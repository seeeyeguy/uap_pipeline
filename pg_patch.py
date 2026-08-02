#!/usr/bin/env python3
"""
Surgical patch publish: push a SMALL set of changed documents to the serving
stores in minutes, without the full rebuild -> embed -> assemble -> republish
cycle (which is hours and exists for corpus-wide changes only).

Handles the common small-change case: document text changed (re-OCR, merge)
or enrichment changed for a handful of stems. For each stem it re-chunks
(identically to rebuild_chunks.py — shared code), embeds the new chunks
locally (a few hundred chunks: seconds on GPU, ~a minute on CPU), then in
ONE pg transaction per target:

  - stages rows via COPY (one round trip — tunnel-friendly)
  - DELETE + INSERT the doc's chunks (tsv columns are generated; the HNSW
    index absorbs inserts incrementally)
  - refreshes the doc's corpus.catalog_docs row  [KEEP IN SYNC: uap-api
    db/catalog.sql — same expressions, filtered to the patched filenames]
  - refreshes the doc's corpus.entities rows     [KEEP IN SYNC: uap-api
    db/search_upgrade.sql src CTE]
  - CALL corpus.backfill_source_tier()  (only rows missing the key — i.e.
    exactly the fresh inserts — are touched)

Known limits (full publish still owns these): incidents/events linkage for
NEW docs, analytics tables, and the Chroma build store's HNSW hygiene
(--chroma patches it additively; tombstones are reclaimed by the next full
rebuild).

Usage:
  pg_patch.py --changed                # stems from data/merge_changed.txt, dry run
  pg_patch.py --stems A B --apply      # patch local pg (localhost:5439)
  PG_DSN=postgresql://…@localhost:15439/db pg_patch.py --changed --apply
  pg_patch.py --changed --apply --chroma   # also patch the build store
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import psycopg

from rebuild_chunks import doc_chunks

ROOT = Path(__file__).parent


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text() \
        if (ROOT.parent / "uap-api/.env").exists() \
        else Path("/apps/uap-api/.env").read_text()
    g = lambda k: re.search(rf"^{k}=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{g('POSTGRES_USER')}:{g('POSTGRES_PASSWORD')}@localhost:5439/{g('POSTGRES_DB')}"


def embed(texts: list[str], cpu: bool) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    device = "cpu" if cpu else None  # None -> cuda when available
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    model.max_seq_length = 1024
    vecs = model.encode(texts, batch_size=8 if cpu else 12,
                        show_progress_bar=len(texts) > 50, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


# ── targeted derived-artifact refresh ────────────────────────────────────
# KEEP IN SYNC with uap-api db/catalog.sql (doc CTE + final SELECT). A column
# drift fails loudly (column count mismatch), never silently.
CATALOG_SEEN = """
INSERT INTO corpus.catalog_seen (filename)
SELECT DISTINCT meta->>'filename' FROM corpus.chunks
WHERE meta->>'filename' = ANY(%(files)s)
ON CONFLICT (filename) DO NOTHING
"""

CATALOG_DELETE = "DELETE FROM corpus.catalog_docs WHERE filename = ANY(%(files)s)"

CATALOG_INSERT = """
INSERT INTO corpus.catalog_docs
    (filename, source_program, document_type, ocr_quality, language, pages,
     originating_agency, classification_level, official_source,
     radar_confirmation, physical_evidence, multiple_witnesses, display_title,
     event_date, document_date, city, region, country, summary,
     named_incidents, shapes, witness_count, year, first_seen, chunk_count,
     events_count, tsv)
WITH doc AS (
    SELECT DISTINCT ON (meta->>'filename')
        meta->>'filename'                                   AS filename,
        NULLIF(meta->>'source_program', '')                 AS source_program,
        NULLIF(meta->>'document_type', '')                  AS document_type,
        NULLIF(meta->>'ocr_quality', '')                    AS ocr_quality,
        NULLIF(meta->>'language', '')                       AS language,
        NULLIF(meta->>'pages', '')                          AS pages,
        NULLIF(meta->>'originating_agency', '')             AS originating_agency,
        NULLIF(meta->>'classification_level', '')           AS classification_level,
        COALESCE(lower(meta->>'official_source') = 'true', false)    AS official_source,
        COALESCE(lower(meta->>'radar_confirmation') = 'true', false) AS radar_confirmation,
        COALESCE(lower(meta->>'physical_evidence') = 'true', false)  AS physical_evidence,
        COALESCE(lower(meta->>'multiple_witnesses') = 'true', false) AS multiple_witnesses,
        NULLIF(meta->>'title', '')                          AS display_title,
        NULLIF(meta->>'event_date', '')                     AS event_date,
        NULLIF(meta->>'document_date', '')                  AS document_date,
        NULLIF(meta->>'city', '')                           AS city,
        NULLIF(meta->>'region', '')                         AS region,
        NULLIF(meta->>'country', '')                        AS country,
        NULLIF(meta->>'summary', '')                        AS summary,
        CASE WHEN jsonb_typeof(meta->'named_incidents') = 'array' THEN meta->'named_incidents'
             WHEN (meta->>'named_incidents') ~ '^\\[' THEN (meta->>'named_incidents')::jsonb
        END                                                 AS named_incidents,
        CASE WHEN jsonb_typeof(meta->'shapes') = 'array' THEN meta->'shapes'
             WHEN (meta->>'shapes') ~ '^\\[' THEN (meta->>'shapes')::jsonb
        END                                                 AS shapes,
        NULLIF(regexp_replace(meta->>'witness_count', '\\D', '', 'g'), '')::int
                                                            AS witness_count
    FROM corpus.chunks
    WHERE meta->>'chunk_id' = '0' AND meta->>'filename' = ANY(%(files)s)
    ORDER BY meta->>'filename', length(COALESCE(meta->>'summary', '')) DESC
),
counts AS (
    SELECT meta->>'filename' AS filename, count(*) AS chunk_count
    FROM corpus.chunks WHERE meta->>'filename' = ANY(%(files)s) GROUP BY 1
),
evt AS (
    SELECT doc_ref, count(*) AS events_count
    FROM corpus.events WHERE source = 'corpus' GROUP BY 1
)
SELECT
    doc.filename, doc.source_program, doc.document_type, doc.ocr_quality,
    doc.language, doc.pages, doc.originating_agency, doc.classification_level,
    doc.official_source, doc.radar_confirmation, doc.physical_evidence,
    doc.multiple_witnesses, doc.display_title, doc.event_date,
    doc.document_date, doc.city, doc.region, doc.country, doc.summary,
    doc.named_incidents, doc.shapes, doc.witness_count,
    COALESCE(substring(doc.event_date FROM '^\\d{4}'),
             substring(doc.document_date FROM '^\\d{4}'))::int,
    seen.first_seen,
    counts.chunk_count,
    COALESCE(evt.events_count, 0),
    setweight(to_tsvector('simple', translate(regexp_replace(doc.filename, '\\.pdf$', ''), '-_.', '   ')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(doc.display_title, '')), 'A') ||
    setweight(to_tsvector('simple', concat_ws(' ',
        doc.city, doc.region, doc.country, doc.originating_agency,
        COALESCE((SELECT string_agg(x, ' ') FROM jsonb_array_elements_text(doc.named_incidents) x), ''))), 'B') ||
    setweight(to_tsvector('english', COALESCE(doc.summary, '')), 'C')
FROM doc
JOIN counts USING (filename)
JOIN corpus.catalog_seen seen USING (filename)
LEFT JOIN evt ON evt.doc_ref = regexp_replace(doc.filename, '\\.pdf$', '')
"""

# KEEP IN SYNC with uap-api db/search_upgrade.sql (entities src CTE).
# KEEP IN SYNC with corpus.backfill_source_tier() (db/search_upgrade.sql):
# same tier CASE, scoped to the patched filenames — the proc's find-unfiled
# loop full-scans the table, which is bulk-publish economics, not patch.
TIER_REFRESH = """
UPDATE corpus.chunks c
SET meta = c.meta || jsonb_build_object('source_tier',
    CASE
        WHEN c.meta->>'source_program' = 'nuforc' THEN '4'
        WHEN c.meta->>'official_source' = 'True'
             OR c.meta->>'source_program' IN
                ('project_blue_book','project_sign','project_grudge',
                 'project_saucer','aaro','mod_uk','sepra_cnes',
                 'ejercito_del_aire','esercito_del_aire','sefaa_cefaa')
            THEN '1'
        WHEN c.meta->>'source_program' = 'condon_committee'
             OR c.meta->>'document_type' IN
                ('scientific_analysis','photo_analysis','video_analysis',
                 'investigation_report','witness_statement','case_index')
            THEN '2'
        WHEN c.meta->>'document_type' IN ('press_clipping','periodical')
            THEN '3'
        WHEN c.meta->>'document_type' IN
                ('book_or_periodical','transcript','interview_transcript','mixed')
            THEN '5'
        ELSE '4'
    END)
WHERE c.meta->>'filename' = ANY(%(files)s) AND NOT c.meta ? 'source_tier'
"""

ENTITIES_DELETE = "DELETE FROM corpus.entities WHERE filename = ANY(%(files)s)"

ENTITIES_INSERT = """
WITH src AS (
    SELECT
        meta->>'filename'       AS filename,
        meta->>'source_program' AS source_program,
        meta->>'event_date'     AS event_date,
        k.etype,
        CASE WHEN pg_input_is_valid(meta->>k.key, 'jsonb')
             THEN (meta->>k.key)::jsonb END AS arr
    FROM corpus.chunks
    CROSS JOIN (VALUES
        ('people',          'person'),
        ('organizations',   'organization'),
        ('named_incidents', 'incident')
    ) AS k(key, etype)
    WHERE meta->>'filename' = ANY(%(files)s)
      AND coalesce(meta->>k.key,'') NOT IN ('', '[]')
)
INSERT INTO corpus.entities (name, etype, filename, source_program, event_date)
SELECT DISTINCT ON (btrim(e.name), src.etype, src.filename)
    btrim(e.name), src.etype, src.filename, src.source_program, src.event_date
FROM src
CROSS JOIN LATERAL jsonb_array_elements_text(src.arr) AS e(name)
WHERE jsonb_typeof(src.arr) = 'array'
  AND length(btrim(e.name)) BETWEEN 3 AND 120
ON CONFLICT DO NOTHING
"""


def patch_pg(pg, docs: dict[str, list[dict]], vecs: dict[str, list[float]]):
    files = sorted({c["metadata"]["filename"] for cs in docs.values() for c in cs})
    t0 = time.time()
    with pg.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE _patch_chunks (
                id TEXT, text TEXT, meta JSONB, embedding vector(1024)
            ) ON COMMIT DROP""")
        with cur.copy("COPY _patch_chunks (id, text, meta, embedding) FROM STDIN") as cp:
            for cs in docs.values():
                for c in cs:
                    cp.write_row((c["id"], c["text"],
                                  json.dumps(c["metadata"], ensure_ascii=False),
                                  "[" + ",".join(f"{x:.7f}" for x in vecs[c["id"]]) + "]"))
        cur.execute("DELETE FROM corpus.chunks WHERE meta->>'filename' = ANY(%s)", (files,))
        deleted = cur.rowcount
        cur.execute("""
            INSERT INTO corpus.chunks (id, text, meta, embedding)
            SELECT id, text, meta, embedding FROM _patch_chunks""")
        inserted = cur.rowcount
        for stmt in (TIER_REFRESH, CATALOG_SEEN, CATALOG_DELETE, CATALOG_INSERT,
                     ENTITIES_DELETE, ENTITIES_INSERT):
            cur.execute(stmt, {"files": files})
    pg.commit()
    print(f"pg: {len(files)} docs · {deleted} chunks out, {inserted} in "
          f"({time.time()-t0:.0f}s)")


def patch_chroma(docs: dict[str, list[dict]], vecs: dict[str, list[float]]):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(ROOT / "data/vectordb"),
                                       settings=Settings(anonymized_telemetry=False))
    col = client.get_collection("uap_documents")
    files = sorted({c["metadata"]["filename"] for cs in docs.values() for c in cs})
    col.delete(where={"filename": {"$in": files}})
    ids, texts, metas, embs = [], [], [], []
    for cs in docs.values():
        for c in cs:
            ids.append(c["id"]); texts.append(c["text"])
            metas.append(c["metadata"]); embs.append(vecs[c["id"]])
    for i in range(0, len(ids), 5000):  # chroma batch cap is 5,461
        col.add(ids=ids[i:i+5000], documents=texts[i:i+5000],
                metadatas=metas[i:i+5000], embeddings=embs[i:i+5000])
    print(f"chroma: {len(files)} docs re-added ({len(ids)} chunks) — additive; "
          f"tombstones reclaimed by the next full rebuild")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="*")
    ap.add_argument("--stems-file")
    ap.add_argument("--changed", action="store_true",
                    help="stems from data/merge_changed.txt")
    ap.add_argument("--apply", action="store_true", help="write Postgres")
    ap.add_argument("--chroma", action="store_true", help="also patch the build store")
    ap.add_argument("--cpu", action="store_true", help="embed on CPU (GPU busy)")
    args = ap.parse_args()

    stems = list(args.stems or [])
    if args.stems_file:
        stems += [l.strip() for l in open(args.stems_file) if l.strip()]
    if args.changed:
        stems += [l.strip() for l in (ROOT / "data/merge_changed.txt").open() if l.strip()]
    stems = sorted(set(stems))
    if not stems:
        ap.error("no stems (use --stems / --stems-file / --changed)")
    if len(stems) > 500:
        ap.error(f"{len(stems)} stems — that's a rebuild, not a patch. "
                 "Use the full publish path above ~500 docs.")

    docs, missing = {}, []
    for s in stems:
        cs = doc_chunks(s)
        (docs.__setitem__(s, cs) if cs else missing.append(s))
    n_chunks = sum(len(c) for c in docs.values())
    print(f"{len(docs)} docs -> {n_chunks} chunks to patch"
          + (f" · {len(missing)} stems missing/short: {missing[:5]}" if missing else ""))
    if not args.apply:
        print("dry run — pass --apply to write")
        return

    pg = psycopg.connect(dsn())
    # Preserve display titles: enriched_v2 predates the title backfill, so a
    # re-chunked meta may lack meta.title — carry the serving store's forward
    # rather than wiping it (catalog display_title derives from it).
    files = sorted({c["metadata"]["filename"] for cs in docs.values() for c in cs})
    titles = dict(pg.execute(
        """SELECT DISTINCT ON (meta->>'filename') meta->>'filename', meta->>'title'
           FROM corpus.chunks
           WHERE meta->>'filename' = ANY(%s) AND COALESCE(meta->>'title','') != ''
           ORDER BY meta->>'filename'""", (files,)).fetchall())
    kept = 0
    for cs in docs.values():
        for c in cs:
            if not c["metadata"].get("title") and titles.get(c["metadata"]["filename"]):
                c["metadata"]["title"] = titles[c["metadata"]["filename"]]
                kept += 1
    if kept:
        print(f"preserved titles on {len(titles)} docs ({kept} chunks)")

    t0 = time.time()
    all_chunks = [c for cs in docs.values() for c in cs]
    vec_list = embed([c["text"] for c in all_chunks], cpu=args.cpu)
    vecs = {c["id"]: v for c, v in zip(all_chunks, vec_list)}
    print(f"embedded {len(vecs)} chunks ({time.time()-t0:.0f}s)")

    patch_pg(pg, docs, vecs)
    if args.chroma:
        patch_chroma(docs, vecs)
    print("PATCH DONE")


if __name__ == "__main__":
    main()
