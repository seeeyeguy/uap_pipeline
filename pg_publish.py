#!/usr/bin/env python3
"""
Publish the corpus to Postgres (pgvector) — the serving store.

Chroma (data/vectordb) stays the BUILD store the ingest pipeline writes;
this script syncs it, plus the analytics tables from data/analytics.duckdb,
into the `corpus` schema of the app's Postgres:

  corpus.chunks        id, text, meta jsonb, embedding vector(1024),
                       tsv_simple + tsv_en (generated) for lexical search
  corpus.events        the unified analytics events table
  corpus.locations     GeoNames dimension
  corpus.geo_clusters  hotspot summary (spatial)
  corpus.st_clusters   wave summary (spatiotemporal)
  corpus.time_clusters flap-period summary (temporal)
  corpus.stats_*       statistics passes

Each table loads into <name>_new and swaps in a single transaction, so the
retriever and API never see a partial corpus and need no restart. Rollback
window: the previous table survives as <name>_old until the next publish.

DSN: $PG_DSN, else built from ../uap-api/.env (host localhost:5439).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import chromadb
import duckdb
import psycopg
from chromadb.config import Settings

ROOT = Path(__file__).parent
VDB = ROOT / "data/vectordb"
ADB = ROOT / "data/analytics.duckdb"
DIM = 1024
BATCH = 2000


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    pw = re.search(r"^POSTGRES_PASSWORD=(.*)$", env, re.M).group(1).strip()
    user = re.search(r"^POSTGRES_USER=(.*)$", env, re.M).group(1).strip()
    db = re.search(r"^POSTGRES_DB=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{user}:{pw}@localhost:5439/{db}"


def swap(cur, name):
    """Atomically replace corpus.<name> with corpus.<name>_new.

    Indexes keep their names through ALTER TABLE RENAME, so the demoted
    table's indexes are suffixed _old (freed when <name>_old drops next
    publish) and the staging table's <name>_new_* indexes are canonicalized
    to <name>_* — otherwise the next publish's CREATE INDEX collides with
    names now owned by the live table.
    """
    cur.execute(f"DROP TABLE IF EXISTS corpus.{name}_old CASCADE")
    cur.execute("BEGIN")
    cur.execute("""SELECT indexname FROM pg_indexes
                   WHERE schemaname='corpus' AND tablename=%s""", (name,))
    for (idx,) in cur.fetchall():
        cur.execute(f'ALTER INDEX corpus."{idx}" RENAME TO "{idx}_old"')
    cur.execute(f"ALTER TABLE IF EXISTS corpus.{name} RENAME TO {name}_old")
    cur.execute("""SELECT indexname FROM pg_indexes
                   WHERE schemaname='corpus' AND tablename=%s""", (f"{name}_new",))
    for (idx,) in cur.fetchall():
        canonical = idx.replace(f"{name}_new_", f"{name}_", 1)
        cur.execute(f'ALTER INDEX corpus."{idx}" RENAME TO "{canonical}"')
    cur.execute(f"ALTER TABLE corpus.{name}_new RENAME TO {name}")
    cur.execute("COMMIT")


def sync_chunks(pg):
    col = (chromadb.PersistentClient(path=str(VDB),
                                     settings=Settings(anonymized_telemetry=False))
           .get_collection("uap_documents"))
    total = col.count()
    print(f"chunks: syncing {total} from build index", flush=True)
    cur = pg.cursor()
    cur.execute("DROP TABLE IF EXISTS corpus.chunks_new CASCADE")
    cur.execute(f"""CREATE TABLE corpus.chunks_new(
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        meta JSONB NOT NULL DEFAULT '{{}}',
        embedding vector({DIM}) NOT NULL,
        tsv_simple tsvector GENERATED ALWAYS AS
            (to_tsvector('simple', left(text, 100000))) STORED,
        tsv_en tsvector GENERATED ALWAYS AS
            (to_tsvector('english', left(text, 100000))) STORED)""")
    pg.commit()

    t0 = time.time()
    done = 0
    with cur.copy("COPY corpus.chunks_new (id, text, meta, embedding) FROM STDIN") as cp:
        offset = 0
        while offset < total:
            page = col.get(limit=BATCH, offset=offset,
                           include=["documents", "metadatas", "embeddings"])
            offset += len(page["ids"])
            for cid, doc, meta, emb in zip(page["ids"], page["documents"],
                                           page["metadatas"], page["embeddings"]):
                vec = "[" + ",".join(f"{x:.7g}" for x in emb) + "]"
                # OCR text can carry NUL bytes; postgres TEXT rejects them
                doc = (doc or "").replace("\x00", "")
                mjson = json.dumps(meta or {}).replace("\\u0000", "")
                cp.write_row((cid, doc, mjson, vec))
            done = offset
            if done % 20000 < BATCH:
                print(f"  copied {done}/{total} ({time.time()-t0:.0f}s)", flush=True)
    pg.commit()
    print(f"  copy done: {done} rows in {time.time()-t0:.0f}s; indexing…", flush=True)

    cur.execute("SET maintenance_work_mem = '2GB'")
    cur.execute("SET max_parallel_maintenance_workers = 8")
    t0 = time.time()
    # Parallel HNSW builds allocate multi-GB shared-memory segments — more
    # than a container's /dev/shm may allow as the corpus grows (prod's 2 GB
    # cap failed at 384k chunks). Single-threaded is slower but bounded.
    cur.execute("SET max_parallel_maintenance_workers = 0")
    cur.execute("""CREATE INDEX chunks_new_embedding_idx ON corpus.chunks_new
                   USING hnsw (embedding vector_cosine_ops)""")
    cur.execute("RESET max_parallel_maintenance_workers")
    print(f"  hnsw index: {time.time()-t0:.0f}s", flush=True)
    cur.execute("CREATE INDEX chunks_new_tsv_simple_idx ON corpus.chunks_new USING gin (tsv_simple)")
    cur.execute("CREATE INDEX chunks_new_tsv_en_idx ON corpus.chunks_new USING gin (tsv_en)")
    cur.execute("CREATE INDEX chunks_new_meta_idx ON corpus.chunks_new USING gin (meta jsonb_path_ops)")
    # Filename equality drives the entity/document/citation endpoints; the
    # gin meta index cannot serve ->> equality, and dropping this on every
    # swap silently regressed those lookups from ~20ms to 15s scans.
    cur.execute("CREATE INDEX chunks_new_meta_filename_idx"
                " ON corpus.chunks_new ((meta->>'filename'))")
    pg.commit()
    swap(cur, "chunks")
    pg.commit()
    print("  swapped in as corpus.chunks", flush=True)


def sync_analytics(pg):
    con = duckdb.connect(str(ADB), read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    cur = pg.cursor()
    for t in tables:
        cols = con.execute(f"DESCRIBE {t}").fetchall()
        typemap = {"VARCHAR": "TEXT", "DOUBLE": "DOUBLE PRECISION",
                   "BIGINT": "BIGINT", "INTEGER": "INTEGER", "HUGEINT": "NUMERIC"}
        defs = ", ".join(f'"{c[0]}" {typemap.get(c[1], "TEXT")}' for c in cols)
        cur.execute(f"DROP TABLE IF EXISTS corpus.{t}_new CASCADE")
        cur.execute(f"CREATE TABLE corpus.{t}_new ({defs})")
        rows = con.execute(f"SELECT * FROM {t}").fetchall()
        with cur.copy(f"COPY corpus.{t}_new FROM STDIN") as cp:
            for r in rows:
                cp.write_row(r)
        pg.commit()
        swap(cur, t)
        pg.commit()
        print(f"  corpus.{t}: {len(rows)} rows", flush=True)
    # tables retired from the build store linger in corpus.* — drop them
    cur.execute("DROP TABLE IF EXISTS corpus.stats_waves CASCADE")
    cur.execute("DROP TABLE IF EXISTS corpus.stats_waves_old CASCADE")
    # indexes the API needs (recreate every publish — tables were swapped)
    for ddl in ("CREATE INDEX IF NOT EXISTS events_cluster_idx ON corpus.events (geo_cluster)",
                "CREATE INDEX IF NOT EXISTS events_st_cluster_idx ON corpus.events (st_cluster)",
                "CREATE INDEX IF NOT EXISTS events_time_cluster_idx ON corpus.events (time_cluster)",
                "CREATE INDEX IF NOT EXISTS events_year_idx ON corpus.events (event_year)",
                "CREATE INDEX IF NOT EXISTS events_region_idx ON corpus.events (loc_cc, loc_region)",
                # advanced search: ILIKE over descriptions/citations (map filters)
                "CREATE EXTENSION IF NOT EXISTS pg_trgm",
                "CREATE INDEX IF NOT EXISTS events_desc_trgm_idx ON corpus.events USING gin (description gin_trgm_ops)",
                "CREATE INDEX IF NOT EXISTS events_sref_trgm_idx ON corpus.events USING gin (source_ref gin_trgm_ops)"):
        cur.execute(ddl)
    pg.commit()
    con.close()


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    with psycopg.connect(dsn(), autocommit=False) as pg:
        pg.cursor().execute("CREATE SCHEMA IF NOT EXISTS corpus")
        pg.commit()
        if only in ("all", "analytics"):
            print("analytics tables -> corpus.*", flush=True)
            sync_analytics(pg)
        if only in ("all", "chunks"):
            sync_chunks(pg)
    print("PG PUBLISH COMPLETE")


if __name__ == "__main__":
    main()
