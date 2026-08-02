#!/usr/bin/env python3
"""
Rebuild step 1 (local, CPU): chunk final text + v2 enrichment into
chunks.jsonl — one line per chunk: {id, text, metadata}.

Uses the page-aligned chunker from main.py. Maps the rich enrich_v2
schema into flat ChromaDB-compatible metadata (scalars + JSON strings),
carrying the searchable facets: location hierarchy, shapes, sensors,
witness types, credibility, incidents, source program, page span.

Output feeds rebuild_embed.py (GPU embed → parquet) and
rebuild_assemble.py (build the Chroma collection).
"""
import hashlib
import json
from pathlib import Path

from main import _page_aligned_splits, PAGE_MARKER, PAGE_CHUNK_MAX, \
    RecursiveCharacterTextSplitter, CHUNK_SIZE, CHUNK_OVERLAP
from textqc import chunk_junk_reason

ROOT = Path(__file__).parent
TEXT = ROOT / "data/text"
ENR = ROOT / "data/enriched_v2"
OUT = ROOT / "data/rebuild_chunks.jsonl"

TEXT_NATIVE_SRC = ("nuforc", "geipan")  # (retired NUFORC excluded already)


def _obj(v):
    """LLM occasionally emits a nested field as a list/scalar; coerce to dict."""
    return v if isinstance(v, dict) else {}


def _lst(v):
    return v if isinstance(v, list) else ([] if v is None else [v])


def _s(v):
    """Guarantee a scalar string for Chroma metadata: the LLM sometimes
    emits a scalar field (city, region...) as a list or None."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v if x is not None)
    if isinstance(v, bool):
        return str(v)
    return str(v)


def flat_meta(stem, e, pages, source):
    e = _obj(e)
    loc = _obj(e.get("event_location"))
    obs = _obj(e.get("observation"))
    ev = _obj(e.get("evidence"))
    disp = _obj(e.get("disposition"))
    cred = _obj(e.get("credibility_indicators"))
    people = _lst(e.get("people"))
    return {
        "filename": stem + ".pdf",
        "source": source,
        "pages": pages,
        "language": _s(e.get("language")),
        "summary": _s(e.get("summary"))[:1000],
        "document_type": _s(e.get("document_type")),
        "originating_agency": _s(e.get("originating_agency")),
        "source_program": _s(e.get("source_program")),
        "document_date": _s(e.get("document_date")),
        "event_date": _s(e.get("event_date")),
        "date_precision": _s(e.get("date_precision")),
        "event_time_of_day": _s(e.get("event_time_of_day")),
        # location hierarchy — the facet fix
        "country": _s(loc.get("country")),
        "region": _s(loc.get("region")),
        "city": _s(loc.get("city")),
        "site": _s(loc.get("site")),
        "nearest_named_place": _s(loc.get("nearest_named_place")),
        # observation
        "shapes": json.dumps(_lst(obs.get("shapes"))),
        "colors": json.dumps(_lst(obs.get("colors"))),
        "motions": json.dumps(_lst(obs.get("motions"))),
        "object_count": obs.get("object_count") if isinstance(obs.get("object_count"), int) else -1,
        # evidence
        "sensor_types": json.dumps(_lst(ev.get("sensor_types"))),
        "witness_count": ev.get("witness_count") if isinstance(ev.get("witness_count"), int) else -1,
        "witness_types": json.dumps(_lst(ev.get("witness_types"))),
        # disposition
        "explanation_status": _s(disp.get("explanation_status")),
        # entities / graph seeds
        "people": json.dumps([p.get("name") for p in people if isinstance(p, dict)]),
        "people_roles": json.dumps(people),
        "organizations": json.dumps(_lst(e.get("organizations"))),
        "named_incidents": json.dumps(_lst(e.get("named_incidents"))),
        "related_case_ids": json.dumps(_lst(e.get("related_case_ids"))),
        # credibility booleans (stringified for chroma)
        "official_source": str(cred.get("official_source", False)),
        "radar_confirmation": str(cred.get("radar_confirmation", False)),
        "physical_evidence": str(cred.get("physical_evidence_mentioned", False)),
        "multiple_witnesses": str(cred.get("multiple_witnesses", False)),
        "topics": json.dumps(_lst(e.get("topics"))),
        "classification_level": _s(e.get("classification_level")),
        "ocr_quality": _s(e.get("ocr_quality")),
    }


def doc_chunks(stem: str, splitter=None):
    """Chunk one document exactly as the full rebuild does. Returns a list of
    {id, text, metadata} dicts (empty when the doc is too short/absent).
    Shared by main() and pg_patch.py so patch and rebuild can never drift."""
    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "])
    tp = TEXT / f"{stem}.txt"
    if not tp.exists():
        return []
    text = tp.read_text(encoding="utf-8", errors="ignore")
    if len(text.strip()) < 50:
        return []
    ep = ENR / f"{stem}.json"
    e = json.loads(ep.read_text()) if ep.exists() else {}
    source = "corpus"  # single collection; source dir not tracked per-doc here
    meta = flat_meta(stem, e, "", source)

    if PAGE_MARKER.search(text):
        splits = _page_aligned_splits(text, splitter)
    else:
        splits = [(s, "") for s in splitter.split_text(text)]
    # drop degenerate chunks (OCR loops, alphabet-free table shred)
    splits = [(b, p) for b, p in splits if not chunk_junk_reason(b)]
    summary = (e.get("summary") or "").strip()
    if summary:
        splits = [(summary, "summary")] + splits

    src_tag = hashlib.sha1(source.encode()).hexdigest()[:8]
    out = []
    for i, (body, pages) in enumerate(splits, start=(-1 if summary else 0)):
        m = dict(meta); m["pages"] = pages; m["chunk_id"] = i
        out.append({"id": f"{stem}_{src_tag}_chunk_{i}", "text": body, "metadata": m})
    return out


def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "])
    n_docs = n_chunks = 0
    with open(OUT, "w") as fout:
        for tp in sorted(TEXT.glob("*.txt")):
            chunks = doc_chunks(tp.stem, splitter)
            if not chunks:
                continue
            for c in chunks:
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
            n_chunks += len(chunks)
            n_docs += 1
            if n_docs % 2000 == 0:
                print(f"{n_docs} docs, {n_chunks} chunks", flush=True)
    print(f"DONE: {n_docs} docs -> {n_chunks} chunks -> {OUT}")


if __name__ == "__main__":
    main()
