#!/usr/bin/env python3
"""
Scan the BUILD vector index for degenerate chunks (textqc.chunk_junk_reason)
and delete them. Dry-run by default: writes data/junk_chunks.jsonl with id,
reason, source, and a text sample for review; pass --delete to actually
remove them from data/vectordb.

A one-time delete of a few thousand ids only tombstones HNSW slots — the
328GB link_lists blowup came from repeated --force re-UPSERTS allocating new
internal ids, which this does not do.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.config import Settings

from textqc import chunk_junk_reason

ROOT = Path(__file__).parent
VDB = ROOT / "data/vectordb"
OUT = ROOT / "data/junk_chunks.jsonl"
BATCH = 5000

ap = argparse.ArgumentParser()
ap.add_argument("--delete", action="store_true",
                help="delete flagged chunks (default: scan + report only)")
args = ap.parse_args()

client = chromadb.PersistentClient(path=str(VDB),
                                   settings=Settings(anonymized_telemetry=False))
col = client.get_collection("uap_documents")
total = col.count()
print(f"scanning {total} chunks", flush=True)

junk, reasons, by_source = [], Counter(), Counter()
offset = 0
with open(OUT, "w") as f:
    while offset < total:
        page = col.get(limit=BATCH, offset=offset,
                       include=["documents", "metadatas"])
        offset += len(page["ids"])
        for cid, doc, meta in zip(page["ids"], page["documents"], page["metadatas"]):
            reason = chunk_junk_reason(doc or "")
            if reason:
                junk.append(cid)
                reasons[reason] += 1
                src = (meta or {}).get("filename", "?")
                by_source[src] += 1
                f.write(json.dumps({"id": cid, "reason": reason, "source": src,
                                    "sample": (doc or "")[:160]}) + "\n")
        if offset % 50000 < BATCH:
            print(f"  {offset}/{total} scanned, {len(junk)} junk", flush=True)

print(f"\nflagged {len(junk)}/{total} chunks ({len(junk)/total:.2%}) — {dict(reasons)}")
print("worst sources:")
for src, n in by_source.most_common(10):
    print(f"  {n:>5}  {src[:80]}")
print(f"details -> {OUT}")

if args.delete and junk:
    for i in range(0, len(junk), BATCH):
        col.delete(ids=junk[i:i + BATCH])
    print(f"DELETED {len(junk)} chunks: {total} -> {col.count()}")
elif junk:
    print("dry run — re-run with --delete to remove them")
