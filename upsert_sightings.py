#!/usr/bin/env python3
"""
Upsert sighting chunks + their bge-m3 vectors into the live build index
(data/vectordb). Purely additive — new IDs only, so no HNSW tombstone
churn. Run `pipeline.py publish` afterward to serve.
"""
import json
from pathlib import Path

import chromadb
import pyarrow.parquet as pq
from chromadb.config import Settings

ROOT = Path(__file__).parent
CHUNKS = ROOT / "data/sightings_chunks.jsonl"
EMB = ROOT / "data/emb_sightings"
DB = ROOT / "data/vectordb"

vecs = {}
for f in sorted(EMB.glob("emb_*.parquet")):
    t = pq.read_table(f)
    for i, v in zip(t.column("id").to_pylist(), t.column("vector").to_pylist()):
        vecs[i] = v
print(f"loaded {len(vecs)} sighting vectors")

client = chromadb.PersistentClient(path=str(DB), settings=Settings(anonymized_telemetry=False))
col = client.get_collection("uap_documents")
before = col.count()

ids, embs, docs, metas = [], [], [], []
added = missing = 0


def flush():
    global added
    if ids:
        col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        added += len(ids)
        ids.clear(); embs.clear(); docs.clear(); metas.clear()


with open(CHUNKS) as f:
    for line in f:
        r = json.loads(line)
        v = vecs.get(r["id"])
        if v is None:
            missing += 1
            continue
        ids.append(r["id"]); embs.append(v)
        docs.append(r["text"]); metas.append(r["metadata"])
        if len(ids) >= 2000:
            flush()
flush()
print(f"UPSERT COMPLETE: {before} -> {col.count()} chunks (+{added}); {missing} had no vector")
