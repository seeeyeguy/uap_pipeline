#!/usr/bin/env python3
"""
Rebuild step 3 (local): build a FRESH ChromaDB collection from
rebuild_chunks.jsonl (text + metadata) + emb/*.parquet (id -> vector).

Fresh build into a new dir — NEVER upsert into the old index (avoids the
HNSW tombstone bloat). Result lands in data/vectordb_rebuild/, ready for
`pipeline.py publish` to validate/copy/serve.
"""
import json
from pathlib import Path

import chromadb
import pyarrow.parquet as pq
from chromadb.config import Settings

ROOT = Path(__file__).parent
CHUNKS = ROOT / "data/rebuild_chunks.jsonl"
EMB = ROOT / "data/emb"
DB = ROOT / "data/vectordb_rebuild"
COLLECTION = "uap_documents"
BATCH = 2000

# id -> vector from all parquet shards
vecs = {}
for f in sorted(EMB.glob("emb_*.parquet")):
    t = pq.read_table(f)
    for i, v in zip(t.column("id").to_pylist(), t.column("vector").to_pylist()):
        vecs[i] = v
print(f"loaded {len(vecs)} vectors")

if DB.exists():
    import shutil
    shutil.rmtree(DB)
client = chromadb.PersistentClient(path=str(DB), settings=Settings(anonymized_telemetry=False))
col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

ids, embs, docs, metas = [], [], [], []
missing = 0
added = 0


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
        ids.append(r["id"])
        embs.append(v)
        docs.append(r["text"])
        metas.append(r["metadata"])
        if len(ids) >= BATCH:
            flush()
            if added % 40000 == 0:
                print(f"added {added}", flush=True)
flush()
print(f"BUILD COMPLETE: {added} chunks in {DB} (collection '{COLLECTION}'); "
      f"{missing} chunks had no vector")
print(f"count: {col.count()}")
