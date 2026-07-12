#!/usr/bin/env python3
"""
Rebuild step 2 (pod GPU): embed chunks.jsonl with bge-m3, write
embeddings.parquet (id, vector). The portable artifact — feeds the Chroma
build now and pgvector later. Resumable: skips ids already in the parquet
shards. Run:
  /workspace/venv_embed/bin/python rebuild_embed.py /workspace/rebuild_chunks.jsonl /workspace/emb
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

CHUNKS = Path(sys.argv[1])
OUTDIR = Path(sys.argv[2])
OUTDIR.mkdir(parents=True, exist_ok=True)
SHARD = 20000     # rows per parquet shard
BATCH = 12

done_ids = set()
for f in OUTDIR.glob("emb_*.parquet"):
    done_ids |= set(pq.read_table(f, columns=["id"]).column("id").to_pylist())
print(f"resuming: {len(done_ids)} ids already embedded", flush=True)

model = SentenceTransformer("BAAI/bge-m3", device="cuda")
model.max_seq_length = 1024
shard_idx = len(list(OUTDIR.glob("emb_*.parquet")))
buf_ids, buf_txt = [], []
total = 0


def flush(ids, vecs, idx):
    t = pa.table({"id": ids, "vector": [v.tolist() for v in vecs]})
    pq.write_table(t, OUTDIR / f"emb_{idx:04d}.parquet")
    print(f"wrote emb_{idx:04d}.parquet ({len(ids)} rows)", flush=True)


with open(CHUNKS) as f:
    for line in f:
        r = json.loads(line)
        if r["id"] in done_ids:
            continue
        buf_ids.append(r["id"])
        buf_txt.append(r["text"][:8000])
        if len(buf_ids) >= SHARD:
            vecs = model.encode(buf_txt, batch_size=BATCH, normalize_embeddings=True,
                                show_progress_bar=True)
            flush(buf_ids, vecs, shard_idx)
            total += len(buf_ids)
            shard_idx += 1
            buf_ids, buf_txt = [], []
if buf_ids:
    vecs = model.encode(buf_txt, batch_size=BATCH, normalize_embeddings=True,
                        show_progress_bar=True)
    flush(buf_ids, vecs, shard_idx)
    total += len(buf_ids)
print(f"EMBED COMPLETE: {total} new vectors across {OUTDIR}", flush=True)
