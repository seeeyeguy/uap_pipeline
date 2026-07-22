#!/usr/bin/env python3
"""
UAP Retrieval Sidecar
---------------------
A tiny HTTP service that the Go API (uap-api) calls for research-mode RAG.

Two backends:

  pg (default when PG_DSN is set)
      Hybrid retrieval over Postgres/pgvector (corpus.chunks, synced by
      pg_publish.py): dense cosine search + two lexical arms (tsv_simple
      preserves exact tokens like callsigns and case numbers; tsv_en adds
      English stemming), fused with reciprocal-rank fusion. Fixes the
      dense-only blind spot where "LACY 17" retrieves nothing.

  chroma (fallback)
      The original dense-only search against the mounted ChromaDB release.

Queries embed with the pipeline's exact model (BAAI/bge-m3), so they land
in the same 1024-dim space the corpus was built in.

Query expansion (pg backend): a static domain synonym map (query_expansion.py)
adds OR-variants of the query to the two LEXICAL arms only — the dense arm
always embeds the user's exact wording.

Optional rerank stage (RERANK=1, default off): after RRF fusion the full
candidate pool (POOL=50) is rescored by a cross-encoder
(BAAI/bge-reranker-v2-m3 via sentence-transformers CrossEncoder) against the
query text, and the top-k by reranker score are returned. The model is
lazy-loaded on the first reranked request. MEMORY COST: the reranker adds
roughly 2 GB RSS (~560M params + activations) on top of bge-m3 — raise the
container memory limit accordingly before enabling in prod. Response stays
backward-compatible: existing fields are unchanged; each chunk additionally
carries "rerank_score" (and "score" keeps the RRF fusion score).

Endpoints:
  POST /retrieve  {"question": str, "n": int=5, "filters": {..}?,
                   "mode": "hybrid"|"dense"?}
       -> {"chunks": [{"id","text","distance","score","source",
                       "document_type","summary"[,"rerank_score"]}]}
  GET  /health    -> {"status":"ok","backend":..,"chunks":N}

Run (from the pipeline venv, so bge-m3 + the GPU are available):
  source .venv/bin/activate
  PG_DSN=postgresql://uap:...@localhost:5439/uapdb python retriever.py
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from main import load_embed_model, load_vectordb, COLLECTION_NAME
from query_expansion import expand_query

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retriever")

HOST = os.environ.get("RETRIEVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("RETRIEVER_PORT", "8001"))
PG_DSN = os.environ.get("PG_DSN", "")
RRF_K = 60          # standard reciprocal-rank-fusion constant
POOL = 50           # candidates fetched per arm before fusion

# Optional cross-encoder rerank stage (see module docstring; ~2GB extra RSS).
RERANK = os.environ.get("RERANK", "") == "1"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "1024"))

log.info("Loading embedder (bge-m3)…")
EMBEDDER = load_embed_model()
# Serialize GPU embedding + store access; the research UI is low-QPS and this
# avoids concurrent-CUDA / driver reentrancy surprises under ThreadingHTTPServer.
_LOCK = threading.Lock()


# ── filter translation: chroma-style dict -> parameterized SQL over meta ──
_OPS = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def _cond(key, value, params):
    if isinstance(value, dict):
        clauses = []
        for op, v in value.items():
            if op == "$in":
                if not isinstance(v, list) or not v:
                    raise ValueError("$in needs a non-empty list")
                ph = ",".join("%s" for _ in v)
                if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                    clauses.append(f"(meta->>'{key}')::numeric IN ({ph})")
                else:
                    clauses.append(f"meta->>'{key}' IN ({ph})")
                params.extend(v)
            elif op in _OPS:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    clauses.append(f"(meta->>'{key}')::numeric {_OPS[op]} %s")
                else:
                    clauses.append(f"meta->>'{key}' {_OPS[op]} %s")
                params.append(v)
            else:
                raise ValueError(f"unsupported operator {op}")
        return "(" + " AND ".join(clauses) + ")"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        params.append(value)
        return f"(meta->>'{key}')::numeric = %s"
    params.append(str(value))
    return f"meta->>'{key}' = %s"


def filters_to_sql(filters):
    """-> (where_fragment, params). Empty filters -> ('TRUE', [])."""
    if not filters:
        return "TRUE", []
    params = []

    def walk(node):
        if "$and" in node:
            return "(" + " AND ".join(walk(x) for x in node["$and"]) + ")"
        if "$or" in node:
            return "(" + " OR ".join(walk(x) for x in node["$or"]) + ")"
        parts = [_cond(k, v, params) for k, v in node.items()
                 if not k.startswith("$") and re_key(k)]
        if not parts:
            raise ValueError("no valid filter keys")
        return "(" + " AND ".join(parts) + ")"

    return walk(filters), params


def re_key(k):
    """metadata keys are code-controlled; reject anything exotic outright"""
    import re
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", k):
        raise ValueError(f"bad filter key {k!r}")
    return True


# ── backends ──
class PgBackend:
    name = "pg"

    def __init__(self, dsn):
        import psycopg
        self._psycopg = psycopg
        self._dsn = dsn
        self._conn = None
        self.connect()

    def connect(self):
        self._conn = self._psycopg.connect(self._dsn, autocommit=True)

    def _run(self, sql, params):
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
        except self._psycopg.OperationalError:
            log.warning("pg connection lost — reconnecting")
            self.connect()
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def count(self):
        return self._run("SELECT COUNT(*) FROM corpus.chunks", [])[0][0]

    def query(self, emb, question, n, filters, mode):
        where, fparams = filters_to_sql(filters)
        vec = "[" + ",".join(f"{x:.7g}" for x in emb) + "]"
        pool = max(POOL, n * 4)

        dense = self._run(
            f"""SELECT id, text, meta, embedding <=> %s::vector AS dist
                FROM corpus.chunks WHERE {where}
                ORDER BY embedding <=> %s::vector LIMIT %s""",
            [vec, *fparams, vec, pool])
        arms = [dense]
        if mode != "dense":
            # Domain synonym expansion, lexical arms only: OR the tsqueries
            # of the original question and its single-substitution variants.
            # (tsquery || tsquery is OR; an empty tsquery operand is a no-op.)
            variants = expand_query(question)
            tsq = " || ".join(["websearch_to_tsquery(%s, %s)"] * len(variants))
            for cfg, col in (("simple", "tsv_simple"), ("english", "tsv_en")):
                vparams = [p for v in variants for p in (cfg, v)]
                arms.append(self._run(
                    f"""SELECT id, text, meta, NULL::float AS dist
                        FROM corpus.chunks,
                             (SELECT {tsq}) AS qx(q)
                        WHERE {col} @@ q AND {where}
                        ORDER BY ts_rank_cd({col}, q) DESC LIMIT %s""",
                    [*vparams, *fparams, pool]))

        # reciprocal-rank fusion across arms
        fused = {}
        for arm in arms:
            for rank, (cid, text, meta, dist) in enumerate(arm):
                e = fused.setdefault(cid, {"id": cid, "text": text,
                                           "meta": meta or {}, "dist": None,
                                           "score": 0.0})
                e["score"] += 1.0 / (RRF_K + rank + 1)
                if dist is not None:
                    e["dist"] = dist
        top = sorted(fused.values(), key=lambda e: -e["score"])[:n]
        return [(e["id"], e["text"], e["meta"], e["dist"], e["score"]) for e in top]


class ChromaBackend:
    name = "chroma"

    def __init__(self):
        self._col = load_vectordb()

    def count(self):
        return self._col.count()

    def query(self, emb, question, n, filters, mode):
        kwargs = {"query_embeddings": [emb], "n_results": max(1, min(n, 25)),
                  "include": ["documents", "distances", "metadatas"]}
        if filters:
            kwargs["where"] = filters
        res = self._col.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        return [(ids[i],
                 docs[i] if i < len(docs) else "",
                 metas[i] if i < len(metas) and metas[i] else {},
                 dists[i] if i < len(dists) else None,
                 None)
                for i in range(len(ids))]


if PG_DSN:
    BACKEND = PgBackend(PG_DSN)
else:
    log.info("PG_DSN not set — falling back to the ChromaDB release")
    BACKEND = ChromaBackend()
log.info("Ready: backend=%s, %d chunks", BACKEND.name, BACKEND.count())


_RERANKER = None


def _get_reranker():
    """Lazy-load the cross-encoder on the first reranked request (~2GB RSS).
    Callers must hold _LOCK."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        log.info("Loading reranker %s (first RERANK request)…", RERANK_MODEL)
        _RERANKER = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
        log.info("Reranker loaded.")
    return _RERANKER


def retrieve(question, n=5, filters=None, mode="hybrid"):
    n = max(1, min(n, 50))  # deep research profiles page the full fusion pool
    # With reranking on, pull the whole fused pool and let the cross-encoder
    # pick the final top-n; otherwise fetch exactly n as before.
    fetch = max(POOL, n) if RERANK else n
    with _LOCK:
        emb = EMBEDDER.encode([question], normalize_embeddings=True)[0].tolist()
        rows = BACKEND.query(emb, question, fetch, filters, mode)
        rerank_scores = None
        if RERANK and rows:
            scores = _get_reranker().predict(
                [(question, r[1] or "") for r in rows])
            order = sorted(range(len(rows)),
                           key=lambda i: -float(scores[i]))[:n]
            rerank_scores = [float(scores[i]) for i in order]
            rows = [rows[i] for i in order]
    out = []
    for i, (cid, text, meta, dist, score) in enumerate(rows):
        chunk = {
            "id":            cid,
            "text":          text,
            "distance":      dist,
            "score":         score,
            "source":        meta.get("filename") or meta.get("source") or "",
            "document_type": meta.get("document_type", ""),
            "summary":       meta.get("summary", ""),
        }
        if rerank_scores is not None:
            chunk["rerank_score"] = rerank_scores[i]
        out.append(chunk)
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "backend": BACKEND.name,
                             "collection": COLLECTION_NAME,
                             "chunks": BACKEND.count()})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/retrieve":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = (payload.get("question") or "").strip()
            if not question:
                self._send(400, {"error": "question is required"})
                return
            n = int(payload.get("n") or 5)
            mode = payload.get("mode") or "hybrid"
            chunks = retrieve(question, n=n, filters=payload.get("filters"),
                              mode=mode)
            self._send(200, {"chunks": chunks})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001 — surface any retrieval error to the caller
            log.exception("retrieve failed")
            self._send(500, {"error": str(e)})

    def log_message(self, *args):  # keep the console quiet
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("Retriever listening on http://%s:%d", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.shutdown()
