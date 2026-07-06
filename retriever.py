#!/usr/bin/env python3
"""
UAP Retrieval Sidecar
---------------------
A tiny HTTP service that the Go API (uap-api) calls for research-mode RAG.

It reuses the pipeline's *exact* embedding model (BAAI/bge-base-en-v1.5) and
ChromaDB store (data/vectordb, collection "uap_documents"), so a query lands in
the same 768-dim embedding space the corpus was built in. The Go server can't
do this itself: Chroma's default embedder is a different model/dimension, which
is why query_texts against this collection fails.

Endpoints:
  POST /retrieve  {"question": str, "n": int=5, "filters": {..}?}
       -> {"chunks": [{"id","text","distance","source","document_type","summary"}]}
  GET  /health    -> {"status":"ok","collection":..,"chunks":N}

Run (from the pipeline venv, so bge-base + the GPU are available):
  source .venv/bin/activate
  RETRIEVER_PORT=8001 python retriever.py
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from main import load_embed_model, load_vectordb, COLLECTION_NAME

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("retriever")

HOST = os.environ.get("RETRIEVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("RETRIEVER_PORT", "8001"))

log.info("Loading embedder + vector DB (this pulls the bge-base model)…")
EMBEDDER = load_embed_model()
COLLECTION = load_vectordb()
# Serialize GPU embedding + Chroma access; the research UI is low-QPS and this
# avoids concurrent-CUDA / Chroma reentrancy surprises under ThreadingHTTPServer.
_LOCK = threading.Lock()
log.info("Ready: collection '%s' has %d chunks", COLLECTION_NAME, COLLECTION.count())


def retrieve(question: str, n: int = 5, filters: dict | None = None) -> list[dict]:
    with _LOCK:
        emb = EMBEDDER.encode([question], normalize_embeddings=True).tolist()
        kwargs = {"query_embeddings": emb, "n_results": max(1, min(n, 25)),
                  "include": ["documents", "distances", "metadatas"]}
        if filters:
            kwargs["where"] = filters
        res = COLLECTION.query(**kwargs)

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    out = []
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        out.append({
            "id":            ids[i],
            "text":          docs[i] if i < len(docs) else "",
            "distance":      dists[i] if i < len(dists) else None,
            "source":        meta.get("filename") or meta.get("source") or "",
            "document_type": meta.get("document_type", ""),
            "summary":       meta.get("summary", ""),
        })
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
            self._send(200, {"status": "ok", "collection": COLLECTION_NAME,
                             "chunks": COLLECTION.count()})
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
            chunks = retrieve(question, n=n, filters=payload.get("filters"))
            self._send(200, {"chunks": chunks})
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
