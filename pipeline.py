#!/usr/bin/env python3
"""
UAP Pipeline — One-Shot Orchestrator
------------------------------------
Single entry point that scours the internet for every known UAP/UFO
document source, manifests them, downloads everything, and builds the
vector database:

    python pipeline.py all                # the one-shot: discover -> download -> ingest

Individual stages:

    python pipeline.py discover           # static registry only (offline)
    python pipeline.py discover --scrape  # + live scrapers (full enumeration)
    python pipeline.py download           # fetch everything in sources.json
    python pipeline.py ingest             # OCR/enrich/embed data/downloads/
    python pipeline.py status             # progress report
    python pipeline.py query "roswell"    # search the vector DB

Useful filters:

    --sources wargov_pursue,fbi_vault     # restrict any stage to given sources
    --include-videos                      # also download video evidence
    --max-size-gb 5                       # skip resources larger than this
    --no-enrich                           # skip LLM enrichment (no API key needed)
    --force                               # redo completed work

Outputs:
    sources.json / SOURCES.md             # the downloadable-resource manifest
    data/downloads/<source>/              # raw corpus
    data/vectordb/                        # ChromaDB collection
    data/training/uap_qa_dataset.jsonl    # fine-tuning Q&A dataset
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")


def _sources_list(arg: str):
    return [s.strip() for s in arg.split(",") if s.strip()] if arg else None


def cmd_discover(args) -> dict:
    from manifest import discover, write_manifest
    m = discover(
        scrape=args.scrape,
        only_sources=_sources_list(args.sources),
        opts={
            "cia_max_pages":        args.cia_max_pages,
            "blackvault_max_pages": args.blackvault_max_pages,
            "afu_max_files":        args.afu_max_files,
            "crawl_max_pages":      args.crawl_max_pages,
            "ia_max_items":         args.ia_max_items,
        },
    )
    write_manifest(m, out_dir=args.manifest_dir)
    print(f"\nManifest: {m['resource_count']} resources from "
          f"{m['source_count']} sources -> "
          f"{Path(args.manifest_dir) / 'sources.json'} + SOURCES.md")
    if m["scrape_errors"]:
        print(f"Scrape errors ({len(m['scrape_errors'])}): "
              f"{', '.join(m['scrape_errors'])} — see SOURCES.md")
    return m


def _load_manifest(args) -> dict:
    path = Path(args.manifest_dir) / "sources.json"
    if not path.exists():
        log.error(f"{path} not found — run `python pipeline.py discover` first.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_download(args, manifest: dict = None):
    from downloader import download_all
    manifest = manifest or _load_manifest(args)
    return download_all(
        manifest,
        only_sources=_sources_list(args.sources),
        include_videos=args.include_videos,
        max_size_gb=args.max_size_gb,
        force=args.force,
        total_gb=args.total_gb,
    )


def cmd_ingest(args):
    from main import ingest_tree
    ingest_tree(args.downloads_dir, force=args.force, enrich=not args.no_enrich,
                sources=_sources_list(args.sources))


def cmd_all(args):
    print("═" * 60)
    print("  STAGE 1/3 — DISCOVER  (scraping every known source)")
    print("═" * 60)
    args.scrape = True
    m = cmd_discover(args)

    print("═" * 60)
    print("  STAGE 2/3 — DOWNLOAD")
    print("═" * 60)
    stats = cmd_download(args, manifest=m)

    print("═" * 60)
    print("  STAGE 3/3 — INGEST  (OCR -> enrich -> chunk -> embed)")
    print("═" * 60)
    cmd_ingest(args)

    print("═" * 60)
    print(f"  ONE-SHOT COMPLETE — downloads ok={stats['ok']} "
          f"failed={stats['failed']}; vector DB at ./data/vectordb")
    print("  Try: python pipeline.py query \"what happened at Roswell\"")
    print("═" * 60)


def cmd_publish(args):
    """
    Publish the build index (data/vectordb) to the serving side.

    The pipeline always ingests into data/vectordb (the BUILD index); the
    retriever serves a frozen, versioned COPY under data/vectordb_releases/.
    This lets ingestion run while the app stays up. Publish:

      1. refuses to run while an ingest is writing the build index
      2. validates the build index opens and counts its chunks
      3. copies it to data/vectordb_releases/v-<timestamp>/
      4. re-validates the copy
      5. points VECTORDB_HOST_DIR in <api-dir>/.env at the new release
      6. recreates the retriever container (~30-60s) and health-checks it
      7. prunes old releases (keeps --keep, never the one being served)

    Rollback: set VECTORDB_HOST_DIR to a previous release dir and
    `docker compose up -d retriever` in the api repo.
    """
    import re
    import shutil
    import subprocess
    import time
    from datetime import datetime, timezone

    import requests as rq

    build = Path("data/vectordb").resolve()
    releases_root = Path("data/vectordb_releases").resolve()
    api_dir = Path(args.api_dir).resolve()
    env_path = api_dir / ".env"

    # 1. never copy a database that's being written. Match only actual python
    #    processes: a wrapping shell's command line can contain the literal
    #    text "pipeline.py ingest" (e.g. a `cmd1 && cmd2` chain) and would
    #    self-match a plain pgrep -f.
    probe = subprocess.run(
        ["pgrep", "-f", r"pipeline\.py (ingest|all)"], capture_output=True, text=True)
    writers = []
    for pid in probe.stdout.split():
        try:
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
            if "python" in comm and int(pid) != os.getpid():
                writers.append(pid)
        except OSError:
            continue
    if writers:
        log.error(f"An ingest is running (pid {','.join(writers)}) — the build "
                  f"index is being written. Publish after it finishes.")
        sys.exit(1)

    # 2. validate the build index
    import chromadb
    from chromadb.config import Settings
    n_build = (chromadb.PersistentClient(path=str(build),
                                         settings=Settings(anonymized_telemetry=False))
               .get_collection("uap_documents").count())
    log.info(f"Build index OK: {n_build} chunks")

    # 3. copy to a versioned release
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    release = releases_root / f"v-{stamp}"
    log.info(f"Copying build index -> {release}")
    shutil.copytree(build, release)

    # 4. validate the copy independently
    n_rel = (chromadb.PersistentClient(path=str(release),
                                       settings=Settings(anonymized_telemetry=False))
             .get_collection("uap_documents").count())
    if n_rel != n_build:
        log.error(f"Release copy is inconsistent ({n_rel} != {n_build}) — "
                  f"removing it; serving config untouched.")
        shutil.rmtree(release)
        sys.exit(1)

    # 5. point the retriever mount at the release
    line = f"VECTORDB_HOST_DIR={release}"
    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if "VECTORDB_HOST_DIR" in env_text:
        env_text = re.sub(r"^VECTORDB_HOST_DIR=.*$", line, env_text, flags=re.M)
    else:
        env_text += f"\n# serving copy of the vector index (managed by pipeline.py publish)\n{line}\n"
    env_path.write_text(env_text, encoding="utf-8")
    log.info(f"{env_path} -> {release.name}")

    # 6. recreate the retriever and wait for health
    if args.no_restart:
        log.info("--no-restart: restart the retriever yourself to serve the new release.")
    else:
        subprocess.run(["docker", "compose", "up", "-d", "retriever"],
                       cwd=api_dir, check=True)
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                h = rq.get("http://127.0.0.1:8001/health", timeout=4).json()
                if h.get("chunks") == n_rel:
                    log.info(f"Retriever healthy, serving {h['chunks']} chunks "
                             f"from {release.name}")
                    break
                log.warning(f"Retriever serves {h.get('chunks')} chunks, "
                            f"expected {n_rel} — still starting?")
            except Exception:
                pass
            time.sleep(5)
        else:
            log.error("Retriever did not become healthy within 180s — "
                      "check `docker compose logs retriever` in the api repo.")
            sys.exit(1)

    # 7. prune old releases (never the one just published/served)
    old = sorted(d for d in releases_root.iterdir()
                 if d.is_dir() and d.name.startswith("v-") and d != release)
    for d in old[:max(0, len(old) - (args.keep - 1))]:
        log.info(f"Pruning old release: {d.name}")
        shutil.rmtree(d)


def cmd_inventory(args):
    """
    Generate INVENTORY.md + inventory.json — a complete index of everything
    searchable in the vector DB, joined with provenance and ingest records:

      vector index      -> what's actually searchable (per-file chunk counts,
                           incl. files that arrived inside ZIPs)
      data/downloads.json -> origin URL per downloaded file
      data/progress.json  -> ingest timestamp + sha256
      data/enriched/*.json -> document_type / ocr_quality / summary

    Run it after each ingest+publish; commit the outputs if you want the
    corpus history in git.
    """
    import re
    import sqlite3
    from collections import defaultdict
    from datetime import datetime, timezone

    index_dir = Path(args.index_dir)
    db = sqlite3.connect(f"file:{index_dir / 'chroma.sqlite3'}?mode=ro", uri=True)

    # Pivot chunk metadata: one row per embedding with the fields we need.
    rows = db.execute("""
        SELECT
          MAX(CASE WHEN key='source'        THEN string_value END),
          MAX(CASE WHEN key='filename'      THEN string_value END),
          MAX(CASE WHEN key='document_type' THEN string_value END),
          MAX(CASE WHEN key='ocr_quality'   THEN string_value END)
        FROM embedding_metadata
        WHERE key IN ('source','filename','document_type','ocr_quality')
        GROUP BY id""")

    def collection_of(src_path: str) -> tuple:
        """-> (collection_name, delivery) from a chunk's source path."""
        parts = Path(src_path or "").parts
        if "downloads" in parts:
            return parts[parts.index("downloads") + 1], "download"
        if len(parts) > 2 and parts[:2] == ("data", "raw"):
            return parts[2], "zip"
        return (parts[0] if parts else "unknown"), "unknown"

    # CSV/JSON row groups are pseudo-documents ("events__part0007.csv");
    # collapse them back into their parent file so the inventory reflects
    # what was actually ingested.
    part_re = re.compile(r"__part\d+(?=\.[^.]+$)")

    files = {}
    for src_path, fn, dtype, ocrq in rows:
        coll, delivery = collection_of(src_path)
        parent = part_re.sub("", fn or "")
        rec = files.setdefault((coll, parent), {
            "filename": parent, "source_path": src_path, "delivery": delivery,
            "chunks": 0, "_partnames": set(),
            "document_type": None, "ocr_quality": None,
        })
        rec["chunks"] += 1
        if parent != fn:
            rec["_partnames"].add(fn)
        rec["document_type"] = rec["document_type"] or dtype
        rec["ocr_quality"] = rec["ocr_quality"] or ocrq

    for rec in files.values():
        rec["parts"] = len(rec.pop("_partnames"))

    # Join: origin URLs (download ledger, keyed by URL -> local path)
    url_by_path = {}
    dl_path = Path("data/downloads.json")
    if dl_path.exists():
        dl = json.loads(dl_path.read_text(encoding="utf-8"))
        url_by_path = {v["path"]: u for u, v in dl.get("completed", {}).items()
                       if v.get("path")}

    # Join: ingest ledger (timestamp + sha256), matched by bare filename
    prog_by_name = {}
    prog_path = Path("data/progress.json")
    if prog_path.exists():
        prog = json.loads(prog_path.read_text(encoding="utf-8"))
        for k, v in prog.get("completed", {}).items():
            prog_by_name[Path(k).name] = v

    # Join: enrichment metadata, matched by filename stem
    enriched_by_stem = {}
    for f in Path("data/enriched").glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            enriched_by_stem[f.stem] = d
        except (json.JSONDecodeError, OSError):
            pass

    for (coll, fn), rec in files.items():
        p = prog_by_name.get(Path(fn).name, {})
        rec["ingested_at"] = (p.get("completed_at") or "")[:19]
        rec["sha256"] = p.get("sha256", "")
        rec["origin_url"] = url_by_path.get(rec["source_path"], "")
        e = enriched_by_stem.get(Path(fn).stem, {})
        rec["ocr_quality"] = rec["ocr_quality"] or e.get("ocr_quality") or ""
        rec["document_type"] = rec["document_type"] or e.get("document_type") or ""
        rec["summary"] = (e.get("summary") or "")[:200]

    # Aggregate per collection
    colls = defaultdict(list)
    for (coll, _), rec in files.items():
        colls[coll].append(rec)
    for recs in colls.values():
        recs.sort(key=lambda r: r["filename"] or "")

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total_chunks = sum(r["chunks"] for r in files.values())
    out = {
        "generated_at": generated,
        "index_dir": str(index_dir),
        "totals": {"collections": len(colls), "documents": len(files),
                   "chunks": total_chunks},
        "collections": {
            c: {"files": recs, "documents": len(recs),
                "chunks": sum(r["chunks"] for r in recs),
                "delivery": recs[0]["delivery"]}
            for c, recs in sorted(colls.items(),
                                  key=lambda kv: -sum(r["chunks"] for r in kv[1]))
        },
    }
    out_dir = Path(args.out_dir)
    (out_dir / "inventory.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# Corpus Inventory",
          f"",
          f"Generated {generated} from `{index_dir}` — "
          f"**{len(colls)} collections, {len(files)} documents, "
          f"{total_chunks:,} chunks**.",
          f""]
    for c, info in out["collections"].items():
        md.append(f"## {c} — {info['documents']} documents, "
                  f"{info['chunks']:,} chunks ({info['delivery']})")
        md.append("")
        md.append("| file | chunks | type | ocr | ingested | origin |")
        md.append("|---|---:|---|---|---|---|")
        for r in info["files"]:
            origin = f"[↗]({r['origin_url']})" if r["origin_url"] else ""
            name = r["filename"] + (f" ({r['parts']} parts)" if r["parts"] else "")
            md.append(f"| {name} | {r['chunks']} "
                      f"| {r['document_type'] or ''} | {r['ocr_quality'] or ''} "
                      f"| {r['ingested_at']} | {origin} |")
        md.append("")
    (out_dir / "INVENTORY.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Inventory: {len(colls)} collections, {len(files)} documents, "
          f"{total_chunks:,} chunks")
    print(f"  -> {out_dir / 'INVENTORY.md'}")
    print(f"  -> {out_dir / 'inventory.json'}")


def cmd_status(args):
    from main import load_progress, print_progress_report
    from downloader import load_ledger
    ledger = load_ledger()
    total = sum(v.get("bytes") or 0 for v in ledger["completed"].values())
    print(f"\nDownloads: {len(ledger['completed'])} complete "
          f"({total / (1 << 30):.2f} GB), {len(ledger['failed'])} failed")
    for url, info in list(ledger["failed"].items())[:20]:
        print(f"  FAILED [{info.get('source')}] {url}: {info.get('error', '')[:120]}")
    print_progress_report(load_progress())


def cmd_query(args):
    from main import Models, query, print_results
    models = Models()
    filters = {}
    if args.type:
        filters["document_type"] = args.type
    if args.period:
        filters["time_period"] = args.period
    if args.official:
        filters["official_source"] = "True"
    if args.radar:
        filters["radar_confirmation"] = "True"
    results = query(args.question, models.embedder, models.collection,
                    n_results=args.n, filters=filters or None)
    print_results(results)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="One-shot UAP document pipeline: discover -> download -> ingest -> query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Outputs:")[0],
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--sources", default=None,
                        help="comma-separated source ids to restrict to")
        sp.add_argument("--manifest-dir", default=".",
                        help="where sources.json / SOURCES.md live (default: repo root)")
        sp.add_argument("--force", action="store_true",
                        help="redo already-completed work")

    def discover_opts(sp):
        sp.add_argument("--scrape", action="store_true",
                        help="run live discovery scrapers (default for `all`)")
        sp.add_argument("--cia-max-pages", type=int, default=150)
        sp.add_argument("--blackvault-max-pages", type=int, default=25)
        sp.add_argument("--afu-max-files", type=int, default=2000)
        sp.add_argument("--crawl-max-pages", type=int, default=300)
        sp.add_argument("--ia-max-items", type=int, default=500,
                        help="max items to enumerate per archive.org collection (1 HTTP call each)")

    def download_opts(sp):
        sp.add_argument("--include-videos", action="store_true",
                        help="also download video resources (large!)")
        sp.add_argument("--max-size-gb", type=float, default=None,
                        help="skip individual resources with a size hint above this")
        sp.add_argument("--total-gb", type=float, default=None,
                        help="stop downloading once this many GB of NEW data has been fetched")

    def ingest_opts(sp):
        sp.add_argument("--downloads-dir", default="./data/downloads")
        sp.add_argument("--no-enrich", action="store_true",
                        help="skip Claude enrichment (no ANTHROPIC_API_KEY needed)")

    sp = sub.add_parser("discover", help="build sources.json + SOURCES.md")
    common(sp); discover_opts(sp)
    sp.set_defaults(fn=cmd_discover)

    sp = sub.add_parser("download", help="download everything in the manifest")
    common(sp); download_opts(sp)
    sp.set_defaults(fn=cmd_download)

    sp = sub.add_parser("ingest", help="OCR/enrich/embed the downloaded corpus")
    common(sp); ingest_opts(sp)
    sp.set_defaults(fn=cmd_ingest)

    sp = sub.add_parser("all", help="one-shot: discover --scrape, download, ingest")
    common(sp); discover_opts(sp); download_opts(sp); ingest_opts(sp)
    sp.set_defaults(fn=cmd_all)

    sp = sub.add_parser("publish",
                        help="copy the build index to a versioned release and point the retriever at it")
    sp.add_argument("--api-dir", default="/apps/uap-api",
                    help="uap-api repo (compose project + .env with VECTORDB_HOST_DIR)")
    sp.add_argument("--keep", type=int, default=3,
                    help="how many releases to retain (default 3)")
    sp.add_argument("--no-restart", action="store_true",
                    help="update .env but don't recreate the retriever container")
    sp.set_defaults(fn=cmd_publish)

    sp = sub.add_parser("inventory",
                        help="write INVENTORY.md + inventory.json indexing every searchable file")
    sp.add_argument("--index-dir", default="data/vectordb",
                    help="vector index to inventory (default: the build index)")
    sp.add_argument("--out-dir", default=".",
                    help="where to write INVENTORY.md / inventory.json")
    sp.set_defaults(fn=cmd_inventory)

    sp = sub.add_parser("status", help="download + ingestion progress report")
    common(sp)
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("query", help="query the vector DB")
    sp.add_argument("question")
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--type", default=None, help="document_type filter")
    sp.add_argument("--period", default=None, help="time_period filter")
    sp.add_argument("--official", action="store_true")
    sp.add_argument("--radar", action="store_true")
    sp.set_defaults(fn=cmd_query)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
