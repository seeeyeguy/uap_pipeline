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

    def download_opts(sp):
        sp.add_argument("--include-videos", action="store_true",
                        help="also download video resources (large!)")
        sp.add_argument("--max-size-gb", type=float, default=None,
                        help="skip resources with a size hint above this")

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
