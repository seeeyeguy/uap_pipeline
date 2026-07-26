"""
Build a PDF-only download manifest for the REMAINDER of the Internet Archive
`project-blue-book` collection (10,764 items; we hold ~1,181).

Past IA tranches pulled every derivative (.gz/.zip/_djvu.txt) — 2.5x the bytes
for no corpus value. This manifest lists ONLY the original PDFs (falling back
to the derivative Text PDF when an item has no original), so the downloader
(`downloader.py download_all`, or main.py download --manifest) stays PDF-only.

Skips items that already have any file in the ledger. Output:
  data/manifest_bluebook/sources.json   (downloader-compatible)

Usage: .venv/bin/python build_bluebook_manifest.py
"""
import concurrent.futures as cf
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

COLLECTION = "project-blue-book"
OUT_DIR = Path("data/manifest_bluebook")
LEDGER = Path("data/downloads.json")
THREADS = 8

session = requests.Session()
session.headers["User-Agent"] = "uap-pipeline/1.0 (bluebook manifest builder)"


def scrape_collection() -> dict:
    """identifier -> item_size for the whole collection."""
    items, cursor = {}, None
    while True:
        url = (f"https://archive.org/services/search/v1/scrape"
               f"?q=collection%3A{COLLECTION}&fields=identifier,item_size&count=10000")
        if cursor:
            url += f"&cursor={cursor}"
        d = session.get(url, timeout=60).json()
        for x in d["items"]:
            items[x["identifier"]] = x.get("item_size", 0)
        cursor = d.get("cursor")
        if not cursor:
            return items


def item_pdfs(ident: str, retries: int = 3):
    """[(filename, bytes)] for the item's original PDFs (else derivative PDFs)."""
    for attempt in range(retries):
        try:
            d = session.get(f"https://archive.org/metadata/{ident}", timeout=60).json()
            files = d.get("files", [])
            pdfs = [f for f in files if f.get("name", "").lower().endswith(".pdf")]
            orig = [f for f in pdfs if f.get("source") == "original"]
            chosen = orig or pdfs
            return [(f["name"], int(f.get("size", 0))) for f in chosen]
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)


def main():
    have = set()
    if LEDGER.exists():
        for url in json.loads(LEDGER.read_text()).get("completed", {}):
            m = re.match(r"https?://archive\.org/download/([^/]+)/", url)
            if m:
                have.add(m.group(1))

    print("scraping collection listing…")
    all_items = scrape_collection()
    todo = sorted(set(all_items) - have)
    print(f"{len(all_items)} items in {COLLECTION}; {len(have & set(all_items))} held; "
          f"{len(todo)} to enumerate")

    resources, failed, done = [], [], 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(THREADS) as ex:
        for ident, pdfs in zip(todo, ex.map(item_pdfs, todo)):
            done += 1
            if pdfs is None:
                failed.append(ident)
            else:
                for name, size in pdfs:
                    resources.append({
                        "source": "internet_archive",
                        "title": f"{ident}/{name}",
                        "url": f"https://archive.org/download/{ident}/{quote(name)}",
                        "kind": "pdf",
                        "category": "government_archive",
                        "size_hint": str(size),
                        "sha256": None,
                        "requires_ocr": True,
                        "verified": True,
                        "notes": "project-blue-book completion (PDF-only)",
                        "discovered_by": "build_bluebook_manifest",
                    })
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)} items ({rate:.1f}/s, "
                      f"{(len(todo) - done) / rate / 60:.0f} min left), "
                      f"{len(resources)} PDFs so far", flush=True)

    total_bytes = sum(int(r["size_hint"]) for r in resources)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "build_bluebook_manifest",
        "scrape": True,
        "scrape_errors": {"metadata_failed": failed} if failed else {},
        "source_count": 1,
        "resource_count": len(resources),
        "sources": {"internet_archive": {
            "title": "IA project-blue-book completion (PDF-only)"}},
        "resources": resources,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sources.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT_DIR}/sources.json: {len(resources)} PDFs, "
          f"{total_bytes / 1e9:.1f} GB, {len(failed)} items failed metadata")


if __name__ == "__main__":
    main()
