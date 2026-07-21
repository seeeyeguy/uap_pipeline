#!/usr/bin/env python3
"""
Drain every AARO PDF (aaro.mil/Portals/136/PDFs/*) via the Wayback Machine.

aaro.mil's WAF blocks scripted fetches outright (curl, headless Chrome, even
page navigation), but the Internet Archive crawls it fine — so enumerate via
the CDX index and pull each file's latest snapshot with the `id_` raw-bytes
form. Files are ledgered against their canonical aaro.mil URL, so the app's
"original source" links point at AARO, not the Wayback copy.

Exact-duplicate uploads (renamed variants of the same file) are skipped by
sha256 against both the existing ledger and this run.

Usage:  python drain_aaro_wayback.py [--apply]   # dry-run lists the plan
"""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
LEDGER = ROOT / "data/downloads.json"
DEST = ROOT / "data/downloads/aaro"
CDX = ("http://web.archive.org/cdx/search/cdx?url=aaro.mil/Portals/136/PDFs*"
       "&output=json&fl=original,timestamp,statuscode&limit=2000")


def fetch(url, timeout=180, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "uap-archive-drain/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rows = json.loads(fetch(CDX))[1:]
    # newest OK snapshot per canonical URL
    latest = {}
    for original, ts, status in rows:
        if status != "200" or not original.lower().endswith(".pdf"):
            continue
        if original not in latest or ts > latest[original]:
            latest[original] = ts

    ledger = json.loads(LEDGER.read_text())
    have_urls = set(ledger["completed"])
    have_shas = {e.get("sha256") for e in ledger["completed"].values()}

    todo = [(u, t) for u, t in sorted(latest.items()) if u not in have_urls]
    print(f"{len(latest)} archived AARO PDFs | {len(latest) - len(todo)} already "
          f"ledgered | {len(todo)} to fetch")
    if not args.apply:
        for u, t in todo:
            print("  would fetch:", urllib.parse.unquote(u).split("/PDFs/", 1)[1], "@", t[:8])
        print("dry run — pass --apply to download")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    added = skipped_dup = failed = 0
    for u, ts in todo:
        name = re.sub(r"[^\w.\- ]", "_",
                      urllib.parse.unquote(u).rsplit("/", 1)[-1]).strip()
        snap = f"http://web.archive.org/web/{ts}id_/{u}"
        try:
            raw = fetch(snap)
        except Exception as e:
            print(f"  FAIL {name}: {str(e)[:80]}", flush=True)
            failed += 1
            continue
        if not raw.startswith(b"%PDF"):
            print(f"  FAIL {name}: not a PDF ({len(raw)} bytes)", flush=True)
            failed += 1
            continue
        sha = hashlib.sha256(raw).hexdigest()
        if sha in have_shas:
            print(f"  dup  {name} (identical to an existing file)", flush=True)
            skipped_dup += 1
            continue
        path = DEST / name
        path.write_bytes(raw)
        ledger["completed"][u] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha, "bytes": len(raw),
            "source": "aaro", "kind": "pdf",
            "via": f"wayback:{ts}",
        }
        have_shas.add(sha)
        added += 1
        print(f"  ok   {name} ({len(raw):,} bytes)", flush=True)
        time.sleep(1.5)  # be kind to the Wayback Machine

    LEDGER.write_text(json.dumps(ledger, indent=1))
    print(f"\ndone: {added} added, {skipped_dup} duplicates skipped, {failed} failed")


if __name__ == "__main__":
    sys.exit(main())
