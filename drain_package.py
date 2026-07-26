#!/usr/bin/env python3
"""
Package drain (2026-07): the under-$250 acquisition batch.

  bluebook   172 missing case items + SIGN/misc/admin items + status reports
             + indexes + the 7 free NARA reels (text-PDF variants) — all from
             the Internet Archive
  gross      Loren Gross "UFOs: A History" complete series from SOHP
             (sohp.us/collections/ufos-a-history/)
  friedman   stantonfriedman.com article pages via the Wayback Machine

Every file is ledgered against its canonical source URL; exact duplicates
are skipped by sha256 against the whole ledger. Idempotent — re-runs skip
already-ledgered URLs.

Usage: python drain_package.py [bluebook|gross|friedman|all]
"""
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
UA = {"User-Agent": "uap-archive-drain/1.0"}

BB_MISC_ITEMS = ["usaf-signm", "misc-pbb", "misc-afcs-9", "misc-afosr-4",
                 "BlueBookArtifacts", "project-blue-book-complete-status-reports",
                 "ProjectBlueBookIndexes"]


def fetch(url, timeout=300, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


class Ledger:
    def __init__(self):
        self.data = json.loads(LEDGER.read_text())
        self.shas = {e.get("sha256") for e in self.data["completed"].values()}
        self.added = self.dups = self.failed = 0

    def has(self, url):
        return url in self.data["completed"]

    def add(self, url, raw, dest: Path, kind, note=""):
        sha = hashlib.sha256(raw).hexdigest()
        if sha in self.shas:
            self.dups += 1
            print(f"  dup  {dest.name}", flush=True)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        self.data["completed"][url] = {
            "path": str(dest.relative_to(ROOT)), "sha256": sha,
            "bytes": len(raw), "source": dest.relative_to(
                ROOT / "data/downloads").parts[0], "kind": kind,
            **({"via": note} if note else {}),
        }
        self.shas.add(sha)
        self.added += 1
        print(f"  ok   {dest.name} ({len(raw):,}b)", flush=True)

    def save(self):
        LEDGER.write_text(json.dumps(self.data, indent=1))
        print(f"ledger saved: +{self.added}, {self.dups} dups, {self.failed} failed")


def ia_item_pdfs(identifier):
    """(name, url) for the content PDFs of an IA item. Prefers the *_text.pdf
    variant when both exist (same pages, compressed, OCR text layer)."""
    m = json.loads(fetch(f"https://archive.org/metadata/{identifier}"))
    files = [f["name"] for f in m.get("files", [])
             if f["name"].lower().endswith(".pdf")]
    keep = []
    for n in files:
        if n.endswith("_text.pdf") or (n[:-4] + "_text.pdf") not in files:
            keep.append(n)
    base = f"https://archive.org/download/{identifier}/"
    return [(n, base + urllib.parse.quote(n)) for n in keep]


def drain_bluebook(led):
    dest = ROOT / "data/downloads/internet_archive"
    missing = (ROOT / "data/bluebook_gap/missing_items.txt").read_text().split()
    for i, ident in enumerate(missing + BB_MISC_ITEMS, 1):
        try:
            for name, url in ia_item_pdfs(ident):
                if led.has(url):
                    continue
                safe = re.sub(r"[^\w.\- ]", "_", name)
                led.add(url, fetch(url), dest / safe, "pdf", f"ia:{ident}")
                time.sleep(1)
        except Exception as e:
            led.failed += 1
            print(f"  FAIL {ident}: {str(e)[:80]}", flush=True)
        if i % 25 == 0:
            led.save()
            print(f"[bluebook {i}/{len(missing) + len(BB_MISC_ITEMS)}]", flush=True)


def drain_gross(led):
    dest = ROOT / "data/downloads/sohp_gross"
    page = fetch("https://sohp.us/collections/ufos-a-history/").decode("utf-8", "replace")
    links = sorted({urllib.parse.urljoin("https://sohp.us/", h)
                    for h in re.findall(r'href="([^"]+\.pdf)"', page, re.I)})
    print(f"gross: {len(links)} PDFs listed on SOHP", flush=True)
    for url in links:
        if led.has(url):
            continue
        try:
            name = re.sub(r"[^\w.\- ]", "_", urllib.parse.unquote(url.rsplit("/", 1)[-1]))
            raw = fetch(url)
            if not raw.startswith(b"%PDF"):
                led.failed += 1
                continue
            led.add(url, raw, dest / name, "pdf", "sohp")
            time.sleep(1.5)
        except Exception as e:
            led.failed += 1
            print(f"  FAIL {url.rsplit('/',1)[-1][:50]}: {str(e)[:60]}", flush=True)


def drain_friedman(led):
    dest = ROOT / "data/downloads/stantonfriedman"
    rows = json.loads(fetch(
        "http://web.archive.org/cdx/search/cdx?url=stantonfriedman.com*"
        "&output=json&collapse=urlkey&fl=original,timestamp,statuscode&limit=3000"))[1:]
    # keep content pages: dated articles plus top-level named pages
    keep = {}
    for u, ts, s in rows:
        if s != "200":
            continue
        uq = urllib.parse.unquote(u)
        if re.search(r"ptp=articles&fdt=[\d.]+", uq) or \
           re.search(r"stantonfriedman\.com/?(index\.\w+)?(\?ptp=\w+)?$", uq):
            key = re.sub(r"&prt=\d+", "", uq)
            if key not in keep or ts > keep[key][1]:
                keep[key] = (u, ts)
    print(f"friedman: {len(keep)} distinct pages", flush=True)
    for key, (u, ts) in sorted(keep.items()):
        if led.has(u):
            continue
        try:
            raw = fetch(f"http://web.archive.org/web/{ts}id_/{u}")
            m = re.search(r"fdt=([\d.]+)", key)
            name = ("article_" + m.group(1) if m else
                    re.sub(r"[^\w.\-]", "_", key.split(".com", 1)[1] or "home")[:60]) + ".html"
            led.add(u, raw, dest / name, "html", f"wayback:{ts}")
            time.sleep(1.5)
        except Exception as e:
            led.failed += 1
            print(f"  FAIL {key[:60]}: {str(e)[:60]}", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    led = Ledger()
    if which in ("bluebook", "all"):
        drain_bluebook(led)
    if which in ("gross", "all"):
        drain_gross(led)
    if which in ("friedman", "all"):
        drain_friedman(led)
    led.save()


if __name__ == "__main__":
    sys.exit(main())
