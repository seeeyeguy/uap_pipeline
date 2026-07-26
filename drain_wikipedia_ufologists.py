#!/usr/bin/env python3
"""
Biographical layer: Wikipedia articles for every major figure in ufology,
plus their cited references.

- Population: the Category:Ufologists tree (biographies) plus the UFO topic
  trees (sightings/incidents, phenomena, conspiracy theories, organizations,
  ufology, lists), plus a curated seed of adjacent majors. Fiction, film,
  media, religion and convention branches are excluded. The category graph
  is cyclic (conspiracy theories <-> sightings) — a shared seen-set guards
  the walk.
- Each article saved as HTML (text-native ingest path), ledgered against its
  canonical wikipedia URL. CC BY-SA; attribution is the source link.
- Each article's external links (up to REF_CAP) are acquired: direct fetch
  first with a browser UA; if that fails or returns junk, fall back to the
  newest archive.ph snapshot, then Wayback. Failures are recorded, not fatal.
  archive.ph rate-limits aggressively (429s at 3s spacing) — hits are spaced
  >=20s with backoff, and a double-429 puts it in a 10-minute cooldown.

Resume-safe: URLs already in the ledger are skipped. Run detached — the
polite delays put a full run in the hours range.

Usage: python drain_wikipedia_ufologists.py [--figures-only] [--limit N]
"""
import argparse
import gzip
import hashlib
import signal
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
LEDGER = ROOT / "data/downloads.json"
BIO_DIR = ROOT / "data/downloads/wikipedia_ufology"
REF_DIR = ROOT / "data/downloads/wikipedia_refs"
FAILED = ROOT / "data/wikipedia_refs_failed.txt"
WIKI_UA = {"User-Agent": "uap-archive-research/1.0 (hello@darkforestarchive.com)"}
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
REF_CAP = 50

SEED = [
    "J. Allen Hynek", "Jacques Vallée", "Stanton T. Friedman", "Donald Keyhoe",
    "Edward J. Ruppelt", "Edward Condon", "James E. McDonald", "Philip J. Klass",
    "Kenneth Arnold", "Betty and Barney Hill", "John E. Mack", "Jesse Marcel",
    "Leslie Kean", "Luis Elizondo", "David Grusch", "Christopher Mellon",
    "Harold E. Puthoff", "Carl Sagan", "Condon Committee", "Robertson Panel",
    "Lonnie Zamora", "Travis Walton", "Charles Hickson", "Kelly Johnston",
]

# (root category, recursion depth). Sightings runs deep: continent -> country
# -> sub-lists is where the individual incident articles live.
ROOTS = [
    ("Category:Ufologists", 2),
    ("Category:UFO sightings", 4),
    ("Category:UFO-related phenomena", 2),
    ("Category:UFO conspiracy theories", 2),
    ("Category:UFO organizations", 2),
    ("Category:Ufology", 2),
    ("Category:UFO-related lists", 1),
    ("Category:UFOs by type", 2),
    ("Category:Unidentified flying objects", 0),
]
EXCLUDE_CAT = re.compile(
    r"media|films?|fiction|novels?|television|comics?|games?|music|"
    r"religions?|conventions?|songs?|albums", re.I)
SKIP_TITLE = re.compile(
    r"^(Portal|Template|Draft|Wikipedia|File|Book|Module|Help|Talk|Category):")

SKIP_REF_HOSTS = re.compile(
    r"wikipedia\.org|wikimedia\.org|wikidata\.org|wiktionary\.org|"
    r"archive\.today|archive\.ph|worldcat\.org|doi\.org|jstor\.org$|"
    r"google\.[a-z.]+/search|google\.[a-z.]+/books")


class ArchivePh:
    """Throttled archive.today client: >=20s between hits, rotates mirror
    domains (same backend, separately throttled frontends) on 429."""
    DOMAINS = ["archive.ph", "archive.is", "archive.md", "archive.li"]

    def __init__(self):
        self.last = 0.0
        self.cooldown_until = 0.0
        self.di = 0

    def get(self, url):
        if time.monotonic() < self.cooldown_until:
            return None
        wait = self.last + 20 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        got429 = 0
        for attempt in range(len(self.DOMAINS)):
            dom = self.DOMAINS[self.di % len(self.DOMAINS)]
            self.last = time.monotonic()
            try:
                return fetch(f"https://{dom}/newest/" + url,
                             BROWSER_UA, timeout=60, retries=1)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    got429 += 1
                    self.di += 1        # try the next mirror
                    time.sleep(5)
                    continue
                return None
            except Exception:
                self.di += 1
                continue
        if got429 >= len(self.DOMAINS):
            self.cooldown_until = time.monotonic() + 600
        return None


def wayback_newest(url):
    """Newest Wayback snapshot as raw bytes, or None."""
    raw, _ = fetch("https://archive.org/wayback/available?"
                   + urllib.parse.urlencode({"url": url}), WIKI_UA, timeout=45)
    snap = (json.loads(raw).get("archived_snapshots") or {}).get("closest") or {}
    if not snap.get("available"):
        return None
    su = re.sub(r"(/web/\d{14})/", r"\1id_/", snap["url"])
    return fetch(su, BROWSER_UA, timeout=60, retries=1)


def arquivo_newest(url):
    """Newest snapshot from arquivo.pt (Portuguese national web archive —
    crawls international sites too). The Memento TimeTravel aggregator would
    be preferable but its domain is NXDOMAIN as of 2026-07."""
    raw, _ = fetch("https://arquivo.pt/wayback/cdx?"
                   + urllib.parse.urlencode({"url": url, "output": "json",
                                             "filter": "status:200"}),
                   WIKI_UA, timeout=45, retries=1)
    lines = raw.strip().splitlines()
    if not lines or b"urlkey" not in lines[-1]:
        return None
    rec = json.loads(lines[-1])  # oldest-first order; last line = newest capture
    return fetch(f"https://arquivo.pt/wayback/{rec['timestamp']}id_/{rec['url']}",
                 BROWSER_UA, timeout=60, retries=1)


_CC_INDEXES = []


def commoncrawl_fetch(url):
    """Pull the page straight out of Common Crawl's WARC files (newest three
    crawls) via the index API + an HTTP range request."""
    global _CC_INDEXES
    if not _CC_INDEXES:
        raw, _ = fetch("https://index.commoncrawl.org/collinfo.json",
                       WIKI_UA, timeout=45, retries=1)
        _CC_INDEXES = [c["id"] for c in json.loads(raw)[:3]]
    for idx in _CC_INDEXES:
        try:
            raw, _ = fetch(f"https://index.commoncrawl.org/{idx}-index?"
                           + urllib.parse.urlencode({"url": url, "output": "json",
                                                     "limit": "1", "filter": "status:200"}),
                           WIKI_UA, timeout=45, retries=1)
            rec = json.loads(raw.splitlines()[0])
            hdrs = dict(BROWSER_UA)
            off, ln = int(rec["offset"]), int(rec["length"])
            hdrs["Range"] = f"bytes={off}-{off + ln - 1}"
            seg, _ = fetch("https://data.commoncrawl.org/" + rec["filename"],
                           hdrs, timeout=60, retries=1)
            # gzipped WARC record: warc headers \r\n\r\n http headers \r\n\r\n body
            body = gzip.decompress(seg).split(b"\r\n\r\n", 2)
            if len(body) == 3 and len(body[2]) > 1024:
                return body[2], rec.get("mime", "text/html")
        except Exception:
            continue
    return None


def fetch(url, headers, timeout=45, retries=2):
    # urllib's timeout is per socket op — a drip-feeding server can pin a
    # fetch for hours. SIGALRM enforces a hard wall-clock deadline (main
    # thread only, which is all this script uses).
    def _alarm(signum, frame):
        raise TimeoutError(f"hard deadline: {url}")
    for attempt in range(retries):
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout * 2 + 30)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), (r.headers.get("Content-Type") or "")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    raise RuntimeError("unreachable")


def api(**params):
    params.update(action="query", format="json")
    u = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    raw, _ = fetch(u, WIKI_UA)
    return json.loads(raw)


def category_tree(cat, depth=2, seen=None):
    seen = seen if seen is not None else set()
    pages, subcats, cont = [], [], {}
    while True:
        r = api(list="categorymembers", cmtitle=cat, cmlimit=500,
                cmtype="page|subcat", **cont)
        for m in r["query"]["categorymembers"]:
            (subcats if m["ns"] == 14 else pages).append(m["title"])
        cont = r.get("continue") or {}
        if not cont:
            break
    out = {p for p in pages if not SKIP_TITLE.match(p)}
    if depth > 0:
        for sc in subcats:
            if sc in seen or EXCLUDE_CAT.search(sc.split(":", 1)[-1]):
                continue
            seen.add(sc)
            out |= category_tree(sc, depth - 1, seen)
    return out


def extlinks(title):
    links, cont = [], {}
    while True:
        r = api(prop="extlinks", titles=title, ellimit=500, redirects=1, **cont)
        for page in r["query"]["pages"].values():
            for l in page.get("extlinks", []):
                links.append(l["*"])
        cont = r.get("continue") or {}
        if not cont:
            break
    out, seen = [], set()
    for u in links:
        if not u.startswith("http") or SKIP_REF_HOSTS.search(
                urllib.parse.urlparse(u).netloc + urllib.parse.urlparse(u).path):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:REF_CAP]


def slug(s, n=70):
    return re.sub(r"[^\w.\-]+", "_", s).strip("_")[:n]


class Ledger:
    def __init__(self):
        self.data = json.loads(LEDGER.read_text())
        self.shas = {e.get("sha256") for e in self.data["completed"].values()}
        self.dirty = 0

    def has(self, url):
        return url in self.data["completed"]

    def add(self, url, raw, dest, source, kind, via=""):
        sha = hashlib.sha256(raw).hexdigest()
        if sha in self.shas:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        self.data["completed"][url] = {
            "path": str(dest.relative_to(ROOT)), "sha256": sha,
            "bytes": len(raw), "source": source, "kind": kind,
            **({"via": via} if via else {})}
        self.shas.add(sha)
        self.dirty += 1
        if self.dirty % 25 == 0:
            self.save()
        return True

    def save(self):
        LEDGER.write_text(json.dumps(self.data, indent=1))


def acquire_ref(led, aph, fig, idx, url, failures):
    if led.has(url):
        return "have"
    host = urllib.parse.urlparse(url).netloc.replace("www.", "")
    base = f"{slug(fig, 40)}__{idx:02d}_{slug(host, 30)}"

    def save(raw, ctype, via):
        ext = ".pdf" if "pdf" in ctype or raw[:5] == b"%PDF-" else ".html"
        led.add(url, raw, REF_DIR / (base + ext),
                "wikipedia_refs", ext[1:], via)

    # direct first
    try:
        raw, ctype = fetch(url, BROWSER_UA, timeout=40)
        if len(raw) > 1024 and ("pdf" in ctype or raw[:5] == b"%PDF-"
                                or "html" in ctype or "text" in ctype):
            save(raw, ctype, "direct")
            return "direct"
    except Exception:
        pass
    # archive.ph newest-snapshot fallback (throttled)
    got = aph.get(url)
    if got and len(got[0]) > 2048 and b"No results" not in got[0][:4096]:
        save(got[0], got[1], "archive.ph")
        return "archive.ph"
    # Wayback tertiary
    try:
        got = wayback_newest(url)
        if got and len(got[0]) > 1024:
            save(got[0], got[1], "wayback")
            return "wayback"
    except Exception:
        pass
    # arquivo.pt quaternary
    try:
        got = arquivo_newest(url)
        if got and len(got[0]) > 1024:
            save(got[0], got[1], "arquivo.pt")
            return "arquivo.pt"
    except Exception:
        pass
    # Common Crawl WARC extraction
    try:
        got = commoncrawl_fetch(url)
        if got:
            save(got[0], got[1], "commoncrawl")
            return "commoncrawl"
    except Exception:
        pass
    failures.append(f"{fig}\t{url}")
    return "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--population-file", help="skip the category walk; one title per line")
    ap.add_argument("--start", type=int, default=0, help="slice start (index into sorted population)")
    ap.add_argument("--end", type=int, default=0, help="slice end, exclusive (0 = to the end)")
    ap.add_argument("--skip-failed", help="failed-list from a prior run; don't retry those URLs")
    ap.add_argument("--retry-failed", help="failed-list to re-attempt with the full fallback chain, then exit")
    args = ap.parse_args()

    if args.population_file:
        figures = [l for l in Path(args.population_file).read_text().splitlines() if l.strip()]
    else:
        pop, seen = set(SEED), set()
        for cat, depth in ROOTS:
            pop |= category_tree(cat, depth, seen)
        figures = sorted(pop)
    total = len(figures)
    figures = figures[args.start:args.end or None]
    if args.limit:
        figures = figures[:args.limit]
    print(f"{len(figures)} figures (slice {args.start}:{args.end or total} of {total})", flush=True)

    skipfail_lines = []
    skipfail = set()
    if args.skip_failed and Path(args.skip_failed).exists():
        skipfail_lines = [l for l in Path(args.skip_failed).read_text().splitlines() if l.strip()]
        skipfail = {l.split("\t")[-1] for l in skipfail_lines}
        print(f"{len(skipfail)} known-dead URLs will not be retried", flush=True)

    led = Ledger()
    aph = ArchivePh()
    failures = list(skipfail_lines)
    stats = {"bio": 0, "direct": 0, "archive.ph": 0, "wayback": 0,
             "arquivo.pt": 0, "commoncrawl": 0, "failed": 0, "have": 0,
             "skipfail": 0}

    if args.retry_failed:
        # dedicated pass over a failed-list with the full fallback chain
        lines = [l.split("\t") for l in Path(args.retry_failed).read_text().splitlines()
                 if "\t" in l]
        print(f"retry pass over {len(lines)} failed refs", flush=True)
        failures = []
        try:
            for i, (fig, url) in enumerate(lines, 1):
                stats[acquire_ref(led, aph, fig, i % 100, url, failures)] += 1
                time.sleep(1.5)
                if i % 25 == 0:
                    print(f"[retry {i}/{len(lines)}] {stats}", flush=True)
        finally:
            led.save()
            FAILED.write_text("\n".join(failures) + "\n")
        print(f"\nDONE {stats} | still-dead refs in {FAILED}", flush=True)
        return 0
    for i, fig in enumerate(figures, 1):
        wiki_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
            fig.replace(" ", "_"))
        if not led.has(wiki_url):
            try:
                raw, _ = fetch("https://en.wikipedia.org/api/rest_v1/page/html/"
                               + urllib.parse.quote(fig.replace(" ", "_"), safe=""),
                               WIKI_UA, timeout=60)
                led.add(wiki_url, raw, BIO_DIR / (slug(fig) + ".html"),
                        "wikipedia_ufology", "html", "wikipedia-rest")
                stats["bio"] += 1
            except Exception as e:
                print(f"  BIO FAIL {fig}: {str(e)[:60]}", flush=True)
                continue
            time.sleep(1.5)
        if args.figures_only:
            continue
        try:
            refs = extlinks(fig)
        except Exception:
            refs = []
        try:
            for idx, url in enumerate(refs):
                if url in skipfail:
                    stats["skipfail"] += 1
                    continue
                stats[acquire_ref(led, aph, fig, idx, url, failures)] += 1
                time.sleep(1.5)
        finally:
            led.save()
            FAILED.write_text("\n".join(failures) + "\n")
        print(f"[{i}/{len(figures)}] {fig}: {len(refs)} refs | {stats}", flush=True)

    led.save()
    print(f"\nDONE {stats} | failures listed in {FAILED}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
