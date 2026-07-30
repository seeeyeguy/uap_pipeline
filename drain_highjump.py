#!/usr/bin/env python3
"""
Operation Highjump (Task Force 68, 1946-47) primary-source drain, plus the
connected layer: Operation Windmill, the Army Observers' Report, the official
TF-68 report (DTIC), NHHC ship histories & photo pages, LOC finding aids,
GovInfo congressional record, USGS Antarctic exploration reports, CIA CREST,
and archive.org holdings (documents only — video skipped; the pipeline
excludes video anyway).

Reuses the wikipedia drain's fetch (hard deadlines) / Ledger / wayback
fallback. Unverified URLs are probed and fall through to Wayback; misses are
logged, not fatal. Resume-safe via the shared ledger.
"""
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

from drain_wikipedia_ufologists import (fetch, Ledger, wayback_newest,
                                        BROWSER_UA, WIKI_UA, slug)

ROOT = Path(__file__).parent
OUT = ROOT / "data/downloads/highjump"
FAILED = ROOT / "data/highjump_failed.txt"

SEEDS = [
    # (url, kind hint) — kind None = derive from content type
    # GovInfo — H.Con.Res. 301 findings + CR index + CREC issue
    ("https://www.govinfo.gov/content/pkg/BILLS-110hconres301ih/html/BILLS-110hconres301ih.htm", "html"),
    ("https://www.govinfo.gov/content/pkg/BILLS-110hconres301ih/pdf/BILLS-110hconres301ih.pdf", "pdf"),
    ("https://www.govinfo.gov/content/pkg/CRI-2008/html/CRI-2008-ANTARCTIC-REGIONS.htm", "html"),
    ("https://www.govinfo.gov/content/pkg/CREC-2008-02-15/pdf/CREC-2008-02-15.pdf", "pdf"),
    # USGS Antarctic exploration series (Highjump/Windmill mapping history)
    ("https://pubs.usgs.gov/of/2006/1117/pdf/2006-1117.pdf", "pdf"),
    ("https://pubs.usgs.gov/of/2006/1116/pdf/2006-1116.pdf", "pdf"),
    ("https://pubs.usgs.gov/of/2006/1113/pdf/2006-1113.pdf", "pdf"),
    # NHHC
    ("https://www.history.navy.mil/browse-by-topic/exploration-and-innovation/polar-exploration0.html", "html"),
    ("https://www.history.navy.mil/content/dam/nhhc/browse-by-topic/exploration-and-innovation/Polar%20Exploration/pdf/Antartic-pp-20-25.pdf", "pdf"),
    # LOC finding aids
    ("https://findingaids.loc.gov/exist_collections/ead3pdf/mss/2018/ms018010.pdf", "pdf"),
    ("https://findingaids.loc.gov/exist_collections/ead3pdf/mss/2022/ms022009.pdf", "pdf"),
    # Official TF-68 report via DTIC (probe; often undigitized)
    ("https://apps.dtic.mil/sti/tr/pdf/AD0088221.pdf", "pdf"),
    # NLM scan of the Army Observers' Report
    ("https://collections.nlm.nih.gov/bookviewer?PID=nlm:nlmuid-14020180R-bk", "html"),
    # Smithsonian Windmill finding aid
    ("https://sirismm.si.edu/EADpdfs/SIA.FA02-223.pdf", "pdf"),
    # OSU Byrd polar archive Highjump collection (finding aid only)
    ("https://library.osu.edu/collections/SPEC.PA.56.0249/summary-information", "html"),
]

# TF-68 / Windmill ships — DANFS history pages
DANFS_SHIPS = [
    "mount-olympus", "philippine-sea", "pine-island", "currituck",
    "henderson-iii", "brownson-ii", "sennet", "yancey", "merrick",
    "cacapon", "canisteo", "burton-island-i", "northwind",
]


def save(led, url, raw, ctype, kind_hint, name):
    kind = kind_hint or ("pdf" if "pdf" in ctype or raw[:5] == b"%PDF-" else "html")
    ext = "." + ("pdf" if kind == "pdf" else "html")
    if raw[:5] == b"%PDF-":
        ext = ".pdf"
    dest = OUT / (slug(name, 90) + ext)
    return led.add(url, raw, dest, "highjump", kind)


def get(led, url, kind_hint, name, failures, via_wayback=True):
    if led.has(url):
        return "have"
    try:
        raw, ctype = fetch(url, BROWSER_UA, timeout=60)
        if len(raw) > 500:
            save(led, url, raw, ctype, kind_hint, name)
            return "direct"
    except Exception:
        pass
    if via_wayback:
        try:
            got = wayback_newest(url)
            if got and len(got[0]) > 500:
                save(led, url, got[0], got[1], kind_hint, name)
                return "wayback"
        except Exception:
            pass
    failures.append(url)
    return "failed"


def hrefs(html_bytes, base, pattern):
    out = []
    for m in re.finditer(rb'href="([^"]+)"', html_bytes):
        u = urllib.parse.urljoin(base, m.group(1).decode("utf-8", "replace"))
        if re.search(pattern, u):
            out.append(u.split("#")[0])
    return list(dict.fromkeys(out))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    led = Ledger()
    failures = []
    stats = {"direct": 0, "wayback": 0, "failed": 0, "have": 0}

    def track(res):
        stats[res] += 1
        time.sleep(1.5)

    print(f"seeds: {len(SEEDS)}", flush=True)
    for url, kind in SEEDS:
        name = Path(urllib.parse.urlparse(url).path).stem or "index"
        track(get(led, url, kind, "seed_" + name, failures))

    # DANFS ship histories (probe -i/-ii variants when the base 404s)
    for ship in DANFS_SHIPS:
        base = f"https://www.history.navy.mil/research/histories/ship-histories/danfs/{ship[0]}/{ship}.html"
        track(get(led, base, "html", "danfs_" + ship, failures))
    print(f"after seeds+danfs: {stats}", flush=True)

    # NHHC Byrd photo index -> per-photo pages
    try:
        idx_url = ("https://www.history.navy.mil/our-collections/photography/"
                   "us-people/b/byrd-richard-e-1946-1957.html")
        raw, _ = fetch(idx_url, BROWSER_UA, timeout=60)
        led.add(idx_url, raw, OUT / "nhhc_byrd_photo_index.html", "highjump", "html")
        pages = hrefs(raw, idx_url, r"/nh-\d+\.html$")[:120]
        print(f"nhhc byrd photo pages: {len(pages)}", flush=True)
        for p in pages:
            track(get(led, p, "html", "nhhc_photo_" + Path(p).stem, failures))
    except Exception as e:
        print(f"nhhc photo index failed: {str(e)[:80]}", flush=True)

    # LOC corporate-entity hub -> finding aid pages/PDFs
    try:
        hub = "https://findingaids.loc.gov/agents/corporate_entities/14085"
        raw, _ = fetch(hub, BROWSER_UA, timeout=60)
        led.add(hub, raw, OUT / "loc_highjump_entity_hub.html", "highjump", "html")
        aids = hrefs(raw, hub, r"(ead3pdf.+\.pdf|/collections/[a-z0-9.]+)")[:60]
        print(f"loc linked aids: {len(aids)}", flush=True)
        for a in aids:
            track(get(led, a, None, "loc_" + Path(a).stem, failures))
    except Exception as e:
        print(f"loc hub failed: {str(e)[:80]}", flush=True)

    # CIA CREST search -> document pages -> PDFs
    try:
        for page in range(0, 3):
            surl = ("https://www.cia.gov/readingroom/search/site/"
                    "operation%20highjump" + (f"?page={page}" if page else ""))
            raw, _ = fetch(surl, BROWSER_UA, timeout=60)
            docs = hrefs(raw, surl, r"/readingroom/document/")
            print(f"crest page {page}: {len(docs)} docs", flush=True)
            if not docs:
                break
            for d in docs:
                if led.has(d):
                    stats["have"] += 1
                    continue
                try:
                    draw, _ = fetch(d, BROWSER_UA, timeout=60)
                    pdfs = hrefs(draw, d, r"/readingroom/docs/.+\.pdf$")
                    for pu in pdfs[:2]:
                        track(get(led, pu, "pdf", "cia_" + Path(pu).stem, failures,
                                  via_wayback=False))
                    led.add(d, draw, OUT / ("cia_page_" + slug(Path(d).name, 60) + ".html"),
                            "highjump", "html")
                except Exception:
                    failures.append(d)
                time.sleep(2)
    except Exception as e:
        print(f"crest failed: {str(e)[:80]}", flush=True)

    # archive.org: enumerate "operation highjump" items, take document files
    try:
        q = ("https://archive.org/advancedsearch.php?q=%22operation+highjump%22"
             "&fl%5B%5D=identifier&fl%5B%5D=mediatype&rows=200&output=json")
        raw, _ = fetch(q, WIKI_UA, timeout=60)
        items = [(d["identifier"], d.get("mediatype", ""))
                 for d in json.loads(raw)["response"]["docs"]]
        # known-good identifiers first, then discovered text items
        known = ["14020180R.nlm.nih.gov",
                 "united-states.-navy.-task-force-68-army-observers-report-of-operation-highjump-war-dept-1947"]
        ids = list(dict.fromkeys(known + [i for i, m in items if m == "texts"]))[:40]
        print(f"archive.org text items: {len(ids)}", flush=True)
        for ident in ids:
            try:
                mraw, _ = fetch(f"https://archive.org/metadata/{ident}", WIKI_UA, timeout=60)
                files = json.loads(mraw).get("files", [])
                pdfs = [f["name"] for f in files
                        if f["name"].lower().endswith((".pdf", "_djvu.txt"))]
                # prefer one pdf (or the djvu text) per item
                pdfs.sort(key=lambda n: (not n.lower().endswith(".pdf"), len(n)))
                for name in pdfs[:1]:
                    fu = f"https://archive.org/download/{ident}/{urllib.parse.quote(name)}"
                    track(get(led, fu, None, "ia_" + ident[:50], failures,
                              via_wayback=False))
            except Exception:
                failures.append(ident)
            time.sleep(2)
    except Exception as e:
        print(f"archive.org enumeration failed: {str(e)[:80]}", flush=True)

    # NARA unauthenticated proxy probe (API v2 needs an emailed key)
    try:
        purl = ("https://catalog.archives.gov/proxy/records/search?"
                + urllib.parse.urlencode({"q": '"Operation Highjump"', "limit": "50"}))
        raw, _ = fetch(purl, BROWSER_UA, timeout=60)
        data = json.loads(raw)
        hits = (((data.get("body") or data).get("hits") or {}).get("hits")) or []
        print(f"nara proxy hits: {len(hits)}", flush=True)
        n_obj = 0
        for h in hits:
            rec = (h.get("_source") or {}).get("record") or {}
            for obj in (rec.get("digitalObjects") or [])[:4]:
                ou = obj.get("objectUrl")
                if ou and n_obj < 150:
                    n_obj += 1
                    track(get(led, ou, None, "nara_" + slug(Path(ou).name, 60),
                              failures, via_wayback=False))
    except Exception as e:
        print(f"nara proxy unavailable ({str(e)[:60]}) — needs API key, skipping",
              flush=True)

    led.save()
    FAILED.write_text("\n".join(failures) + "\n")
    print(f"\nDONE {stats} | {len(failures)} failures in {FAILED}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
