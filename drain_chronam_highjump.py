#!/usr/bin/env python3
"""Chronicling America (loc.gov) drain: 1946-48 newspaper pages covering
Operation Highjump / Admiral Byrd / Task Force 68. Page full-text preferred,
page PDF fallback. Ledgered under the highjump source."""
import json
import time
import urllib.parse
from pathlib import Path

from drain_wikipedia_ufologists import fetch, Ledger, WIKI_UA, slug

ROOT = Path(__file__).parent
OUT = ROOT / "data/downloads/highjump"
QUERIES = ['"operation highjump"', '"admiral byrd" antarctic',
           '"task force 68" antarctic']


def main():
    led = Ledger()
    got = 0
    seen = set()
    for q in QUERIES:
        for page in range(1, 4):
            u = ("https://www.loc.gov/collections/chronicling-america/?"
                 + urllib.parse.urlencode({"q": q, "dates": "1946/1948",
                                           "fo": "json", "c": "50", "sp": str(page)}))
            try:
                raw, _ = fetch(u, WIKI_UA, timeout=90)
                d = json.loads(raw)
            except Exception as e:
                print(q, page, "search fail", str(e)[:60], flush=True)
                break
            results = d.get("results", [])
            if not results:
                break
            for r in results:
                rid = r.get("id") or r.get("url")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                rid = rid.replace("http://", "https://")
                date = (r.get("date") or "")[:10]
                pt = r.get("partof_title")
                title = pt[0] if isinstance(pt, list) and pt else "paper"
                name = f"ChronAm_{date}_{slug(title, 40)}"
                try:
                    join = "&" if "?" in rid else "?"
                    iraw, _ = fetch(rid + join + "fo=json", WIKI_UA, timeout=90)
                    item = json.loads(iraw)
                    res = (item.get("resources") or [{}])[0]
                    ok = False
                    ftu = res.get("fulltext_file")
                    if ftu:
                        traw, _ = fetch(ftu, WIKI_UA, timeout=90)
                        if len(traw) > 500 and b"<html" not in traw[:200].lower():
                            ok = led.add(rid, traw, OUT / (name + ".txt"),
                                         "highjump", "txt", "chronam")
                    if not ok and res.get("pdf"):
                        praw, _ = fetch(res["pdf"], WIKI_UA, timeout=120)
                        if praw[:5] == b"%PDF-":
                            ok = led.add(rid, praw, OUT / (name + ".pdf"),
                                         "highjump", "pdf", "chronam")
                    if ok:
                        got += 1
                        time.sleep(3)
                        continue
                except Exception as e:
                    print("item json fail", rid[:60], str(e)[:50], flush=True)
                # legacy ocr.txt fallback: /resource/{lccn}/{date}/ed-1/?sp=N
                m = None
                import re as _re
                m = _re.search(r"/resource/([a-z0-9]+)/([0-9-]+)/(ed-\d+)/\?sp=(\d+)", rid)
                if m:
                    ou = (f"https://chroniclingamerica.loc.gov/lccn/{m.group(1)}/"
                          f"{m.group(2)}/{m.group(3)}/seq-{m.group(4)}/ocr.txt")
                    try:
                        oraw, _ = fetch(ou, WIKI_UA, timeout=90)
                        if len(oraw) > 500 and b"<html" not in oraw[:200].lower():
                            if led.add(rid, oraw, OUT / (name + ".txt"),
                                       "highjump", "txt", "chronam-ocr"):
                                got += 1
                    except Exception as e:
                        print("ocr fail", ou[:60], str(e)[:50], flush=True)
                time.sleep(3)
            time.sleep(4)
    led.save()
    print("DONE chronam pages acquired:", got, flush=True)


if __name__ == "__main__":
    main()
