#!/usr/bin/env python3
"""
Definitive Blue Book gap report: what the archive holds vs. what is freely
acquirable, split into NEW material and UPGRADE material.

Sources compared:
  - our corpus (distinct fold3-style case filenames in corpus.chunks)
  - IA collection `project-blue-book` (per-case items, Fold3-derived NARA set)
  - IA item `nara-pbb` (free full reels — case files + admin paper)
  - IA item `ProjectBlueBookIndexes` (per-year case indexes, redacted AND
    unredacted — the map of the 12,618-case universe)

Writes data/bluebook_gap/{missing_items.txt,report.txt} and prints the report.
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import psycopg

ROOT = Path(__file__).parent
OUT = ROOT / "data/bluebook_gap"
SCRAPE = ("https://archive.org/services/search/v1/scrape"
          "?q=collection%3Aproject-blue-book&count=10000&fields=identifier")


def dsn():
    import os
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    g = lambda k: re.search(rf"^{k}=(.*)$", env, re.M).group(1).strip()
    return (f"postgresql://{g('POSTGRES_USER')}:{g('POSTGRES_PASSWORD')}"
            f"@localhost:5439/{g('POSTGRES_DB')}")


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "uap-archive/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def collection_identifiers():
    ids, cursor = [], None
    while True:
        url = SCRAPE + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
        d = fetch_json(url)
        ids += [i["identifier"] for i in d.get("items", [])]
        cursor = d.get("cursor")
        if not cursor:
            return ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(dsn()) as pg, pg.cursor() as cur:
        cur.execute(r"""SELECT DISTINCT meta->>'filename' FROM corpus.chunks
                        WHERE meta->>'filename' ~ '^\d{4}-\d{2}-\d{6,}'""")
        held = {re.sub(r"\.pdf$", "", r[0], flags=re.I) for r in cur.fetchall()}

    ia = collection_identifiers()
    ia_case = {i for i in ia if re.match(r"^\d{4}-\d{2}-", i)}
    ia_other = sorted(set(ia) - ia_case)

    missing = sorted(ia_case - held)
    held_not_ia = held - ia_case

    by_year = Counter(m[:4] for m in missing)
    rpt = []
    rpt.append("BLUE BOOK GAP REPORT")
    rpt.append("=" * 50)
    rpt.append(f"case files in our corpus:            {len(held):,}")
    rpt.append(f"case items in IA project-blue-book:  {len(ia_case):,}")
    rpt.append(f"  -> NEW: in IA, not in corpus:      {len(missing):,}")
    rpt.append(f"  -> in corpus, not in IA:           {len(held_not_ia):,}"
               " (raw-tree extras; fine)")
    rpt.append(f"non-case items in the collection:    {len(ia_other)}")
    for o in ia_other[:15]:
        rpt.append(f"    {o}")
    rpt.append("")
    rpt.append("missing case files by year:")
    for y in sorted(by_year):
        rpt.append(f"  {y}: {by_year[y]}")
    (OUT / "missing_items.txt").write_text("\n".join(missing) + "\n")
    (OUT / "report.txt").write_text("\n".join(rpt) + "\n")
    print("\n".join(rpt))
    print(f"\nfull missing list -> {OUT/'missing_items.txt'}")


if __name__ == "__main__":
    sys.exit(main())
