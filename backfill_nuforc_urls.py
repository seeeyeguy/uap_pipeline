#!/usr/bin/env python3
"""
Backfill per-report NUFORC URLs into corpus.chunks metadata.

The narrative sightings layer (import_sightings.py) was built from the
scrubbed Kaggle CSV, which carries no report URLs — so meta->>'event_url'
is empty for all 76k NUFORC chunks and the app can only link to NUFORC's
event-month index. LinkWentz's scrape (nuforc_events_complete.csv, already
in data/downloads) has an event_url for every row; joining on
event date + city + state (shape, then duration text, as tiebreakers)
uniquely resolves ~96% of our sightings.

Metadata-only: embeddings key on chunk text, so no re-embed, no retriever
restart. The API already prefers event_url when present. Idempotent — the
UPDATE skips rows whose event_url already matches. Ambiguous or unmatched
sightings are left empty and keep the app's month-index fallback.

URLs are normalized to the modern form (https://nuforc.org/sighting/?id=N):
the scrape's old webreports/.../S<N>.html links 301 there anyway.

Usage:
    python backfill_nuforc_urls.py             # dry run: counts only
    python backfill_nuforc_urls.py --apply     # write to the DB

DSN: $PG_DSN, else built from ../uap-api/.env (host localhost:5439).
For prod, run with PG_DSN pointed at the prod database (out-of-band, like
schema changes — the deploy pipeline never touches the DB).
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

ROOT = Path(__file__).parent
CSV = ROOT / "data/downloads/github_mirrors/NUFORC-Dataset/nuforc_events_complete.csv"
BATCH = 5000

# Same normalizations import_sightings.py applied at ingest, so the join
# keys line up with what's in chunk metadata.
STATES = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "de": "Delaware",
    "fl": "Florida", "ga": "Georgia", "hi": "Hawaii", "id": "Idaho",
    "il": "Illinois", "in": "Indiana", "ia": "Iowa", "ks": "Kansas",
    "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi",
    "mo": "Missouri", "mt": "Montana", "ne": "Nebraska", "nv": "Nevada",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico", "ny": "New York",
    "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio", "ok": "Oklahoma",
    "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
    "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah",
    "vt": "Vermont", "va": "Virginia", "wa": "Washington", "wv": "West Virginia",
    "wi": "Wisconsin", "wy": "Wyoming", "dc": "District of Columbia",
}
SHAPE_MAP = {"disk": "disc", "delta": "triangle", "changing": "irregular",
             "chevron": "chevron", "cross": "cross", "cigar": "cigar",
             "flash": "light", "other": "irregular", "unknown": "unknown"}

REPORT_ID = re.compile(r"/S?(\d+)\.html$")


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    pw = re.search(r"^POSTGRES_PASSWORD=(.*)$", env, re.M).group(1).strip()
    user = re.search(r"^POSTGRES_USER=(.*)$", env, re.M).group(1).strip()
    db = re.search(r"^POSTGRES_DB=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{user}:{pw}@localhost:5439/{db}"


def norm_shape(s):
    s = (s or "").lower().strip()
    return SHAPE_MAP.get(s, s or "unknown")


def canonical_url(raw):
    """Modern report URL, or None if the row's URL is unusable."""
    raw = (raw or "").strip()
    m = REPORT_ID.search(raw)
    if m:
        return "https://nuforc.org/sighting/?id=" + m.group(1)
    # Unrecognized shape — keep it only if it's already a plain http(s) URL.
    return raw if re.match(r"^https?://", raw) else None


def build_index():
    """(event_date, city_lower, region) -> [(shape, duration_lower, url)]"""
    idx = defaultdict(list)
    rows = 0
    with open(CSV, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            url = canonical_url(row.get("event_url"))
            date = (row.get("event_date") or "").strip()
            if not url or not date:
                continue
            city = (row.get("city") or "").strip().lower()
            state = (row.get("state") or "").strip().lower()
            region = STATES.get(state, state.upper())
            idx[(date, city, region)].append(
                (norm_shape(row.get("shape")),
                 (row.get("duration") or "").strip().lower(), url))
            rows += 1
    print(f"indexed {rows} scrape rows with report URLs")
    return idx


def resolve(idx, date, city, region, shapes_json, duration):
    cands = idx.get((date, (city or "").lower(), region or ""), [])
    if len(cands) == 1:
        return cands[0][2]
    if not cands:
        return None
    try:
        shapes = set(json.loads(shapes_json or "[]"))
    except ValueError:
        shapes = set()
    tie = [c for c in cands if c[0] in shapes] if shapes else cands
    if len(tie) != 1 and duration:
        tie2 = [c for c in tie if c[1] == (duration or "").lower()]
        if tie2:
            tie = tie2
    return tie[0][2] if len(tie) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write matches to the DB (default: dry-run counts)")
    args = ap.parse_args()

    idx = build_index()
    matched, ambiguous, unmatched, already = [], 0, 0, 0

    with psycopg.connect(dsn()) as pg, pg.cursor() as cur:
        cur.execute("""
            SELECT id, meta->>'event_date', meta->>'city', meta->>'region',
                   meta->>'shapes', meta->>'duration_text',
                   COALESCE(meta->>'event_url','')
            FROM corpus.chunks WHERE meta->>'source_program' = 'nuforc'""")
        for cid, date, city, region, shapes, dur, current in cur.fetchall():
            url = resolve(idx, date, city, region, shapes, dur)
            if url is None:
                # distinguish "no candidates" from "couldn't disambiguate"
                if idx.get((date or "", (city or "").lower(), region or "")):
                    ambiguous += 1
                else:
                    unmatched += 1
            elif url == current:
                already += 1
            else:
                matched.append((cid, url))

        total = len(matched) + ambiguous + unmatched + already
        print(f"chunks {total} | to update {len(matched)} | already set {already}"
              f" | ambiguous {ambiguous} | no match {unmatched}")
        if not args.apply:
            print("dry run — pass --apply to write")
            return
        if not matched:
            print("nothing to write")
            return

        cur.execute("""
            CREATE TEMP TABLE nuforc_url_backfill
            (id TEXT PRIMARY KEY, url TEXT NOT NULL) ON COMMIT DROP""")
        with cur.copy("COPY nuforc_url_backfill (id, url) FROM STDIN") as cp:
            for cid, url in matched:
                cp.write_row((cid, url))
        cur.execute("""
            UPDATE corpus.chunks c
            SET meta = jsonb_set(c.meta, '{event_url}', to_jsonb(b.url))
            FROM nuforc_url_backfill b WHERE c.id = b.id""")
        print(f"updated {cur.rowcount} chunks")
        pg.commit()


if __name__ == "__main__":
    sys.exit(main())
