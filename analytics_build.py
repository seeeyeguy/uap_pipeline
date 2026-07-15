#!/usr/bin/env python3
"""
Build the unified analytics store (DuckDB): one `events` table combining
every sighting/event across sources, with the fields analytics needs
(date, geocoded location, shape, quality, source lineage). Foundation for
the location dimension, clustering, and statistics.

Sources:
  - UFOSINT ufo_public.db  : 618k deduplicated, 418k geocoded
  - NUFORC scrubbed CSV     : 76k geocoded narratives (also in vector index)
  - corpus enrich_v2        : 17k documents' event_location/date/shape

Output: data/analytics.duckdb  (table `events`)
"""
import csv
import glob
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent
DB = ROOT / "data/analytics.duckdb"
UFOSINT = ROOT / "data/downloads/ufosint_flags/ufo_public.db"
NUFORC = ROOT / "data/retired/nuforc/ufo-scrubbed-geocoded-time-standardized.csv"
ENR = ROOT / "data/enriched_v2"

if DB.exists():
    DB.unlink()
con = duckdb.connect(str(DB))
con.execute("""CREATE TABLE events(
    event_id VARCHAR, source VARCHAR, origin VARCHAR,
    event_date VARCHAR, event_year INTEGER,
    city VARCHAR, region VARCHAR, country VARCHAR,
    lat DOUBLE, lng DOUBLE,
    shape VARCHAR, quality DOUBLE,
    doc_ref VARCHAR,
    description VARCHAR, duration VARCHAR, num_witnesses INTEGER,
    hynek VARCHAR, vallee VARCHAR, explanation VARCHAR, source_ref VARCHAR)""")

rows = []
COLS = ["event_id", "source", "origin", "event_date", "event_year", "city",
        "region", "country", "lat", "lng", "shape", "quality", "doc_ref",
        "description", "duration", "num_witnesses", "hynek", "vallee",
        "explanation", "source_ref"]

# Detail-field hygiene (UFOSINT): explanation mixes real values ("Meteor",
# "Hoax") with catalog symbols ("+", "-", "*") — keep only lettered values.
# source_ref can run to ~6k chars of stacked citations — cap it.
def _txt(v, cap=None):
    if isinstance(v, (list, tuple)):
        v = ", ".join(str(x) for x in v)
    elif v is not None and not isinstance(v, str):
        v = str(v)
    v = (v or "").replace("\x00", "").strip()
    return v[:cap] if cap else v


def _explanation(v):
    v = _txt(v)
    return v if any(c.isalpha() for c in v) else ""


def yr(d):
    try:
        return int(str(d)[:4])
    except (ValueError, TypeError):
        return None


def flush():
    # DuckDB executemany is slow for bulk; register a DataFrame and INSERT
    # from it (columnar, ~100x faster).
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=COLS)
        con.register("_batch", df)
        con.execute("INSERT INTO events SELECT * FROM _batch")
        con.unregister("_batch")
        rows.clear()


# ── UFOSINT ──
sc = sqlite3.connect(UFOSINT)
n = 0
for r in sc.execute("""SELECT s.id, o.name, s.date_event, l.city, l.state, l.country,
                              l.latitude, l.longitude, s.standardized_shape, s.quality_score,
                              s.description, s.duration, s.num_witnesses,
                              s.hynek, s.vallee, s.explanation, s.source_ref
                       FROM sighting s LEFT JOIN location l ON s.location_id=l.id
                       LEFT JOIN source_origin o ON s.origin_id=o.id"""):
    (sid, origin, date, city, state, country, lat, lng, shape, q,
     desc, dur, wit, hynek, vallee, expl, sref) = r
    d = str(date)[:10] if date else None
    try:
        wit = int(wit) if wit is not None else None
    except (ValueError, TypeError):
        wit = None
    rows.append([f"ufosint_{sid}", "ufosint", origin or "", d, yr(date),
                 city or "", state or "", country or "",
                 lat, lng, (shape or "").lower(), q, "",
                 _txt(desc), _txt(dur, 100), wit,
                 _txt(hynek, 10), _txt(vallee, 10), _explanation(expl),
                 _txt(sref, 600)])
    n += 1
    if len(rows) >= 20000:
        flush()
flush()
print(f"UFOSINT: {n} events", flush=True)

# ── NUFORC ──
STATES = {}  # reuse import_sightings mapping lightly (region already full in vector import;
# here we keep raw state for the dimension pass to normalize)
n = 0
with open(NUFORC, encoding="utf-8", errors="ignore") as f:
    for row in csv.reader(f):
        if len(row) < 11:
            continue
        dt, city, state, country, shape, _, dur_txt, desc, _, lat, lng = row[:11]
        try:
            latf, lngf = float(lat), float(lng)
        except ValueError:
            latf = lngf = None
        d = None
        for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y"):
            try:
                d = datetime.strptime(dt.strip(), fmt).strftime("%Y-%m-%d"); break
            except ValueError:
                continue
        rows.append([f"nuforc_{n}", "nuforc", "NUFORC", d, yr(d),
                     city.title(), state.upper(), (country or "").upper(),
                     latf, lngf, (shape or "").lower(), None, "",
                     _txt(desc), _txt(dur_txt, 100), None, "", "", "", ""])
        n += 1
        if len(rows) >= 20000:
            flush()
flush()
print(f"NUFORC: {n} events", flush=True)

# ── corpus enrich_v2 (documents that describe a datable/locatable event) ──
n = 0
for p in glob.glob(str(ENR / "*.json")):
    try:
        e = json.load(open(p))
    except Exception:
        continue
    loc = e.get("event_location") if isinstance(e.get("event_location"), dict) else {}
    date = e.get("event_date")
    if not (loc.get("city") or loc.get("region")) and not date:
        continue
    obs = e.get("observation") if isinstance(e.get("observation"), dict) else {}
    shapes = obs.get("shapes") if isinstance(obs.get("shapes"), list) else []
    rows.append([f"corpus_{Path(p).stem}", "corpus", e.get("source_program") or "",
                 str(date)[:10] if date else None, yr(date),
                 loc.get("city") or "", loc.get("region") or "", loc.get("country") or "",
                 None, None, (shapes[0] if shapes else "").lower(), None, Path(p).stem,
                 _txt(e.get("summary")), _txt(e.get("duration"), 100),
                 None, "", "", "", ""])
    n += 1
    if len(rows) >= 20000:
        flush()
flush()
print(f"corpus: {n} events", flush=True)

tot = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
geo = con.execute("SELECT COUNT(*) FROM events WHERE lat IS NOT NULL").fetchone()[0]
print(f"\nTOTAL events: {tot} | geocoded: {geo}")
print("by source:", con.execute("SELECT source, COUNT(*) FROM events GROUP BY source").fetchall())
con.close()
print(f"-> {DB}")
