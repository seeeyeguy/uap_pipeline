#!/usr/bin/env python3
"""
One-off, in-place enrichment of the events stores with detail fields —
description, duration, num_witnesses, hynek, vallee, explanation,
source_ref — WITHOUT rebuilding or re-clustering.

analytics_build.py now carries these columns natively, so any future full
rebuild produces them from scratch; this script back-fills the two stores
that already exist so cluster ids stay stable today:

  data/analytics.duckdb  events   (build store — keeps a partial
                                   `pg_publish.py analytics` from dropping
                                   the new columns again)
  corpus.events (Postgres)        (serving store the API reads)

Keys replicate analytics_build.py exactly: ufosint_{sighting.id},
nuforc_{n} in CSV row order, corpus_{json stem}.

Run: .venv/bin/python enrich_events_inplace.py
"""
import csv
import glob
import json
import re
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd
import psycopg

ROOT = Path(__file__).parent
UFOSINT = ROOT / "data/downloads/ufosint_flags/ufo_public.db"
NUFORC = ROOT / "data/retired/nuforc/ufo-scrubbed-geocoded-time-standardized.csv"
ENR = ROOT / "data/enriched_v2"
ADB = ROOT / "data/analytics.duckdb"

NEWCOLS = [("description", "VARCHAR", "TEXT"), ("duration", "VARCHAR", "TEXT"),
           ("num_witnesses", "INTEGER", "INTEGER"), ("hynek", "VARCHAR", "TEXT"),
           ("vallee", "VARCHAR", "TEXT"), ("explanation", "VARCHAR", "TEXT"),
           ("source_ref", "VARCHAR", "TEXT")]


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


def collect():
    rows = []  # [event_id, description, duration, num_witnesses, hynek, vallee, explanation, source_ref]

    sc = sqlite3.connect(UFOSINT)
    for r in sc.execute("""SELECT id, description, duration, num_witnesses,
                                  hynek, vallee, explanation, source_ref FROM sighting"""):
        sid, desc, dur, wit, hynek, vallee, expl, sref = r
        try:
            wit = int(wit) if wit is not None else None
        except (ValueError, TypeError):
            wit = None
        rows.append([f"ufosint_{sid}", _txt(desc), _txt(dur, 100), wit,
                     _txt(hynek, 10), _txt(vallee, 10), _explanation(expl),
                     _txt(sref, 600)])
    n_ufo = len(rows)

    # NUFORC: n MUST count exactly like analytics_build.py (every row with
    # >= 11 fields increments, no other skips) or the keys drift.
    n = 0
    with open(NUFORC, encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if len(row) < 11:
                continue
            desc, dur_txt = row[7], row[6]
            if desc or dur_txt:
                rows.append([f"nuforc_{n}", _txt(desc), _txt(dur_txt, 100),
                             None, "", "", "", ""])
            n += 1
    n_nuf = len(rows) - n_ufo

    for p in glob.glob(str(ENR / "*.json")):
        try:
            e = json.load(open(p))
        except Exception:
            continue
        loc = e.get("event_location") if isinstance(e.get("event_location"), dict) else {}
        date = e.get("event_date")
        if not (loc.get("city") or loc.get("region")) and not date:
            continue  # same membership rule as analytics_build
        rows.append([f"corpus_{Path(p).stem}", _txt(e.get("summary")),
                     _txt(e.get("duration"), 100), None, "", "", "", ""])

    print(f"collected: {n_ufo:,} ufosint / {n_nuf:,} nuforc / "
          f"{len(rows) - n_ufo - n_nuf:,} corpus")
    return pd.DataFrame(rows, columns=["event_id"] + [c[0] for c in NEWCOLS])


def update_duckdb(df):
    con = duckdb.connect(str(ADB))
    for col, ddl, _ in NEWCOLS:
        con.execute(f"ALTER TABLE events ADD COLUMN IF NOT EXISTS {col} {ddl}")
    con.register("_enrich", df)
    sets = ", ".join(f"{c[0]} = _enrich.{c[0]}" for c in NEWCOLS)
    con.execute(f"UPDATE events SET {sets} FROM _enrich WHERE events.event_id = _enrich.event_id")
    got = con.execute("SELECT COUNT(*) FROM events WHERE description IS NOT NULL AND description != ''").fetchone()[0]
    con.close()
    print(f"duckdb: updated; {got:,} events now carry a description")


def dsn():
    import os
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    pw = re.search(r"^POSTGRES_PASSWORD=(.*)$", env, re.M).group(1).strip()
    user = re.search(r"^POSTGRES_USER=(.*)$", env, re.M).group(1).strip()
    db = re.search(r"^POSTGRES_DB=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{user}:{pw}@localhost:5439/{db}"


def update_pg(df):
    with psycopg.connect(dsn()) as pg:
        cur = pg.cursor()
        cur.execute("""CREATE TEMP TABLE _enrich(
            event_id TEXT PRIMARY KEY, description TEXT, duration TEXT,
            num_witnesses INTEGER, hynek TEXT, vallee TEXT,
            explanation TEXT, source_ref TEXT)""")
        wit_ix = df.columns.get_loc("num_witnesses")
        with cur.copy("COPY _enrich FROM STDIN") as cp:
            for r in df.itertuples(index=False):
                vals = list(r)
                # pandas stores nullable ints as float64 — undo "4.0"
                w = vals[wit_ix]
                vals[wit_ix] = None if (w is None or pd.isna(w)) else int(w)
                cp.write_row(tuple(None if (isinstance(v, float) and pd.isna(v)) else v
                                   for v in vals))
        for col, _, ddl in NEWCOLS:
            cur.execute(f"ALTER TABLE corpus.events ADD COLUMN IF NOT EXISTS {col} {ddl}")
        sets = ", ".join(f"{c[0]} = _enrich.{c[0]}" for c in NEWCOLS)
        cur.execute(f"""UPDATE corpus.events SET {sets} FROM _enrich
                        WHERE corpus.events.event_id = _enrich.event_id""")
        n = cur.rowcount
        pg.commit()
        cur.execute("""SELECT COUNT(*) FROM corpus.events
                       WHERE description IS NOT NULL AND description != ''""")
        got = cur.fetchone()[0]
        print(f"postgres: {n:,} rows updated; {got:,} events now carry a description")


if __name__ == "__main__":
    frame = collect()
    if ADB.exists():
        update_duckdb(frame)
    update_pg(frame)
    print("DONE")
