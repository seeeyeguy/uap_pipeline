#!/usr/bin/env python3
"""
Searchability (plan task 3): denormalize the analytics-derived values —
geo_cluster, wave_cluster, canonical GeoNames location hierarchy — into the
chunk metadata of the BUILD vector index (data/vectordb), so the retriever's
ChromaDB `where` filters can answer "sightings in cluster 944" or
"events in the 1965 Midwest wave". Run `pipeline.py publish` afterward.

Chunk id -> analytics event mapping:
  nuforc_<sha16>            one chunk per sighting. analytics_build keyed
                            events by CSV row number, so the sighting CSV is
                            re-walked here replicating BOTH schemes to map
                            chunk id <-> event id (also written to
                            events.doc_ref for first-class lineage).
  <stem>_<tag8>_chunk_<i>   corpus documents; event id is corpus_<stem>.

Metadata updates REPLACE the whole dict per id in Chroma, so existing
metadata is fetched and merged, never overwritten blind.
"""
import csv
import hashlib
from datetime import datetime
from pathlib import Path

import chromadb
import duckdb
from chromadb.config import Settings

ROOT = Path(__file__).parent
DB = ROOT / "data/analytics.duckdb"
VDB = ROOT / "data/vectordb"
CSV_PATH = ROOT / "data/retired/nuforc/ufo-scrubbed-geocoded-time-standardized.csv"
CORPUS_TAG = hashlib.sha1(b"corpus").hexdigest()[:8]
BATCH = 5000


def clean(s):
    return (s or "").replace("&#44;", ",").replace("&amp;", "&").replace("&quot;", '"').strip()


def iso_date(raw):
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


# ── chunk id -> event id map for NUFORC (replicates import_sightings dedup) ──
chunk2event = {}
seen = set()
n = 0
with open(CSV_PATH, encoding="utf-8", errors="ignore") as f:
    for row in csv.reader(f):
        if len(row) < 11:
            continue
        event_id = f"nuforc_{n}"          # analytics_build keyed by row counter
        n += 1
        dt, city, desc = row[0], row[1], row[7]
        desc = clean(desc)
        city = clean(city).title()
        if len(desc) < 25:
            continue
        key = hashlib.sha1(f"{iso_date(dt)}|{city}|{desc[:120]}".encode()).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        chunk2event[f"nuforc_{key}"] = event_id
print(f"NUFORC map: {len(chunk2event)} chunk ids over {n} events", flush=True)

# ── analytics values per event ──
con = duckdb.connect(str(DB))
events = {}
for (eid, gc, wc, lid, lcity, lregion, lcountry, lcc, lat, lng) in con.execute(
        """SELECT event_id, geo_cluster, st_cluster, location_id, loc_city,
                  loc_region, loc_country, loc_cc, lat, lng FROM events"""
        ).fetchall():
    events[eid] = (gc, wc, lid, lcity, lregion, lcountry, lcc, lat, lng)

# lineage: store the vector-index chunk id on the NUFORC events
con.execute("BEGIN")
con.executemany("UPDATE events SET doc_ref=? WHERE event_id=?",
                [(cid, eid) for cid, eid in chunk2event.items()])
con.execute("COMMIT")
print(f"events.doc_ref set for {len(chunk2event)} NUFORC events", flush=True)
con.close()

# ── walk the build index, merge in the derived values ──
client = chromadb.PersistentClient(path=str(VDB),
                                   settings=Settings(anonymized_telemetry=False))
col = client.get_collection("uap_documents")
total = col.count()
print(f"build index: {total} chunks", flush=True)

updated = unmapped = offset = 0
while offset < total:
    page = col.get(limit=BATCH, offset=offset, include=["metadatas"])
    offset += len(page["ids"])
    up_ids, up_metas = [], []
    for cid, meta in zip(page["ids"], page["metadatas"]):
        if cid.startswith("nuforc_"):
            eid = chunk2event.get(cid)
        elif f"_{CORPUS_TAG}_chunk_" in cid:
            eid = "corpus_" + cid.rsplit(f"_{CORPUS_TAG}_chunk_", 1)[0]
        else:
            eid = None
        ev = events.get(eid) if eid else None
        if ev is None:
            unmapped += 1
            continue
        gc, wc, lid, lcity, lregion, lcountry, lcc, lat, lng = ev
        m = dict(meta or {})
        m["geo_cluster"] = int(gc) if gc is not None else -1
        m["wave_cluster"] = wc or ""
        m["location_id"] = int(lid) if lid is not None else -1
        m["loc_city"] = lcity or ""
        m["loc_region"] = lregion or ""
        m["loc_country"] = lcountry or ""
        m["loc_cc"] = lcc or ""
        # corpus chunks had no coordinates; sightings already carry them
        if lat is not None and float(m.get("latitude", -999.0)) == -999.0:
            m["latitude"], m["longitude"] = float(lat), float(lng)
        up_ids.append(cid)
        up_metas.append(m)
    if up_ids:
        col.update(ids=up_ids, metadatas=up_metas)
        updated += len(up_ids)
    print(f"  {offset}/{total} scanned, {updated} updated", flush=True)

print(f"\nDONE: {updated} chunks updated, {unmapped} without an analytics event "
      f"(documents with no datable/locatable content)")
print("Now run: .venv/bin/python pipeline.py publish")
