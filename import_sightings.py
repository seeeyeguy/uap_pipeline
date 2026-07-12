#!/usr/bin/env python3
"""
Import NUFORC narrative sightings as text-native documents — one chunk per
sighting, deterministic metadata (no LLM enrichment), reference labels
pointing at the ultimate source (NUFORC + date + place).

Source: data/retired/nuforc/ufo-scrubbed-geocoded-time-standardized.csv
(80,331 sightings WITH narratives + geocoded lat/long).

Metadata is aligned to the corpus enrich_v2 schema so cross-corpus filters
work uniformly: region uses full US state names (matching the LLM's
"California", not "ca"); document_type=sighting_report; source_program=nuforc.

Note: UFOSINT's 618k deduplicated structured records (narratives stripped)
are a better fit for the analytics/clustering layer than the vector index,
so they are NOT imported here — this is the *narrative* sightings layer.

Writes sightings_chunks.jsonl (same shape as rebuild_chunks.jsonl) for
embedding + upsert into the live index.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
CSV = ROOT / "data/retired/nuforc/ufo-scrubbed-geocoded-time-standardized.csv"
OUT = ROOT / "data/sightings_chunks.jsonl"

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
COUNTRY = {"us": "United States", "gb": "United Kingdom", "ca": "Canada",
           "au": "Australia", "de": "Germany"}


def clean(s):
    return (s or "").replace("&#44;", ",").replace("&amp;", "&").replace("&quot;", '"').strip()


def iso_date(raw):
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


def main():
    seen = set()
    n = dup = 0
    with open(CSV, encoding="utf-8", errors="ignore") as f, open(OUT, "w") as out:
        for row in csv.reader(f):
            if len(row) < 11:
                continue
            dt, city, state, country, shape, dur_s, dur_txt, desc, posted, lat, lng = row[:11]
            desc = clean(desc)
            city = clean(city).title()
            if len(desc) < 25:
                continue
            date = iso_date(dt)
            # dedup: same date + city + narrative prefix
            key = hashlib.sha1(f"{date}|{city}|{desc[:120]}".encode()).hexdigest()[:16]
            if key in seen:
                dup += 1
                continue
            seen.add(key)

            region = STATES.get(state.lower().strip(), state.upper().strip() if state.strip() else "")
            cty = COUNTRY.get((country or "").lower().strip(), (country or "").upper())
            shp = SHAPE_MAP.get(shape.lower().strip(), shape.lower().strip() or "unknown")
            ref = f"NUFORC sighting — {city}, {region}" + (f" — {date}" if date else "")
            # embed the narrative with a light reference header for context
            text = f"{ref}\nShape: {shp or 'unknown'} · Duration: {clean(dur_txt) or 'unknown'}\n\n{desc}"

            try:
                latf = float(lat); lngf = float(lng)
            except ValueError:
                latf = lngf = None

            cid = "nuforc_" + key
            meta = {
                "filename": ref, "source": "nuforc", "pages": "",
                "chunk_id": 0, "language": "en",
                "summary": desc[:500],
                "document_type": "sighting_report", "source_program": "nuforc",
                "originating_agency": "National UFO Reporting Center",
                "event_date": date, "date_precision": "day" if date else "unknown",
                "document_date": iso_date(posted),
                "country": cty, "region": region, "city": city, "site": "",
                "nearest_named_place": f"{city}, {region}".strip(", "),
                "latitude": latf if latf is not None else -999.0,
                "longitude": lngf if lngf is not None else -999.0,
                "shapes": json.dumps([shp] if shp and shp != "unknown" else []),
                "sensor_types": json.dumps(["visual"]),
                "witness_count": -1, "witness_types": json.dumps(["civilian"]),
                "duration_text": clean(dur_txt),
                "explanation_status": "unexplained",
                "ocr_quality": "good",  # text-native, no OCR
                "official_source": "False",
                "event_url": clean(posted) if posted.startswith("http") else "",
            }
            out.write(json.dumps({"id": cid, "text": text, "metadata": meta},
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"imported {n} sightings ({dup} intra-source dupes skipped) -> {OUT}")


if __name__ == "__main__":
    main()
