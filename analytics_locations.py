#!/usr/bin/env python3
"""
Location dimension (plan task 2): build a GeoNames-backed `locations` table
in analytics.duckdb and resolve every event's free-text (city, region,
country) to a canonical geoname_id + city -> region -> country hierarchy.

Deterministic offline pass — the LLM only ever extracted *named places*;
coordinates come from GeoNames (cities500, 235k places), never the model.
Events that arrived without coordinates (the enriched corpus docs) get
lat/lng from their matched city, so the clustering pass can include them.

Match tiers, most to least specific (recorded in events.loc_match):
  city_exact   country + admin1 + city name
  city_cc      country + city name (admin1 unknown/failed)
  city_global  city name only (no country could be resolved; top population)
  region       admin1 resolved but city not in GeoNames
  country      only the country resolved
  none         nothing usable
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).parent
GEO = ROOT / "data/geonames"
DB = ROOT / "data/analytics.duckdb"

# ── GeoNames reference data ──
COUNTRY_NAME = {}   # cc -> canonical name
COUNTRY_CODE = {}   # casefolded name/alias -> cc
with open(GEO / "countryInfo.txt", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) > 4 and p[0]:
            COUNTRY_NAME[p[0]] = p[4]
            COUNTRY_CODE[p[4].casefold()] = p[0]
COUNTRY_CODE.update({
    "usa": "US", "united states of america": "US", "us": "US", "u.s.": "US",
    "uk": "GB", "great britain": "GB", "england": "GB", "scotland": "GB",
    "wales": "GB", "northern ireland": "GB", "russia": "RU", "south korea": "KR",
    "north korea": "KP", "iran": "IR", "vietnam": "VN", "syria": "SY",
    "venezuela": "VE", "bolivia": "BO", "tanzania": "TZ", "czech republic": "CZ",
    "the netherlands": "NL", "holland": "NL", "brasil": "BR",
})

ADMIN1_NAME = {}    # (cc, adm1_code) -> canonical region name
ADMIN1_CODE = {}    # (cc, casefolded region name) -> adm1_code
with open(GEO / "admin1.txt", encoding="utf-8") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        cc, code = p[0].split(".", 1)
        ADMIN1_NAME[(cc, code)] = p[1]
        ADMIN1_CODE[(cc, p[1].casefold())] = code
        ADMIN1_CODE[(cc, p[2].casefold())] = code

US_STATES = {
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
CA_PROV = {"ab": "Alberta", "bc": "British Columbia", "mb": "Manitoba",
           "nb": "New Brunswick", "nf": "Newfoundland and Labrador",
           "nl": "Newfoundland and Labrador", "ns": "Nova Scotia",
           "nt": "Northwest Territories", "nu": "Nunavut", "on": "Ontario",
           "pe": "Prince Edward Island", "pq": "Quebec", "qc": "Quebec",
           "sk": "Saskatchewan", "yt": "Yukon", "yk": "Yukon"}

# ── city indexes from cities500 ──
# value: (priority, population, geoname_id) — primary/ascii names beat
# alternate names, then population decides.
IDX3 = defaultdict(list)   # (cc, adm1, name) -> candidates
IDX2 = defaultdict(list)   # (cc, name)       -> candidates
IDX1 = defaultdict(list)   # (name,)          -> candidates
CITY = {}                  # geoname_id -> (name, adm1_code, cc, lat, lng, pop)

with open(GEO / "cities500.txt", encoding="utf-8") as f:
    for p in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
        gid, name, ascii_, alts = int(p[0]), p[1], p[2], p[3]
        lat, lng, cc, adm1, pop = float(p[4]), float(p[5]), p[8], p[10], int(p[14] or 0)
        CITY[gid] = (name, adm1, cc, lat, lng, pop)
        names = {name.casefold(): 0, ascii_.casefold(): 0}
        for a in alts.split(","):
            a = a.strip().casefold()
            if a and a not in names:
                names[a] = 1
        for nm, prio in names.items():
            cand = (prio, -pop, gid)
            IDX3[(cc, adm1, nm)].append(cand)
            IDX2[(cc, nm)].append(cand)
            IDX1[nm].append(cand)
print(f"GeoNames loaded: {len(CITY)} cities, {len(COUNTRY_NAME)} countries, "
      f"{len(ADMIN1_NAME)} admin1 regions", flush=True)

PAREN = re.compile(r"\s*\([^)]*\)")


def clean_city(s):
    return PAREN.sub("", s or "").replace(".", "").strip().casefold()


def resolve_country(country, region):
    c = (country or "").strip().casefold()
    if c:
        if c in COUNTRY_CODE:
            return COUNTRY_CODE[c]
        if len(c) == 2 and c.upper() in COUNTRY_NAME:
            return c.upper()
        return None
    r = (region or "").strip().casefold()
    if r in US_STATES or r in {v.casefold() for v in US_STATES.values()}:
        return "US"
    if r in CA_PROV:
        return "CA"
    return None


def resolve_region(cc, region):
    r = (region or "").strip().casefold()
    if not cc or not r:
        return None
    if cc == "US" and r in US_STATES:
        r = US_STATES[r].casefold()
    if cc == "CA" and r in CA_PROV:
        r = CA_PROV[r].casefold()
    return ADMIN1_CODE.get((cc, r))


def best(cands):
    return min(cands)[2]


def resolve(city, region, country):
    """-> (geoname_id, cc, adm1_code, match_level)

    A RESOLVED region is authoritative: when (cc, adm1, city) misses, stay
    at region level rather than matching the city country-wide — the
    population tie-break would otherwise relocate e.g. Farmington/Wisconsin
    to Farmington/New Mexico. Country-wide and global city tiers apply only
    when no narrower scope resolved.
    """
    cc = resolve_country(country, region)
    adm1 = resolve_region(cc, region)
    c = clean_city(city)
    if c:
        if cc and adm1:
            if (cc, adm1, c) in IDX3:
                return best(IDX3[(cc, adm1, c)]), cc, adm1, "city_exact"
        elif cc and (cc, c) in IDX2:
            gid = best(IDX2[(cc, c)])
            return gid, cc, CITY[gid][1], "city_cc"
        elif not cc and c in IDX1:
            gid = best(IDX1[c])
            return gid, CITY[gid][2], CITY[gid][1], "city_global"
    if adm1:
        return None, cc, adm1, "region"
    if cc:
        return None, cc, None, "country"
    return None, None, None, "none"


con = duckdb.connect(str(DB))

# ── locations dimension table ──
con.execute("DROP TABLE IF EXISTS locations")
con.execute("""CREATE TABLE locations(
    location_id BIGINT PRIMARY KEY, city VARCHAR, region VARCHAR,
    region_code VARCHAR, country VARCHAR, country_code VARCHAR,
    lat DOUBLE, lng DOUBLE, population BIGINT)""")
ldf = pd.DataFrame(
    [(gid, n, ADMIN1_NAME.get((cc, a), ""), a, COUNTRY_NAME.get(cc, cc), cc, la, ln, pop)
     for gid, (n, a, cc, la, ln, pop) in CITY.items()],
    columns=["location_id", "city", "region", "region_code", "country",
             "country_code", "lat", "lng", "population"])
con.register("_loc", ldf)
con.execute("INSERT INTO locations SELECT * FROM _loc")
con.unregister("_loc")

# The geo columns must exist BEFORE the reset below: analytics_build
# recreates `events` bare, so a cold run has no geo_src yet (the reset
# then correctly touches 0 rows).
for col, typ in [("location_id", "BIGINT"), ("loc_city", "VARCHAR"),
                 ("loc_region", "VARCHAR"), ("loc_country", "VARCHAR"),
                 ("loc_cc", "VARCHAR"), ("loc_match", "VARCHAR"),
                 ("geo_src", "VARCHAR")]:
    con.execute(f"ALTER TABLE events ADD COLUMN IF NOT EXISTS {col} {typ}")

# re-runs must not keep coordinates a previous (possibly wrong) city match
# assigned — only source-provided coordinates survive the reset
con.execute("UPDATE events SET lat=NULL, lng=NULL WHERE geo_src='geonames'")

# ── resolve every distinct raw (city, region, country) ──
tuples = con.execute(
    "SELECT DISTINCT city, region, country FROM events").fetchall()
print(f"resolving {len(tuples)} distinct location tuples", flush=True)
out, counts = [], defaultdict(int)
for city, region, country in tuples:
    gid, cc, adm1, level = resolve(city, region, country)
    counts[level] += 1
    loc_city = CITY[gid][0] if gid else ""
    loc_region = ADMIN1_NAME.get((cc, adm1), "") if cc and adm1 else ""
    loc_country = COUNTRY_NAME.get(cc, "") if cc else ""
    glat = CITY[gid][3] if gid else None
    glng = CITY[gid][4] if gid else None
    out.append((city, region, country, gid, loc_city, loc_region,
                loc_country, cc or "", level, glat, glng))
print("match levels:", dict(sorted(counts.items())), flush=True)

res = pd.DataFrame(out, columns=["city", "region", "country", "location_id",
                                 "loc_city", "loc_region", "loc_country",
                                 "loc_cc", "loc_match", "glat", "glng"])
con.register("_res", res)
con.execute("""UPDATE events e SET
    location_id=r.location_id, loc_city=r.loc_city, loc_region=r.loc_region,
    loc_country=r.loc_country, loc_cc=r.loc_cc, loc_match=r.loc_match,
    geo_src=CASE WHEN e.lat IS NOT NULL THEN 'source'
                 WHEN r.glat IS NOT NULL THEN 'geonames' ELSE NULL END,
    lat=COALESCE(e.lat, r.glat), lng=COALESCE(e.lng, r.glng)
    FROM _res r
    WHERE e.city=r.city AND e.region=r.region AND e.country=r.country""")
con.unregister("_res")

for src, n, m, g in con.execute("""SELECT source, COUNT(*),
        COUNT(*) FILTER (location_id IS NOT NULL),
        COUNT(*) FILTER (lat IS NOT NULL)
        FROM events GROUP BY source ORDER BY 2 DESC""").fetchall():
    print(f"  {src:8}: {n:>6} events | {m:>6} city-matched | {g:>6} geocoded")
newly = con.execute(
    "SELECT COUNT(*) FROM events WHERE geo_src='geonames'").fetchone()[0]
print(f"newly geocoded from GeoNames: {newly}")
con.close()
print(f"-> locations dimension + event location columns in {DB.name}")
