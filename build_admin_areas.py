#!/usr/bin/env python3
"""
County-layer build: assign every GeoNames town in corpus.locations to a
geoBoundaries ADM2 area (county / district / département …) and compute
which areas share a border — the data behind "search Oxford, MA and see
Auburn and Webster too" at the county level, worldwide.

Sources
  - corpus.locations (GeoNames cities500, already in Postgres)
  - geoBoundaries gbOpen ADM2 simplified polygons, one file per country
    (CC-BY / public-domain per country; metadata records each license).
    Files live behind Git LFS: fetch the pointer from raw.githubusercontent,
    then resolve the object through the LFS batch API. Cached in
    data/geoboundaries/ so re-runs are offline.

Method
  - Membership: point-in-polygon via shapely STRtree; unmatched towns
    (coastal simplification gaps) fall back to the nearest polygon within
    ~0.1 degree.
  - Adjacency: within-country only (by design — county expansion never
    crosses a national border). Simplified geometry can leave hairline
    gaps between neighbours, so adjacency is "within ~150 m", not strict
    touches.
  - Area→region: majority vote of member towns' region strings, so
    "Worcester (Massachusetts)" disambiguates from other Worcesters.

Publishes (with --apply; dry-run prints counts only)
  corpus.admin_areas       area_id, name, kind, country_code, region
  corpus.admin_area_towns  area_id, location_id
  corpus.admin_area_adj    area_id, adj_area_id  (symmetric)

Usage
  python build_admin_areas.py                    # dry run, all countries
  python build_admin_areas.py --countries US,GB  # subset
  python build_admin_areas.py --apply            # write to Postgres

DSN: $PG_DSN, else built from ../uap-api/.env (host localhost:5439).
Prod runs out-of-band with PG_DSN, like every schema/data change.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import psycopg
from shapely import STRtree
from shapely.geometry import Point, shape
from shapely.prepared import prep

ROOT = Path(__file__).parent
GEO = ROOT / "data/geonames"
CACHE = ROOT / "data/geoboundaries"
API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/ADM2/"
LFS_BATCH = "https://github.com/wmgeolab/geoBoundaries.git/info/lfs/objects/batch"
ADJ_EPS = 0.0015   # ~150 m in degrees: bridges simplification gaps
SNAP_EPS = 0.1     # ~10 km: max distance for nearest-polygon fallback


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    pw = re.search(r"^POSTGRES_PASSWORD=(.*)$", env, re.M).group(1).strip()
    user = re.search(r"^POSTGRES_USER=(.*)$", env, re.M).group(1).strip()
    db = re.search(r"^POSTGRES_DB=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{user}:{pw}@localhost:5439/{db}"


def http_json(url, data=None, headers=None, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def http_bytes(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def iso2_to_iso3():
    m = {}
    with open(GEO / "countryInfo.txt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and p[0] and p[1]:
                m[p[0]] = p[1]
    return m


def fetch_boundaries(iso3):
    """Country ADM2 geojson (cached). Returns (features, kind) or (None, why)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    gj_path = CACHE / f"{iso3}_ADM2.geojson"
    meta_path = CACHE / f"{iso3}_ADM2.meta.json"
    if gj_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return json.loads(gj_path.read_text())["features"], meta.get("boundaryCanonical") or "Districts"

    try:
        meta = http_json(API.format(iso3=iso3))
    except Exception:
        return None, "no ADM2 dataset"
    url = meta.get("simplifiedGeometryGeoJSON") or meta.get("gjDownloadURL")
    if not url:
        return None, "no download URL"
    # github.com/<o>/<r>/raw/<ref>/<path> redirects to a dead media host for
    # LFS files; the raw.githubusercontent.com form serves the LFS pointer.
    url = re.sub(r"^https://github\.com/([^/]+)/([^/]+)/raw/",
                 r"https://raw.githubusercontent.com/\1/\2/", url)

    raw = http_bytes(url)
    # Git-LFS pointer? Resolve through the batch API for the real object.
    if raw.startswith(b"version https://git-lfs"):
        m_oid = re.search(rb"oid sha256:([0-9a-f]+)", raw)
        m_size = re.search(rb"size (\d+)", raw)
        if not m_oid or not m_size:
            return None, "malformed LFS pointer"
        oid = m_oid.group(1).decode()
        size = int(m_size.group(1))
        batch = http_json(
            LFS_BATCH,
            data=json.dumps({
                "operation": "download", "transfers": ["basic"],
                "objects": [{"oid": oid, "size": size}],
            }).encode(),
            headers={"Accept": "application/vnd.git-lfs+json",
                     "Content-Type": "application/vnd.git-lfs+json"})
        href = batch["objects"][0]["actions"]["download"]["href"]
        # The href is server-supplied — never follow anything but https.
        if not href.startswith("https://"):
            return None, "non-https LFS href"
        raw = http_bytes(href)

    gj = json.loads(raw)
    gj_path.write_bytes(raw)
    meta_path.write_text(json.dumps(meta))
    return gj["features"], meta.get("boundaryCanonical") or "Districts"


def load_towns(pg):
    with pg.cursor() as cur:
        cur.execute("""
            SELECT location_id, city, region, country_code, lat, lng
            FROM corpus.locations
            WHERE lat IS NOT NULL AND lng IS NOT NULL""")
        towns = defaultdict(list)
        for lid, city, region, cc, lat, lng in cur.fetchall():
            towns[cc].append((lid, city, region or "", lat, lng))
    return towns


def process_country(cc, iso3, towns):
    """Returns (areas, memberships, adjacency, unmatched_count) or None."""
    feats, kind = fetch_boundaries(iso3)
    if feats is None:
        return None, kind

    polys, ids, names = [], [], []
    for f in feats:
        p = f.get("properties", {})
        sid = p.get("shapeID") or p.get("shapeGroup", iso3) + "-" + str(len(ids))
        try:
            g = shape(f["geometry"])
        except Exception:
            continue
        if g.is_empty:
            continue
        polys.append(g)
        ids.append(sid)
        names.append((p.get("shapeName") or "").strip() or sid)
    if not polys:
        return None, "no usable polygons"

    tree = STRtree(polys)
    prepared = [prep(g) for g in polys]

    memberships = []       # (area_id, location_id)
    region_votes = defaultdict(Counter)
    unmatched = 0
    for lid, _city, region, lat, lng in towns:
        pt = Point(lng, lat)
        hit = None
        for i in tree.query(pt):
            if prepared[i].covers(pt):
                hit = i
                break
        if hit is None:
            # nearest polygon within SNAP_EPS (simplification gaps, coasts)
            cands = tree.query(pt.buffer(SNAP_EPS))
            best, bestd = None, SNAP_EPS
            for i in cands:
                d = polys[i].distance(pt)
                if d < bestd:
                    best, bestd = i, d
            hit = best
        if hit is None:
            unmatched += 1
            continue
        memberships.append((ids[hit], lid))
        if region:
            region_votes[ids[hit]][region] += 1

    # Adjacency: bounding-box candidates, then distance test with tolerance.
    adj = set()
    for i, g in enumerate(polys):
        for j in tree.query(g.buffer(ADJ_EPS)):
            if j <= i:
                continue
            if g.distance(polys[j]) <= ADJ_EPS:
                adj.add((ids[i], ids[j]))

    areas = []
    for i, sid in enumerate(ids):
        region = ""
        if region_votes[sid]:
            region = region_votes[sid].most_common(1)[0][0]
        areas.append((sid, names[i], kind, cc, region))
    return (areas, memberships, sorted(adj), unmatched), None


def publish(pg, areas, members, adj):
    with pg.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS corpus.admin_area_adj, corpus.admin_area_towns,
                                 corpus.admin_areas;
            CREATE TABLE corpus.admin_areas (
                area_id      TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                kind         TEXT NOT NULL,
                country_code TEXT NOT NULL,
                region       TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE corpus.admin_area_towns (
                area_id     TEXT NOT NULL REFERENCES corpus.admin_areas(area_id),
                location_id BIGINT NOT NULL,
                PRIMARY KEY (area_id, location_id)
            );
            CREATE TABLE corpus.admin_area_adj (
                area_id     TEXT NOT NULL REFERENCES corpus.admin_areas(area_id),
                adj_area_id TEXT NOT NULL REFERENCES corpus.admin_areas(area_id),
                PRIMARY KEY (area_id, adj_area_id)
            );""")
        with cur.copy("COPY corpus.admin_areas FROM STDIN") as cp:
            for row in areas:
                cp.write_row(row)
        seen = set()
        with cur.copy("COPY corpus.admin_area_towns FROM STDIN") as cp:
            for row in members:
                if row not in seen:
                    seen.add(row)
                    cp.write_row(row)
        with cur.copy("COPY corpus.admin_area_adj FROM STDIN") as cp:
            for a, b in adj:
                cp.write_row((a, b))
                cp.write_row((b, a))
        cur.execute("""
            CREATE INDEX ON corpus.admin_areas (lower(name));
            CREATE INDEX ON corpus.admin_areas (country_code);
            CREATE INDEX ON corpus.admin_area_towns (location_id);""")
    pg.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", help="comma-separated ISO2 subset")
    ap.add_argument("--apply", action="store_true", help="write to Postgres")
    args = ap.parse_args()
    if args.apply and args.countries:
        # publish() replaces the tables wholesale — a subset apply would
        # silently drop every other country. Subsets are for dry-run QA.
        ap.error("--apply publishes ALL countries; drop --countries "
                 "(downloads are cached, the full run is cheap)")

    iso3 = iso2_to_iso3()
    all_areas, all_members, all_adj = [], [], []
    skipped, total_unmatched = [], 0

    with psycopg.connect(dsn()) as pg:
        towns = load_towns(pg)
        ccs = sorted(towns)
        if args.countries:
            want = {c.strip().upper() for c in args.countries.split(",")}
            ccs = [c for c in ccs if c in want]

        for n, cc in enumerate(ccs, 1):
            if cc not in iso3:
                skipped.append((cc, "no ISO3"))
                continue
            # One flaky download (the API 504s now and then) must not kill a
            # 200-country run — record and move on; a re-run picks it up
            # from cache-miss.
            try:
                result, why = process_country(cc, iso3[cc], towns[cc])
            except Exception as e:
                skipped.append((cc, "error: " + str(e)[:80]))
                continue
            if result is None:
                skipped.append((cc, why))
                continue
            areas, members, adj, unmatched = result
            all_areas.extend(areas)
            all_members.extend(members)
            all_adj.extend(adj)
            total_unmatched += unmatched
            print(f"[{n}/{len(ccs)}] {cc}: {len(areas)} areas, "
                  f"{len(members)} towns, {len(adj)} borders, "
                  f"{unmatched} unmatched", flush=True)

        print(f"\nTOTAL: {len(all_areas)} areas | {len(all_members)} town "
              f"memberships | {len(all_adj)} borders | "
              f"{total_unmatched} unmatched towns | {len(skipped)} countries skipped")
        for cc, why in skipped:
            print(f"  skip {cc}: {why}")
        if not args.apply:
            print("dry run — pass --apply to publish")
            return
        publish(pg, all_areas, all_members, all_adj)
        print("published corpus.admin_areas / admin_area_towns / admin_area_adj")


if __name__ == "__main__":
    sys.exit(main())
