#!/usr/bin/env python3
"""
Clustering over the analytics events (task 4) — three independent views,
one column each on the events table:

  geo_cluster   spatial: HDBSCAN over all geocoded events (hotspots).
  st_cluster    spatiotemporal: HDBSCAN over (position, scaled event time),
                so a wave = events dense in space AND time.
  time_cluster  temporal: flap periods — weeks where the report rate spikes
                far above a rolling 1-year baseline. Density clustering can't
                find these directly: reporting volume grows ~100x over the
                century, so any global density threshold just selects the
                modern era. Baseline-relative burst detection is
                era-independent.

Coordinates are mapped to 3D unit-sphere points so Boruvka can use a KDTree:
chord distance is a monotonic function of great-circle distance, so the
HDBSCAN hierarchy matches true haversine exactly. The first run OOM-killed
(exit 137) using the BallTree/haversine path with core_dist_n_jobs=-1 on 20
cores; jobs are capped at 4 here.

For the spatiotemporal pass, event time joins as a 4th euclidean axis scaled
by KM_PER_DAY: two same-place events N days apart are as distant as two
same-day events N*KM_PER_DAY km apart (axis units = earth radii, matching
the chord axes). Only events with a day-precision event_date participate;
year-only dates would pile up on Jan 1 and fabricate clusters.
"""
import numpy as np
import pandas as pd
import duckdb
import hdbscan

DB = "data/analytics.duckdb"
JOBS = 4
R_EARTH_KM = 6371.0
KM_PER_DAY = 20.0      # spatiotemporal coupling: 7 days apart ~ 140 km apart.
                       # At 5 km/day, big cities' continuous reporting chained
                       # into multi-year worms (London 2013-2021); 20 keeps a
                       # wave to the weeks/months a flap actually lasts.
DATE_LO, DATE_HI = "1900-01-02", "2029-12-31"  # junk dates exist (year 19, 2925)
# Jan 1 carries 13.5k events vs a 1.7k median day-of-year: sources that only
# know the year default the date to Jan 1, so both temporal passes skip it
# (the DATE_LO bound plus the dayofyear filter below).
DATED = (f"TRY_CAST(event_date AS DATE) BETWEEN DATE '{DATE_LO}' AND DATE '{DATE_HI}' "
         f"AND dayofyear(TRY_CAST(event_date AS DATE)) != 1")

FLAP_RATIO = 3.0       # weekly count must exceed RATIO x rolling median ...
FLAP_MIN_WEEK = 25     # ... and this absolute floor (early decades are sparse)
FLAP_MIN_EVENTS = 150  # discard merged flap periods smaller than this

con = duckdb.connect(DB)

df = con.execute(f"""SELECT event_id, lat, lng, event_year,
                            TRY_CAST(event_date AS DATE) event_day
                     FROM events
                     WHERE lat IS NOT NULL AND lng IS NOT NULL
                     AND lat BETWEEN -90 AND 90 AND lng BETWEEN -180 AND 180""").df()
lat = np.radians(df["lat"].to_numpy(dtype=np.float64))
lng = np.radians(df["lng"].to_numpy(dtype=np.float64))
xyz = np.column_stack([np.cos(lat) * np.cos(lng),
                       np.cos(lat) * np.sin(lng),
                       np.sin(lat)])
print(f"clustering {len(df)} geocoded events", flush=True)

# ── spatial clusters — min_cluster_size tuned for hotspot granularity ──
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10,
                            metric="euclidean", algorithm="boruvka_kdtree",
                            core_dist_n_jobs=JOBS)
df["geo_cluster"] = clusterer.fit_predict(xyz)
n_clusters = df["geo_cluster"].max() + 1
noise = int((df["geo_cluster"] == -1).sum())
print(f"spatial clusters: {n_clusters}, noise points: {noise}", flush=True)

# ── spatiotemporal pass: waves = dense in space AND time ──
days = (df["event_day"] - pd.Timestamp("1970-01-01")).dt.days
dated = (df["event_day"] >= pd.Timestamp(DATE_LO)) & \
        (df["event_day"] <= pd.Timestamp(DATE_HI)) & \
        (df["event_day"].dt.dayofyear != 1)
mask = dated.to_numpy()
feats = np.column_stack([xyz[mask],
                         days.to_numpy(dtype=np.float64)[mask] * (KM_PER_DAY / R_EARTH_KM)])
print(f"spatiotemporal pass over {len(feats)} dated+geocoded events", flush=True)
st = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=10,
                     metric="euclidean", algorithm="boruvka_kdtree",
                     core_dist_n_jobs=JOBS).fit_predict(feats)
df["st_cluster"] = -1
df.loc[mask, "st_cluster"] = st
n_st = int(st.max()) + 1
print(f"spatiotemporal clusters: {n_st}, noise points: {int((st == -1).sum())}", flush=True)

# ── temporal pass: flap periods over ALL dated events (geocoding not needed) ──
tdf = con.execute(f"""SELECT event_id, TRY_CAST(event_date AS DATE) event_day
                      FROM events WHERE {DATED}""").df()
tweek = ((tdf["event_day"] - pd.Timestamp("1970-01-01")).dt.days // 7).astype(int)
lo, hi = int(tweek.min()), int(tweek.max())
cnt = tweek.value_counts().reindex(np.arange(lo, hi + 1), fill_value=0).sort_index()
base = cnt.rolling(53, center=True, min_periods=13).median()
hot = cnt >= np.maximum(FLAP_RATIO * base.to_numpy(), FLAP_MIN_WEEK)
print(f"temporal pass over {len(tdf)} dated events, "
      f"{int(hot.sum())} hot weeks", flush=True)

# merge hot weeks into flap intervals, bridging single-week gaps
intervals = []
run = None
for w, h in zip(cnt.index, hot.to_numpy()):
    if h:
        if run and w - run[1] <= 2:
            run[1] = w
        else:
            if run:
                intervals.append(tuple(run))
            run = [w, w]
if run:
    intervals.append(tuple(run))

tdf["time_cluster"] = -1
flap_meta = []  # (id, peak_date, intensity) — the SQL summary joins this in
fid = 0
for w0, w1 in intervals:
    m = (tweek >= w0) & (tweek <= w1)
    n = int(m.sum())
    if n < FLAP_MIN_EVENTS:
        continue
    tdf.loc[m, "time_cluster"] = fid
    span = cnt.loc[w0:w1]
    peak_w = int(span.idxmax())
    peak_date = (pd.Timestamp("1970-01-01") + pd.Timedelta(days=peak_w * 7)).date()
    b = float(base.loc[peak_w]) if base.loc[peak_w] and base.loc[peak_w] > 0 else 1.0
    # a near-zero baseline in a sparse era would print an absurd multiple
    flap_meta.append((fid, str(peak_date), min(round(float(span.max()) / b, 1), 999.0)))
    fid += 1
print(f"flap periods kept: {fid}", flush=True)

# ── write back via registered frames; executemany UPDATE is far too slow ──
con.execute("ALTER TABLE events DROP COLUMN IF EXISTS wave_cluster")
con.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS geo_cluster INTEGER")
con.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS st_cluster INTEGER")
con.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS time_cluster INTEGER")
con.execute("UPDATE events SET geo_cluster=NULL, st_cluster=NULL, time_cluster=NULL")
con.register("_geo", df[["event_id", "geo_cluster", "st_cluster"]])
con.execute("""UPDATE events SET geo_cluster=l.geo_cluster, st_cluster=l.st_cluster
               FROM _geo l WHERE events.event_id=l.event_id""")
con.unregister("_geo")
con.register("_time", tdf[["event_id", "time_cluster"]])
con.execute("""UPDATE events SET time_cluster=l.time_cluster
               FROM _time l WHERE events.event_id=l.event_id""")
con.unregister("_time")

# cluster summary tables: centroid, size, span, top region
con.execute("DROP TABLE IF EXISTS geo_clusters")
con.execute("""CREATE TABLE geo_clusters AS
    SELECT geo_cluster,
           COUNT(*) n,
           AVG(lat) clat, AVG(lng) clng,
           MIN(event_year) yr_min, MAX(event_year) yr_max,
           MODE(COALESCE(NULLIF(loc_region,''), NULLIF(region,''))) top_region,
           MODE(COALESCE(NULLIF(loc_country,''), NULLIF(country,''))) top_country,
           MODE(NULLIF(loc_city,'')) top_city
    FROM events WHERE geo_cluster >= 0 GROUP BY geo_cluster ORDER BY n DESC""")

con.execute("DROP TABLE IF EXISTS st_clusters")
con.execute("""CREATE TABLE st_clusters AS
    SELECT st_cluster,
           COUNT(*) n,
           AVG(lat) clat, AVG(lng) clng,
           CAST(MIN(TRY_CAST(event_date AS DATE)) AS VARCHAR) date_min,
           CAST(MAX(TRY_CAST(event_date AS DATE)) AS VARCHAR) date_max,
           MIN(event_year) yr_min, MAX(event_year) yr_max,
           MODE(COALESCE(NULLIF(loc_region,''), NULLIF(region,''))) top_region,
           MODE(COALESCE(NULLIF(loc_country,''), NULLIF(country,''))) top_country,
           MODE(NULLIF(loc_city,'')) top_city
    FROM events WHERE st_cluster >= 0 GROUP BY st_cluster ORDER BY n DESC""")

flaps = pd.DataFrame(flap_meta, columns=["time_cluster", "peak_date", "intensity"])
con.register("_flaps", flaps)
con.execute("DROP TABLE IF EXISTS time_clusters")
con.execute("""CREATE TABLE time_clusters AS
    SELECT e.time_cluster,
           COUNT(*) n,
           COUNT(*) FILTER (e.lat IS NOT NULL) n_geo,
           AVG(e.lat) clat, AVG(e.lng) clng,
           CAST(MIN(TRY_CAST(e.event_date AS DATE)) AS VARCHAR) date_min,
           CAST(MAX(TRY_CAST(e.event_date AS DATE)) AS VARCHAR) date_max,
           MIN(e.event_year) yr_min, MAX(e.event_year) yr_max,
           MODE(COALESCE(NULLIF(e.loc_region,''), NULLIF(e.region,''))) top_region,
           MODE(COALESCE(NULLIF(e.loc_country,''), NULLIF(e.country,''))) top_country,
           MODE(NULLIF(e.loc_city,'')) top_city,
           ANY_VALUE(f.peak_date) peak_date,
           ANY_VALUE(f.intensity) intensity
    FROM events e JOIN _flaps f USING (time_cluster)
    WHERE e.time_cluster >= 0
    GROUP BY e.time_cluster ORDER BY e.time_cluster""")
con.unregister("_flaps")

print("\ntop 12 spatial clusters:", flush=True)
for r in con.execute("""SELECT geo_cluster, n, ROUND(clat,2), ROUND(clng,2),
                              top_region, top_country FROM geo_clusters LIMIT 12""").fetchall():
    print(f"  cluster {r[0]:>4}: {r[1]:>6} events @ ({r[2]},{r[3]}) — {r[4]}, {r[5]}")

print("\ntop 12 spatiotemporal waves:", flush=True)
for r in con.execute("""SELECT st_cluster, n, date_min, date_max,
                              top_city, top_region, top_country
                        FROM st_clusters LIMIT 12""").fetchall():
    print(f"  wave {r[0]:>4}: {r[1]:>6} events {r[2]}..{r[3]} — {r[4]}, {r[5]}, {r[6]}")

print("\nflap periods:", flush=True)
for r in con.execute("""SELECT time_cluster, n, date_min, date_max, intensity,
                              top_region, top_country
                        FROM time_clusters ORDER BY time_cluster""").fetchall():
    print(f"  flap {r[0]:>3}: {r[1]:>6} events {r[2]}..{r[3]} "
          f"(x{r[4]} baseline) — {r[5]}, {r[6]}")

con.close()
print("\nwrote geo_cluster + st_cluster + time_cluster columns "
      "and geo_clusters/st_clusters/time_clusters tables")
