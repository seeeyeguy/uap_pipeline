#!/usr/bin/env python3
"""
Geo-clustering over the analytics events (task 4). HDBSCAN over all geocoded
events finds spatial clusters (hotspots); a per-decade pass flags flap/wave
concentrations. Cluster IDs written back to the events table.

Coordinates are mapped to 3D unit-sphere points so Boruvka can use a KDTree:
chord distance is a monotonic function of great-circle distance, so the
HDBSCAN hierarchy matches true haversine exactly. The first run OOM-killed
(exit 137) using the BallTree/haversine path with core_dist_n_jobs=-1 on 20
cores; jobs are capped at 4 here.
"""
import numpy as np
import duckdb
import hdbscan

DB = "data/analytics.duckdb"
JOBS = 4
con = duckdb.connect(DB)

df = con.execute("""SELECT event_id, lat, lng, event_year FROM events
                    WHERE lat IS NOT NULL AND lng IS NOT NULL
                    AND lat BETWEEN -90 AND 90 AND lng BETWEEN -180 AND 180""").df()
lat = np.radians(df["lat"].to_numpy(dtype=np.float64))
lng = np.radians(df["lng"].to_numpy(dtype=np.float64))
xyz = np.column_stack([np.cos(lat) * np.cos(lng),
                       np.cos(lat) * np.sin(lng),
                       np.sin(lat)])
years = df["event_year"].fillna(0).to_numpy(dtype=np.int64)
print(f"clustering {len(df)} geocoded events", flush=True)

# ── spatial clusters — min_cluster_size tuned for hotspot granularity ──
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10,
                            metric="euclidean", algorithm="boruvka_kdtree",
                            core_dist_n_jobs=JOBS)
df["geo_cluster"] = clusterer.fit_predict(xyz)
n_clusters = df["geo_cluster"].max() + 1
noise = int((df["geo_cluster"] == -1).sum())
print(f"spatial clusters: {n_clusters}, noise points: {noise}", flush=True)

# ── per-decade pass: flap/wave concentrations (denser threshold, short window) ──
df["wave_cluster"] = ""
for dec in range(1900, 2030, 10):
    mask = (years >= dec) & (years < dec + 10)
    n = int(mask.sum())
    if n < 200:
        continue
    sub = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=10,
                          metric="euclidean", algorithm="boruvka_kdtree",
                          core_dist_n_jobs=JOBS).fit_predict(xyz[mask])
    df.loc[mask, "wave_cluster"] = [f"{dec}s:{l}" if l >= 0 else "" for l in sub]
    waves = len(set(sub)) - (1 if -1 in sub else 0)
    print(f"  {dec}s: {n} events -> {waves} wave clusters", flush=True)

# ── write back via a registered frame; executemany UPDATE is far too slow ──
con.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS geo_cluster INTEGER")
con.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS wave_cluster VARCHAR")
con.register("_labels", df[["event_id", "geo_cluster", "wave_cluster"]])
con.execute("""UPDATE events SET geo_cluster=l.geo_cluster,
                                 wave_cluster=NULLIF(l.wave_cluster, '')
               FROM _labels l WHERE events.event_id=l.event_id""")
con.unregister("_labels")

# cluster summary table: centroid, size, span of years, top region
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

print("\ntop 12 spatial clusters:", flush=True)
for r in con.execute("""SELECT geo_cluster, n, ROUND(clat,2), ROUND(clng,2),
                              top_region, top_country FROM geo_clusters LIMIT 12""").fetchall():
    print(f"  cluster {r[0]:>4}: {r[1]:>6} events @ ({r[2]},{r[3]}) — {r[4]}, {r[5]}")

con.close()
print("\nwrote geo_cluster + wave_cluster columns and geo_clusters table")
