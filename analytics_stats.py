#!/usr/bin/env python3
"""
Corpus-wide statistics (plan task 1): analytical passes over the unified
events store. Persists stats_* tables in analytics.duckdb and renders
ANALYTICS.md — the human-readable snapshot of the whole corpus.

Run after analytics_build.py / analytics_locations.py / analytics_cluster.py.
"""
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent
DB = ROOT / "data/analytics.duckdb"
OUT = ROOT / "ANALYTICS.md"

con = duckdb.connect(str(DB))
md = []


def table(title, sql, headers, fmt=None):
    """Run sql, persist nothing; render a markdown table section."""
    rows = con.execute(sql).fetchall()
    md.append(f"\n## {title}\n")
    md.append("| " + " | ".join(headers) + " |")
    md.append("|" + "---|" * len(headers))
    for r in rows:
        cells = fmt(r) if fmt else [str(c) if c is not None else "" for c in r]
        md.append("| " + " | ".join(cells) + " |")
    return rows


def persist(name, sql):
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"CREATE TABLE {name} AS {sql}")


# ── headline numbers ──
tot, geo, dated, city, clustered = con.execute("""SELECT COUNT(*),
    COUNT(*) FILTER (lat IS NOT NULL),
    COUNT(*) FILTER (event_year IS NOT NULL),
    COUNT(*) FILTER (location_id IS NOT NULL),
    COUNT(*) FILTER (geo_cluster >= 0) FROM events""").fetchone()
n_clusters = con.execute("SELECT COUNT(*) FROM geo_clusters").fetchone()[0]
n_waves = con.execute(
    "SELECT COUNT(DISTINCT wave_cluster) FROM events WHERE wave_cluster IS NOT NULL").fetchone()[0]

md.append("# UAP Corpus Analytics")
md.append(f"\n_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
          f"by analytics_stats.py — do not edit by hand._\n")
md.append(f"- **{tot:,} events** across sources (UFOSINT structured, NUFORC "
          f"narratives, enriched document corpus)")
md.append(f"- **{geo:,} geocoded** ({geo/tot:.0%}) · **{dated:,} dated** "
          f"({dated/tot:.0%}) · **{city:,} resolved to a canonical GeoNames "
          f"city** ({city/tot:.0%})")
md.append(f"- **{n_clusters:,} spatial hotspot clusters** covering "
          f"{clustered:,} events · **{n_waves:,} decade-windowed wave clusters**")

# ── per source ──
persist("stats_sources", """
    SELECT source, COUNT(*) events,
           COUNT(*) FILTER (lat IS NOT NULL) geocoded,
           COUNT(*) FILTER (event_year IS NOT NULL) dated,
           COUNT(*) FILTER (location_id IS NOT NULL) city_matched,
           MIN(event_year) earliest, MAX(event_year) latest
    FROM events GROUP BY source ORDER BY events DESC""")
table("Sources", "SELECT * FROM stats_sources",
      ["source", "events", "geocoded", "dated", "city matched", "earliest", "latest"],
      lambda r: [r[0], f"{r[1]:,}", f"{r[2]:,}", f"{r[3]:,}", f"{r[4]:,}",
                 str(r[5] or ""), str(r[6] or "")])

# ── per decade ──
persist("stats_decades", """
    SELECT (event_year//10)*10 AS decade, COUNT(*) events,
           COUNT(*) FILTER (source='ufosint') ufosint,
           COUNT(*) FILTER (source='nuforc') nuforc,
           COUNT(*) FILTER (source='corpus') corpus
    FROM events WHERE event_year BETWEEN 1900 AND 2029
    GROUP BY decade ORDER BY decade""")
table("Events per decade", "SELECT * FROM stats_decades",
      ["decade", "events", "ufosint", "nuforc", "corpus"],
      lambda r: [f"{r[0]}s", f"{r[1]:,}", f"{r[2]:,}", f"{r[3]:,}", f"{r[4]:,}"])

# ── shapes ──
persist("stats_shapes", """
    SELECT shape, COUNT(*) n,
           MIN(event_year) earliest, MAX(event_year) latest
    FROM events WHERE shape IS NOT NULL AND shape NOT IN ('', 'unknown')
    GROUP BY shape ORDER BY n DESC""")
table("Reported shapes (top 15)", "SELECT * FROM stats_shapes LIMIT 15",
      ["shape", "events", "earliest", "latest"],
      lambda r: [r[0], f"{r[1]:,}", str(r[2] or ""), str(r[3] or "")])

# shape mix shift by era: share of each era's reports
persist("stats_shape_eras", """
    WITH era AS (
      SELECT CASE WHEN event_year < 1970 THEN '1947-1969'
                  WHEN event_year < 2000 THEN '1970-1999'
                  ELSE '2000-present' END era, shape
      FROM events
      WHERE event_year >= 1947 AND shape NOT IN ('', 'unknown') AND shape IS NOT NULL)
    SELECT era, shape, COUNT(*) n,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (PARTITION BY era), 1) pct
    FROM era GROUP BY era, shape""")
table("Shape share by era (top 6 per era)", """
    SELECT era, shape, n, pct FROM (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY era ORDER BY n DESC) rk
      FROM stats_shape_eras) WHERE rk <= 6 ORDER BY era, n DESC""",
      ["era", "shape", "events", "% of era"],
      lambda r: [r[0], r[1], f"{r[2]:,}", f"{r[3]}%"])

# ── geography ──
persist("stats_countries", """
    SELECT COALESCE(NULLIF(loc_country,''), 'unresolved') country, COUNT(*) n
    FROM events GROUP BY 1 ORDER BY n DESC""")
table("Top countries", "SELECT * FROM stats_countries LIMIT 12",
      ["country", "events"], lambda r: [r[0], f"{r[1]:,}"])

persist("stats_us_states", """
    SELECT loc_region region, COUNT(*) n
    FROM events WHERE loc_cc='US' AND loc_region != ''
    GROUP BY 1 ORDER BY n DESC""")
table("Top US states", "SELECT * FROM stats_us_states LIMIT 15",
      ["state", "events"], lambda r: [r[0], f"{r[1]:,}"])

# ── hotspots ──
table("Top 20 spatial hotspots (all-time)", """
    SELECT geo_cluster, n, top_city, top_region, top_country,
           ROUND(clat,2), ROUND(clng,2), yr_min, yr_max
    FROM geo_clusters LIMIT 20""",
      ["cluster", "events", "city", "region", "country", "lat", "lng", "from", "to"],
      lambda r: [str(r[0]), f"{r[1]:,}", r[2] or "", r[3] or "", r[4] or "",
                 str(r[5]), str(r[6]), str(r[7] or ""), str(r[8] or "")])

# ── waves: dense decade-windowed concentrations ──
persist("stats_waves", """
    SELECT wave_cluster, COUNT(*) n,
           MODE(NULLIF(loc_city,'')) top_city,
           MODE(COALESCE(NULLIF(loc_region,''), NULLIF(region,''))) top_region,
           MODE(COALESCE(NULLIF(loc_country,''), NULLIF(country,''))) top_country,
           MIN(event_year) yr_min, MAX(event_year) yr_max
    FROM events WHERE wave_cluster IS NOT NULL
    GROUP BY wave_cluster ORDER BY n DESC""")
table("Largest wave concentrations per decade", """
    SELECT wave_cluster, n, top_city, top_region, top_country FROM (
      SELECT *, ROW_NUMBER() OVER (
        PARTITION BY split_part(wave_cluster, ':', 1) ORDER BY n DESC) rk
      FROM stats_waves) WHERE rk = 1 ORDER BY wave_cluster""",
      ["wave", "events", "city", "region", "country"],
      lambda r: [r[0], f"{r[1]:,}", r[2] or "", r[3] or "", r[4] or ""])

# ── quality (UFOSINT quality_score is the only per-event quality signal) ──
persist("stats_quality", """
    SELECT ROUND(quality,1) q, COUNT(*) n FROM events
    WHERE quality IS NOT NULL GROUP BY 1 ORDER BY 1""")
qrows = con.execute("""SELECT COUNT(*), ROUND(AVG(quality),3),
    ROUND(MEDIAN(quality),3) FROM events WHERE quality IS NOT NULL""").fetchone()
md.append(f"\n## Quality\n\nUFOSINT quality_score present on {qrows[0]:,} "
          f"events — mean {qrows[1]}, median {qrows[2]} "
          f"(distribution in `stats_quality`).")

OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
con.close()
print(f"wrote {OUT.name} + stats_* tables "
      f"(sources, decades, shapes, shape_eras, countries, us_states, waves, quality)")
