#!/usr/bin/env python3
"""
Backfill location metadata on chunks whose enrichment never resolved a place.

The Portage County incident: the Blue Book case file
1966-04-7104469-Ravenna-Mantua-Ohio.pdf carried loc_city/loc_region "" and
was excluded by every region-filtered retrieval. ~2k case-file documents
share the gap while their filenames literally name the place.

Passes (all skip chunks that already have a value — never overwrites):

  A. Filename parsing: Blue Book-style names (YYYY-MM-ID-City[-City2]-State)
     with CamelCase splitting, state-abbreviation expansion (Pa, Ariz,
     Calif...), trailing junk (case numbers, _ILLEGIBLE_) dropped, and
     validation against the GeoNames dimension (corpus.locations). Also
     resolves City-Country and bare-Country names.
  B. Intra-document propagation: chunks missing geography inherit their
     document's known location when sibling chunks have one.
  C. (--llm) Haiku extraction for still-blank case-file documents from their
     summary text, GeoNames-validated, verdicts cached on disk.

Books, journals, transcripts and indexes are left alone on purpose: they
have no single location, and retrieval's $eq_or_unknown filter semantics
keep them reachable.

Targets: Postgres (--apply; PG_DSN-aware, so dev and prod both work) and the
Chroma build store (--chroma) so every future pg_publish carries the fix.

Usage:
  python backfill_locations.py                 # dry run, report only
  python backfill_locations.py --apply         # write Postgres
  python backfill_locations.py --apply --chroma  # also fix the build store
  python backfill_locations.py --apply --llm   # include pass C
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import psycopg

ROOT = Path(__file__).parent
VERDICT_CACHE = ROOT / "data/location_verdicts.json"
LLM_MODEL = "claude-haiku-4-5-20251001"

STATE_ABBREV = {
    "ala": "Alabama", "ariz": "Arizona", "ark": "Arkansas", "calif": "California",
    "colo": "Colorado", "conn": "Connecticut", "del": "Delaware", "fla": "Florida",
    "ga": "Georgia", "ida": "Idaho", "ill": "Illinois", "ind": "Indiana",
    "ia": "Iowa", "kan": "Kansas", "kans": "Kansas", "ky": "Kentucky",
    "la": "Louisiana", "md": "Maryland", "mass": "Massachusetts",
    "mich": "Michigan", "minn": "Minnesota", "miss": "Mississippi",
    "mo": "Missouri", "mont": "Montana", "neb": "Nebraska", "nebr": "Nebraska",
    "nev": "Nevada", "nm": "New Mexico", "nmex": "New Mexico",
    "ny": "New York", "nj": "New Jersey", "nc": "North Carolina",
    "nd": "North Dakota", "okla": "Oklahoma", "ore": "Oregon", "oreg": "Oregon",
    "pa": "Pennsylvania", "penn": "Pennsylvania", "ri": "Rhode Island",
    "sc": "South Carolina", "sd": "South Dakota", "tenn": "Tennessee",
    "tex": "Texas", "vt": "Vermont", "va": "Virginia", "wash": "Washington",
    "wva": "West Virginia", "wis": "Wisconsin", "wisc": "Wisconsin",
    "wyo": "Wyoming", "me": "Maine", "nh": "New Hampshire", "ut": "Utah",
    "okl": "Oklahoma", "mex": "Mexico",
}

FNAME_RE = re.compile(r"^[0-9x]{4}-[0-9x]{2}-[0-9]+-(.+?)\.pdf$", re.I)
JUNK_SEG = re.compile(r"^(\d+|_?illegible_?|)$", re.I)
COORDish = re.compile(r"\d")


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    g = lambda k: re.search(rf"^{k}=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{g('POSTGRES_USER')}:{g('POSTGRES_PASSWORD')}@localhost:5439/{g('POSTGRES_DB')}"


def camel_split(s: str) -> str:
    """BlackRiverFalls -> Black River Falls; leaves ALLCAPS alone."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).strip()


class Gazetteer:
    """City/region/country lookups over the GeoNames dimension."""

    def __init__(self, pg):
        rows = pg.execute(
            "SELECT city, region, COALESCE(region_code,''), country"
            " FROM corpus.locations").fetchall()
        self.city_region = defaultdict(set)   # lower city -> {(region, country)}
        self.regions = {}                     # lower region name/code -> region
        self.region_country = {}
        self.countries = set()
        for city, region, rcode, country in rows:
            if city:
                self.city_region[city.lower()].add((region or "", country or ""))
            if region:
                self.regions[region.lower()] = region
                self.region_country[region] = country or ""
                if rcode:
                    self.regions.setdefault(rcode.lower(), region)
            if country:
                self.countries.add(country)
        self.countries_l = {c.lower(): c for c in self.countries}

    def region(self, token: str):
        t = token.lower()
        if t in STATE_ABBREV:
            return self.regions.get(STATE_ABBREV[t].lower())
        return self.regions.get(t)

    def country(self, token: str):
        return self.countries_l.get(token.lower())

    def city_in(self, city: str, region: str | None, country: str | None):
        """Return the canonical city name if it exists (in region/country)."""
        for cand_region, cand_country in self.city_region.get(city.lower(), ()):
            if region and cand_region != region:
                continue
            if country and cand_country != country:
                continue
            return city
        return None


def parse_filename(fname: str, gaz: Gazetteer):
    """-> (city, region, country) with "" for unknowns, or None if unparseable."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    segs = [s for s in m.group(1).split("-") if not JUNK_SEG.match(s)]
    # trailing case numbers / coordinate-ish segments are noise
    while segs and COORDish.search(segs[-1]):
        segs.pop()
    if not segs:
        return None
    segs = [camel_split(s) for s in segs if not COORDish.search(s)]
    if not segs:
        return None

    last = segs[-1]
    region = gaz.region(last)
    if region:
        country = gaz.region_country.get(region, "United States")
        city = ""
        for cand in segs[:-1]:
            hit = gaz.city_in(cand, region, None)
            if hit:
                city = hit
                break
        if not city and len(segs) > 1:
            city = segs[0]  # unvalidated but parsed — still useful for display
        return (city, region, country or "United States")

    country = gaz.country(last)
    if country:
        city = ""
        for cand in segs[:-1]:
            hit = gaz.city_in(cand, None, country)
            if hit:
                city = hit
                break
        return (city, country if False else "", country)

    # single segment that is a known region ("Indiana") or city was junk
    if len(segs) == 1:
        r = gaz.region(last)
        if r:
            return ("", r, gaz.region_country.get(r, "United States"))
    return None


BLANK = "COALESCE(meta->>'loc_region','') = '' AND COALESCE(meta->>'loc_city','') = ''"


def pass_a(pg, gaz):
    rows = pg.execute(
        f"SELECT DISTINCT meta->>'filename' FROM corpus.chunks WHERE {BLANK}"
        " AND meta->>'filename' ~ '^[0-9x]{4}-[0-9x]{2}-[0-9]+-'").fetchall()
    out = {}
    for (fname,) in rows:
        parsed = parse_filename(fname or "", gaz)
        if parsed and (parsed[0] or parsed[1] or parsed[2]):
            out[fname] = parsed
    return out


def pass_b(pg):
    rows = pg.execute(f"""
        WITH known AS (
            SELECT meta->>'filename' AS f,
                   mode() WITHIN GROUP (ORDER BY meta->>'loc_city') AS city,
                   mode() WITHIN GROUP (ORDER BY meta->>'loc_region') AS region,
                   mode() WITHIN GROUP (ORDER BY meta->>'loc_country') AS country
            FROM corpus.chunks
            WHERE COALESCE(meta->>'loc_region','') <> ''
            GROUP BY 1
        )
        SELECT k.f, k.city, k.region, k.country
        FROM known k
        WHERE EXISTS (SELECT 1 FROM corpus.chunks c
                      WHERE c.meta->>'filename' = k.f AND {BLANK})""").fetchall()
    return {f: (city or "", region or "", country or "") for f, city, region, country in rows}


def pass_c(pg, gaz, updates):
    import anthropic
    cache = json.loads(VERDICT_CACHE.read_text()) if VERDICT_CACHE.exists() else {}
    rows = pg.execute(f"""
        SELECT meta->>'filename', max(left(text, 600))
        FROM corpus.chunks
        WHERE {BLANK}
          AND meta->>'document_type' IN ('sighting_report','investigation_report',
              'government_memo','intelligence_report','witness_statement')
          AND COALESCE(meta->>'filename','') <> ''
        GROUP BY 1""").fetchall()
    todo = [(f, t) for f, t in rows if f not in updates and f not in cache]
    print(f"pass C: {len(rows)} candidate docs ({len(todo)} need adjudication)")
    client = anthropic.Anthropic() if todo else None
    for start in range(0, len(todo), 20):
        batch = todo[start:start + 20]
        lines = [f"Doc {i} ({f}): {t[:400]}" for i, (f, t) in enumerate(batch)]
        msg = client.messages.create(
            model=LLM_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content":
                "For each numbered UAP case document below, extract the sighting "
                "location. Answer ONLY a JSON array of "
                '{"id": n, "city": "...", "region": "...", "country": "..."} '
                '(use "" when not stated; region = state/province full name; '
                "never guess).\n\n" + "\n".join(lines)}])
        text = msg.content[0].text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            verdicts = json.loads(text.strip())
        except json.JSONDecodeError:
            continue
        for v in verdicts:
            try:
                fname = batch[int(v["id"])][0]
            except (KeyError, ValueError, IndexError):
                continue
            cache[fname] = {"city": v.get("city", ""), "region": v.get("region", ""),
                            "country": v.get("country", "")}
        VERDICT_CACHE.write_text(json.dumps(cache, indent=1))
        print(f"  adjudicated {min(start + 20, len(todo))}/{len(todo)}")
    added = 0
    for fname, v in cache.items():
        if fname in updates:
            continue
        region = gaz.regions.get((v.get("region") or "").lower(), "")
        country = gaz.countries_l.get((v.get("country") or "").lower(), "")
        if not (region or country):
            continue
        if region and not country:
            country = gaz.region_country.get(region, "")
        city = gaz.city_in(v.get("city") or "", region or None, country or None) or ""
        updates[fname] = (city, region, country)
        added += 1
    print(f"pass C: {added} docs resolved from text")
    return updates


def apply_pg(pg, updates):
    n = 0
    with pg.cursor() as cur:
        for fname, (city, region, country) in updates.items():
            patch = {}
            if city:
                patch["loc_city"] = city
            if region:
                patch["loc_region"] = region
            if country:
                patch["loc_country"] = country
            if not patch:
                continue
            cur.execute(
                f"UPDATE corpus.chunks SET meta = meta || %s::jsonb"
                f" WHERE meta->>'filename' = %s AND {BLANK}",
                (json.dumps(patch), fname))
            n += cur.rowcount
    pg.commit()
    print(f"postgres: {n} chunks updated across {len(updates)} docs")


def apply_chroma(updates):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(ROOT / "data/vectordb"), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection("uap_documents")
    n = 0
    for fname, (city, region, country) in updates.items():
        got = col.get(where={"filename": fname}, include=["metadatas"])
        ids, metas = [], []
        for cid, meta in zip(got["ids"], got["metadatas"]):
            meta = dict(meta or {})
            if (meta.get("loc_region") or meta.get("loc_city")):
                continue
            if city:
                meta["loc_city"] = city
            if region:
                meta["loc_region"] = region
            if country:
                meta["loc_country"] = country
            ids.append(cid)
            metas.append(meta)
        if ids:
            col.update(ids=ids, metadatas=metas)
            n += len(ids)
    print(f"chroma build store: {n} chunks updated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write Postgres")
    ap.add_argument("--chroma", action="store_true", help="also update the build store")
    ap.add_argument("--llm", action="store_true", help="pass C: text extraction")
    ap.add_argument("--report", type=int, default=20)
    args = ap.parse_args()

    with psycopg.connect(dsn()) as pg:
        gaz = Gazetteer(pg)
        updates = pass_a(pg, gaz)
        print(f"pass A (filenames): {len(updates)} docs")
        for f, loc in pass_b(pg).items():
            updates.setdefault(f, loc)
        print(f"pass B (propagation): {len(updates)} docs total")
        if args.llm:
            updates = pass_c(pg, gaz, updates)

        for fname, (city, region, country) in list(updates.items())[:args.report]:
            print(f"  {fname[:52]:54s} -> {city or '·'}, {region or '·'}, {country or '·'}")

        if args.apply:
            apply_pg(pg, updates)
        else:
            print("\ndry run — pass --apply to write")
    if args.apply and args.chroma:
        apply_chroma(updates)


if __name__ == "__main__":
    main()
