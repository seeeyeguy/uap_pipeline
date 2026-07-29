#!/usr/bin/env python3
"""
Backfill semantic display titles (meta.title) for corpus documents.

Users see raw archival filenames today — "1949-03-6792959-Rodeo-NewMexico.pdf",
"stantonfriedman-fbi1.pdf". This writes a human title per document:
"Project Blue Book case file — Rodeo, New Mexico, March 1949",
"FBI file — Stanton Friedman". Titles are display-only; filenames stay the
stable document key (URLs, citations, dedup).

Doc classes:
  * ID-ish filenames (Blue Book case pattern, numeric IDs, <2 real words):
    titled by Haiku from doc metadata (summary, people, orgs, type, place,
    date) — 25 docs per call, verdicts cached in data/title_verdicts.json.
  * Wordy filenames (articles, books, HTML refs): deterministic cleanup only
    (that text is already the best title; no LLM spend).
  * LLM abstentions fall back to a metadata template, then cleaned filename.

Duplicate titles get a disambiguating suffix (event date, else file id).

Usage:
  python backfill_titles.py --sample 30          # preview: print proposed titles
  python backfill_titles.py                      # dry run, full report
  python backfill_titles.py --apply              # write Postgres (PG_DSN-aware)
  python backfill_titles.py --apply --chroma     # also the Chroma build store
Never overwrites an existing meta.title unless --force.
"""
import argparse
import json
import os
import re
from pathlib import Path

import psycopg

ROOT = Path(__file__).parent
VERDICT_CACHE = ROOT / "data/title_verdicts.json"
LLM_MODEL = "claude-haiku-4-5-20251001"
BATCH = 25

BLUEBOOK_RE = re.compile(r"^[0-9x]{4}-[0-9x]{2}-[0-9]+", re.I)
WORD_RE = re.compile(r"[A-Za-z]{4,}")

MONTHS = ["", "January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    # works from the main checkout and from .worktrees/<name> alike
    env_file = next(p / "uap-api/.env" for p in (ROOT.parent, *ROOT.parents)
                    if (p / "uap-api/.env").exists())
    env = env_file.read_text()
    g = lambda k: re.search(rf"^{k}=(.*)$", env, re.M).group(1).strip()
    return f"postgresql://{g('POSTGRES_USER')}:{g('POSTGRES_PASSWORD')}@localhost:5439/{g('POSTGRES_DB')}"


def clean_filename(fname: str) -> str:
    t = re.sub(r"\.(pdf|html?|txt|csv)$", "", fname, flags=re.I)
    t = re.sub(r"__\d+_[a-z0-9.\-]+$", "", t)      # scraper suffixes (…__00_canada.com)
    t = t.replace("_", " ").replace("-", " ")
    return " ".join(t.split())


def is_wordy(fname: str) -> bool:
    """Filenames that already read as titles skip the LLM."""
    if BLUEBOOK_RE.match(fname):
        return False
    return len(WORD_RE.findall(clean_filename(fname))) >= 3


def month_year(date: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})", date or "")
    if not m:
        return (date or "")[:4]
    y, mo = m.group(1), int(m.group(2))
    return f"{MONTHS[mo]} {y}" if 1 <= mo <= 12 else y


def template_title(d: dict) -> str:
    """Deterministic fallback from metadata."""
    dtype = (d.get("document_type") or "document").replace("_", " ")
    prog = (d.get("source_program") or "").replace("_", " ")
    lead = f"{prog} {dtype}".strip() if prog and prog.lower() not in dtype.lower() else dtype
    place = ", ".join(x for x in (d.get("city"), d.get("region")) if x) \
        or (d.get("country") or "")
    when = month_year(d.get("event_date") or "")
    tail = ", ".join(x for x in (place, when) if x)
    title = f"{lead} — {tail}" if tail else lead
    return title[0].upper() + title[1:]


PROMPT = """You are titling documents for a public archive of UAP/UFO government \
records. For each document below, write a short display title (max 75 chars) a \
researcher would recognize in a list.

Rules:
- Lead with the collection or agency when known (FBI file, CIA memo, Project Blue \
Book case file, NARA microfilm…), then the central subject: a person, named \
incident, or place + date.
- Use ONLY the facts given. Never invent names, dates, or places.
- Person-centric files: "FBI file — Stanton Friedman". Sighting case files: \
"Project Blue Book case file — Rodeo, New Mexico, March 1949".
- Plain text only: no quotes, no trailing period, no markdown.
- If the metadata is too thin to improve on the filename, return "SKIP".

Return one JSON object per line: {{"id": <n>, "title": "<title or SKIP>"}}

Documents:
{docs}"""


def llm_titles(todo: list[dict], cache: dict) -> None:
    import anthropic
    client = anthropic.Anthropic()
    pending = [d for d in todo if d["filename"] not in cache]
    print(f"LLM pass: {len(pending)} docs to title ({len(todo) - len(pending)} cached)")
    for start in range(0, len(pending), BATCH):
        batch = pending[start:start + BATCH]
        lines = []
        for i, d in enumerate(batch):
            bits = {k: d[k] for k in ("filename", "document_type", "source_program",
                                      "originating_agency", "event_date", "city",
                                      "region", "country", "people", "organizations")
                    if d.get(k)}
            bits["summary"] = (d.get("summary") or "")[:350]
            lines.append(f"Doc {i}: {json.dumps(bits, ensure_ascii=False)}")
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=1500,
            messages=[{"role": "user", "content": PROMPT.format(docs="\n".join(lines))}])
        for line in resp.content[0].text.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                v = json.loads(line)
                d = batch[int(v["id"])]
            except (ValueError, KeyError, IndexError):
                continue
            title = str(v.get("title", "")).strip()
            cache[d["filename"]] = "" if title.upper() == "SKIP" else title[:90]
        VERDICT_CACHE.parent.mkdir(exist_ok=True)
        VERDICT_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
        print(f"  {min(start + BATCH, len(pending))}/{len(pending)}")


def load_docs(pg, limit: int = 0, only_untitled: bool = True):
    q = """
        SELECT DISTINCT ON (meta->>'filename')
               meta->>'filename', meta->>'document_type', meta->>'source_program',
               meta->>'originating_agency', meta->>'event_date',
               meta->>'city', meta->>'region', meta->>'country',
               meta->>'people', meta->>'organizations', meta->>'summary',
               meta->>'title'
        FROM corpus.chunks
        WHERE COALESCE(meta->>'filename','') != ''
        ORDER BY meta->>'filename', (meta->>'chunk_id')"""
    cols = ("filename", "document_type", "source_program", "originating_agency",
            "event_date", "city", "region", "country", "people", "organizations",
            "summary", "title")
    docs = []
    for row in pg.execute(q):
        d = dict(zip(cols, row))
        if only_untitled and d["title"]:
            continue
        for k in ("people", "organizations"):
            try:
                d[k] = ", ".join(json.loads(d[k] or "[]")[:4])
            except (ValueError, TypeError):
                d[k] = ""
        docs.append(d)
        if limit and len(docs) >= limit:
            break
    return docs


def resolve(docs: list[dict], cache: dict) -> dict[str, str]:
    """filename -> final title, deduplicated."""
    titles = {}
    for d in docs:
        f = d["filename"]
        if is_wordy(f):
            t = clean_filename(f)
        else:
            t = cache.get(f) or template_title(d)
            # a bare "Unknown"-class template beats nothing, but the cleaned
            # filename beats both — thin docs keep their name, not a shrug
            if t.lower().startswith("unknown") and len(WORD_RE.findall(clean_filename(f))) >= 1:
                t = clean_filename(f)
        titles[f] = t
    # disambiguate duplicates with event date, else the archival id
    seen = {}
    for f, t in titles.items():
        seen.setdefault(t.lower(), []).append(f)
    for dup_files in (v for v in seen.values() if len(v) > 1):
        for f in dup_files:
            d = next(x for x in docs if x["filename"] == f)
            suffix = d.get("event_date") or ""
            if not suffix:
                m = re.search(r"\d{5,}", f)
                suffix = f"file {m.group(0)}" if m else f
            titles[f] = f"{titles[f]} ({suffix})"
    return titles


def apply_pg(pg, titles: dict[str, str]):
    n = 0
    with pg.cursor() as cur:
        for f, t in titles.items():
            cur.execute(
                "UPDATE corpus.chunks SET meta = meta || %s::jsonb"
                " WHERE meta->>'filename' = %s",
                (json.dumps({"title": t}), f))
            n += cur.rowcount
    pg.commit()
    print(f"pg: {len(titles)} docs / {n} chunks updated")


def apply_chroma(titles: dict[str, str]):
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(ROOT / "data/vectordb"), settings=Settings(anonymized_telemetry=False))
    col = client.get_collection("uap_documents")
    updated = 0
    for f, t in titles.items():
        got = col.get(where={"filename": f}, include=["metadatas"])
        if not got["ids"]:
            continue
        for m in got["metadatas"]:
            m["title"] = t
        col.update(ids=got["ids"], metadatas=got["metadatas"])
        updated += len(got["ids"])
    print(f"chroma: {updated} chunks updated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write Postgres")
    ap.add_argument("--chroma", action="store_true", help="also update the build store")
    ap.add_argument("--no-llm", action="store_true", help="template/cleanup only")
    ap.add_argument("--force", action="store_true", help="retitle docs that have one")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="preview N ID-ish docs: old name -> proposed title")
    args = ap.parse_args()

    cache = json.loads(VERDICT_CACHE.read_text()) if VERDICT_CACHE.exists() else {}
    pg = psycopg.connect(dsn())
    docs = load_docs(pg, limit=args.limit, only_untitled=not args.force)
    idish = [d for d in docs if not is_wordy(d["filename"])]
    print(f"{len(docs)} untitled docs: {len(idish)} ID-ish (LLM), "
          f"{len(docs) - len(idish)} wordy (cleanup only)")

    if args.sample:
        subset = idish[: args.sample]
        if not args.no_llm:
            llm_titles(subset, cache)
        for d in subset:
            t = cache.get(d["filename"]) or template_title(d) + "  [template]"
            print(f"  {d['filename'][:52]:52} -> {t}")
        return

    if not args.no_llm:
        llm_titles(idish, cache)
    titles = resolve(docs, cache)
    if not args.apply:
        print("dry run — pass --apply to write. Example titles:")
        for f in list(titles)[:15]:
            print(f"  {f[:52]:52} -> {titles[f]}")
        return
    apply_pg(pg, titles)
    if args.chroma:
        apply_chroma(titles)


if __name__ == "__main__":
    main()
