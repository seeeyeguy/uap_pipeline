#!/usr/bin/env python3
"""
LLM batch notes for the landing page's "fresh from the archive" section.

For each recent ingest batch (a distinct catalog_seen.first_seen date), ask
Claude for an editorial note: what the batch is, why it matters, what's
notable — grounded in the batch's own stats and document summaries. Written
to corpus.catalog_batch_notes; /api/public/catalog/recent joins it in.

Run after `db/catalog.sql` (which stamps new filenames). Idempotent: batches
that already have a note are skipped unless --rebuild.

Usage: python build_batch_notes.py [--batches 3] [--rebuild]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import psycopg
from anthropic import Anthropic

ROOT = Path(__file__).parent
MODEL = "claude-sonnet-4-6"

PROMPT = """\
You write a short editorial note for the landing page of the Dark Forest
Archive (a searchable archive of UAP records) describing a batch of newly
ingested documents. Given the batch's statistics and sample documents,
return JSON: {"headline": "...", "note": "..."}.

- headline: under 60 chars, concrete, no hype ("Four decades of local-press
  UFO clippings", not "Amazing new documents!").
- note: 2-4 sentences. What the material is, why it matters to research,
  and one specific notable item. Sober, curious tone. State only facts
  present in the input. No markdown.
"""


def dsn():
    if os.environ.get("PG_DSN"):
        return os.environ["PG_DSN"]
    env = (ROOT.parent / "uap-api/.env").read_text()
    g = lambda k: re.search(rf"^{k}=(.*)$", env, re.M).group(1).strip()
    return (f"postgresql://{g('POSTGRES_USER')}:{g('POSTGRES_PASSWORD')}"
            f"@localhost:5439/{g('POSTGRES_DB')}")


def batch_context(cur, day):
    cur.execute("""
        SELECT COALESCE(source_program,'unknown'), COUNT(*)
        FROM corpus.catalog_docs WHERE first_seen = %s
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10""", (day,))
    programs = cur.fetchall()
    cur.execute("""
        SELECT COALESCE(document_type,'unknown'), COUNT(*)
        FROM corpus.catalog_docs WHERE first_seen = %s
        GROUP BY 1 ORDER BY 2 DESC LIMIT 8""", (day,))
    types = cur.fetchall()
    cur.execute("""
        SELECT filename, LEFT(COALESCE(summary,''), 400)
        FROM corpus.catalog_docs
        WHERE first_seen = %s AND summary IS NOT NULL AND summary <> ''
        ORDER BY official_source DESC, pages DESC NULLS LAST
        LIMIT 12""", (day,))
    samples = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM corpus.catalog_docs WHERE first_seen = %s",
                (day,))
    total = cur.fetchone()[0]
    return {
        "batch_date": str(day), "total_documents": total,
        "by_source_program": [{"program": p, "docs": n} for p, n in programs],
        "by_document_type": [{"type": t, "docs": n} for t, n in types],
        "sample_documents": [{"filename": f, "summary": s} for f, s in samples],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    client = Anthropic()
    with psycopg.connect(dsn()) as pg, pg.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS corpus.catalog_batch_notes (
                first_seen date PRIMARY KEY,
                headline   text NOT NULL,
                note       text NOT NULL,
                built_at   timestamptz NOT NULL DEFAULT now()
            )""")
        cur.execute("""
            SELECT DISTINCT first_seen FROM corpus.catalog_docs
            ORDER BY first_seen DESC LIMIT %s""", (args.batches,))
        days = [r[0] for r in cur.fetchall()]
        for day in days:
            if not args.rebuild:
                cur.execute("SELECT 1 FROM corpus.catalog_batch_notes WHERE first_seen = %s",
                            (day,))
                if cur.fetchone():
                    print(f"{day}: note exists, skipping")
                    continue
            ctx = batch_context(cur, day)
            msg = client.messages.create(
                model=MODEL, max_tokens=500,
                system=PROMPT,
                messages=[{"role": "user", "content": json.dumps(ctx)}])
            raw = msg.content[0].text.strip()
            raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
            note = json.loads(raw)
            cur.execute("""
                INSERT INTO corpus.catalog_batch_notes (first_seen, headline, note, built_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (first_seen)
                DO UPDATE SET headline = EXCLUDED.headline,
                              note = EXCLUDED.note, built_at = now()""",
                (day, note["headline"][:120], note["note"][:1200]))
            pg.commit()
            print(f"{day}: {note['headline']}")


if __name__ == "__main__":
    sys.exit(main())
