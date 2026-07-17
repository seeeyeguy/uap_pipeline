#!/usr/bin/env python3
"""
Batch agreement/contradiction notes for multi-program events.
-------------------------------------------------------------
For every corpus.incidents row with program_count >= 2 (the event is attested
by documents from at least two source programs), pull the event's document
summaries from corpus.chunks and ask Claude (claude-haiku-4-5-20251001) to
emit {"agreements": [...], "contradictions": [...]}.

Results stream into data/event_notes.json incrementally (atomic tmp+rename
every FLUSH_EVERY events) keyed by event_id, so an interrupted run resumes
where it left off — already-noted events are skipped.

The corpus.incidents table (and its program_count column) is being built by a
separate job; if the column doesn't exist yet, the script polls briefly and
then reports zero eligible events instead of crashing.

Usage:
    export PG_DSN=postgresql://uap:...@localhost:5439/uapdb
    export ANTHROPIC_API_KEY=sk-ant-...
    .venv/bin/python scripts/event_contradictions.py --dry-run   # no API calls
    .venv/bin/python scripts/event_contradictions.py             # real run
    .venv/bin/python scripts/event_contradictions.py --limit 100 # first 100

Cost: ~$0.002/event at haiku-4.5 rates ($1/MTok in, $5/MTok out; roughly
1.5k input + 0.4k output tokens per event). --dry-run prints the estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg

MODEL = os.environ.get("EVENT_NOTES_MODEL", "claude-haiku-4-5-20251001")
OUT_PATH = Path("data/event_notes.json")
FLUSH_EVERY = 10          # events between atomic writes of the notes file
COST_PER_EVENT = 0.002    # USD, rough (~1.5k in + 0.4k out at $1/$5 per MTok)
MAX_DOCS_PER_EVENT = 12   # cap prompt size on heavily-documented events

SYSTEM_PROMPT = (
    "You are an analyst comparing multiple document summaries that describe "
    "the same UAP/UFO event, drawn from different collection programs "
    "(e.g. Blue Book, NICAP, foreign archives). Identify where the accounts "
    "AGREE on substantive facts (date, location, object description, "
    "behavior, witnesses, outcome) and where they CONTRADICT each other. "
    "Respond with valid JSON only, no markdown fences, in this exact shape: "
    '{"agreements": ["..."], "contradictions": ["..."]}. '
    "Each entry is one short factual sentence. Empty lists are fine."
)

USER_PROMPT = """Event record:
  event_id: {event_id}
  date: {event_date}
  location: {location}
  shape: {shape}
  description: {description}

Document summaries from {n_programs} program(s):

{doc_block}

Compare the accounts and return the agreements/contradictions JSON."""


def connect():
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        sys.exit("PG_DSN not set (postgresql://uap:...@localhost:5439/uapdb)")
    return psycopg.connect(dsn, autocommit=True)


def wait_for_events_table(cur, attempts: int = 3, delay: float = 5.0) -> bool:
    """True once corpus.incidents.program_count exists. The table is being
    (re)built by another job, so poll briefly instead of failing."""
    for i in range(attempts):
        cur.execute(
            """SELECT 1 FROM information_schema.columns
               WHERE table_schema='corpus' AND table_name='events'
                 AND column_name='program_count'"""
        )
        if cur.fetchone():
            return True
        if i < attempts - 1:
            print(f"corpus.incidents.program_count not present yet — "
                  f"retrying in {delay:.0f}s ({i + 1}/{attempts})")
            time.sleep(delay)
    return False


def fetch_events(cur, limit: int | None):
    sql = """SELECT event_id, event_date, city, region, country, shape,
                    COALESCE(description, ''), COALESCE(doc_ref, '')
             FROM corpus.incidents
             WHERE program_count >= 2
             ORDER BY event_id"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    return cur.fetchall()


def fetch_doc_summaries(cur, event_id: str, doc_ref: str):
    """Distinct (filename, source_program, summary) for the event's docs.

    Linkage: corpus-origin events carry doc_ref = the source document's
    filename stem; chunks store the full filename in meta. Also matches
    chunks whose related_case_ids mention the event_id, which is how the
    cross-program linkage lands. Adjust here if the events build settles
    on a dedicated docs column.
    """
    rows = []
    if doc_ref:
        cur.execute(
            """SELECT DISTINCT meta->>'filename', meta->>'source_program',
                               meta->>'summary'
               FROM corpus.chunks
               WHERE meta->>'summary' <> ''
                 AND (meta->>'filename' LIKE %s
                      OR meta->>'related_case_ids' LIKE %s)""",
            [doc_ref + ".%", "%" + doc_ref + "%"])
        rows = cur.fetchall()
    return rows[:MAX_DOCS_PER_EVENT]


def build_prompt(event, docs) -> str:
    (event_id, event_date, city, region, country, shape, desc, _doc_ref) = event
    location = ", ".join(x for x in (city, region, country) if x) or "unknown"
    programs = {d[1] or "unknown" for d in docs}
    doc_block = "\n\n".join(
        f"[{i + 1}] {fn or 'unknown file'} (program: {prog or 'unknown'})\n{summ}"
        for i, (fn, prog, summ) in enumerate(docs)
    ) or "(no document summaries found — use the event description only)"
    return USER_PROMPT.format(
        event_id=event_id, event_date=event_date or "unknown",
        location=location, shape=shape or "unknown",
        description=(desc or "")[:2000], n_programs=len(programs),
        doc_block=doc_block)


def parse_response(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    obj = json.loads(raw)
    return {"agreements": list(obj.get("agreements", [])),
            "contradictions": list(obj.get("contradictions", []))}


def atomic_write(notes: dict):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(notes, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(OUT_PATH)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first prompt + count/cost estimate; "
                         "no API calls, nothing written")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N events")
    args = ap.parse_args()

    conn = connect()
    cur = conn.cursor()

    if not wait_for_events_table(cur, attempts=3 if args.dry_run else 6):
        print("corpus.incidents has no program_count column yet (table still "
              "being built by the events job).")
        print(f"Estimated eligible events: unknown (stubbed to 0). "
              f"Cost estimate: 0 x ${COST_PER_EVENT:.3f} = $0.00. "
              f"Re-run once the column exists.")
        return

    events = fetch_events(cur, args.limit)
    est_cost = len(events) * COST_PER_EVENT
    print(f"{len(events)} events with program_count >= 2 | model {MODEL} | "
          f"estimated cost ~${est_cost:,.2f} (at ~${COST_PER_EVENT}/event)")

    if args.dry_run:
        if events:
            docs = fetch_doc_summaries(cur, events[0][0], events[0][7])
            print("\n--- first prompt (system) ---\n" + SYSTEM_PROMPT)
            print("\n--- first prompt (user) ---\n" + build_prompt(events[0], docs))
        print("\nDry run complete — no API calls made, nothing written.")
        return

    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

    notes: dict = {}
    if OUT_PATH.exists():
        notes = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"Resuming: {len(notes)} events already noted in {OUT_PATH}")

    pending = [e for e in events if e[0] not in notes]
    print(f"{len(pending)} events to process")

    done_since_flush = 0
    try:
        for i, event in enumerate(pending):
            event_id = event[0]
            docs = fetch_doc_summaries(cur, event_id, event[7])
            prompt = build_prompt(event, docs)
            try:
                resp = client.messages.create(
                    model=MODEL, max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}])
                if resp.stop_reason == "refusal" or not resp.content:
                    print(f"  {event_id}: refused (stop_reason="
                          f"{resp.stop_reason}) — skipping, no cache")
                    continue
                notes[event_id] = parse_response(resp.content[0].text)
            except json.JSONDecodeError as e:
                print(f"  {event_id}: unparseable response ({e}) — skipping")
                continue
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 500, 529):
                    print(f"  {event_id}: {e.status_code}, backing off 30s")
                    time.sleep(30)
                    continue  # not cached -> retried on next run
                raise
            done_since_flush += 1
            if done_since_flush >= FLUSH_EVERY:
                atomic_write(notes)
                done_since_flush = 0
                print(f"  {len(notes)}/{len(events)} events noted "
                      f"({i + 1}/{len(pending)} this run)")
    finally:
        atomic_write(notes)
        print(f"Wrote {len(notes)} event notes -> {OUT_PATH}")


if __name__ == "__main__":
    main()
