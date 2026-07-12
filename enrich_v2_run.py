#!/usr/bin/env python3
"""
Enrichment v2 batch orchestrator (pass 1: Haiku, rich metadata, no Q&A).

Submits Message Batches for every SAFE staged document and polls to
completion. Safe = documents whose final text is certainly the fresh OCR:
  - the sub-0.30 band (data/reocr_manifest.json stems), plus
  - docs that never had text before (stem absent from data/text/)
The 0.30-0.55 band is EXCLUDED until the text-layer merge decides its text.

Re-runnable: state ledger (data/enrich_v2_state.json) tracks submitted and
completed stems, so later runs (as more OCR lands in data/text_new/) only
submit new work. Results cache to data/enriched_v2/<stem>.json.
Batch-errored docs are recorded and retried on the next run.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from enrich_v2 import ENRICH_V2_SCHEMA, ENRICH_V2_SYSTEM, ENRICH_V2_USER

ROOT = Path(__file__).parent
# Wave 2+: read FINAL merged text. (Wave 1 read data/text_new pre-merge;
# its docs were the sub-0.30 safe subset, whose final text == fresh OCR,
# so those cached enrichments remain valid.)
STAGED = ROOT / "data/text"
OUT = ROOT / "data/enriched_v2"
STATE_F = ROOT / "data/enrich_v2_state.json"
MODEL = "claude-haiku-4-5"
BATCH_SIZE = 1000
SCHEMA_STR = json.dumps(ENRICH_V2_SCHEMA, indent=1)


def load_state():
    if STATE_F.exists():
        return json.loads(STATE_F.read_text())
    return {"cid_map": {}, "submitted": {}, "batches": [], "failed": {}}


def save_state(s):
    STATE_F.write_text(json.dumps(s, indent=1))


def safe_stems():
    # Post-merge: text is final. Enrich every non-empty document.
    return {p.stem for p in STAGED.glob("*.txt") if p.stat().st_size >= 200}


def cid(stem):
    return "d" + hashlib.sha1(stem.encode()).hexdigest()[:20]


def main():
    OUT.mkdir(exist_ok=True)
    client = anthropic.Anthropic()
    state = load_state()

    done = {p.stem for p in OUT.glob("*.json")}
    todo = sorted(safe_stems() - done - set(state["submitted"]))
    # failed docs from previous runs get retried
    retries = sorted(set(state.get("failed", {})) - done)
    for stem in retries:
        state["submitted"].pop(stem, None)
    todo = sorted(set(todo) | set(retries))
    state["failed"] = {}
    print(f"safe+staged: {len(safe_stems())} | cached: {len(done)} | submitting: {len(todo)}", flush=True)

    # ── submit ──
    for i in range(0, len(todo), BATCH_SIZE):
        chunk = todo[i:i + BATCH_SIZE]
        reqs = []
        for stem in chunk:
            text = (STAGED / f"{stem}.txt").read_text(encoding="utf-8", errors="ignore")[:12000]
            c = cid(stem)
            state["cid_map"][c] = stem
            reqs.append(Request(
                custom_id=c,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL, max_tokens=3000,
                    system=ENRICH_V2_SYSTEM,
                    messages=[{"role": "user", "content": ENRICH_V2_USER.format(
                        filename=stem + ".pdf", text=text, schema=SCHEMA_STR)}],
                )))
        b = client.messages.batches.create(requests=reqs)
        for stem in chunk:
            state["submitted"][stem] = b.id
        state["batches"].append(b.id)
        save_state(state)
        print(f"submitted batch {b.id} ({len(chunk)} docs)", flush=True)

    # ── poll ──
    pending = [bid for bid in state["batches"]]
    while pending:
        time.sleep(60)
        still = []
        for bid in pending:
            try:
                b = client.messages.batches.retrieve(bid)
            except Exception as e:
                print(f"poll error {bid}: {e}", flush=True)
                still.append(bid)
                continue
            if b.processing_status != "ended":
                still.append(bid)
                continue
            ok = err = 0
            for res in client.messages.batches.results(bid):
                stem = state["cid_map"].get(res.custom_id)
                if not stem:
                    continue
                if res.result.type == "succeeded":
                    msg = res.result.message
                    raw = next((bl.text for bl in msg.content if bl.type == "text"), "").strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1].removeprefix("json")
                    try:
                        parsed = json.loads(raw)
                        (OUT / f"{stem}.json").write_text(
                            json.dumps(parsed, indent=1, ensure_ascii=False), encoding="utf-8")
                        ok += 1
                    except ValueError:
                        state["failed"][stem] = "parse_error"
                        err += 1
                else:
                    state["failed"][stem] = res.result.type
                    err += 1
            print(f"batch {bid} ended: {ok} ok, {err} failed", flush=True)
            save_state(state)
        pending = still
        done_n = len(list(OUT.glob("*.json")))
        print(f"[{time.strftime('%H:%M')}] {len(pending)} batches pending | {done_n} docs cached", flush=True)

    print(f"ALL BATCHES COMPLETE: {len(list(OUT.glob('*.json')))} enriched, "
          f"{len(state['failed'])} failed (retried next run)", flush=True)


if __name__ == "__main__":
    main()
