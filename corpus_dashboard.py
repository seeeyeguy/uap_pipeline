#!/usr/bin/env python3
"""
Live dashboard for the corpus: new-document downloads, ingest progress, and
the published corpus. Stdlib only — no new dependencies.

    .venv/bin/python corpus_dashboard.py   # then open http://<host>:8877

Replaces the cloud-OCR dashboard (ocr_dashboard.py): the RunPod fleet, rsync
upload, and enrichment-v2 batch cards all tracked a finished phase and are
gone. Data sources now:

  - data/*download*.log, data/tranche*.log   freshest tqdm line = active fetch
  - data/progress.json                       ingest ledger (doc -> when)
  - postgres (docker exec ... psql)          published chunks/docs/events
  - data/vectordb_releases/                  latest published vector release
"""
import json
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT     = Path(__file__).parent
PORT     = 8877
POLL_S   = 30      # logs + ledger
PG_S     = 120     # published-corpus counts
PG_EXEC  = ["docker", "exec", "uap-api-postgres-1",
            "psql", "-U", "uap", "-d", "uapdb", "-tA", "-F", "|", "-c"]

STATE = {"updated": None, "history": deque(maxlen=480),   # (t, ingested_total)
         "ncs_history": deque(maxlen=480)}                # (t, ncs_enriched)


def _read_tail(path, n=6000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def download_status():
    """Freshest download log with a tqdm line wins — works for the current
    AFU tranche and for whatever run replaces it."""
    logs = [p for pat in ("*download*.log", "tranche*.log")
            for p in (ROOT / "data").glob(pat)]
    best = None
    for p in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
        tail = _read_tail(p).replace("\r", "\n")
        m = None
        for m2 in re.finditer(
                r"Downloading:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*"
                r"\[([\d:]+)<([\d:?]+),\s*([\d.]+)\s*file/s\]", tail):
            m = m2
        if m:
            best = {"log": p.name, "pct": int(m.group(1)),
                    "n": int(m.group(2)), "total": int(m.group(3)),
                    "elapsed": m.group(4), "eta": m.group(5),
                    "rate": float(m.group(6)),
                    "stale_s": round(time.time() - p.stat().st_mtime)}
            break
    return best


def proc_status():
    def alive(pat):
        return subprocess.run(["pgrep", "-f", pat],
                              capture_output=True).returncode == 0
    return {"download":  alive(r"pipeline\.py download"),
            "ingest":    alive(r"pipeline\.py (ingest|all)"),
            "publish":   alive(r"pipeline\.py publish|pg_publish"),
            "dashboard": True}


def ingest_status():
    """The ingest ledger: every completed document with its timestamp."""
    try:
        led = json.loads((ROOT / "data/progress.json").read_text())
    except Exception as e:
        return {"error": str(e)[:120]}
    done, failed = led.get("completed", {}), led.get("failed", {})
    now = datetime.now()
    per_source, last24, last1h, latest = {}, 0, 0, None
    for key, info in done.items():
        src = key.split("/", 1)[0] if "/" in key else "(root)"
        per_source[src] = per_source.get(src, 0) + 1
        ts = info.get("completed_at")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if latest is None or t > latest[0]:
            latest = (t, key)
        if now - t <= timedelta(hours=24):
            last24 += 1
            if now - t <= timedelta(hours=1):
                last1h += 1
    return {
        "total": len(done),
        "failed": len(failed),
        "last24h": last24,
        "last1h": last1h,
        "latest": {"doc": latest[1], "at": latest[0].strftime("%m-%d %H:%M")}
                  if latest else None,
        "per_source": dict(sorted(per_source.items(),
                                  key=lambda kv: -kv[1])[:10]),
    }


def corpus_status():
    """Published corpus, straight from the serving Postgres."""
    q = ("SELECT (SELECT count(*) FROM corpus.chunks), "
         "(SELECT count(DISTINCT meta->>'filename') FROM corpus.chunks), "
         "(SELECT count(*) FROM corpus.events)")
    try:
        out = subprocess.run(PG_EXEC + [q], capture_output=True,
                             text=True, timeout=30)
        chunks, docs, events = out.stdout.strip().split("|")
        rel = sorted(p.name for p in (ROOT / "data/vectordb_releases").iterdir()
                     if p.is_dir())
        return {"chunks": int(chunks), "docs": int(docs), "events": int(events),
                "release": rel[-1] if rel else None}
    except Exception as e:
        return {"error": str(e)[:120]}


def ncs_status():
    """UFO Newsclipping Service ingest (2026-07): 492 issues moving through
    text-extraction -> enrichment -> embed. Enriched count is the live phase
    signal (embedding + ledger land in one batch at the end)."""
    pat = "UFO Newsclipping Service*"
    target = len(list((ROOT / "data/ingest_ncs/afu_se").glob("*.pdf"))) or 492
    text = len(list((ROOT / "data/text").glob(pat + ".txt")))
    enriched = len(list((ROOT / "data/enriched").glob(pat + ".json")))
    ocr = len(list((ROOT / "data/images").glob(pat)))
    ledger = 0
    try:
        led = json.loads((ROOT / "data/progress.json").read_text())
        ledger = sum(1 for k in led.get("completed", {}) if "Newsclipping" in k)
    except Exception:
        pass
    latest = None
    cands = sorted((ROOT / "data/enriched").glob(pat + ".json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        latest = {"doc": cands[0].stem,
                  "age_s": round(time.time() - cands[0].stat().st_mtime)}
    # Enrichment runs through the Message Batches API: results land in one
    # merge, so surface the submitted-batch state or the file stays "quiet".
    batch = None
    tail = _read_tail(ROOT / "data/ncs_ingest.log", 20000)
    for m in re.finditer(r"Enrichment batch (msgbatch_\w+): (\d+) docs submitted"
                         r"|batch (msgbatch_\w+) ended: (\d+) ok", tail):
        if m.group(1):
            batch = {"id": m.group(1)[:18] + "…", "docs": int(m.group(2)),
                     "state": "polling"}
        else:
            batch = {"id": m.group(3)[:18] + "…", "docs": int(m.group(4)),
                     "state": "done"}
    return {"target": target, "text": text, "enriched": enriched,
            "ocr_fallbacks": ocr, "embedded": ledger, "latest": latest,
            "batch": batch}


def ncs_rate():
    """Issues enriched per hour over the sample window, with a naive ETA."""
    h = list(STATE["ncs_history"])
    cutoff = time.time() - 3600
    win = [x for x in h if x[0] >= cutoff] or h
    if len(win) < 2 or win[-1][0] <= win[0][0]:
        return None
    gain = win[-1][1] - win[0][1]
    if gain <= 0:
        return None
    per_hr = gain / ((win[-1][0] - win[0][0]) / 3600)
    ncs = STATE.get("ncs") or {}
    left = max(0, (ncs.get("target") or 0) - (ncs.get("enriched") or 0))
    return {"per_hr": round(per_hr, 1), "eta_h": round(left / per_hr, 1)}


def poller():
    while True:
        STATE["download"] = download_status()
        STATE["ingest"] = ingest_status()
        STATE["ncs"] = ncs_status()
        STATE["procs"] = proc_status()
        STATE["updated"] = time.time()
        if isinstance(STATE["ingest"].get("total"), int):
            STATE["history"].append((time.time(), STATE["ingest"]["total"]))
        if isinstance(STATE["ncs"].get("enriched"), int):
            STATE["ncs_history"].append((time.time(), STATE["ncs"]["enriched"]))
        time.sleep(POLL_S)


def pg_poller():
    while True:
        STATE["corpus"] = corpus_status()
        time.sleep(PG_S)


def ingest_rate():
    """Docs/hour over the last hour of samples (positive deltas only)."""
    h = list(STATE["history"])
    cutoff = time.time() - 3600
    win = [x for x in h if x[0] >= cutoff] or h
    if len(win) < 2 or win[-1][0] <= win[0][0]:
        return None
    gain = sum(max(0, b[1] - a[1]) for a, b in zip(win, win[1:]))
    return round(gain / ((win[-1][0] - win[0][0]) / 3600), 1)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>UAP Corpus Dashboard</title>
<style>
 body{background:#101418;color:#dde3ea;font-family:ui-monospace,Menlo,monospace;margin:2rem}
 h1{font-size:1.2rem;color:#7ec8f8} .cards{display:flex;flex-wrap:wrap;gap:1rem}
 .card{background:#1a2027;border:1px solid #2a323c;border-radius:8px;padding:1rem 1.2rem;min-width:280px;flex:1}
 .card h2{font-size:.85rem;color:#8fa3b8;margin:0 0 .6rem;text-transform:uppercase;letter-spacing:.08em}
 .big{font-size:1.6rem;color:#fff} .bar{background:#2a323c;border-radius:4px;height:10px;margin:.5rem 0}
 .fill{background:#4aa3df;height:10px;border-radius:4px} .ok{color:#5fd38d}.bad{color:#e46a6a}.warn{color:#e5b567}
 .dim{color:#67788c;font-size:.8rem} table{width:100%;font-size:.78rem;border-collapse:collapse}
 td{padding:.15rem .3rem;border-bottom:1px solid #232b34;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:420px}
</style></head><body>
<h1>UAP corpus dashboard</h1>
<div class="cards" id="cards">loading…</div>
<p class="dim" id="foot"></p>
<script>
const esc = t => String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;');
function bar(p){return `<div class="bar"><div class="fill" style="width:${p}%"></div></div>`}
async function tick(){
  const r = await fetch('/api/status'); const s = await r.json();
  let html = '';

  const d = s.download;
  if (d) {
    const stale = d.stale_s > 180;
    html += `<div class="card"><h2>New downloads — ${esc(d.log)}</h2>
      <div class="big">${d.n.toLocaleString()} / ${d.total.toLocaleString()}</div>
      ${bar(d.pct)}
      <div>${d.pct}% · ${d.rate} files/s · eta ${esc(d.eta)}</div>
      <div class="${stale?'warn':'dim'}">${stale?'log quiet for '+Math.round(d.stale_s/60)+' min':'elapsed '+esc(d.elapsed)}</div></div>`;
  } else {
    html += `<div class="card"><h2>New downloads</h2>
      <div class="big dim">idle</div><div class="dim">no active download log</div></div>`;
  }

  const n = s.ncs;
  if (n && n.target) {
    const pct = Math.round(100*(n.enriched||0)/n.target);
    const quiet = n.latest && n.latest.age_s > 900;
    html += `<div class="card"><h2>Newsclipping Service ingest (1969–2011)</h2>
      <div class="big">${n.enriched||0} / ${n.target} issues</div>
      ${bar(pct)}
      <div>${pct}% enriched · ${n.text||0} text-extracted · ${n.ocr_fallbacks||0} OCR fallbacks · ${n.embedded||0} embedded</div>
      <div>${n.batch?('Batch API ('+esc(n.batch.state)+'): '+n.batch.docs+' docs · '+esc(n.batch.id)):(s.ncs_rate?(s.ncs_rate.per_hr+' issues/hr · eta ~'+s.ncs_rate.eta_h+' h'):'<span class="dim">extracting text…</span>')}</div>
      <div class="${quiet?'warn':'dim'}">${n.latest?((quiet?'quiet '+Math.round(n.latest.age_s/60)+' min — ':'')+'latest: '+esc(n.latest.doc)):'not started'}</div></div>`;
  }

  const i = s.ingest||{};
  html += `<div class="card"><h2>Document ingest</h2>
    <div class="big">${(i.total??0).toLocaleString()} docs</div>
    <div><span class="ok">+${i.last24h??0}</span> last 24 h · +${i.last1h??0} last hour
      ${s.ingest_rate!=null?('· '+s.ingest_rate+' docs/hr'):''}</div>
    <div class="${i.failed?'warn':'dim'}">${i.failed||0} failed</div>
    <div class="dim">${i.latest?('latest: '+esc(i.latest.doc)+' @ '+i.latest.at):'no ingests recorded'}</div></div>`;

  const srcs = Object.entries(i.per_source||{}).map(([k,v])=>
    `<tr><td>${esc(k)}</td><td>${v.toLocaleString()}</td></tr>`).join('');
  html += `<div class="card"><h2>Ingested by source</h2>
    <table>${srcs||'<tr><td class="dim">none</td></tr>'}</table></div>`;

  const c = s.corpus||{};
  html += `<div class="card"><h2>Published corpus (serving)</h2>
    <div class="big">${c.chunks!=null?c.chunks.toLocaleString():'—'} chunks</div>
    <div>${c.docs!=null?c.docs.toLocaleString()+' documents':''} · ${c.events!=null?c.events.toLocaleString()+' events':''}</div>
    <div class="dim">${c.release?('release '+esc(c.release)):(c.error?esc(c.error):'')}</div></div>`;

  const procs = Object.entries(s.procs||{}).map(([k,v])=>
    `<tr><td>${esc(k)}</td><td class="${v?'ok':'dim'}">${v?'running':'idle'}</td></tr>`).join('');
  html += `<div class="card"><h2>Pipeline processes</h2><table>${procs}</table></div>`;

  document.getElementById('cards').innerHTML = html;
  document.getElementById('foot').textContent =
    'updated ' + new Date(s.updated*1000).toLocaleTimeString() +
    ' · ledger+logs every '+s.poll_s+'s · postgres every '+s.pg_s+'s · page refreshes every 15s';
}
tick(); setInterval(tick, 15000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/status":
            body = json.dumps({
                "updated": STATE["updated"],
                "download": STATE.get("download"),
                "ingest": STATE.get("ingest", {}),
                "ingest_rate": ingest_rate(),
                "ncs": STATE.get("ncs"),
                "ncs_rate": ncs_rate(),
                "corpus": STATE.get("corpus", {}),
                "procs": STATE.get("procs", {}),
                "poll_s": POLL_S, "pg_s": PG_S,
            }).encode()
            ct = "application/json"
        else:
            body = PAGE.encode()
            ct = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    threading.Thread(target=poller, daemon=True).start()
    threading.Thread(target=pg_poller, daemon=True).start()
    print(f"corpus dashboard on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
