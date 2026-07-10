#!/usr/bin/env python3
"""
Live dashboard for the cloud OCR run: upload, download drain, pod workers,
throughput and ETA. Stdlib only — no new dependencies.

    .venv/bin/python ocr_dashboard.py   # then open http://<host>:8877

Data sources:
  - data/upload.log            rsync --info=progress2 output (local upload)
  - data/download_drain.log    manifest drain (local downloads)
  - data/upload_list.txt       full corpus file list (denominator)
  - data/text/*.txt            page counts per stem (for pages-based ETA)
  - ssh to the pod             done count, worker logs, GPU stats
"""
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT        = Path(__file__).parent
POD_HOST    = "root@209.170.80.132"
PODS        = {"1": "13976", "2": "10045", "3": "10106", "4": "13925", "5": "14092", "6": "10960"}   # name -> ssh port (shared volume)


def ssh_cmd(port):
    return ["ssh", "-p", port, "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ControlMaster=auto", "-o", f"ControlPath=/tmp/ssh-ocrdash-{port}",
            "-o", "ControlPersist=600", POD_HOST]


SSH = ssh_cmd(PODS["1"])  # primary pod: volume-level stats
POLL_S      = 45      # heavy poll: done counts, GPU, uploaded files
LOGPOLL_S   = 12      # light poll: worker log tails
PORT        = 8877

STATE = {"updated": None, "pod": {}, "workers": {}, "history": deque(maxlen=240)}
PAGES = {}          # stem -> page count (from local text cache)
TOTAL_DOCS = 0
TOTAL_PAGES = 0


def _read_tail(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def local_status():
    s = {}
    # upload: last rsync progress2 line, e.g. " 10,637,240,140  27%  1.48MB/s  5:03:59"
    tail = _read_tail(ROOT / "data/upload.log").replace("\r", "\n")
    m = None
    for m2 in re.finditer(r"([\d,]{7,})\s+(\d+)%\s+([\d.]+\w?B/s)\s+([\d:]+)", tail):
        m = m2
    if m:
        s["upload"] = {"bytes": int(m.group(1).replace(",", "")), "pct": int(m.group(2)),
                       "rate": m.group(3), "eta": m.group(4)}
    if "total size is" in tail or "speedup is" in tail:
        s["upload"] = {**s.get("upload", {}), "done": True, "pct": 100}
    # drain: last tqdm line "Downloading:  57%|...| 2963/5172 [...]"
    dtail = _read_tail(ROOT / "data/download_drain.log").replace("\r", "\n")
    dm = None
    for dm2 in re.finditer(r"Downloading:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)", dtail):
        dm = dm2
    if dm:
        s["drain"] = {"pct": int(dm.group(1)), "n": int(dm.group(2)), "total": int(dm.group(3))}
    if "DRAIN: complete" in _read_tail(ROOT / "data/tranche_loop.out"):
        s["drain"] = {**s.get("drain", {}), "done": True}
    # local processes
    def alive(pat):
        return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
    s["procs"] = {"rsync upload": alive(r"rsync.*upload_list"),
                  "download drain": alive(r"pipeline\.py download")}
    return s


def pod_status():
    # one ssh round-trip gathering everything, field-separated by @@
    cmd = (
        "ls /workspace/out 2>/dev/null; echo @@; "
        "ps -eo args | grep -c '^python3 cloud_ocr.py' || true; echo @@; "
        "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu "
        "--format=csv,noheader 2>/dev/null; echo @@; "
        "find /workspace/corpus -type f 2>/dev/null | wc -l; echo @@; "
        "for f in /workspace/worker_*.log; do echo \"$f: $(grep '^OK' $f | tail -1)\"; done; echo @@; "
        "grep -hc '^FAIL' /workspace/worker_*.log | awk '{s+=$1} END {print s}'; echo @@; "
        "for f in /workspace/worker_*.log; do p=$(grep '^PAGE' $f | tail -1); "
        "echo \"${f##*worker_}: ${p:-rendering pages...}\"; done; echo @@; "
        "cat /tmp/vol_used_gb 2>/dev/null; echo @@; "
        # ground truth: every PAGE log line = one 4-page OCR batch, counted
        # across current AND rotated logs so restarts don't reset the meter
        "grep -hc '^PAGE' /workspace/worker_*.log* 2>/dev/null | awk '{s+=$1} END {print s*4}'"
    )
    try:
        out = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=30).stdout
        parts = [p.strip() for p in out.split("@@")]
        stems = [l[:-4] for l in parts[0].splitlines() if l.endswith(".txt")]
        # per-pod worker + GPU stats
        per_pod = {}
        for name, port in PODS.items():
            try:
                o = subprocess.run(
                    ssh_cmd(port) + ["ps -eo args | grep -c '^venv/bin/python [c]loud_ocr' || true; "
                                     "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader"],
                    capture_output=True, text=True, timeout=20).stdout.splitlines()
                per_pod[name] = {"workers": int(o[0] or 0), "gpu": o[1] if len(o) > 1 else "?"}
            except Exception:
                per_pod[name] = {"workers": None, "gpu": "unreachable"}
        pod = {"reachable": True,
               "done": len(stems),
               "per_pod": per_pod,
               "done_pages": sum(PAGES.get(s, 1) for s in stems),
               "workers": sum(p["workers"] or 0 for p in per_pod.values()),
               "gpu": " · ".join(f"pod{n}: {p['gpu']}" for n, p in sorted(per_pod.items())),
               "uploaded_files": int(parts[3] or 0),
               "last_per_worker": [l for l in parts[4].splitlines() if l.strip()],
               "failed": int(parts[5] or 0),
               "in_flight": [l for l in parts[6].splitlines() if l.strip()] if len(parts) > 6 else [],
               "disk_pct": round(100 * int(parts[7]) / 90) if len(parts) > 7 and parts[7].strip().isdigit() else None,
               "pages_ocrd": int(parts[8]) if len(parts) > 8 and parts[8].strip().isdigit() else None}
    except Exception as e:
        pod = {"reachable": False, "error": str(e)[:120]}
    return pod


def log_poller():
    """Light, frequent poll: per-worker log tails + ok/fail counts."""
    cmd = (
        "for f in /workspace/worker_*.log; do "
        "echo \"===$f===$(grep -c '^OK' $f)===$(grep -c '^FAIL' $f)\"; "
        "tail -c 1600 $f; done"
    )
    while True:
        try:
            out = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=25).stdout
            workers = {}
            cur = None
            for line in out.splitlines():
                if line.startswith("==="):
                    _, path, ok, fail = line.split("===")
                    cur = re.search(r"worker_(\w+)\.log", path).group(1)
                    workers[cur] = {"ok": int(ok or 0), "fail": int(fail or 0), "tail": []}
                elif cur is not None and "unauthenticated requests" not in line and "Kwargs passed" not in line:
                    workers[cur]["tail"].append(line[-160:])
            for w in workers.values():
                w["tail"] = w["tail"][-14:]
            STATE["workers"] = workers
            # live in-flight page positions (PAGE n/m lines) so the page
            # counter moves continuously, not only at document completion
            inflight = 0
            for w in workers.values():
                for line in reversed(w["tail"]):
                    m = re.match(r"PAGE (\d+)/(\d+)", line)
                    if m:
                        inflight += int(m.group(1))
                        break
            STATE["inflight_pages"] = inflight
        except Exception:
            pass
        time.sleep(LOGPOLL_S)


_H = {"done": None, "fails": 0, "workers": None, "stall_since": None,
      "unreach": 0, "milestone": 0, "balance": None, "balance_t": 0}
ALERTS = deque(maxlen=60)


def _alert(kind, msg):
    ALERTS.append({"t": time.time(), "kind": kind, "msg": msg})
    with open(ROOT / "data/ocr_alerts.log", "a") as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')} [{kind}] {msg}\n")


def _balance():
    try:
        key = open(os.path.expanduser("~/.runpod_key")).read().split("=", 1)[1].strip()
        r = subprocess.run(
            ["curl", "-s", "-H", f"Authorization: Bearer {key}",
             "-H", "Content-Type: application/json",
             "-d", '{"query":"query { myself { clientBalance } }"}',
             "https://api.runpod.io/graphql"],
            capture_output=True, text=True, timeout=20)
        m = re.search(r'"clientBalance":([\d.]+)', r.stdout)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def evaluate_health(pod):
    """Same checks as the session health monitor, rendered for the dashboard."""
    checks = {}
    now = time.time()

    if not pod.get("reachable"):
        _H["unreach"] += 1
        if _H["unreach"] == 2:
            _alert("alert", "pod unreachable (2 consecutive polls)")
        checks["pod reachable"] = {"status": "alert", "detail": f"unreachable ×{_H['unreach']}"}
        STATE["health"] = {"checks": checks, "alerts": list(ALERTS)}
        return
    if _H["unreach"] >= 2:
        _alert("info", "pod reachable again")
    _H["unreach"] = 0
    checks["pod reachable"] = {"status": "ok", "detail": "ssh ok"}

    expect = len(PODS)
    w = pod.get("workers", 0)
    if _H["workers"] is not None and w != _H["workers"]:
        if w < expect:
            _alert("alert", f"worker count dropped to {w} (expect {expect})")
        elif _H["workers"] < expect:
            _alert("info", f"recovered: {w} workers running")
    _H["workers"] = w
    checks[f"workers (expect {expect})"] = {"status": "ok" if w >= expect else "alert",
                                            "detail": f"{w} running across {expect} pods"}

    fails = pod.get("failed", 0)
    if fails - _H["fails"] >= 5:
        _alert("alert", f"failures jumped {_H['fails']} → {fails}")
    _H["fails"] = fails
    checks["failure rate"] = {"status": "ok" if fails < 20 else "warn", "detail": f"{fails} total FAILs"}

    # stall: gpu idle AND done frozen for >= 9 minutes (max util across pods)
    utils = []
    for p in (pod.get("per_pod") or {}).values():
        m = re.match(r"\s*(\d+)\s*%", p.get("gpu") or "")
        if m:
            utils.append(int(m.group(1)))
    gpu_pct = max(utils) if utils else 0
    done = pod.get("done", 0)
    stalled = w >= 1 and done == _H["done"] and gpu_pct < 10
    if stalled:
        _H["stall_since"] = _H["stall_since"] or now
        dur = now - _H["stall_since"]
        if 540 <= dur < 540 + POLL_S:
            _alert("alert", f"possible stall — GPU idle {dur/60:.0f}min, done stuck at {done}")
        checks["progress"] = {"status": "warn" if dur < 540 else "alert",
                              "detail": f"gpu {gpu_pct}%, no completions for {dur/60:.0f}min"}
    else:
        if _H["stall_since"] and now - _H["stall_since"] >= 540:
            _alert("info", f"progress resumed (done={done}, gpu={gpu_pct}%)")
        _H["stall_since"] = None
        checks["progress"] = {"status": "ok", "detail": f"gpu {gpu_pct}%, done={done}"}

    if _H["done"] is not None:
        if _H["done"] == 0 and done > 0:
            _alert("info", f"milestone: first documents completed (done={done})")
        if done // 1000 > _H["milestone"]:
            _H["milestone"] = done // 1000
            _alert("info", f"milestone: {done} documents done")
    _H["done"] = done

    d = pod.get("disk_pct")
    if d is not None:
        if d >= 90:
            _alert("alert", f"network volume {d}% full")
        checks["volume disk"] = {"status": "ok" if d < 90 else "alert", "detail": f"{d}% used"}

    # balance every ~30 min
    if now - _H["balance_t"] > 1800:
        _H["balance_t"] = now
        b = _balance()
        if b is not None:
            _H["balance"] = b
            if b < 8:
                _alert("alert", f"RunPod balance low: ${b:.2f}")
    if _H["balance"] is not None:
        checks["runpod credits"] = {"status": "ok" if _H["balance"] >= 8 else "alert",
                                    "detail": f"${_H['balance']:.2f}"}

    STATE["health"] = {"checks": checks, "alerts": list(ALERTS)}


def poller():
    while True:
        pod = pod_status()
        STATE["pod"] = pod
        STATE["local"] = local_status()
        STATE["updated"] = time.time()
        if pod.get("reachable"):
            # prefer the PAGE-line meter (exact, includes in-flight, survives
            # the incomplete local page map); fall back to the old estimate
            live_pages = pod.get("pages_ocrd") or (
                pod.get("done_pages", 0) + STATE.get("inflight_pages", 0))
            STATE["history"].append((time.time(), pod["done"], live_pages))
        evaluate_health(pod)
        time.sleep(POLL_S)


def compute_rates():
    h = list(STATE["history"])
    if len(h) < 2:
        return {}
    # rate over up to the last hour of samples; pages drive the ETA
    # (documents vary 1-1100 pages, so doc-rate ETA is meaningless)
    cutoff = time.time() - 3600
    win = [x for x in h if x[0] >= cutoff] or h
    if len(win) < 2 or win[-1][0] <= win[0][0]:
        return {}
    # sum positive deltas between consecutive samples: robust against the
    # page meter resetting (worker log truncation on fleet restarts)
    pages_gain = sum(max(0, b[2] - a[2]) for a, b in zip(win, win[1:]))
    docs_gain = sum(max(0, b[1] - a[1]) for a, b in zip(win, win[1:]))
    hrs = (win[-1][0] - win[0][0]) / 3600
    if pages_gain == 0:
        return {}
    pages_hr = pages_gain / hrs
    docs_hr = docs_gain / hrs
    # remaining from the per-doc completed counter (survives log truncation)
    done_pages = (STATE.get("pod") or {}).get("done_pages", 0)
    remaining_pages = max(0, TOTAL_PAGES - done_pages)
    eta_h = remaining_pages / pages_hr if pages_hr else None
    return {"pages_per_hour": round(pages_hr),
            "docs_per_hour": round(docs_hr, 1),
            "eta_hours": round(eta_h, 1) if eta_h is not None else None,
            "remaining_pages": remaining_pages}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>UAP OCR Dashboard</title>
<style>
 body{background:#101418;color:#dde3ea;font-family:ui-monospace,Menlo,monospace;margin:2rem}
 h1{font-size:1.2rem;color:#7ec8f8} .cards{display:flex;flex-wrap:wrap;gap:1rem}
 .card{background:#1a2027;border:1px solid #2a323c;border-radius:8px;padding:1rem 1.2rem;min-width:280px;flex:1}
 .card h2{font-size:.85rem;color:#8fa3b8;margin:0 0 .6rem;text-transform:uppercase;letter-spacing:.08em}
 .big{font-size:1.6rem;color:#fff} .bar{background:#2a323c;border-radius:4px;height:10px;margin:.5rem 0}
 .fill{background:#4aa3df;height:10px;border-radius:4px} .ok{color:#5fd38d}.bad{color:#e46a6a}.warn{color:#e5b567}
 .dim{color:#67788c;font-size:.8rem} table{width:100%;font-size:.78rem;border-collapse:collapse}
 td{padding:.15rem .3rem;border-bottom:1px solid #232b34;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:420px}
 .logtail{background:#12161b;border:1px solid #232b34;border-radius:6px;padding:.6rem;font-size:.72rem;
   line-height:1.35;white-space:pre-wrap;word-break:break-all;margin:0;max-height:16em;overflow-y:auto;color:#a8b8c8}
</style></head><body>
<h1>UAP corpus — cloud OCR dashboard</h1>
<div class="cards" id="cards">loading…</div>
<p class="dim" id="foot"></p>
<script>
function bar(p){return `<div class="bar"><div class="fill" style="width:${p}%"></div></div>`}
async function tick(){
  const r = await fetch('/api/status'); const s = await r.json();
  const pod = s.pod||{}, loc = s.local||{}, up = loc.upload||{}, dr = loc.drain||{};
  const pct = s.total_docs ? (100*pod.done/s.total_docs).toFixed(1) : 0;
  let html = '';
  html += `<div class="card"><h2>OCR progress</h2><div class="big">${pod.done??'?'} / ${s.total_docs}</div>
    ${bar(pct)}<div>${pct}% of docs · ~${s.done_pages??'?'} / ${s.total_pages} pages</div>
    <div class="dim">${pod.failed||0} failed · output on pod volume</div></div>`;
  html += `<div class="card"><h2>Throughput / ETA</h2>
    <div class="big">${s.rates?.pages_per_hour!=null?s.rates.pages_per_hour.toLocaleString():'—'} pages/hr</div>
    <div>${s.rates?.docs_per_hour??'—'} docs/hr · ${s.rates?.remaining_pages!=null?s.rates.remaining_pages.toLocaleString()+' pages left':''}</div>
    <div>ETA: ${s.rates?.eta_hours!=null?s.rates.eta_hours+' h':'measuring…'}</div>
    <div class="dim">page-based, rate window: last hour (includes not-yet-uploaded pages in the denominator)</div></div>`;
  html += `<div class="card"><h2>Upload (local → pod)</h2>
    <div class="big">${up.done?'complete':(up.pct??'?')+'%'}</div>${bar(up.pct||0)}
    <div>${up.rate||''} ${up.eta?('· eta '+up.eta):''}</div>
    <div class="dim">${(pod.uploaded_files??'?')} files landed on pod</div></div>`;
  html += `<div class="card"><h2>Download drain (manifest → local)</h2>
    <div class="big">${dr.done?'complete':(dr.pct!=null?dr.pct+'%':'?')}</div>${bar(dr.pct||0)}
    <div>${dr.n!=null?dr.n+' / '+dr.total+' resources':''}</div></div>`;
  const procs = Object.entries(loc.procs||{}).map(([k,v])=>
    `<tr><td>${k}</td><td class="${v?'ok':'bad'}">${v?'running':'stopped'}</td></tr>`).join('');
  const gpu = pod.gpu||'';
  const podRows = Object.entries(pod.per_pod||{}).sort().map(([n,p])=>
    `<tr><td>pod ${n} GPU (util, mem, temp)</td><td class="${p.workers?'ok':'bad'}">${p.gpu}</td>
     <td class="dim">${p.workers==null?'unreachable':p.workers+' worker'}</td></tr>`).join('');
  html += `<div class="card"><h2>Processes</h2><table>
    ${procs}
    ${podRows}
    <tr><td>total workers</td><td class="${pod.workers>=3?'ok':'bad'}">${pod.workers??0} running</td></tr>
    <tr><td>pod reachable</td><td class="${pod.reachable?'ok':'bad'}">${pod.reachable}</td></tr></table></div>`;
  const hc = Object.entries((s.health||{}).checks||{}).map(([k,v])=>
    `<tr><td>${k}</td><td class="${v.status==='ok'?'ok':(v.status==='warn'?'warn':'bad')}">${v.status}</td><td class="dim">${v.detail}</td></tr>`).join('');
  html += `<div class="card"><h2>Health monitor</h2><table>${hc||'<tr><td>waiting for first poll…</td></tr>'}</table></div>`;
  const al = ((s.health||{}).alerts||[]).slice(-8).reverse().map(a=>
    `<tr><td class="dim">${new Date(a.t*1000).toLocaleTimeString()}</td>
     <td class="${a.kind==='alert'?'bad':'ok'}">${a.kind}</td><td>${a.msg}</td></tr>`).join('');
  html += `<div class="card"><h2>Alert history</h2><table>${al||'<tr><td>no alerts — healthy run</td></tr>'}</table></div>`;
  for (const [name, w] of Object.entries(s.workers||{}).sort()) {
    const tail = (w.tail||[]).map(l=>l.replace(/&/g,'&amp;').replace(/</g,'&lt;')).join('\\n');
    html += `<div class="card" style="flex-basis:100%"><h2>worker ${name} —
      <span class="ok">${w.ok} done</span> · <span class="${w.fail?'bad':'dim'}">${w.fail} failed</span></h2>
      <pre class="logtail">${tail||'(no output yet)'}</pre></div>`;
  }
  document.getElementById('cards').innerHTML = html;
  document.getElementById('foot').textContent =
    'updated ' + new Date(s.updated*1000).toLocaleTimeString() + ' · polls pod every '+s.poll_s+'s · auto-refreshes every 10s';
}
tick(); setInterval(tick, 10000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/api/status":
            pod = STATE.get("pod", {})
            body = json.dumps({
                "updated": STATE["updated"], "pod": pod, "local": STATE.get("local", {}),
                "workers": STATE.get("workers", {}),
                "health": STATE.get("health", {}),
                "total_docs": TOTAL_DOCS, "total_pages": TOTAL_PAGES,
                "done_pages": pod.get("done_pages"),
                "rates": compute_rates(), "poll_s": POLL_S,
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
    global TOTAL_DOCS, TOTAL_PAGES
    for t in (ROOT / "data/text").glob("*.txt"):
        try:
            PAGES[t.stem] = max(1, open(t, encoding="utf-8", errors="ignore").read().count("--- Page "))
        except OSError:
            pass
    full = [l.strip() for l in open(ROOT / "data/ocr_full_list.txt") if l.strip()]
    sample = [l.strip() for l in open(ROOT / "data/ocr_sample_list.txt") if l.strip()]
    TOTAL_DOCS = len(full) + len(sample)
    # full band: known page counts (delta docs unknown -> assume corpus median ~12);
    # sample band: ~15 pages per doc regardless of size
    TOTAL_PAGES = sum(PAGES.get(Path(f).stem, 12) for f in full) + 15 * len(sample)
    threading.Thread(target=poller, daemon=True).start()
    threading.Thread(target=log_poller, daemon=True).start()
    print(f"dashboard on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
