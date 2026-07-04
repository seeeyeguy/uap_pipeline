"""
Bulk Downloader
---------------
Consumes sources.json and pulls every resource to disk:

  data/downloads/<source_id>/<filename>      regular files
  data/downloads/<source_id>/<repo_name>/    git clones (kind=git)

Features:
  - resume via HTTP Range for partial files
  - retries with exponential backoff
  - SHA-256 verification when the manifest carries a checksum
  - ledger (data/downloads.json) so completed files are never re-fetched
  - filters: by source, kind, max file size; videos excluded by default
    (the OCR pipeline is document-oriented)
"""

import hashlib
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests
from tqdm import tqdm

log = logging.getLogger(__name__)

LEDGER_FILE = "./data/downloads.json"
DOWNLOAD_DIR = "./data/downloads"
CHUNK = 1024 * 256
DEFAULT_EXCLUDED_KINDS = {"video", "html", "torrent"}


def load_ledger() -> dict:
    p = Path(LEDGER_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"completed": {}, "failed": {}}


def save_ledger(ledger: dict):
    p = Path(LEDGER_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_filename(url: str) -> str:
    name = unquote(urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]) or "index"
    name = re.sub(r"[^\w.\- ]+", "_", name)[:180]
    # FBI Vault URLs end in /at_download/file — synthesize a name
    if name in ("file", "index"):
        parts = [unquote(p) for p in urlparse(url).path.split("/") if p]
        name = re.sub(r"[^\w.\- ]+", "_", "_".join(parts[-3:]))[:180] or "download"
    return name


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _download_file(session: requests.Session, url: str, dest: Path,
                   expected_sha256: str = None, retries: int = 4) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            headers = {}
            mode = "wb"
            done = 0
            if tmp.exists():
                done = tmp.stat().st_size
                headers["Range"] = f"bytes={done}-"
                mode = "ab"

            with session.get(url, stream=True, timeout=120, headers=headers) as r:
                if r.status_code == 416:      # range past EOF — file complete
                    pass
                elif r.status_code in (200, 206):
                    if r.status_code == 200 and mode == "ab":
                        mode, done = "wb", 0  # server ignored Range
                    total = int(r.headers.get("content-length", 0)) + done
                    with open(tmp, mode) as f, tqdm(
                            total=total or None, initial=done, unit="B",
                            unit_scale=True, desc=dest.name[:40], leave=False) as bar:
                        for chunk in r.iter_content(chunk_size=CHUNK):
                            f.write(chunk)
                            bar.update(len(chunk))
                else:
                    raise requests.HTTPError(f"HTTP {r.status_code}")

            digest = sha256_file(tmp)
            if expected_sha256 and digest.lower() != expected_sha256.lower():
                tmp.unlink(missing_ok=True)
                raise IOError(f"SHA-256 mismatch (got {digest[:12]}…)")
            tmp.rename(dest)
            return {"ok": True, "sha256": digest, "bytes": dest.stat().st_size}

        except (requests.RequestException, IOError, OSError) as e:
            if attempt == retries:
                return {"ok": False, "error": str(e)}
            wait = 2 ** attempt
            log.warning(f"{dest.name}: attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    return {"ok": False, "error": "exhausted retries"}


def _clone_repo(url: str, dest_dir: Path) -> dict:
    name = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    dest = dest_dir / name
    try:
        if dest.exists():
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"],
                           check=True, capture_output=True, text=True, timeout=1800)
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", url, str(dest)],
                           check=True, capture_output=True, text=True, timeout=3600)
        return {"ok": True, "path": str(dest)}
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err = getattr(e, "stderr", "") or str(e)
        return {"ok": False, "error": err.strip()[:500]}


def _size_hint_bytes(size_hint) -> int:
    """Parse '2.0 GB' / '342MB' / '123456' into bytes; 0 if unknown."""
    if not size_hint:
        return 0
    s = str(size_hint).strip().upper().replace(" ", "")
    m = re.match(r"^([\d.]+)(GB|MB|KB|B)?$", s)
    if not m:
        return 0
    mult = {"GB": 1 << 30, "MB": 1 << 20, "KB": 1 << 10, "B": 1, None: 1}[m.group(2)]
    return int(float(m.group(1)) * mult)


def download_all(manifest: dict, only_sources: list[str] = None,
                 include_kinds: set = None, include_videos: bool = False,
                 max_size_gb: float = None, force: bool = False) -> dict:
    """Download every manifest resource that passes the filters."""
    ledger = load_ledger()
    session = requests.Session()
    session.headers.update({"User-Agent": "uap-pipeline/1.0 (archival research)"})

    excluded = set(DEFAULT_EXCLUDED_KINDS)
    if include_videos:
        excluded.discard("video")

    todo = []
    for r in manifest["resources"]:
        if only_sources and r["source"] not in only_sources:
            continue
        if include_kinds and r["kind"] not in include_kinds:
            continue
        if not include_kinds and r["kind"] in excluded:
            continue
        if max_size_gb and _size_hint_bytes(r.get("size_hint")) > max_size_gb * (1 << 30):
            log.info(f"Skipping (over size cap): {r['title']}")
            continue
        if not force and r["url"] in ledger["completed"]:
            continue
        todo.append(r)

    log.info(f"{len(todo)} resources to download "
             f"({len(ledger['completed'])} already complete).")

    stats = {"ok": 0, "failed": 0}
    for r in tqdm(todo, desc="Downloading", unit="file"):
        source_dir = Path(DOWNLOAD_DIR) / r["source"]
        if r["kind"] == "git":
            result = _clone_repo(r["url"], source_dir)
        else:
            dest = source_dir / _safe_filename(r["url"])
            result = _download_file(session, r["url"], dest,
                                    expected_sha256=r.get("sha256"))
            if result.get("ok"):
                result["path"] = str(dest)

        if result.get("ok"):
            stats["ok"] += 1
            ledger["completed"][r["url"]] = {
                "path": result.get("path"),
                "sha256": result.get("sha256"),
                "bytes": result.get("bytes"),
                "source": r["source"],
                "kind": r["kind"],
            }
            ledger["failed"].pop(r["url"], None)
        else:
            stats["failed"] += 1
            log.error(f"FAILED {r['url']}: {result.get('error')}")
            ledger["failed"][r["url"]] = {"error": result.get("error"),
                                          "source": r["source"]}
        save_ledger(ledger)

    log.info(f"Download pass done: {stats['ok']} ok, {stats['failed']} failed. "
             f"Ledger: {LEDGER_FILE}")
    return stats
