#!/usr/bin/env python3
"""
Merge the prod-server wikipedia drain results into the local ledger.

Reads the prod ledger snapshot + staged files (data/prod_drain_sync/, kept
fresh by rsync), and for every prod entry absent locally: sha-dedupes,
copies the file into the real data/downloads/ tree, and adds the ledger
entry. Idempotent — re-run after each rsync refresh.

Usage: python merge_wikipedia_drain.py [--apply]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
LEDGER = ROOT / "data/downloads.json"
STAGE = ROOT / "data/prod_drain_sync"
PROD_LEDGER = STAGE / "downloads.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    local = json.loads(LEDGER.read_text())
    prod = json.loads(PROD_LEDGER.read_text())
    shas = {e.get("sha256") for e in local["completed"].values()}

    new, sha_dup, missing_file = [], 0, 0
    for url, e in prod["completed"].items():
        if url in local["completed"]:
            continue
        if e.get("sha256") in shas:
            sha_dup += 1
            continue
        src = STAGE / "downloads" / Path(e["path"]).relative_to("data/downloads")
        if not src.exists():
            missing_file += 1
            continue
        new.append((url, e, src))

    print(f"prod entries: {len(prod['completed'])} | already local: "
          f"{len(prod['completed']) - len(new) - sha_dup - missing_file} | "
          f"sha-dup: {sha_dup} | staged file missing: {missing_file} | "
          f"to merge: {len(new)}")

    if not args.apply:
        print("dry run — pass --apply to merge")
        return 0

    for url, e, src in new:
        dest = ROOT / e["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        local["completed"][url] = e
        shas.add(e.get("sha256"))
    LEDGER.write_text(json.dumps(local, indent=1))
    print(f"merged {len(new)} entries; ledger now "
          f"{len(local['completed'])} completed URLs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
