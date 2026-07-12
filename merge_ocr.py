#!/usr/bin/env python3
"""
Merge fresh cloud OCR (data/text_new) with the existing text layer
(data/text), deciding each document's final text.

Policy per stem:
  - only in text_new (never had text)          -> take new
  - only in text (not re-OCR'd)                -> keep old
  - both present:
      * empty new (corrupt-source marker)      -> keep old, flag corrupt
      * shingle-Jaccard agreement >= AGREE     -> keep old (text layer
        validated; its enrichment cache stays valid) [decision: "agree"]
      * disagreement -> whichever scores higher on lexical quality wins;
        flag for review                        [decision: "new"/"old"]
      * both poor (< MINQ)                      -> take higher, flag human

Writes final text to data/text_final/, a decision report to
data/merge_report.json, and a changed-stems list (whose enrichment cache
must be invalidated) to data/merge_changed.txt.
"""
import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from main import text_quality_score

ROOT = Path(__file__).parent
OLD = ROOT / "data/text"
NEW = ROOT / "data/text_new"
FINAL = ROOT / "data/text_final"
AGREE = 0.55      # shingle-Jaccard: substantial agreement keeps the layer
MINQ = 0.30       # below this, text is still junk


def shingles(text, k=5):
    words = re.sub(r"--- Page \d+ ---", " ", text).lower().split()
    return set(" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1)))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    FINAL.mkdir(exist_ok=True)
    old = {p.stem: p for p in OLD.glob("*.txt")}
    new = {p.stem: p for p in NEW.glob("*.txt")}
    report = {"take_new": 0, "keep_old": 0, "agree_keep": 0,
              "new_wins": 0, "old_wins": 0, "both_poor": 0,
              "corrupt": 0, "flagged": []}
    changed = []

    for stem in sorted(set(old) | set(new)):
        op = old.get(stem)
        np = new.get(stem)
        otext = op.read_text(encoding="utf-8", errors="ignore") if op else ""
        ntext = np.read_text(encoding="utf-8", errors="ignore") if np else ""

        if np and not op:
            final, decision = ntext, "take_new"; changed.append(stem)
        elif op and not np:
            final, decision = otext, "keep_old"
        elif not ntext.strip():                 # empty new = corrupt source
            final, decision = otext, "corrupt"
            report["corrupt"] += 1
            report["flagged"].append({"stem": stem, "why": "corrupt_source"})
        else:
            sim = jaccard(shingles(otext), shingles(ntext))
            oq, nq = text_quality_score(otext), text_quality_score(ntext)
            if sim >= AGREE:
                final, decision = otext, "agree_keep"   # layer validated
            elif nq >= oq:
                final, decision = ntext, "new_wins"; changed.append(stem)
                if nq < MINQ and oq < MINQ:
                    decision = "both_poor"
                    report["flagged"].append({"stem": stem, "why": "both_poor",
                                              "new_q": round(nq, 2), "old_q": round(oq, 2)})
            else:
                final, decision = otext, "old_wins"
                report["flagged"].append({"stem": stem, "why": "old_beat_new",
                                          "new_q": round(nq, 2), "old_q": round(oq, 2), "sim": round(sim, 2)})
        key = {"take_new": "take_new", "keep_old": "keep_old", "corrupt": "corrupt",
               "agree_keep": "agree_keep", "new_wins": "new_wins",
               "old_wins": "old_wins", "both_poor": "both_poor"}[decision]
        report[key] = report.get(key, 0) + 1
        (FINAL / f"{stem}.txt").write_text(final, encoding="utf-8")

    (ROOT / "data/merge_report.json").write_text(json.dumps(report, indent=1))
    (ROOT / "data/merge_changed.txt").write_text("\n".join(changed) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "flagged"}, indent=1))
    print(f"final texts: {len(list(FINAL.glob('*.txt')))} | changed (re-enrich): {len(changed)} | flagged: {len(report['flagged'])}")


if __name__ == "__main__":
    main()
