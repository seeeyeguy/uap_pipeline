"""
AFU tranche OCR sizing: sample PDFs from data/downloads/afu_se, classify each
via the same rules as main.extract_pdf_text_layer (chars/page floor + lexical
gate), count pages, and extrapolate the page-weighted OCR workload.

Classification is approximated on the first SAMPLE_PAGES pages per doc (the
lexical score is stable well before that); page counts are exact.

Usage: .venv/bin/python sample_afu_ocr.py [N_SAMPLE]
"""
import json
import random
import re
import sys
from pathlib import Path

from pypdf import PdfReader

# Mirrors main.py's gate (not imported: main.py loads torch/transformers at
# module level). Keep in sync with TEXT_LAYER_MIN_* / text_quality_score.
TEXT_LAYER_MIN_CHARS_PER_PAGE = 200
TEXT_LAYER_MIN_QUALITY = 0.30


def text_quality_score(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 20:
        return 1.0
    plausible = sum(
        1 for t in tokens
        if re.fullmatch(r"[A-Za-z]{3,}[.,;:!?)]?|\(?\d{1,4}([-/.:]\d{1,4})*[.,;:]?", t)
    )
    return plausible / len(tokens)

AFU_DIR = Path("data/downloads/afu_se")
SAMPLE_PAGES = 12
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400

pdfs = sorted(p for p in AFU_DIR.rglob("*.pdf") if p.stat().st_size > 0)
random.seed(42)
sample = random.sample(pdfs, min(N, len(pdfs)))
print(f"population: {len(pdfs)} PDFs; sampling {len(sample)}")

classes = {"clean_layer": [], "garbage_layer": [], "image_only": [], "error": []}
for i, path in enumerate(sample, 1):
    try:
        reader = PdfReader(str(path))
        n_pages = len(reader.pages)
        probe = [(pg.extract_text() or "").strip()
                 for pg in reader.pages[:SAMPLE_PAGES]]
        avg_chars = sum(len(t) for t in probe) / max(1, len(probe))
        if avg_chars < TEXT_LAYER_MIN_CHARS_PER_PAGE:
            cls = "image_only"
        elif text_quality_score(" ".join(probe)) < TEXT_LAYER_MIN_QUALITY:
            cls = "garbage_layer"
        else:
            cls = "clean_layer"
    except Exception:
        cls, n_pages = "error", 0
    classes[cls].append(n_pages)
    if i % 50 == 0:
        print(f"  {i}/{len(sample)}…", flush=True)

total_sampled_pages = sum(sum(v) for v in classes.values())
print(f"\n{'class':14s} {'docs':>5s} {'share':>6s} {'pages':>7s} {'pg-share':>8s} {'avg pg':>6s}")
for k, v in classes.items():
    if not v:
        continue
    print(f"{k:14s} {len(v):5d} {len(v)/len(sample):6.1%} {sum(v):7d} "
          f"{sum(v)/max(1,total_sampled_pages):8.1%} {sum(v)/len(v):6.1f}")

ocr_pages_sampled = sum(classes["garbage_layer"]) + sum(classes["image_only"])
scale = len(pdfs) / len(sample)
est = {
    "population_pdfs": len(pdfs),
    "sampled": len(sample),
    "est_total_pages": int(total_sampled_pages * scale),
    "est_ocr_docs": int((len(classes["garbage_layer"]) + len(classes["image_only"])) * scale),
    "est_ocr_pages": int(ocr_pages_sampled * scale),
}
print(f"\nextrapolated: {est['est_total_pages']:,} total pages; "
      f"OCR needed: {est['est_ocr_docs']:,} docs / {est['est_ocr_pages']:,} pages")
print(f"cost @ $1.05/kpage: ${est['est_ocr_pages'] / 1000 * 1.05:,.0f}")
json.dump(est, open("data/afu_ocr_estimate.json", "w"), indent=1)
print("wrote data/afu_ocr_estimate.json")
