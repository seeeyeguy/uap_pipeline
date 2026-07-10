#!/usr/bin/env python3
"""
Standalone GPU OCR worker for RunPod pods.

Reads source files (PDFs/images) mirrored under --corpus, writes one text
file per document to --out using the same page format as the local
pipeline's ocr_pdf(), so the results drop straight into data/text/.

Resumable: documents whose output file already exists are skipped, so any
number of restarts (or spot interruptions) lose at most one document.
Shardable: --shard K/N processes every file whose index % N == K, letting
several worker processes (and several pods) share one file list.

Usage (ONE batched worker per 24GB card — batching beats concurrent
single-stream workers by ~2.5-3x aggregate):
  nohup python3 cloud_ocr.py --files /workspace/files.txt \
    --corpus /workspace/corpus --out /workspace/out \
    --shard 0/1 > /workspace/worker_0.log 2>&1 &
"""
import argparse
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import torch
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

OCR_MODEL_ID   = "zai-org/GLM-OCR"
MAX_NEW_TOKENS = 8192
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_model():
    processor = AutoProcessor.from_pretrained(OCR_MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        OCR_MODEL_ID, dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    return processor, model


BATCH_SIZE = 4   # measured sweet spot on a 24GB 4090 (batch 8 regresses)


def ocr_batch(image_paths, processor, model):
    """OCR several page images in one batched generate() call.

    Batching amortizes the weight reads that dominate single-stream decode:
    measured ~2.5-3x aggregate throughput vs 3 concurrent single-stream
    workers on the same card. Left padding keeps input ends aligned so one
    slice strips every prompt.
    """
    imgs = []
    for path in image_paths:
        with Image.open(path) as im:
            imgs.append(im.convert("RGB"))
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": im},
            {"type": "text",  "text": "Document Parsing:"},
        ],
    }] for im in imgs]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", padding=True,
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    tail = generated_ids[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in processor.batch_decode(tail, skip_special_tokens=True)]


def ocr_image(image_path, processor, model):
    return ocr_batch([image_path], processor, model)[0]


RENDER_CHUNK = 32   # pages rendered per poppler call, prefetched one ahead


def pdf_page_count(pdf_path):
    import subprocess as sp
    info = sp.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True)
    for line in info.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[-1])
    raise RuntimeError("pdfinfo found no page count")


def ocr_pdf(pdf_path, processor, model, tmp_root, rel=""):
    """OCR a PDF with render/OCR pipelining.

    Rendering the whole document up front leaves the GPU idle for the
    entire rasterization (10-25 min on 1000-page books). Instead render
    RENDER_CHUNK pages at a time in a prefetch thread while the GPU works
    on the previous chunk — both stages stay busy.
    """
    from concurrent.futures import ThreadPoolExecutor
    total = pdf_page_count(pdf_path)

    def render(first):
        last = min(first + RENDER_CHUNK - 1, total)
        d = Path(tempfile.mkdtemp(dir=tmp_root))
        paths = convert_from_path(str(pdf_path), dpi=200, first_page=first,
                                  last_page=last, output_folder=str(d),
                                  fmt="jpeg", output_file="page_", paths_only=True)
        return sorted(paths), d

    full_text = []
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(render, 1)
        first = 1
        while first <= total:
            paths, d = fut.result()
            nxt = first + RENDER_CHUNK
            if nxt <= total:
                fut = ex.submit(render, nxt)   # prefetch while GPU works
            try:
                for i in range(0, len(paths), BATCH_SIZE):
                    chunk = paths[i:i + BATCH_SIZE]
                    for j, text in enumerate(ocr_batch(chunk, processor, model)):
                        full_text.append(f"--- Page {first+i+j} ---\n{text}")
                    print(f"PAGE {min(first+i+BATCH_SIZE-1, total)}/{total} {rel}", flush=True)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            first = nxt
    return "\n\n".join(full_text)


def ocr_pdf_sampled(pdf_path, processor, model, tmp_root, n_sample, rel=""):
    """OCR ~n_sample evenly spaced pages — text-layer verification mode.

    Output keeps true page numbers ("--- Page 37 ---") so the local
    comparison can align sampled pages against the embedded text layer.
    Converts only the sampled pages (first_page/last_page), not the doc.
    """
    import subprocess as sp
    info = sp.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True)
    total = 0
    for line in info.stdout.splitlines():
        if line.startswith("Pages:"):
            total = int(line.split()[-1])
    if total == 0:
        raise RuntimeError("pdfinfo found no pages")
    if total <= n_sample:
        idxs = list(range(1, total + 1))
    else:
        step = total / n_sample
        idxs = sorted({int(step * k) + 1 for k in range(n_sample)})
    img_dir = Path(tempfile.mkdtemp(dir=tmp_root))
    try:
        pages = []
        for pageno in idxs:
            got = convert_from_path(str(pdf_path), dpi=200, first_page=pageno,
                                    last_page=pageno, output_folder=str(img_dir),
                                    fmt="jpeg", paths_only=True)
            if got:
                pages.append((pageno, got[0]))
        texts = []
        for i in range(0, len(pages), BATCH_SIZE):
            chunk = pages[i:i + BATCH_SIZE]
            for (pageno, _), text in zip(chunk, ocr_batch([p for _, p in chunk], processor, model)):
                texts.append(f"--- Page {pageno} ---\n{text}")
        print(f"PAGE sampled {len(pages)}/{total} {rel}", flush=True)
        return "\n\n".join(texts)
    finally:
        shutil.rmtree(img_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="list of paths relative to --corpus")
    ap.add_argument("--corpus", default="/workspace/corpus")
    ap.add_argument("--out", default="/workspace/out")
    ap.add_argument("--shard", default="0/1", help="K/N: process files where idx %% N == K")
    ap.add_argument("--sample-pages", type=int, default=0,
                    help="if >0, OCR only ~N evenly spaced pages per doc (verification mode)")
    args = ap.parse_args()

    k, n = (int(x) for x in args.shard.split("/"))
    corpus, out = Path(args.corpus), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp_root = Path("/tmp/ocr_pages"); tmp_root.mkdir(exist_ok=True)

    files = [l.strip() for l in open(args.files) if l.strip()]
    mine = [f for i, f in enumerate(files) if i % n == k]
    print(f"shard {k}/{n}: {len(mine)} of {len(files)} files", flush=True)

    processor, model = load_model()
    done = failed = skipped = 0
    t0 = time.time()
    for rel in mine:
        src = corpus / rel
        dst = out / (src.stem + ".txt")
        if dst.exists():
            done += 1
            continue
        if not src.exists():
            # usually a not-yet-uploaded file — skipped, retried next run
            print(f"MISSING (skip): {rel}", flush=True)
            skipped += 1
            continue
        # cross-worker claim: whoever creates the marker first owns the doc;
        # stale claims (crashed worker) are stolen after 2h
        claim = out / (src.stem + ".claim")
        try:
            with open(claim, "x") as f:
                f.write(f"{time.time()}")
        except FileExistsError:
            try:
                age = time.time() - claim.stat().st_mtime
            except OSError:
                age = 0.0
            if age < 7200:
                continue  # another worker owns it
            claim.touch()  # stale — steal it
        try:
            t = time.time()
            if src.suffix.lower() in IMAGE_EXTS:
                text = ocr_image(src, processor, model)
            elif args.sample_pages > 0:
                text = ocr_pdf_sampled(src, processor, model, tmp_root,
                                       args.sample_pages, rel=rel)
            else:
                text = ocr_pdf(src, processor, model, tmp_root, rel=rel)
            tmp = dst.with_suffix(".txt.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.rename(dst)
            done += 1
            print(f"OK ({time.time()-t:.0f}s, {done}/{len(mine)}): {rel}", flush=True)
        except Exception as e:
            failed += 1
            print(f"FAIL: {rel}: {e}\n{traceback.format_exc(limit=2)}", flush=True)
        finally:
            claim.unlink(missing_ok=True)
    print(f"shard {k}/{n} finished: {done} ok, {failed} failed, {skipped} skipped, "
          f"{(time.time()-t0)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()
