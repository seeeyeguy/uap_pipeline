"""
UAP Document Ingestion & RAG Pipeline
---------------------------------------
Flow: ZIP → Extract → OCR → LLM Enrichment → Chunk → Embed → ChromaDB → Query

LLM enrichment extracts structured metadata AND generates Q&A pairs that serve
as both RAG context and future fine-tuning data for a domain-specific model.
"""

import os
import re
import json
import zipfile
import logging
import threading
import gc
import requests
import torch

# Guards the shared training-JSONL append (enrichment is single-threaded now
# that it runs via the Batch API, but the lock is cheap insurance).
_TRAINING_LOCK = threading.Lock()
from pathlib import Path
from datetime import datetime
from typing import Optional

import anthropic
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from langchain_text_splitters import RecursiveCharacterTextSplitter

from textqc import chunk_junk_reason
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DIRS = {
    "zips":      "./data/zips",
    "raw":       "./data/raw",
    "images":    "./data/images",
    "text":      "./data/text",
    "enriched":  "./data/enriched",    # JSON metadata per document
    "training":  "./data/training",    # JSONL fine-tuning dataset
    "vectordb":  os.environ.get("VECTORDB_DIR", "./data/vectordb"),
}

OCR_MODEL_ID    = "zai-org/GLM-OCR"
# bge-m3: multilingual with cross-lingual alignment — English queries must
# retrieve the Swedish/Spanish/Italian/Portuguese documents in this corpus.
# Dimension change (768->1024) means any index built with bge-base is
# incompatible; switched at the 2026-07 full rebuild.
EMBED_MODEL_ID  = "BAAI/bge-m3"
LLM_MODEL       = "claude-sonnet-4-6"
COLLECTION_NAME = "uap_documents"

# Chunking (2026-07 rebuild): pages are the primary unit for OCR'd docs.
# CHUNK_SIZE/OVERLAP (in CHARACTERS) apply to unmarked text and to
# subdividing oversized single pages.
CHUNK_SIZE        = 2000
CHUNK_OVERLAP     = 200
PAGE_CHUNK_TARGET = 2500   # pack consecutive pages up to this many chars
PAGE_CHUNK_MAX    = 3000   # single pages above this get subdivided
MAX_NEW_TOKENS = 8192

# PDFs whose embedded text layer averages at least this many characters
# per page skip GPU OCR entirely (most modern government releases are
# born-digital or already OCRed).
TEXT_LAYER_MIN_CHARS_PER_PAGE = 200
# Below this plausible-word ratio a text is treated as garbage OCR:
# the PDF text layer is rejected (re-OCR instead) and LLM enrichment is
# skipped (uncached, so it re-runs once better text exists).
TEXT_LAYER_MIN_QUALITY = 0.30

# Text-native formats ingested without OCR
TEXT_NATIVE_EXTS = {".txt", ".md", ".csv", ".json", ".jsonl", ".html", ".htm"}
CSV_ROWS_PER_DOC = 100   # structured rows grouped per pseudo-document

# Set your Anthropic API key in your environment:
#   export ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def setup_dirs():
    for d in DIRS.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    log.info("Directories ready.")


def load_ocr_model():
    log.info(f"Loading OCR model: {OCR_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(OCR_MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        OCR_MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    log.info("OCR model loaded.")
    return processor, model


def load_embed_model():
    log.info(f"Loading embedding model: {EMBED_MODEL_ID}")
    embedder = SentenceTransformer(EMBED_MODEL_ID)
    log.info("Embedding model loaded.")
    return embedder


def load_vectordb():
    client = chromadb.PersistentClient(
        path=DIRS["vectordb"],
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(f"Vector DB ready. '{COLLECTION_NAME}' has {collection.count()} chunks.")
    return collection


def load_llm_client():
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Export it before running.")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


class Models:
    """
    Lazy loader so a run that never needs a component never pays for it —
    e.g. a corpus of born-digital PDFs and CSVs loads no OCR model at all.
    """
    def __init__(self):
        self._ocr = None
        self._embedder = None
        self._collection = None
        self._llm = None

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = load_ocr_model()
        return self._ocr

    def unload_ocr(self):
        """Free GLM-OCR's VRAM before the embedder loads. On an 8 GB card the
        two don't fit together and the embed phase OOMs; the lazy property
        transparently reloads OCR if a later group needs it again."""
        if self._ocr is not None:
            self._ocr = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("OCR model unloaded — VRAM freed for embedding.")

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = load_embed_model()
        return self._embedder

    @property
    def collection(self):
        if self._collection is None:
            self._collection = load_vectordb()
        return self._collection

    @property
    def llm(self):
        if self._llm is None:
            self._llm = load_llm_client()
        return self._llm


# ─────────────────────────────────────────────
# STEP 1 — DOWNLOAD & EXTRACT
# ─────────────────────────────────────────────

def download_zip(url: str, filename: Optional[str] = None) -> Path:
    filename = filename or url.split("/")[-1]
    if not filename.endswith(".zip"):
        filename += ".zip"
    dest = Path(DIRS["zips"]) / filename
    if dest.exists():
        log.info(f"ZIP already downloaded: {dest}")
        return dest
    log.info(f"Downloading {url} -> {dest}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path) -> Path:
    out_dir = Path(DIRS["raw"]) / zip_path.stem
    if out_dir.exists():
        log.info(f"Already extracted: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Extracting {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir


def collect_files(directory: Path, prefer: str = "pdf") -> dict:
    """
    Collect PDFs and images from a directory, resolving two kinds of duplicates:

    1. PREFIX MATCH — an image file whose name starts with a PDF's stem
       (e.g. report.pdf + report_page_001.jpg).  We keep whichever format
       is preferred via the `prefer` argument ("pdf" or "image").

    2. CONTENT DUPLICATES — after OCR, near-identical text across files
       with unrelated names is caught by deduplicate_texts() further down
       the pipeline.

    prefer: "pdf"   → skip images whose prefix matches a PDF stem (default)
            "image" → skip PDFs that have matching pre-rendered images
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}

    all_pdfs   = []
    all_images = []
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() == ".pdf":
            all_pdfs.append(p)
        elif p.suffix.lower() in IMAGE_EXTS:
            all_images.append(p)

    # Build a set of PDF stems for fast prefix lookups
    pdf_stems = {pdf.stem.lower() for pdf in all_pdfs}

    # Group images by the PDF stem they match (if any)
    # An image matches if its filename starts with a pdf stem followed by
    # a non-alpha character (underscore, hyphen, space, digit) or end of stem.
    import re
    def image_matches_pdf(img: Path) -> Optional[str]:
        img_lower = img.stem.lower()
        for stem in pdf_stems:
            # Match stem exactly, or stem followed by separator + anything
            if img_lower == stem or re.match(rf"^{re.escape(stem)}[\W_]", img_lower):
                return stem
        return None

    matched_image_stems: set[str] = set()   # PDF stems that have matching images
    shadowed_images:     set[Path] = set()  # images to skip when preferring PDF

    for img in all_images:
        matched = image_matches_pdf(img)
        if matched:
            matched_image_stems.add(matched)
            shadowed_images.add(img)

    if prefer == "pdf":
        # Skip images that are pre-rendered versions of a PDF we already have
        kept_pdfs   = all_pdfs
        kept_images = [img for img in all_images if img not in shadowed_images]
        if shadowed_images:
            log.info(
                f"Prefix dedup (prefer=pdf): skipping {len(shadowed_images)} images "
                f"that match {len(matched_image_stems)} PDF stem(s)."
            )
    else:
        # Skip PDFs whose pages are already available as images
        shadowed_pdfs = [pdf for pdf in all_pdfs if pdf.stem.lower() in matched_image_stems]
        kept_pdfs     = [pdf for pdf in all_pdfs if pdf.stem.lower() not in matched_image_stems]
        kept_images   = all_images
        if shadowed_pdfs:
            log.info(
                f"Prefix dedup (prefer=image): skipping {len(shadowed_pdfs)} PDFs "
                f"that already have pre-rendered images."
            )

    log.info(
        f"Collected {len(kept_pdfs)} PDFs and {len(kept_images)} images "
        f"from {directory} (after prefix dedup)."
    )
    return {"pdfs": kept_pdfs, "images": kept_images}


# ─────────────────────────────────────────────
# STEP 2 — OCR
# ─────────────────────────────────────────────

def ocr_image(image_path: Path, processor, model) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "url": str(image_path)},
            {"type": "text",  "text": "Document Parsing:"},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def text_quality_score(text: str) -> float:
    """
    Cheap lexical quality estimate: the fraction of tokens that look like
    real words or numbers. Clean text (born-digital or good OCR) scores
    0.6+; mangled embedded OCR layers from microfilm scans score under 0.2.
    Very short texts return 1.0 — the ratio is too noisy to judge them.
    """
    tokens = text.split()
    if len(tokens) < 20:
        return 1.0
    plausible = sum(
        1 for t in tokens
        if re.fullmatch(r"[A-Za-z]{3,}[.,;:!?)]?|\(?\d{1,4}([-/.:]\d{1,4})*[.,;:]?", t)
    )
    return plausible / len(tokens)


def extract_pdf_text_layer(pdf_path: Path) -> Optional[str]:
    """
    Pull the embedded text layer from a PDF. Returns None when the PDF is
    image-only, too sparse, or the layer is garbage OCR (e.g. Internet
    Archive microfilm scans) — all signalling that our own OCR is required.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        if not pages:
            return None
        avg_chars = sum(len(p) for p in pages) / len(pages)
        if avg_chars < TEXT_LAYER_MIN_CHARS_PER_PAGE:
            return None
        score = text_quality_score(" ".join(pages))
        if score < TEXT_LAYER_MIN_QUALITY:
            log.info(f"Text layer is garbage (lexical score {score:.2f} < "
                     f"{TEXT_LAYER_MIN_QUALITY}), falling back to OCR: {pdf_path.name}")
            return None
        return "\n\n".join(f"--- Page {i+1} ---\n{p}" for i, p in enumerate(pages))
    except Exception as e:
        log.warning(f"Text-layer extraction failed for {pdf_path.name}: {e}")
        return None


def ocr_pdf(pdf_path: Path, processor, model) -> str:
    img_dir = Path(DIRS["images"]) / pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Converting PDF to images: {pdf_path.name}")
    # paths_only + output_folder: poppler streams each rendered page straight
    # to disk, so memory stays flat at ~1 page. Holding the PIL images in RAM
    # OOM-killed the process on a 1126-page scan (>11 GB of pixels).
    img_paths = convert_from_path(
        str(pdf_path),
        # OCR_DPI: lower (e.g. 130) rescues dense-graphics scans whose pages
        # allocate ~1 GB at 200 DPI and OOM the 8 GB card (the CARET set).
        dpi=int(os.environ.get("OCR_DPI", "200")),
        output_folder=str(img_dir),
        fmt="jpeg",
        output_file="page_",
        paths_only=True,
    )
    full_text = []
    for i, img_path in enumerate(tqdm(img_paths, desc=f"OCR {pdf_path.name}")):
        page_text = ocr_image(Path(img_path), processor, model)
        full_text.append(f"--- Page {i+1} ---\n{page_text}")
    return "\n\n".join(full_text)


def save_text(text: str, source_path: Path) -> Path:
    out_path = Path(DIRS["text"]) / (source_path.stem + ".txt")
    out_path.write_text(text, encoding="utf-8")
    return out_path


def deduplicate_texts(docs: list[dict], threshold: float = 0.85) -> list[dict]:
    """
    Content-based deduplication for documents with unrelated filenames.

    Uses a fast shingling approach (no heavy ML needed):
      - Builds a set of 5-word shingles from each document's text
      - Computes Jaccard similarity between every pair
      - Drops the shorter document when similarity exceeds `threshold`

    threshold: 0.85 means 85% shingle overlap = considered duplicate.
    Tune down to 0.7 for fuzzier matching, up to 0.95 for near-exact only.
    """
    if not docs:
        return docs

    def shingles(text: str, k: int = 5) -> set:
        words = text.lower().split()
        return set(" ".join(words[i:i+k]) for i in range(max(1, len(words) - k + 1)))

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    log.info(f"Running content deduplication on {len(docs)} documents...")

    shingle_sets = [shingles(doc["text"]) for doc in docs]
    keep = [True] * len(docs)

    for i in range(len(docs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(docs)):
            if not keep[j]:
                continue
            sim = jaccard(shingle_sets[i], shingle_sets[j])
            if sim >= threshold:
                # Drop the shorter document, keep the longer one
                shorter = i if len(docs[i]["text"]) < len(docs[j]["text"]) else j
                keep[shorter] = False
                log.info(
                    f"Duplicate detected (similarity={sim:.2f}): "
                    f"dropping '{docs[shorter]['filename']}' "
                    f"(duplicate of '{docs[i if shorter == j else j]['filename']}')"
                )

    kept   = [doc for doc, k in zip(docs, keep) if k]
    dropped = len(docs) - len(kept)
    if dropped:
        log.info(f"Content dedup: removed {dropped} duplicate(s), {len(kept)} documents remain.")
    else:
        log.info("Content dedup: no duplicates found.")
    return kept


def process_files(files: dict, models: "Models", force_ocr: bool = False) -> list[dict]:
    results = []
    for pdf in files["pdfs"]:
        # Per-file guard: archives contain the occasional corrupt PDF
        # (broken XRef tables etc.) — one must never abort the whole batch.
        try:
            txt_path = Path(DIRS["text"]) / (pdf.stem + ".txt")
            if txt_path.exists():
                log.info(f"Skipping OCR (cached): {pdf.name}")
                text = txt_path.read_text(encoding="utf-8")
            else:
                text = None if force_ocr else extract_pdf_text_layer(pdf)
                if text:
                    log.info(f"Text layer found, skipping OCR: {pdf.name}")
                else:
                    text = ocr_pdf(pdf, *models.ocr)
                save_text(text, pdf)
        except Exception as e:
            log.error(f"Unprocessable PDF, skipping: {pdf.name}: {e}")
            continue
        results.append({"source": str(pdf), "filename": pdf.name, "text": text})

    for img in files["images"]:
        try:
            txt_path = Path(DIRS["text"]) / (img.stem + ".txt")
            if txt_path.exists():
                log.info(f"Skipping OCR (cached): {img.name}")
                text = txt_path.read_text(encoding="utf-8")
            else:
                text = ocr_image(img, *models.ocr)
                save_text(text, img)
        except Exception as e:
            log.error(f"Unprocessable image, skipping: {img.name}: {e}")
            continue
        results.append({"source": str(img), "filename": img.name, "text": text})

    return results


def read_text_native(path: Path) -> list[dict]:
    """
    Ingest text-native files without OCR.

    - .txt/.md          -> one document
    - .html/.htm        -> tag-stripped text, one document
    - .csv              -> row groups of CSV_ROWS_PER_DOC as pseudo-documents
                           (header repeated per group so rows stay readable)
    - .json/.jsonl      -> pretty-printed records, grouped like CSV rows
    """
    import csv as _csv
    import io as _io

    suffix = path.suffix.lower()
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning(f"Cannot read {path}: {e}")
        return []

    def doc(text, part=None):
        fname = path.name if part is None else f"{path.stem}__part{part:04d}{path.suffix}"
        return {"source": str(path), "filename": fname, "text": text,
                "text_native": True}

    if suffix in (".txt", ".md"):
        return [doc(raw)] if raw.strip() else []

    if suffix in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(raw, "html.parser").get_text(separator="\n")
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            return [doc(text)] if text else []
        except ImportError:
            return [doc(raw)]

    if suffix == ".csv":
        try:
            rows = list(_csv.reader(_io.StringIO(raw)))
        except _csv.Error as e:
            log.warning(f"CSV parse failed for {path.name}: {e}")
            return [doc(raw)]
        if not rows:
            return []
        header, body = rows[0], rows[1:]
        docs = []
        for part, i in enumerate(range(0, len(body), CSV_ROWS_PER_DOC)):
            lines = [", ".join(header)]
            lines += [", ".join(str(c) for c in row) for row in body[i:i + CSV_ROWS_PER_DOC]]
            docs.append(doc("\n".join(lines), part))
        return docs

    if suffix in (".json", ".jsonl"):
        records = []
        if suffix == ".jsonl":
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        records.append(line)
        else:
            try:
                data = json.loads(raw)
                records = data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                return [doc(raw)]
        docs = []
        for part, i in enumerate(range(0, len(records), CSV_ROWS_PER_DOC)):
            chunk = records[i:i + CSV_ROWS_PER_DOC]
            docs.append(doc("\n".join(json.dumps(r, ensure_ascii=False) for r in chunk), part))
        return docs

    return []


# ─────────────────────────────────────────────
# STEP 3 — LLM ENRICHMENT
# ─────────────────────────────────────────────

ENRICHMENT_SYSTEM_PROMPT = """You are an expert analyst specializing in UAP (Unidentified Aerial Phenomena),
UFO historical documentation, government records, and aerospace anomalies.
You are processing declassified documents, witness testimonies, government memos, and research reports.
Your job is to extract structured metadata and generate high-quality question-answer pairs from document text.
Always respond with valid JSON only. No preamble, no explanation, no markdown fences."""

ENRICHMENT_USER_PROMPT = """Analyze the following document text and return a JSON object with this exact structure:

{{
  "title": "short display title (max 75 chars) a researcher would recognize in a list: lead with the collection or agency when known (FBI file, CIA memo, Project Blue Book case file...), then the central subject — a person, named incident, or place + date. Use only facts from the text; plain text, no trailing period",
  "summary": "2-4 sentence factual summary of what this document contains",
  "document_type": "one of: [sighting_report, government_memo, witness_testimony, research_report, news_article, investigation_report, correspondence, unknown]",
  "event_date": "ISO date string if determinable (YYYY-MM-DD), or date range (YYYY/YYYY), or null",
  "event_location": {{
    "country": "string or null",
    "region": "state/province/region or null",
    "city": "string or null (if the document covers MULTIPLE nearby towns, use the primary/first one — never null just because there are several)",
    "site": "specific site name (e.g. air base, lake) or null"
  }},
  "entities": {{
    "people": ["list of named individuals mentioned"],
    "organizations": ["agencies, military units, companies mentioned"],
    "craft_descriptions": ["any descriptions of UAP/UFO appearance, behavior, or capabilities"]
  }},
  "classification_level": "one of: [unclassified, confidential, secret, top_secret, unknown, not_applicable]",
  "credibility_indicators": {{
    "official_source": true or false,
    "multiple_witnesses": true or false,
    "physical_evidence_mentioned": true or false,
    "radar_confirmation": true or false,
    "government_acknowledgment": true or false
  }},
  "topics": ["list of relevant topic tags, e.g. close_encounter, abduction, crash_retrieval, government_coverup, military_encounter, nuclear_connection, mass_sighting"],
  "time_period": "one of: [pre_1947, 1947_1969, 1970_1989, 1990_2009, 2010_present, unknown]",
  "ocr_quality": "one of: [good, degraded, garbage] — good: text is clean and complete; degraded: legible but with OCR noise, gaps, or garbled passages; garbage: mostly unreadable, not usable as source material",
  "ocr_notes": "one short sentence describing any OCR/legibility problems, or null if none",
  "qa_pairs": [
    {{
      "question": "A natural question a researcher or journalist would ask about this document",
      "answer": "A thorough, factual answer grounded strictly in the document text",
      "question_type": "one of: [factual, analytical, contextual, comparative]"
    }}
  ]
}}

If ocr_quality is "garbage", return an empty qa_pairs array — do not fabricate answers from unreadable text.
Otherwise generate between 3 and 8 qa_pairs depending on document richness. Questions should cover:
- What happened / what is described
- Who was involved
- When and where
- What evidence or corroboration exists
- What the significance of the document is

Document text:
{text}"""


def _enrichment_cache_path(doc: dict) -> Path:
    return Path(DIRS["enriched"]) / (Path(doc["filename"]).stem + ".json")


def _enrichment_request_params(doc: dict) -> dict:
    """Messages-API params for one enrichment call — shared by the sync and
    Batch API paths so both produce byte-identical requests."""
    # Truncate very long texts to avoid huge token bills.
    # Enrichment works on a representative sample; full text goes to chunking.
    text_sample = doc["text"][:12000]
    return {
        "model": LLM_MODEL,
        # Output is metadata + up to 8 Q&A pairs (~1-2k tokens) — the old
        # clean_text echo is gone, so 4096 is comfortable headroom.
        "max_tokens": 4096,
        "system": ENRICHMENT_SYSTEM_PROMPT,
        "messages": [{
            "role": "user",
            "content": ENRICHMENT_USER_PROMPT.format(text=text_sample),
        }],
    }


def _parse_enrichment_response(doc: dict, response) -> dict:
    """Turn a Messages API response into an enriched dict (defaults on
    refusal or parse failure)."""
    try:
        # The model can decline via stop_reason="refusal" with empty content
        # (common on declassified intelligence docs). Detect it explicitly so
        # it's logged as a refusal, not a cryptic "list index out of range".
        if response.stop_reason == "refusal" or not response.content:
            log.warning(f"Enrichment refused by model for {doc['filename']} "
                        f"(stop_reason={response.stop_reason}); using defaults.")
            return _default_enrichment(doc["filename"])
        raw = response.content[0].text.strip()
        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        log.warning(f"Enrichment failed for {doc['filename']}: {e}. Using defaults.")
        return _default_enrichment(doc["filename"])


def enrich_document(doc: dict, llm_client) -> dict:
    """
    Pass document text through Claude to extract structured metadata
    and generate Q&A pairs for both RAG and future model training.
    """
    cache_path = _enrichment_cache_path(doc)
    if cache_path.exists():
        log.info(f"Skipping enrichment (cached): {doc['filename']}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    log.info(f"Enriching: {doc['filename']}")
    response = llm_client.messages.create(**_enrichment_request_params(doc))
    enriched = _parse_enrichment_response(doc, response)

    return _finalize_enrichment(doc, enriched)


def _finalize_enrichment(doc: dict, enriched: dict) -> dict:
    """Attach source info, write the per-doc cache, and append training data."""
    enriched["source"]       = doc["source"]
    enriched["filename"]     = doc["filename"]
    enriched["ingested_at"]  = datetime.utcnow().isoformat()

    _enrichment_cache_path(doc).write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

    # Append Q&A pairs to the training JSONL dataset (raw text as context —
    # the LLM-cleaned echo was dropped; see ocr_quality instead)
    _append_training_data(enriched, doc_text=doc.get("text", ""))

    return enriched


def _default_enrichment(filename: str) -> dict:
    """Fallback enrichment if LLM call fails."""
    return {
        "title": "",
        "summary": "",
        "document_type": "unknown",
        "event_date": None,
        "event_location": {"country": None, "region": None, "city": None, "site": None},
        "entities": {"people": [], "organizations": [], "craft_descriptions": []},
        "classification_level": "unknown",
        "credibility_indicators": {
            "official_source": False,
            "multiple_witnesses": False,
            "physical_evidence_mentioned": False,
            "radar_confirmation": False,
            "government_acknowledgment": False,
        },
        "topics": [],
        "time_period": "unknown",
        "ocr_quality": "unknown",
        "ocr_notes": None,
        "qa_pairs": [],
    }


def _append_training_data(enriched: dict, doc_text: str = ""):
    """
    Write Q&A pairs to a JSONL file in a format compatible with:
      - Anthropic fine-tuning API
      - OpenAI fine-tuning API
      - HuggingFace SFTTrainer (minor column renaming needed)

    Each record includes a system prompt with document context so the model
    learns to answer grounded in source material — critical for a domain
    model that will eventually run without retrieval.
    """
    training_path = Path(DIRS["training"]) / "uap_qa_dataset.jsonl"

    system_ctx = (
        "You are an expert on UAP/UFO historical documentation. "
        "Answer questions based on the following document:\n\n"
        f"Source: {enriched.get('filename', 'unknown')}\n"
        f"Type: {enriched.get('document_type', 'unknown')}\n"
        f"Summary: {enriched.get('summary', '')}\n\n"
        f"Document text:\n{doc_text[:3000]}"
    )

    # Garbage-OCR docs produce no usable Q&A pairs, and any that slip through
    # would poison the fine-tuning set with answers grounded in noise.
    if enriched.get("ocr_quality") == "garbage":
        return

    # Guard the shared append so concurrent enrichment threads don't interleave.
    with _TRAINING_LOCK, open(training_path, "a", encoding="utf-8") as f:
        for qa in enriched.get("qa_pairs", []):
            record = {
                "system": system_ctx,
                "messages": [
                    {"role": "user",      "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ],
                # Extra metadata preserved for dataset filtering/analysis later
                "metadata": {
                    "source":        enriched.get("filename"),
                    "document_type": enriched.get("document_type"),
                    "time_period":   enriched.get("time_period"),
                    "topics":        enriched.get("topics", []),
                    "question_type": qa.get("question_type"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _enrich_via_batch_api(docs: list[dict], llm_client):
    """
    Enrich via the Message Batches API — 50% off all token usage, and this
    workload is exactly batch-shaped (thousands of independent docs, zero
    latency sensitivity). Most batches complete within an hour.

    Succeeded results are cached like the sync path. Errored/expired results
    write NO cache, so the next run retries them (a cached default would
    never retry — the cache-of-failures trap).
    """
    import time
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    BATCH_CHUNK = 1000  # well under the 100k-request/256MB limits; bounds blast radius

    for start in range(0, len(docs), BATCH_CHUNK):
        group = docs[start:start + BATCH_CHUNK]
        by_id = {f"doc-{start + i}": d for i, d in enumerate(group)}
        batch = llm_client.messages.batches.create(requests=[
            Request(custom_id=cid,
                    params=MessageCreateParamsNonStreaming(**_enrichment_request_params(d)))
            for cid, d in by_id.items()
        ])
        log.info(f"Enrichment batch {batch.id}: {len(group)} docs submitted "
                 f"({start + len(group)}/{len(docs)} total). Polling…")

        while True:
            b = llm_client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            c = b.request_counts
            log.info(f"  batch {batch.id}: {c.succeeded} ok, {c.errored} err, "
                     f"{c.processing} processing")
            time.sleep(30)

        for result in llm_client.messages.batches.results(batch.id):
            doc = by_id[result.custom_id]
            if result.result.type == "succeeded":
                enriched = _parse_enrichment_response(doc, result.result.message)
                _finalize_enrichment(doc, enriched)
            else:
                # No cache written → retried on the next run
                log.warning(f"Batch enrichment {result.result.type} for "
                            f"{doc['filename']} — will retry next run.")


def enrich_all(docs: list[dict], llm_client) -> list[dict]:
    # Hard pause: while data/ENRICH_PAUSED exists, no enrichment API calls are
    # made and nothing is cached — cached results are still used, uncached
    # docs get defaults and will enrich normally once the sentinel is removed.
    if (Path(DIRS["text"]).parent / "ENRICH_PAUSED").exists():
        log.warning(f"ENRICH_PAUSED sentinel present — skipping enrichment API "
                    f"calls for {len(docs)} doc(s) (rm data/ENRICH_PAUSED to resume).")
        out = []
        for doc in docs:
            cache = _enrichment_cache_path(doc)
            enriched = (json.loads(cache.read_text(encoding="utf-8"))
                        if cache.exists() else _default_enrichment(doc["filename"]))
            out.append({**doc, **enriched})
        return out

    # Quality gate: don't pay to enrich text the OCR mangled — metadata and
    # Q&A pairs generated from garbage text are worthless, and these docs
    # will be re-OCR'd. No cache is written for skipped docs, so enrichment
    # runs (once) as soon as better text exists.
    skipped_keys = {
        id(d) for d in docs
        if not _enrichment_cache_path(d).exists()
        and text_quality_score(d.get("text", "")) < TEXT_LAYER_MIN_QUALITY
    }
    if skipped_keys:
        log.info(f"Enrichment skipped for {len(skipped_keys)} low-quality doc(s) "
                 f"(lexical score < {TEXT_LAYER_MIN_QUALITY}) — deferred until re-OCR.")

    pending = [d for d in docs
               if id(d) not in skipped_keys and not _enrichment_cache_path(d).exists()]

    # Batch API for real workloads; the sync path for tiny runs (batch
    # round-trip latency isn't worth it) or when forced via ENRICH_SYNC=1.
    if pending:
        if len(pending) <= 5 or os.environ.get("ENRICH_SYNC") == "1":
            for d in tqdm(pending, desc="LLM enrichment (sync)"):
                enrich_document(d, llm_client)
        else:
            _enrich_via_batch_api(pending, llm_client)

    enriched_docs = []
    for doc in docs:
        cache = _enrichment_cache_path(doc)
        if id(doc) in skipped_keys:
            enriched = _default_enrichment(doc["filename"])
            enriched["ocr_quality"] = "garbage"
            enriched["ocr_notes"] = "enrichment skipped by lexical quality gate"
        elif cache.exists():
            enriched = json.loads(cache.read_text(encoding="utf-8"))
        else:
            # batch result errored/expired — defaults for this run, no cache
            enriched = _default_enrichment(doc["filename"])
        enriched_docs.append({**doc, **enriched})
    log.info(f"Enrichment complete for {len(enriched_docs)} documents "
             f"({len(pending)} newly enriched, {len(skipped_keys)} quality-gated).")
    return enriched_docs


def process_doc_batch(docs: list[dict], models: "Models", enrich: bool = True) -> int:
    """
    Shared tail of the pipeline: dedupe -> enrich -> chunk -> embed -> store.

    Text-native pseudo-documents (CSV/JSON row groups) skip LLM enrichment:
    they are already structured, and enriching thousands of row groups
    would burn tokens for no metadata gain. Returns the chunk count.
    """
    # Content dedup is an O(n²) shingle comparison — worthwhile only for OCR'd
    # documents with unrelated filenames. Structured/text-native pseudo-docs
    # (CSV/JSON row groups) are numerous and identity-stable, so deduping them
    # is both semantically wrong and prohibitively slow (thousands of rows →
    # millions of pairwise comparisons that peg one CPU for many minutes).
    ocr_docs    = [d for d in docs if not d.get("text_native")]
    native_docs = [d for d in docs if d.get("text_native")]
    ocr_docs    = deduplicate_texts(ocr_docs, threshold=0.85)

    # ENRICH_NATIVE=1: push text-native docs through Claude enrichment too.
    # The default skip exists for structured row groups (CSV/JSON), where
    # enrichment buys nothing — but prose HTML (e.g. the Wikipedia biography/
    # incident corpus) needs enrichment or its people/organizations arrays
    # stay empty and the docs never reach corpus.entities / the entity graph.
    enrich_native = enrich and os.environ.get("ENRICH_NATIVE") == "1"

    enriched_docs = []
    if ocr_docs:
        if enrich:
            enriched_docs += enrich_all(ocr_docs, models.llm)
        else:
            enriched_docs += ocr_docs
    if native_docs and enrich_native:
        for d in enrich_all(native_docs, models.llm):
            enriched_docs.append({**d, "ocr_quality": "good"})
    else:
        for d in native_docs:
            enriched_docs.append({
                **d,
                **_default_enrichment(d["filename"]),
                "document_type": "structured_dataset",
                "ocr_quality": "good",  # text-native — no OCR involved
            })

    chunks = chunk_documents(enriched_docs)
    # OCR is finished for this batch — free its VRAM before the embedder
    # loads (the two never fit together on the 8 GB card). A later group
    # that needs OCR again reloads it lazily.
    models.unload_ocr()
    embed_and_store(chunks, models.embedder, models.collection)
    return len(chunks)


# ─────────────────────────────────────────────
# STEP 4 — CHUNK
# ─────────────────────────────────────────────

PAGE_MARKER = re.compile(r"^--- Page (\d+) ---$", re.M)


def _page_aligned_splits(text: str, splitter) -> list[tuple[str, str]]:
    """
    Split OCR text on its page markers, then pack pages into chunks:
      - merge consecutive pages up to PAGE_CHUNK_TARGET chars
        (a record card is one page; a memo is 2-3 — pages are the natural
        semantic unit of scanned government documents)
      - subdivide any single page over PAGE_CHUNK_MAX with the recursive
        splitter so dense book pages don't produce diluted mega-chunks
    Returns (chunk_text, page_span) pairs, e.g. ("...", "3-5").
    """
    parts = PAGE_MARKER.split(text)
    # parts = [preamble, pageno, body, pageno, body, ...]
    pages = []
    if parts[0].strip():
        pages.append(("1", parts[0].strip()))
    for pageno, body in zip(parts[1::2], parts[2::2]):
        if body.strip():
            pages.append((pageno, body.strip()))
    if not pages:
        return []

    out = []
    buf, buf_start, buf_end = "", None, None
    for pageno, body in pages:
        if len(body) > PAGE_CHUNK_MAX:
            if buf:
                out.append((buf, _span(buf_start, buf_end)))
                buf, buf_start = "", None
            for piece in splitter.split_text(body):
                out.append((piece, pageno))
            continue
        if buf and len(buf) + len(body) > PAGE_CHUNK_TARGET:
            out.append((buf, _span(buf_start, buf_end)))
            buf, buf_start = "", None
        buf = f"{buf}\n\n{body}" if buf else body
        buf_start = buf_start or pageno
        buf_end = pageno
    if buf:
        out.append((buf, _span(buf_start, buf_end)))
    return out


def _span(a, b):
    return a if a == b else f"{a}-{b}"


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Chunk the raw document text. All enriched metadata is attached to every
    chunk so filtering works at query time without needing a separate
    metadata store.

    Chunking policy (2026-07 rebuild):
      - text-native pseudo-docs (CSV/JSON rows, sightings): one chunk per
        record — never page-split structured data
      - OCR'd documents: page-aligned packing (see _page_aligned_splits)
      - anything without page markers: recursive split at CHUNK_SIZE
      - every chunk carries `pages` + parent identifiers (filename/source)
        so the serving layer can do small-to-big retrieval: match on the
        chunk, hand the LLM the surrounding page span from data/text
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = []
    for doc in docs:
        # Chunk the FULL raw document text — never an enrichment-derived
        # sample. (The old clean_text echo covered only the first ~12k chars
        # and silently dropped ~96% of large documents from the index.)
        text_to_chunk = doc.get("text", "")
        if not text_to_chunk.strip():
            log.warning(f"Empty text for {doc['filename']}, skipping.")
            continue

        if doc.get("text_native"):
            # structured rows are already one semantic unit each
            splits = [(text_to_chunk, "")] if len(text_to_chunk) <= PAGE_CHUNK_MAX \
                else [(s, "") for s in splitter.split_text(text_to_chunk)]
        elif PAGE_MARKER.search(text_to_chunk):
            splits = _page_aligned_splits(text_to_chunk, splitter)
        else:
            splits = [(s, "") for s in splitter.split_text(text_to_chunk)]

        # drop degenerate chunks (OCR loops, alphabet-free table shred) —
        # they embed near any short query and pollute retrieval
        splits = [(s, p) for s, p in splits if not chunk_junk_reason(s)]

        # English enrichment summary as an auxiliary chunk (chunk_id -1):
        # gives every document — especially non-English ones — an English
        # semantic entry point in the index, on top of bge-m3's native
        # cross-lingual matching.
        summary = (doc.get("summary") or "").strip()
        if summary:
            splits = [(summary, "summary")] + splits
        for i, (split, pages) in enumerate(splits, start=(-1 if summary else 0)):
            chunks.append({
                "text":                 split,
                "source":               doc["source"],
                "filename":             doc["filename"],
                "chunk_id":             i,
                "pages":                pages,
                # Enriched fields — all stringified for ChromaDB compatibility
                "summary":              doc.get("summary", ""),
                "title":                doc.get("title") or "",
                "document_type":        doc.get("document_type", "unknown"),
                "event_date":           doc.get("event_date") or "",
                "country":              (doc.get("event_location") or {}).get("country") or "",
                "region":               (doc.get("event_location") or {}).get("region") or "",
                "city":                 (doc.get("event_location") or {}).get("city") or "",
                "time_period":          doc.get("time_period", "unknown"),
                "classification_level": doc.get("classification_level", "unknown"),
                "ocr_quality":          doc.get("ocr_quality", "unknown"),
                "topics":               json.dumps(doc.get("topics", [])),
                "people":               json.dumps((doc.get("entities") or {}).get("people", [])),
                "organizations":        json.dumps((doc.get("entities") or {}).get("organizations", [])),
                "craft_descriptions":   json.dumps((doc.get("entities") or {}).get("craft_descriptions", [])),
                "official_source":      str((doc.get("credibility_indicators") or {}).get("official_source", False)),
                "radar_confirmation":   str((doc.get("credibility_indicators") or {}).get("radar_confirmation", False)),
                "physical_evidence":    str((doc.get("credibility_indicators") or {}).get("physical_evidence_mentioned", False)),
            })

    log.info(f"Created {len(chunks)} chunks from {len(docs)} documents.")
    return chunks


# ─────────────────────────────────────────────
# STEP 5 — EMBED & STORE
# ─────────────────────────────────────────────

def embed_and_store(chunks: list[dict], embedder, collection):
    if not chunks:
        log.warning("No chunks to embed.")
        return

    import hashlib
    # Key on filename, not source: CSV/JSON row-group pseudo-docs share one
    # source path but get unique filenames ("foo__part0007.csv"), so keying on
    # source stem collides across all parts. A short source-path hash also
    # disambiguates identical filenames from different sources. Deterministic,
    # so re-runs upsert idempotently.
    def _chunk_key(c):
        src_tag = hashlib.sha1(c["source"].encode("utf-8")).hexdigest()[:8]
        return f"{Path(c['filename']).stem}_{src_tag}_chunk_{c['chunk_id']}"

    texts     = [c["text"] for c in chunks]
    ids       = [_chunk_key(c) for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    log.info(f"Embedding {len(texts)} chunks...")
    # batch_size only affects peak VRAM, not the output vectors. bge-m3 with
    # 8192-token sequences OOMs an 8GB card at the ST default of 32.
    embed_batch = int(os.environ.get("EMBED_BATCH", "32"))
    embeddings = embedder.encode(texts, normalize_embeddings=True,
                                 batch_size=embed_batch, show_progress_bar=True)

    batch_size = 500
    for i in range(0, len(texts), batch_size):
        collection.upsert(
            documents=texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size],
        )
    log.info(f"Stored {len(texts)} chunks. Collection total: {collection.count()}")


# ─────────────────────────────────────────────
# STEP 6 — QUERY
# ─────────────────────────────────────────────

def query(
    question: str,
    embedder,
    collection,
    n_results: int = 5,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Retrieve top-n relevant chunks for a question.

    Optional metadata filters (ChromaDB where clause):
      {"document_type": "government_memo"}
      {"time_period": "1947_1969"}
      {"official_source": "True"}
      {"radar_confirmation": "True"}
    """
    embedding = embedder.encode([question], normalize_embeddings=True).tolist()
    kwargs = {"query_embeddings": embedding, "n_results": n_results}
    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)
    output = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        output.append({
            "text":             doc,
            "source":           meta.get("filename"),
            "score":            round(1 - results["distances"][0][i], 4),
            "document_type":    meta.get("document_type"),
            "event_date":       meta.get("event_date"),
            "time_period":      meta.get("time_period"),
            "summary":          meta.get("summary"),
            "topics":           json.loads(meta.get("topics", "[]")),
            "official_source":  meta.get("official_source"),
        })
    return output


def print_results(results: list[dict]):
    print("\n" + "="*70)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['source']}  |  Score: {r['score']}  |  {r['document_type']}  |  {r['time_period']}")
        if r.get("event_date"):
            print(f"    Date: {r['event_date']}")
        if r.get("topics"):
            print(f"    Topics: {', '.join(r['topics'])}")
        if r.get("summary"):
            print(f"    Summary: {r['summary'][:200]}...")
        print("-"*70)
        print(r["text"])
    print("="*70 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# PROGRESS TRACKER
# ─────────────────────────────────────────────

PROGRESS_FILE = "./data/progress.json"

def load_progress() -> dict:
    """
    Load the progress ledger from disk.
    Structure:
    {
      "completed": {
        "some_file.zip": {
          "completed_at": "2025-01-01T00:00:00",
          "sha256": "abc123...",
          "doc_count": 42
        }
      },
      "failed": {
        "bad_file.zip": {
          "failed_at": "...",
          "error": "..."
        }
      }
    }
    """
    p = Path(PROGRESS_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"completed": {}, "failed": {}}


def save_progress(progress: dict):
    Path(PROGRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(PROGRESS_FILE).write_text(
        json.dumps(progress, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def file_sha256(path: Path, chunk_size: int = 65536) -> str:
    """SHA256 of a file for reliable change detection."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_size):
            h.update(block)
    return h.hexdigest()


def mark_completed(progress: dict, zip_path: Path, doc_count: int):
    progress["completed"][zip_path.name] = {
        "completed_at": datetime.utcnow().isoformat(),
        "sha256":       file_sha256(zip_path),
        "path":         str(zip_path),
        "doc_count":    doc_count,
    }
    progress["failed"].pop(zip_path.name, None)
    save_progress(progress)


def mark_failed(progress: dict, zip_path: Path, error: str, key: str | None = None):
    # key lets callers use the same scoped key ("source/file.pdf") the success
    # path writes; a later success must find and pop this exact entry, or the
    # failure lingers in the ledger forever.
    progress["failed"][key or zip_path.name] = {
        "failed_at": datetime.utcnow().isoformat(),
        "path":      str(zip_path),
        "error":     error,
    }
    save_progress(progress)


def is_already_done(progress: dict, zip_path: Path) -> bool:
    """
    A ZIP is considered done if it is in completed AND its SHA256 has not
    changed — so dropping a new version of the same filename re-processes it.
    """
    entry = progress["completed"].get(zip_path.name)
    if not entry:
        return False
    if file_sha256(zip_path) != entry.get("sha256"):
        log.info(f"ZIP changed since last run: {zip_path.name} — reprocessing.")
        return False
    return True


def scan_folder(folder: str) -> list:
    """Return all ZIP files found in a folder, sorted by name."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    zips = sorted(folder_path.rglob("*.zip"))
    log.info(f"Found {len(zips)} ZIP file(s) in {folder}")
    return zips


def print_progress_report(progress: dict):
    completed  = progress["completed"]
    failed     = progress["failed"]
    total_docs = sum(v.get("doc_count", 0) for v in completed.values())
    print("\n" + "="*60)
    print(f"  PROGRESS REPORT")
    print(f"  Completed : {len(completed)} ZIPs  ({total_docs} documents)")
    print(f"  Failed    : {len(failed)} ZIPs")
    if failed:
        print("\n  Failed files:")
        for name, info in failed.items():
            print(f"    - {name}: {info['error']}")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────

def ingest(zip_sources: list, force: bool = False):
    """
    zip_sources can be:
      - A list of ZIP file paths or URLs
      - A single folder path (all ZIPs inside will be processed)

    force=True reprocesses ZIPs even if already in the ledger.
    """
    setup_dirs()
    progress = load_progress()

    # Expand folder paths to individual ZIP files
    zip_paths = []
    for s in zip_sources:
        if s.startswith("http://") or s.startswith("https://"):
            zip_paths.append(download_zip(s))
        else:
            p = Path(s)
            if p.is_dir():
                zip_paths.extend(scan_folder(s))
            elif p.exists():
                zip_paths.append(p)
            else:
                log.error(f"Not found: {s}")

    # Filter already-completed ZIPs unless forced
    pending = []
    for zp in zip_paths:
        if not force and is_already_done(progress, zp):
            log.info(f"Skipping (already processed): {zp.name}")
        else:
            pending.append(zp)

    if not pending:
        log.info("Nothing new to process.")
        print_progress_report(progress)
        return

    log.info(f"{len(pending)} ZIP(s) to process, {len(zip_paths) - len(pending)} already done.")

    # Lazily loaded — nothing heavy loads until a document actually needs it
    models = Models()

    for zip_path in pending:
        log.info(f"Processing: {zip_path.name}")
        try:
            extracted_dir = extract_zip(zip_path)
            files         = collect_files(extracted_dir, prefer="pdf")
            docs          = process_files(files, models)
            chunks        = process_doc_batch(docs, models)

            mark_completed(progress, zip_path, doc_count=len(docs))
            log.info(f"Done: {zip_path.name} ({len(docs)} docs, {chunks} chunks)")

        except Exception as e:
            log.error(f"Failed: {zip_path.name}: {e}", exc_info=True)
            mark_failed(progress, zip_path, error=str(e))
            log.info("Continuing to next ZIP...")
            continue

    training_path = Path(DIRS["training"]) / "uap_qa_dataset.jsonl"
    if training_path.exists():
        line_count = sum(1 for _ in open(training_path))
        log.info(f"Training dataset: {training_path} ({line_count} Q&A examples)")

    print_progress_report(progress)
    log.info("Ingestion complete.")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def ingest_tree(root: str, force: bool = False, enrich: bool = True,
                sources: list[str] | None = None):
    """
    Ingest an arbitrary directory tree — the shape downloader.py produces
    (data/downloads/<source_id>/...), including git clones and IA mirrors:

      *.zip                  -> the classic ZIP flow (extract/OCR/enrich)
      *.pdf, images          -> OCR (or PDF text layer) + enrichment
      *.txt/.csv/.json/.html -> text-native ingestion, no OCR, no enrichment

    Loose files are processed in batches per top-level subdirectory (one
    batch per source), and tracked in the progress ledger by relative path
    + SHA-256 so re-runs only touch new material.
    """
    setup_dirs()
    progress = load_progress()
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Ingest root not found: {root}")

    models = Models()

    def in_scope(p: Path) -> bool:
        # top-level subdir under the downloads root = source id
        if sources is None:
            return True
        rel = p.relative_to(root_path)
        return bool(rel.parts) and rel.parts[0] in sources

    # 1. ZIPs go through the battle-tested ZIP flow
    for zip_path in sorted(root_path.rglob("*.zip")):
        if not in_scope(zip_path):
            continue
        if not force and is_already_done(progress, zip_path):
            log.info(f"Skipping (already processed): {zip_path.name}")
            continue
        log.info(f"Processing ZIP: {zip_path}")
        try:
            extracted_dir = extract_zip(zip_path)
            files         = collect_files(extracted_dir, prefer="pdf")
            docs          = process_files(files, models)
            chunks        = process_doc_batch(docs, models, enrich=enrich)
            mark_completed(progress, zip_path, doc_count=len(docs))
            log.info(f"Done: {zip_path.name} ({len(docs)} docs, {chunks} chunks)")
        except Exception as e:
            log.error(f"Failed: {zip_path.name}: {e}", exc_info=True)
            mark_failed(progress, zip_path, error=str(e))

    # 2. Loose files, batched per top-level subdirectory (= per source)
    subdirs = sorted(d for d in root_path.iterdir()
                     if d.is_dir() and (sources is None or d.name in sources)) \
              or ([root_path] if sources is None else [])
    for subdir in subdirs:
        loose = [p for p in sorted(subdir.rglob("*"))
                 if p.is_file()
                 and ".git" not in p.parts
                 and p.suffix.lower() != ".zip"
                 and (p.suffix.lower() == ".pdf"
                      or p.suffix.lower() in IMAGE_EXTS
                      or p.suffix.lower() in TEXT_NATIVE_EXTS)]

        pending = []
        for p in loose:
            key = str(p.relative_to(root_path))
            entry = progress["completed"].get(key)
            if not force and entry and entry.get("sha256") == file_sha256(p):
                continue
            pending.append(p)

        if not pending:
            continue
        log.info(f"── Ingesting {len(pending)} loose file(s) from {subdir}")

        docs = []
        files = {"pdfs": [], "images": []}
        for p in pending:
            suffix = p.suffix.lower()
            try:
                if suffix == ".pdf":
                    files["pdfs"].append(p)
                elif suffix in IMAGE_EXTS:
                    files["images"].append(p)
                else:
                    docs.extend(read_text_native(p))
            except Exception as e:
                log.error(f"Failed to read {p}: {e}")

        try:
            docs = process_files(files, models) + docs
            chunks = process_doc_batch(docs, models, enrich=enrich)
            for p in pending:
                rel = str(p.relative_to(root_path))
                progress["completed"][rel] = {
                    "completed_at": datetime.utcnow().isoformat(),
                    "sha256":       file_sha256(p),
                    "path":         str(p),
                    "doc_count":    1,
                }
                progress["failed"].pop(rel, None)
                progress["failed"].pop(p.name, None)  # legacy bare-name keys
            save_progress(progress)
            log.info(f"Done: {subdir.name} ({len(docs)} docs, {chunks} chunks)")
        except Exception as e:
            log.error(f"Batch failed for {subdir}: {e}", exc_info=True)
            for p in pending:
                mark_failed(progress, p, error=str(e),
                            key=str(p.relative_to(root_path)))

    print_progress_report(progress)
    log.info("Tree ingestion complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  Folder:  python pipeline.py ./.data/")
        print("  Files:   python pipeline.py file1.zip file2.zip")
        print("  URL:     python pipeline.py https://example.com/docs.zip")
        print("  Force:   python pipeline.py ./.data/ --force")
        print("  Status:  python pipeline.py --status")
        print("  Query:   python pipeline.py --query \"your question\"")
        print("  Filter:  python pipeline.py --query \"roswell\" --type government_memo --period 1947_1969 --official")
        sys.exit(1)

    if sys.argv[1] == "--status":
        print_progress_report(load_progress())

    elif sys.argv[1] == "--query":
        args     = sys.argv[2:]
        question = ""
        filters  = {}

        i = 0
        while i < len(args):
            if args[i] == "--type" and i+1 < len(args):
                filters["document_type"] = args[i+1]; i += 2
            elif args[i] == "--period" and i+1 < len(args):
                filters["time_period"] = args[i+1]; i += 2
            elif args[i] == "--official":
                filters["official_source"] = "True"; i += 1
            elif args[i] == "--radar":
                filters["radar_confirmation"] = "True"; i += 1
            else:
                question += args[i] + " "; i += 1

        embedder   = load_embed_model()
        collection = load_vectordb()
        results    = query(question.strip(), embedder, collection, filters=filters or None)
        print_results(results)

    else:
        args    = sys.argv[1:]
        force   = "--force" in args
        sources = [a for a in args if not a.startswith("--")]
        ingest(sources, force=force)
