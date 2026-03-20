"""
UAP Document Ingestion & RAG Pipeline
---------------------------------------
Flow: ZIP → Extract → OCR → LLM Enrichment → Chunk → Embed → ChromaDB → Query

LLM enrichment extracts structured metadata AND generates Q&A pairs that serve
as both RAG context and future fine-tuning data for a domain-specific model.
"""

import os
import json
import zipfile
import logging
import requests
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional

import anthropic
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    "vectordb":  "./data/vectordb",
}

OCR_MODEL_ID    = "zai-org/GLM-OCR"
EMBED_MODEL_ID  = "BAAI/bge-base-en-v1.5"
LLM_MODEL       = "claude-sonnet-4-20250514"
COLLECTION_NAME = "uap_documents"

CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
MAX_NEW_TOKENS = 8192

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


def ocr_pdf(pdf_path: Path, processor, model) -> str:
    img_dir = Path(DIRS["images"]) / pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Converting PDF to images: {pdf_path.name}")
    pages = convert_from_path(str(pdf_path), dpi=200)
    full_text = []
    for i, page in enumerate(tqdm(pages, desc=f"OCR {pdf_path.name}")):
        img_path = img_dir / f"page_{i+1:04d}.jpg"
        page.save(str(img_path), "JPEG")
        page_text = ocr_image(img_path, processor, model)
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


def process_files(files: dict, processor, model) -> list[dict]:
    results = []
    for pdf in files["pdfs"]:
        txt_path = Path(DIRS["text"]) / (pdf.stem + ".txt")
        if txt_path.exists():
            log.info(f"Skipping OCR (cached): {pdf.name}")
            text = txt_path.read_text(encoding="utf-8")
        else:
            text = ocr_pdf(pdf, processor, model)
            save_text(text, pdf)
        results.append({"source": str(pdf), "filename": pdf.name, "text": text})

    for img in files["images"]:
        txt_path = Path(DIRS["text"]) / (img.stem + ".txt")
        if txt_path.exists():
            log.info(f"Skipping OCR (cached): {img.name}")
            text = txt_path.read_text(encoding="utf-8")
        else:
            text = ocr_image(img, processor, model)
            save_text(text, img)
        results.append({"source": str(img), "filename": img.name, "text": text})

    return results


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
  "summary": "2-4 sentence factual summary of what this document contains",
  "document_type": "one of: [sighting_report, government_memo, witness_testimony, research_report, news_article, investigation_report, correspondence, unknown]",
  "event_date": "ISO date string if determinable (YYYY-MM-DD), or date range (YYYY/YYYY), or null",
  "event_location": {{
    "country": "string or null",
    "region": "state/province/region or null",
    "city": "string or null",
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
  "clean_text": "The document text cleaned of OCR artifacts, headers/footers, page numbers, and formatting noise. Preserve all substantive content.",
  "qa_pairs": [
    {{
      "question": "A natural question a researcher or journalist would ask about this document",
      "answer": "A thorough, factual answer grounded strictly in the document text",
      "question_type": "one of: [factual, analytical, contextual, comparative]"
    }}
  ]
}}

Generate between 3 and 8 qa_pairs depending on document richness. Questions should cover:
- What happened / what is described
- Who was involved
- When and where
- What evidence or corroboration exists
- What the significance of the document is

Document text:
{text}"""


def enrich_document(doc: dict, llm_client) -> dict:
    """
    Pass document text through Claude to extract structured metadata
    and generate Q&A pairs for both RAG and future model training.
    """
    cache_path = Path(DIRS["enriched"]) / (Path(doc["filename"]).stem + ".json")
    if cache_path.exists():
        log.info(f"Skipping enrichment (cached): {doc['filename']}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    log.info(f"Enriching: {doc['filename']}")

    # Truncate very long texts to avoid huge token bills
    # Enrichment works on a representative sample; full text goes to chunking
    text_sample = doc["text"][:12000]

    try:
        response = llm_client.messages.create(
            model=LLM_MODEL,
            max_tokens=4096,
            system=ENRICHMENT_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": ENRICHMENT_USER_PROMPT.format(text=text_sample),
            }],
        )
        raw = response.content[0].text.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        enriched = json.loads(raw)

    except (json.JSONDecodeError, Exception) as e:
        log.warning(f"Enrichment failed for {doc['filename']}: {e}. Using defaults.")
        enriched = _default_enrichment(doc["filename"])

    # Attach source info
    enriched["source"]       = doc["source"]
    enriched["filename"]     = doc["filename"]
    enriched["ingested_at"]  = datetime.utcnow().isoformat()

    # Cache enriched JSON to disk
    cache_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

    # Append Q&A pairs to the training JSONL dataset
    _append_training_data(enriched)

    return enriched


def _default_enrichment(filename: str) -> dict:
    """Fallback enrichment if LLM call fails."""
    return {
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
        "clean_text": "",
        "qa_pairs": [],
    }


def _append_training_data(enriched: dict):
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
        f"Document text:\n{enriched.get('clean_text', '')[:3000]}"
    )

    with open(training_path, "a", encoding="utf-8") as f:
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


def enrich_all(docs: list[dict], llm_client) -> list[dict]:
    enriched_docs = []
    for doc in tqdm(docs, desc="LLM enrichment"):
        enriched = enrich_document(doc, llm_client)
        enriched_docs.append({**doc, **enriched})
    log.info(f"Enrichment complete for {len(enriched_docs)} documents.")
    return enriched_docs


# ─────────────────────────────────────────────
# STEP 4 — CHUNK
# ─────────────────────────────────────────────

def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Chunk using clean_text if available (LLM-cleaned), else raw OCR text.
    All enriched metadata is attached to every chunk so filtering works
    at query time without needing a separate metadata store.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = []
    for doc in docs:
        text_to_chunk = doc.get("clean_text") or doc.get("text", "")
        if not text_to_chunk.strip():
            log.warning(f"Empty text for {doc['filename']}, skipping.")
            continue

        splits = splitter.split_text(text_to_chunk)
        for i, split in enumerate(splits):
            chunks.append({
                "text":                 split,
                "source":               doc["source"],
                "filename":             doc["filename"],
                "chunk_id":             i,
                # Enriched fields — all stringified for ChromaDB compatibility
                "summary":              doc.get("summary", ""),
                "document_type":        doc.get("document_type", "unknown"),
                "event_date":           doc.get("event_date") or "",
                "country":              (doc.get("event_location") or {}).get("country") or "",
                "time_period":          doc.get("time_period", "unknown"),
                "classification_level": doc.get("classification_level", "unknown"),
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

    texts     = [c["text"] for c in chunks]
    ids       = [f"{Path(c['source']).stem}_chunk_{c['chunk_id']}" for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    log.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

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


def mark_failed(progress: dict, zip_path: Path, error: str):
    progress["failed"][zip_path.name] = {
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

    # Only load heavy models if there is actual work to do
    processor, ocr_model = load_ocr_model()
    embedder             = load_embed_model()
    collection           = load_vectordb()
    llm_client           = load_llm_client()

    for zip_path in pending:
        log.info(f"Processing: {zip_path.name}")
        try:
            extracted_dir = extract_zip(zip_path)
            files         = collect_files(extracted_dir, prefer="pdf")
            docs          = process_files(files, processor, ocr_model)
            docs          = deduplicate_texts(docs, threshold=0.85)
            enriched_docs = enrich_all(docs, llm_client)
            chunks        = chunk_documents(enriched_docs)
            embed_and_store(chunks, embedder, collection)

            mark_completed(progress, zip_path, doc_count=len(docs))
            log.info(f"Done: {zip_path.name} ({len(docs)} docs, {len(chunks)} chunks)")

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
