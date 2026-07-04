# UAP Pipeline

One-shot pipeline that scours the internet for every publicly available
UAP/UFO document source, downloads the corpus, OCRs it, enriches it with
Claude, and builds a queryable vector database — plus a Q&A fine-tuning
dataset as a by-product.

```
discover ──> sources.json + SOURCES.md   (the downloadable-resource manifest)
download ──> data/downloads/<source>/    (resumable, checksummed, ledgered)
ingest   ──> OCR/text-layer -> LLM enrichment -> chunk -> embed -> ChromaDB
query    ──> semantic search with metadata filters
```

## One-shot

```bash
pip install -r requirements.txt          # see notes inside for torch/poppler
export ANTHROPIC_API_KEY=sk-ant-...      # or use --no-enrich

python pipeline.py all
```

That single command:

1. **Discovers** — runs 15 live scrapers across every cataloged source and
   writes `sources.json` (machine manifest) + `SOURCES.md` (human catalog).
2. **Downloads** — pulls every manifest resource with resume, retries,
   SHA-256 verification (where published), and a ledger so re-runs are
   incremental. Videos are skipped by default (`--include-videos` to keep).
3. **Ingests** — ZIPs, loose PDFs, images, CSV/JSON datasets, and HTML.
   PDFs with a usable embedded text layer skip GPU OCR entirely; image-only
   scans go through GLM-OCR. Documents get Claude metadata enrichment + Q&A
   generation; structured datasets are chunked/embedded directly.

Everything is resumable: kill it anytime and re-run — completed work is
skipped via SHA-tracked ledgers (`data/progress.json`, `data/downloads.json`).

## What's covered

24 sources across five categories (full detail in [SOURCES.md](SOURCES.md)):

| Category | Sources |
|---|---|
| **U.S. government** | Department of War **PURSUE releases** (war.gov/UFO, with per-file SHA-256 manifests), **NARA** UAP bulk downloads (RG 615 + Project Blue Book), **CIA** FOIA reading room, **FBI Vault** (UFO 16 parts, Majestic 12, Roswell, Guy Hottel), **NSA**, **AARO**, **ODNI** assessments, **DoW OIG**, Navy/NAVAIR |
| **International** | **GEIPAN** (France, CSV open data), **UK MoD** files, **Canada** LAC, **Australia** NAA, **Brazil** Arquivo Nacional, **New Zealand** NZDF |
| **Community archives** | **The Black Vault** (incl. the 2 GB processed UFO Files Release ZIP and CIA CD-ROM collection), **Internet Archive** (complete Blue Book NARA microfilm + 10k case files + Canada/Australia/Brazil consolidated sets), **NICAP**, **AFU.se** (Sweden), Majestic Documents, Blue Book Archive |
| **Structured datasets** | **NUFORC** sighting reports (140k+, geocoded CSVs), richgel999/ufo_data chronology |
| **Mirrors** | 9 GitHub mirrors of the PURSUE releases and NUFORC tooling (redundancy when war.gov throttles) |

Sources marked `manual` in SOURCES.md have no bulk endpoint (e.g. UK
Discovery pay-per-download); their holdings arrive via a mirrored source
where one exists, and they're listed so nothing is silently omitted.

## Stage-by-stage usage

```bash
python pipeline.py discover              # offline: static registry only
python pipeline.py discover --scrape     # full live enumeration (thousands of files)
python pipeline.py download              # fetch the manifest
python pipeline.py ingest                # build the vector DB from downloads
python pipeline.py status                # download + ingestion progress
python pipeline.py query "1976 Tehran incident" --official --n 8
```

Useful flags:

| Flag | Effect |
|---|---|
| `--sources wargov_pursue,fbi_vault` | restrict any stage to specific sources |
| `--max-size-gb 5` | skip resources with a size hint above N GB |
| `--include-videos` | also download video evidence (large) |
| `--no-enrich` | skip Claude enrichment (no API key needed) |
| `--force` | redo completed work |
| `--blackvault-max-pages 999` | full Black Vault category crawl (default 25 pages) |
| `--afu-max-files 20000` | raise the AFU.se crawl cap (default 2000) |

Legacy entry points still work: `python main.py ./zips/` ingests ZIPs
directly, `python main.py --query "..."` queries.

## Expectations & scale

- A **full** run is hundreds of GB and days of OCR on a single GPU. For a
  first pass, try:
  `python pipeline.py all --sources wargov_pursue,nsa_ufo,odni,aaro,nuforc --max-size-gb 3`
- Discovery scrapers are polite (0.5 s/host delay) and **best-effort**: a
  dead site logs a warning into `SOURCES.md` and the run continues.
- war.gov PURSUE and NARA RG 615 are **rolling releases** — re-run
  `discover --scrape` + `download` periodically to pick up new tranches;
  everything is incremental.
- LLM enrichment cost scales with document count (one Claude call per
  document, ~12k chars sampled). Use `--no-enrich` for a metadata-free
  build, or restrict sources first.

## Layout

```
pipeline.py     orchestrator CLI (discover/download/ingest/all/status/query)
sources.py      master source registry (add new sources here)
scrapers.py     per-source discovery scrapers (add new scrapers here)
manifest.py     sources.json + SOURCES.md generation
downloader.py   resumable bulk downloader + ledger
main.py         ingestion core: OCR, enrichment, chunking, embedding, query
SOURCES.md      human-readable catalog  ─┐ regenerate with
sources.json    machine manifest         ─┘ `pipeline.py discover [--scrape]`
```

### Adding a source

1. Add a `Source(...)` to `sources.py` (with any known static URLs).
2. If it needs enumeration, write a scraper in `scrapers.py` returning
   `resource(...)` dicts and register it in `SCRAPERS`.
3. `python pipeline.py discover --scrape --sources your_id` to verify.
