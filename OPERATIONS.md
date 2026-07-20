# UAP Pipeline — Operations Guide

A practical runbook for the discover → download → ingest → query pipeline.
For architecture/source-catalog detail see [README.md](README.md) and [SOURCES.md](SOURCES.md).

---

## 1. What it does

```
discover ──▶ sources.json + SOURCES.md      15 scrapers enumerate every known UAP source
download ──▶ data/downloads/<source>/        resumable, checksummed, ledgered bulk fetch
ingest   ──▶ OCR / text-layer ─▶ Claude enrich ─▶ chunk ─▶ embed ─▶ ChromaDB
query    ──▶ semantic search with metadata filters
```

- **Discover** hits live sites (spoofed browser UA, 0.5 s/host delay) and writes a manifest. Best-effort: a dead site logs a warning and the run continues.
- **Download** pulls each manifest resource with HTTP-Range resume, exponential backoff, and SHA-256 verification where published. Skips `video`/`html`/`torrent` by default.
- **Ingest** OCRs image-only scans on the GPU (GLM-OCR); born-digital PDFs take a **pypdf text-layer fast path** and skip the GPU entirely. Documents get Claude metadata enrichment + Q&A generation; `.txt/.csv/.json/.html` are ingested text-native (no OCR, no enrichment).
- Everything is **resumable** — kill and re-run; completed work is skipped via SHA-tracked ledgers.

---

## 2. One-time setup

```bash
cd /apps/uap_pipeline
python3 -m venv .venv && source .venv/bin/activate

# PyTorch first, matching your CUDA (cp314 wheels exist on cu128):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# transformers from source (required by the GLM-OCR model):
git clone --depth 1 https://github.com/huggingface/transformers.git /apps/transformers
pip install -e /apps/transformers

pip install -r requirements.txt          # everything else

export ANTHROPIC_API_KEY=sk-ant-...       # needed only for the enrich step
```

**System dep:** poppler (`pdftoppm`) for PDF→image rendering — `sudo pacman -S poppler` / `apt install poppler-utils`.

**This environment:** venv on **Python 3.14.5**, torch **2.11+cu128**, GPU **RTX 3070 Ti Laptop, 8 GB VRAM** (the OCR bottleneck).

---

## 3. Running it

### One-shot
```bash
python pipeline.py all                    # discover --scrape ─▶ download ─▶ ingest (EVERYTHING)
```
A full run is **hundreds of GB and days of OCR** — scope it (see below) before using bare `all`.

### Stage by stage (recommended)
```bash
python pipeline.py discover               # offline: static registry only, no scraping
python pipeline.py discover --scrape      # live crawl → sources.json + SOURCES.md
python pipeline.py download               # fetch everything in the manifest
python pipeline.py ingest                 # OCR/enrich/embed data/downloads/
python pipeline.py status                 # download + ingestion progress report
python pipeline.py query "1976 Tehran incident" --official --n 8
```

### Scoped first run (a few GB, bounded cost)
```bash
python pipeline.py all \
  --sources wargov_pursue,nsa_ufo,odni,aaro,nuforc \
  --max-size-gb 3
```

---

## 4. Command / flag reference

| Command | Purpose |
|---|---|
| `discover` | build `sources.json` + `SOURCES.md` (add `--scrape` for live enumeration) |
| `download` | download every manifest resource that passes the filters |
| `ingest` | OCR/enrich/embed the downloaded corpus |
| `all` | one-shot: `discover --scrape` → `download` → `ingest` |
| `status` | download + ingestion progress report |
| `query "..."` | semantic search over the vector DB |

**Shared filters** (any stage): `--sources a,b,c` · `--manifest-dir .` · `--force` (redo completed work)

**Discover:** `--scrape` · `--cia-max-pages 150` · `--blackvault-max-pages 25` · `--afu-max-files 2000` · `--crawl-max-pages 300`

**Download:** `--include-videos` (large!) · `--max-size-gb N` (skip resources whose size hint exceeds N)

**Ingest:** `--downloads-dir ./data/downloads` · `--no-enrich` (skip Claude — no API key needed)

**Query:** `--n 5` (results) · `--type <document_type>` · `--period <time_period>` · `--official` · `--radar`

> Note: `--sources` is accepted by `ingest` but ignored — ingest processes the whole `--downloads-dir` tree. Restrict at download time instead.

### Scoped AFU drains

`--afu-dirs` restricts the AFU crawl to specific directories of files.afu.se —
use a separate `--manifest-dir` so the scoped manifest never clobbers the main
one (the ledger `data/downloads.json` is shared and append-only either way).

The UFO Newsclipping Service drain (492 monthly issues, 1969–2011 — local-press
UFO stories worldwide, the primary source for "reported only in the county
paper" lore; first drained 2026-07-20):

```bash
python pipeline.py discover --scrape --sources afu_se \
  --afu-dirs "Magazines/United States/UFO Newsclipping Service" \
  --afu-max-files 600 --manifest-dir data/manifest_ncs
python pipeline.py download --sources afu_se --manifest-dir data/manifest_ncs
```

Nearby shelves worth the same treatment later: `Magazines/United States/APCIC
clipping service`, the `Clippings/` tree (US/UK/Sweden), and the state-MUFON
newsletter runs under `Magazines/United States/MUFON *`.

### Post-publish backfills — ALWAYS re-run after pg_publish

`pg_publish.py` regenerates `corpus.chunks` from the build store, which
discards any metadata applied directly to Postgres. Currently that means the
NUFORC per-report URLs. After every publish (dev AND prod):

```bash
python backfill_nuforc_urls.py --apply          # idempotent, ~8 min
```

`corpus.admin_areas` / `admin_area_towns` / `admin_area_adj` are published by
`build_admin_areas.py`, not pg_publish, and survive the swap untouched — but
if `corpus.locations` gains/loses rows, re-run `build_admin_areas.py --apply`
so town membership stays aligned.

---

## 5. Sources covered

24 sources across five categories (full detail in SOURCES.md). Highlights:

- **U.S. government** — war.gov PURSUE releases (per-file SHA-256), NARA bulk (RG 615 / Blue Book), CIA FOIA reading room, FBI Vault, NSA, AARO, ODNI, DoW OIG, Navy.
- **International** — GEIPAN (France CSV), UK MoD, Canada LAC, Australia NAA, Brazil, New Zealand.
- **Community** — The Black Vault, Internet Archive, NICAP, AFU.se, Majestic Documents, Blue Book Archive.
- **Structured datasets** — NUFORC (140k+ geocoded reports), ufo_data chronology.
- **Mirrors** — GitHub redundancy for PURSUE + NUFORC tooling.

Add a source: `Source(...)` in `sources.py`, an optional scraper in `scrapers.py` (register in `SCRAPERS`), then `discover --scrape --sources your_id`.

---

## 6. Outputs & layout

```
sources.json / SOURCES.md          manifest (machine + human) — regenerated by `discover`
data/downloads/<source>/           raw corpus
data/downloads.json                download ledger (per-URL: path, sha256, bytes)
data/vectordb/                     ChromaDB collection "uap_documents"
data/progress.json                 ingest ledger (per-file sha256 + doc counts)
data/enriched/<doc>.json           per-document Claude metadata
data/training/uap_qa_dataset.jsonl fine-tuning Q&A dataset (by-product)
data/text/ data/images/            OCR intermediates (cached, skip re-OCR)
```

Code map: `pipeline.py` (CLI) · `sources.py` (registry) · `scrapers.py` (discovery) · `manifest.py` (manifest gen) · `downloader.py` (bulk fetch) · `main.py` (OCR/enrich/chunk/embed/query).

---

## 7. Operational notes & gotchas

- **`discover` overwrites** the committed `sources.json` / `SOURCES.md`. `git checkout` them to restore.
- **GPU is the bottleneck.** 8 GB VRAM + GLM-OCR fp16 @ 200 DPI. Prefer born-digital PDFs (text-layer fast path); big image-only scans are slow and can OOM.
- **Enrichment cost** scales with document count (one Claude call per doc, ~12k chars sampled). Use `--no-enrich` for a metadata-free build.
- **Resumable everywhere.** Re-runs only touch new/changed material (SHA-256 tracked). `--force` overrides.
- **Downloader UA** matches the scraper's browser UA — some gov WAFs 403 non-browser agents.
- **Known upstream gap:** ~25 war.gov "FBI Photo" images 403 because the manifest lists a stale `.../release_1/*.png` path war.gov no longer serves (the live docs are under `.../release_03/documents/`). These are FBI photos — backfill via the `fbi_vault` source. Not a WAF/UA issue; no client-side fix.
- **Videos excluded** by default (`--include-videos` to keep). HTML and torrents also excluded from download.

---

## 8. Analytics layer

A structured events store (DuckDB) unifying every sighting/event across
sources, with a GeoNames location dimension, HDBSCAN hotspot/wave clustering,
and corpus statistics. Rebuild order:

```bash
.venv/bin/python analytics_build.py       # data/analytics.duckdb: unified `events` table
                                          #   (UFOSINT 618k + NUFORC 76k + corpus enrich_v2)
.venv/bin/python analytics_locations.py   # GeoNames `locations` dimension; canonical
                                          #   city→region→country + geocode coord-less events
.venv/bin/python analytics_cluster.py     # geo_cluster hotspots + per-decade wave_cluster
.venv/bin/python analytics_stats.py       # stats_* tables + ANALYTICS.md report
.venv/bin/python analytics_index_meta.py  # denormalize cluster/location values into the
                                          #   build index chunk metadata, then publish
.venv/bin/python pipeline.py publish
```

GeoNames inputs live in `data/geonames/` (cities500, admin1, countryInfo).
After publish, the retriever's `filters` accept the derived fields, e.g.
`{"geo_cluster": 944}` or `{"loc_region": "New Mexico", "wave_cluster": "1960s:12"}`.

**Serving store is Postgres/pgvector** (`corpus` schema in the uap-api DB,
synced by `pg_publish.py`, which `pipeline.py publish` runs as its final
step — `--no-pg` to skip). The retriever does HYBRID retrieval there:
dense pgvector cosine + two tsvector arms (`simple` preserves exact tokens
like callsigns — "LACY 17" — and case numbers; `english` adds stemming),
fused with reciprocal-rank fusion. Chroma (`data/vectordb`) remains the
build-side store the ingest pipeline writes; if `PG_DSN` is unset the
retriever falls back to dense-only search on the mounted Chroma release.
Chunk quality: `textqc.chunk_junk_reason` gates every chunk-producing path
(degenerate OCR repetition / alphabet-free shred); `purge_junk_chunks.py`
scans and cleans the build index retroactively.

---

## 9. Query examples

```bash
python pipeline.py query "what happened at Roswell"
python pipeline.py query "radar-confirmed military encounters" --radar --n 10
python pipeline.py query "nuclear facility overflights" --type government_memo --period 1947_1969 --official
```

Metadata filters map to ChromaDB `where` clauses: `document_type`, `time_period`, `official_source`, `radar_confirmation`.
```
