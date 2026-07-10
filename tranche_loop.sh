#!/usr/bin/env zsh
# Run download -> ingest -> publish tranches back-to-back until the manifest
# is drained (a download that finishes without hitting its --total-gb budget
# is the last slice). Logs each cycle to data/tranche<N>.log.
#
# Usage: ./tranche_loop.sh [start-tranche-number]  (default 7)

# Export ONLY the API key. Sourcing the whole api .env exported
# VECTORDB_HOST_DIR, which shadows the .env flip publish makes (shell env
# beats .env in docker compose) — the retriever then never got recreated.
export ANTHROPIC_API_KEY="$(grep '^ANTHROPIC_API_KEY=' /apps/uap-api/.env | cut -d= -f2-)"
cd "$(dirname "$0")"

SOURCES=internet_archive,afu_se
SLICE_GB=2.5
i=${1:-7}
MAX=25   # hard stop so a bug can never loop forever

# Download-only drain mode (rm data/DOWNLOAD_ONLY to restore normal loop):
# fetch EVERYTHING left in the manifest, no budget slices, no ingest/publish.
# Used while OCR is offloaded to cloud GPUs — ingest runs after the OCR'd
# text is synced back into data/text (ingest then skips OCR via that cache).
if [[ -f data/DOWNLOAD_ONLY ]]; then
  log=data/download_drain.log
  echo "=== DOWNLOAD-ONLY DRAIN START $(date -Iseconds) === ($SOURCES, no budget)" >> $log
  if .venv/bin/python pipeline.py download --sources $SOURCES >> $log 2>&1; then
    echo "DOWNLOAD DRAIN: complete $(date -Iseconds) — manifest empty, NOT ingested (waiting on cloud OCR)"
  else
    echo "DOWNLOAD DRAIN: FAILED — see $log"
    exit 1
  fi
  exit 0
fi

while (( i <= MAX )); do
  log=data/tranche$i.log
  echo "=== TRANCHE $i START $(date -Iseconds) === (auto-loop: download+ingest+publish $SOURCES, $SLICE_GB GB slice)" >> $log

  if ! .venv/bin/python pipeline.py download --sources $SOURCES --total-gb $SLICE_GB >> $log 2>&1; then
    echo "TRANCHE $i: download FAILED — stopping loop"; exit 1
  fi
  if ! .venv/bin/python pipeline.py ingest --sources $SOURCES >> $log 2>&1; then
    echo "TRANCHE $i: ingest FAILED — stopping loop"; exit 1
  fi
  if ! .venv/bin/python pipeline.py publish >> $log 2>&1; then
    echo "TRANCHE $i: publish FAILED — stopping loop"; exit 1
  fi

  echo "TRANCHE $i: complete $(date -Iseconds)"
  if ! grep -q "budget reached" $log; then
    echo "=== MANIFEST DRAINED — tranche $i downloaded everything remaining ===" | tee -a $log
    break
  fi
  (( i++ ))
done
echo "tranche loop finished at tranche $i"
