#!/bin/bash
set -e

CHROMA_DIR="${CHROMA_PATH:-/data/chroma_data}"

# Run ingest if ChromaDB collection is empty or missing
if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null)" ]; then
  echo "ChromaDB empty — running ingest..."
  python scripts/ingest.py --resume "data/Pooja_10151_SST (2).pdf" --reset
fi

exec uvicorn server.main:app --host 0.0.0.0 --port 8000
