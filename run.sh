#!/usr/bin/env bash
set -euo pipefail

VCTK_DIR="data/VCTK-Corpus-0.92"
OUTPUT_DIR="results"
LIMIT="5"

# uv sync

uv run python eval.py \
    --vctk-dir  "$VCTK_DIR"   \
    --output-dir "$OUTPUT_DIR" \
    --limit $LIMIT

# echo ""
# echo "Results written to $OUTPUT_DIR/"
