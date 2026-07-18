#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${1:-eval_output/hf_modern_baseline}"
EVAL_CONFIG="${2:-configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json}"
MANIFEST_DIR="${3:-models/aemo/dt/hf_v2_modern}"

python3 scripts/create_hf_surface_manifest.py --output-dir "$MANIFEST_DIR"
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path "$MANIFEST_DIR/hf_modern_surface_manifest.json" \
  --evaluation-config "$EVAL_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device auto
