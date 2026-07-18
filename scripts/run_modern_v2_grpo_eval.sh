#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SURFACE_MANIFEST="${1:-models/aemo/dt/grpo_modern_single/grpo_surface_manifest.json}"
OUTPUT_DIR="${2:-eval_output/grpo_modern_final}"
EVAL_CONFIG="${3:-configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json}"

python3 src/autoresearch_evaluator.py \
  --surface-manifest-path "$SURFACE_MANIFEST" \
  --evaluation-config "$EVAL_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device auto
