#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="${1:-models/aemo/dt/grpo_modern_single}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 src/run_grpo_posttraining.py \
  --region NSW1 \
  --start-date 2024-01-01 \
  --end-date 2024-01-14 \
  --iterations 1 \
  --group-size 2 \
  --update-epochs 1 \
  --minibatch-size 4 \
  --baseline-eval-episodes 1 \
  --episode-hours 24 \
  --step-duration 0.083333 \
  --output-dir "$OUTPUT_DIR"
