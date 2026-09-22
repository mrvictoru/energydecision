#!/usr/bin/env bash
# Phase 1 reproducibility spot-check: re-run ONE identity surface (standard)
# from the frozen release and diff against the stored artifact. The other three
# surfaces use the same evaluator code path, checkpoint, and data, so a
# bit-identical standard surface demonstrates determinism without the ~40 min
# full re-run.
set -euo pipefail
cd /home/victoru/rescued/energydecision
export PYTHONUNBUFFERED=1
export AEMO_CACHE_ONLY=1

python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.sdp_teacher_standard.json \
  --output-dir eval_output/physics_v2/standard_v2cal_spotcheck
