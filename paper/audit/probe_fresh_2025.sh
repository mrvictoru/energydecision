#!/usr/bin/env bash
set -euo pipefail
cd /home/victoru/rescued/energydecision
export PYTHONUNBUFFERED=1 AEMO_CACHE_ONLY=1
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.sdp_teacher_2025_fresh.json \
  --output-dir eval_output/physics_v2/2025_v2cal_fresh
