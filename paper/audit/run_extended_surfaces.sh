#!/usr/bin/env bash
# Phase 1 small-n fix: run the three extended identity surfaces with the shipped
# v2cal checkpoint. Sequential (single GPU). Canonical configs/artifacts are
# untouched; these write to new *_v2cal dirs and use fresh reference caches.
set -euo pipefail
cd /home/victoru/rescued/energydecision
export PYTHONUNBUFFERED=1
export AEMO_CACHE_ONLY=1

MANIFEST=models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json

for pair in \
  "standard_year:standard_year_v2cal" \
  "dispatch_year:dispatch_year_v2cal" \
  "expanded_full:expanded_full_v2cal"
do
  cfg="${pair%%:*}"
  out="${pair##*:}"
  echo "==================== ${cfg} -> ${out} ===================="
  python3 scripts/autoresearch_evaluator.py \
    --surface-manifest-path "${MANIFEST}" \
    --evaluation-config "configs/aemo_autoresearch_evaluator.sdp_teacher_${cfg}.json" \
    --output-dir "eval_output/physics_v2/${out}"
done
echo "ALL EXTENDED SURFACES COMPLETE"
