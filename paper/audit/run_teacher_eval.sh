#!/usr/bin/env bash
# Stage-A teacher (waypoint DT + honest SDP executor, deg calibration 0.12) on
# the four extended surfaces, for the retention computation.
set -euo pipefail
cd /mnt/rescue-hdd/energydecision
export PYTHONUNBUFFERED=1
export AEMO_CACHE_ONLY=1

MANIFEST=models/aemo/dt/soc_waypoint_dt_loss_surface_manifest.json
WP=models/aemo/dt/soc_waypoint_dt_best.pt
for pair in \
  "soc_sdp_standard_year:teacher_best_standard_year" \
  "soc_sdp_dispatch_year:teacher_best_dispatch_year" \
  "soc_sdp_expanded_full:teacher_best_expanded_full" \
  "soc_sdp_2025_fresh:teacher_best_2025_fresh"
do
  cfg="${pair%%:*}"
  out="${pair##*:}"
  echo "==================== ${cfg} -> ${out} ===================="
  python3 scripts/autoresearch_evaluator.py \
    --surface-manifest-path "${MANIFEST}" \
    --model-path "${WP}" \
    --evaluation-config "configs/aemo_autoresearch_evaluator.${cfg}.json" \
    --output-dir "eval_output/physics_v2/${out}"
done
echo "TEACHER_EVAL_DONE"
