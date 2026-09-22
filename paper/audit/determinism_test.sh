#!/usr/bin/env bash
set -euo pipefail
cd /home/victoru/rescued/energydecision
export PYTHONUNBUFFERED=1 AEMO_CACHE_ONLY=1
for tag in a b; do
  python3 scripts/autoresearch_evaluator.py \
    --surface-manifest-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json \
    --evaluation-config configs/aemo_autoresearch_evaluator.det_nsw1may_${tag}.json \
    --output-dir eval_output/physics_v2/det_nsw1may_${tag}
done
echo DET_TEST_DONE
