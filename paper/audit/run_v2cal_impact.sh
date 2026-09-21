#!/usr/bin/env bash
# Provenance audit (Phase 1): run the canonical market-impact gate with the
# shipped physics-v2 v2cal checkpoint. The Aug/Sep impact gate artifacts only
# cover the v1 fullcorpus checkpoint (stagec_h3h1_auto / stagec_fix_20260913),
# so this closes the missing v2cal impact evidence for report.md §8.2.0.
set -euo pipefail
cd /home/victoru/rescued/energydecision
export PYTHONUNBUFFERED=1

python3 scripts/phase3_impact_eval.py \
  --impact-config configs/impact_benchmark.json \
  --checkpoint models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt \
  --config models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json \
  --label stagec_v2cal_auto \
  --rtg-mode auto
