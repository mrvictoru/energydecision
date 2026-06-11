# eval_output/ — Evaluation Artifacts

This directory contains outputs from all evaluation runs. Each subdirectory is an eval graphics output unless stated otherwise.

## Structure

```
eval_output/
├── README.md          ← This file
│
├── household/         ← Household environment evaluation results
│   ├── baseline/      ← Baseline comparison (Rule, SDP, MRDP, RL, DT, Oracle)
│   ├── dt_sensitivity/ ← DT RTG sensitivity study
│   ├── risk_metrics.csv
│   ├── pairwise_summary.csv
│   └── pairwise_significance_heatmap.svg
│
├── aemo/              ← AEMO/NEM utility-scale evaluation results
│   ├── notebook/      ← Static AEMO notebook outputs (dispatch replay images)
│   ├── baseline/      ← AEMO baseline comparisons
│   ├── cache_prewarm/ ← AEMO data cache prewarming manifests
│   └── autoresearch/  ← Autoresearch program evaluation runs
│       ├── comparison_plots/  ← Combined comparison SVGs
│       │   └── expanded/      ← 135-episode expanded comparison SVGs
│       ├── full/              ← Canonical full evaluator runs
│       │   ├── tuned_dt/              ← Original 4-ep tuned DT eval
│       │   ├── pretrain_dt/           ← Original 4-ep pretrain DT eval
│       │   ├── expanded_dt/           ← 135-ep full-pretrained DT
│       │   ├── expanded_pretrain/     ← 135-ep old pretrain DT
│       │   ├── resume_test_best/      ← Resume test best model eval
│       │   ├── rtg_calibrated/        ← RTG calibration eval
│       │   └── pilot_pretrain_baseline/ ← Pilot pretrain baseline eval
│       └── sweeps/           ← Optimizer and hyperparameter sweep evals
│
├── cache/             ← Non-graphic compute caches (NOT eval output)
│   └── reference_rollouts/  ← Cached reference policy rollouts
│
└── training/          ← Training monitor logs (NOT eval output)
    └── monitor/       ← Telemetry logs from DT training runs
```
