# eval_output/aemo/ — AEMO/NEM Utility-Scale Evaluation Results

Contains all evaluation outputs for the AEMO battery trading environment.

## Subdirectories

- **`notebook/`** — Static images from AEMO notebooks (e.g., dispatch replay visualizations)
- **`baseline/`** — AEMO baseline policy comparisons (CSV metrics + evaluation SVGs)
- **`cache_prewarm/`** — AEMO data cache prewarming manifests (non-graphic)
- **`autoresearch/`** — All outputs from the autoresearch program's held-out evaluator runs
  - `comparison_plots/` — Combined comparison SVGs across policies
  - `full/` — Canonical full evaluator runs
  - `sweeps/` — Optimizer and hyperparameter sweep evaluator runs
