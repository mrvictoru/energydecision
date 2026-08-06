# Phase 6 v2 — Conditional Diffusion for Synthetic FCAS Prices

Self-contained implementation plan for the Phase 6 **v2** generator (new PR, branched
off main after PR #33). v1 (HMM/copula) and the evaluation harness already live on
main: `src/synthetic_fcas.py`, `src/fcas_generator_eval.py`, `scripts/eval_fcas_generator.py`.
Full context + diary: `docs/market_impact_plan.md` → "Phase 6 v2 design (handoff spec)".

## Goal

Learn `p(FCAS prices | exogenous market state)` and sample realistic FCAS price
paths that pass the existing harness gates. This unlocks unlimited FCAS-rich
training data for the DT and (later) Phase 7 (impact + synthetic combined).

## Why v2 (diffusion) and not v1

v1 transfers co-occurrence (0.34–0.38 vs real 0.39–0.40) and spike rates
(~0.005 error) to held-out periods but **fails the tail-value KS** on
same-period splits (p≈1e-22–1e-25): empirical-tail resampling replays rare
price-cap events (e.g. LOWER6SEC hit 16,600 in training; holdout max 317).
Decision: go **straight to conditional diffusion**; the v1.5 parametric-tail fix
is kept only as a fallback if diffusion training is blocked.

## Data

- Train: cached 5-min processed parquets, H1 2024, SA1 + NSW1 + QLD1 (~52k
  rows/region). No NEMOSIS fetches needed (`AEMO_CACHE_ONLY=1`).
- Target channels (T × 9): `[RRP] + 8×FCAS`, in **log1p space**, clipped at the
  service cap (16,600 contingency / 999 regulation) exactly as
  `scripts/eval_fcas_generator.py::load()` does.
- Conditioning channels: TOTALDEMAND, GEN_wind, GEN_solar, hour_sin/cos,
  day_sin/cos, and a **lagged RRP-spike indicator** (spike within the trailing
  12 bars — NOT the current bar, to avoid circularity with the jointly
  generated RRP).
- Windows: T=288 (24 h), stride ~12 → ~13k windows for training.

## Model

- **1D temporal U-Net DDPM** over the (T × 9) target channels, ~10–20M params.
- Sinusoidal timestep embedding; conditioning injected by channel concatenation
  + AdaLN on the time embedding.
- Plain PyTorch, hand-written DDPM/DDIM loop (~200 lines). **Do not add
  `diffusers` or other new dependencies** unless one is already installed.
- ~100–200 diffusion steps for training; DDIM 20–50 for sampling.
- Tail handling: log1p space + higher loss weight on the upper-tail quantiles
  of the 8 FCAS channels.

## Training

- 2080 Ti (22 GB) — fits comfortably; a few hours of GPU.
- Use the telemetry wrapper pattern from AGENTS.md
  (`scripts/run_full_learning_baseline.sh`) for long runs; only shut down after
  `SAFE_TO_SHUTDOWN.txt` exists.

## Acceptance gates (same-period split: fit Jan–Apr, holdout Apr–Jun per region)

Scored with the existing `src/fcas_generator_eval.py` harness:

| Metric | Gate |
|---|---|
| Tail KS p-value | ≥ 0.05 on all 8 services × 3 regions |
| Within-direction spike co-occurrence | within ±0.10 of real data |
| Spike-rate error | ≤ 0.01 |
| ACF lag1 absolute error | ≤ 0.10 (mean over services) |
| Discriminator AUC | ≤ 0.65 |

Cross-period H1→H2 is **reported, not gated** — even real-vs-real fails it
(p≈1e-32–1e-61) because the 2024 regime shift dominates.

## Long-episode generation (DT episodes are 4,032 steps / 14 days)

Generate window-by-window with a small overlap and linear crossfade blending,
then verify ACF continuity at the seams.
🐴 ceiling: overlap blending can lose multi-day dependence; upgrade path:
autoregressive conditioning where each window is generated conditioned on the
previous window's tail.

## Downstream DT validation

1. Swap synthetic RRP + FCAS columns into real exogenous frames → same schema.
2. Roll out policies (PPO, fcas_rule, dt_v2) via the `generate_fcas_dataset.py`
   machinery → assemble episode parquet.
3. Train DT via `scripts/pretrain_aemo_decision_transformer.py`.
4. Evaluate on the standard + dispatch-matched surfaces vs the real-data-trained
   DT baseline ($1,522/ep on the example evaluator).

**Success = synthetic-trained DT is within a reasonable band of the
real-data-trained DT** (the test is whether synthetic-only data carries the FCAS
signal, not necessarily beating it).

## Out of scope for this PR

- Phase 7 (combine impact model + synthetic FCAS) — contingent on v2 passing gates.
- "Distributional conditioning" follow-up (generator spike-risk features into
  the DT as a revisit of §8.2.8) — documented in the plan; no code until v2
  passes the gates.
