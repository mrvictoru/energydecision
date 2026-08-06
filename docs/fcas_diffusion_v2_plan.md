# Phase 6 v2 — Conditional Diffusion for Synthetic FCAS Prices

Self-contained implementation plan for the Phase 6 **v2** generator (new PR, branched
off main after PR #33). v1 (HMM/copula) and the evaluation harness already live on
main: `src/synthetic_fcas.py`, `src/fcas_generator_eval.py`, `scripts/eval_fcas_generator.py`.
Full context + diary: `docs/market_impact_plan.md` → "Phase 6 v2 design (handoff spec)".

## Implementation status (this PR)

The repository now contains an initial v2 implementation:

- `src/synthetic_fcas.py`
  - keeps the existing `FCASRegimeCopulaGenerator` v1 baseline,
  - adds `FCASDiffusionGenerator`,
  - adds shared helpers for service caps, conditioning features, windowing, and
    target transforms,
  - adds checkpoint helpers `save()` / `load()` for the diffusion model.
- `scripts/eval_fcas_generator.py`
  - now supports `--generator v1|v2`,
  - runs the same same-period and cross-period harness for either generator,
  - writes generator-specific output JSON (`generator_eval_v1.json` or
    `generator_eval_v2.json`).
- `tests/test_synthetic_fcas.py`
  - adds a CPU smoke test for the diffusion generator interface and caps,
  - adds a unit test for the lagged RRP-spike feature.

This is an **implementation landing**, not a completed empirical result: the
acceptance-gate runs on the real H1/H2 datasets still need to be executed in
the recommended Distrobox environment.

## Latest acceptance run (2026-08-06)

Executed in `energydecision-gpu` with working CUDA/PyTorch:

```bash
python3 scripts/eval_fcas_generator.py --generator v2 --device cuda
```

This used the evaluator's current **calendar-2024** default global train set:

```text
SA1:2024-01-01:2024-07-01
NSW1:2024-01-01:2024-07-01
QLD1:2024-01-01:2024-07-01
SA1:2024-07-01:2025-01-01
NSW1:2024-07-01:2025-01-01
QLD1:2024-07-01:2025-01-01
```

Result: **same-period gate FAIL**.

Per-region same-period tail KS minimum p-values:

- `SA1_samesplit`: `3.35e-13`
- `NSW1_samesplit`: `5.43e-41`
- `VIC1_samesplit`: `3.98e-43`

Key failure pattern from the report:

- Tail KS is far below the `>= 0.05` gate in all three regions.
- Spike co-occurrence is pathological in multiple cells (often `1.0` for
  synthetic within-family and cross-family spike coupling).
- Discriminator AUC remains too high (`~0.79` to `~0.83` on same-period splits).
- Lag-1 ACF error is also above the `<= 0.10` gate.

Interpretation:

- The current training/sampling formulation is **not yet learning calibrated
  spike structure**. The generated series appears to over-synchronize spike
  events and under-match the held-out tail shape.
- Expanding the global train set to a full calendar year **did not by itself**
  solve the gating problem.

Artifacts:

- Report written to `eval_output/phase6_fcas/generator_eval_v2.json`.

## Second iteration run (2026-08-06, epsilon-prediction sampler)

Implemented a second generator iteration before rerunning the harness:

- switched the diffusion loss from direct clean-target regression to
  **epsilon prediction**,
- reconstructed `x0` from predicted noise during sampling,
- used a DDIM-style update with configurable `--sample-eta`.

Executed in `energydecision-gpu` with:

```bash
python3 scripts/eval_fcas_generator.py \
  --generator v2 \
  --device cuda \
  --epochs 12 \
  --sample-steps 48 \
  --sample-eta 0.05
```

Result: **same-period gate still FAIL**.

Per-region same-period tail KS minimum p-values:

- `SA1_samesplit`: `3.34e-13`
- `NSW1_samesplit`: `4.80e-26`
- `VIC1_samesplit`: `2.08e-25`

What improved a bit:

- Same-period discriminator AUC moved down from roughly `0.79–0.83` to roughly
  `0.64–0.72`.
- RMSE / MAE improved slightly in several cells.

What did **not** improve materially:

- Tail KS still misses the gate by many orders of magnitude.
- Synthetic spike co-occurrence is still pathological, with many
  within-family / cross-family synthetic values pinned at `1.0`.
- Lag-1 ACF error remains well above target.

Interpretation of the second run:

- The failure is **not primarily a sampler-formulation bug anymore**. Switching
  to epsilon prediction helped realism somewhat, but the dominant error pattern
  persists.
- The stronger suspicion now is that the current single-stage joint diffusion
  setup is not learning the **sparse spike-event structure** well enough from
  the available conditioning signal, so it falls back to overly synchronized
  multi-service spike bursts.

Recommended next move:

1. Stop spending cycles on small hyperparameter sweeps alone.
2. Move to a **hybrid / two-stage formulation**, for example:
   - stage A: explicit spike-probability model per service/family (or a shared
     spike-structure head),
   - stage B: conditional generator for magnitudes given the spike state.
3. Alternatively, keep diffusion for magnitudes but add an explicit auxiliary
   spike classification loss / head so the model is directly trained on the
   p99-style event structure that the harness scores.
4. If staying single-stage, consider autoregressive window conditioning or
   previous-window latent carryover before broader sweeps.

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

- Train:
  - **Original plan:** cached 5-min processed parquets, H1 2024, SA1 + NSW1 +
    QLD1 (~52k rows/region).
  - **Current evaluator default:** full-year 2024 across the same train regions
    by concatenating H1 2024 + H2 2024 slices for SA1 + NSW1 + QLD1.
  - No NEMOSIS fetches needed (`AEMO_CACHE_ONLY=1`).
- Historical-extension note:
  - The evaluator now accepts arbitrary `--train-spec REGION:START:END`
    arguments and can slice from larger cached parquet ranges.
  - On this machine, **H2 2023 is not uniformly available** across the current
    train trio: SA1 and VIC1 have late-2023 coverage, but NSW1 / QLD1 cached
    history currently stops at 2023-04-01.
  - Because of that, the landed default uses **calendar 2024** as the clean
    whole-year representation across all three train regions.
- Target channels (T × 9): `[RRP] + 8×FCAS`.
  - FCAS channels use **log1p space** after clipping to the service cap
    (16,600 contingency / 999 regulation).
  - **Implementation note:** real processed `RRP` values can be negative
    (for example SA1 H1 2024 reaches about `-999.999`), so the landed code uses
    **signed log1p** for `RRP` instead of plain log1p.
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

### Landed architecture details

- The current model is a plain-PyTorch 1D temporal U-Net with:
  - timestep sinusoidal embeddings,
  - conditioning by channel concatenation,
  - timestep-conditioned adaptive scaling/shifting inside residual blocks,
  - deterministic DDIM-style sampling over a reduced timestep schedule.
- The training loss currently predicts the clean normalized target window
  directly (rather than epsilon) and applies the extra tail weight on the FCAS
  channels above the configured quantile threshold.
- Long sequences are generated window-by-window with overlap crossfade in
  `FCASDiffusionGenerator.sample()`.

## Training

- 2080 Ti (22 GB) — fits comfortably; a few hours of GPU.
- Use the telemetry wrapper pattern from AGENTS.md
  (`scripts/run_full_learning_baseline.sh`) for long runs; only shut down after
  `SAFE_TO_SHUTDOWN.txt` exists.

### Current runnable entrypoint

From the repo root inside the recommended Distrobox:

```bash
python3 scripts/eval_fcas_generator.py --generator v2
```

This now defaults to the **calendar-2024** global train set:

```text
SA1:2024-01-01:2024-07-01
NSW1:2024-01-01:2024-07-01
QLD1:2024-01-01:2024-07-01
SA1:2024-07-01:2025-01-01
NSW1:2024-07-01:2025-01-01
QLD1:2024-07-01:2025-01-01
```

Custom historical slices are supported, for example:

```bash
python3 scripts/eval_fcas_generator.py \
  --generator v2 \
  --train-spec SA1:2023-07-01:2023-12-01 \
  --train-spec VIC1:2023-07-01:2023-12-01 \
  --train-spec SA1:2024-01-01:2024-07-01 \
  --train-spec NSW1:2024-01-01:2024-07-01 \
  --train-spec QLD1:2024-01-01:2024-07-01
```

That mixed 2023/2024 setup is valid mechanically, but note the region-coverage
imbalance above.

Useful tuning knobs exposed by the script:

```bash
python3 scripts/eval_fcas_generator.py \
  --generator v2 \
  --epochs 8 \
  --batch-size 32 \
  --diffusion-steps 128 \
  --sample-steps 32 \
  --base-channels 64 \
  --window-size 288 \
  --stride 12 \
  --overlap 48
```

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

## Handoff notes for the next agent session

What is done:

- v2 code has been added and is wired into the existing evaluation harness.
- The evaluation loader now clips FCAS services with the intended per-service
  caps (`999` for regulation, `16,600` for contingency).
- A targeted smoke test exists for the new generator surface.

What still needs to be done:

1. Run `tests/test_synthetic_fcas.py` and any adjacent targeted tests inside the
   repo's Distrobox workflow.
2. Run the real-data harness with `python3 scripts/eval_fcas_generator.py --generator v2`
   in the GPU-capable Distrobox / torch-working environment.
3. Inspect `eval_output/phase6_fcas/generator_eval_v2.json` against the gates.
4. If the gates fail, iterate on:
   - model width/depth (`--base-channels`),
   - training length (`--epochs`),
   - diffusion/sample step counts,
   - overlap/blending behavior,
   - tail weighting / quantile threshold.
   - **and likely the core objective/sampling formulation**, since the latest
     run's failure mode is not a near-miss but a structural spike-calibration
     problem.
5. Only after gate performance is acceptable, move on to the downstream
   synthetic-data DT training loop.

Known implementation caveats:

- The current overlap blending is a practical first pass; it will not preserve
  multi-day dependence as well as an autoregressive window-conditioning scheme.
- The current implementation uses real historical RRP only for the **lagged**
  spike-conditioning feature; it does not feed the current real RRP into the
  model input while also generating synthetic current-bar RRP.
- In the current `energydecision` Distrobox on this machine, importing the
  installed PyTorch build can fail because it expects CUDA runtime libraries
  that are not present there. The code now degrades cleanly for v1/non-torch
  paths, but full v2 training/evaluation should be run in a working torch
  environment (for example the GPU-capable Distrobox).
- The current diffusion implementation predicts the clean normalized target
  window directly and then performs deterministic DDIM-style updates from that
  estimate. Based on the first acceptance run this was a leading suspect.
- After the second iteration, the code now uses epsilon prediction plus a
  DDIM-style update with `sample_eta`, but the spike-coupling failure still
  persists. That points toward a broader modeling mismatch rather than just the
  original sampler formulation.

## Out of scope for this PR

- Phase 7 (combine impact model + synthetic FCAS) — contingent on v2 passing gates.
- "Distributional conditioning" follow-up (generator spike-risk features into
  the DT as a revisit of §8.2.8) — documented in the plan; no code until v2
  passes the gates.
