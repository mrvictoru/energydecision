# Household Workflow Guide

This is the main workflow guide for the household solar-battery track.

Use this document when you need:

- the end-to-end household workflow
- the main notebooks for data generation, RL training, and evaluation
- the canonical CLI entrypoint for household Decision Transformer training
- the expected artifact locations for logs, models, and evaluation output

If you only need environment mechanics, read [environment.md](environment.md). If you want the full household docs map, start with [README.md](README.md).

## Recommended Entry Points

- Environment and baseline sanity check: `notebooks/testrun.ipynb`
- Household log generation: `notebooks/test_simrun.ipynb`
- Household SB3 training: `notebooks/test_sb3train.ipynb`
- Canonical DT training: `scripts/pretrain_decision_transformer.py`
- Household evaluation: `notebooks/test_eval.ipynb`

## Standard Household Workflow

### 1. Prepare raw household data

Place the Ausgrid household CSV files under:

- `data/household/raw/`

The household preprocessing flow converts raw customer data into the schema expected by `SolarBatteryEnv`.

### 2. Generate baseline or rollout logs

Use `notebooks/test_simrun.ipynb` to:

- inspect transformed household data
- run rule-based or planning baselines
- write parquet logs under `data/household/logs/`

These logs are one of the main inputs for offline DT training.

### 3. Train online RL baselines if needed

Use `notebooks/test_sb3train.ipynb` to train SB3 policies and optionally export rollout logs.

Typical model outputs live under:

- `models/household/sb3/`

### 4. Train a household Decision Transformer

Use the canonical CLI surface:

```bash
python scripts/pretrain_decision_transformer.py \
  --data-dir data/household/logs \
  --patterns train_episode_01 train_episode_02 \
  --epochs 2 \
  --batch-size 6 \
  --lr 2e-5 \
  --save-path models/household/dt/dt_model.pt \
  --checkpoint-path models/household/dt/dt_model_checkpoint.pt \
  --loss-csv-path models/household/dt/dt_model_loss_history.csv
```

This is the shared DT trainer used across the repo. The household track usually differs from the AEMO track in:

- data source
- state and action dimensions
- artifact locations

### 5. Evaluate household policies

Use `notebooks/test_eval.ipynb` for notebook-driven comparison across:

- rule-based baselines
- planning baselines
- SB3 models
- Decision Transformer policies

Typical evaluation outputs live under:

- `eval_output/`

### 6. Build the synthetic diverse-household corpus

H1.5 recomposes complete normalized 5-minute days; it does not add row-wise
noise. The generator clusters real load profiles by season and weekday/weekend,
samples five explicit archetypes, injects optional EV/AC/pool blocks under a
60% daily-energy cap, scales the real solar curve, and rejects candidates that
fail any G1–G6 validation gate.

From the repository root in the GPU Distrobox:

```bash
python3 scripts/build_household_synth_corpus.py \
  --normalized-dir data/household/real/normalized \
  --output-dir data/household/synth \
  --episodes 1200 \
  --days-per-episode 7 \
  --seed 42
```

Each episode is an env-view-compatible parquet under
`data/household/synth/<archetype>/`, with `SolarGen` and `HouseLoad` stored as
kWh per 5-minute step and the day-ahead `FutureSolar`/`FutureLoad` columns.
`manifest.json` records the seed, source dates and clusters, archetype,
lambda scale, appliance parameters, solar and battery configuration, gate
metrics, split (`train`/`val`/`test`), and the real source dates reserved for
OOD evaluation. The real household remains the OOD surface of record and must
not be included in synthetic training data.

The H4.1 extension supports a balanced horizon/scenario matrix. Use
`--horizons 1w 2w 6m 2y` to cycle across one-week, two-week, six-month, and
two-year episodes while retaining the five archetypes, seasonal/day-type
sampling, solar/battery variation, and per-episode provenance. For example:

```bash
python3 scripts/build_household_synth_corpus.py \
  --output-dir data/household/synth_h4_1 \
  --episodes 240 --horizons 1w 2w 6m 2y --seed 20260830
```

The manifest records the horizon and degradation configuration. The generated
data provides held-out synthetic surfaces; real household segments remain the
primary OOD surface.

### 6a. Train a fresh modern SB3 baseline

The legacy PPO checkpoint is not a modern-data baseline. Train a new PPO
directly on the H4.1 corpus with parallel CPU environments:

```bash
python3 scripts/train_household_sb3.py \
  --corpus-dir data/household/synth_h4_1 \
  --output-dir models/household/sb3/h4_3 \
  --timesteps 250000 --n-envs 12 \
  --capacity-kwh 5 --max-flow-kw 3.3
```

The H4.3 pilot used the realistic tariff and matched 5 kWh/3.3 kW settings.
Its fresh PPO saved +$27/year on the five real OOD segments, below the
+$81/year rule baseline; this does not establish PPO as the preferred
household policy.

### 6b. Forecast-feature experiments

`FutureSolar` and `FutureLoad` are part of the 12-dimensional environment
observation. The current values are an honest 24-hour persistence forecast:
the same time slot from the preceding day, with a first-day fallback to the
current value. They are not a learned weather or load forecast. H4.2 compares
the trained policy with these channels preserved, zeroed, and shuffled before
considering causal rolling/seasonal forecast retraining. Real OOD segments
remain fixed across all modes.

The initial inference-only H4.2 ablation uses ten deterministic seven-day
windows (two per real segment) and the corrected H2 DT with J_t(soc). Annualized
savings were +$92.28 for persistence, +$92.88 with forecasts zeroed, and
+$95.78 with forecasts shuffled. These small differences show that the current
DT does not use the persistence channels beneficially. A stronger forecast must
be generated causally and included during matched policy retraining; replacing
inputs only at inference is not sufficient.

The no-forecast control has now been retrained from the same SDP-teacher corpus
with both forecast channels set to zero in every training observation
(`h4_2_no_forecast_8x512_ctx576`). On the same ten windows, evaluated with
zeroed channels, it saved +$232.32/year (95% CI for annualized bill:
$597–$1,172; n=10). This is a retrained-policy comparison, not directly
comparable to the earlier inference-only zeroing result.

Granite TTM-R3 is integrated as an **offline sidecar**, never as a dependency
of the simulator or main training container. `Containerfile.ttm` provides a
separate Python 3.12/Torch/CUDA environment, and the wrapper creates the
`energydecision-ttm` Distrobox with NVIDIA integration:

```bash
# Precompute a mirrored synthetic corpus with causal one-hour-ahead forecasts.
bash scripts/run_household_ttm_forecasts.sh \
  --synth-dir data/household/synth \
  --output data/household/synth_ttm \
  --device cuda --batch-size 512

# Precompute a timestamp-keyed real-OOD sidecar.
bash scripts/run_household_ttm_forecasts.sh \
  --normalized-dir data/household/real/normalized \
  --output data/household/real/household_ttm_forecasts.parquet \
  --device cuda --batch-size 512
```

The pinned `512-48-dec-512-r3` checkpoint uses 512 historical samples and
predicts 48 future samples. The environment columns use the 12th prediction
(one hour ahead at five-minute cadence). Each output records forecast issuance
and target timestamps, invalid warm-up rows, model revision, and forecast
quality. The full synthetic build improved solar/load MAE by 34.6%/12.2%;
the real-OOD sidecar improved them by 39.2%/17.5%.

Matched J_t(soc) DT evaluation on the fixed ten-window surface saved
+$381.76/year with TTM versus +$264.40/year without forecasts. The paired
TTM advantage was +$117.36/year (bootstrap 95% CI +$64.55–$163.82, 9/10
windows, one-sided Wilcoxon p=0.0029). This is supporting evidence only:
J_t(soc) uses future actuals when constructing its inference prompt.

The deployment-style control retrains all policies with
`--rtg-source constant`, identical architecture/optimizer/seed/data labels,
and `--stride 288`. Inference updates RTG only from realized rewards. The
shared fixed prompt is RTG=-2, selected from the training RTG median
(approximately -1.76), not from OOD policy performance:

```bash
python3 scripts/pretrain_decision_transformer.py \
  --surface-preset household_baseline \
  --data-dir data/household/dt \
  --patterns h4_2_ttm_sdp_train \
  --val-data-dir data/household/dt \
  --val-patterns h4_2_ttm_sdp_val \
  --split-policy explicit_validation \
  --context-length 576 --stride 288 \
  --n-block 8 --h-dim 512 --n-heads 8 --drop-p 0.15 \
  --batch-size 16 --epochs 5 --lr 3e-5 --seed 42 \
  --rtg-source constant --return-scale 1.0 \
  --action-loss-weight 0.999 --state-loss-weight 0.002 \
  --return-loss-weight 0.0001 --device cuda --amp-mode auto

python3 scripts/evaluate_household_ood_baselines.py \
  --dt-rtg-mode standard --dt-rtg-value -2 \
  --forecast-sidecar data/household/real/household_ttm_forecasts.parquet \
  --tariff realistic --window-days 7 --windows-per-segment 2 \
  --skip-reference-policies --skip-ppo --device cuda
```

Change the train/validation patterns and evaluation forecast input for the
matched persistence and no-forecast controls. On the same ten real-OOD
windows, annualized savings were:

| Forecast input | Savings vs no battery |
|---|---:|
| TTM-R3 | +$258.50/year |
| 24-hour persistence | +$216.74/year |
| No forecast | +$155.12/year |

TTM beat persistence by +$41.75/year (paired bootstrap 95% CI
+$16.56–$69.43, 9/10 windows, one-sided Wilcoxon p=0.0068) and no forecast
by +$103.37/year (95% CI +$78.23–$127.67, 10/10, p=0.0010). Persistence
also beat no forecast by +$61.62/year (95% CI +$33.96–$90.77, 8/10,
p=0.0049).

Prompt calibration matters: at the optimistic out-of-distribution RTG=0,
TTM saved -$4.21/year and no forecast saved +$38.53/year. Therefore use a
prompt justified from the training distribution and report it with every
result. The matched RTG=-2 experiment supports retaining offline TTM
forecasts in the observation pipeline; broader households and prompt
robustness belong to H4.4.

### 6c. H4.4 full-corpus forecast generalization (reproducible pipeline)

H4.2 was trained on the controlled H2 corpus (1,200 fixed seven-day,
5/10/20 kWh episodes). H4.4 repeats the matched three-way comparison on the
full horizon/scenario-diverse H4.1 corpus: 240 episodes, one per
archetype × season × battery-capacity × horizon cell (horizons `1w`, `2w`,
`6m`, `2y`; 55,860 episode-days), split 165 train / 35 val / 40 test with
the same 158 real source dates held out for OOD. Build it with:

```bash
python3 scripts/build_household_synth_corpus.py \
  --output-dir data/household/synth_h4_1 \
  --episodes 240 --horizons 1w 2w 6m 2y --seed 20260830
```

Precompute the causal TTM mirror (offline, isolated `energydecision-ttm`
Distrobox; 16.1M rows took ~3 h on the 2080 Ti at batch 2048):

```bash
bash scripts/run_household_ttm_forecasts.sh \
  --synth-dir data/household/synth_h4_1 \
  --output data/household/synth_h4_1_ttm \
  --device cuda --batch-size 2048
```

Generate matched SDP-teacher trajectories. The teacher optimizes against
actual solar/load, so action labels, rewards, and `rtg_value` are identical
across variants; only the stored forecast channels in the observations
differ:

```bash
# 24-hour persistence (corpus default channels)
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1 --split train \
  --forecast-mode persistence \
  --out data/household/dt/h4_4_persistence_sdp_train.parquet
# (repeat --split val)

# no forecast (both channels zeroed)
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1 --split train \
  --forecast-mode zero \
  --out data/household/dt/h4_4_no_forecast_sdp_train.parquet
# (repeat --split val)

# TTM (mirror corpus already carries causal one-hour-ahead channels)
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1_ttm --split train \
  --forecast-mode persistence \
  --out data/household/dt/h4_4_ttm_sdp_train.parquet
# (repeat --split val)
```

Train the three deployment-style standard-RTG DTs with the exact H4.2
recipe (state 12, act 1, 8×512, ctx 576, drop 0.15, batch 16, lr 3e-5,
5 epochs, seed 42, `--stride 288`, `--rtg-source constant`,
`--return-scale 1.0`, loss weights 0.999/0.002/0.0001). The shared
inference prompt RTG=-2 is again justified from the training RTG median
(-1.78 on the full corpus, vs -1.76 on H4.2), not from OOD rankings:

```bash
for variant in ttm persistence no_forecast; do
python3 scripts/pretrain_decision_transformer.py \
  --surface-preset household_baseline \
  --data-dir data/household/dt \
  --patterns h4_4_${variant}_sdp_train \
  --val-data-dir data/household/dt \
  --val-patterns h4_4_${variant}_sdp_val \
  --split-policy explicit_validation \
  --context-length 576 --stride 288 \
  --n-block 8 --h-dim 512 --n-heads 8 --drop-p 0.15 \
  --batch-size 16 --epochs 5 --lr 3e-5 --seed 42 \
  --rtg-source constant --return-scale 1.0 \
  --action-loss-weight 0.999 --state-loss-weight 0.002 \
  --return-loss-weight 0.0001 --device cuda --amp-mode auto \
  --save-path models/household/dt/h4_4_${variant}_standard_rtg_8x512_ctx576.pt \
  --checkpoint-path models/household/dt/h4_4_${variant}_standard_rtg_8x512_ctx576_checkpoint.pt \
  --loss-csv-path models/household/dt/h4_4_${variant}_standard_rtg_8x512_ctx576_loss.csv
done
```

Train the matching full-corpus PPO baseline (not the H4.3 pilot):

```bash
python3 scripts/train_household_sb3.py \
  --corpus-dir data/household/synth_h4_1 \
  --output-dir models/household/sb3/h4_4_full \
  --timesteps 500000 --n-envs 12 \
  --capacity-kwh 5 --max-flow-kw 3.3 \
  --battery-life-cost 5000 --seed 20260830 \
  --model-name ppo_h4_4_fullcorpus.zip
```

Evaluate on the fixed ten-window real-OOD surface (TTM from sidecars only;
TTM is never run live inside the simulator) and on the synthetic test
split with `--synth-dir` (per-episode battery configs from the manifest,
`--limit-windows` for a deterministic subsample), then compute paired
window-level bootstrap CIs, win counts, and one-sided Wilcoxon tests with
`scripts/household_forecast_stats.py` over the three `summary.json` files.

**H4.4 outcome (2026-09-02).** On the fixed 10-window real-OOD surface, the
full-corpus matched standard-RTG DTs saved: **TTM +$357.29/yr**, **24-hour
persistence +$309.35/yr**, **no forecast +$310.90/yr**, with fresh full-corpus
PPO +$23.66/yr, rule +$58.03/yr, and oracle +$738.96/yr as references. Paired:
TTM beats persistence by +$47.94/yr (95% CI +$27.99–$67.59, 9/10, Wilcoxon
p=0.0020) and no-forecast by +$46.39/yr (95% CI +$23.78–$68.91, 9/10,
p=0.0020). Every arm improved on the H4.2 seven-day-corpus run (TTM +$98.79),
so the offline-forecast benefit generalizes to the diverse corpus. The
persistence-vs-no-forecast gap collapsed to −$1.55/yr (p=0.46): with broad
training data the policy only exploits the genuinely-better TTM channel. On the
20-window synthetic test surface (mixed 5/10/20 kWh per-episode batteries) the
three DTs were statistically indistinguishable (TTM−no-forecast −$15.49/yr,
p=0.86, per-horizon mixed) — the forecast advantage is proven on the held-out
real household, not yet on the broad multi-battery synthetic surface. A matched
TTM/no-forecast/persistence sidecar layout bug (the forecast sidecar previously
reordered the Future columns) was fixed in `src/household_forecast.py`;
the mirror corpus and TTM trajectories were regenerated so all three arms share
identical action/reward/RTG labels and observation layout, differing only in
dims 6–7.


### 7. Compare observed and optimized real-battery dispatch

Use the H3 harness to replay recorded VPP actions and compare them with a
new cost-minimizing dispatch over each complete real day. It uses normalized
kW telemetry directly (not `build_year_dataset()`, whose values are already
converted to kWh per step), fits one corpus-wide action sign, and never spans
a gap seam:

```bash
python3 scripts/evaluate_household_tariffs.py \
  --normalized-dir data/household/real/normalized \
  --capacity-kwh 5 --max-flow-kw 3.3 --roundtrip-eff 0.80
```

The output is `eval_output/household/tariff_optimization/summary.json` with
per-day bootstrap CIs for observed replay, optimized dispatch, the
optimization gap, and the no-battery baseline under flat and free-window ToU
tariffs. Do not label a spot-pass-through result until a time-aligned retail
spot-price series is supplied.

### 8. Evaluate legacy policies on the real OOD surface

The legacy benchmark script evaluates the existing rule, PPO, and cloning-era
DT checkpoints with a fresh `SolarBatteryEnv` for every contiguous real-data
segment, alongside a daily perfect-foresight oracle. This isolates the
renovation gap and reports bootstrap CIs over segments:

```bash
python3 scripts/evaluate_household_ood_baselines.py \
  --capacity-kwh 5 --max-flow-kw 3.3
```

On the current five-segment OOD surface, the legacy PPO saves only $2/year and
the cloning-era DT loses $6/year against no battery (annualized segment
bootstrap means); the rule saves $70/year and the oracle indicates $729/year
is available. These legacy checkpoints therefore do not transfer to modern
household telemetry. See `eval_output/household/ood_baselines/summary.json`
for exact CI values and do not conflate this with an H2-trained policy.

### 9. Generate H2 SDP-teacher trajectories

The H2 handoff reuses the shared DT data schema. It solves a deterministic
per-day cost-to-go table over each synthetic environment episode, rolls its
actions through `SolarBatteryEnv`, and writes `rtg_value = -J_t(soc)` alongside
the normalized observation, action, and realized reward:

```bash
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth --split train \
  --out data/household/dt/sdp_teacher_train.parquet
```

Build a matching `--split val` corpus before invoking the sanctioned shared
trainer. Do not include real OOD segments in either corpus. The full corpus
build is now complete: 840 training and 180 validation episodes emitted
1,693,440 and 362,880 rows respectively. **Teacher data regenerated with
realistic tariff (31.042c import, free 11:00–14:00, 1c FiT) and RTE=0.80.**

**Standard-RTG baseline:** A 2×128 standalone DT trained for 5 epochs achieves
0.0424 validation total loss; on real OOD it saves **+$254/yr** (beats rule
+$82/yr).

**J_t(soc) at AEMO scale:** An 8×512 ctx576 model trained on corrected-RTE
teacher data achieves **+$300/yr** with `J_t(soc)` inference (3.6× rule, 24%
of oracle gap). This is a **positive transfer** result — planner distillation
transfers across scales when (1) model capacity ≥ AEMO scale, (2) RTE matches
the environment (0.80), (3) teacher data uses realistic tariff, and (4)
configs persist correctly.

The teacher data carries exact `rtg_value = -J_t(soc)`. The H2 inference
evaluation now precomputes an exact deterministic table per segment-local
calendar day and supplies the current-SOC prompt before each DT action; it
does not use the standard realized-reward recurrence for this policy. The
trainer also writes `<checkpoint-stem>_model_kwargs.json` beside every
checkpoint, so evaluation must use that file rather than a legacy model config.

## Main Artifacts

Common household artifact locations:

- `data/household/raw/`
- `data/household/logs/`
- `models/household/sb3/`
- `models/household/dt/`
- `eval_output/household/`

## Related Modules

- `src/helper.py`: household data transformation, evaluation, visualization
- `src/EnergySimEnv.py`: household simulation environment
- `src/decision.py`: agents and rollout helpers
- `src/sdp_algorithm.py`: planning baseline
- `src/mrdp_algorithm.py`: multi-resolution planning baseline
- `src/sb3train.py`: SB3 helper functions
- `src/decision_transformer.py`: DT model implementation
- `src/transformer_training.py`: DT training engine
- `src/household_synthetic.py`: clustered day library, archetypes, appliance/solar synthesis, validation gates, and episode export
- `scripts/build_household_synth_corpus.py`: reproducible H1.5 corpus builder
- `src/household_optimization.py`: deterministic dispatch optimizer and bootstrap CI helper
- `scripts/evaluate_household_tariffs.py`: H3 replay-gap and tariff evaluation
- `scripts/evaluate_household_ood_baselines.py`: H1 rule/oracle/PPO/DT real-OOD evaluation
- `scripts/generate_household_sdp_trajectories.py`: H2 synthetic SDP-teacher trajectory builder
- `scripts/train_household_sb3.py`: fresh modern-data SB3 (PPO) baseline trainer over a synthetic corpus
- `scripts/household_forecast_stats.py`: paired bootstrap/Wilcoxon stats for the H4.2/H4.4 forecast ablation
- `scripts/generate_household_ttm_forecasts.py` / `src/household_forecast.py`: offline causal TTM-R3 forecast sidecar
- `scripts/dump_household_behavior.py`: per-step rollouts (solar/load/price/power/SOC) of the matched H4.4 arms on one real-OOD window, feeding the website household behaviour charts
- `scripts/h4_degradation_study.py`: degradation-aware policy study (H4.5) — trains and evaluates DTs across multiple degradation modes (disabled, cycle-only, full, high/low battery-life-cost)

### 6d. H4.5 degradation-aware policy study

Run the five matched teacher/DT conditions from the repository root in the GPU
Distrobox:

```bash
python3 scripts/h4_degradation_study.py \
  --config all --train --eval \
  --output-dir results/h4_5_degradation
```

The study uses the H4.1 horizon-diverse corpus and holds the DT architecture,
optimizer, seed, and training schedule fixed while changing the teacher's
degradation mode and battery-life cost. Per-configuration models and
trajectories are written under `results/h4_5_degradation/`; real-OOD
evaluation summaries are written under
`eval_output/household/h4_5_degradation/`.

**H4.5 outcome (2026-09-04).** Annualized savings versus no battery on the
fixed ten-window real-OOD surface were A$395.33 (disabled), A$320.96
(cycle-only), A$329.06 (full realistic, A$5,000), A$212.87 (high cost,
A$10,000), and A$199.43 (low cost, A$1,000). Degradation-aware policies
therefore changed the outcome relative to the disabled control, but the
cost sweep was not monotonic. Full versus cycle-only differed by only A$8.10
per year. Read the dollar figures as **grid-bill reduction only** (wear cost
excluded); all five policies were scored under the same default full/$5,000
environment, so the ranking reflects cross-regime policy behaviour in the
real world, not net-of-wear economics. These results are short-window
economic evidence, not a direct measurement of cycling restraint: the current
evaluator does not record throughput/capacity-fade metrics or the
TTM/no-forecast forecast controls.

## Validation And Iteration

For code changes that affect the household track, use pytest as the main validation path:

```bash
python -m pytest tests/ -v
```

If you are changing only a narrow area, prefer a single relevant test file first.

## Notes

- Treat `scripts/pretrain_decision_transformer.py` as the canonical household DT entrypoint.
- Treat notebooks as the best surface for exploration, inspection, and demonstration.
- Keep household results separate from AEMO results in both reporting and interpretation.