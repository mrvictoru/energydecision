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

The optional `--use-ttm --ttm-mode {gap_imputation,weather_residual}` path is
intentionally isolated and fails explicitly until the Granite TTM runtime is
provisioned; TTM is not used as the primary generator.

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