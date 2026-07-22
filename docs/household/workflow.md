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