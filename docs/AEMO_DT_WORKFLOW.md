# AEMO Decision Transformer Workflow

This workflow turns the existing AEMO notebook path into a reproducible offline-DT pipeline:

1. fetch + preprocess AEMO market data
2. collect rule and/or dispatch-replay trajectories
3. merge them into a DT parquet dataset
4. write an AEMO-specific DT model config + dataset manifest
5. optionally launch `<repo_root>/src/pretrain_decision_transformer.py`

## Files added for this workflow

- `<repo_root>/src/aemo_dt_workflow.py`
- `<repo_root>/configs/aemo_decision_transformer_model_kwargs.json`

## Default AEMO DT assumptions

- observation space: **18D**
- action mode: **`multi_market`**
- DT action dimension: **3**
- degradation defaults:
  - `degradation_mode='real_world'`
  - `degradation_chemistry='LFP'`
  - `degradation_temperature=30.0`

## Recommended first run

Use a small SA1 run first:

```bash
cd <repo_root>
python src/aemo_dt_workflow.py \
  --mode both \
  --region SA1 \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --episode-hours 24 \
  --step-duration 0.0833333333 \
  --action-mode multi_market \
  --num-rule-episodes 8 \
  --num-dispatch-episodes 4 \
  --dispatch-station hornsdale \
  --random-episode-start \
  --output-dir <repo_root>/data/aemo_dt \
  --cache-dir <repo_root>/data/aemo
```

## Output artifacts

With the defaults above, the workflow writes:

- dataset parquet  
  `<repo_root>/data/aemo_dt/aemo_dt_dataset.parquet`
- dataset manifest  
  `<repo_root>/data/aemo_dt/aemo_dt_manifest.json`
- raw rule logs  
  `<repo_root>/data/aemo_dt/raw_logs/aemo_dt_rule_logs.parquet`
- raw dispatch logs  
  `<repo_root>/data/aemo_dt/raw_logs/aemo_dt_dispatch_dispatch_logs.parquet`
- AEMO DT model config  
  `<repo_root>/configs/aemo_decision_transformer_model_kwargs.json`
- trained model checkpoint outputs  
  `<repo_root>/data/aemo_dt/aemo_dt_dt_model.pt`  
  `<repo_root>/data/aemo_dt/aemo_dt_dt_checkpoint.pt`  
  `<repo_root>/data/aemo_dt/aemo_dt_dt_loss_history.csv`

## Collect only

```bash
cd <repo_root>
python src/aemo_dt_workflow.py \
  --mode collect \
  --region SA1 \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --episode-hours 24 \
  --dispatch-station hornsdale \
  --random-episode-start
```

## Train only on an existing dataset

```bash
cd <repo_root>
python src/aemo_dt_workflow.py \
  --mode train \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --dataset-path <repo_root>/data/aemo_dt/aemo_dt_dataset.parquet \
  --model-config-path <repo_root>/configs/aemo_decision_transformer_model_kwargs.json \
  --output-dir <repo_root>/data/aemo_dt
```

## Using extra behavior-policy logs

If you already have parquet logs from RL runs, include them in the merged offline dataset:

```bash
cd <repo_root>
python src/aemo_dt_workflow.py \
  --mode collect \
  --region SA1 \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --num-rule-episodes 4 \
  --num-dispatch-episodes 2 \
  --dispatch-station hornsdale \
  --include-log /absolute/path/to/aemo_mm_ppo_logs.parquet /absolute/path/to/aemo_mm_sac_logs.parquet
```

Each included parquet is split by `episode_id` when present, then merged into the final DT dataset with a `source_policy` tag for provenance.

## Manifest contents

The manifest records:

- experiment definition (`region`, dates, step duration, degradation settings)
- battery/env settings
- model config used for training
- raw source files used to build the dataset
- per-source episode/row counts
- per-episode index with detected `state_dim` and `act_dim`

That gives you the provenance needed to keep AEMO dataset builds reproducible.
