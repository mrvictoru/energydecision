# AEMO notebook-first offline RL workflow

The AEMO offline RL workflow is now centered on notebooks so you can inspect intermediate tables, debug trajectories, and change configs without leaving the notebook UI.

## Main files

- `<repo_root>/aemo_simrun.ipynb`
- `<repo_root>/aemo_sb3train.ipynb`
- `<repo_root>/src/aemo_notebook_utils.py`
- `<repo_root>/configs/aemo_decision_transformer_model_kwargs.json`

## What `aemo_simrun.ipynb` does

`aemo_simrun.ipynb` is the main offline-data notebook. It is structured so you can stop after any stage and inspect the objects in memory.

It covers:

1. fetch + preprocess + cache AEMO market data
2. configure a target AEMO experiment
3. sweep multiple battery sizes
4. collect behavior trajectories from:
   - rule-based AEMO agent
   - dispatch replay
   - SB3-based models
5. save raw parquet logs per policy / battery variant
6. merge those logs into a DT-ready parquet dataset
7. write a dataset manifest + AEMO DT model config
8. optionally launch DT training

## Notebook configuration points

Inside `aemo_simrun.ipynb`, edit these cells first:

- `REGION`, `START_DATE`, `END_DATE`
- `STEP_DURATION`, `EPISODE_HOURS`
- `ACTION_MODE`
- `DEGRADATION_MODE`, `DEGRADATION_CHEMISTRY`, `DEGRADATION_TEMPERATURE`
- `BATTERY_VARIANTS`
- `BEHAVIOR_RUNS`

### Battery-size sweeps

`BATTERY_VARIANTS` lets you run multiple battery sizes in the same workflow. Each variant is tagged separately in the saved logs and dataset manifest.

Example shape:

```python
BATTERY_VARIANTS = [
    {"name": "small", "capacity_mwh": 2.0, "max_power_mw": 1.0, "init_soc_ratio": 0.5},
    {"name": "medium", "capacity_mwh": 10.0, "max_power_mw": 5.0, "init_soc_ratio": 0.5},
    {"name": "large", "capacity_mwh": 50.0, "max_power_mw": 25.0, "init_soc_ratio": 0.5},
]
```

### Behavior-policy sweeps

`BEHAVIOR_RUNS` controls which policies are simulated.

Supported notebook policies:

- `rule`
- `dispatch`
- `sb3`

Example shape:

```python
BEHAVIOR_RUNS = [
    {"policy": "rule", "episodes": 4, "battery_variants": ["small", "medium", "large"]},
    {"policy": "dispatch", "episodes": 2, "battery_variants": ["medium"], "station_name": "hornsdale"},
    {
        "policy": "sb3",
        "episodes": 2,
        "battery_variants": ["small", "medium"],
        "algorithm": "PPO",
        "model_path": <repo_root>/models/aemo_ppo_model.zip,
    },
]
```

## DT dataset expectations

The notebook writes parquet logs that are then merged into the schema expected by `TrajectoryDataset`:

- `episode_id`
- `step`
- `norm_observation`
- `action`
- `reward`

The helper validates the AEMO dimensions before writing the final DT dataset:

- observation space: **18D**
- `multi_market` → `act_dim=3`
- `simple` → `act_dim=1`

## Output artifacts from `aemo_simrun.ipynb`

By default the notebook writes:

- DT dataset parquet  
  `<repo_root>/data/aemo_dt/aemo_dt_dataset.parquet`
- dataset manifest  
  `<repo_root>/data/aemo_dt/aemo_dt_manifest.json`
- raw logs directory  
  `<repo_root>/data/aemo_dt/raw_logs/`
- AEMO DT model config  
  `<repo_root>/configs/aemo_decision_transformer_model_kwargs.json`

If DT training is enabled in the notebook config, it also writes:

- `<repo_root>/data/aemo_dt/aemo_dt_dt_model.pt`
- `<repo_root>/data/aemo_dt/aemo_dt_dt_checkpoint.pt`
- `<repo_root>/data/aemo_dt/aemo_dt_dt_loss_history.csv`

## What `aemo_sb3train.ipynb` does

`aemo_sb3train.ipynb` is the online RL notebook for AEMO + SB3.

It covers:

1. fetch + cache AEMO data
2. define one or more battery variants
3. choose an SB3 algorithm (`PPO`, `A2C`, `DDPG`, `SAC`, `TD3`)
4. train the online RL model on AEMO environments
5. save the trained model
6. export rollout parquet logs for later offline-DT use

Default rollout outputs go under:

- `<repo_root>/data/aemo_sb3/raw_logs/`

## Recommended usage

1. Open `<repo_root>/aemo_sb3train.ipynb` if you need a fresh SB3 policy.
2. Save the trained model.
3. Open `<repo_root>/aemo_simrun.ipynb`.
4. Add that SB3 model to `BEHAVIOR_RUNS`.
5. Collect rule + dispatch + SB3 trajectories across your battery variants.
6. Build the DT dataset and manifest.
7. Optionally launch DT training from the notebook.

## Helper module notes

`<repo_root>/src/aemo_notebook_utils.py` contains the reusable notebook helpers for:

- AEMO data fetching / caching
- battery-variant resolution
- rule / dispatch / SB3 rollout collection
- parquet log persistence
- DT dataset assembly
- DT model-config writing
- DT training launch
- SB3 online training
