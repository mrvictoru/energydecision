# AEMO notebook-first offline RL and DT workflow

This is the **workflow guide** for AEMO data generation, notebook-based experiments, and Decision Transformer training.

Use this document when you need:

- the end-to-end notebook flow
- dataset generation steps
- SB3 training entrypoints
- AEMO DT dataset and training artifacts

If you only need environment mechanics, read [environment.md](environment.md). If you want the full AEMO docs map, start with [README.md](README.md).

The AEMO offline RL workflow is now centered on notebooks so you can inspect intermediate tables, debug trajectories, and change configs without leaving the notebook UI.

## Main files

- `<repo_root>/notebooks/aemo_simrun.ipynb`
- `<repo_root>/notebooks/aemo_sb3train.ipynb`
- `<repo_root>/src/aemo_notebook_utils.py`
- `<repo_root>/configs/aemo_decision_transformer_model_kwargs.json`

## Manual fallback for AEMO generator metadata

Most AEMO data is fetched from NEMWeb archives, but the generator / battery registry mapping used by
`fetch_aemo_generation_by_fuel()`, `get_available_battery_units()`, and dispatch replay station discovery
comes from AEMO's **NEM Registration and Exemption List** static spreadsheet.

- Direct file URL used by NEMOSIS:
  `https://www.aemo.com.au/-/media/Files/Electricity/NEM/Participant_Information/NEM-Registration-and-Exemption-List.xls`
- Human-facing AEMO page to use if the direct link changes:
  `https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/participate-in-the-market/registration/registered-participants`
- Recommended manual fallback location in this repo:
  `<repo_root>/data/aemo/manual/NEM Registration and Exemption List.xls`
- Alternate override:
  set `AEMO_GENERATORS_FILE=/absolute/path/to/NEM Registration and Exemption List.xls`

The runtime keeps NEMOSIS-managed static downloads in `<repo_root>/data/aemo/_nemosis_static/` so a bad web
response should not overwrite the manual fallback copy.

## Manual fallback for monthly MMS cache

When post-2024-07 monthly MMS downloads fail through `nemosis`, use the repo's archive fetcher to
stage the raw monthly CSVs directly into `<repo_root>/data/aemo/` with the exact filenames NEMOSIS
expects.

Example:

```bash
python3 src/fetch_aemo_monthly_cache.py --year 2025
```

Default tables:

- `DISPATCHLOAD`
- `DISPATCHPRICE`
- `DISPATCHREGIONSUM`
- `DISPATCH_UNIT_SCADA`

The tool downloads the NEMWeb monthly archive zips, validates that each zip contains the expected
`PUBLIC_ARCHIVE#...#FILE01#...CSV`, and only then writes or replaces the local cache file. That
means a failed HTML/text response should not overwrite a valid local monthly cache file.

## What `notebooks/aemo_simrun.ipynb` does

`notebooks/aemo_simrun.ipynb` is the main offline-data notebook. It is structured so you can stop after any stage and inspect the objects in memory.

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

Inside `notebooks/aemo_simrun.ipynb`, edit these cells first:

- `REGION`, `START_DATE`, `END_DATE`
- `STEP_DURATION`, `EPISODE_HOURS`
- `ACTION_MODE`
- `DEGRADATION_MODE`, `DEGRADATION_CHEMISTRY`, `DEGRADATION_TEMPERATURE`
- `BATTERY_VARIANTS`
- `BEHAVIOR_RUNS`
- `DISPATCH_RUNS`

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
- `sb3`

Example shape:

```python
BEHAVIOR_RUNS = [
    {"policy": "rule", "episodes": 4, "battery_variants": ["small", "medium", "large"]},
    {
        "policy": "sb3",
        "episodes": 2,
        "battery_variants": ["small", "medium"],
        "algorithm": "PPO",
        "model_path": <repo_root>/models/aemo_sb3/ppo_aemo_model.zip,
    },
]
```

### Dispatch replay configs

`DISPATCH_RUNS` is separate from `BATTERY_VARIANTS` because dispatch replay should use the
station sizing implied by the AEMO battery record rather than a synthetic battery variant.

Example shape:

```python
DISPATCH_RUNS = [
    {
        "label": "dalrymple_north_replay",
        "episodes": 1,
        "station_name": "dalrymple_north",
        "init_soc_ratio": 0.5,
    },
    {
        "label": "torrens_island_replay",
        "episodes": 1,
        "station_name": "torrens_island",
        "init_soc_ratio": 0.5,
    },
]
```

Notes:

- `episodes` may be greater than 1, but repeated dispatch replays are expected to be nearly identical
  unless you intentionally vary initial conditions or economics.
- Station capacity and max power come from the resolved AEMO unit metadata.
- `init_soc_ratio` is applied after station sizing is resolved.
- `battery_life_cost` or `battery_cost_per_kwh` can be supplied if you want to override the
  degradation economics for dispatch replay.
- When you want multiple replay baselines in the evaluator, prefer two stations from the same
  window that both have `DISPATCHLOAD` coverage. For the SA1 winter 2024 window, `dalrymple_north`
  and `torrens_island` both work while `hornsdale` does not.

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

## Output artifacts from `notebooks/aemo_simrun.ipynb`

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

## Training large AEMO datasets safely

The combined AEMO DT parquet can be too large for the current in-memory `TrajectoryDataset`
implementation when it is loaded as one file and expanded into all sliding windows at once.

### Training tiers

Use two distinct DT training tiers for AEMO:

- **`aemo_proxy`** — compact, cheap, and intended for quick idea ranking only
- **`aemo_learning_baseline`** — broader baseline training with explicit held-out validation and a longer context

Do not treat the narrow proxy slice (`aemo_dt_dataset_train_subset_007`) as the main learning baseline. It is useful for smoke tests and cheap sweeps, but not for establishing the branch point that future AEMO experiments should build on.

Use `src/pretrain_aemo_decision_transformer.py` subset mode for large runs:

```bash
python3 src/pretrain_aemo_decision_transformer.py \
  --dataset-path data/aemo_dt/aemo_dt_dataset.parquet \
  --surface-preset aemo_learning_baseline \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --train-in-subsets \
  --subset-episodes 24 \
  --epochs-per-subset 1 \
  --num-workers 0
```

If you are already inside the `energydecision` Distrobox shell at the repo root, run the same
command as `python3 src/pretrain_aemo_decision_transformer.py` and keep normal `data/...` paths.
The wrapper now forwards a small approved set of direct-trainer shape knobs such as `--context-length`,
`--state-dim`, and `--rope-max-position`, but it still expects one dataset path at a time.
For manual mixed-corpus runs that rely on `--patterns`, call `src/pretrain_decision_transformer.py`
directly instead.

For most CLI training runs, prefer the higher-level launcher instead:

```bash
python3 src/launch_aemo_training.py --run-tier proxy-baseline
python3 src/launch_aemo_training.py --run-tier learning-baseline
```

It makes the intended run tier explicit, writes `aemo_training_launch_plan.json` next to the run
artifacts, starts the live progress dashboard automatically, and re-enters the preferred Distrobox
when launched from the host shell.
The launcher also forwards direct DT shape overrides (for example `--context-length`, `--h-dim`,
`--n-heads`, and RoPE flags) to the AEMO wrapper so you can run controlled context/model sweeps
without rewriting the full command line.

Practical guidance:

- start serious baseline refreshes from `aemo_learning_baseline`
- use explicit validation subsets/files rather than tiny episode splits
- start with `lr=3e-5`
- prefer `context_len=288` for learning baselines; `120` is a reasonable fallback if runtime is too high
- keep `aemo_proxy` for rapid triage only

For interactive autoresearch loops, it is often better to pin a small **fixed pilot train/val split**
for the proxy tier than to repeatedly launch the full learning-baseline subset pipeline. The wrapper now
accepts an explicit validation parquet via `--val-dataset-path`, so you can keep the AEMO entrypoint while
holding the train/validation pair constant.

The repository now includes a reproducible pilot builder that refreshes the fixed split from the full
AEMO dataset using a curated set of week-long cross-region episode slices:

```bash
python3 src/build_aemo_autoresearch_pilot.py
```

That command rewrites:

- `data/aemo_dt/autoresearch_pilot/aemo_dt_train_pilot.parquet`
- `data/aemo_dt/autoresearch_pilot/aemo_dt_val_pilot.parquet`
- `data/aemo_dt/autoresearch_pilot/aemo_dt_autoresearch_pilot_manifest.json`

Use the generated split like this:

```bash
python3 src/pretrain_aemo_decision_transformer.py \
  --dataset-path data/aemo_dt/autoresearch_pilot/aemo_dt_train_pilot.parquet \
  --val-dataset-path data/aemo_dt/autoresearch_pilot/aemo_dt_val_pilot.parquet \
  --surface-preset aemo_proxy \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --epochs 1 \
  --batch-size 128 \
  --lr 3e-5 \
  --num-workers 0
```

Use that fixed pilot split only for cheap inner-loop ranking. Once a proxy change looks promising, rerun it
on the heavier learning-baseline path before treating it as a serious branch point. Each pilot example is
about one week of history, or roughly 2k 5-minute rows.

For the proxy tier, prefer **best validation action loss** as the ranking metric and keep
**best validation total loss** as the guardrail. The training surface manifest now writes both, plus a
`pilot_ranking_metric` recommendation for the current preset.

For simulator checks, use:

- `configs/aemo_autoresearch_evaluator.mini.json` for quick pilot screening
- `configs/aemo_autoresearch_evaluator.example.json` for the fuller held-out comparison

Both evaluator configs can share cached non-DT reference rollouts through `reference_cache_dir`, so fixed
baselines like `rule`, dispatch replay, and unchanged SB3 references do not need to rerun for every pilot
experiment.
They also now include `heldout.parallel_workers` with `parallelize_candidate_dt=false`, so evaluator
rollouts run in parallel by default for reference policies while DT candidate rollouts stay serial unless
you explicitly opt in.

If the broader AEMO corpus is still too dominated by the original multi-year episode spans, you can
resegment it into shorter fixed-horizon episodes before launching the learning baseline. The repository
includes a builder for that workflow:

```bash
python3 src/build_aemo_short_horizon_dataset.py \
  --dataset-tag aemo_dt_8week \
  --target-episode-hours 1344 \
  --subset-episodes 24
```

That command:

- rewrites the existing non-eval dataset into contiguous **8-week** episodes
- preserves the original **rule + SB3 + dispatch replay** source-policy mix
- writes:
  - `data/aemo_dt/aemo_dt_8week_dataset.parquet`
  - `data/aemo_dt/aemo_dt_8week_manifest.json`
  - `data/aemo_dt/aemo_dt_8week_scenario_manifest.json`
  - `data/aemo_dt/aemo_dt_8week_dataset_subsets/`

You can then point the broader training tier at the refreshed dataset:

```bash
python3 src/launch_aemo_training.py \
  --run-tier learning-baseline \
  --dataset-path data/aemo_dt/aemo_dt_8week_dataset.parquet
```

If you want the live progress dashboard to update during a long subset epoch instead of only at the end,
set `--checkpoints-per-epoch` above `1` so the trainer writes intermediate snapshot updates.

Subset mode works like this:

1. The combined dataset parquet is split once at the episode level into a global train set and a global validation set.
2. The train and validation episodes are each written into smaller parquet subset files, while preserving whole episodes.
3. The generic DT trainer is launched once per train subset file and receives the full validation subset set explicitly.
4. The first subset starts normally.
5. Later subset runs add `--resume`, reusing the same checkpoint and optimizer state.
6. The target epoch value is cumulative across subset stages, so resumed runs continue training instead of stopping immediately at the checkpoint epoch.

This avoids the worst memory spikes from loading the entire AEMO corpus into one `TrajectoryDataset`.

Current tradeoff:

- Validation now comes from one consistent global held-out split, but it is still materialized in memory inside the generic trainer when those validation subset parquet files are loaded.

### Prewarming evaluator caches

Held-out evaluator runs reuse cached processed scenario files under `data/aemo/`. Precomputing those
scenario windows before a long autoresearch session can reduce evaluator turnaround and avoid mixing
cache generation work with DT training runs. Make sure that cache directory is writable by the user
running autoresearch; stale root-owned processed parquet files can block later evaluator reruns.

Use the helper below to preflight permissions and warm the fixed evaluator windows before a search session:

```bash
python3 src/prewarm_aemo_cache.py \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json
```

## What `notebooks/aemo_sb3train.ipynb` does

`notebooks/aemo_sb3train.ipynb` is the online RL notebook for AEMO + SB3.

It covers:

1. fetch + cache AEMO data
2. define one or more battery variants
3. choose an SB3 algorithm (`PPO`, `A2C`, `DDPG`, `SAC`, `TD3`)
4. train the online RL model on AEMO environments
5. save the trained model
6. export rollout parquet logs for later offline-DT use

Default rollout outputs go under:

- `<repo_root>/models/aemo_sb3/raw_logs/`

## Recommended usage

1. Open `<repo_root>/notebooks/aemo_sb3train.ipynb` if you need a fresh SB3 policy.
2. Save the trained model.
3. Open `<repo_root>/notebooks/aemo_simrun.ipynb`.
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
