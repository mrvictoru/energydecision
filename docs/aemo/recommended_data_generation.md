# Recommended AEMO data-generation commands

This runbook creates a **new additive AEMO DT dataset** that targets the diversity gaps in the current corpus:

1. more **distinct 2024 time windows**
2. more **dispatch replay coverage** across regions
3. more **episodes sampled from shorter windows** instead of only a few very long episodes
4. continued **rule + SB3 + dispatch** behavior mixing

It writes to a **new output directory** so you can compare it with the existing dataset instead of overwriting it.

## What this recipe adds

Compared with the current dataset, this recipe focuses on:

- `2024_h1` and `2024_h2` windows instead of only long 2021-2023 blocks
- 5 NEM regions: `NSW1`, `QLD1`, `SA1`, `TAS1`, `VIC1`
- 90-day episodes to increase the number of distinct starts per scenario
- dispatch replay aliases that already resolve in this repo:
  - `wallgrove`
  - `wandoan`
  - `hornsdale`
  - `lake_bonney`
  - `victorian_big_battery`

The default safe version keeps SB3 rollouts on `small` and `medium`, matching the current dataset pattern. Rule data still covers `small`, `medium`, and `large`.

## Runtime

Run this from the repo root inside the recommended Distrobox:

```bash
distrobox enter energydecision-gpu
cd /path/to/energydecision
```

If you only want to use the notebook UI instead, start:

```bash
python3 -m jupyter lab notebooks/aemo_simrun.ipynb
```

Then copy the configuration block from the **Notebook config block** section below into the config cell and run the same logical stages as the CLI script.

## Preflight

Make sure the SB3 checkpoints referenced by the script exist:

```bash
ls models/aemo_sb3/*_aemo_model.zip
```

Expected files:

- `models/aemo_sb3/a2c_aemo_model.zip`
- `models/aemo_sb3/ddpg_aemo_model.zip`
- `models/aemo_sb3/ppo_aemo_model.zip`
- `models/aemo_sb3/sac_aemo_model.zip`
- `models/aemo_sb3/td3_aemo_model.zip`

If any are missing, either regenerate them from `notebooks/aemo_sb3train.ipynb` or remove the corresponding run from `BEHAVIOR_RUNS` in the script below.

## One-shot CLI data-generation command

Run this from the repo root:

```bash
python3 - <<'PY'
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT / "src"))

from aemo_notebook_utils import (
    build_dispatch_selection,
    build_dt_dataset_from_logs,
    build_model_config,
    fetch_and_preprocess_aemo_scenarios,
    fit_aemo_global_stats,
    prepare_run_paths,
    resolve_battery_variants,
    resolve_dispatch_battery_life_cost,
    resolve_dispatch_replay_runs,
    run_rule_episodes,
    run_sb3_episodes,
    should_run_dispatch_for_scenario,
    validate_aemo_dt_dimensions,
    write_combined_episode_logs,
    write_json,
)
from dispatch_utils import run_dispatch_replay

DATASET_TAG = "aemo_dt_diverse_2024"
OUTPUT_DIR = REPO_ROOT / "data" / DATASET_TAG
CACHE_DIR = REPO_ROOT / "data" / "aemo"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / f"{DATASET_TAG}_model_kwargs.json"
SCENARIO_MANIFEST_PATH = OUTPUT_DIR / f"{DATASET_TAG}_scenario_manifest.json"

STEP_DURATION = 5 / 60
EPISODE_HOURS = 24 * 90
ACTION_MODE = "multi_market"
DEGRADATION_MODE = "real_world"
DEGRADATION_CHEMISTRY = "LFP"
DEGRADATION_TEMPERATURE = 30.0
CONTEXT_LENGTH = 288

SCENARIOS = [
    {"label": "nsw1_2024_h1", "region": "NSW1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "nsw1_2024_h2", "region": "NSW1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "qld1_2024_h1", "region": "QLD1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "qld1_2024_h2", "region": "QLD1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "sa1_2024_h1", "region": "SA1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "sa1_2024_h2", "region": "SA1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "tas1_2024_h1", "region": "TAS1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "tas1_2024_h2", "region": "TAS1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "vic1_2024_h1", "region": "VIC1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "vic1_2024_h2", "region": "VIC1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
]

BATTERY_VARIANTS = [
    {"name": "small", "capacity_mwh": 2.0, "max_power_mw": 1.0, "init_soc_ratio": 0.5},
    {"name": "medium", "capacity_mwh": 10.0, "max_power_mw": 5.0, "init_soc_ratio": 0.5},
    {"name": "large", "capacity_mwh": 50.0, "max_power_mw": 25.0, "init_soc_ratio": 0.5},
]

BEHAVIOR_RUNS = [
    {
        "policy": "rule",
        "episodes": 4,
        "battery_variants": ["small", "medium", "large"],
        "random_episode_start": True,
        "seed": 42,
    },
    {
        "policy": "sb3",
        "name": "a2c",
        "episodes": 2,
        "battery_variants": ["small", "medium"],
        "algorithm": "A2C",
        "model_path": REPO_ROOT / "models" / "aemo_sb3" / "a2c_aemo_model.zip",
        "deterministic": True,
        "random_episode_start": True,
    },
    {
        "policy": "sb3",
        "name": "ddpg",
        "episodes": 2,
        "battery_variants": ["small", "medium"],
        "algorithm": "DDPG",
        "model_path": REPO_ROOT / "models" / "aemo_sb3" / "ddpg_aemo_model.zip",
        "deterministic": True,
        "random_episode_start": True,
    },
    {
        "policy": "sb3",
        "name": "ppo",
        "episodes": 2,
        "battery_variants": ["small", "medium"],
        "algorithm": "PPO",
        "model_path": REPO_ROOT / "models" / "aemo_sb3" / "ppo_aemo_model.zip",
        "deterministic": True,
        "random_episode_start": True,
    },
    {
        "policy": "sb3",
        "name": "sac",
        "episodes": 2,
        "battery_variants": ["small", "medium"],
        "algorithm": "SAC",
        "model_path": REPO_ROOT / "models" / "aemo_sb3" / "sac_aemo_model.zip",
        "deterministic": True,
        "random_episode_start": True,
    },
    {
        "policy": "sb3",
        "name": "td3",
        "episodes": 2,
        "battery_variants": ["small", "medium"],
        "algorithm": "TD3",
        "model_path": REPO_ROOT / "models" / "aemo_sb3" / "td3_aemo_model.zip",
        "deterministic": True,
        "random_episode_start": True,
    },
]

DISPATCH_RUNS = [
    {"label": "wallgrove_replay", "episodes": 1, "station_name": "wallgrove", "init_soc_ratio": 0.5},
    {"label": "wandoan_replay", "episodes": 1, "station_name": "wandoan", "init_soc_ratio": 0.5},
    {"label": "hornsdale_replay", "episodes": 1, "station_name": "hornsdale", "init_soc_ratio": 0.5},
    {"label": "lake_bonney_replay", "episodes": 1, "station_name": "lake_bonney", "init_soc_ratio": 0.5},
    {"label": "victorian_big_battery_replay", "episodes": 1, "station_name": "victorian_big_battery", "init_soc_ratio": 0.5},
]

run_paths = prepare_run_paths(output_dir=OUTPUT_DIR, dataset_tag=DATASET_TAG)

global_stats, scenario_manifest = fit_aemo_global_stats(
    scenarios=SCENARIOS,
    cache_dir=CACHE_DIR,
    step_duration=STEP_DURATION,
    refresh=False,
)

processed_by_label, _ = fetch_and_preprocess_aemo_scenarios(
    scenarios=SCENARIOS,
    cache_dir=CACHE_DIR,
    step_duration=STEP_DURATION,
    refresh=False,
    fixed_stats=global_stats,
)

scenario_manifest_payload = [
    {
        **entry,
        "start_date": entry["start_date"].isoformat(),
        "end_date": entry["end_date"].isoformat(),
    }
    for entry in scenario_manifest
]
scenario_lookup = {entry["label"]: entry for entry in scenario_manifest}
scenario_payloads = [
    (scenario_lookup[entry["label"]], processed_by_label[entry["label"]])
    for entry in scenario_manifest
]

resolved_battery_variants = resolve_battery_variants(BATTERY_VARIANTS)
resolved_dispatch_runs = resolve_dispatch_replay_runs(DISPATCH_RUNS)
MAX_STEP = int(round(EPISODE_HOURS / STEP_DURATION))
MAX_TIMESTEP = MAX_STEP

write_json(
    SCENARIO_MANIFEST_PATH,
    {
        "global_stats": global_stats,
        "scenarios": scenario_manifest_payload,
    },
)

all_logs = {}
raw_outputs = {}

for scenario_entry, scenario_processed_data in scenario_payloads:
    for run in BEHAVIOR_RUNS:
        selected_labels = set(run.get("battery_variants", [variant["label"] for variant in resolved_battery_variants]))
        selected_variants = [variant for variant in resolved_battery_variants if variant["label"] in selected_labels]

        for variant in selected_variants:
            tag = f"{scenario_entry['label']}__{run.get('name', run['policy'])}__{variant['label']}"
            print(f"Collecting {tag}...")

            if run["policy"] == "rule":
                episodes = run_rule_episodes(
                    processed_data=scenario_processed_data,
                    num_episodes=run["episodes"],
                    battery_capacity=variant["battery_capacity"],
                    max_battery_flow=variant["max_battery_flow"],
                    init_soc=variant["init_soc"],
                    step_duration=STEP_DURATION,
                    battery_life_cost=variant["battery_life_cost"],
                    max_step=MAX_STEP,
                    action_mode=ACTION_MODE,
                    degradation_mode=DEGRADATION_MODE,
                    degradation_chemistry=DEGRADATION_CHEMISTRY,
                    degradation_temperature=DEGRADATION_TEMPERATURE,
                    random_episode_start=run.get("random_episode_start", True),
                    base_seed=run.get("seed", 42),
                )
                dispatch_label = None
            elif run["policy"] == "sb3":
                episodes = run_sb3_episodes(
                    processed_data=scenario_processed_data,
                    battery_variant=variant,
                    model_path=run["model_path"],
                    algorithm=run["algorithm"],
                    num_episodes=run["episodes"],
                    max_step=MAX_STEP,
                    step_duration=STEP_DURATION,
                    action_mode=ACTION_MODE,
                    degradation_mode=DEGRADATION_MODE,
                    degradation_chemistry=DEGRADATION_CHEMISTRY,
                    degradation_temperature=DEGRADATION_TEMPERATURE,
                    random_episode_start=run.get("random_episode_start", True),
                    deterministic=run.get("deterministic", True),
                )
                dispatch_label = None
            else:
                raise ValueError(f"Unsupported policy in BEHAVIOR_RUNS: {run['policy']}")

            tagged_episodes = [
                episode.with_columns(
                    pl.lit(scenario_entry["label"]).alias("scenario_label"),
                    pl.lit(scenario_entry["region"]).alias("scenario_region"),
                    pl.lit(scenario_entry["start_date"].isoformat()).alias("scenario_start_date"),
                    pl.lit(scenario_entry["end_date"].isoformat()).alias("scenario_end_date"),
                    pl.lit(run["policy"]).alias("policy_name"),
                    pl.lit(variant["label"]).alias("battery_label"),
                    pl.lit(dispatch_label, dtype=pl.Utf8).alias("dispatch_label"),
                )
                for episode in episodes
            ]

            raw_path = run_paths["raw_dir"] / f"{tag}_logs.parquet"
            write_combined_episode_logs(episodes=tagged_episodes, output_path=raw_path)
            all_logs[tag] = tagged_episodes
            raw_outputs[tag] = str(raw_path)

    for dispatch_run in resolved_dispatch_runs:
        tag = f"{scenario_entry['label']}__dispatch__{dispatch_run['label']}"
        try:
            should_run_dispatch, dispatch_region = should_run_dispatch_for_scenario(
                scenario_region=scenario_entry["region"],
                dispatch_station=dispatch_run.get("station_name"),
                dispatch_duid=dispatch_run.get("dispatch_duid"),
                start_date=scenario_entry["start_date"],
                end_date=scenario_entry["end_date"],
            )

            if not should_run_dispatch:
                print(
                    f"Skipping {tag}: dispatch target region {dispatch_region} does not match "
                    f"scenario region {scenario_entry['region']}."
                )
                continue

            print(f"Collecting {tag}...")
            selection = build_dispatch_selection(
                region=scenario_entry["region"],
                start_date=scenario_entry["start_date"],
                end_date=scenario_entry["end_date"],
                cache_dir=CACHE_DIR,
                dispatch_station=dispatch_run.get("station_name"),
                dispatch_duid=dispatch_run.get("dispatch_duid"),
                dispatch_index=dispatch_run["dispatch_index"],
                battery_capacity=10.0,
                max_battery_flow=5.0,
                init_soc=0.0,
                init_soc_ratio=dispatch_run["init_soc_ratio"],
            )

            dispatch_battery_life_cost = resolve_dispatch_battery_life_cost(
                dispatch_run=dispatch_run,
                station_capacity_mwh=selection["battery_capacity"],
            )

            episodes, incident_logs, _ = run_dispatch_replay(
                processed_data=scenario_processed_data,
                selection=selection,
                start_date=scenario_entry["start_date"],
                end_date=scenario_entry["end_date"],
                region=scenario_entry["region"],
                cache_dir=str(CACHE_DIR),
                num_episodes=dispatch_run["episodes"],
                step_duration=STEP_DURATION,
                battery_life_cost=dispatch_battery_life_cost,
                max_step=MAX_STEP,
                output_dir=None,
                run_tag=tag,
                action_mode=ACTION_MODE,
                degradation_mode=DEGRADATION_MODE,
                degradation_chemistry=DEGRADATION_CHEMISTRY,
                degradation_temperature=DEGRADATION_TEMPERATURE,
            )
        except ValueError as exc:
            print(f"Skipping {tag}: {exc}")
            continue

        if any(df.height > 0 for df in incident_logs):
            incident_path = run_paths["raw_dir"] / f"{tag}_incident_logs.parquet"
            pl.concat(
                [df.with_columns(pl.lit(i).alias("episode_id")) for i, df in enumerate(incident_logs) if df.height > 0],
                how="diagonal_relaxed",
            ).write_parquet(incident_path)
            raw_outputs[f"{tag}__incidents"] = str(incident_path)

        dispatch_station_label = selection.get("station_key") or selection.get("station_name") or dispatch_run["label"]
        tagged_episodes = [
            episode.with_columns(
                pl.lit(scenario_entry["label"]).alias("scenario_label"),
                pl.lit(scenario_entry["region"]).alias("scenario_region"),
                pl.lit(scenario_entry["start_date"].isoformat()).alias("scenario_start_date"),
                pl.lit(scenario_entry["end_date"].isoformat()).alias("scenario_end_date"),
                pl.lit("dispatch").alias("policy_name"),
                pl.lit(None, dtype=pl.Utf8).alias("battery_label"),
                pl.lit(dispatch_run["label"]).alias("dispatch_label"),
                pl.lit(dispatch_station_label).alias("dispatch_station"),
            )
            for episode in episodes
        ]

        raw_path = run_paths["raw_dir"] / f"{tag}_logs.parquet"
        write_combined_episode_logs(episodes=tagged_episodes, output_path=raw_path)
        all_logs[tag] = tagged_episodes
        raw_outputs[tag] = str(raw_path)

dataset, manifest = build_dt_dataset_from_logs(all_logs)
validate_aemo_dt_dimensions(manifest, action_mode=ACTION_MODE)
model_kwargs = build_model_config(
    action_mode=ACTION_MODE,
    context_len=CONTEXT_LENGTH,
    max_timestep=MAX_TIMESTEP,
    output_path=MODEL_CONFIG_PATH,
)

dataset.write_parquet(run_paths["dataset_path"])

behavior_runs_json = [
    {
        **run,
        "model_path": str(run["model_path"]) if run.get("model_path") is not None else None,
    }
    for run in BEHAVIOR_RUNS
]
dispatch_runs_json = [{**run} for run in resolved_dispatch_runs]

manifest.update(
    {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(run_paths["dataset_path"]),
        "manifest_path": str(run_paths["manifest_path"]),
        "scenario_manifest_path": str(SCENARIO_MANIFEST_PATH),
        "cache_dir": str(CACHE_DIR),
        "region_count": len({entry["region"] for entry in scenario_manifest}),
        "scenario_count": len(scenario_manifest),
        "global_stats": global_stats,
        "scenarios": scenario_manifest_payload,
        "step_duration": STEP_DURATION,
        "episode_hours": EPISODE_HOURS,
        "max_step": MAX_STEP,
        "action_mode": ACTION_MODE,
        "degradation_mode": DEGRADATION_MODE,
        "degradation_chemistry": DEGRADATION_CHEMISTRY,
        "degradation_temperature": DEGRADATION_TEMPERATURE,
        "battery_variants": resolved_battery_variants,
        "behavior_runs": behavior_runs_json,
        "dispatch_runs": dispatch_runs_json,
        "model_config_path": str(MODEL_CONFIG_PATH),
        "model_kwargs": model_kwargs,
        "raw_outputs": raw_outputs,
    }
)

write_json(run_paths["manifest_path"], manifest)

print("\\nDone.")
print(f"dataset:  {run_paths['dataset_path']}")
print(f"manifest: {run_paths['manifest_path']}")
print(f"raw logs: {run_paths['raw_dir']}")
print(f"episodes: {manifest['episode_count']}")
print(f"rows:     {manifest['row_count']}")
PY
```

## Optional: include large-battery SB3 rollouts

If your current SB3 checkpoints are known to behave sensibly on the `large` battery, change each SB3 run block from:

```python
"battery_variants": ["small", "medium"],
```

to:

```python
"battery_variants": ["small", "medium", "large"],
```

Use this only if you have already validated those rollouts. The current repo dataset did **not** include large-battery SB3 trajectories, so the safe default above preserves that behavior.

## Inspect the generated dataset

After generation:

```bash
python3 - <<'PY'
from pathlib import Path
import json
import polars as pl

root = Path.cwd()
manifest_path = root / "data" / "aemo_dt_diverse_2024" / "aemo_dt_diverse_2024_manifest.json"
dataset_path = root / "data" / "aemo_dt_diverse_2024" / "aemo_dt_diverse_2024_dataset.parquet"

manifest = json.loads(manifest_path.read_text())
df = pl.read_parquet(dataset_path, columns=["episode_id", "source_policy"])

print("episode_count:", manifest["episode_count"])
print("row_count:", manifest["row_count"])
print("scenario_count:", manifest["scenario_count"])
print("battery_variants:", [v["label"] for v in manifest["battery_variants"]])
print("source policies:", df["source_policy"].n_unique())
print(df.group_by("source_policy").len().sort("len", descending=True))
PY
```

## Notebook config block

If you prefer `notebooks/aemo_simrun.ipynb`, replace the main config cell with the same values used above:

```python
SCENARIOS = [
    {"label": "nsw1_2024_h1", "region": "NSW1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "nsw1_2024_h2", "region": "NSW1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "qld1_2024_h1", "region": "QLD1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "qld1_2024_h2", "region": "QLD1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "sa1_2024_h1", "region": "SA1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "sa1_2024_h2", "region": "SA1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "tas1_2024_h1", "region": "TAS1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "tas1_2024_h2", "region": "TAS1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
    {"label": "vic1_2024_h1", "region": "VIC1", "start_date": datetime.fromisoformat("2024-01-01"), "end_date": datetime.fromisoformat("2024-07-01")},
    {"label": "vic1_2024_h2", "region": "VIC1", "start_date": datetime.fromisoformat("2024-07-01"), "end_date": datetime.fromisoformat("2025-01-01")},
]

STEP_DURATION = 5 / 60
EPISODE_HOURS = 24 * 90
ACTION_MODE = "multi_market"
DEGRADATION_MODE = "real_world"
DEGRADATION_CHEMISTRY = "LFP"
DEGRADATION_TEMPERATURE = 30.0
CONTEXT_LENGTH = 288
DATASET_TAG = "aemo_dt_diverse_2024"
OUTPUT_DIR = REPO_ROOT / "data" / DATASET_TAG
CACHE_DIR = REPO_ROOT / "data" / "aemo"
MODEL_CONFIG_PATH = REPO_ROOT / "configs" / f"{DATASET_TAG}_model_kwargs.json"
SCENARIO_MANIFEST_PATH = OUTPUT_DIR / f"{DATASET_TAG}_scenario_manifest.json"
```

Keep the rest of the notebook flow the same:

1. resolve/cache scenarios
2. collect rule + SB3 trajectories
3. collect dispatch replays
4. build the merged DT dataset + manifest

## After the new data exists

Train it through the hardened launcher:

```bash
python3 src/launch_aemo_training.py \
  --runtime-mode require-distrobox \
  --run-tier proxy-baseline \
  --dataset-path data/aemo_dt_diverse_2024/aemo_dt_diverse_2024_dataset.parquet \
  --model-config configs/aemo_dt_diverse_2024_model_kwargs.json
```

For the heavier baseline:

```bash
python3 src/launch_aemo_training.py \
  --runtime-mode require-distrobox \
  --run-tier learning-baseline \
  --dataset-path data/aemo_dt_diverse_2024/aemo_dt_diverse_2024_dataset.parquet \
  --model-config configs/aemo_dt_diverse_2024_model_kwargs.json
```
