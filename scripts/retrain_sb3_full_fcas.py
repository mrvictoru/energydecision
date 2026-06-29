"""Retrain all SB3 models with full_fcas action mode.

Trains ONE model per algorithm on ALL 5 AEMO regions combined,
matching the original notebook workflow.

Run from repo root:
    SINGLE_PROCESS_TRAINING=1 python3 scripts/retrain_sb3_full_fcas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl
from aemo_notebook_utils import train_sb3_model_on_aemo

# Regions with 2021-2023 training data (same as original FCAS dataset)
SCENARIOS = [
    {"label": "nsw1", "region": "NSW1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"label": "qld1", "region": "QLD1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"label": "sa1",  "region": "SA1",  "start_date": "2022-04-01", "end_date": "2023-12-01"},
    {"label": "tas1", "region": "TAS1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"label": "vic1", "region": "VIC1", "start_date": "2021-04-01", "end_date": "2023-12-01"},
]

BATTERIES = [
    {"name": "medium", "capacity_mwh": 10.0, "max_power_mw": 5.0, "init_soc_ratio": 0.5},
    {"name": "small",  "capacity_mwh": 2.0,  "max_power_mw": 1.0, "init_soc_ratio": 0.5},
    {"name": "large",  "capacity_mwh": 50.0, "max_power_mw": 25.0, "init_soc_ratio": 0.5},
]

CACHE_DIR = ROOT / "data" / "aemo"
MODELS_DIR = ROOT / "models" / "aemo_sb3"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STEP_DURATION = 5 / 60  # 5 minutes
EPISODE_HOURS = 24 * 7  # 1 week
MAX_STEP = int(EPISODE_HOURS / STEP_DURATION)

TOTAL_TIMESTEPS = 200_000
TEST_TIMESTEPS = 20_000
EPISODES_PER_VARIANT = 1
N_TRIALS = 0
N_JOBS = 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrain SB3 models with full_fcas")
    parser.add_argument("--algorithms", type=str, default="PPO,A2C,DDPG,SAC,TD3",
                        help="Comma-separated list of algorithms")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help="Total training timesteps")
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",")]
    tt = args.timesteps
    test_tt = max(20_000, tt // 10)

    # Step 1: Load and combine ALL 5 regions into one dataset
    print("Loading and combining all 5 regions...")
    frames = []
    for scenario in SCENARIOS:
        candidates = sorted(CACHE_DIR.glob(
            f"processed_{scenario['region']}_*_*_0.0833h.parquet"
        ))
        if candidates:
            path = candidates[0]
            df = pl.read_parquet(str(path))
            df = df.with_columns(pl.lit(scenario["label"]).alias("scenario_label"))
            frames.append(df)
            print(f"  {scenario['label']}: {path.name} ({df.height} rows)")
        else:
            print(f"  {scenario['label']}: no cached data found, skipping")

    if not frames:
        print("ERROR: No cached processed data found for any region.")
        sys.exit(1)

    combined = pl.concat(frames, how="diagonal_relaxed").sort(["SETTLEMENTDATE", "scenario_label"])
    # Fill nulls from diagonal_relaxed concat (some regions lack GEN_solar, etc.)
    numeric_cols = [c for c in combined.columns if c not in {"SETTLEMENTDATE", "scenario_label"}]
    combined = combined.with_columns([
        pl.col(c).fill_null(0.0) for c in numeric_cols if c != "scenario_label"
    ])
    print(f"\nCombined dataset: {combined.height} rows across {len(frames)} regions")
    print(f"  Columns: {[c for c in combined.columns if not c.endswith('_normalized')]}")

    # Step 2: Train ONE model per algorithm on ALL regions combined
    for algo in algorithms:
        out_path = MODELS_DIR / f"{algo.lower()}_aemo_fcas_model.zip"
        print(f"\n{'='*60}")
        print(f"Training {algo} on ALL 5 regions combined with full_fcas...")
        print(f"  Timesteps: {tt}")
        print(f"  3 battery variants, {EPISODE_HOURS}h episodes")
        print(f"  Output: {out_path}")
        print(f"{'='*60}")

        model, _ = train_sb3_model_on_aemo(
            processed_data=combined,
            algorithm=algo,
            battery_variants=BATTERIES,
            episodes_per_variant=EPISODES_PER_VARIANT,
            max_step=MAX_STEP,
            step_duration=STEP_DURATION,
            action_mode="full_fcas",
            degradation_mode="real_world",
            degradation_chemistry="LFP",
            degradation_temperature=30.0,
            random_episode_start=True,
            test_timesteps=test_tt,
            total_timesteps=tt,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            default_model=True,
        )

        model.save(str(out_path))
        print(f"✅ {algo} saved: {out_path}")

    print(f"\n{'='*60}")
    print("All models trained on combined 5-region data!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
