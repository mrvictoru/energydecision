"""Retrain all SB3 models with full_fcas action mode.

Run from repo root:
    bash scripts/retrain_sb3_full_fcas.sh
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
    {"region": "NSW1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"region": "QLD1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"region": "SA1",  "start_date": "2022-04-01", "end_date": "2023-12-01"},
    {"region": "TAS1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"region": "VIC1", "start_date": "2021-04-01", "end_date": "2023-12-01"},
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

# Training configuration
TOTAL_TIMESTEPS = 200_000  # moderate for demonstration
TEST_TIMESTEPS = 20_000
EPISODES_PER_VARIANT = 1
N_TRIALS = 0  # no optuna, just default
N_JOBS = 1


def find_processed_data(region: str) -> pl.DataFrame:
    """Find and load the processed data file for a region."""
    candidates = sorted(CACHE_DIR.glob(f"processed_{region}_*_*_0.0833h.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No processed data for {region} in {CACHE_DIR}")
    path = candidates[0]
    print(f"  Loading {path.name} ({path.stat().st_size // 1_000_000} MB)")
    return pl.read_parquet(str(path))


def train_one(algorithm: str, processed_data: pl.DataFrame, region: str, total_timesteps: int, test_timesteps: int) -> None:
    """Train one SB3 model and save to models/aemo_sb3/."""
    out_path = MODELS_DIR / f"{algorithm.lower()}_aemo_fcas_model.zip"
    print(f"\n{'='*60}")
    print(f"Training {algorithm} on {region} with full_fcas...")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  3 battery variants, {EPISODE_HOURS}h episodes")
    print(f"{'='*60}")

    model, _ = train_sb3_model_on_aemo(
        processed_data=processed_data,
        algorithm=algorithm,
        battery_variants=BATTERIES,
        episodes_per_variant=EPISODES_PER_VARIANT,
        max_step=MAX_STEP,
        step_duration=STEP_DURATION,
        action_mode="full_fcas",
        degradation_mode="real_world",
        degradation_chemistry="LFP",
        degradation_temperature=30.0,
        random_episode_start=True,
        test_timesteps=test_timesteps,
        total_timesteps=total_timesteps,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
        default_model=True,
    )

    model.save(str(out_path))
    print(f"  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Retrain SB3 models with full_fcas")
    parser.add_argument("--algorithms", type=str, default="PPO,A2C,DDPG,SAC,TD3",
                        help="Comma-separated list of algorithms")
    parser.add_argument("--region", type=str, default=None,
                        help="Single region (default: all 5)")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help="Total training timesteps")
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",")]
    tt = args.timesteps
    test_tt = max(20_000, tt // 10)

    regions = [s["region"] for s in SCENARIOS]
    if args.region:
        regions = [args.region]

    for region in regions:
        print(f"\n{'#'*60}")
        print(f"# Loading data for {region}")
        print(f"{'#'*60}")
        processed = find_processed_data(region)
        for algo in algorithms:
            train_one(algo, processed, region, tt, test_tt)

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
