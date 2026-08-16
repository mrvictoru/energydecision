#!/usr/bin/env python3
"""Generate SDP-teacher DT training trajectories for Stage B (standalone DT).

For each (region, horizon, battery) slot, slice the cached 2021-2023 processed
parquet into episode-length windows with random starts and replay the honest
SDP executor (AEMOAgent dt_soc_oracle executor='sdp') through the env,
capturing per-step (norm_observation, action[9], reward). These triples are
self-consistent (the reward is what the teacher earns from the action, the obs
are the env's normalized state), so a Decision Transformer retrained on them
is a standalone clone of the honest planner — no solver at inference.

Data source: data/aemo/processed_{REGION}_*.parquet (the SAME preprocessed
frames the evaluator uses at inference), so train/eval observation spaces are
identical.

Usage (pilot):
  python3 scripts/generate_sdp_dt_trajectories.py \
    --regions NSW1 QLD1 SA1 TAS1 VIC1 \
    --horizons short medium \
    --batteries medium_1c fast_375c \
    --episodes-per-slot 8 \
    --out data/aemo_dt_sdp/dt_trajectories.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_notebook_utils import create_aemo_env  # noqa: E402
from decision import AEMOAgent  # noqa: E402
from decision_transformer import DecisionTransformer  # noqa: E402

BATTERY_SPECS = {
    "medium_1c": {"capacity": 10.0, "max_flow": 10.0},
    "fast_375c": {"capacity": 8.0, "max_flow": 30.0},
}
# 5-min steps per episode by horizon (short=14d, medium=8wk)
HORIZON_STEPS = {"short": 3456, "medium": 16128, "long": 74880}

# Region -> longest pre-2024 processed parquet
REGION_FILES = {
    "NSW1": "data/aemo/processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet",
    "QLD1": "data/aemo/processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet",
    "SA1": "data/aemo/processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet",
    "TAS1": "data/aemo/processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet",
    "VIC1": "data/aemo/processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet",
}


def generate_slot(
    processed: pl.DataFrame,
    region: str,
    horizon: str,
    battery_name: str,
    n_eps: int,
    rng: np.random.Generator,
    model: DecisionTransformer,
    deg_cost_per_mwh: float,
    rtg_value: float,
    seed_base: int,
) -> list[pl.DataFrame]:
    cap = BATTERY_SPECS[battery_name]["capacity"]
    max_flow = BATTERY_SPECS[battery_name]["max_flow"]
    n_steps = HORIZON_STEPS[horizon]
    max_start = processed.height - n_steps
    frames = []
    for ep in range(n_eps):
        if max_start <= 0:
            break
        start = int(rng.integers(0, max_start))
        sliced = processed.slice(start, n_steps)
        env = create_aemo_env(
            processed_data=sliced,
            battery_variant={"battery_capacity": cap, "max_battery_flow": max_flow,
                             "init_soc": cap * 0.5, "battery_life_cost": 1312500.0},
            max_step=n_steps, step_duration=5.0 / 60.0, action_mode="full_fcas",
            random_episode_start=False,
        )
        agent = AEMOAgent(env, algorithm="dt_soc_oracle", model=model,
                          rtg_value=rtg_value, executor="sdp",
                          deg_cost_per_mwh=deg_cost_per_mwh,
                          reset_seed=seed_base + ep)
        episode_df, _ = agent.run_episode()
        episode_df = episode_df.with_columns(
            pl.lit(f"{region}__{horizon}__{battery_name}__ep{ep:03d}").alias("episode_id"),
            pl.lit("sdp_teacher").alias("source_policy"),
            pl.lit(region).alias("region"),
            pl.lit(battery_name).alias("battery"),
        )
        frames.append(episode_df)
    return frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", nargs="+", default=["NSW1", "QLD1", "SA1", "TAS1", "VIC1"])
    p.add_argument("--horizons", nargs="+", default=["short", "medium"])
    p.add_argument("--batteries", nargs="+", default=["medium_1c", "fast_375c"])
    p.add_argument("--episodes-per-slot", type=int, default=8)
    p.add_argument("--model-manifest", type=Path,
                   default=Path("models/aemo/dt/soc_waypoint_dt_loss_surface_manifest.json"))
    p.add_argument("--model-path", type=Path,
                   default=Path("models/aemo/dt/soc_waypoint_dt_best.pt"))
    p.add_argument("--deg-cost-per-mwh", type=float, default=50.0)
    p.add_argument("--rtg-value", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("data/aemo_dt_sdp/dt_trajectories.parquet"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    import json
    manifest = json.loads(args.model_manifest.read_text())
    model = DecisionTransformer(**manifest["model_kwargs"])
    model.load_from_checkpoint(str(args.model_path), map_location="cpu")
    model.eval()

    rng = np.random.default_rng(args.seed)
    all_frames: list[pl.DataFrame] = []
    for region in args.regions:
        src = REGION_FILES.get(region)
        if src is None or not Path(src).exists():
            print(f"[stage_b] skip {region}: no training parquet")
            continue
        processed = pl.read_parquet(src)
        print(f"[stage_b] {region}: {processed.height} rows", flush=True)
        for horizon in args.horizons:
            for battery in args.batteries:
                frames = generate_slot(
                    processed, region, horizon, battery, args.episodes_per_slot,
                    rng, model, args.deg_cost_per_mwh, args.rtg_value, args.seed,
                )
                n_rows = sum(f.height for f in frames)
                print(f"  {horizon}/{battery}: {len(frames)} eps, {n_rows} rows", flush=True)
                all_frames.extend(frames)

    combined = pl.concat(all_frames) if all_frames else pl.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(args.out)
    print(f"\nWrote {args.out} ({combined.height} rows, "
          f"{combined['episode_id'].n_unique()} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
