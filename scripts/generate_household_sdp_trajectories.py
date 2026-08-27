#!/usr/bin/env python3
"""Generate deterministic SDP-teacher household trajectories for H2 DT training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from EnergySimEnv import SolarBatteryEnv
from household_optimization import optimize_dispatch
from household_replay import Tariff


def tariff_for_name(name: str) -> Tariff:
    if name == "realistic":
        return Tariff(
            import_cents_per_kwh=31.042,
            feed_in_cents_per_kwh=1.0,
            free_window_start_hour=11,
            free_window_end_hour=14,
        )
    if name == "legacy_flat":
        return Tariff(
            import_cents_per_kwh=30.0,
            feed_in_cents_per_kwh=5.0,
            free_window_start_hour=24,
            free_window_end_hour=24,
        )
    raise ValueError(f"Unsupported tariff {name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth-dir", type=Path, default=ROOT / "data/household/synth")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--out", type=Path, default=ROOT / "data/household/dt/sdp_teacher_train.parquet")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--soc-resolution", type=int, default=31)
    parser.add_argument("--action-resolution", type=int, default=21)
    parser.add_argument("--tariff", choices=("realistic", "legacy_flat"), default="realistic")
    return parser.parse_args()


def _trajectory_for_episode(
    frame: pl.DataFrame, episode_id: int, capacity: float, flow: float,
    soc_resolution: int, action_resolution: int, tariff: Tariff,
) -> pl.DataFrame:
    """Roll out per-day deterministic DP actions through the actual env."""
    priced_frame = frame.with_columns([
        pl.Series("ImportEnergyPrice", [tariff.import_price(ts) for ts in frame["Timestamp"]]),
        pl.lit(tariff.feed_in_price()).alias("ExportEnergyPrice"),
    ])
    env = SolarBatteryEnv(
        priced_frame, battery_capacity=capacity, max_battery_flow=flow,
        init_battery_level=capacity / 2.0, max_step=len(frame),
    )
    observation, _ = env.reset()
    records = []
    for day_start in range(0, len(frame), 288):
        day = frame.slice(day_start, 288)
        raw_kw = day.with_columns([
            (pl.col("HouseLoad") * 12.0).alias("HouseLoad"),
            (pl.col("SolarGen") * 12.0).alias("SolarGen"),
        ])
        solution = optimize_dispatch(
            raw_kw, tariff=tariff, capacity_kwh=capacity, max_flow_kw=flow,
            roundtrip_eff=1.0, initial_soc=env.battery_level / capacity,
            soc_resolution=soc_resolution, action_resolution=action_resolution,
        )
        for offset, action_kw in enumerate(solution.actions_kw):
            step = day_start + offset
            state_index = int(np.argmin(np.abs(solution.soc_levels_kwh - env.battery_level)))
            # Reward is -cost, so RTG is negative cost-to-go at the state.
            rtg_value = -float(solution.cost_to_go[offset, state_index])
            next_observation, reward, terminated, truncated, _ = env.step([
                float(action_kw / flow)
            ])
            records.append({
                "episode_id": str(episode_id),
                "step": step,
                "norm_observation": observation.tolist(),
                "action": [float(action_kw / flow)],
                "reward": float(reward),
                "rtg_value": rtg_value,
                "source_policy": "sdp_teacher",
            })
            observation = next_observation
            if terminated or truncated:
                break
        if env.current_step >= len(frame):
            break
    return pl.DataFrame(records)


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.synth_dir / "manifest.json").read_text())
    episodes = [entry for entry in manifest["episodes"] if entry["split"] == args.split]
    if args.limit_episodes is not None:
        episodes = episodes[:args.limit_episodes]
    if not episodes:
        raise ValueError(f"No {args.split} episodes in {args.synth_dir / 'manifest.json'}")
    frames = []
    tariff = tariff_for_name(args.tariff)
    for entry in episodes:
        frame = pl.read_parquet(args.synth_dir / entry["path"])
        battery = entry["battery"]
        frames.append(_trajectory_for_episode(
            frame, entry["episode_id"], float(battery["capacity_kwh"]),
            float(battery["max_flow_kw"]), args.soc_resolution, args.action_resolution, tariff,
        ))
    result = pl.concat(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(args.out)
    print(f"Wrote {result.height} rows from {len(frames)} {args.split} SDP-teacher episodes to {args.out}")


if __name__ == "__main__":
    main()
