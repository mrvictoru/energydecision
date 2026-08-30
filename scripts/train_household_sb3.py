#!/usr/bin/env python3
"""Train and evaluate a fresh SB3 household policy on synthetic episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from EnergySimEnv import SolarBatteryEnv
from household_replay import Tariff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/household/sb3/h4_3")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=12)
    parser.add_argument("--capacity-kwh", type=float, default=10.0)
    parser.add_argument("--max-flow-kw", type=float, default=5.0)
    parser.add_argument("--battery-life-cost", type=float, default=5000.0)
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def _tariff_frame(frame: pl.DataFrame) -> pl.DataFrame:
    tariff = Tariff(31.042, 1.0, 11, 14)
    hours = frame["Timestamp"].dt.hour()
    import_price = np.where(
        (hours.to_numpy() >= tariff.free_window_start_hour)
        & (hours.to_numpy() < tariff.free_window_end_hour),
        0.0,
        tariff.import_cents_per_kwh / 100.0,
    )
    return frame.with_columns([
        pl.Series("ImportEnergyPrice", import_price),
        pl.lit(tariff.feed_in_cents_per_kwh / 100.0).alias("ExportEnergyPrice"),
    ])


def _load_entries(corpus_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    return manifest, manifest["episodes"]


def _make_env(frame: pl.DataFrame, capacity: float, flow: float, life_cost: float):
    def factory() -> SolarBatteryEnv:
        return SolarBatteryEnv(
            frame,
            battery_capacity=capacity,
            max_battery_flow=flow,
            init_battery_level=capacity / 2.0,
            max_step=len(frame),
            battery_life_cost=life_cost,
        )

    return factory


def _episode_return(model: PPO, frame: pl.DataFrame, capacity: float, flow: float, life_cost: float) -> float:
    env = _make_env(frame, capacity, flow, life_cost)()
    obs, _ = env.reset()
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
    env.close()
    return total


def main() -> None:
    args = parse_args()
    if args.timesteps < 1 or args.n_envs < 1:
        raise ValueError("--timesteps and --n-envs must be positive")
    manifest, entries = _load_entries(args.corpus_dir)
    frames: dict[str, pl.DataFrame] = {}
    for entry in entries:
        if entry["split"] == "train":
            frames[entry["path"]] = _tariff_frame(pl.read_parquet(args.corpus_dir / entry["path"]))
    if not frames:
        raise ValueError("Corpus has no training episodes")
    paths = list(frames)
    n_envs = min(args.n_envs, len(paths))
    env_fns = [
        _make_env(frames[paths[index % len(paths)]], args.capacity_kwh, args.max_flow_kw, args.battery_life_cost)
        for index in range(n_envs)
    ]
    vec_env = SubprocVecEnv(env_fns, start_method="forkserver")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs={"net_arch": [256, 256]},
        device="cpu",
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "ppo_h4_3_modern.zip"
    model.save(model_path)
    vec_env.close()

    evaluation: dict[str, list[float]] = {}
    for split in ("val", "test"):
        returns = []
        for entry in entries:
            if entry["split"] != split:
                continue
            frame = _tariff_frame(pl.read_parquet(args.corpus_dir / entry["path"]))
            returns.append(_episode_return(model, frame, args.capacity_kwh, args.max_flow_kw, args.battery_life_cost))
        evaluation[split] = returns
    metadata = {
        "corpus_manifest_schema": manifest["schema_version"],
        "corpus_dir": str(args.corpus_dir),
        "model_path": str(model_path),
        "algorithm": "PPO",
        "training_split": "train",
        "timesteps": args.timesteps,
        "n_envs": n_envs,
        "seed": args.seed,
        "tariff": {"import_cents": 31.042, "free_start": 11, "free_end": 14, "export_cents": 1.0},
        "battery": {
            "capacity_kwh": args.capacity_kwh,
            "max_flow_kw": args.max_flow_kw,
            "battery_life_cost": args.battery_life_cost,
        },
        "evaluation": {
            split: {
                "episodes": len(values),
                "mean_reward": float(np.mean(values)) if values else None,
                "std_reward": float(np.std(values)) if values else None,
            }
            for split, values in evaluation.items()
        },
    }
    (args.output_dir / "training_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["evaluation"], indent=2))


if __name__ == "__main__":
    main()
