#!/usr/bin/env python3
"""Evaluate legacy household policies on gap-separated real-data OOD segments."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from EnergySimEnv import SolarBatteryEnv
from decision import Agent
from decision_transformer import DecisionTransformer
from household_ingest import build_year_dataset, env_view, split_segments
from household_optimization import (
    apply_tariff,
    bootstrap_mean_ci,
    build_j_t_soc_prompt_provider,
    optimize_dispatch,
)
from household_replay import Tariff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=ROOT / "data/household/real/normalized")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_output/household/ood_baselines")
    parser.add_argument("--ppo-path", type=Path, default=ROOT / "models/household/sb3/ppo_model.zip")
    parser.add_argument("--dt-path", type=Path, default=ROOT / "models/household/dt/dt_model_new_best.pt")
    parser.add_argument("--dt-config", type=Path, default=ROOT / "models/household/dt/decision_transformer_model_kwargs.json")
    parser.add_argument("--dt-rtg-mode", choices=("standard", "j_t_soc"), default="standard")
    parser.add_argument(
        "--additional-dt",
        action="append",
        default=[],
        metavar="NAME:CHECKPOINT:CONFIG:RTG_MODE",
        help="Additional DT evaluated in the same run; RTG_MODE is standard or j_t_soc.",
    )
    parser.add_argument("--tariff", choices=("legacy_flat", "realistic"), default="legacy_flat")
    parser.add_argument("--capacity-kwh", type=float, default=5.0)
    parser.add_argument("--max-flow-kw", type=float, default=3.3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dt", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    return parser.parse_args()


def _bill_from_logs(frame: pl.DataFrame, logs: pl.DataFrame) -> float:
    """Price env-physical grid energy with import/export prices from the frame."""
    rows = logs["info"].to_list()
    bill = 0.0
    for index, info in enumerate(rows):
        grid_energy = float(info["grid_energy"])
        if grid_energy >= 0:
            bill += grid_energy * float(frame["ImportEnergyPrice"][index])
        else:
            bill += grid_energy * float(frame["ExportEnergyPrice"][index])
    return bill


def tariff_for_name(name: str) -> Tariff:
    if name == "realistic":
        return Tariff(31.042, 1.0, 11, 14)
    return Tariff(30.0, 5.0, 24, 24)


def _run_agent(
    frame: pl.DataFrame, algorithm: str, model, capacity: float, flow: float,
    tariff: Tariff, rtg_mode: str = "standard",
) -> float:
    env = SolarBatteryEnv(
        frame, battery_capacity=capacity, max_battery_flow=flow,
        init_battery_level=capacity / 2.0, max_step=len(frame),
    )
    prompt_provider = (
        build_j_t_soc_prompt_provider(
            frame, tariff=tariff, capacity_kwh=capacity, max_flow_kw=flow
        )
        if algorithm == "dt" and rtg_mode == "j_t_soc" else None
    )
    logs, _ = Agent(env, algorithm=algorithm, model=model, horizon=288,
                    soc_resolution=31, action_resolution=21,
                    rtg_prompt_provider=prompt_provider).run_episode()
    return _bill_from_logs(frame, logs)


def _oracle_bill(frame: pl.DataFrame, capacity: float, flow: float, tariff: Tariff) -> float:
    """Perfect-foresight daily DP, retaining the segment boundary discipline."""
    raw_kw = frame.with_columns([
        (pl.col("HouseLoad") * 12.0).alias("HouseLoad"),
        (pl.col("SolarGen") * 12.0).alias("SolarGen"),
    ])
    costs = []
    with_date = raw_kw.with_columns(pl.col("Timestamp").dt.date().alias("_date"))
    for day in with_date.partition_by("_date", maintain_order=True):
        if len(day) == 288:
            costs.append(optimize_dispatch(
                day.drop("_date"), tariff=tariff, capacity_kwh=capacity,
                max_flow_kw=flow, roundtrip_eff=1.0,
            ).bill_aud)
    return float(sum(costs))


def _load_dt(path: Path, config: Path) -> DecisionTransformer:
    model = DecisionTransformer(**json.loads(config.read_text()))
    model.load_from_checkpoint(str(path), map_location="cpu")
    return model


def _parse_additional_dt(spec: str) -> tuple[str, Path, Path, str]:
    parts = spec.split(":", 3)
    if len(parts) != 4 or parts[3] not in {"standard", "j_t_soc"}:
        raise ValueError("--additional-dt must be NAME:CHECKPOINT:CONFIG:RTG_MODE")
    return parts[0], Path(parts[1]), Path(parts[2]), parts[3]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tariff = tariff_for_name(args.tariff)
    segments = [
        apply_tariff(env_view(segment), tariff)
        for segment in split_segments(build_year_dataset(args.normalized_dir))
    ]
    if not segments:
        raise ValueError("No contiguous real OOD segments found")

    models: dict[str, tuple[object, str]] = {}
    if not args.skip_ppo:
        models["ppo"] = (PPO.load(str(args.ppo_path), device="cpu"), "standard")
    if not args.skip_dt:
        models["dt"] = (_load_dt(args.dt_path, args.dt_config), args.dt_rtg_mode)
    for spec in args.additional_dt:
        name, checkpoint, config, rtg_mode = _parse_additional_dt(spec)
        if name in models or name in {"rule", "oracle", "no_battery"}:
            raise ValueError(f"Duplicate or reserved policy name: {name}")
        models[name] = (_load_dt(checkpoint, config), rtg_mode)

    bills: dict[str, list[float]] = {"rule": [], "oracle": []}
    for name in models:
        bills[name] = []
    for segment in segments:
        bills["rule"].append(_run_agent(
            segment, "rule", None, args.capacity_kwh, args.max_flow_kw, tariff
        ))
        bills["oracle"].append(_oracle_bill(segment, args.capacity_kwh, args.max_flow_kw, tariff))
        for name, (model, rtg_mode) in models.items():
            bills[name].append(_run_agent(
                segment, "rl" if name == "ppo" else "dt", model,
                args.capacity_kwh, args.max_flow_kw, tariff, rtg_mode,
            ))

    no_battery = []
    for segment in segments:
        grid = segment["HouseLoad"] - segment["SolarGen"]
        no_battery.append(float(
            (grid.clip(lower_bound=0.0) * segment["ImportEnergyPrice"]).sum()
            - ((-grid).clip(lower_bound=0.0) * segment["ExportEnergyPrice"]).sum()
        ))
    segment_days = [len(segment) / 288.0 for segment in segments]
    output = {
        "surface": "real normalized telemetry; held-out OOD; each environment instantiated per contiguous segment",
        "hardware": {"capacity_kwh": args.capacity_kwh, "max_flow_kw": args.max_flow_kw},
        "tariff": args.tariff,
        "segments": len(segments),
        "results": {},
    }
    for name, values in {**bills, "no_battery": no_battery}.items():
        annualized = [bill / days * 365.0 for bill, days in zip(values, segment_days)]
        output["results"][name] = {
            "bill_aud_per_year": bootstrap_mean_ci(annualized, n_bootstrap=args.bootstrap, seed=args.seed),
            "segment_bills_aud": values,
        }
    base = np.asarray(output["results"]["no_battery"]["bill_aud_per_year"]["mean"])
    for name in bills:
        output["results"][name]["savings_vs_no_battery_aud_per_year"] = float(
            base - output["results"][name]["bill_aud_per_year"]["mean"]
        )
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
