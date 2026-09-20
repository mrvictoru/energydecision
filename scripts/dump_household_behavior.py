#!/usr/bin/env python3
"""Dump per-step rollout data for the household behaviour chart on the website.

Runs the matched full-corpus (H4.4) arms on ONE deterministic 7-day real-OOD
window — TTM-forecast DT, persistence DT, no-forecast DT, and the rule
baseline — under identical battery/tariff settings, and writes a JSON with
10-minute means of solar, load, import price, battery power, and SOC per policy.

Usage (from the repository root, GPU Distrobox):
    python3 scripts/dump_household_behavior.py --window 6 \
        --output eval_output/household/h4_4_behavior/week.json
    python3 scripts/dump_household_behavior.py --window 6 --policies persist --merge \
        --output eval_output/household/h4_4_behavior/week.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from EnergySimEnv import SolarBatteryEnv  # noqa: E402
from decision import Agent  # noqa: E402
from decision_transformer import DecisionTransformer  # noqa: E402
from evaluate_household_ood_baselines import (  # noqa: E402
    _apply_forecast_mode,
    _bounded_windows,
    tariff_for_name,
)
from household_forecast import apply_forecast_sidecar  # noqa: E402
from household_ingest import build_year_dataset, env_view, split_segments  # noqa: E402
from household_optimization import apply_tariff  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=ROOT / "data/household/real/normalized")
    parser.add_argument("--forecast-sidecar", type=Path,
                        default=ROOT / "data/household/real/household_ttm_forecasts.parquet")
    parser.add_argument("--dt-dir", type=Path, default=ROOT / "models/household/dt")
    parser.add_argument("--window", type=int, default=6,
                        help="Index among the evaluator's ten 7-day windows (0-based).")
    parser.add_argument("--capacity-kwh", type=float, default=5.0)
    parser.add_argument("--max-flow-kw", type=float, default=3.3)
    parser.add_argument("--rtg-value", type=float, default=-2.0)
    parser.add_argument("--bin-steps", type=int, default=2, help="5-min steps per output bin (2 = 10-min).")
    parser.add_argument(
        "--policies",
        default="ttm,nofc,persist,rule",
        help="Comma-separated arms to dump: ttm, nofc, persist, rule.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge dumped arms into an existing --output JSON instead of rewriting it.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_dt(path: Path, config: Path, device: str) -> DecisionTransformer:
    model = DecisionTransformer(**json.loads(config.read_text()))
    model.load_from_checkpoint(str(path), map_location=device)
    model.to(device)
    model.eval()
    return model


def rollout(frame: pl.DataFrame, algorithm: str, model, capacity: float, flow: float,
            rtg_value: float) -> pl.DataFrame:
    env = SolarBatteryEnv(
        frame, battery_capacity=capacity, max_battery_flow=flow,
        init_battery_level=capacity / 2.0, max_step=len(frame),
    )
    agent = Agent(env, algorithm=algorithm, model=model, horizon=288,
                  soc_resolution=31, action_resolution=21, rtg_value=rtg_value)
    logs, _ = agent.run_episode()
    return logs


def downsample(values: np.ndarray, bin_steps: int) -> list[float]:
    trimmed = values[: len(values) - (len(values) % bin_steps)]
    return trimmed.reshape(-1, bin_steps).mean(axis=1).tolist()


def main() -> None:
    args = parse_args()
    tariff = tariff_for_name("realistic")
    segments = [
        apply_tariff(env_view(segment), tariff)
        for segment in split_segments(build_year_dataset(args.normalized_dir))
    ]
    windows, provenance = _bounded_windows(segments, 7, 2)
    frame_base = windows[args.window]
    prov = provenance[args.window]
    sidecar = pl.read_parquet(args.forecast_sidecar)

    stems = {
        "ttm": "h4_4_ttm_standard_rtg_8x512_ctx576",
        "nofc": "h4_4_no_forecast_standard_rtg_8x512_ctx576",
        "persist": "h4_4_persistence_standard_rtg_8x512_ctx576",
    }
    wanted = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in {*stems, "rule"}]
    if unknown:
        raise SystemExit(f"unknown --policies: {unknown}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def pack(logs: pl.DataFrame) -> dict[str, list[float]]:
        info = logs["info"]
        return {
            "batt_kw": downsample(
                np.array([row["battery_flow_energy"] for row in info]) * 12.0,
                args.bin_steps,
            ),
            "soc_pct": downsample(
                np.array([row["battery_level"] for row in info]) / args.capacity_kwh * 100.0,
                args.bin_steps,
            ),
            "grid_kwh": downsample(
                np.array([row["grid_energy"] for row in info]),
                args.bin_steps,
            ),
        }

    series: dict[str, dict[str, list[float]]] = {}

    for arm, stem in stems.items():
        if arm not in wanted:
            continue
        if arm == "ttm":
            frame = apply_forecast_sidecar(frame_base, sidecar)
        elif arm == "persist":
            frame = _apply_forecast_mode(frame_base, "persistence", 42)
        else:
            frame = _apply_forecast_mode(frame_base, "zero", 42)
        model = load_dt(args.dt_dir / f"{stem}_best.pt", args.dt_dir / f"{stem}_model_kwargs.json", device)
        logs = rollout(frame, "dt", model, args.capacity_kwh, args.max_flow_kw, args.rtg_value)
        series[arm] = pack(logs)
        print(f"[behavior] {arm} done", flush=True)

    if "rule" in wanted:
        logs_rule = rollout(
            _apply_forecast_mode(frame_base, "persistence", 42),
            "rule", None, args.capacity_kwh, args.max_flow_kw, 0.0,
        )
        series["rule"] = pack(logs_rule)
        print("[behavior] rule done", flush=True)

    step_kw = 12.0  # kWh per 5-min step -> kW
    n = len(frame_base)
    timestamps: list[dt.datetime] = frame_base["Timestamp"].to_list()
    labels = [ts.strftime("%a %H:%M") for ts in timestamps[:: args.bin_steps]][: n // args.bin_steps]
    payload = {
        "schema": "energydecision.household_behavior.v1",
        "window": prov,
        "hardware": {"capacity_kwh": args.capacity_kwh, "max_flow_kw": args.max_flow_kw},
        "tariff": "realistic",
        "rtg_value": args.rtg_value,
        "bin_minutes": 5 * args.bin_steps,
        "labels": labels,
        "shared": {
            "solar_kw": downsample(frame_base["SolarGen"].to_numpy() * step_kw, args.bin_steps),
            "load_kw": downsample(frame_base["HouseLoad"].to_numpy() * step_kw, args.bin_steps),
            "price_c": downsample(frame_base["ImportEnergyPrice"].to_numpy() * 100.0, args.bin_steps),
        },
        "policies": series,
    }
    def _round(v: float) -> float:
        return round(float(v), 3)
    payload["shared"] = {k: [_round(v) for v in arr] for k, arr in payload["shared"].items()}
    for arm in payload["policies"]:
        payload["policies"][arm] = {k: [_round(v) for v in arr] for k, arr in payload["policies"][arm].items()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and args.output.exists():
        existing = json.loads(args.output.read_text())
        if existing.get("labels") != payload["labels"]:
            raise SystemExit("refusing --merge: labels do not match existing output")
        existing.setdefault("policies", {}).update(payload["policies"])
        payload = existing
    args.output.write_text(json.dumps(payload))
    print(f"[behavior] wrote {args.output} ({len(payload['labels'])} bins, policies={list(payload['policies'])})")


if __name__ == "__main__":
    main()
