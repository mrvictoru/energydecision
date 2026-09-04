#!/usr/bin/env python3
"""Compare observed household dispatch with re-optimized tariff dispatch."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from household_ingest import split_segments
from household_optimization import bootstrap_mean_ci, optimize_dispatch
from household_replay import Tariff, detect_action_sign, load_normalized_year, replay


TARIFFS = {
    "flat": Tariff(import_cents_per_kwh=31.042, feed_in_cents_per_kwh=1.0,
                   free_window_start_hour=24, free_window_end_hour=24),
    "tou_free": Tariff(import_cents_per_kwh=31.042, feed_in_cents_per_kwh=1.0,
                       free_window_start_hour=11, free_window_end_hour=14),
    # Public spot prices are intentionally not invented. Supply a priced
    # dataset/adapter before claiming a spot-pass-through result.
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=ROOT / "data/household/real/normalized")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_output/household/tariff_optimization")
    parser.add_argument("--capacity-kwh", type=float, default=5.0)
    parser.add_argument("--max-flow-kw", type=float, default=3.3)
    parser.add_argument("--roundtrip-eff", type=float, default=0.80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel worker processes for day optimization (default: min(8, cpu_count)).",
    )
    return parser.parse_args()


def _evaluate_single_day(
    args_tuple: tuple[pl.DataFrame, float, float, Tariff, int, float]
) -> tuple[float, float, float]:
    day, capacity_kwh, max_flow_kw, tariff, action_sign, roundtrip_eff = args_tuple
    observed = replay(
        day, capacity_kwh, max_flow_kw, tariff,
        action_sign=action_sign, roundtrip_eff=roundtrip_eff,
    )
    optimized = optimize_dispatch(
        day, tariff=tariff, capacity_kwh=capacity_kwh,
        max_flow_kw=max_flow_kw, roundtrip_eff=roundtrip_eff,
    )
    no_battery = float(sum(
        max(load - solar, 0.0) * (5.0 / 60.0) * tariff.import_price(ts)
        - max(solar - load, 0.0) * (5.0 / 60.0) * tariff.feed_in_price()
        for ts, load, solar in zip(day["Timestamp"], day["HouseLoad"], day["SolarGen"])
    ))
    return float(observed["bill_aud"]), float(optimized.bill_aud), no_battery


def complete_days(segments: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """Return complete days independently, never joining gap-separated segments."""
    result = []
    for segment in segments:
        with_date = segment.with_columns(pl.col("Timestamp").dt.date().alias("_date"))
        for day in with_date.partition_by("_date", maintain_order=True):
            if len(day) == 288:
                result.append(day.drop("_date"))
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Optimization and recorded-action replay consume normalized kW. In
    # contrast, build_year_dataset() converts to the environment's kWh/step
    # representation and must not be used here.
    days = complete_days(split_segments(load_normalized_year(args.normalized_dir)))
    if not days:
        raise ValueError("No complete contiguous real days are available")

    output = {
        "hardware": {
            "capacity_kwh": args.capacity_kwh,
            "max_flow_kw": args.max_flow_kw,
            "roundtrip_eff": args.roundtrip_eff,
        },
        "evaluation_unit": "complete contiguous real day; gap seams are never crossed",
        "days": len(days),
        "tariffs": {},
    }
    for name, tariff in TARIFFS.items():
        # Sign fitting needs the full observed SOC trajectory. Per-day fitting
        # is underidentified and can invert the recorded convention.
        action_sign = detect_action_sign(
            pl.concat(days), tariff, args.capacity_kwh, args.max_flow_kw
        )
        day_tasks = [
            (day, args.capacity_kwh, args.max_flow_kw, tariff, action_sign, args.roundtrip_eff)
            for day in days
        ]
        if args.workers > 1 and len(days) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(_evaluate_single_day, day_tasks, chunksize=16))
        else:
            results = [_evaluate_single_day(task) for task in day_tasks]

        observed_bills = [r[0] for r in results]
        optimized_bills = [r[1] for r in results]
        no_battery_bills = [r[2] for r in results]
        gap = [observed - optimized for observed, optimized in zip(observed_bills, optimized_bills)]
        output["tariffs"][name] = {
            "definition": {
                "import_cents_per_kwh": tariff.import_cents_per_kwh,
                "feed_in_cents_per_kwh": tariff.feed_in_cents_per_kwh,
                "free_window": [tariff.free_window_start_hour, tariff.free_window_end_hour],
            },
            "observed_action_sign": action_sign,
            "observed_replay_bill_aud_per_day": bootstrap_mean_ci(observed_bills, n_bootstrap=args.bootstrap, seed=args.seed),
            "optimized_bill_aud_per_day": bootstrap_mean_ci(optimized_bills, n_bootstrap=args.bootstrap, seed=args.seed),
            "optimization_gap_aud_per_day": bootstrap_mean_ci(gap, n_bootstrap=args.bootstrap, seed=args.seed),
            "no_battery_bill_aud_per_day": bootstrap_mean_ci(no_battery_bills, n_bootstrap=args.bootstrap, seed=args.seed),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
