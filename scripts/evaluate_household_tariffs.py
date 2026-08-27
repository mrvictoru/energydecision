#!/usr/bin/env python3
"""Compare observed household dispatch with re-optimized tariff dispatch."""
from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


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
        observed_bills = []
        optimized_bills = []
        no_battery_bills = []
        # Sign fitting needs the full observed SOC trajectory. Per-day fitting
        # is underidentified and can invert the recorded convention.
        action_sign = detect_action_sign(
            pl.concat(days), tariff, args.capacity_kwh, args.max_flow_kw
        )
        for day in days:
            observed = replay(
                day, args.capacity_kwh, args.max_flow_kw, tariff,
                action_sign=action_sign, roundtrip_eff=args.roundtrip_eff,
            )
            optimized = optimize_dispatch(
                day, tariff=tariff, capacity_kwh=args.capacity_kwh,
                max_flow_kw=args.max_flow_kw, roundtrip_eff=args.roundtrip_eff,
            )
            net = (day["HouseLoad"] - day["SolarGen"]).sum() * (5.0 / 60.0)
            # Calculate the no-battery day bill with zero actions through replay's
            # tariff-consistent grid accounting.
            no_battery = float(sum(
                max(load - solar, 0.0) * (5.0 / 60.0) * tariff.import_price(ts)
                - max(solar - load, 0.0) * (5.0 / 60.0) * tariff.feed_in_price()
                for ts, load, solar in zip(day["Timestamp"], day["HouseLoad"], day["SolarGen"])
            ))
            del net
            observed_bills.append(observed["bill_aud"])
            optimized_bills.append(optimized.bill_aud)
            no_battery_bills.append(no_battery)
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
