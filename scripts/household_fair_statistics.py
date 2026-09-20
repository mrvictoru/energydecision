#!/usr/bin/env python3
"""Paired statistics for fair household DT versus SB3 evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _window_key(window: dict) -> tuple:
    return (
        window.get("episode_path"),
        window.get("source_segment"),
        window.get("start"),
        window.get("end"),
    )


def _paired_bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _compare(dt: dict, baseline: dict, baseline_name: str, draws: int, seed: int) -> dict:
    dt_name = next(name for name in dt["results"] if name != "no_battery")
    baseline_key = next(name for name in baseline["results"] if name != "no_battery")
    dt_windows = {_window_key(window): index for index, window in enumerate(dt["windows"])}
    baseline_windows = {_window_key(window): index for index, window in enumerate(baseline["windows"])}
    keys = [key for key in dt_windows if key in baseline_windows]
    if not keys:
        raise ValueError(f"No matching windows for {baseline_name}")

    dt_result = dt["results"][dt_name]
    baseline_result = baseline["results"][baseline_key]
    dt_values = np.array(
        [dt_result["net_savings_segment_aud_per_year"][dt_windows[key]] for key in keys],
        dtype=float,
    )
    baseline_values = np.array(
        [baseline_result["net_savings_segment_aud_per_year"][baseline_windows[key]] for key in keys],
        dtype=float,
    )
    differences = dt_values - baseline_values
    if np.allclose(differences, 0.0):
        p_value = 1.0
        statistic = 0.0
    else:
        test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    lower, upper = _paired_bootstrap(differences, np.random.default_rng(seed), draws)
    return {
        "baseline": baseline_name,
        "n_windows": len(keys),
        "mean_dt_savings_aud_per_year": float(dt_values.mean()),
        "mean_baseline_savings_aud_per_year": float(baseline_values.mean()),
        "mean_paired_difference_aud_per_year": float(differences.mean()),
        "bootstrap_ci95_aud_per_year": [lower, upper],
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt-summary", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_paths = sorted(args.baseline_root.glob("*/summary.json"))
    if not baseline_paths:
        raise FileNotFoundError(f"No summary.json files under {args.baseline_root}")
    dt = _load(args.dt_summary)
    comparisons = []
    for index, path in enumerate(baseline_paths):
        if path.resolve() == args.dt_summary.resolve():
            continue
        try:
            comparisons.append(
                _compare(dt, _load(path), path.parent.name, args.draws, args.seed + index)
            )
        except ValueError as error:
            if "No matching windows" not in str(error):
                raise
    result = {
        "dt_summary": str(args.dt_summary),
        "baseline_root": str(args.baseline_root),
        "metric": "net_savings_segment_aud_per_year",
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
