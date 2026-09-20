#!/usr/bin/env python3
"""Paired forecast-ablation statistics for household DT evaluation summaries.

Reads multiple ``summary.json`` files produced by
``scripts/evaluate_household_ood_baselines.py`` on the SAME ordered evaluation
windows and reports, for every pair of policies across runs:

* mean paired savings difference (AUD/year),
* paired window-level bootstrap 95% CI,
* win count,
* one-sided Wilcoxon signed-rank p-value (H1: first policy saves more).

Savings per window are ``no_battery_bill - policy_bill`` annualized, so the
no-battery term cancels in the paired difference and both runs must share
identical windows.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Run name -> summary.json path; repeatable. Each summary must "
             "contain the same windows in the same order.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy",
        default="dt",
        help="Policy key in each summary.json to score against no_battery "
             "(default 'dt'; use 'ppo' when scoring PPO).",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _annualized_savings(summary: dict, policy: str) -> tuple[np.ndarray, list[dict]]:
    windows = summary["windows"]
    days = np.asarray([w["days"] for w in windows], dtype=float)
    no_battery = np.asarray(summary["results"]["no_battery"]["segment_bills_aud"], dtype=float)
    if policy not in summary["results"]:
        raise ValueError(f"Policy {policy!r} not found in summary results {list(summary['results'])}")
    bills = np.asarray(summary["results"][policy]["segment_bills_aud"], dtype=float)
    savings = (no_battery - bills) / days * 365.0
    return savings, windows


def _paired_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> dict:
    diff = a - b
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_bootstrap)])
    wins = int(np.sum(diff > 0))
    nonzero = diff[diff != 0]
    if len(nonzero) >= 6:
        p_two = float(stats.wilcoxon(nonzero, alternative="two-sided").pvalue)
        p_greater = float(stats.wilcoxon(nonzero, alternative="greater").pvalue)
        p_less = float(stats.wilcoxon(nonzero, alternative="less").pvalue)
    else:
        p_two = p_greater = p_less = float("nan")
    return {
        "mean_diff_aud_per_year": float(diff.mean()),
        "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wins_a": wins,
        "wins_b": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
        "n_windows": int(len(diff)),
        "wilcoxon_two_sided_p": p_two,
        "wilcoxon_p_a_greater": p_greater,
        "wilcoxon_p_b_greater": p_less,
    }


def main() -> None:
    args = parse_args()
    runs: dict[str, np.ndarray] = {}
    windows: list[dict] | None = None
    for spec in args.summary:
        name, _, path = spec.partition("=")
        summary = json.loads(Path(path).read_text())
        savings, run_windows = _annualized_savings(summary, args.policy)
        if windows is not None and run_windows != windows:
            raise ValueError(f"Window provenance differs between runs: {name}")
        windows = run_windows
        runs[name] = savings
    rng = np.random.default_rng(args.seed)
    report: dict[str, object] = {
        "surface": summary["surface"],
        "tariff": summary["tariff"],
        "windows": windows,
        "mean_annualized_savings_aud_per_year": {
            name: float(values.mean()) for name, values in runs.items()
        },
        "savings_ci95": {
            name: [
                float(np.percentile([values[rng.integers(0, len(values), len(values))].mean() for _ in range(args.bootstrap)], 2.5)),
                float(np.percentile([values[rng.integers(0, len(values), len(values))].mean() for _ in range(args.bootstrap)], 97.5)),
            ]
            for name, values in runs.items()
        },
        "paired": {},
    }
    for a, b in itertools.combinations(runs, 2):
        report["paired"][f"{a}_vs_{b}"] = _paired_stats(runs[a], runs[b], rng, args.bootstrap)
    if windows and any("horizon" in w for w in windows):
        indices_by_group: dict[str, list[int]] = {}
        for index, window in enumerate(windows):
            indices_by_group.setdefault(str(window.get("horizon")), []).append(index)
        report["per_group"] = {
            group: {
                f"{a}_vs_{b}": _paired_stats(
                    runs[a][np.asarray(idx)], runs[b][np.asarray(idx)],
                    np.random.default_rng(args.seed), args.bootstrap,
                )
                for a, b in itertools.combinations(runs, 2)
            }
            for group, idx in sorted(indices_by_group.items())
        }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
