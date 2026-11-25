#!/usr/bin/env python3
"""Build the paper's within-cell three-seed PPO aggregate from audit outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SURFACES = (
    "standard_year_2024",
    "dispatch_year_2024",
    "expanded_broad_2024",
    "2025_ood",
)
SEEDS = ("tuned_seed0", "tuned_seed1", "tuned_seed2")


def percentile_stats(dt: np.ndarray, ppo: np.ndarray, seed: int, n_boot: int) -> dict:
    diff = dt - ppo
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    return {
        "n": len(diff),
        "dt_mean": float(dt.mean()),
        "dt_ci": [float(x) for x in np.percentile(dt[idx].mean(axis=1), [2.5, 97.5])],
        "ppo_mean": float(ppo.mean()),
        "ppo_ci": [float(x) for x in np.percentile(ppo[idx].mean(axis=1), [2.5, 97.5])],
        "ratio_of_means": float(dt.mean() / ppo.mean()),
        "diff_mean": float(diff.mean()),
        "diff_ci": [float(x) for x in np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])],
        "win_rate": float((diff > 0).mean()),
        "wilcoxon_p": float(wilcoxon(diff).pvalue),
    }


def build(args: argparse.Namespace) -> dict:
    stats = json.loads(Path(args.stats).read_text())
    out = {
        "schema": "energydecision.ppo_retrain_seed_aggregate.v1",
        "generated_from": str(Path(args.stats)),
        "seed_rule": "Mean PPO result per matched (scenario,battery) cell across tuned_seed0/1/2",
        "n_boot": args.boot,
        "seed": args.seed,
        "surfaces": {},
    }
    for surface in SURFACES:
        maps = []
        for tag in SEEDS:
            maps.append({
                (cell["scenario_label"], cell["battery_label"]): cell
                for cell in stats["surface_seeds"][surface][tag]["cells"]
            })
        keys = sorted(set.intersection(*(set(one) for one in maps)))
        if any(set(one) != set(keys) for one in maps):
            raise ValueError(f"PPO seed cell mismatch for {surface}")
        dt = np.asarray([maps[0][key]["dt"] for key in keys], dtype=float)
        ppo_seeds = np.asarray([[one[key]["ppo"] for key in keys] for one in maps], dtype=float)
        row = percentile_stats(dt, ppo_seeds.mean(axis=0), args.seed, args.boot)
        row["ppo_seed_means"] = [float(v.mean()) for v in ppo_seeds]
        row["ci_excludes_zero"] = row["diff_ci"][0] > 0 or row["diff_ci"][1] < 0
        out["surfaces"][surface] = row

    impact_files = [Path(args.impact_seed0), Path(args.impact_seed1), Path(args.impact_seed2)]
    dt_by_key: dict[str, float] = {}
    ppo_maps: list[dict[str, float]] = []
    for path in impact_files:
        one: dict[str, float] = {}
        for row in json.loads(path.read_text()):
            label = row["label"]
            if "piecewise_merit_order" not in label:
                continue
            if label.startswith("piecewise_merit_order_stagec_v2cal_auto_rtg0.0_"):
                dt_by_key[label.split("rtg0.0_", 1)[1]] = float(row["profit"])
            elif label.startswith("piecewise_merit_order_ppo_"):
                one[label.split("_ppo_", 1)[1]] = float(row["profit"])
        ppo_maps.append(one)
    keys = sorted(dt_by_key)
    if any(set(one) != set(keys) for one in ppo_maps):
        raise ValueError("Impact PPO seed cell mismatch")
    dt = np.asarray([dt_by_key[key] for key in keys])
    ppo_seeds = np.asarray([[one[key] for key in keys] for one in ppo_maps])
    impact = percentile_stats(dt, ppo_seeds.mean(axis=0), args.seed, args.boot)
    impact.update({
        "impact": "piecewise_merit_order",
        "per_battery_ratio_of_means": {},
        "ppo_seed_ratios": [float(dt.sum() / seed.sum()) for seed in ppo_seeds],
        "ppo_seed_win_rates": [float((dt > seed).mean()) for seed in ppo_seeds],
        "ci_excludes_zero": impact["diff_ci"][0] > 0 or impact["diff_ci"][1] < 0,
    })
    for battery in ("small", "hornsdale", "torrens"):
        ix = [i for i, key in enumerate(keys) if key.startswith(battery + "_")]
        impact["per_battery_ratio_of_means"][battery] = float(
            dt[ix].mean() / ppo_seeds[:, ix].mean()
        )
    out["impact"] = impact
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="paper/audit/artifacts/ppo_retrain_stats_20261125.json")
    ap.add_argument("--impact-seed0", default="eval_output/ppo_retrain_20261125/phase3_impact/results.json")
    ap.add_argument("--impact-seed1", default="eval_output/ppo_retrain_20261125/phase3_impact/results_seed1_ppo_only.json")
    ap.add_argument("--impact-seed2", default="eval_output/ppo_retrain_20261125/phase3_impact/results_seed2_ppo_only.json")
    ap.add_argument("--out", default="paper/audit/artifacts/ppo_retrain_seed_aggregate_20261125.json")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    payload = build(args)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
