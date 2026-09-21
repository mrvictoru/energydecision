"""Provenance-audit companion: reproducible identity-surface significance for
Stage C DT vs PPO, parameterised by surface directory.

The shipped `scripts/stagec_statistical_significance.py` hardcodes the v1
`stagec_jtsoc_*_rtgjtsoc` directories, so its output
(`eval_output/stagec_statistical_significance.json`) describes the **v1
fullcorpus** checkpoint, not the shipped physics-v2 `aemo_dt_sdp_jtsoc_v2cal.pt`.
This script makes the surface directories explicit so the v2cal headline CIs
(report.md §8.2.0 line 735) are traceable to a saved artifact.

Usage:
    python3 paper/audit/reproduce_significance.py \
        --out paper/audit/artifacts/stagec_v2cal_significance.json

Reads `heldout_metrics_by_scenario.csv` from each surface dir (columns
`policy_name`, `scenario_label`, `avg_profit_per_episode`).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]

SURFACE_SETS = {
    # Shipped physics-v2 checkpoint.
    "v2cal": {
        "standard_oct": "eval_output/physics_v2/standard_v2cal",
        "dispatch_matched": "eval_output/physics_v2/dispatch_v2cal",
        "expanded_broad_2024": "eval_output/physics_v2/expanded_v2cal",
        "2025_ood": "eval_output/physics_v2/2025_v2cal",
    },
    # Superseded v1 fullcorpus checkpoint (what the shipped significance script
    # actually reads). Kept for side-by-side comparison.
    "v1_fullcorpus": {
        "standard_oct": "eval_output/stagec_jtsoc_standard_rtgjtsoc",
        "dispatch_matched": "eval_output/stagec_jtsoc_dispatch_rtgjtsoc",
        "expanded_broad_2024": "eval_output/stagec_jtsoc_expanded_rtgjtsoc",
        "2025_ood": "eval_output/stagec_jtsoc_2025_rtgjtsoc",
    },
}


def bootstrap_ci(values, n_boot, rng, conf=0.95):
    vals = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    alpha = (1.0 - conf) / 2.0
    return (
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
        float((means > 0).mean()),
    )


def paired_stats(dt_vals, ppo_vals, label, n_boot, seed):
    dt = np.asarray(dt_vals, dtype=float)
    ppo = np.asarray(ppo_vals, dtype=float)
    assert len(dt) == len(ppo), f"{label}: unpaired cells"
    diffs = dt - ppo
    rng = np.random.default_rng(seed)
    lo_d, hi_d, p_gt = bootstrap_ci(diffs, n_boot, rng)
    lo_dt, hi_dt, _ = bootstrap_ci(dt, n_boot, rng)
    lo_ppo, hi_ppo, _ = bootstrap_ci(ppo, n_boot, rng)
    nonzero = diffs[diffs != 0]
    if len(nonzero) >= 1 and not np.all(nonzero == nonzero[0]):
        try:
            _, w_p = stats.wilcoxon(diffs)
        except ValueError:
            w_p = float("nan")
    else:
        w_p = float("nan")
    return {
        "surface": label,
        "n": len(diffs),
        "dt_mean": float(dt.mean()),
        "dt_ci": [lo_dt, hi_dt],
        "ppo_mean": float(ppo.mean()),
        "ppo_ci": [lo_ppo, hi_ppo],
        "diff_mean": float(diffs.mean()),
        "diff_ci": [lo_d, hi_d],
        "ci_excludes_zero": bool(lo_d > 0),
        "p_dt_greater": p_gt,
        "win_rate": float((diffs > 0).mean()),
        "wilcoxon_p": None if np.isnan(w_p) else float(w_p),
    }


def run_set(name, surfaces, n_boot, seed):
    out = []
    for label, rel in surfaces.items():
        csv_path = ROOT / rel / "heldout_metrics_by_scenario.csv"
        df = pl.read_csv(csv_path)
        dt = df.filter(pl.col("policy_name") == "candidate_dt").sort("scenario_label")
        ppo = df.filter(pl.col("policy_name") == "ppo_reference").sort("scenario_label")
        if dt.height != ppo.height:
            raise RuntimeError(f"{name}/{label}: row mismatch {dt.height} vs {ppo.height}")
        out.append(
            paired_stats(
                dt["avg_profit_per_episode"].to_list(),
                ppo["avg_profit_per_episode"].to_list(),
                label,
                n_boot,
                seed,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    payload = {
        "method": "bootstrap 95% CI over matched scenario cells + paired Wilcoxon; "
                  "primary evidence is the paired-difference CI (n<10 bounds Wilcoxon).",
        "n_boot": args.boot,
        "seed": args.seed,
        "generated_by": "paper/audit/reproduce_significance.py",
        "surface_sets": {},
    }
    for name, surfaces in SURFACE_SETS.items():
        payload["surface_sets"][name] = run_set(name, surfaces, args.boot, args.seed)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
