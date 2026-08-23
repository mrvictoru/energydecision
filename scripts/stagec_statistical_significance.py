"""Statistical significance for the Stage C DT headlines (report.md §8.2.10).

Part A — identity surfaces: pairs candidate_dt vs ppo_reference per scenario from
the autoresearch evaluator's heldout_metrics_by_scenario.csv (_rtgjtsoc runs,
which are the shipped rtg_mode="auto" headline numbers).

Part B — impact gate: pairs stagec_h3h1_auto vs ppo per (battery, scenario)
from eval_output/phase3_impact/results.json under both identity and
piecewise_merit_order.

Methods: bootstrap CIs over matched cells (scenario-level resampling, the only
independent unit — each cell is one deterministic episode), plus the paired
Wilcoxon signed-rank test. With n<10 the Wilcoxon minimum attainable p is
bounded (n=5 -> 0.0625 two-sided), so it is reported as indicative; the
bootstrap CI on the paired difference is the primary evidence.

Usage: python3 scripts/stagec_statistical_significance.py [--boot N] [--seed S]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

SURFACES = {
    "standard_oct": "eval_output/stagec_jtsoc_standard_rtgjtsoc",
    "dispatch_matched": "eval_output/stagec_jtsoc_dispatch_rtgjtsoc",
    "expanded_broad_2024": "eval_output/stagec_jtsoc_expanded_rtgjtsoc",
    "2025_ood": "eval_output/stagec_jtsoc_2025_rtgjtsoc",
}

BOOT_DEFAULT = 10000


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
            w_stat, w_p = stats.wilcoxon(diffs)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), float("nan")

    print(f"\n=== {label} (n={len(diffs)} matched scenarios) ===")
    print(f"  DT  profit : ${dt.mean():>12,.0f}   95% CI [{lo_dt:>10,.0f}, {hi_dt:>10,.0f}]")
    print(f"  PPO profit : ${ppo.mean():>12,.0f}   95% CI [{lo_ppo:>10,.0f}, {hi_ppo:>10,.0f}]")
    print(f"  Diff (DT-PPO): ${diffs.mean():>12,.0f}   95% CI [{lo_d:>10,.0f}, {hi_d:>10,.0f}]")
    print(f"  P(DT > PPO | bootstrap) = {p_gt:.4f}   win rate = {(diffs > 0).mean():.2f}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}"
          + ("   [indicative only: n<10]" if len(diffs) < 10 else ""))
    return {
        "surface": label,
        "n": len(diffs),
        "dt_mean": float(dt.mean()), "dt_ci": [lo_dt, hi_dt],
        "ppo_mean": float(ppo.mean()), "ppo_ci": [lo_ppo, hi_ppo],
        "diff_mean": float(diffs.mean()), "diff_ci": [lo_d, hi_d],
        "p_dt_greater": p_gt,
        "win_rate": float((diffs > 0).mean()),
        "wilcoxon_p": None if np.isnan(w_p) else float(w_p),
    }


def part_a_identity_surfaces(n_boot, seed):
    results = []
    for label, rel in SURFACES.items():
        csv_path = ROOT / rel / "heldout_metrics_by_scenario.csv"
        df = pl.read_csv(csv_path)
        dt = (
            df.filter(pl.col("policy_name") == "candidate_dt")
            .sort("scenario_label")
        )
        ppo = (
            df.filter(pl.col("policy_name") == "ppo_reference")
            .sort("scenario_label")
        )
        if dt.height != ppo.height:
            raise RuntimeError(f"{label}: policy row mismatch {dt.height} vs {ppo.height}")
        results.append(
            paired_stats(
                dt["avg_profit_per_episode"].to_list(),
                ppo["avg_profit_per_episode"].to_list(),
                f"{label} (identity, rtg_mode=auto)",
                n_boot,
                seed,
            )
        )
    return results


def part_b_impact_gate(n_boot, seed):
    with open(ROOT / "eval_output/phase3_impact/results.json") as fh:
        records = json.load(fh)

    _CELL_RE = re.compile(r"_((?:small|hornsdale|torrens)_n?\d*\w*)_((?:sa1|vic1)_[a-z]+_\d{4})$")

    def collect(prefix):
        """Collect one profit per (battery, scenario).

        The auto-mode sweeps carry an RTG label that sets the constant-RTG
        fallback value; under merit-order impact only rtg0.0 matches the shipped
        configuration (report.md: 'auto falls back to constant RTG'). Higher
        constant values re-introduce the self-suppression failure mode and are
        not part of the shipped policy. For identity, all labels resolve to
        j_t_soc and agree; we still take rtg0.0-labelled runs for consistency.
        """
        out = {}
        seen_rtg = {}
        for r in records:
            lab = r["label"]
            if not lab.startswith(prefix):
                continue
            m = re.search(r"_rtg([\d.]+)_", lab)
            rtg = m.group(1) if m else ""
            m2 = re.search(r"_(small|hornsdale|torrens)_((?:sa1|vic1)_[a-z]+_\d{4})$", lab)
            if not m2:
                raise ValueError(f"cannot parse battery/scenario from label: {lab}")
            key = (m2.group(1), m2.group(2))
            seen_rtg.setdefault(key, set()).add(rtg)
            # Prefer the rtg0.0-labelled run (shipped fallback); first occurrence wins.
            if key not in out or rtg == "0.0":
                if key in out and out[key][1] == "0.0":
                    continue
                out[key] = (float(r["profit"]), rtg)
        mismatched = {k: v for k, v in seen_rtg.items() if len(v) > 1}
        return {k: v[0] for k, v in out.items()}

    results = []
    for impact, prefix_dt, prefix_ppo in [
        ("piecewise_merit_order", "piecewise_merit_order_stagec_h3h1_auto", "piecewise_merit_order_ppo"),
        ("identity", "identity_stagec_h3h1_auto", "identity_ppo"),
    ]:
        dt_cells = collect(prefix_dt)
        ppo_cells = collect(prefix_ppo)
        # auto runs carry an rtg suffix inside the label; keep rtg0.0 (auto resolves internally)
        keys = sorted(set(dt_cells) & set(ppo_cells))
        results.append(
            paired_stats(
                [dt_cells[k] for k in keys],
                [ppo_cells[k] for k in keys],
                f"impact_gate_{impact}",
                n_boot,
                seed,
            )
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Stage C DT statistical significance (bootstrap over matched cells)")
    print(f"bootstrap iterations: {args.boot}, seed: {args.seed}")

    identity = part_a_identity_surfaces(args.boot, args.seed)
    impact = part_b_impact_gate(args.boot, args.seed)

    out_path = ROOT / "eval_output/stagec_statistical_significance.json"
    with open(out_path, "w") as fh:
        json.dump({"identity_surfaces": identity, "impact_gate": impact}, fh, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
