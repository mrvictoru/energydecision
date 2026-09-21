"""Analyse the v2cal market-impact gate from eval_output/phase3_impact/results.json.

Reports the shipped fixed fallback (rtg_value=0.0) and the best-RTG-per-cell
selection separately, so the preprint cannot conflate them.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval_output/phase3_impact/results.json"
OUT = ROOT / "paper/audit/artifacts/stagec_v2cal_impact_analysis.json"
IMPACT = "piecewise_merit_order"


def load():
    return json.loads(RESULTS.read_text())


def cell(dt_label, kind):
    """Return {(battery, scenario): {rtg: profit}} and ppo {(battery,scenario): profit}."""
    records = load()
    dt = defaultdict(dict)
    ppo = {}
    for r in records:
        lab = r["label"]
        if not lab.startswith(f"{kind}_"):
            continue
        rest = lab[len(kind) + 1:]
        # battery + scenario are the last two/three underscore fields
        import re
        m = re.search(r"_(small|hornsdale|torrens)_((?:sa1|vic1)_[a-z]+_\d{4})$", lab)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        if rest.startswith("ppo_"):
            ppo[key] = float(r["profit"])
            continue
        m2 = re.search(rf"^{re.escape(dt_label)}_rtg([\d.]+)_", rest)
        if m2:
            dt[key][float(m2.group(1))] = float(r["profit"])
    return dt, ppo


def select(dt, mode):
    if mode == "fixed_rtg0":
        return {k: v[0.0] for k, v in dt.items() if 0.0 in v}
    if mode == "best_rtg":
        return {k: max(v.values()) for k, v in dt.items()}
    raise ValueError(mode)


def per_battery_ratios(dt, ppo):
    bybat = defaultdict(list)
    for k in sorted(set(dt) & set(ppo)):
        bybat[k[0]].append(dt[k] / ppo[k])
    return {b: statistics.mean(v) for b, v in bybat.items()}


def ratio_of_means(dt, ppo):
    out = {}
    for b in sorted({k[0] for k in dt}):
        ks = [k for k in dt if k[0] == b and k in ppo]
        out[b] = sum(dt[k] for k in ks) / sum(ppo[k] for k in ks)
    return out


def paired(dt, ppo):
    keys = sorted(set(dt) & set(ppo))
    diffs = np.array([dt[k] - ppo[k] for k in keys])
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diffs), size=(10000, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    try:
        _, wp = stats.wilcoxon(diffs)
    except ValueError:
        wp = float("nan")
    return {
        "n": len(keys),
        "dt_mean": float(np.mean([dt[k] for k in keys])),
        "ppo_mean": float(np.mean([ppo[k] for k in keys])),
        "diff_mean": float(diffs.mean()),
        "diff_ci": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0),
        "win_rate": float((diffs > 0).mean()),
        "wilcoxon_p": None if np.isnan(wp) else float(wp),
        "cells": {f"{k[0]}|{k[1]}": {"dt": dt[k], "ppo": ppo[k], "win": dt[k] > ppo[k]} for k in keys},
    }


def main():
    payload = {"impact": IMPACT, "source": str(RESULTS.relative_to(ROOT))}
    configs = {
        "v2cal_shipped_fixed_rtg0": "stagec_v2cal_auto",
        "v2cal_best_rtg": "stagec_v2cal_auto",
        "v1_fix_best_rtg": "stagec_fix_20260913",
        "v1_h3h1_best_rtg": "stagec_h3h1_auto_20260821",
    }
    for name, dt_label in configs.items():
        mode = "fixed_rtg0" if "fixed" in name else "best_rtg"
        dt_all, ppo = cell(dt_label, IMPACT)
        dt = select(dt_all, mode)
        payload[name] = {
            "dt_label": dt_label,
            "selection": mode,
            "ratio_of_means": ratio_of_means(dt, ppo),
            "mean_of_per_cell_ratios": per_battery_ratios(dt, ppo),
            "paired": paired(dt, ppo),
        }

    OUT.write_text(json.dumps(payload, indent=2))
    for name in configs:
        p = payload[name]
        print(f"\n=== {name}  (selection={p['selection']}) ===")
        for b in p["ratio_of_means"]:
            print(f"   {b:9s} ratio-of-means {p['ratio_of_means'][b]:5.3f}   "
                  f"mean-of-cell-ratios {p['mean_of_per_cell_ratios'][b]:5.3f}")
        q = p["paired"]
        print(f"   paired n={q['n']} DT ${q['dt_mean']:,.0f} PPO ${q['ppo_mean']:,.0f} "
              f"diff ${q['diff_mean']:,.0f} CI[{q['diff_ci'][0]:,.0f},{q['diff_ci'][1]:,.0f}] "
              f"win {q['win_rate']:.2f} excl0={q['ci_excludes_zero']} p={q['wilcoxon_p']}")
        print("   cell detail:")
        for k, v in q["cells"].items():
            print(f"      {k:22s} DT ${v['dt']:>9,.0f} PPO ${v['ppo']:>8,.0f} {'win' if v['win'] else 'LOSS'}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
