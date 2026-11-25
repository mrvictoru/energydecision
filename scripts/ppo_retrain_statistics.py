"""Paired statistics for the retrained-PPO identity surfaces and impact gate.

Method
------
Cell-level statistics reproduce ``paper/audit/reproduce_significance.py`` and
``paper/audit/analyse_v2cal_impact.py``: 10,000-resample percentile bootstrap of
the paired DT-PPO difference, paired Wilcoxon signed-rank p, per-cell win rate.

Two cluster-aware sensitivity analyses are added because the shipped cell-level
CI treats correlated cells as independent:

* ``region_cluster``  -- resample the 5 NEM regions with replacement and keep
  every cell belonging to a drawn region (months/episodes inside a region are
  correlated because they share the region's price regime);
* ``scenario_block``  -- resample (region, month) blocks with replacement; for
  surfaces with several episodes per scenario the block keeps all of them
  together, so the effective n is the number of scenarios, not episodes.

For the impact gate the battery scale is a crossed factor: it is resampled
*within* scenario blocks, and per-battery ratio-of-means is reported alongside
the pooled paired statistic.

Usage:
    python3 scripts/ppo_retrain_statistics.py --manifest <manifest.json> --out <out.json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

N_BOOT_DEFAULT = 10_000
SEED_DEFAULT = 42


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


ROOT = _repo_root()


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator, conf: float = 0.95):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    alpha = (1.0 - conf) / 2.0
    return (
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
        float((means > 0).mean()),
    )


def wilcoxon_p(diffs: np.ndarray) -> float | None:
    nonzero = diffs[diffs != 0]
    if nonzero.size < 1 or np.all(nonzero == nonzero[0]):
        return None
    try:
        return float(stats.wilcoxon(diffs).pvalue)
    except ValueError:
        return None


def load_surface(directory: Path) -> pl.DataFrame:
    path = directory / "heldout_metrics_by_scenario.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return pl.read_csv(path)


def load_policy_cells(directory: Path, policy: str) -> dict[tuple[str, str], float]:
    """Return {(scenario_label, battery_label): avg_profit_per_episode}."""
    df = load_surface(directory)
    rows = df.filter(pl.col("policy_name") == policy)
    if rows.height == 0:
        available = sorted(set(df["policy_name"].to_list()))
        raise SystemExit(f"{directory}: policy {policy!r} absent (available: {available})")
    return {
        (str(r["scenario_label"]), str(r["battery_label"])): float(r["avg_profit_per_episode"])
        for r in rows.iter_rows(named=True)
    }


def pair_sources(
    dt_cells: dict[tuple[str, str], float],
    ppo_cells: dict[tuple[str, str], float],
    *,
    dt_source: str,
    ppo_source: str,
) -> list[dict]:
    keys = sorted(set(dt_cells) & set(ppo_cells))
    if not keys:
        raise SystemExit(f"no shared cells between {dt_source} and {ppo_source}")
    missing_dt = sorted(set(ppo_cells) - set(dt_cells))
    missing_ppo = sorted(set(dt_cells) - set(ppo_cells))
    if missing_dt or missing_ppo:
        raise SystemExit(
            f"cell set differs between {dt_source} and {ppo_source}: "
            f"missing_dt={missing_dt[:5]} missing_ppo={missing_ppo[:5]}"
        )
    return [
        {
            "scenario_label": k[0],
            "battery_label": k[1],
            "region": str(k[0]).split("_")[0].upper(),
            "dt": dt_cells[k],
            "ppo": ppo_cells[k],
        }
        for k in keys
    ]


def region_of(row: dict) -> str:
    region = str(row.get("region") or "").upper()
    if region in {"NSW1", "QLD1", "SA1", "TAS1", "VIC1"}:
        return region
    label = str(row["scenario_label"]).lower()
    for reg in ("nsw1", "qld1", "sa1", "tas1", "vic1"):
        if label.startswith(reg):
            return reg.upper()
    return "UNKNOWN"


def surface_stats(rows: list[dict], n_boot: int, seed: int) -> dict:
    dt = np.array([r["dt"] for r in rows], dtype=float)
    ppo = np.array([r["ppo"] for r in rows], dtype=float)
    diffs = dt - ppo
    rng = np.random.default_rng(seed)
    lo, hi, p_gt = bootstrap_ci(diffs, n_boot, rng)
    lo_dt, hi_dt, _ = bootstrap_ci(dt, n_boot, rng)
    lo_ppo, hi_ppo, _ = bootstrap_ci(ppo, n_boot, rng)
    ratio_rng = np.random.default_rng(seed + 1)
    idx = ratio_rng.integers(0, len(dt), size=(n_boot, len(dt)))
    ratios = dt[idx].mean(axis=1) / ppo[idx].mean(axis=1)
    ratio_ci = [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]

    out = {
        "n": len(dt),
        "dt_mean": float(dt.mean()),
        "dt_ci": [lo_dt, hi_dt],
        "ppo_mean": float(ppo.mean()),
        "ppo_ci": [lo_ppo, hi_ppo],
        "ratio_of_means": float(dt.mean() / ppo.mean()),
        "ratio_of_means_ci": ratio_ci,
        "diff_mean": float(diffs.mean()),
        "diff_ci": [lo, hi],
        "ci_excludes_zero": bool(lo > 0),
        "ci_excludes_zero_negative": bool(hi < 0),
        "p_dt_greater": p_gt,
        "win_rate": float((diffs > 0).mean()),
        "wilcoxon_p": wilcoxon_p(diffs),
        "cells": [
            {**r, "diff": r["dt"] - r["ppo"], "win": r["dt"] > r["ppo"]} for r in rows
        ],
    }

    # --- cluster-aware sensitivity: resample regions with replacement -------
    by_region: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_region[region_of(r)].append(r["dt"] - r["ppo"])
    region_means = {k: float(np.mean(v)) for k, v in by_region.items()}
    regions = list(region_means)
    rng2 = np.random.default_rng(seed + 2)
    draws = []
    for _ in range(n_boot):
        pick = rng2.integers(0, len(regions), size=len(regions))
        draws.append(np.mean([region_means[regions[i]] for i in pick]))
    draws = np.array(draws)
    out["region_cluster"] = {
        "n_clusters": len(regions),
        "cluster_means": region_means,
        "diff_mean_of_cluster_means": float(np.mean(list(region_means.values()))),
        "diff_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "p_positive": float((draws > 0).mean()),
    }

    # --- cluster-aware sensitivity: resample scenario blocks ----------------
    blocks: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        blocks[(region_of(r), str(r["scenario_label"]))].append(r["dt"] - r["ppo"])
    block_means = {k: float(np.mean(v)) for k, v in blocks.items()}
    keys = list(block_means)
    rng3 = np.random.default_rng(seed + 3)
    draws = []
    for _ in range(n_boot):
        pick = rng3.integers(0, len(keys), size=len(keys))
        draws.append(np.mean([block_means[keys[i]] for i in pick]))
    draws = np.array(draws)
    out["scenario_block_cluster"] = {
        "n_blocks": len(keys),
        "block_size_counts": {f"{k[0]}|{k[1]}": len(v) for k, v in blocks.items()},
        "diff_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "p_positive": float((draws > 0).mean()),
    }
    return out


def impact_stats(results_path: Path, dt_label: str, n_boot: int, seed: int) -> dict:
    records = json.loads(results_path.read_text())
    kind = "piecewise_merit_order"
    dt: dict[tuple[str, str], float] = {}
    ppo: dict[tuple[str, str], float] = {}
    for rec in records:
        label = rec["label"]
        if not label.startswith(f"{kind}_"):
            continue
        parts = label[len(kind) + 1:].split("_")
        battery = parts[-4]
        scenario = "_".join(parts[-3:])
        key = (battery, scenario)
        if label.startswith(f"{kind}_ppo_"):
            ppo[key] = float(rec["profit"])
        elif label.startswith(f"{kind}_{dt_label}_"):
            dt[key] = float(rec["profit"])
    keys = sorted(set(dt) & set(ppo))
    if not keys:
        raise SystemExit(f"no paired impact cells found in {results_path}")

    dt_arr = np.array([dt[k] for k in keys])
    ppo_arr = np.array([ppo[k] for k in keys])
    diffs = dt_arr - ppo_arr
    rng = np.random.default_rng(seed)
    lo, hi, p_gt = bootstrap_ci(diffs, n_boot, rng)

    per_battery_ratio = {}
    for battery in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == battery]
        per_battery_ratio[battery] = {
            "ratio_of_means": float(sum(dt[k] for k in ks) / sum(ppo[k] for k in ks)),
            "dt_mean": float(np.mean([dt[k] for k in ks])),
            "ppo_mean": float(np.mean([ppo[k] for k in ks])),
        }

    # region/scenario-cluster sensitivity (here: the 3 SA1/VIC1 scenarios)
    by_scenario: dict[str, list[float]] = defaultdict(list)
    for k in keys:
        by_scenario[k[1]].append(dt[k] - ppo[k])
    scenario_means = {k: float(np.mean(v)) for k, v in by_scenario.items()}
    scen = list(scenario_means)
    rng2 = np.random.default_rng(seed + 1)
    draws = np.array(
        [
            np.mean([scenario_means[scen[i]] for i in rng2.integers(0, len(scen), size=len(scen))])
            for _ in range(n_boot)
        ]
    )
    # battery-crossed: per-scenario ratio of means, then average
    scenario_ratios = {}
    for s in sorted({k[1] for k in keys}):
        ks = [k for k in keys if k[1] == s]
        scenario_ratios[s] = float(sum(dt[k] for k in ks) / sum(ppo[k] for k in ks))

    return {
        "impact": kind,
        "n_cells": len(keys),
        "dt_mean": float(dt_arr.mean()),
        "ppo_mean": float(ppo_arr.mean()),
        "ratio_of_means": float(dt_arr.mean() / ppo_arr.mean()),
        "per_battery": per_battery_ratio,
        "per_scenario_ratio_of_means": scenario_ratios,
        "diff_mean": float(diffs.mean()),
        "diff_ci": [lo, hi],
        "ci_excludes_zero": bool(lo > 0),
        "win_rate": float((diffs > 0).mean()),
        "n_cells_dt_wins": int((diffs > 0).sum()),
        "wilcoxon_p": wilcoxon_p(diffs),
        "scenario_cluster": {
            "n_clusters": len(scen),
            "cluster_means": scenario_means,
            "diff_ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "p_positive": float((draws > 0).mean()),
        },
        "cells": [
            {
                "battery": k[0],
                "scenario": k[1],
                "dt": dt[k],
                "ppo": ppo[k],
                "diff": dt[k] - ppo[k],
                "ratio": dt[k] / ppo[k],
                "win": dt[k] > ppo[k],
            }
            for k in keys
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="JSON describing surface dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--impact-results", default=None, help="optional impact results.json")
    ap.add_argument("--impact-dt-label", default="stagec_v2cal_auto")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    payload: dict = {
        "method": (
            "10,000-resample percentile bootstrap of the paired DT-PPO difference; paired "
            "Wilcoxon signed-rank; cell-level CI as in paper/audit/reproduce_significance.py, "
            "plus region-cluster and scenario-block cluster bootstrap sensitivities."
        ),
        "n_boot": args.boot,
        "seed": args.seed,
        "surfaces": {},
        "surface_seeds": {},
    }
    for surface, cfg in manifest["surfaces"].items():
        dt_policy = cfg.get("dt_policy", "candidate_dt")
        ppo_policy = cfg.get("ppo_policy", "ppo_reference")
        dt_cells = load_policy_cells(ROOT / cfg["dt_dir"], dt_policy)
        primary_ppo_dir = cfg.get("ppo_dir") or next(iter(cfg["ppo_seed_dirs"].values()))
        ppo_cells = load_policy_cells(ROOT / primary_ppo_dir, ppo_policy)
        rows = pair_sources(dt_cells, ppo_cells, dt_source=cfg["dt_dir"], ppo_source=primary_ppo_dir)
        primary_tag = cfg.get("primary_seed_tag", "primary")
        payload["surfaces"][surface] = surface_stats(rows, args.boot, args.seed)
        payload["surfaces"][surface]["dt_dir"] = str(cfg["dt_dir"])
        payload["surfaces"][surface]["dt_cells_source"] = cfg.get("dt_cells_source", cfg["dt_dir"])
        payload["surfaces"][surface]["primary_ppo_dir"] = str(primary_ppo_dir)
        payload["surfaces"][surface]["primary_seed_tag"] = primary_tag

        for seed_tag, ppo_dir in cfg.get("ppo_seed_dirs", {}).items():
            ppo_cells_s = load_policy_cells(ROOT / ppo_dir, ppo_policy)
            rows_s = pair_sources(dt_cells, ppo_cells_s, dt_source=cfg["dt_dir"], ppo_source=ppo_dir)
            stats_s = surface_stats(rows_s, args.boot, args.seed)
            stats_s["ppo_dir"] = ppo_dir
            payload["surface_seeds"].setdefault(surface, {})[seed_tag] = stats_s

    if args.impact_results:
        payload["impact_gate"] = impact_stats(
            ROOT / args.impact_results, args.impact_dt_label, args.boot, args.seed
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    for surface, stats_ in payload["surfaces"].items():
        print(
            f"{surface:22s} n={stats_['n']:2d} DT ${stats_['dt_mean']:>10,.0f} "
            f"PPO ${stats_['ppo_mean']:>10,.0f} ratio {stats_['ratio_of_means']:.3f} "
            f"diff ${stats_['diff_mean']:>10,.0f} "
            f"CI[{stats_['diff_ci'][0]:,.0f},{stats_['diff_ci'][1]:,.0f}] "
            f"win {stats_['win_rate']:.2f} p={stats_['wilcoxon_p']}"
        )
        print(
            f"{'':22s} region-cluster CI[{stats_['region_cluster']['diff_ci'][0]:,.0f},"
            f"{stats_['region_cluster']['diff_ci'][1]:,.0f}] "
            f"scenario-block CI[{stats_['scenario_block_cluster']['diff_ci'][0]:,.0f},"
            f"{stats_['scenario_block_cluster']['diff_ci'][1]:,.0f}]"
        )
    if "impact_gate" in payload:
        ig = payload["impact_gate"]
        print(
            f"impact gate: {ig['n_cells_dt_wins']}/{ig['n_cells']} DT wins, "
            f"ratio {ig['ratio_of_means']:.3f}, diff ${ig['diff_mean']:,.0f} "
            f"CI[{ig['diff_ci'][0]:,.0f},{ig['diff_ci'][1]:,.0f}] p={ig['wilcoxon_p']}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
