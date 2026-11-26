"""Publication figures for the preprint (extended, fresh surfaces).

Supersedes ``scripts/stagec_paper_figures.py`` (which hardcoded the v1
``stagec_jtsoc_*`` paths). Reads the shipped/extended artifacts:

  - paper/audit/artifacts/stagec_v2cal_extended_significance.json   -> fig1
  - paper/audit/artifacts/stagec_v2cal_impact_analysis.json         -> fig4
  - eval_output/physics_v2/{standard_year,dispatch_year,expanded_full,
    2025_v2cal_fresh}_v2cal*/heldout_metrics_by_scenario.csv        -> fig5

Outputs PDF + PNG into ``paper/manuscript/figures/``.

Usage: python3 scripts/stagec_paper_figures_v2.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# arXiv (and most publishers) reject PostScript Type 3 fonts; matplotlib's
# default PDF backend emits them. Type 42 == embedded TrueType.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SIG = ROOT / "paper/audit/artifacts/stagec_v2cal_extended_significance.json"
IMPACT = ROOT / "paper/audit/artifacts/stagec_v2cal_impact_analysis.json"
PPO = ROOT / "paper/audit/artifacts/ppo_retrain_stats_20261125.json"
PPO_SEED_IMPACT = ROOT / "eval_output/ppo_retrain_20261125/phase3_impact"

SURFACE_ORDER = ["standard_year_2024", "dispatch_year_2024", "expanded_broad_2024", "2025_ood"]
SURFACE_LABEL = {
    "standard_year_2024": "Standard\n2024",
    "dispatch_year_2024": "Dispatch-\nmatched",
    "expanded_broad_2024": "Expanded\nbroad-2024",
    "2025_ood": "2025\nOOD",
}
SURFACE_CSV = {
    "standard_year_2024": "eval_output/physics_v2/standard_year_v2cal",
    "dispatch_year_2024": "eval_output/physics_v2/dispatch_year_v2cal",
    "expanded_broad_2024": "eval_output/physics_v2/expanded_full_v2cal",
    "2025_ood": "eval_output/physics_v2/2025_v2cal_fresh",
}
PPO_SURFACE_CSV = {
    "standard_year_2024": "eval_output/ppo_retrain_20261125/standard_year",
    "dispatch_year_2024": "eval_output/ppo_retrain_20261125/dispatch_year",
    "expanded_broad_2024": "eval_output/ppo_retrain_20261125/expanded_full",
    "2025_ood": "eval_output/ppo_retrain_20261125/fresh_2025",
}
PPO_SEEDS = ("tuned_seed0", "tuned_seed1", "tuned_seed2")

C_DT = "#1f77b4"
C_PPO = "#d62728"
C_DEG = "#8c564b"


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / (name + "." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + name + ".png/.pdf")


def _halfwidth(ci):
    return (ci[1] - ci[0]) / 2.0


def _aggregate_new_ppo():
    """Compare each DT cell with the mean of three matched PPO seed rollouts."""
    data = json.loads(PPO.read_text())
    rows = []
    for surface in SURFACE_ORDER:
        seed_sets = data["surface_seeds"][surface]
        seed_maps = []
        for seed in PPO_SEEDS:
            seed_maps.append({(c["scenario_label"], c["battery_label"]): c for c in seed_sets[seed]["cells"]})
        keys = sorted(set(seed_maps[0]))
        if any(set(sm) != set(keys) for sm in seed_maps):
            raise ValueError(f"PPO seed cell mismatch for {surface}")
        dt = np.asarray([seed_maps[0][key]["dt"] for key in keys], dtype=float)
        ppo_seed = np.asarray([[sm[key]["ppo"] for key in keys] for sm in seed_maps], dtype=float)
        ppo = ppo_seed.mean(axis=0)
        diff = dt - ppo
        rng = np.random.default_rng(42)
        boot = diff[rng.integers(0, len(diff), size=(10000, len(diff)))].mean(axis=1)
        rows.append({
            "surface": surface,
            "n": len(keys),
            "dt_mean": float(dt.mean()),
            "dt_ci": [float(x) for x in np.percentile(dt[rng.integers(0, len(dt), size=(10000, len(dt)))].mean(axis=1), [2.5, 97.5])],
            "ppo_mean": float(ppo.mean()),
            "ppo_ci": [float(x) for x in np.percentile(ppo[rng.integers(0, len(ppo), size=(10000, len(ppo)))].mean(axis=1), [2.5, 97.5])],
            "ppo_seed_means": [float(v.mean()) for v in ppo_seed],
            "diff_mean": float(diff.mean()),
            "diff_ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "ci_excludes_zero": bool(np.percentile(boot, 2.5) > 0),
        })
    return rows


def fig1_main_results(out_dir: Path) -> None:
    rows = _aggregate_new_ppo()
    x = np.arange(len(rows))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.45})

    dt = np.array([r["dt_mean"] for r in rows]) / 1e3
    pp = np.array([r["ppo_mean"] for r in rows]) / 1e3
    dt_lo = np.array([r["dt_mean"] - r["dt_ci"][0] for r in rows]) / 1e3
    dt_hi = np.array([r["dt_ci"][1] - r["dt_mean"] for r in rows]) / 1e3
    pp_lo = np.array([r["ppo_mean"] - r["ppo_ci"][0] for r in rows]) / 1e3
    pp_hi = np.array([r["ppo_ci"][1] - r["ppo_mean"] for r in rows]) / 1e3
    ax1.bar(x - w / 2, dt, w, yerr=[dt_lo, dt_hi], capsize=3, color=C_DT, label="Decision Transformer")
    ax1.bar(x + w / 2, pp, w, yerr=[pp_lo, pp_hi], capsize=3, color=C_PPO, label="4M PPO · 3-seed mean")
    for i, r in enumerate(rows):
        seed_means = np.asarray(r["ppo_seed_means"]) / 1e3
        ax1.scatter(np.full(3, x[i] + w / 2), seed_means, color="white", edgecolor=C_PPO, zorder=4, s=22, label="PPO seed means" if i == 0 else None)
        ann = "{:.2f}x\nn={}".format(r["dt_mean"] / r["ppo_mean"], r["n"])
        top = max(dt[i] + dt_hi[i], pp[i] + pp_hi[i], seed_means.max())
        ax1.text(i, top + 1.5, ann, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SURFACE_LABEL[s] for s in SURFACE_ORDER])
    ax1.set_ylabel("profit per episode (k$)")
    ax1.set_title("Identity surfaces: DT vs tuned 4M PPO (3 seeds)")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_ylim(0, 60)

    diff = np.array([r["diff_mean"] for r in rows]) / 1e3
    dlo = np.array([r["diff_mean"] - r["diff_ci"][0] for r in rows]) / 1e3
    dhi = np.array([r["diff_ci"][1] - r["diff_mean"] for r in rows]) / 1e3
    ax2.errorbar(diff, x, xerr=[dlo, dhi], fmt="o", color=C_DT, capsize=3)
    ax2.axvline(0, color="k", lw=0.8, ls="--")
    ax2.set_yticks(x)
    ax2.set_yticklabels([SURFACE_LABEL[s].replace("\n", " ") for s in SURFACE_ORDER], fontsize=8)
    ax2.set_xlabel("paired difference (DT - PPO seed mean), k$")
    ax2.set_title("DT/PPO advantage · dispatch CI crosses zero")
    ax2.invert_yaxis()

    _save(fig, out_dir, "fig1_main_results")


def fig4_impact_resilience(out_dir: Path) -> None:
    dt_rows = json.loads((PPO_SEED_IMPACT / "results.json").read_text())
    ppo_rows = [dt_rows]
    ppo_rows += [json.loads((PPO_SEED_IMPACT / f"results_seed{i}_ppo_only.json").read_text()) for i in (1, 2)]
    dt = {}
    ppo = []
    for seed_rows in ppo_rows:
        one = {}
        for row in seed_rows:
            label = row["label"]
            if "piecewise_merit_order" not in label:
                continue
            if label.startswith("piecewise_merit_order_stagec_v2cal_auto_rtg0.0_"):
                dt[label.split("rtg0.0_", 1)[1]] = row["profit"]
            elif label.startswith("piecewise_merit_order_ppo_"):
                one[label.split("_ppo_", 1)[1]] = row["profit"]
        ppo.append(one)
    bat = ["small", "hornsdale", "torrens"]
    labels = {"small": "Small\n(~8 MWh)", "hornsdale": "Hornsdale\n(150 MW)", "torrens": "Torrens\n(250 MWh)"}
    x = np.arange(len(bat))
    w = 0.38
    dtm = np.array([np.mean([v for k, v in dt.items() if k.startswith(b + "_")]) for b in bat]) / 1e3
    seed_means = np.array([[np.mean([v for k, v in seed.items() if k.startswith(b + "_")]) for b in bat] for seed in ppo]) / 1e3
    ppm = seed_means.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, dtm, w, color=C_DT, label="DT (auto -> const rtg0.0)")
    ax.bar(x + w / 2, ppm, w, color=C_PPO, label="4M PPO · 3-seed mean")
    for i, b in enumerate(bat):
        ax.scatter(np.full(3, x[i] + w / 2), seed_means[:, i], color="white", edgecolor=C_PPO, s=22, zorder=4, label="PPO seed means" if i == 0 else None)
        ax.text(i, max(dtm[i], ppm[i], seed_means[:, i].max()) * 1.03,
                "{:.2f}x".format(dtm[i] / ppm[i]), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[b] for b in bat])
    ax.set_ylabel("mean profit per episode (k$)")
    ax.set_title("Market-impact stress test: PPO seed variance is material")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, max(dtm.max(), seed_means.max()) * 1.25)
    _save(fig, out_dir, "fig4_impact_resilience")


def fig5_revenue_decomposition(out_dir: Path) -> None:
    energy = {0: [], 1: []}
    fcas = {0: [], 1: []}
    deg = {0: [], 1: []}
    for s in SURFACE_ORDER:
        df = pl.read_csv(ROOT / SURFACE_CSV[s] / "heldout_metrics_by_scenario.csv")
        dt = df.filter(pl.col("policy_name") == "candidate_dt")
        energy[0].append(dt["avg_energy_revenue_per_episode"].mean() / 1e3)
        fcas[0].append(dt["avg_fcas_revenue_per_episode"].mean() / 1e3)
        deg[0].append(dt["avg_total_degradation_cost_per_episode"].mean() / 1e3)
        ppo_means = []
        for seed in PPO_SEEDS:
            p = pl.read_csv(ROOT / PPO_SURFACE_CSV[s] / seed / "heldout_metrics_by_scenario.csv")
            sub = p.filter(pl.col("policy_name") == "ppo_reference")
            ppo_means.append((
                sub["avg_energy_revenue_per_episode"].mean() / 1e3,
                sub["avg_fcas_revenue_per_episode"].mean() / 1e3,
                sub["avg_total_degradation_cost_per_episode"].mean() / 1e3,
            ))
        energy[1].append(float(np.mean([p[0] for p in ppo_means])))
        fcas[1].append(float(np.mean([p[1] for p in ppo_means])))
        deg[1].append(float(np.mean([p[2] for p in ppo_means])))

    x = np.arange(len(SURFACE_ORDER))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for i, pol in enumerate(["DT", "PPO"]):
        off = -w / 2 if i == 0 else w / 2
        hatch = None if i == 0 else "//"
        en = np.array(energy[i])
        fc = np.array(fcas[i])
        dg = np.array(deg[i])
        ax.bar(x + off, en, w, color=C_DT, hatch=hatch, label=pol + " energy")
        ax.bar(x + off, fc, w, bottom=en, color=C_PPO, hatch=hatch, label=pol + " FCAS")
        ax.bar(x + off, -dg, w, color=C_DEG, hatch=hatch, label=pol + " degradation")
        for j in range(len(x)):
            ax.text(x[j] + off, en[j] + fc[j] + 0.5,
                    "{:.1f}".format(en[j] + fc[j] - dg[j]), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SURFACE_LABEL[s] for s in SURFACE_ORDER])
    ax.set_ylabel("per episode (k$)")
    ax.set_title("Revenue decomposition: DT vs 4M PPO seed mean")
    ax.set_ylim(-5, 47)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center")
    _save(fig, out_dir, "fig5_revenue_decomposition")


def fig3_jtsoc_heatmap(out_dir: Path) -> None:
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "src"))
    from aemo_sdp_executor import (
        build_rrp_forecast,
        build_seasonal_rrp_profile,
        compute_cost_to_go_table,
    )

    proc = ROOT / "data/aemo/processed_NSW1_2024-10-01_2024-10-14_0.0833h.parquet"
    df = pl.read_parquet(proc)
    horizon = 1728  # 144 h at 5-min steps (standard-surface horizon)
    sliced = df.head(horizon)
    profile = build_seasonal_rrp_profile(ROOT / "data/aemo", "NSW1", 5.0 / 60.0)
    forecast = build_rrp_forecast(sliced, profile)
    env = SimpleNamespace(
        battery_capacity=10.0, max_battery_flow=10.0, step_duration=5.0 / 60.0,
        max_grid_energy=float("inf"), df=None, battery_life_cost=1312500.0,
        degradation_temperature=25.0,
    )
    ctg, soc_levels = compute_cost_to_go_table(
        env, forecast, deg_cost_per_mwh=50.0, deg_calibration=0.12
    )
    rtg = -ctg  # (horizon+1, n_soc)

    hours = np.arange(rtg.shape[0]) * (5.0 / 60.0)
    soc = np.asarray(soc_levels)
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    im = ax.imshow(rtg.T, aspect="auto", origin="lower",
                   extent=[hours[0], hours[-1], soc[0], soc[-1]], cmap="viridis")
    ax.set_xlabel("time within episode (hours)")
    ax.set_ylabel("state of charge (MWh)")
    ax.set_title("J_t(soc) return-to-go prompt (NSW1, Oct 2024, seasonal forecast)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("return-to-go (A$)")
    _save(fig, out_dir, "fig3_jtsoc_heatmap")


def fig6_ceiling_break(out_dir: Path) -> None:
    from matplotlib.patches import Patch

    # Route history, not ceiling evidence. Values are report.md §8.2.7
    # improvement-trajectory numbers, each from a different evaluation surface,
    # era, and physics version: stages 1-5 predate the physics-v2 re-baseline,
    # stage 6 is the perfect-foresight LP, stage 7 is solver-in-the-loop, and
    # stage 8 is the extended dispatch-matched surface. The bars are therefore
    # not directly comparable; the behaviour-cloning ceiling itself is
    # documented by the within-cloning levers in Appendix A.
    stages = ["1 Pilot", "2 Autoresearch", "3 FCAS-rich", "4 GRPO\u2020",
              "5 Modern v2", "6 Hier DT+LP", "7 SDP exec",
              "8 Distilled\n(shipped)"]
    vals = [-10620, -1396, 1522, 8242, 10138, 291841, 59091, 33791]
    colors = ["#9aa5b1"] * 5 + ["#e08a3c"] * 2 + ["#2a9d8f"]

    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors)
    for i, v in enumerate(vals):
        ax.annotate("{:,}".format(v), (i, v), textcoords="offset points",
                    xytext=(0, 4 if v >= 0 else -10), ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_yscale("symlog", linthresh=1000)
    ax.set_ylim(-3e4, 8e5)
    ax.set_yticks([-1e4, -1e3, 0, 1e3, 1e4, 1e5])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=7)
    ax.set_ylabel("profit/ep (A$, symlog; per-stage surfaces)")
    ax.legend(handles=[
        Patch(color="#9aa5b1", label="behaviour cloning"),
        Patch(color="#e08a3c", label="solver in the loop"),
        Patch(color="#2a9d8f", label="distilled, standalone (shipped)"),
    ], frameon=False, fontsize=7, loc="upper left")

    _save(fig, out_dir, "fig6_ceiling_break")


def fig2_pipeline_schematic(out_dir: Path) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#eef2f7", fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.008,rounding_size=0.02",
                     linewidth=1.0, facecolor=fc, edgecolor="#4a5568"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=10, linewidth=1.0, linestyle=ls, color="#4a5568"))

    ax.text(0.02, 0.93, "Forecast", fontsize=9, style="italic")
    ax.text(0.02, 0.62, "Teacher (non-clairvoyant, hybrid)", fontsize=9, style="italic")
    ax.text(0.02, 0.28, "Distillation", fontsize=9, style="italic")

    box(0.02, 0.74, 0.24, 0.14, "cached pre-2024\nAEMO data")
    box(0.32, 0.74, 0.30, 0.14, "seasonal RRP profile\n(month$\\times$hour mean)")
    arrow(0.26, 0.81, 0.32, 0.81)

    box(0.02, 0.40, 0.18, 0.14, "SOC-waypoint DT\n($K{=}8$ targets)", fc="#e8f0fe")
    box(0.26, 0.40, 0.28, 0.14, "discretised DP\n(energy/SOC,\ndegradation-aware)", fc="#e8f0fe")
    box(0.60, 0.40, 0.16, 0.14, "greedy FCAS\nbidder", fc="#e8f0fe")
    box(0.80, 0.40, 0.18, 0.14, "teacher\ntrajectories")
    arrow(0.20, 0.47, 0.26, 0.47)
    arrow(0.54, 0.47, 0.60, 0.47)
    arrow(0.76, 0.47, 0.80, 0.47)
    arrow(0.45, 0.74, 0.40, 0.54)

    box(0.26, 0.06, 0.20, 0.14, "free-terminal\n$J_t(s)$")
    box(0.52, 0.06, 0.20, 0.14, "RTG prompt\n$=-J_t(s)$", fc="#e6f4ea")
    box(0.80, 0.06, 0.18, 0.14, "student DT\n(offline)", fc="#e6f4ea")
    arrow(0.46, 0.13, 0.52, 0.13)
    arrow(0.89, 0.40, 0.89, 0.20)
    arrow(0.72, 0.13, 0.80, 0.13, ls="--")
    arrow(0.34, 0.40, 0.36, 0.20, ls="--")
    _save(fig, out_dir, "fig2_pipeline_schematic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/manuscript/figures")
    args = ap.parse_args()
    out = ROOT / args.out
    print("writing figures to", out)
    fig1_main_results(out)
    fig2_pipeline_schematic(out)
    fig3_jtsoc_heatmap(out)
    fig4_impact_resilience(out)
    fig5_revenue_decomposition(out)
    fig6_ceiling_break(out)
    print("done")


if __name__ == "__main__":
    main()
