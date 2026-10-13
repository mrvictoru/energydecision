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
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SIG = ROOT / "paper/audit/artifacts/stagec_v2cal_extended_significance.json"
IMPACT = ROOT / "paper/audit/artifacts/stagec_v2cal_impact_analysis.json"

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


def fig1_main_results(out_dir: Path) -> None:
    data = json.loads(SIG.read_text())["surface_sets"]["v2cal_extended"]
    by = {s["surface"]: s for s in data}
    rows = [by[s] for s in SURFACE_ORDER]
    x = np.arange(len(rows))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.45})

    dt = np.array([r["dt_mean"] for r in rows]) / 1e3
    pp = np.array([r["ppo_mean"] for r in rows]) / 1e3
    dt_e = np.array([_halfwidth(r["dt_ci"]) for r in rows]) / 1e3
    pp_e = np.array([_halfwidth(r["ppo_ci"]) for r in rows]) / 1e3
    ax1.bar(x - w / 2, dt, w, yerr=dt_e, capsize=3, color=C_DT, label="Decision Transformer")
    ax1.bar(x + w / 2, pp, w, yerr=pp_e, capsize=3, color=C_PPO, label="PPO (online RL)")
    for i, r in enumerate(rows):
        ann = "{:.2f}x\nn={}".format(r["dt_mean"] / r["ppo_mean"], r["n"])
        ax1.text(i, max(dt[i], pp[i]) * 1.03, ann, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SURFACE_LABEL[s] for s in SURFACE_ORDER])
    ax1.set_ylabel("profit per episode (k$)")
    ax1.set_title("Identity surfaces: DT vs PPO (95% CI)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_ylim(0, max(dt) * 1.35)

    diff = np.array([r["diff_mean"] for r in rows]) / 1e3
    dlo = np.array([r["diff_mean"] - r["diff_ci"][0] for r in rows]) / 1e3
    dhi = np.array([r["diff_ci"][1] - r["diff_mean"] for r in rows]) / 1e3
    ax2.errorbar(diff, x, xerr=[dlo, dhi], fmt="o", color=C_DT, capsize=3)
    ax2.axvline(0, color="k", lw=0.8, ls="--")
    ax2.set_yticks(x)
    ax2.set_yticklabels([SURFACE_LABEL[s].replace("\n", " ") for s in SURFACE_ORDER], fontsize=8)
    ax2.set_xlabel("paired difference (DT - PPO), k$")
    ax2.set_title("All four CIs exclude zero")
    ax2.invert_yaxis()

    _save(fig, out_dir, "fig1_main_results")


def fig4_impact_resilience(out_dir: Path) -> None:
    cells = json.loads(IMPACT.read_text())["v2cal_shipped_fixed_rtg0"]["paired"]["cells"]
    bat = ["small", "hornsdale", "torrens"]
    labels = {
        "small": "Small\n(~8 MWh)",
        "hornsdale": "Hornsdale\n(150 MW)",
        "torrens": "Torrens\n(250 MWh)",
    }
    dt = {b: [] for b in bat}
    pp = {b: [] for b in bat}
    for key, v in cells.items():
        b = key.split("|")[0]
        dt[b].append(v["dt"])
        pp[b].append(v["ppo"])

    x = np.arange(len(bat))
    w = 0.38
    dtm = np.array([np.mean(dt[b]) for b in bat]) / 1e3
    ppm = np.array([np.mean(pp[b]) for b in bat]) / 1e3

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(x - w / 2, dtm, w, color=C_DT, label="DT (auto -> const rtg0.0)")
    ax.bar(x + w / 2, ppm, w, color=C_PPO, label="PPO")
    for i, b in enumerate(bat):
        ann = "{:.2f}x".format(np.mean(dt[b]) / np.mean(pp[b]))
        ax.text(i, max(dtm[i], ppm[i]) * 1.02, ann, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[b] for b in bat])
    ax.set_ylabel("mean profit per episode (k$)")
    ax.set_title("Market-impact gate (piecewise merit-order): 9/9 cells beat PPO")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, dtm.max() * 1.2)
    _save(fig, out_dir, "fig4_impact_resilience")


def fig5_revenue_decomposition(out_dir: Path) -> None:
    energy = {0: [], 1: []}
    fcas = {0: [], 1: []}
    deg = {0: [], 1: []}
    for s in SURFACE_ORDER:
        df = pl.read_csv(ROOT / SURFACE_CSV[s] / "heldout_metrics_by_scenario.csv")
        for i, pol in enumerate(["candidate_dt", "ppo_reference"]):
            sub = df.filter(pl.col("policy_name") == pol)
            energy[i].append(sub["avg_energy_revenue_per_episode"].mean() / 1e3)
            fcas[i].append(sub["avg_fcas_revenue_per_episode"].mean() / 1e3)
            deg[i].append(sub["avg_total_degradation_cost_per_episode"].mean() / 1e3)

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
    ax.set_title("Revenue decomposition: energy, FCAS, degradation (DT vs PPO)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, out_dir, "fig5_revenue_decomposition")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/manuscript/figures")
    args = ap.parse_args()
    out = ROOT / args.out
    print("writing figures to", out)
    fig1_main_results(out)
    fig4_impact_resilience(out)
    fig5_revenue_decomposition(out)
    print("done")


if __name__ == "__main__":
    main()
