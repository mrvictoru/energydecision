"""Paper figures for the Stage C DT (report.md §8.2.10).

Generates six publication-ready figures into ``eval_output/paper_figures/``
(both PNG, 200 dpi, and PDF for LaTeX):

1.  ``fig1_main_results``          — DT vs PPO profit on the 4 identity
    surfaces with bootstrap 95% CIs.
2.  ``fig2_behavior``              — SOC trajectory and energy dispatch over
    one held-out episode, DT vs PPO.
3.  ``fig3_jtsoc_heatmap``         — the J_t(soc) cost-to-go table (RTG prompt)
    for one region/episode window.
4.  ``fig4_impact_resilience``     — identity vs merit-order impact across the
    three grid-scale batteries, shipped auto-DT vs PPO.
5.  ``fig5_revenue_decomposition`` — energy / FCAS / degradation decomposition
    per surface.
6.  ``fig6_ceiling_break``         — expanded broad-2024 profit across model
    generations (the behaviour-cloning ceiling breaking).

Usage: python3 scripts/stagec_paper_figures.py [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SURFACES = {
    "Standard\nOct": "eval_output/stagec_jtsoc_standard_rtgjtsoc",
    "Dispatch-\nmatched": "eval_output/stagec_jtsoc_dispatch_rtgjtsoc",
    "Expanded\nbroad-2024": "eval_output/stagec_jtsoc_expanded_rtgjtsoc",
    "2025\nOOD": "eval_output/stagec_jtsoc_2025_rtgjtsoc",
}

C_DT = "#1f77b4"
C_PPO = "#d62728"
C_AUX = "#7f7f7f"


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def fig1_main_results(stats_path, out_dir):
    with open(stats_path) as fh:
        data = json.load(fh)
    rows = data["identity_surfaces"]
    labels = ["Standard\nOct", "Dispatch-\nmatched", "Expanded\nbroad-2024", "2025\nOOD"]
    dt_means = [r["dt_mean"] for r in rows]
    dt_err = [[r["dt_mean"] - r["dt_ci"][0] for r in rows], [r["dt_ci"][1] - r["dt_mean"] for r in rows]]
    ppo_means = [r["ppo_mean"] for r in rows]
    ppo_err = [[r["ppo_mean"] - r["ppo_ci"][0] for r in rows], [r["ppo_ci"][1] - r["ppo_mean"] for r in rows]]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - w / 2, dt_means, w, yerr=dt_err, capsize=3, label="Stage C DT ($rtg\\_mode=auto$)", color=C_DT)
    ax.bar(x + w / 2, ppo_means, w, yerr=ppo_err, capsize=3, label="PPO reference", color=C_PPO)
    for xi, r in zip(x, rows):
        ax.annotate(
            f"+${r['diff_mean'] / 1000:.1f}k",
            (xi, max(r["dt_mean"], r["ppo_mean"])),
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Profit per episode ($)")
    ax.set_title("Stage C DT vs PPO — four identity surfaces (bootstrap 95% CI)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "fig1_main_results")


def _load_policy_logs(surface_dir: Path, policy: str) -> pl.DataFrame:
    return pl.read_parquet(surface_dir / "heldout_logs" / f"{policy}_heldout_logs.parquet")


def fig2_behavior(out_dir):
    surface = ROOT / SURFACES["Standard\nOct"]
    frames = {}
    for pol in ("candidate_dt", "ppo_reference"):
        df = _load_policy_logs(surface, pol)
        df = df.filter(pl.col("scenario_label") == "nsw1_oct_2024").sort("step")
        frames[pol] = df

    def soc_series(df):
        socs, caps = [], []
        for info in df["info"]:
            d = info if isinstance(info, dict) else json.loads(info) if info else {}
            socs.append(float(d.get("battery_soc", np.nan)))
            caps.append(float(d.get("capacity_mwh", np.nan)))
        s = np.array(socs)
        c = np.nanmedian(np.array(caps)) if not all(np.isnan(x) for x in caps) else np.nan
        return s, c

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    steps = frames["candidate_dt"]["step"].to_numpy()
    for ax, pol, color, label in [
        (axes[0], "candidate_dt", C_DT, "Stage C DT"),
        (axes[0], "ppo_reference", C_PPO, "PPO"),
    ]:
        s, cap = soc_series(frames[pol])
        ax.plot(steps, s, lw=1.0, color=color, label=label)
    axes[0].set_ylabel("SOC (MWh)")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0].grid(alpha=0.3)

    a_dt = np.array(frames["candidate_dt"]["action"].to_list(), dtype=float)[:, 0]
    a_ppo = np.array(frames["ppo_reference"]["action"].to_list(), dtype=float)[:, 0]
    axes[1].plot(steps, a_dt, lw=0.6, color=C_DT, label="Stage C DT")
    axes[1].plot(steps, a_ppo, lw=0.6, color=C_PPO, label="PPO")
    axes[1].axhline(0.0, color="k", lw=0.5)
    axes[1].set_ylabel("Energy dispatch\n(-1 charge, +1 discharge)")
    axes[1].set_xlabel("Step (5-min intervals, NSW1 Oct 2024)")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.suptitle("Learned dispatch behavior — one held-out episode", y=0.98)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_behavior")


def fig3_jtsoc_heatmap(out_dir):
    try:
        from AEMOBatteryEnv import AEMOBatteryTradingEnv
        from aemo_sdp_executor import build_rrp_forecast, build_seasonal_rrp_profile, compute_cost_to_go_table
    except ImportError as e:
        print(f"  skip fig3 ({e})")
        return

    parquet = ROOT / "data/aemo/processed_NSW1_2024-01-01_2024-01-14_0.0833h.parquet"
    if not parquet.exists():
        print("  skip fig3 (NSW1 parquet missing)")
        return
    data = pl.read_parquet(parquet).head(1728)  # one 144h episode window
    env = AEMOBatteryTradingEnv(
        aemo_data=data,
        battery_capacity=10.0,
        max_battery_flow=10.0,
        init_battery_level=5.0,
        max_step=len(data),
        action_mode="full_fcas",
    )
    env.reset(seed=0)
    profile = build_seasonal_rrp_profile(ROOT / "data/aemo", "NSW1")
    forecast = build_rrp_forecast(data, profile)
    cost, soc_levels = compute_cost_to_go_table(env, forecast, deg_cost_per_mwh=50.0)

    value = -cost.T  # [soc, t] remaining value ($) = RTG prompt
    # subsample time axis hourly for readability
    stride = 12
    value_s = value[:, ::stride]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    im = ax.imshow(value_s, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Episode time (hours)")
    ax.set_ylabel("SOC level (MWh)")
    ax.set_title("J_t(soc): state-dependent RTG prompt (remaining value), NSW1 seasonal forecast")
    n_y = value.shape[0]
    ax.set_yticks(range(0, n_y, max(1, n_y // 8)), [f"{v:.1f}" for v in soc_levels[:: max(1, n_y // 8)]])
    fig.colorbar(im, ax=ax, label="RTG token = $-J_t(soc)$ ($)")
    _save(fig, out_dir, "fig3_jtsoc_heatmap")


def fig4_impact_resilience(out_dir):
    with open(ROOT / "eval_output/phase3_impact/results.json") as fh:
        records = json.load(fh)

    import re

    def collect(prefix):
        cells = {}
        for r in records:
            lab = r["label"]
            if not lab.startswith(prefix):
                continue
            if "_rtg" in lab and "_rtg0.0_" not in lab + "_":
                pass
            m = re.search(r"_rtg([\d.]+)_", lab)
            rtg = m.group(1) if m else ""
            m2 = re.search(r"_(small|hornsdale|torrens)_((?:sa1|vic1)_[a-z]+_\d{4})$", lab)
            if m2 and (rtg == "0.0" or (m2[1], m2[2]) not in cells):
                key = (m2.group(1), m2.group(2))
                if key not in cells or rtg == "0.0":
                    if key in cells and cells[key][1] == "0.0":
                        continue
                    cells[key] = (float(r["profit"]), rtg)
        return {k: v[0] for k, v in cells.items()}

    bats = ["small", "hornsdale", "torrens"]
    data = {}
    for impact, pre_dt in [("identity", "identity_stagec_h3h1_auto"), ("piecewise", "piecewise_merit_order_stagec_h3h1_auto")]:
        dt_cells = collect(pre_dt)
        ppo_cells = collect(("identity_ppo" if impact == "identity" else "piecewise_merit_order_ppo"))
        data[impact] = {
            b: (
                np.mean([v for (bb, _), v in dt_cells.items() if bb == b]),
                np.mean([v for (bb, _), v in ppo_cells.items() if bb == b]),
            )
            for b in bats
        }

    x = np.arange(len(bats))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, (impact, hatch) in enumerate([("identity", ""), ("piecewise", "//")]):
        offs = (i - 0.5) * 2 * w
        ax.bar(x + offs - w / 2, [data[impact][b][0] for b in bats], w,
               label=f"Stage C DT — {impact}", color=C_DT, hatch=hatch, edgecolor="white")
        ax.bar(x + offs + w / 2, [data[impact][b][1] for b in bats], w,
               label=f"PPO — {impact}", color=C_PPO, hatch=hatch, edgecolor="white")
    ax.set_xticks(x, ["small (8 MWh)", "Hornsdale-class (194 MWh)", "Torrens-class (250 MWh)"])
    ax.set_ylabel("Mean profit per episode ($)")
    ax.set_yscale("symlog", linthresh=1e4)
    ax.set_title("Impact gate — price-taking vs merit-order impact (shipped $rtg\\_mode=auto$, rtg fallback 0.0)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "fig4_impact_resilience")


def fig5_revenue_decomposition(out_dir):
    labels = list(SURFACES.keys())
    comps = {pol: {"energy": [], "fcas": [], "deg": []} for pol in ("candidate_dt", "ppo_reference")}
    for rel in SURFACES.values():
        df = pl.read_csv(ROOT / rel / "heldout_metrics_by_scenario.csv")
        agg = df.group_by("policy_name").agg([
            pl.col("avg_energy_revenue_per_episode").mean(),
            pl.col("avg_fcas_revenue_per_episode").mean(),
            pl.col("avg_total_degradation_cost_per_episode").mean(),
        ])
        for pol in comps:
            row = agg.filter(pl.col("policy_name") == pol)
            comps[pol]["energy"].append(row["avg_energy_revenue_per_episode"][0])
            comps[pol]["fcas"].append(row["avg_fcas_revenue_per_episode"][0])
            comps[pol]["deg"].append(row["avg_total_degradation_cost_per_episode"][0])

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, (pol, color, label) in enumerate([("candidate_dt", C_DT, "Stage C DT"), ("ppo_reference", C_PPO, "PPO")]):
        offs = x + (i - 0.5) * w
        e = np.array(comps[pol]["energy"]); f = np.array(comps[pol]["fcas"]); d = np.array(comps[pol]["deg"])
        ax.bar(offs, e, w, color=color, alpha=0.45, label=f"{label} — energy")
        ax.bar(offs, f, w, bottom=e, color=color, label=f"{label} — FCAS")
        ax.bar(offs, -d, w, bottom=e + f, color=color, alpha=0.25, hatch="//", edgecolor="white",
               label=f"{label} − degradation")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("$ per episode")
    ax.set_title("Revenue decomposition (net of degradation)")
    ax.legend(frameon=False, fontsize=6.5, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "fig5_revenue_decomposition")


def fig6_ceiling_break(out_dir):
    generations = [
        ("Modern v2\nmixed (flagship)", 4596),
        ("FCAS-heavy\ntanh head", 13387),
        ("PPO-only DT\n(modern 8x768)", 17775),
        ("PPO\nreference", 19504),
        ("Stage B\nstandalone", 11987),
        ("Stage C\nconst-RTG", 27068),
        ("Stage C\nj\\_t\\_soc/auto", 34761),
    ]
    colors = [C_AUX, C_AUX, C_AUX, C_PPO, C_AUX, C_DT, C_DT]
    names = [g[0] for g in generations]
    vals = [g[1] for g in generations]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.bar(names, vals, color=colors)
    bars[3].set_edgecolor("black"); bars[3].set_linewidth(1.2)  # PPO reference emphasis
    ax.axhline(vals[3], color=C_PPO, ls="--", lw=1.0, alpha=0.7)
    for bar, v in zip(bars, vals):
        ax.annotate(f"${v/1000:.1f}k", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Profit per episode ($)\nexpanded broad-2024 surface")
    ax.set_title("Breaking the behaviour-cloning ceiling — model generations on the broad surface")
    plt.setp(ax.get_xticklabels(), fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_dir, "fig6_ceiling_break")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_output/paper_figures")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing paper figures to {out_dir}")
    stats_path = ROOT / "eval_output/stagec_statistical_significance.json"
    if stats_path.exists():
        fig1_main_results(stats_path, out_dir)
    else:
        print("  skip fig1 (run stagec_statistical_significance.py first)")
    fig2_behavior(out_dir)
    fig3_jtsoc_heatmap(out_dir)
    fig4_impact_resilience(out_dir)
    fig5_revenue_decomposition(out_dir)
    fig6_ceiling_break(out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
