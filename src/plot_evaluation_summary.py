from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create analytics plots from an AEMO evaluator summary JSON file.")
    parser.add_argument("--summary-path", type=Path, required=True, help="Path to evaluation_summary.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for the generated plots and CSV/JSON artifacts")
    parser.add_argument("--title", type=str, default="AEMO evaluation summary", help="Figure title prefix")
    parser.add_argument(
        "--dispatch-parquet",
        action="append",
        default=None,
        help="Optional parquet path for a dispatch-based episode to include as an extra comparison series",
    )
    parser.add_argument(
        "--dispatch-parquet-dir",
        type=Path,
        default=repo_root() / "data" / "aemo_dispatch_episodes" / "fcas_compare",
        help="Directory to scan for dispatch parquet files when --dispatch-parquet is not provided",
    )
    return parser.parse_args(argv)


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_aggregate_rows(summary: dict[str, Any]) -> pd.DataFrame:
    metrics = summary.get("heldout_evaluation", {}).get("aggregate_metrics", [])
    rows: list[dict[str, Any]] = []
    for item in metrics:
        if not isinstance(item, dict):
            continue
        experiment = str(item.get("experiment", "")).strip()
        if not experiment:
            continue
        rows.append(
            {
                "experiment": experiment,
                "episodes": int(item.get("episodes_evaluated", 0) or 0),
                "profit": float(item.get("avg_profit_per_episode", np.nan)),
                "fcas_revenue": float(item.get("avg_fcas_revenue_per_episode", np.nan)),
                "mean_reward": float(item.get("mean_reward", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def extract_paired_rows(summary: dict[str, Any]) -> pd.DataFrame:
    paired = summary.get("heldout_evaluation", {}).get("paired_comparisons_vs_reference", {})
    rows: list[dict[str, Any]] = []
    for experiment, stats in paired.items():
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "experiment": str(experiment),
                "mean_diff": float(stats.get("mean_diff", np.nan)),
                "median_diff": float(stats.get("median_diff", np.nan)),
                "wilcoxon_p": float(stats.get("wilcoxon_p", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def infer_episode_steps(summary: dict[str, Any]) -> int | None:
    metrics = summary.get("heldout_evaluation", {}).get("aggregate_metrics", [])
    steps: list[int] = []
    for item in metrics:
        if not isinstance(item, dict):
            continue
        raw_value = item.get("avg_episode_steps")
        if raw_value is None:
            continue
        try:
            step_value = int(round(float(raw_value)))
        except (TypeError, ValueError):
            continue
        if step_value > 0:
            steps.append(step_value)
    if not steps:
        return None
    return int(round(float(np.median(steps))))


def _chunk_dispatch_rows(df: pl.DataFrame, *, steps_per_episode: int | None) -> pl.DataFrame:
    if steps_per_episode is None or steps_per_episode <= 0 or df.height == 0:
        return df.with_columns(pl.lit(0).alias("episode_chunk"))
    full_episodes = df.height // steps_per_episode
    if full_episodes <= 0:
        return df.with_columns(pl.lit(0).alias("episode_chunk"))
    trimmed_rows = full_episodes * steps_per_episode
    return df.head(trimmed_rows).with_columns(
        (pl.int_range(0, trimmed_rows, eager=True) // steps_per_episode).alias("episode_chunk")
    )


def extract_dispatch_episode_rows(
    dispatch_paths: Sequence[Path] | None,
    *,
    steps_per_episode: int | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not dispatch_paths:
        return pd.DataFrame(rows)
    for path in dispatch_paths:
        path = path.resolve()
        if not path.is_file():
            continue
        try:
            df = pl.read_parquet(path)
        except Exception as exc:
            print(f"Skipping dispatch parquet {path}: {exc}")
            continue
        if df.height == 0:
            continue
        chunked = _chunk_dispatch_rows(df, steps_per_episode=steps_per_episode)
        if "reward" not in chunked.columns:
            print(f"Skipping dispatch parquet {path}: missing reward column")
            continue
        per_episode = (
            chunked.group_by("episode_chunk")
            .agg(
                [
                    pl.col("reward").sum().alias("reward_sum"),
                    pl.col("fcas_revenue").sum().alias("fcas_sum")
                    if "fcas_revenue" in chunked.columns
                    else pl.lit(0.0).alias("fcas_sum"),
                ]
            )
            .sort("episode_chunk")
        )
        if per_episode.height == 0:
            continue
        experiment_name = f"dispatch_episode_{path.stem}"
        rows.append(
            {
                "experiment": experiment_name,
                "episodes": int(per_episode.height),
                "profit": float(per_episode["reward_sum"].mean() * 1000.0),
                "fcas_revenue": float(per_episode["fcas_sum"].mean()),
                "mean_reward": float(per_episode["reward_sum"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save_bar_chart(
    output_path: Path,
    values: Sequence[float],
    labels: Sequence[str],
    *,
    title: str,
    ylabel: str,
    colors: Sequence[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4.6))
    colors = list(colors or ["#1f77b4"] * len(labels))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.02 * max(1.0, abs(value))) if value >= 0 else value - (0.02 * max(1.0, abs(value))),
            f"{value:,.0f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_paired_diff_chart(output_path: Path, paired_df: pd.DataFrame, *, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(paired_df) * 1.1), 4.6))
    x = np.arange(len(paired_df))
    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in paired_df["mean_diff"].tolist()]
    bars = ax.bar(x, paired_df["mean_diff"].tolist(), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Mean paired reward difference")
    ax.set_xticks(x)
    ax.set_xticklabels(paired_df["experiment"].tolist(), rotation=30, ha="right")
    for bar, row in zip(bars, paired_df.itertuples(index=False)):
        p_value = float(row.wilcoxon_p) if not np.isnan(row.wilcoxon_p) else np.nan
        annotation = f"p={p_value:.3f}" if np.isfinite(p_value) else "p=n/a"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.02 * max(1.0, abs(bar.get_height()))) if bar.get_height() >= 0 else bar.get_height() - (0.02 * max(1.0, abs(bar.get_height()))),
            annotation,
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = args.summary_path.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    output_dir = (args.output_dir or summary_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    aggregate_df = extract_aggregate_rows(summary)
    paired_df = extract_paired_rows(summary)
    steps_per_episode = infer_episode_steps(summary)

    dispatch_paths: list[Path] = []
    if args.dispatch_parquet:
        dispatch_paths = [Path(item).expanduser().resolve() for item in args.dispatch_parquet]
    elif args.dispatch_parquet_dir:
        dispatch_dir = args.dispatch_parquet_dir.expanduser().resolve()
        if dispatch_dir.is_dir():
            dispatch_paths = sorted(dispatch_dir.glob("*.parquet"))

    dispatch_df = extract_dispatch_episode_rows(
        dispatch_paths,
        steps_per_episode=steps_per_episode,
    )
    if not dispatch_df.empty:
        aggregate_df = pd.concat([aggregate_df, dispatch_df], ignore_index=True)

    if aggregate_df.empty:
        raise ValueError("No aggregate metrics found in the evaluation summary")

    aggregate_df = aggregate_df.sort_values("profit", ascending=False)
    profit_path = output_dir / "profit_by_experiment.png"
    save_bar_chart(
        profit_path,
        aggregate_df["profit"].tolist(),
        aggregate_df["experiment"].tolist(),
        title=f"{args.title}: average profit per episode",
        ylabel="Avg profit per episode",
        colors=["#1f77b4" if value >= 0 else "#d62728" for value in aggregate_df["profit"].tolist()],
    )

    fcas_path = output_dir / "fcas_revenue_by_experiment.png"
    save_bar_chart(
        fcas_path,
        aggregate_df["fcas_revenue"].tolist(),
        aggregate_df["experiment"].tolist(),
        title=f"{args.title}: average FCAS revenue per episode",
        ylabel="Avg FCAS revenue per episode",
        colors=["#2ca02c" if value >= 0 else "#ff7f0e" for value in aggregate_df["fcas_revenue"].tolist()],
    )

    paired_path = output_dir / "paired_comparison_mean_diff.png"
    save_paired_diff_chart(
        paired_path,
        paired_df,
        title=f"{args.title}: paired reward difference vs reference",
    )

    metrics_csv = output_dir / "evaluation_metrics_summary.csv"
    aggregate_df.to_csv(metrics_csv, index=False)

    manifest_path = output_dir / "evaluation_plot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "output_dir": str(output_dir),
                "plots": [
                    {"name": "profit_by_experiment", "path": str(profit_path)},
                    {"name": "fcas_revenue_by_experiment", "path": str(fcas_path)},
                    {"name": "paired_comparison_mean_diff", "path": str(paired_path)},
                ],
                "metrics_csv": str(metrics_csv),
                "comparison_scope": summary.get("heldout_evaluation", {}).get("comparison_scope"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
