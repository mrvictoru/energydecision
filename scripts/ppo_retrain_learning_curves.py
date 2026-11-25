"""Collect PPO retrain learning curves into a table + figure.

Reads every ``*_history.json`` produced by ``scripts/ppo_retrain_aemo.py train``
under ``--models-dir`` and writes:

  * ``learning_curves.csv``  (tag, seed, steps, validation mean profit, ...)
  * ``learning_curves.png``  (validation mean profit vs environment steps)

Usage:
    python3 scripts/ppo_retrain_learning_curves.py \
        --models-dir models/aemo_sb3/ppo_retrain_20261125 \
        --out-dir paper/audit/artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--old-ppo-validation", default=None,
                    help="optional validate_old_ppo.json for a reference line")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(models_dir.glob("*_history.json")):
        tag = path.name[: -len("_history.json")]
        meta_path = models_dir / f"{tag}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        history = json.loads(path.read_text())["history"]
        for row in history:
            val = row["validation"]
            rows.append(
                {
                    "tag": tag,
                    "seed": meta.get("seed"),
                    "config": meta.get("config"),
                    "overrides": json.dumps(meta.get("overrides", {})),
                    "steps": row["model_num_timesteps"],
                    "requested_milestone": row.get("requested_milestone"),
                    "val_mean_profit": val["mean_profit"],
                    "val_median_profit": val["median_profit"],
                    "val_std_profit": val["std_profit"],
                    "val_medium_1c": val["mean_profit_by_battery"].get("medium_1c"),
                    "val_large_07c": val["mean_profit_by_battery"].get("large_07c"),
                    "val_small_05c": val["mean_profit_by_battery"].get("small_05c"),
                    "val_fast_375c": val["mean_profit_by_battery"].get("fast_375c"),
                    "steps_per_second": row.get("steps_per_second"),
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )

    if not rows:
        raise SystemExit(f"no *_history.json under {models_dir}")

    csv_path = out_dir / "ppo_retrain_learning_curves.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path} ({len(rows)} rows)")

    fig, ax = plt.subplots(figsize=(8, 5))
    tags = sorted({r["tag"] for r in rows})
    for tag in tags:
        series = sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["steps"])
        ax.plot([r["steps"] for r in series], [r["val_mean_profit"] for r in series],
                marker="o", label=tag, linewidth=1.4)
    if args.old_ppo_validation:
        old = json.loads(Path(args.old_ppo_validation).read_text())["validation"]
        ax.axhline(old["mean_profit"], color="black", linestyle="--", linewidth=1.2,
                   label="old PPO (204,800 steps)")
    ax.set_xscale("log")
    ax.set_xlabel("PPO environment steps (log scale)")
    ax.set_ylabel("held-out pre-2024 validation mean profit (A$/episode)")
    ax.set_title("PPO learning curves — fixed 40-episode validation set")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = out_dir / "ppo_retrain_learning_curves.png"
    fig.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
