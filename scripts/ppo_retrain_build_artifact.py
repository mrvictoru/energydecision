"""Assemble paper/audit/artifacts/ppo_retrain_<date>.json from the run artifacts.

Inputs (all produced by the other ppo_retrain_* scripts):
  * models/<run>/data_split_manifest.json
  * models/<run>/{tag}_meta.json + {tag}_history.json
  * models/<run>/optuna/trials.json
  * paper/audit/artifacts/ppo_retrain_learning_curves.csv
  * the statistics JSON (scripts/ppo_retrain_statistics.py)
  * validate_old_ppo.json / validate_dt.json (optional learning-curve references)

Usage:
    python3 scripts/ppo_retrain_build_artifact.py \
        --run-dir models/aemo_sb3/ppo_retrain_20261125 \
        --stats paper/audit/artifacts/ppo_retrain_stats_20261125.json \
        --out paper/audit/artifacts/ppo_retrain_20261125.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_zip_metadata(path: Path) -> dict:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    keys = ["num_timesteps", "_total_timesteps", "n_envs", "n_steps", "batch_size",
            "n_epochs", "gamma", "learning_rate", "policy_kwargs", "seed", "_n_updates",
            "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "gae_lambda", "target_kl"]
    out = {}
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict) and "value_schedule" in value:
            value = value["value_schedule"]
        out[key] = value
    return out


def _statistics_summary(stats: dict, stats_path: str) -> dict:
    """Compact index of the statistics payload (no per-cell arrays)."""
    surface_fields = (
        "n", "dt_mean", "ppo_mean", "ratio_of_means", "ratio_of_means_ci",
        "diff_mean", "diff_ci", "win_rate", "wilcoxon_p", "primary_seed_tag",
        "dt_cells_source", "dt_dir",
    )
    seed_fields = ("dt_mean", "ppo_mean", "ratio_of_means", "diff_mean", "diff_ci",
                   "win_rate", "wilcoxon_p")
    out: dict = {
        "method": stats.get("method"),
        "n_boot": stats.get("n_boot"),
        "seed": stats.get("seed"),
        "full_statistics_file": stats_path,
        "surfaces": {
            name: {
                **{k: v[k] for k in surface_fields if k in v},
                "region_cluster_diff_ci": (v.get("region_cluster") or {}).get("diff_ci"),
                "scenario_block_diff_ci": (v.get("scenario_block_cluster") or {}).get("diff_ci"),
            }
            for name, v in stats.get("surfaces", {}).items()
        },
        "seed_level": {
            name: {tag: {k: s[k] for k in seed_fields if k in s} for tag, s in seeds.items()}
            for name, seeds in stats.get("surface_seeds", {}).items()
        },
    }
    if "impact_gate" in stats:
        out["impact_gate"] = {
            k: v for k, v in stats["impact_gate"].items() if k != "cells"
        }
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-ppo", default="models/aemo_sb3/ppo_aemo_fcas_model.zip")
    ap.add_argument("--learning-curves-csv", default=None)
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    split_manifest = json.loads((run_dir / "data_split_manifest.json").read_text())
    stats = json.loads((ROOT / args.stats).read_text())

    runs = {}
    for meta_path in sorted(run_dir.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        tag = meta["tag"]
        ckpt = ROOT / meta["final_checkpoint"]
        runs[tag] = {
            "seed": meta["seed"],
            "config": meta["config"],
            "config_json": meta.get("config_json"),
            "overrides": {k: v for k, v in (meta.get("overrides") or {}).items()},
            "requested_timesteps": meta["requested_timesteps"],
            "actual_num_timesteps": meta["actual_num_timesteps"],
            "n_envs": meta["n_envs"],
            "device": meta["device"],
            "total_wall_seconds": meta["total_wall_seconds"],
            "final_checkpoint": meta["final_checkpoint"],
            "final_checkpoint_sha256": meta["final_checkpoint_sha256"],
            "ppo_checkpoint_metadata": read_zip_metadata(ckpt) if ckpt.is_file() else None,
            "history": [
                {
                    "requested_milestone": h.get("requested_milestone"),
                    "model_num_timesteps": h["model_num_timesteps"],
                    "val_mean_profit": h["validation"]["mean_profit"],
                    "val_median_profit": h["validation"]["median_profit"],
                    "val_std_profit": h["validation"]["std_profit"],
                    "val_mean_profit_by_battery": h["validation"]["mean_profit_by_battery"],
                    "val_mean_profit_by_region": h["validation"]["mean_profit_by_region"],
                    "checkpoint": h["checkpoint"],
                    "checkpoint_sha256": h["checkpoint_sha256"],
                }
                for h in meta["history"]
            ],
        }

    optuna_path = run_dir / "optuna" / "trials.json"
    optuna = json.loads(optuna_path.read_text()) if optuna_path.is_file() else None

    old_ppo = ROOT / args.old_ppo
    old_meta = read_zip_metadata(old_ppo) if old_ppo.is_file() else None
    old_sha = sha256(old_ppo) if old_ppo.is_file() else None

    validation_refs = {}
    for name, filename in (("old_ppo", "validate_old_ppo.json"), ("dt", "validate_dt.json")):
        path = run_dir / filename
        if path.is_file():
            payload = json.loads(path.read_text())
            validation = payload.get("validation", payload)
            validation_refs[name] = {
                "mean_profit": validation["mean_profit"],
                "median_profit": validation["median_profit"],
                "mean_profit_by_battery": validation["mean_profit_by_battery"],
                "n_episodes": validation["n_episodes"],
                "source_file": str(path.relative_to(ROOT)),
            }

    # Shipped PPO means on the four identity surfaces (for the old-vs-new table).
    canonical = {
        "standard_year_2024": "eval_output/physics_v2/standard_year_v2cal",
        "dispatch_year_2024": "eval_output/physics_v2/dispatch_year_v2cal",
        "expanded_broad_2024": "eval_output/physics_v2/expanded_full_v2cal",
        "2025_ood": "eval_output/physics_v2/2025_v2cal_fresh",
    }
    old_ppo_surface_means = {}
    dt_surface_means = {}
    for surface, rel in canonical.items():
        path = ROOT / rel / "heldout_metrics_by_scenario.csv"
        if not path.is_file():
            continue
        frame = pl.read_csv(path)
        for policy in ("ppo_reference", "candidate_dt"):
            rows = frame.filter(pl.col("policy_name") == policy)
            if rows.height:
                target = old_ppo_surface_means if policy == "ppo_reference" else dt_surface_means
                target[surface] = {
                    "mean": float(rows["avg_profit_per_episode"].mean()),
                    "n": int(rows.height),
                    "source": rel,
                }

    extra = {}
    impact_seeds_path = ROOT / "paper/audit/artifacts/ppo_retrain_impact_seeds_20261125.json"
    if impact_seeds_path.is_file():
        extra["impact_seed_summary"] = json.loads(impact_seeds_path.read_text())
    default_std = ROOT / "eval_output/ppo_retrain_20261125/standard_year/default_seed0/heldout_metrics_by_scenario.csv"
    if default_std.is_file():
        frame = pl.read_csv(default_std)
        rows = frame.filter(pl.col("policy_name") == "ppo_reference")
        extra["default_config_4m_standard_surface"] = {
            "mean": float(rows["avg_profit_per_episode"].mean()),
            "n": int(rows.height),
            "source": str(default_std.relative_to(ROOT)),
        }

    payload = {
        "schema": "energydecision.ppo_retrain.v1",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "git_status_short": split_manifest.get("git_status_short"),
        "purpose": (
            "Retrain and evaluate a defensible PPO baseline for the AEMO Decision Transformer "
            "paper. The shipped PPO reference is a 204,800-step unseeded default-configuration "
            "checkpoint; the manuscript describes a tuned 4M-step Optuna search."
        ),
        "protocol": split_manifest["protocol"],
        "data_files": split_manifest["data_files"],
        "split": split_manifest["split"],
        "validation_windows": split_manifest["validation_windows"],
        "old_ppo_checkpoint": {
            "path": str(Path(args.old_ppo)),
            "sha256": old_sha,
            "metadata": old_meta,
            "validation": validation_refs.get("old_ppo"),
        },
        "dt_validation_reference": validation_refs.get("dt"),
        "optuna": optuna,
        "runs": runs,
        # Headline statistics only; the full per-cell bootstrap payload lives in
        # --stats (referenced below) so the two files cannot drift apart.
        "statistics_summary": _statistics_summary(stats, args.stats),
        "shipped_reference_means": {
            "old_ppo": old_ppo_surface_means,
            "dt": dt_surface_means,
        },
        "extras": extra,
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
