"""Select tuned-PPO checkpoints by validation and run the four identity surfaces.

Selection rule (pre-specified, validation-only):
  1. for every seed, pick the milestone checkpoint with the highest held-out
     pre-2024 validation mean profit;
  2. the paper-facing ("primary") seed is the one whose selected checkpoint has
     the *median* validation profit of the three seeds -- never the best seed;
  3. every seed is then evaluated on every surface.

Ordering matters because the evaluator's ``fit_global_stats`` path rewrites the
processed scenario parquet files in the data mirror. Standard/expanded/2025 read
the canonical cached state; the dispatch surface needs its own SA1-only global
stats, so it runs last (and first with ``fit_global_stats=true``).

Usage (inside Distrobox, from the repo root):
    python3 scripts/ppo_retrain_run_evaluations.py \
        --run-dir models/aemo_sb3/ppo_retrain_20261125 \
        --eval-root eval_output/ppo_retrain_20261125 \
        --seeds tuned_seed0 tuned_seed1 tuned_seed2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ppo_retrain_make_eval_configs as cfgmod  # noqa: E402

SURFACE_ORDER = ["standard_year", "expanded_full", "fresh_2025", "dispatch_year"]


def select_checkpoints(run_dir: Path, seeds: list[str]) -> dict:
    selection = {}
    for tag in seeds:
        meta = json.loads((run_dir / f"{tag}_meta.json").read_text())
        milestones = [
            {
                "model_num_timesteps": h["model_num_timesteps"],
                "val_mean_profit": h["validation"]["mean_profit"],
                "checkpoint": h["checkpoint"],
                "checkpoint_sha256": h["checkpoint_sha256"],
            }
            for h in meta["history"]
        ]
        best = max(milestones, key=lambda m: m["val_mean_profit"])
        selection[tag] = {
            "seed": meta["seed"],
            "selected": best,
            "milestones": milestones,
        }
    ranked = sorted(selection, key=lambda t: selection[t]["selected"]["val_mean_profit"])
    primary = ranked[len(ranked) // 2]  # median of an odd number of seeds
    return {"primary_tag": primary, "ranking_by_validation": ranked, "seeds": selection}


def write_config(surface: str, model: str, tag: str, *, cache_root: str, data_cache: str,
                 policies: list[str] | None, fit_global_stats: str, parallelize: bool,
                 out_prefix: str) -> Path:
    import json as _json

    rel = cfgmod.SURFACES[surface]
    config = _json.loads((ROOT / rel).read_text())
    out_policies = []
    for policy in config["policies"]:
        policy = dict(policy)
        if policy.get("name") == "ppo_reference":
            policy["model_path"] = model
        out_policies.append(policy)
    if policies is not None:
        keep = set(policies)
        filtered = [p for p in out_policies if p.get("name") in keep]
        for original in out_policies:
            if original.get("kind") == "dispatch":
                filtered.append(original)
        out_policies = filtered
    config["policies"] = out_policies
    config["reference_cache_dir"] = f"{cache_root}/{surface}__{tag}"
    config["heldout"] = dict(config["heldout"])
    config["heldout"]["cache_dir"] = str((ROOT / data_cache).resolve())
    config["heldout"]["parallelize_candidate_dt"] = parallelize
    if fit_global_stats != "inherit":
        config["heldout"]["fit_global_stats"] = fit_global_stats == "true"
    config["_provenance"] = {
        "derived_from": rel,
        "ppo_model_path": model,
        "generated_by": "scripts/ppo_retrain_run_evaluations.py",
    }
    path = ROOT / f"{out_prefix}_{surface}__{tag}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def run_evaluator(config_path: Path, output_dir: Path, log_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "autoresearch_evaluator.py"),
        "--surface-manifest-path",
        "models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json",
        "--model-path",
        "models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt",
        "--loss-csv-path",
        "models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss.csv",
        "--evaluation-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--device",
        "auto",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--cache-root", default="eval_output/ppo_retrain_20261125/reference_cache")
    ap.add_argument("--data-cache", default="data/aemo_eval_mirror_20261125")
    ap.add_argument("--out-prefix", default="configs/aemo_autoresearch_evaluator.ppo_retrain")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    run_dir = ROOT / args.run_dir
    eval_root = ROOT / args.eval_root
    selection = select_checkpoints(run_dir, args.seeds)
    primary = selection["primary_tag"]
    (eval_root / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2))
    print(f"primary seed (median validation): {primary}")
    for tag in selection["ranking_by_validation"]:
        sel = selection["seeds"][tag]
        print(
            f"  {tag}: {sel['selected']['model_num_timesteps']:,} steps "
            f"val ${sel['selected']['val_mean_profit']:,.2f}"
        )

    for surface in SURFACE_ORDER:
        for tag in [primary] + [t for t in args.seeds if t != primary]:
            out_dir = eval_root / surface / tag
            if (out_dir / "heldout_metrics_by_scenario.csv").is_file():
                print(f"skip existing {out_dir}")
                continue
            model = ROOT / selection["seeds"][tag]["selected"]["checkpoint"]
            is_primary = tag == primary
            if surface == "dispatch_year":
                policies = ["candidate_dt", "ppo_reference"] if is_primary else ["ppo_reference"]
                # first (primary) dispatch run must materialise the canonical
                # SA1-only global stats before the other seeds read the cache
                fit = "true" if is_primary else "false"
            else:
                # expanded_full is DT-free (its shipped DT cells are reused: the
                # 150-episode re-rollout is prohibitively slow and two independent
                # spot checks reproduced the shipped cells exactly)
                dt_surfaces = {"standard_year", "fresh_2025"}
                policies = (
                    ["candidate_dt", "ppo_reference"]
                    if is_primary and surface in dt_surfaces
                    else ["ppo_reference"]
                )
                fit = "false"
            config_path = write_config(
                surface,
                str(model),
                tag,
                cache_root=args.cache_root,
                data_cache=args.data_cache,
                policies=policies,
                fit_global_stats=fit,
                parallelize=False,
                out_prefix=args.out_prefix,
            )
            print(f"running {surface}/{tag} (fit_global_stats={fit}, policies={policies})")
            run_evaluator(config_path, out_dir, run_dir / "logs" / f"eval_{surface}_{tag}.log")
            print(f"  done {surface}/{tag}")

    print("all surfaces complete")


if __name__ == "__main__":
    main()
