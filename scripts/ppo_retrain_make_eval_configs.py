"""Generate PPO-retrain evaluation configs without touching the canonical ones.

Reads a canonical evaluator config and writes a copy that
  * points ``ppo_reference.model_path`` at a retrained checkpoint,
  * points ``reference_cache_dir`` at a fresh cache directory,
  * points ``heldout.cache_dir`` at the read-only-safe data mirror,
  * leaves the candidate DT (``rtg_mode=j_t_soc``) and every held-out
    scenario/battery definition untouched.

Usage (repo root):
    python3 scripts/ppo_retrain_make_eval_configs.py \
        --model models/aemo_sb3/ppo_retrain_20261125/tuned_seed0_4014k.zip \
        --tag seed0 \
        --cache-root eval_output/ppo_retrain_20261125/reference_cache \
        --data-cache data/aemo_eval_mirror_20261125

Writes ``configs/aemo_autoresearch_evaluator.ppo_retrain_<surface>__<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SURFACES = {
    "standard_year": "configs/aemo_autoresearch_evaluator.sdp_teacher_standard_year.json",
    "dispatch_year": "configs/aemo_autoresearch_evaluator.sdp_teacher_dispatch_year.json",
    "expanded_full": "configs/aemo_autoresearch_evaluator.sdp_teacher_expanded_full.json",
    "fresh_2025": "configs/aemo_autoresearch_evaluator.sdp_teacher_2025_fresh.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache-root", default="eval_output/ppo_retrain_20261125/reference_cache")
    ap.add_argument("--data-cache", default="data/aemo_eval_mirror_20261125")
    ap.add_argument("--out-prefix", default="configs/aemo_autoresearch_evaluator.ppo_retrain")
    ap.add_argument("--surfaces", nargs="*", default=None,
                    help="subset of surface keys (default: all)")
    ap.add_argument("--policies", default=None,
                    help="comma-separated policy names to keep (default: keep all canonical)")
    ap.add_argument("--fit-global-stats", choices=["inherit", "true", "false"], default="inherit")
    ap.add_argument("--parallelize-candidate-dt", dest="parallelize_candidate_dt",
                    action="store_true", default=None,
                    help="override heldout.parallelize_candidate_dt (default: keep canonical)")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    data_cache = (ROOT / args.data_cache).resolve()
    if not data_cache.is_dir():
        raise SystemExit(f"data cache mirror not found: {data_cache}")

    wanted = list(SURFACES) if not args.surfaces else list(args.surfaces)
    keep = None if not args.policies else {p.strip() for p in args.policies.split(",")}
    for surface in wanted:
        rel = SURFACES[surface]
        src = ROOT / rel
        config = json.loads(src.read_text())
        policies = []
        found = False
        for policy in config["policies"]:
            policy = dict(policy)
            if policy.get("name") == "ppo_reference":
                policy["model_path"] = str(model)
                found = True
            if policy.get("name") == "candidate_dt":
                if policy.get("rtg_mode") != "j_t_soc":
                    raise SystemExit(
                        f"{rel}: candidate_dt rtg_mode is {policy.get('rtg_mode')!r}, expected 'j_t_soc'"
                    )
            policies.append(policy)
        if not found:
            raise SystemExit(f"{rel}: no ppo_reference policy to re-point")
        if keep is not None:
            policies = [p for p in policies if p.get("name") in keep]
            # the dispatch surface validates that its reference policy is present
            if any(p.get("kind") == "dispatch" for p in config.get("policies", [])):
                for original in config["policies"]:
                    if original.get("kind") == "dispatch" and original.get("name") not in keep:
                        policies.append(original)
        config["policies"] = policies
        config["reference_cache_dir"] = f"{args.cache_root}/{surface}__{args.tag}"
        config["heldout"] = dict(config["heldout"])
        config["heldout"]["cache_dir"] = str(data_cache)
        if args.fit_global_stats != "inherit":
            config["heldout"]["fit_global_stats"] = args.fit_global_stats == "true"
        if args.parallelize_candidate_dt is not None:
            config["heldout"]["parallelize_candidate_dt"] = bool(args.parallelize_candidate_dt)
        config["_provenance"] = {
            "derived_from": rel,
            "ppo_model_path": str(model),
            "generated_by": "scripts/ppo_retrain_make_eval_configs.py",
            "note": "candidate_dt, scenarios, battery variants and physics copied verbatim",
        }
        out = ROOT / f"{args.out_prefix}_{surface}__{args.tag}.json"
        out.write_text(json.dumps(config, indent=2) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
