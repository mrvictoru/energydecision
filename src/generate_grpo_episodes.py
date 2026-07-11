"""Generate DT dataset episodes from the best GRPO-tuned model.

Generates episodes on 2021-2023 AEMO data (same period as SB3 models)
using the Phase 1 GRPO model. Outputs episodes in the same format as
``generate_fcas_dataset.py`` so they can be combined with the existing v2
dataset for retraining the improved DecisionTransformer.

Usage:
    python3 src/generate_grpo_episodes.py --output-dir data/aemo_dt_fcas_v3
"""
from __future__ import annotations

import argparse
import sys
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch
from huggingface_hub import hf_hub_download

from aemo_notebook_utils import (
    create_aemo_env,
    fetch_and_preprocess_aemo_scenarios,
    resolve_battery_variants,
)
from decision import AEMOAgent
from decision_transformer import DecisionTransformer
from grpo_posttraining import load_pretrained_dt_for_grpo


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ── Scenario config (same date ranges as SB3 training data) ─────────
SCENARIOS: list[dict[str, Any]] = [
    {"label": "nsw1_2021_2023", "region": "NSW1", "start_date": datetime(2021, 1, 1), "end_date": datetime(2023, 4, 1)},
    {"label": "qld1_2021_2023", "region": "QLD1", "start_date": datetime(2021, 1, 1), "end_date": datetime(2023, 4, 1)},
    {"label": "sa1_2022_2023",  "region": "SA1",  "start_date": datetime(2022, 4, 1), "end_date": datetime(2023, 12, 1)},
    {"label": "tas1_2021_2023", "region": "TAS1", "start_date": datetime(2021, 1, 1), "end_date": datetime(2023, 4, 1)},
    {"label": "vic1_2021_2023", "region": "VIC1", "start_date": datetime(2021, 4, 1), "end_date": datetime(2023, 12, 1)},
]

# ── Episode horizon config ──────────────────────────────────────────
# Only short horizon is used for GRPO-generated data because:
# 1. DT context is only 180 timesteps (15 hours), so longer episodes don't add signal
# 2. Generation is 5-20x faster with short episodes
HORIZONS: dict[str, int] = {
    "short": 3456,    # 12 days  (288h)
}

# ── Battery variants (matches v2 dataset) ────────────────────────────
BATTERIES: dict[str, dict[str, float]] = {
    "small_05c":  {"capacity_mwh": 2.0, "max_power_mw": 1.0,  "init_soc_ratio": 0.5},
    "medium_1c":  {"capacity_mwh": 10.0, "max_power_mw": 10.0, "init_soc_ratio": 0.5},
    "large_07c":  {"capacity_mwh": 50.0, "max_power_mw": 35.0, "init_soc_ratio": 0.5},
    "fast_375c":  {"capacity_mwh": 8.0,  "max_power_mw": 30.0, "init_soc_ratio": 0.5},
}

# ── Episode budget ──────────────────────────────────────────────────
# Total episodes per policy: 180 for PPO-like, 60 for others
# GRPO is the primary contributor
TOTAL_EPISODES = 180
BATTERY_DIST = {"medium_1c": 0.40, "large_07c": 0.25, "small_05c": 0.20, "fast_375c": 0.15}
HORIZON_SPLIT = {"short": 1.0}

STEP_DURATION = 5 / 60  # 5 minutes
POLICY_NAME = "grpo_dt"
BASE_SEED = 2026


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate episodes from GRPO-trained DT")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "data" / "aemo_dt_fcas_v3",
                        help="Output directory for generated data")
    parser.add_argument("--cache-dir", type=Path, default=repo_root() / "data" / "aemo",
                        help="AEMO processed data cache")
    parser.add_argument("--model-repo", default="mrvictoru/energydecision-dt",
                        help="HF repo for the GRPO model")
    parser.add_argument("--model-filename", default="aemo_dt_grpo_model.pt",
                        help="Filename of the GRPO model in the HF repo")
    parser.add_argument("--rtg-value", type=float, default=0.5,
                        help="RTG value for DT inference (optimal from calibration)")
    parser.add_argument("--dt-gamma", type=float, default=0.95,
                        help="Discount factor for RTG updates")
    parser.add_argument("--total-episodes", type=int, default=TOTAL_EPISODES,
                        help="Total episodes to generate")
    parser.add_argument("--parallel-workers", type=int, default=4,
                        help="Number of parallel episode generation workers")
    parser.add_argument("--device", default="auto",
                        help="Device for model inference (auto, cuda, cpu)")
    return parser.parse_args(argv)


def _resolve_device(device: str) -> str:
    d = str(device).strip().lower()
    return "cuda" if d == "auto" and torch.cuda.is_available() else d


def build_episode_plan(total_episodes: int = TOTAL_EPISODES) -> list[dict[str, Any]]:
    """Build a flat list of all episodes to generate."""
    plans = []
    for battery_name, battery_frac in BATTERY_DIST.items():
        for horizon_name, horizon_frac in HORIZON_SPLIT.items():
            n_eps = max(1, round(total_episodes * battery_frac * horizon_frac))
            plans.append({
                "battery": battery_name,
                "horizon": horizon_name,
                "max_step": HORIZONS[horizon_name],
                "num_episodes": n_eps,
            })
    return plans


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    root = repo_root()
    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load GRPO model
    print(f"[GRPO] Loading model from {args.model_repo}/{args.model_filename}...")
    try:
        checkpoint_path = hf_hub_download(repo_id=args.model_repo, filename=args.model_filename)
    except Exception:
        # Fallback: try the Phase 1 local model
        local_path = root / "models" / "aemo" / "dt" / "grpo_phase1" / "dt_model_grpo_multi.pt"
        if local_path.exists():
            checkpoint_path = str(local_path)
            print(f"[GRPO] Using local model: {checkpoint_path}")
        else:
            raise FileNotFoundError(
                f"Model not found on HF ({args.model_repo}/{args.model_filename}) "
                f"or locally ({local_path})"
            )

    model_kwargs = {
        "state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384,
        "context_len": 180, "n_heads": 8, "drop_p": 0.15, "max_timestep": 100000,
    }
    model, reference_model = load_pretrained_dt_for_grpo(
        model_kwargs, checkpoint_path, device=device
    )
    model.eval()
    print(f"[GRPO] Model loaded (return_scale={getattr(model, 'return_scale', 1.0)})")

    # 2. Build episode plan
    plan = build_episode_plan(total_episodes=args.total_episodes)
    total_planned = sum(e["num_episodes"] for e in plan)
    print(f"[GRPO] Episode plan: {total_planned} episodes across {len(plan)} combos")

    # 3. Process each scenario (region)
    print(f"[GRPO] Preprocessing {len(SCENARIOS)} regions...")
    processed_by_label, scenario_manifest = fetch_and_preprocess_aemo_scenarios(
        scenarios=SCENARIOS,
        cache_dir=cache_dir,
        step_duration=STEP_DURATION,
        refresh=False,
    )

    all_episodes: list[dict[str, Any]] = []
    raw_logs_dir = output_dir / "raw_logs"
    parallel_workers = max(1, args.parallel_workers)

    def _generate_single(entry: dict[str, Any], scenario: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate one batch of episodes for a (battery, horizon) combo in one region."""
        region = scenario["region"]
        label = scenario["label"]
        processed_data = processed_by_label[label]
        battery_name = entry["battery"]
        battery_spec = BATTERIES[battery_name]
        horizon_name = entry["horizon"]
        max_step = entry["max_step"]
        n_eps = entry["num_episodes"]

        if n_eps < 1:
            return []

        battery_variant = resolve_battery_variants([{
            "name": battery_name,
            "capacity_mwh": battery_spec["capacity_mwh"],
            "max_power_mw": battery_spec["max_power_mw"],
            "init_soc_ratio": battery_spec["init_soc_ratio"],
        }])[0]

        results = []
        for ep_idx in range(n_eps):
            env = create_aemo_env(
                processed_data=processed_data,
                battery_variant=battery_variant,
                max_step=max_step,
                step_duration=STEP_DURATION,
                action_mode="full_fcas",
                degradation_mode="real_world",
                degradation_chemistry="LFP",
                degradation_temperature=30.0,
                random_episode_start=True,
            )
            agent = AEMOAgent(
                env,
                algorithm="dt",
                model=model,
                rtg_value=args.rtg_value,
                dt_gamma=args.dt_gamma,
                reset_seed=BASE_SEED + hash(f"{label}_{battery_name}_{horizon_name}_{ep_idx}") % (2**31),
            )
            episode_df, _ = agent.run_episode()

            ep_tag = f"{label}__{POLICY_NAME}__{horizon_name}__{battery_name}__ep{ep_idx:03d}"
            ep_path = raw_logs_dir / label / f"{ep_tag}.parquet"
            ep_path.parent.mkdir(parents=True, exist_ok=True)
            episode_df.write_parquet(str(ep_path))

            results.append({
                "path": str(ep_path),
                "scenario": label,
                "region": region,
                "policy": POLICY_NAME,
                "battery": battery_name,
                "horizon": horizon_name,
                "max_step": max_step,
            })
        return results

    # Build flat work list
    work_items: list[dict[str, Any]] = []
    for scenario in scenario_manifest:
        for entry in plan:
            work_items.append({"entry": entry, "scenario": scenario})

    total_work = len(work_items)
    print(f"[GRPO] Generating {total_work} battery×horizon×region combos with {parallel_workers} workers...")

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(_generate_single, item["entry"], item["scenario"]): item
            for item in work_items
        }
        done = 0
        for future in as_completed(futures):
            item = futures[future]
            entry = item["entry"]
            sc = item["scenario"]
            try:
                result = future.result()
                all_episodes.extend(result)
                done += 1
                print(f"  [{done}/{total_work}] {sc['label']} / {entry['battery']} / {entry['horizon']} — {len(result)} eps")
            except Exception as e:
                print(f"  [ERROR] {sc['label']} / {entry['battery']} / {entry['horizon']}: {e}")

    # 4. Save generation manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "policies": [POLICY_NAME],
        "total_episodes": len(all_episodes),
        "episodes": all_episodes,
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"  Generation complete: {len(all_episodes)} episodes")
    print(f"  Manifest: {manifest_path}")
    print(f"  Raw logs: {raw_logs_dir}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
