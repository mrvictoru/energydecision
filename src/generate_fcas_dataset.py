"""Generate FCAS-rich episodes from SB3 models and build DT dataset.

Usage:
  # Test run: one region, one policy, one battery, short horizon
  python3 src/generate_fcas_dataset.py --mode test

  # Full PPO generation (all regions, 3 horizons, all batteries)
  python3 src/generate_fcas_dataset.py --policies ppo

  # Full generation (all SB3 models)
  python3 src/generate_fcas_dataset.py --policies ppo,td3

  # Build final dataset (after all episodes are generated)
  python3 src/generate_fcas_dataset.py --mode assemble
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ── Scenario config (same date ranges as fetched data) ──────────────
SCENARIOS: list[dict[str, Any]] = [
    {"label": "nsw1_2021_2023", "region": "NSW1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"label": "qld1_2021_2023", "region": "QLD1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"label": "sa1_2022_2023",  "region": "SA1",  "start_date": "2022-04-01", "end_date": "2023-12-01"},
    {"label": "tas1_2021_2023", "region": "TAS1", "start_date": "2021-01-01", "end_date": "2023-04-01"},
    {"label": "vic1_2021_2023", "region": "VIC1", "start_date": "2021-04-01", "end_date": "2023-12-01"},
]

# ── Episode horizon config ──────────────────────────────────────────
HORIZONS: dict[str, int] = {
    "short": 3456,    # 12 days  (288h)
    "medium": 16128,  # 8 weeks  (1344h)
    "long": 74880,    # 26 weeks (6240h)
}

# ── Battery variants ────────────────────────────────────────────────
BATTERIES: dict[str, dict[str, float]] = {
    "small":  {"capacity_mwh": 2.0, "max_power_mw": 1.0, "init_soc_ratio": 0.5},
    "medium": {"capacity_mwh": 10.0, "max_power_mw": 5.0, "init_soc_ratio": 0.5},
    "large":  {"capacity_mwh": 50.0, "max_power_mw": 25.0, "init_soc_ratio": 0.5},
}

# ── Policy config: (model_path_suffix, total_episodes, battery_distribution) ──
# Battery distribution: how many episodes per battery variant
POLICIES: dict[str, dict[str, Any]] = {
    "ppo": {
        "model": "ppo_aemo_model.zip",
        "algorithm": "PPO",
        "total_episodes": 180,
        "batteries": {"medium": 0.60, "small": 0.25, "large": 0.15},
        "horizon_split": {"short": 0.33, "medium": 0.33, "long": 0.34},
    },
    "td3": {
        "model": "td3_aemo_model.zip",
        "algorithm": "TD3",
        "total_episodes": 60,
        "batteries": {"medium": 0.50, "small": 0.30, "large": 0.20},
        "horizon_split": {"short": 0.33, "medium": 0.33, "long": 0.34},
    },
    "a2c": {
        "model": "a2c_aemo_model.zip",
        "algorithm": "A2C",
        "total_episodes": 60,
        "batteries": {"medium": 0.50, "small": 0.30, "large": 0.20},
        "horizon_split": {"short": 0.33, "medium": 0.33, "long": 0.34},
    },
    "ddpg": {
        "model": "ddpg_aemo_model.zip",
        "algorithm": "DDPG",
        "total_episodes": 60,
        "batteries": {"medium": 0.50, "small": 0.30, "large": 0.20},
        "horizon_split": {"short": 0.33, "medium": 0.33, "long": 0.34},
    },
    "sac": {
        "model": "sac_aemo_model.zip",
        "algorithm": "SAC",
        "total_episodes": 60,
        "batteries": {"medium": 0.50, "small": 0.30, "large": 0.20},
        "horizon_split": {"short": 0.33, "medium": 0.33, "long": 0.34},
    },
    "fcas_rule": {
        "model": None,
        "algorithm": "fcas_rule",
        "total_episodes": 60,
        "batteries": {"medium": 0.50, "small": 0.30, "large": 0.20},
        "horizon_split": {"short": 0.33, "medium": 0.33, "long": 0.34},
    },
}

STEP_DURATION = 5 / 60


def load_processed_data(region: str, cache_dir: Path) -> Any:
    """Load a cached processed_data DataFrame for a region."""
    import polars as pl
    # Find the cached file
    candidates = sorted(cache_dir.glob(f"processed_{region}_*_*_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No cached data for {region} in {cache_dir}")
    # Use the first match (there should be only one per region now)
    path = candidates[0]
    print(f"  Loading {path.name}")
    return pl.read_parquet(str(path))


def build_episode_plan() -> list[dict[str, Any]]:
    """Build a flat list of all episodes to generate."""
    plans = []
    for policy_name, pcfg in POLICIES.items():
        for battery_name, battery_frac in pcfg["batteries"].items():
            for horizon_name, horizon_frac in pcfg["horizon_split"].items():
                n_eps = max(1, round(pcfg["total_episodes"] * battery_frac * horizon_frac))
                plans.append({
                    "policy": policy_name,
                    "algorithm": pcfg["algorithm"],
                    "model": pcfg["model"],
                    "battery": battery_name,
                    "horizon": horizon_name,
                    "max_step": HORIZONS[horizon_name],
                    "num_episodes": n_eps,
                })
    return plans


def _is_rule_based(policy_name: str) -> bool:
    pcfg = POLICIES.get(policy_name, {})
    return pcfg.get("model") is None or pcfg.get("algorithm", "").lower() in ("rule", "fcas_rule")


def generate_episodes(
    *,
    processed_data: Any,
    model_path: Path | None,
    algorithm: str,
    battery: dict[str, float],
    num_episodes: int,
    max_step: int,
    horizon_name: str,
    policy_name: str,
    scenario_label: str,
    output_dir: Path,
) -> list[Path]:
    """Generate episodes and save raw logs. Returns list of saved parquet paths."""
    import polars as pl
    from aemo_notebook_utils import run_sb3_episodes, run_rule_episodes

    is_rule = _is_rule_based(policy_name)
    deg_kw = dict(action_mode="multi_market", degradation_mode="real_world",
                   degradation_chemistry="LFP", degradation_temperature=30.0)

    if is_rule:
        episodes = run_rule_episodes(
            processed_data=processed_data,
            num_episodes=num_episodes,
            battery_capacity=battery["capacity_mwh"],
            max_battery_flow=battery["max_power_mw"],
            init_soc=battery["capacity_mwh"] * battery["init_soc_ratio"],
            step_duration=STEP_DURATION,
            battery_life_cost=0.0,
            max_step=max_step,
            random_episode_start=True,
            base_seed=42,
            algorithm=algorithm,
            **deg_kw,
        )
    else:
        episodes = run_sb3_episodes(
            processed_data=processed_data,
            battery_variant=battery,
            model_path=str(model_path),
            algorithm=algorithm,
            num_episodes=num_episodes,
            max_step=max_step,
            step_duration=STEP_DURATION,
            random_episode_start=True,
            deterministic=True,
            device="auto",
            **deg_kw,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, ep_df in enumerate(episodes):
        ep_tag = f"{scenario_label}__{policy_name}__{horizon_name}__{battery['name']}__ep{idx:03d}"
        ep_path = output_dir / f"{ep_tag}.parquet"
        ep_df.write_parquet(str(ep_path))
        saved.append(ep_path)
    return saved


def generate_policy_region(
    *,
    policy_name: str,
    scenario: dict[str, Any],
    plan_entries: list[dict[str, Any]],
    output_dir: Path,
    cache_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    """Generate all planned episodes for one (policy, region) combo.

    Loads the processed data and model once, then generates all episodes.
    Returns a summary dict.
    """
    print(f"\n{'='*60}")
    print(f"Generating {policy_name} on {scenario['label']}")
    print(f"{'='*60}")

    # Load processed data once per region
    processed = load_processed_data(scenario["region"], cache_dir)

    is_rule = _is_rule_based(policy_name)
    model_path: Path | None = None
    if not is_rule:
        model_path = models_dir / POLICIES[policy_name]["model"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

    results = []
    for entry in plan_entries:
        battery = dict(BATTERIES[entry["battery"]])
        battery["name"] = entry["battery"]
        horizon = entry["horizon"]
        max_step = entry["max_step"]
        n_eps = entry["num_episodes"]

        if n_eps < 1:
            continue

        print(f"  {entry['battery']:>6} | {horizon:>6} ({max_step:>5} steps) | {n_eps} episodes")

        ep_paths = generate_episodes(
            processed_data=processed,
            model_path=model_path,
            algorithm=POLICIES[policy_name]["algorithm"],
            battery=battery,
            num_episodes=n_eps,
            max_step=max_step,
            horizon_name=horizon,
            policy_name=policy_name,
            scenario_label=scenario["label"],
            output_dir=output_dir / "raw_logs" / f"{scenario['label']}",
        )

        for p in ep_paths:
            results.append({
                "path": str(p),
                "scenario": scenario["label"],
                "region": scenario["region"],
                "policy": policy_name,
                "battery": entry["battery"],
                "horizon": horizon,
                "max_step": max_step,
            })

    return {"policy": policy_name, "scenario": scenario["label"], "episodes": results}


def assemble_dataset(
    *,
    output_dir: Path,
    root: Path,
    keep_old_rule: int = 20,
) -> dict[str, Any]:
    """Scan raw_logs directory for episode files and build a DT dataset."""
    import polars as pl
    from aemo_notebook_utils import build_dt_dataset_from_logs

    print(f"\n{'='*60}")
    print("Assembling final DT dataset")
    print(f"{'='*60}")

    # 1. Scan raw_logs directory for all episode files
    raw_logs_dir = output_dir / "raw_logs"
    if not raw_logs_dir.exists():
        raise FileNotFoundError(f"raw_logs not found: {raw_logs_dir}")

    all_ep_files = sorted(raw_logs_dir.rglob("*.parquet"))
    print(f"  Found {len(all_ep_files)} episode files in raw_logs")

    log_groups: dict[str, list[pl.DataFrame]] = defaultdict(list)
    for path in all_ep_files:
        # Filename format: {scenario}__{policy}__{horizon}__{battery}__ep{idx}.parquet
        stem = path.stem
        parts = stem.split("__")
        # parts[0] = scenario (e.g. nsw1_2021_2023)
        # parts[1] = policy (e.g. ppo, td3)
        # parts[-1] = epXXX
        if len(parts) >= 4:
            policy = parts[1]
        else:
            policy = "unknown"
        df = pl.read_parquet(str(path))
        if "episode_id" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("episode_id"))
        log_groups[policy].append(df)

    # 2. Load old rule episodes (keep `keep_old_rule` of them)
    old_dataset_path = root / "data" / "aemo_dt" / "aemo_dt_dataset.parquet"
    if old_dataset_path.exists():
        old_df = pl.read_parquet(str(old_dataset_path))
        # Filter to rule episodes
        old_rule = old_df.filter(pl.col("source_policy").str.contains("rule"))
        # Sample up to keep_old_rule unique episodes
        rule_ep_ids = old_rule["episode_id"].unique().to_list()
        import random
        random.seed(42)
        selected = random.sample(rule_ep_ids, min(keep_old_rule, len(rule_ep_ids)))
        old_rule_sample = old_rule.filter(pl.col("episode_id").is_in(selected))
        log_groups["old_rule"] = [old_rule_sample]
        print(f"  Added {len(selected)} old rule episodes (from {len(rule_ep_ids)} available)")

    # 3. Build DT dataset
    dataset, manifest = build_dt_dataset_from_logs(dict(log_groups))

    # 4. Save
    out_path = output_dir / "aemo_fcas_dataset.parquet"
    dataset.write_parquet(str(out_path))
    manifest_path = output_dir / "aemo_fcas_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    total_eps = manifest["episode_count"]
    print(f"  Saved {total_eps} episodes ({dataset.height:,} rows)")
    print(f"  Dataset: {out_path}")
    print(f"  Manifest: {manifest_path}")

    return manifest


def main() -> None:
    root = repo_root()
    sys.path.insert(0, str(root / "src"))

    parser = argparse.ArgumentParser(description="Generate FCAS-rich DT dataset")
    parser.add_argument("--mode", choices=["generate", "assemble", "test"], default="generate",
                        help="generate: run SB3 rollouts; assemble: build final dataset; test: quick test")
    parser.add_argument("--policies", type=str, default="ppo",
                        help="Comma-separated policies to generate (ppo,td3,a2c,ddpg,sac)")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "aemo_dt_fcas",
                        help="Output directory for generated data")
    parser.add_argument("--cache-dir", type=Path, default=root / "data" / "aemo",
                        help="AEMO processed data cache")
    parser.add_argument("--models-dir", type=Path, default=root / "models" / "aemo_sb3",
                        help="SB3 model directory")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    models_dir = args.models_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "assemble":
        manifest = assemble_dataset(
            output_dir=output_dir,
            root=root,
        )
        print(json.dumps(manifest, indent=2))
        return

    if args.mode == "test":
        # Quick test: one region, one policy, one battery, short horizon
        policies = ["ppo"]
        test_output = output_dir / "test"
        test_output.mkdir(parents=True, exist_ok=True)

        for policy_name in policies:
            scenario = SCENARIOS[0]  # NSW1
            processed = load_processed_data(scenario["region"], cache_dir)
            model_path = models_dir / POLICIES[policy_name]["model"]

            battery = dict(BATTERIES["medium"])
            battery["name"] = "medium"

            n_eps = 2
            max_step = HORIZONS["short"]

            print(f"\nTest: {policy_name} on {scenario['label']}, medium, short, {n_eps} episodes")
            paths = generate_episodes(
                processed_data=processed,
                model_path=model_path,
                algorithm=POLICIES[policy_name]["algorithm"],
                battery=battery,
                num_episodes=n_eps,
                max_step=max_step,
                horizon_name="short",
                policy_name=policy_name,
                scenario_label=scenario["label"],
                output_dir=test_output / "raw_logs",
            )
            print(f"  Generated {len(paths)} episodes in {test_output}")
            # Check file sizes
            for p in paths:
                size_mb = p.stat().st_size / 1024 / 1024
                print(f"  {p.name}: {size_mb:.1f} MB")
        return

    # ── Full generation mode ──────────────────────────────
    policies = [p.strip() for p in args.policies.split(",")]
    plan = build_episode_plan()
    filtered_plan = [e for e in plan if e["policy"] in policies]

    # Group by policy for progress tracking
    by_policy = defaultdict(list)
    for e in filtered_plan:
        by_policy[e["policy"]].append(e)

    all_episodes: list[dict[str, Any]] = []

    for policy_name in policies:
        policy_entries = by_policy[policy_name]
        for scenario in SCENARIOS:
            try:
                result = generate_policy_region(
                    policy_name=policy_name,
                    scenario=scenario,
                    plan_entries=policy_entries,
                    output_dir=output_dir,
                    cache_dir=cache_dir,
                    models_dir=models_dir,
                )
                all_episodes.extend(result["episodes"])
            except Exception as e:
                print(f"  ERROR generating {policy_name} on {scenario['label']}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next

    # Save generation manifest
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "policies": policies,
        "total_episodes": len(all_episodes),
        "episodes": all_episodes,
    }
    manifest_path = output_dir / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nGeneration manifest: {manifest_path}")
    print(f"Total episodes generated: {len(all_episodes)}")


if __name__ == "__main__":
    main()
