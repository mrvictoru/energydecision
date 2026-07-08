from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from aemo_notebook_utils import (
    create_aemo_env,
    fetch_and_preprocess_aemo_scenarios,
    resolve_battery_variants,
)
from decision import AEMOAgent
from decision_transformer import DecisionTransformer
from huggingface_hub import hf_hub_download
from grpo_posttraining import (
    GRPOPrompt,
    GRPOTrainer,
    load_pretrained_dt_for_grpo,
    sample_rtg_values,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-region GRPO post-training for AEMO DT")

    parser.add_argument("--hf-repo", default="mrvictoru/energydecision-dt")
    parser.add_argument("--hf-filename", default="aemo_dt_fcas_model.pt")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "models" / "aemo" / "dt" / "grpo_multi")

    # Regions (comma-separated)
    parser.add_argument("--regions", default="NSW1,SA1,QLD1", help="Comma-separated AEMO regions")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-01-14")
    parser.add_argument("--episode-hours", type=float, default=144.0)

    # GRPO params
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coeff", type=float, default=0.02)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)
    parser.add_argument("--rtg-count", type=int, default=4)
    parser.add_argument("--rtg-spread", type=float, default=3.0)
    parser.add_argument("--rtg-dist", default="gaussian")
    parser.add_argument("--dt-gamma", type=float, default=1.0)
    parser.add_argument("--sync-reference-every", type=int, default=0)
    parser.add_argument("--adaptive-rtg", action="store_true", default=False)
    parser.add_argument("--adaptive-rtg-ewma-alpha", type=float, default=0.1)
    parser.add_argument("--deg-penalty-weight", type=float, default=1.0)

    parser.add_argument("--action-mode", default="full_fcas")
    parser.add_argument("--battery-capacity", type=float, default=10.0)
    parser.add_argument("--max-power", type=float, default=5.0)
    parser.add_argument("--step-duration", type=float, default=0.5)
    parser.add_argument("--baseline-eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")

    return parser.parse_args(argv)


def _resolve_device(device: str) -> str:
    d = str(device).strip().lower()
    return "cuda" if d == "auto" and torch.cuda.is_available() else d


def evaluate_dt_policy(
    dt_model: DecisionTransformer,
    env_factory,
    episodes: int = 3,
    rtg_value: float = 0.0,
    dt_gamma: float = 1.0,
    base_seed: int = 2026,
) -> pl.DataFrame:
    rows = []
    for seed in range(episodes):
        env = env_factory()
        agent = AEMOAgent(env, algorithm="dt", model=dt_model, rtg_value=rtg_value, dt_gamma=dt_gamma, reset_seed=base_seed + seed)
        ep_df, _ = agent.run_episode()
        info = ep_df["info"].struct.unnest()
        rows.append({
            "episode": seed,
            "reward_sum": float(ep_df["reward"].sum()),
            "energy_revenue": float(info["energy_revenue"].sum()) if "energy_revenue" in info.columns else 0.0,
            "fcas_revenue": float(info["fcas_revenue"].sum()) if "fcas_revenue" in info.columns else 0.0,
        })
    return pl.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    root = repo_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download HF model
    print(f"[GRPO] Downloading DT from {args.hf_repo}/{args.hf_filename}...")
    checkpoint_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename)

    model_kwargs = {"state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384, "context_len": 180, "n_heads": 8, "drop_p": 0.15, "max_timestep": 100000}

    # 2. Preprocess all regions
    regions = [r.strip() for r in args.regions.split(",")]
    print(f"[GRPO] Preprocessing {len(regions)} regions: {regions}...")
    scenarios = []
    for region in regions:
        scenarios.append({
            "label": f"{region.lower()}_{args.start_date[:7]}",
            "region": region,
            "start_date": datetime.fromisoformat(args.start_date),
            "end_date": datetime.fromisoformat(args.end_date),
        })

    cache_dir = root / "data" / "aemo"
    processed_by_label, scenario_manifest = fetch_and_preprocess_aemo_scenarios(
        scenarios=scenarios, cache_dir=cache_dir, step_duration=args.step_duration, refresh=False,
    )

    battery_variant = resolve_battery_variants([{"name": "medium", "capacity_mwh": args.battery_capacity, "max_power_mw": args.max_power, "init_soc_ratio": 0.5}])[0]
    max_step = max(1, int(round(args.episode_hours / args.step_duration)))

    # Build per-region processed data store
    region_data: dict[str, pl.DataFrame] = {}
    for sm in scenario_manifest:
        region_data[sm["region"]] = processed_by_label[sm["label"]]
    region_names = list(region_data.keys())

    # 3. Create multi-region env factory
    def make_env():
        region = random.choice(region_names)
        return create_aemo_env(
            processed_data=region_data[region],
            battery_variant=battery_variant,
            max_step=max_step,
            step_duration=args.step_duration,
            action_mode=args.action_mode,
            random_episode_start=True,
        )

    # 4. Load model
    print(f"[GRPO] Loading model on {device}...")
    model, reference_model = load_pretrained_dt_for_grpo(model_kwargs, checkpoint_path, device=device)
    optimal_rtg = float(getattr(model, "return_scale", 1.0))
    print(f"[GRPO] return_scale = {optimal_rtg}")

    # 5. Baseline eval (on a single region for simplicity)
    print(f"[GRPO] Baseline eval ({args.baseline_eval_episodes} episodes)...")
    single_env = lambda: create_aemo_env(
        processed_data=region_data[region_names[0]],
        battery_variant=battery_variant,
        max_step=max_step,
        step_duration=args.step_duration,
        action_mode=args.action_mode,
        random_episode_start=True,
    )
    baseline = evaluate_dt_policy(model, single_env, episodes=args.baseline_eval_episodes, dt_gamma=args.dt_gamma, base_seed=args.seed)
    print(f"[GRPO] Baseline reward: {float(baseline['reward_sum'].mean()):.2f}")

    # 6. GRPO training
    target_rtg = args.rtg_count * optimal_rtg / 2  # center RTG on return_scale
    rtg_values = sample_rtg_values(optimum=target_rtg, spread=args.rtg_spread, count=args.rtg_count, distribution=args.rtg_dist, seed=args.seed)
    print(f"[GRPO] RTG prompts: {[round(v, 2) for v in rtg_values]}")

    prompts = [
        GRPOPrompt(seed=args.seed + idx, options={"random_episode_start": True}, rtg_value=rtg, max_steps=max_step)
        for idx, rtg in enumerate(rtg_values)
    ]

    print(
        "[GRPO] Phase 1 config: "
        f"sync_reference_every={args.sync_reference_every}, "
        f"adaptive_rtg={args.adaptive_rtg}, "
        f"deg_penalty_weight={args.deg_penalty_weight}"
    )

    trainer = GRPOTrainer(
        model,
        reference_model=reference_model,
        device=device,
        lr=args.lr,
        kl_coeff=args.kl_coeff,
        entropy_coeff=args.entropy_coeff,
        degradation_penalty_weight=args.deg_penalty_weight,
    )
    history = trainer.train(
        make_env,
        prompts=prompts,
        iterations=args.iterations,
        group_size=args.group_size,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        dt_gamma=args.dt_gamma,
        sync_reference_every=args.sync_reference_every,
        adaptive_rtg=args.adaptive_rtg,
        adaptive_rtg_spread=args.rtg_spread,
        adaptive_rtg_dist=args.rtg_dist,
        adaptive_rtg_ewma_alpha=args.adaptive_rtg_ewma_alpha,
        adaptive_rtg_seed=args.seed,
    )

    # 7. Post-GRPO eval
    print(f"[GRPO] Post-GRPO eval...")
    post = evaluate_dt_policy(model, single_env, episodes=args.baseline_eval_episodes, dt_gamma=args.dt_gamma, base_seed=args.seed + 100)
    imp = float(post["reward_sum"].mean() - baseline["reward_sum"].mean())
    print(f"[GRPO] Improvement: {imp:.2f}")

    # 8. Save
    save_path = output_dir / "dt_model_grpo_multi.pt"
    loss_csv = output_dir / "grpo_loss_history.csv"
    manifest_path = output_dir / "grpo_surface_manifest.json"

    torch.save({"model_state_dict": model.state_dict(), "meta": {"return_scale": float(getattr(model, "return_scale", 1.0))}}, save_path)

    pl.DataFrame([{
        "epoch": int(h.get("iteration", 0)), "train_total": h.get("loss", 0.0), "train_action": h.get("policy_loss", 0.0),
    } for h in history]).write_csv(loss_csv)

    manifest = {
        "schema": "energydecision.dt_training_surface.v1", "run_tag": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "surface_preset": "grpo_multi_region", "model_variant": "full_fcas", "action_mode": "full_fcas",
        "model_kwargs": model_kwargs,
        "training_kwargs": {
            "iterations": args.iterations,
            "lr": args.lr,
            "kl_coeff": args.kl_coeff,
            "entropy_coeff": args.entropy_coeff,
            "sync_reference_every": args.sync_reference_every,
            "adaptive_rtg": args.adaptive_rtg,
            "adaptive_rtg_ewma_alpha": args.adaptive_rtg_ewma_alpha,
            "degradation_penalty_weight": args.deg_penalty_weight,
        },
        "paths": {"save_path": str(save_path), "loss_csv_path": str(loss_csv)},
        "grpo_config": {
            "regions": regions,
            "iterations": args.iterations,
            "group_size": args.group_size,
            "rtg_count": args.rtg_count,
            "rtg_spread": args.rtg_spread,
            "rtg_dist": args.rtg_dist,
            "dt_gamma": args.dt_gamma,
        },
        "scenario": {"regions": regions, "start_date": args.start_date, "end_date": args.end_date, "episode_hours": args.episode_hours, "action_mode": args.action_mode},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"\n{'='*60}")
    print(f"Multi-region GRPO Complete")
    print(f"  Regions:     {regions}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Baseline:    {float(baseline['reward_sum'].mean()):.2f}")
    print(f"  Post-GRPO:   {float(post['reward_sum'].mean()):.2f}")
    print(f"  Improvement: {imp:+.2f}")
    print(f"  Model:       {save_path}")
    print(f"  Manifest:    {manifest_path}")
    print(f"  Eval next: python3 src/autoresearch_evaluator.py \\")
    print(f"    --surface-manifest-path {manifest_path} \\")
    print(f"    --evaluation-config <config> \\")
    print(f"    --output-dir eval_output/autoresearch/grpo_multi")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
