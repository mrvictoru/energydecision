from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch

from aemo_dt_hf import (
    MODERN_V2_HF_FILENAME,
    MODERN_V2_HF_REPO,
    load_model_kwargs,
    modern_v2_model_config_path,
)
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
    parser = argparse.ArgumentParser(description="GRPO post-training for AEMO Decision Transformer")

    # HF model
    parser.add_argument("--hf-repo", default=MODERN_V2_HF_REPO, help="HuggingFace repo")
    parser.add_argument("--hf-filename", default=MODERN_V2_HF_FILENAME, help="Checkpoint filename")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Local checkpoint path (overrides HF download).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=modern_v2_model_config_path(),
        help="Path to the Decision Transformer model kwargs JSON.",
    )

    # Output paths
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "models" / "aemo" / "dt" / "grpo", help="Output directory for model + artifacts")

    # GRPO config
    parser.add_argument("--iterations", type=int, default=5, help="GRPO iterations")
    parser.add_argument("--group-size", type=int, default=4, help="Group size per iteration")
    parser.add_argument("--update-epochs", type=int, default=2, help="Update epochs per iteration")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--kl-coeff", type=float, default=0.02, help="KL coefficient")
    parser.add_argument("--entropy-coeff", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--initial-log-std", type=float, default=-1.0, help="Initial log std")
    parser.add_argument("--no-trainable-log-std", action="store_true", default=False, help="Disable trainable log std")
    parser.add_argument("--dt-gamma", type=float, default=1.0, help="Discount factor for RTG updates in DT inference (1.0 = undiscounted)")

    # RTG sampling
    parser.add_argument("--rtg-count", type=int, default=4, help="Number of RTG values per group")
    parser.add_argument("--rtg-spread", type=float, default=3.0, help="RTG spread")
    parser.add_argument("--target-rtg", type=float, default=None, help="Override target RTG (default: model.return_scale)")

    # Environment config
    parser.add_argument("--region", default="NSW1", help="AEMO region")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-01-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--episode-hours", type=float, default=24.0, help="Episode length in hours")
    parser.add_argument("--action-mode", default="full_fcas", help="Action mode (full_fcas, multi_market, simple)")
    parser.add_argument("--battery-capacity", type=float, default=10.0, help="Battery capacity MWh")
    parser.add_argument("--max-power", type=float, default=5.0, help="Max power MW")
    parser.add_argument("--baseline-eval-episodes", type=int, default=5, help="Baseline eval episodes")
    parser.add_argument("--seed", type=int, default=2026, help="Base seed")

    # Step duration
    parser.add_argument("--step-duration", type=float, default=5.0 / 60.0, help="Step duration in hours")

    # Device
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")

    return parser.parse_args(argv)


def _resolve_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def evaluate_dt_policy(
    dt_model: DecisionTransformer,
    make_env,
    episodes: int = 5,
    rtg_value: float = 0.0,
    dt_gamma: float = 1.0,
    base_seed: int = 2026,
    random_episode_start: bool = True,
) -> pl.DataFrame:
    episode_rows = []
    for seed in range(episodes):
        env = make_env()
        agent = AEMOAgent(
            env,
            algorithm="dt",
            model=dt_model,
            rtg_value=rtg_value,
            dt_gamma=dt_gamma,
            reset_seed=base_seed + seed if random_episode_start else None,
        )
        episode_df, _ = agent.run_episode()
        info_series = episode_df["info"].struct.unnest()
        total_revenue = float(info_series["total_revenue"].tail(1).item()) if "total_revenue" in info_series.columns else float(episode_df["reward"].sum())
        episode_rows.append(
            {
                "episode": seed,
                "reward_sum": float(episode_df["reward"].sum()),
                "energy_revenue": float(info_series["energy_revenue"].sum()) if "energy_revenue" in info_series.columns else 0.0,
                "fcas_revenue": float(info_series["fcas_revenue"].sum()) if "fcas_revenue" in info_series.columns else 0.0,
                "total_revenue": total_revenue,
            }
        )
    return pl.DataFrame(episode_rows)


def build_surface_manifest(
    *,
    model_kwargs: dict[str, Any],
    save_path: Path,
    loss_csv_path: Path,
    run_tag: str,
    iterations: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": "energydecision.dt_training_surface.v1",
        "run_tag": run_tag,
        "surface_preset": "grpo_posttraining",
        "model_variant": "full_fcas",
        "action_mode": "full_fcas",
        "model_kwargs": model_kwargs,
        "training_kwargs": {
            "iterations": iterations,
            "lr": args.lr,
            "clip_ratio": args.clip_ratio,
            "kl_coeff": args.kl_coeff,
            "entropy_coeff": args.entropy_coeff,
        },
        "paths": {
            "save_path": str(save_path),
            "loss_csv_path": str(loss_csv_path),
        },
        "grpo_config": {
            "iterations": args.iterations,
            "group_size": args.group_size,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "rtg_count": args.rtg_count,
            "rtg_spread": args.rtg_spread,
            "dt_gamma": getattr(args, "dt_gamma", 1.0),
        },
        "scenario": {
            "region": args.region,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "episode_hours": args.episode_hours,
            "action_mode": args.action_mode,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    root = repo_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- 1. Load pretrained model ---
    if args.checkpoint is not None:
        checkpoint_path = str(args.checkpoint.resolve())
        print(f"[GRPO] Using local checkpoint: {checkpoint_path}")
    else:
        print(f"[GRPO] Downloading pretrained DT from {args.hf_repo}/{args.hf_filename} ...")
        checkpoint_path = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename)
        print(f"[GRPO] Checkpoint: {checkpoint_path}")
    print(f"[GRPO] Model config: {args.model_config.resolve()}")

    # Model kwargs matching the full_fcas checkpoint
    model_kwargs = load_model_kwargs(args.model_config)

    # --- 2. Fetch/preprocess AEMO data ---
    print(f"[GRPO] Setting up AEMO env for {args.region} {args.start_date}..{args.end_date} ...")
    scenario = {
        "label": f"{args.region.lower()}_{args.start_date[:7]}",
        "region": args.region,
        "start_date": datetime.fromisoformat(args.start_date),
        "end_date": datetime.fromisoformat(args.end_date),
    }
    cache_dir = root / "data" / "aemo"

    processed_by_label, scenario_manifest = fetch_and_preprocess_aemo_scenarios(
        scenarios=[scenario],
        cache_dir=cache_dir,
        step_duration=args.step_duration,
        refresh=False,
    )
    scenario_label = scenario_manifest[0]["label"]
    processed_df = processed_by_label[scenario_label]

    battery_variant = resolve_battery_variants(
        [
            {
                "name": "medium",
                "capacity_mwh": args.battery_capacity,
                "max_power_mw": args.max_power,
                "init_soc_ratio": 0.5,
            }
        ]
    )[0]
    max_step = max(1, min(processed_df.height, int(round(args.episode_hours / args.step_duration))))

    def make_env():
        return create_aemo_env(
            processed_data=processed_df,
            battery_variant=battery_variant,
            max_step=max_step,
            step_duration=args.step_duration,
            action_mode=args.action_mode,
            random_episode_start=True,
        )

    # --- 3. Load model ---
    print(f"[GRPO] Loading model on {device} ...")
    model, reference_model = load_pretrained_dt_for_grpo(model_kwargs, checkpoint_path, device=device)
    optimal_rtg = float(model.return_scale)
    print(f"[GRPO] Model return_scale = {optimal_rtg}")

    # --- 4. Baseline evaluation ---
    print(f"[GRPO] Baseline evaluation ({args.baseline_eval_episodes} episodes, rtg=0)...")
    baseline_eval = evaluate_dt_policy(
        model, make_env, episodes=args.baseline_eval_episodes, rtg_value=0.0,
        dt_gamma=args.dt_gamma, base_seed=args.seed
    )
    print(f"[GRPO] Baseline mean reward: {float(baseline_eval['reward_sum'].mean()):.2f}")
    print(f"[GRPO] Baseline mean FCAS revenue: {float(baseline_eval['fcas_revenue'].mean()):.2f}")

    # --- 5. GRPO training ---
    target_rtg = args.target_rtg if args.target_rtg is not None else optimal_rtg
    rtg_values = sample_rtg_values(
        optimum=target_rtg,
        spread=args.rtg_spread,
        count=args.rtg_count,
        seed=args.seed,
    )
    print(f"[GRPO] RTG prompts: {[round(v, 2) for v in rtg_values]}")

    prompts = [
        GRPOPrompt(
            seed=args.seed + idx,
            options={"random_episode_start": True},
            rtg_value=rtg,
            max_steps=max_step,
        )
        for idx, rtg in enumerate(rtg_values)
    ]

    trainer = GRPOTrainer(
        model,
        reference_model=reference_model,
        device=device,
        lr=args.lr,
        clip_ratio=args.clip_ratio,
        kl_coeff=args.kl_coeff,
        entropy_coeff=args.entropy_coeff,
        initial_log_std=args.initial_log_std,
        trainable_log_std=not args.no_trainable_log_std,
    )

    history = trainer.train(
        make_env,
        prompts=prompts,
        iterations=args.iterations,
        group_size=args.group_size,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        dt_gamma=args.dt_gamma,
    )

    # Log training history
    history_df = pl.DataFrame(history)
    print(f"[GRPO] Training history:\n{history_df}")

    # --- 6. Post-GRPO evaluation ---
    print(f"[GRPO] Post-GRPO evaluation ({args.baseline_eval_episodes} episodes, rtg=0)...")
    post_grpo_eval = evaluate_dt_policy(
        model, make_env, episodes=args.baseline_eval_episodes, rtg_value=0.0,
        dt_gamma=args.dt_gamma, base_seed=args.seed + 100
    )

    mean_reward_improvement = float(post_grpo_eval["reward_sum"].mean() - baseline_eval["reward_sum"].mean())
    mean_fcas_improvement = float(post_grpo_eval["fcas_revenue"].mean() - baseline_eval["fcas_revenue"].mean())
    print(f"[GRPO] Mean reward improvement: {mean_reward_improvement:.2f}")
    print(f"[GRPO] Mean FCAS revenue improvement: {mean_fcas_improvement:.2f}")

    # --- 7. Save model ---
    save_path = output_dir / "dt_model_grpo.pt"
    loss_csv_path = output_dir / "grpo_loss_history.csv"
    surface_manifest_path = output_dir / "grpo_surface_manifest.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": {"return_scale": float(getattr(model, "return_scale", 1.0))},
        },
        save_path,
    )
    print(f"[GRPO] Model saved to {save_path}")

    # Save a loss CSV for evaluator compatibility
    loss_rows = []
    for h in history:
        loss_rows.append(
            {
                "epoch": int(h.get("iteration", 0)),
                "train_total": h.get("loss", 0.0),
                "train_action": h.get("policy_loss", 0.0),
                "val_total": None,
                "val_action": None,
            }
        )
    pl.DataFrame(loss_rows).write_csv(loss_csv_path)
    print(f"[GRPO] Loss CSV saved to {loss_csv_path}")

    # Save surface manifest
    surface_manifest = build_surface_manifest(
        model_kwargs=model_kwargs,
        save_path=save_path,
        loss_csv_path=loss_csv_path,
        run_tag=run_tag,
        iterations=args.iterations,
        args=args,
    )
    surface_manifest_path.write_text(json.dumps(surface_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[GRPO] Surface manifest saved to {surface_manifest_path}")

    # --- 8. Summary ---
    print(f"\n{'='*60}")
    print(f"GRPO Post-Training Complete")
    print(f"{'='*60}")
    print(f"  Run tag:          {run_tag}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Iterations:       {args.iterations}")
    print(f"  Baseline reward:  {float(baseline_eval['reward_sum'].mean()):.2f}")
    print(f"  Post-GRPO reward: {float(post_grpo_eval['reward_sum'].mean()):.2f}")
    print(f"  Improvement:      {mean_reward_improvement:+.2f}")
    print(f"  Baseline FCAS:    {float(baseline_eval['fcas_revenue'].mean()):.2f}")
    print(f"  Post-GRPO FCAS:   {float(post_grpo_eval['fcas_revenue'].mean()):.2f}")
    print(f"  FCAS improvement: {mean_fcas_improvement:+.2f}")
    print(f"  Model:            {save_path}")
    print(f"  Manifest:         {surface_manifest_path}")
    print(f"{'='*60}")

    # Print evaluation command for next step
    print(f"\nTo evaluate with the dispatch_matched config:\n")
    print(f"  python3 src/autoresearch_evaluator.py \\")
    print(f"    --surface-manifest-path {surface_manifest_path} \\")
    print(f"    --evaluation-config configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json \\")
    print(f"    --output-dir eval_output/autoresearch/grpo_{run_tag} \\")
    print(f"    --device auto")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
