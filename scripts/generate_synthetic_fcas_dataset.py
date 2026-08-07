"""Generate synthetic-price and real-price FCAS episodes for the Phase 6 v2
downstream DT validation.

For each (policy, battery, horizon) cell, episodes are rolled out twice:
1. on a **synthetic** processed frame (real exogenous features, RRP + 8x FCAS
   replaced by FCASDiffusionGenerator samples), and
2. on the **real** processed frame (the control, same rollouts).

Both are normalized into DT datasets with identical schema, so training a DT on
each and evaluating both isolates whether synthetic-only prices carry the FCAS
signal needed for trading.

Usage:
  # Smoke: 2 episodes per cell, generator 1 epoch
  python3 scripts/generate_synthetic_fcas_dataset.py --smoke

  # Real run
  python3 scripts/generate_synthetic_fcas_dataset.py \
      --region NSW1 \
      --train-start 2024-07-01 --train-end 2025-01-01 \
      --generate-start 2024-07-01 --generate-end 2025-01-01 \
      --policies fcas_rule,ppo \
      --batteries medium_1c,small_05c \
      --horizon short \
      --episodes-per-cell 15 \
      --generator-epochs 12 \
      --output-dir data/aemo_dt_synth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import polars as pl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default="NSW1")
    parser.add_argument("--train-start", default="2024-07-01")
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument(
        "--train-spec",
        action="append",
        default=[],
        help="Repeatable REGION:YYYY-MM-DD:YYYY-MM-DD slice for the generator "
             "train set. When provided, overrides --region/--train-start/--train-end "
             "and concatenates all slices (e.g. a full-year multi-region set).",
    )
    parser.add_argument("--generate-start", default="2024-07-01")
    parser.add_argument("--generate-end", default="2025-01-01")
    parser.add_argument("--policies", default="fcas_rule,ppo")
    parser.add_argument("--batteries", default="medium_1c,small_05c")
    parser.add_argument("--horizon", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--episodes-per-cell", type=int, default=15)
    parser.add_argument("--generator-epochs", type=int, default=12)
    parser.add_argument("--generator-base-channels", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--synthetic-frame-path",
        type=Path,
        default=None,
        help="Path to a pre-generated synthetic price frame. When provided, the "
             "generator fit/sample is skipped and this frame is used (with "
             "--price-mode splicing applied). Reuses an expensive generator fit.",
    )
    parser.add_argument(
        "--price-mode",
        choices=["both", "fcas_only", "rrp_only"],
        default="both",
        help="both: synthetic RRP + FCAS; fcas_only: real RRP + synthetic FCAS; "
             "rrp_only: synthetic RRP + real FCAS. Used to attribute the "
             "downstream transfer failure to a specific price channel.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "aemo_dt_synth")
    return parser.parse_args()


def load_slice(region: str, start: str, end: str) -> pl.DataFrame:
    from eval_fcas_generator import load_interval

    return load_interval(region, start, end)


def build_generator(frame: pl.DataFrame, args: argparse.Namespace):
    from synthetic_fcas import FCASDiffusionGenerator

    return FCASDiffusionGenerator(
        window_size=288,
        stride=12,
        overlap=48,
        diffusion_steps=128,
        sample_steps=args.sample_steps,
        base_channels=args.generator_base_channels,
        epochs=args.generator_epochs,
        batch_size=32,
        tail_quantile=0.95,
        tail_weight=4.0,
        spike_quantile=0.99,
        sample_eta=0.05,
        schedule_seed=args.seed,
        tail_mode="schedule",
        seed=args.seed,
    ).fit(frame)


def synthetic_frame(gen, frame: pl.DataFrame, real: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Generate a synthetic price frame and, for isolation modes, splice in the
    real counterpart channel so only one price channel is synthetic."""
    from fcas_generator_eval import FCAS_COLS

    synth = gen.sample(frame)
    if mode == "fcas_only":
        synth = synth.with_columns(pl.Series("RRP", real["RRP"].to_numpy()))
    elif mode == "rrp_only":
        synth = synth.with_columns([pl.Series(col, real[col].to_numpy()) for col in FCAS_COLS])
    return synth


def run_cell(
    *,
    processed_data: pl.DataFrame,
    policy_name: str,
    battery: dict[str, float],
    battery_name: str,
    horizon: str,
    max_step: int,
    num_episodes: int,
    output_dir: Path,
    tag: str,
    seed: int,
) -> list[Path]:
    from generate_fcas_dataset import POLICIES, STEP_DURATION, _is_rule_based, generate_episodes

    battery["name"] = battery_name
    model_path = None
    if not _is_rule_based(policy_name):
        model_path = ROOT / "models" / "aemo_sb3" / POLICIES[policy_name]["model"]
        if not model_path.exists():
            raise FileNotFoundError(f"SB3 model not found: {model_path}")
    paths = generate_episodes(
        processed_data=processed_data,
        model_path=model_path,
        algorithm=POLICIES[policy_name]["algorithm"],
        battery=battery,
        num_episodes=num_episodes,
        max_step=max_step,
        horizon_name=horizon,
        policy_name=policy_name,
        scenario_label=f"{tag}",
        output_dir=output_dir,
    )
    return paths


def assemble(
    *,
    episode_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    from aemo_notebook_utils import _normalize_episode_dataframe

    frames: list[pl.DataFrame] = []
    index: list[dict[str, object]] = []
    for ep_id, path in enumerate(sorted(episode_paths)):
        df = pl.read_parquet(path)
        stem = path.stem
        policy = stem.split("__")[1] if len(stem.split("__")) >= 3 else "unknown"
        normalized, row = _normalize_episode_dataframe(df, source_policy=policy, episode_id=ep_id)
        frames.append(normalized)
        index.append({"episode_id": ep_id, "source_policy": policy, "rows": int(normalized.height), "path": str(path)})

    dataset = pl.concat(frames, how="diagonal_relaxed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(str(output_path))
    manifest = {
        "episode_count": len(index),
        "row_count": int(dataset.height),
        "episode_index": index,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from generate_fcas_dataset import BATTERIES, HORIZONS, POLICIES

    policies = [p.strip() for p in args.policies.split(",")]
    batteries = [b.strip() for b in args.batteries.split(",")]
    horizon = args.horizon
    max_step = HORIZONS[horizon]
    eps_per_cell = 1 if args.smoke else args.episodes_per_cell

    if args.synthetic_frame_path:
        if not args.synthetic_frame_path.is_file():
            raise FileNotFoundError(f"synthetic frame not found: {args.synthetic_frame_path}")
        print(f"Using pre-generated synthetic frame: {args.synthetic_frame_path}")
        gen = None
        train = None
    elif args.train_spec:
        from eval_fcas_generator import _parse_dataset_spec, load_interval

        specs = [_parse_dataset_spec(s) for s in args.train_spec]
        train = pl.concat([load_interval(r, s, e) for r, s, e in specs])
        print(f"Training generator on {len(specs)} slices ({train.height:,} rows):")
        for r, s, e in specs:
            print(f"  {r}:{s}:{e}")
    else:
        print(f"Loading {args.region} {args.train_start}..{args.train_end}")
        train = load_slice(args.region, args.train_start, args.train_end)
        print(f"  train rows: {train.height}")

    if train is not None:
        print(f"Fitting FCASDiffusionGenerator (epochs={args.generator_epochs})...")
        gen = build_generator(train, args)

    print(f"Loading {args.region} {args.generate_start}..{args.generate_end}")
    real = load_slice(args.region, args.generate_start, args.generate_end)
    print(f"  generate rows: {real.height}")

    if args.synthetic_frame_path:
        synth = pl.read_parquet(args.synthetic_frame_path)
        if args.price_mode == "fcas_only":
            synth = synth.with_columns(pl.Series("RRP", real["RRP"].to_numpy()))
        elif args.price_mode == "rrp_only":
            from fcas_generator_eval import FCAS_COLS
            synth = synth.with_columns([pl.Series(col, real[col].to_numpy()) for col in FCAS_COLS])
        print(f"  loaded synthetic frame + spliced price_mode={args.price_mode}")
    else:
        print("Generating synthetic prices over the generate period...")
        synth = synthetic_frame(gen, real, real, args.price_mode)
        print(f"  synthetic frame done (price_mode={args.price_mode})")

    synth_path = output_dir / f"synth_{args.region}_{args.generate_start}_{args.generate_end}.parquet"
    synth.write_parquet(synth_path)
    print(f"  saved synthetic frame -> {synth_path}")

    syn_eps: list[Path] = []
    real_eps: list[Path] = []
    for policy_name in policies:
        algo = POLICIES[policy_name]["algorithm"]
        for battery_name in batteries:
            battery = dict(BATTERIES[battery_name])
            n = eps_per_cell
            print(f"  {policy_name} x {battery_name} x {horizon} ({n} eps) ...")
            syn_paths = run_cell(
                processed_data=synth,
                policy_name=policy_name,
                battery=battery,
                battery_name=battery_name,
                horizon=horizon,
                max_step=max_step,
                num_episodes=n,
                output_dir=output_dir / "raw_logs_synth",
                tag=f"{args.region}",
                seed=args.seed,
            )
            real_paths = run_cell(
                processed_data=real,
                policy_name=policy_name,
                battery=dict(BATTERIES[battery_name]),
                battery_name=battery_name,
                horizon=horizon,
                max_step=max_step,
                num_episodes=n,
                output_dir=output_dir / "raw_logs_real",
                tag=f"{args.region}",
                seed=args.seed,
            )
            syn_eps.extend(syn_paths)
            real_eps.extend(real_paths)

    print("Assembling synthetic dataset...")
    syn_manifest = assemble(
        episode_paths=syn_eps,
        output_path=output_dir / "aemo_fcas_dataset_synth.parquet",
        manifest_path=output_dir / "aemo_fcas_manifest_synth.json",
    )
    print(f"  synthetic: {syn_manifest['episode_count']} episodes, {syn_manifest['row_count']:,} rows")

    print("Assembling real dataset...")
    real_manifest = assemble(
        episode_paths=real_eps,
        output_path=output_dir / "aemo_fcas_dataset_real.parquet",
        manifest_path=output_dir / "aemo_fcas_manifest_real.json",
    )
    print(f"  real: {real_manifest['episode_count']} episodes, {real_manifest['row_count']:,} rows")

    summary = {
        "region": args.region,
        "train": [args.train_start, args.train_end],
        "generate": [args.generate_start, args.generate_end],
        "policies": policies,
        "batteries": batteries,
        "horizon": horizon,
        "max_step": max_step,
        "episodes_per_cell": eps_per_cell,
        "generator_epochs": args.generator_epochs,
        "price_mode": args.price_mode,
        "synthetic": {"path": str(output_dir / "aemo_fcas_dataset_synth.parquet"), **syn_manifest},
        "real": {"path": str(output_dir / "aemo_fcas_dataset_real.parquet"), **real_manifest},
    }
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSummary -> {output_dir / 'generation_summary.json'}")


if __name__ == "__main__":
    main()
