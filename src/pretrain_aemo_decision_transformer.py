from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_episode_subsets(
    *,
    df: pl.DataFrame,
    episode_ids: Sequence[int],
    subset_episode_count: int,
    output_dir: Path,
    stem_prefix: str,
) -> list[dict[str, Any]]:
    subsets: list[dict[str, Any]] = []
    for subset_index, start_idx in enumerate(range(0, len(episode_ids), subset_episode_count), start=1):
        subset_episode_ids = [int(episode_id) for episode_id in episode_ids[start_idx : start_idx + subset_episode_count]]
        subset_df = df.filter(pl.col("episode_id").is_in(subset_episode_ids))
        subset_path = output_dir / f"{stem_prefix}_subset_{subset_index:03d}.parquet"
        subset_df.write_parquet(str(subset_path))
        subsets.append(
            {
                "subset_index": subset_index,
                "path": str(subset_path),
                "episode_count": len(subset_episode_ids),
                "row_count": int(subset_df.height),
                "episode_ids": subset_episode_ids,
            }
        )
    return subsets


def _partition_dt_dataset_for_subset_training_fallback(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    subset_episode_count: int,
    val_split: float,
    seed: int,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    if subset_episode_count < 1:
        raise ValueError("subset_episode_count must be at least 1.")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"DT dataset not found: {dataset_path}")

    df = pl.read_parquet(str(dataset_path))
    if df.height == 0:
        raise ValueError("DT dataset is empty; cannot partition into training subsets.")
    if "episode_id" not in df.columns:
        raise ValueError("DT dataset is missing required column 'episode_id'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_ids = sorted(int(episode_id) for episode_id in df["episode_id"].unique().to_list())
    rng = np.random.default_rng(seed)
    shuffled_episode_ids = list(episode_ids)
    rng.shuffle(shuffled_episode_ids)

    n_val = int(len(shuffled_episode_ids) * max(0.0, min(1.0, float(val_split))))
    val_episode_ids = sorted(shuffled_episode_ids[:n_val])
    train_episode_ids = sorted(shuffled_episode_ids[n_val:])
    if not train_episode_ids:
        raise RuntimeError(
            f"val_split={val_split!r} is too large; no episodes remain for subset training."
        )

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_subsets = _write_episode_subsets(
        df=df,
        episode_ids=train_episode_ids,
        subset_episode_count=subset_episode_count,
        output_dir=train_dir,
        stem_prefix=f"{dataset_path.stem}_train",
    )
    val_subsets = _write_episode_subsets(
        df=df,
        episode_ids=val_episode_ids,
        subset_episode_count=subset_episode_count,
        output_dir=val_dir,
        stem_prefix=f"{dataset_path.stem}_val",
    )

    manifest = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "subset_episode_count": int(subset_episode_count),
        "seed": int(seed),
        "requested_val_split": float(val_split),
        "total_episode_count": len(episode_ids),
        "total_row_count": int(df.height),
        "train_episode_count": len(train_episode_ids),
        "val_episode_count": len(val_episode_ids),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "train_subsets": train_subsets,
        "val_subsets": val_subsets,
    }
    manifest_path = output_dir / f"{dataset_path.stem}_subset_training_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def partition_dt_dataset_for_subset_training(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    subset_episode_count: int,
    val_split: float,
    seed: int,
) -> dict[str, Any]:
    try:
        from aemo_notebook_utils import partition_dt_dataset_for_subset_training as notebook_partition
    except ModuleNotFoundError as exc:
        missing_module = exc.name or ""
        print(
            "[WARN] Falling back to the lightweight AEMO subset partitioner because "
            f"aemo_notebook_utils could not be imported ({missing_module})."
        )
        return _partition_dt_dataset_for_subset_training_fallback(
            dataset_path=dataset_path,
            output_dir=output_dir,
            subset_episode_count=subset_episode_count,
            val_split=val_split,
            seed=seed,
        )
    return notebook_partition(
        dataset_path=dataset_path,
        output_dir=output_dir,
        subset_episode_count=subset_episode_count,
        val_split=val_split,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Decision Transformer on AEMO trajectory logs.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=repo_root() / "data" / "aemo_dt" / "aemo_dt_dataset.parquet",
        help="Path to the AEMO DT parquet dataset generated by notebooks/aemo_simrun.ipynb.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=repo_root() / "configs" / "aemo_decision_transformer_model_kwargs.json",
        help="Path to the AEMO DT model kwargs JSON.",
    )
    parser.add_argument(
        "--val-dataset-path",
        type=Path,
        default=None,
        help="Optional explicit validation parquet path to forward to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--surface-preset",
        type=str,
        default="aemo_learning_baseline",
        help=(
            "DT training surface preset forwarded to src/pretrain_decision_transformer.py. "
            "Use aemo_learning_baseline for broader baseline training or aemo_proxy for cheap triage."
        ),
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help="Optional DT model variant forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Optional DT context length forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="Optional DT state dimension forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--act-dim",
        type=int,
        default=None,
        help="Optional DT action dimension forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--n-block",
        type=int,
        default=None,
        help="Optional DT block count forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--h-dim",
        type=int,
        default=None,
        help="Optional DT hidden size forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="Optional DT attention head count forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--drop-p",
        type=float,
        default=None,
        help="Optional DT dropout forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=None,
        help="Optional DT max timestep forwarded to src/pretrain_decision_transformer.py.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=repo_root() / "models" / "aemo" / "dt" / "aemo_dt_model.pt",
        help="Where to store the trained AEMO DT weights.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=repo_root() / "models" / "aemo" / "dt" / "aemo_dt_checkpoint.pt",
        help="Optional checkpoint path for intermediate saves.",
    )
    parser.add_argument(
        "--loss-csv-path",
        type=Path,
        default=repo_root() / "models" / "aemo" / "dt" / "aemo_dt_loss_history.csv",
        help="Optional path to save AEMO DT loss history as CSV.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of samples reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string understood by torch (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--amp-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Automatic mixed precision mode: auto (GPU only), on (force on CUDA), off (disable).",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--checkpoints-per-epoch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--return-scale", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.01)
    parser.add_argument("--return-loss-weight", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.set_defaults(persistent_workers=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--rope-enabled", action="store_true")
    parser.add_argument("--rope-max-position", type=int, default=None)
    parser.add_argument("--rope-base", type=float, default=None)
    parser.add_argument(
        "--train-in-subsets",
        action="store_true",
        help="Split the AEMO parquet into episode-based subset files and train across them sequentially.",
    )
    parser.add_argument(
        "--subset-episodes",
        type=int,
        default=None,
        help="Number of full episodes to include in each generated subset parquet when --train-in-subsets is enabled.",
    )
    parser.add_argument(
        "--subset-output-dir",
        type=Path,
        default=None,
        help="Directory where episode-based AEMO subset parquet files will be written.",
    )
    parser.add_argument(
        "--epochs-per-subset",
        type=int,
        default=None,
        help="Epoch count to use for each subset run. Defaults to --epochs when omitted.",
    )
    return parser.parse_args()


def build_training_command(
    *,
    root: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    epochs: int,
    resume: bool,
    val_data_dir: Path | None = None,
    val_patterns: list[str] | None = None,
) -> list[str]:
    model_config_path = args.model_config.resolve()
    save_path = args.save_path.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    loss_csv_path = args.loss_csv_path.resolve()

    command = [
        sys.executable,
        str(root / "src" / "pretrain_decision_transformer.py"),
        "--surface-preset",
        args.surface_preset,
        "--data-dir",
        str(dataset_path.parent),
        "--patterns",
        dataset_path.stem,
        "--model-config",
        str(model_config_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--val-split",
        str(0.0 if val_data_dir is not None else args.val_split),
        "--seed",
        str(args.seed),
        "--save-path",
        str(save_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--loss-csv-path",
        str(loss_csv_path),
        "--amp-mode",
        args.amp_mode,
        "--return-scale",
        str(args.return_scale),
        "--action-loss-weight",
        str(args.action_loss_weight),
        "--state-loss-weight",
        str(args.state_loss_weight),
        "--return-loss-weight",
        str(args.return_loss_weight),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--checkpoints-per-epoch",
        str(args.checkpoints_per_epoch),
    ]

    if args.context_length is not None:
        command.extend(["--context-length", str(args.context_length)])
    if args.state_dim is not None:
        command.extend(["--state-dim", str(args.state_dim)])
    if args.act_dim is not None:
        command.extend(["--act-dim", str(args.act_dim)])
    if args.n_block is not None:
        command.extend(["--n-block", str(args.n_block)])
    if args.h_dim is not None:
        command.extend(["--h-dim", str(args.h_dim)])
    if args.n_heads is not None:
        command.extend(["--n-heads", str(args.n_heads)])
    if args.drop_p is not None:
        command.extend(["--drop-p", str(args.drop_p)])
    if args.max_timestep is not None:
        command.extend(["--max-timestep", str(args.max_timestep)])

    if val_data_dir is not None:
        command.extend(["--val-data-dir", str(val_data_dir)])
        if val_patterns:
            command.append("--val-patterns")
            command.extend(val_patterns)
    if args.model_variant is not None:
        command.extend(["--model-variant", args.model_variant])

    if resume:
        command.append("--resume")
    if args.device is not None:
        command.extend(["--device", args.device])
    if not args.persistent_workers:
        command.append("--no-persistent-workers")
    if args.rope_enabled:
        command.append("--rope-enabled")
    if args.rope_max_position is not None:
        command.extend(["--rope-max-position", str(args.rope_max_position)])
    if args.rope_base is not None:
        command.extend(["--rope-base", str(args.rope_base)])
    return command


def build_training_commands(
    *,
    root: Path,
    args: argparse.Namespace,
    dataset_paths: list[Path],
    epochs_per_stage: int,
    initial_epoch_offset: int = 0,
    val_dataset_paths: list[Path] | None = None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    val_data_dir: Path | None = None
    val_patterns: list[str] | None = None
    if val_dataset_paths:
        val_data_dir = val_dataset_paths[0].parent
        val_patterns = [path.stem for path in val_dataset_paths]
    for index, dataset_path in enumerate(dataset_paths):
        commands.append(
            build_training_command(
                root=root,
                args=args,
                dataset_path=dataset_path,
                epochs=initial_epoch_offset + (epochs_per_stage * (index + 1)),
                resume=bool(args.resume or index > 0),
                val_data_dir=val_data_dir,
                val_patterns=val_patterns,
            )
        )
    return commands


def get_checkpoint_epoch(checkpoint_path: Path) -> int:
    if not checkpoint_path.is_file():
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return int(checkpoint.get("epoch", 0))


def main() -> None:
    args = parse_args()
    root = repo_root()

    dataset_path = args.dataset_path.resolve()
    model_config_path = args.model_config.resolve()
    val_dataset_path = args.val_dataset_path.resolve() if args.val_dataset_path is not None else None

    if not dataset_path.is_file():
        raise FileNotFoundError(f"AEMO dataset not found: {dataset_path}")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"AEMO model config not found: {model_config_path}")
    if val_dataset_path is not None and not val_dataset_path.is_file():
        raise FileNotFoundError(f"AEMO validation dataset not found: {val_dataset_path}")

    dataset_paths = [dataset_path]
    val_dataset_paths: list[Path] | None = [val_dataset_path] if val_dataset_path is not None else None
    epochs_per_stage = int(args.epochs)
    initial_epoch_offset = get_checkpoint_epoch(args.checkpoint_path.resolve()) if args.resume else 0

    if args.surface_preset == "aemo_learning_baseline":
        if not args.train_in_subsets:
            raise ValueError(
                "--surface-preset=aemo_learning_baseline requires --train-in-subsets so the wrapper can "
                "create explicit held-out validation subsets."
            )
        if args.subset_episodes is None:
            args.subset_episodes = 24

    if args.train_in_subsets:
        if args.subset_episodes is None:
            raise ValueError("--subset-episodes is required when --train-in-subsets is enabled.")
        subset_output_dir = args.subset_output_dir
        if subset_output_dir is None:
            subset_output_dir = dataset_path.parent / f"{dataset_path.stem}_subsets"
        subset_manifest = partition_dt_dataset_for_subset_training(
            dataset_path=dataset_path,
            output_dir=subset_output_dir,
            subset_episode_count=args.subset_episodes,
            val_split=args.val_split,
            seed=args.seed,
        )
        dataset_paths = [Path(entry["path"]) for entry in subset_manifest["train_subsets"]]
        val_dataset_paths = [Path(entry["path"]) for entry in subset_manifest["val_subsets"]]
        epochs_per_stage = int(args.epochs_per_subset or args.epochs)
        print(
            f"Partitioned {dataset_path.name} into {len(dataset_paths)} subset files "
            f"with up to {args.subset_episodes} episodes each."
        )
        print(f"Subset manifest: {subset_manifest['manifest_path']}")
        print(
            f"Global split: train episodes={subset_manifest['train_episode_count']}, "
            f"val episodes={subset_manifest['val_episode_count']}"
        )
    elif args.val_split < 0.0 or args.val_split > 1.0:
        raise ValueError("--val-split must be between 0.0 and 1.0.")

    commands = build_training_commands(
        root=root,
        args=args,
        dataset_paths=dataset_paths,
        epochs_per_stage=epochs_per_stage,
        initial_epoch_offset=initial_epoch_offset,
        val_dataset_paths=val_dataset_paths,
    )
    for subset_index, command in enumerate(commands, start=1):
        if len(commands) > 1:
            print(f"Launching subset {subset_index}/{len(commands)} via:")
        else:
            print("Launching AEMO Decision Transformer training via:")
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
