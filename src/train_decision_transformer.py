from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from decision_transformer import DecisionTransformer
from transformer_training import (
    TrajectoryDataset,
    concat_trajectory_datasets,
    train_decision_transformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Decision Transformer on stored trajectory logs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory that holds parquet trajectory logs.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["train_episode_01", "train_episode_02", "test_episode_01"],
        help="Filename substrings to include when collecting parquet logs.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to JSON file with Decision Transformer kwargs.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override context length used when building datasets.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="Override state dimension if not stored in the config.",
    )
    parser.add_argument(
        "--act-dim",
        type=int,
        default=None,
        help="Override action dimension if not stored in the config.",
    )
    parser.add_argument(
        "--max-timestep",
        type=int,
        default=None,
        help="Override max timestep embedding length.",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor passed to TrajectoryDataset.",
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
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
        default="off",
        help="Automatic mixed precision mode: auto (GPU only), on (force on CUDA), off (disable).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Where to store the trained model weights.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint path for intermediate saves.",
    )
    parser.add_argument(
        "--loss-csv-path",
        type=Path,
        default=None,
        help="Optional path to save training/validation loss history as CSV.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--checkpoints-per-epoch", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--return-scale", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.1)
    parser.add_argument("--return-loss-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # DataLoader performance tuning
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers. Increase to improve throughput if CPU can keep up.",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable persistent DataLoader workers.",
    )
    parser.set_defaults(persistent_workers=True)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched per worker (only applies when num-workers > 0).",
    )
    parser.add_argument(
        "--rope-enabled",
        action="store_true",
        help="Enable rotary positional embeddings inside transformer blocks.",
    )
    parser.add_argument(
        "--rope-max-position",
        type=int,
        default=None,
        help="Override the maximum sequence length cached for RoPE (defaults to context length * 3).",
    )
    parser.add_argument(
        "--rope-base",
        type=float,
        default=None,
        help="Override the RoPE frequency base (defaults to 10000.0).",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def collect_parquet_files(data_dir: Path, patterns: Sequence[str]) -> list[Path]:
    matches: list[Path] = []
    for entry in sorted(data_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".parquet":
            continue
        if any(pattern in entry.name for pattern in patterns):
            matches.append(entry)
    return matches


def load_model_kwargs(config_path: Path | None) -> dict:
    if config_path and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def assemble_model_kwargs(args: argparse.Namespace, base_kwargs: dict) -> dict:
    defaults = {
        "state_dim": 12,
        "act_dim": 1,
        "n_block": 2,
        "h_dim": 128,
        "context_len": 60,
        "n_heads": 8,
        "drop_p": 0.1,
        "max_timestep": 17567,
        "rope_enabled": False,
        "rope_max_position": None,
        "rope_base": 10000.0,
    }
    model_kwargs = {**defaults, **base_kwargs}
    if args.state_dim is not None:
        model_kwargs["state_dim"] = args.state_dim
    if args.act_dim is not None:
        model_kwargs["act_dim"] = args.act_dim
    if args.context_length is not None:
        model_kwargs["context_len"] = args.context_length
    if args.max_timestep is not None:
        model_kwargs["max_timestep"] = args.max_timestep
    if args.rope_enabled:
        model_kwargs["rope_enabled"] = True
    if args.rope_max_position is not None:
        model_kwargs["rope_max_position"] = args.rope_max_position
    if args.rope_base is not None:
        model_kwargs["rope_base"] = args.rope_base
    # ensure values are concrete ints where expected
    model_kwargs["state_dim"] = int(model_kwargs["state_dim"])
    model_kwargs["act_dim"] = int(model_kwargs["act_dim"])
    model_kwargs["context_len"] = int(model_kwargs["context_len"])
    model_kwargs["n_block"] = int(model_kwargs["n_block"])
    model_kwargs["h_dim"] = int(model_kwargs["h_dim"])
    model_kwargs["n_heads"] = int(model_kwargs["n_heads"])
    model_kwargs["max_timestep"] = int(model_kwargs.get("max_timestep", model_kwargs["context_len"]))
    model_kwargs["drop_p"] = float(model_kwargs["drop_p"])
    rope_max_pos = model_kwargs.get("rope_max_position")
    if rope_max_pos is None:
        rope_max_pos = 3 * model_kwargs["context_len"]
    model_kwargs["rope_max_position"] = int(rope_max_pos)
    model_kwargs["rope_base"] = float(model_kwargs.get("rope_base", 10000.0))
    model_kwargs["rope_enabled"] = bool(model_kwargs.get("rope_enabled", False))
    return model_kwargs


def main() -> None:
    args = parse_args()
    root = repo_root()

    data_dir = (args.data_dir or (root / "data")).resolve()
    model_dir = (root / "models")
    config_path = (args.model_config or (model_dir / "decision_transformer_model_kwargs.json")).resolve()
    save_path = (args.save_path or (model_dir / "dt_model.pt")).resolve()
    checkpoint_path = (args.checkpoint_path or (model_dir / "dt_model_checkpoint.pt")).resolve()
    loss_csv_path = (args.loss_csv_path or (model_dir / "dt_model_loss_history.csv")).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = collect_parquet_files(data_dir, args.patterns)
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet trajectories matched {args.patterns} in {data_dir}"
        )

    base_kwargs = load_model_kwargs(config_path)
    model_kwargs = assemble_model_kwargs(args, base_kwargs)

    context_length = model_kwargs["context_len"]
    state_dim = model_kwargs["state_dim"]
    act_dim = model_kwargs["act_dim"]

    print("Using model kwargs:")
    print(json.dumps(model_kwargs, indent=2))

    datasets: list[TrajectoryDataset] = []
    for parquet_path in parquet_files:
        print(f"Loading dataset from {parquet_path}")
        ds = TrajectoryDataset(
            data_path=str(parquet_path),
            context_length=context_length,
            state_dim=state_dim,
            act_dim=act_dim,
            discount_factor=args.discount,
        )
        datasets.append(ds)
        print(f"  -> collected {len(ds)} sliding windows")

    combined_dataset = concat_trajectory_datasets(datasets)
    total_samples = len(combined_dataset)
    if total_samples == 0:
        raise RuntimeError("Combined dataset has no samples to train on.")
    print(f"Combined dataset size: {total_samples}")

    val_split = max(0.0, min(1.0, args.val_split))
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    if train_size == 0:
        raise RuntimeError("Validation split too large; no samples left for training.")

    generator = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset,
        [train_size, val_size],
        generator=generator,
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = DecisionTransformer(**model_kwargs)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path_str = str(checkpoint_path)
    else:
        checkpoint_path_str = None

    trained_model, train_losses, val_losses = train_decision_transformer(
        ds=train_dataset,
        model=model,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        save_path=str(save_path),
        checkpoint_path=checkpoint_path_str,
        checkpoint_interval=args.checkpoint_interval,
        checkpoints_per_epoch=args.checkpoints_per_epoch,
        val_ds=val_dataset if val_size > 0 else None,
        resume=args.resume,
        action_loss_weight=args.action_loss_weight,
        state_loss_weight=args.state_loss_weight,
        return_loss_weight=args.return_loss_weight,
        weight_decay=args.weight_decay,
        return_scale=args.return_scale,
        amp_mode=args.amp_mode,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # store training loss and validation loss history in csv
    with open(loss_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses or [])):
            writer.writerow([epoch, train_loss, val_loss])

    print(f"Training finished; final training loss {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final validation loss {val_losses[-1]:.6f}")
    print(f"Trained weights available at {save_path}")


if __name__ == "__main__":
    main()
