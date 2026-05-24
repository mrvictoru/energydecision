from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from decision_transformer import DecisionTransformer
from transformer_training import (
    TrajectoryDataset,
    episode_train_val_split,
    train_decision_transformer,
)


EDITABLE_TRAINING_SURFACE_FILE = str(Path(__file__).resolve())
SUPPORTED_MODEL_CONFIG_KEYS = frozenset(
    {
        "state_dim",
        "act_dim",
        "n_block",
        "h_dim",
        "context_len",
        "n_heads",
        "drop_p",
        "max_timestep",
        "rope_enabled",
        "rope_max_position",
        "rope_base",
    }
)
APPROVED_OPTIMIZERS = ("adamw",)
APPROVED_SCHEDULERS = ("steplr",)
ACTION_MODE_TO_ACT_DIM = {"simple": 1, "multi_market": 3}
VALID_AEMO_ACT_DIMS = frozenset(ACTION_MODE_TO_ACT_DIM.values())
AEMO_STATE_DIM = 18
SEARCHABLE_KNOBS = (
    "surface_preset",
    "model_variant",
    "state_dim",
    "act_dim",
    "n_block",
    "h_dim",
    "n_heads",
    "drop_p",
    "context_len",
    "max_timestep",
    "rope_enabled",
    "rope_max_position",
    "rope_base",
    "batch_size",
    "epochs",
    "lr",
    "val_split",
    "discount",
    "return_scale",
    "action_loss_weight",
    "state_loss_weight",
    "return_loss_weight",
    "weight_decay",
    "amp_mode",
    "num_workers",
    "persistent_workers",
    "prefetch_factor",
    "checkpoint_interval",
    "checkpoints_per_epoch",
    "seed",
    "device",
)
TRAINING_KNOB_TO_ARG_DEST = {
    "batch_size": "batch_size",
    "lr": "lr",
    "epochs": "epochs",
    "discount": "discount",
    "val_split": "val_split",
    "seed": "seed",
    "device": "device",
    "amp_mode": "amp_mode",
    "checkpoint_interval": "checkpoint_interval",
    "checkpoints_per_epoch": "checkpoints_per_epoch",
    "resume": "resume",
    "return_scale": "return_scale",
    "action_loss_weight": "action_loss_weight",
    "state_loss_weight": "state_loss_weight",
    "return_loss_weight": "return_loss_weight",
    "weight_decay": "weight_decay",
    "num_workers": "num_workers",
    "persistent_workers": "persistent_workers",
    "prefetch_factor": "prefetch_factor",
    "optimizer": "optimizer",
    "scheduler": "scheduler",
}
FROZEN_INVARIANTS = (
    "editable_training_surface_file",
    "decision_transformer_module=decision_transformer.py",
    "training_engine=transformer_training.py",
    "dataset_schema=episode_id,step,norm_observation,action,reward",
    "evaluation_pipeline_and_env_modules_are_read_only",
    "artifact_contract=model,checkpoint,loss_csv,metadata_sidecars",
    "adapter_contract=pretrain_aemo_decision_transformer.py and aemo_notebook_utils.py",
    "canonical_entrypoint=src/pretrain_decision_transformer.py",
    "optimizer_impl=adamw",
    "scheduler_impl=steplr",
)
SURFACE_CONSTRAINTS = {
    "h_dim_divisible_by_n_heads": "h_dim must be divisible by n_heads",
}
SAFE_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "state_dim": (1, 256),
    "act_dim": (1, 16),
    "n_block": (1, 16),
    "h_dim": (16, 2048),
    "n_heads": (1, 32),
    "drop_p": (0.0, 0.95),
    "context_len": (1, 4096),
    "max_timestep": (1, 2_000_000),
    "rope_max_position": (1, 16_384),
    "rope_base": (1.0, 1_000_000.0),
    "batch_size": (1, 4096),
    "epochs": (1, 100_000),
    "lr": (1e-8, 1.0),
    "val_split": (0.0, 0.95),
    "discount": (0.0, 1.0),
    "return_scale": (1e-12, 1_000_000_000.0),
    "action_loss_weight": (0.0, 1_000_000.0),
    "state_loss_weight": (0.0, 1_000_000.0),
    "return_loss_weight": (0.0, 1_000_000.0),
    "weight_decay": (0.0, 10.0),
    "num_workers": (0, 128),
    "prefetch_factor": (1, 128),
    "checkpoint_interval": (1, 10_000),
    "checkpoints_per_epoch": (0, 10_000),
    "seed": (0, 2**32 - 1),
}


@dataclass(frozen=True)
class SurfacePreset:
    description: str
    model_variant: str = "baseline"
    model_overrides: dict[str, Any] = field(default_factory=dict)
    train_overrides: dict[str, Any] = field(default_factory=dict)
    requires_explicit_validation: bool = False
    min_train_episodes: int | None = None
    disallowed_train_patterns: tuple[str, ...] = ()


MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "compact": {
        "n_block": 2,
        "h_dim": 96,
        "n_heads": 4,
        "drop_p": 0.1,
    },
    "wide": {
        "n_block": 4,
        "h_dim": 256,
        "n_heads": 8,
        "drop_p": 0.1,
    },
    "deeper_wider": {
        "n_block": 8,
        "h_dim": 384,
        "n_heads": 8,
        "drop_p": 0.1,
    },
    "aemo_multimarket": {
        "state_dim": 18,
        "act_dim": 3,
        "n_block": 4,
        "h_dim": 128,
        "context_len": 288,
        "n_heads": 8,
        "drop_p": 0.1,
        "max_timestep": 2016,
        "rope_enabled": True,
    },
}
SURFACE_PRESETS: dict[str, SurfacePreset] = {
    "legacy": SurfacePreset(
        description="Preserve the current DT CLI behavior while centralizing safe experiment selection.",
    ),
    "household_baseline": SurfacePreset(
        description="Household-oriented baseline preset using the legacy model defaults.",
        model_variant="baseline",
    ),
    "aemo_multimarket": SurfacePreset(
        description="AEMO multi-market preset aligned with the shared notebook and wrapper defaults.",
        model_variant="aemo_multimarket",
    ),
    "aemo_proxy": SurfacePreset(
        description=(
            "Fast compact AEMO proxy loop for cheap triage. Use this for quick ranking, not as the "
            "primary learning baseline."
        ),
        model_variant="compact",
        model_overrides={
            "state_dim": 18,
            "act_dim": 3,
            "context_len": 60,
            "max_timestep": 2016,
            "rope_enabled": True,
            "rope_max_position": 180,
        },
        train_overrides={
            "batch_size": 128,
            "epochs": 1,
            "lr": 3e-5,
            "amp_mode": "auto",
            "num_workers": 0,
            "checkpoint_interval": 1,
            "checkpoints_per_epoch": 1,
        },
    ),
    "aemo_proxy_frontier": SurfacePreset(
        description=(
            "Pilot-ranking AEMO proxy preset baked from the current frontier defaults. Use this for the "
            "canonical fixed-split proxy baseline."
        ),
        model_variant="deeper_wider",
        model_overrides={
            "state_dim": 18,
            "act_dim": 3,
            "context_len": 180,
            "max_timestep": 2016,
            "rope_enabled": True,
            "rope_max_position": 540,
        },
        train_overrides={
            "batch_size": 16,
            "epochs": 2,
            "lr": 3e-5,
            "amp_mode": "auto",
            "num_workers": 0,
            "checkpoint_interval": 1,
            "checkpoints_per_epoch": 4,
        },
    ),
    "aemo_learning_baseline": SurfacePreset(
        description=(
            "Broader AEMO learning baseline with explicit validation, longer context, and evaluator-backed "
            "optimizer defaults."
        ),
        model_variant="aemo_multimarket",
        train_overrides={
            "batch_size": 32,
            "epochs": 1,
            "lr": 3e-5,
            "amp_mode": "auto",
            "checkpoint_interval": 1,
            "checkpoints_per_epoch": 1,
        },
        requires_explicit_validation=True,
        min_train_episodes=8,
        disallowed_train_patterns=("aemo_dt_dataset_train_subset_007",),
    ),
    "autoresearch_safe": SurfacePreset(
        description="General-purpose preset for constrained autoresearch over approved knobs.",
        model_variant="baseline",
        train_overrides={"optimizer": "adamw", "scheduler": "steplr"},
    ),
}


@dataclass(frozen=True)
class ResolvedTrainingSurface:
    preset_name: str
    preset_description: str
    model_variant: str
    optimizer: str
    scheduler: str
    split_policy: str
    action_mode: str | None
    model_kwargs: dict[str, Any]
    training_kwargs: dict[str, Any]


def summarize_dataset_shape(dataset: TrajectoryDataset, *, file_count: int) -> dict[str, int]:
    return {
        "file_count": int(file_count),
        "episode_count": int(len(dataset.episodes)),
        "window_count": int(len(dataset)),
    }


def _best_loss_value(values: Sequence[float]) -> float | None:
    candidates = [float(value) for value in values if np.isfinite(value)]
    return min(candidates) if candidates else None


def recommend_pilot_ranking(
    *,
    surface_preset: str,
    best_val_total_loss: float | None,
    best_val_action_loss: float | None,
) -> dict[str, Any]:
    if surface_preset == "aemo_proxy" and best_val_action_loss is not None:
        return {
            "pilot_ranking_metric": "best_val_action_loss",
            "pilot_ranking_value": best_val_action_loss,
            "pilot_ranking_guardrail_metric": "best_val_total_loss",
            "pilot_ranking_guardrail_value": best_val_total_loss,
        }
    return {
        "pilot_ranking_metric": "best_val_total_loss",
        "pilot_ranking_value": best_val_total_loss,
        "pilot_ranking_guardrail_metric": None,
        "pilot_ranking_guardrail_value": None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        "--val-data-dir",
        type=Path,
        default=None,
        help="Optional directory that holds parquet trajectory logs reserved for validation.",
    )
    parser.add_argument(
        "--val-patterns",
        nargs="+",
        default=None,
        help="Optional filename substrings to include when collecting validation parquet logs.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Path to JSON file with Decision Transformer kwargs.",
    )
    parser.add_argument(
        "--surface-preset",
        choices=sorted(SURFACE_PRESETS),
        default="legacy",
        help="Approved experiment preset exposed through the editable DT training surface.",
    )
    parser.add_argument(
        "--model-variant",
        choices=sorted(MODEL_VARIANTS),
        default=None,
        help="Approved architecture variant selected by the editable DT training surface.",
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(APPROVED_OPTIMIZERS),
        default="adamw",
        help="Approved optimizer selection exposed by the training surface.",
    )
    parser.add_argument(
        "--scheduler",
        choices=sorted(APPROVED_SCHEDULERS),
        default="steplr",
        help="Approved scheduler selection exposed by the training surface.",
    )
    parser.add_argument(
        "--split-policy",
        choices=["auto", "episode", "explicit_validation"],
        default="auto",
        help="Dataset split policy. Defaults to episode splitting unless explicit validation files are provided.",
    )
    parser.add_argument(
        "--action-mode",
        choices=sorted(ACTION_MODE_TO_ACT_DIM),
        default=None,
        help="Optional action mode guard for AEMO-style datasets.",
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
        "--n-block",
        type=int,
        default=None,
        help="Override the number of transformer blocks.",
    )
    parser.add_argument(
        "--h-dim",
        type=int,
        default=None,
        help="Override transformer hidden size.",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=None,
        help="Override the number of attention heads.",
    )
    parser.add_argument(
        "--drop-p",
        type=float,
        default=None,
        help="Override dropout probability.",
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
    parser.add_argument(
        "--progress-snapshot-path",
        type=Path,
        default=None,
        help="Optional path for a live JSON progress snapshot that the progress runner can watch.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--checkpoints-per-epoch", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--return-scale", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.01)
    parser.add_argument("--return-loss-weight", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    namespace = parser.parse_args(argv)
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit_cli_args: set[str] = set()
    for token in raw_argv:
        if not token.startswith("-") or token == "-":
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit_cli_args.add(dest)
    namespace.explicit_cli_args = explicit_cli_args
    return namespace


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


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        output_path.write_text(serialized, encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Failed to write JSON payload to {output_path}: {exc}") from exc


def validate_supported_model_kwargs(model_kwargs: dict[str, Any], *, source: str) -> None:
    unknown_keys = sorted(set(model_kwargs) - SUPPORTED_MODEL_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unsupported model config keys in {source}: {unknown_keys}. "
            f"Allowed keys: {sorted(SUPPORTED_MODEL_CONFIG_KEYS)}"
        )


def load_model_kwargs(config_path: Path | None) -> dict[str, Any]:
    if config_path and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        validate_supported_model_kwargs(loaded, source=str(config_path))
        return loaded
    return {}


def validate_numeric_range(name: str, value: float | int | None) -> None:
    if value is None or name not in SAFE_NUMERIC_RANGES:
        return
    lower, upper = SAFE_NUMERIC_RANGES[name]
    if not (lower <= float(value) <= upper):
        raise ValueError(f"{name} must be between {lower} and {upper}, received {value}.")


def validate_surface_constraints(model_kwargs: dict[str, Any]) -> None:
    if model_kwargs["h_dim"] % model_kwargs["n_heads"] != 0:
        raise ValueError(
            f"{SURFACE_CONSTRAINTS['h_dim_divisible_by_n_heads']}: "
            f"h_dim={model_kwargs['h_dim']}, n_heads={model_kwargs['n_heads']}."
        )


def resolve_model_variant(args: argparse.Namespace) -> str:
    preset = SURFACE_PRESETS[args.surface_preset]
    return args.model_variant or preset.model_variant


def resolve_split_policy(args: argparse.Namespace) -> str:
    if args.split_policy == "auto":
        return "explicit_validation" if args.val_data_dir is not None else "episode"
    if args.split_policy == "explicit_validation" and args.val_data_dir is None:
        raise ValueError("--split-policy=explicit_validation requires --val-data-dir.")
    if args.split_policy == "episode" and args.val_data_dir is not None:
        raise ValueError(
            "--split-policy=episode cannot be combined with --val-data-dir; "
            "use --split-policy=explicit_validation or omit the validation directory."
        )
    return args.split_policy


def assemble_model_kwargs(args: argparse.Namespace, base_kwargs: dict[str, Any]) -> dict[str, Any]:
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
    preset = SURFACE_PRESETS[args.surface_preset]
    variant_name = resolve_model_variant(args)
    model_kwargs = {
        **defaults,
        **base_kwargs,
        **MODEL_VARIANTS[variant_name],
        **preset.model_overrides,
    }
    cli_overrides = {
        "state_dim": args.state_dim,
        "act_dim": args.act_dim,
        "context_len": args.context_length,
        "n_block": args.n_block,
        "h_dim": args.h_dim,
        "n_heads": args.n_heads,
        "drop_p": args.drop_p,
        "max_timestep": args.max_timestep,
        "rope_max_position": args.rope_max_position,
        "rope_base": args.rope_base,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            model_kwargs[key] = value
    if args.rope_enabled:
        model_kwargs["rope_enabled"] = True

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
    rope_max_pos = int(rope_max_pos)
    min_rope_max_pos = 3 * model_kwargs["context_len"]
    if bool(model_kwargs.get("rope_enabled", False)) and rope_max_pos < min_rope_max_pos:
        if args.rope_max_position is not None:
            raise ValueError(
                "rope_max_position is too small for the requested context length: "
                f"received rope_max_position={rope_max_pos}, but need at least {min_rope_max_pos} "
                f"for context_len={model_kwargs['context_len']}."
            )
        rope_max_pos = min_rope_max_pos
    model_kwargs["rope_max_position"] = rope_max_pos
    model_kwargs["rope_base"] = float(model_kwargs.get("rope_base", 10000.0))
    model_kwargs["rope_enabled"] = bool(model_kwargs.get("rope_enabled", False))

    for key in (
        "state_dim",
        "act_dim",
        "n_block",
        "h_dim",
        "n_heads",
        "context_len",
        "max_timestep",
        "rope_max_position",
        "rope_base",
        "drop_p",
    ):
        validate_numeric_range(key, model_kwargs[key])
    validate_surface_constraints(model_kwargs)
    return model_kwargs


def assemble_training_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    preset = SURFACE_PRESETS[args.surface_preset]
    explicit_cli_args = set(getattr(args, "explicit_cli_args", set()))
    training_kwargs = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "discount": args.discount,
        "val_split": args.val_split,
        "seed": args.seed,
        "device": args.device,
        "amp_mode": args.amp_mode,
        "checkpoint_interval": args.checkpoint_interval,
        "checkpoints_per_epoch": args.checkpoints_per_epoch,
        "resume": args.resume,
        "return_scale": args.return_scale,
        "action_loss_weight": args.action_loss_weight,
        "state_loss_weight": args.state_loss_weight,
        "return_loss_weight": args.return_loss_weight,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "persistent_workers": args.persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
    }
    for key, value in preset.train_overrides.items():
        arg_dest = TRAINING_KNOB_TO_ARG_DEST.get(key)
        if arg_dest is not None and arg_dest in explicit_cli_args:
            continue
        training_kwargs[key] = value
    for key, value in training_kwargs.items():
        validate_numeric_range(key, value)
    return training_kwargs


def resolve_training_surface(
    args: argparse.Namespace,
    *,
    base_kwargs: dict[str, Any],
) -> ResolvedTrainingSurface:
    split_policy = resolve_split_policy(args)
    model_kwargs = assemble_model_kwargs(args, base_kwargs)
    training_kwargs = assemble_training_kwargs(args)
    action_mode = args.action_mode
    if action_mode is not None:
        expected_act_dim = ACTION_MODE_TO_ACT_DIM[action_mode]
        if model_kwargs["act_dim"] != expected_act_dim:
            raise ValueError(
                f"action_mode={action_mode!r} requires act_dim={expected_act_dim}, "
                f"received act_dim={model_kwargs['act_dim']}."
            )
    if model_kwargs["state_dim"] == AEMO_STATE_DIM and model_kwargs["act_dim"] not in VALID_AEMO_ACT_DIMS:
        raise ValueError(
            "AEMO DT configs must use act_dim=1 for simple mode or act_dim=3 for multi_market mode."
        )
    preset = SURFACE_PRESETS[args.surface_preset]
    return ResolvedTrainingSurface(
        preset_name=args.surface_preset,
        preset_description=preset.description,
        model_variant=resolve_model_variant(args),
        optimizer=training_kwargs["optimizer"],
        scheduler=training_kwargs["scheduler"],
        split_policy=split_policy,
        action_mode=action_mode,
        model_kwargs=model_kwargs,
        training_kwargs=training_kwargs,
    )


def validate_preset_dataset_policy(
    *,
    surface: ResolvedTrainingSurface,
    parquet_files: Sequence[Path],
    train_episode_count: int,
) -> None:
    preset = SURFACE_PRESETS[surface.preset_name]
    if preset.requires_explicit_validation and surface.split_policy != "explicit_validation":
        raise ValueError(
            f"surface_preset={surface.preset_name!r} requires explicit validation parquet files. "
            "Pass --val-data-dir/--val-patterns or choose a proxy preset for cheap triage."
        )
    if (
        preset.disallowed_train_patterns
        and len(parquet_files) == 1
        and any(pattern in parquet_files[0].name for pattern in preset.disallowed_train_patterns)
    ):
        raise ValueError(
            f"surface_preset={surface.preset_name!r} cannot use the narrow proxy slice "
            f"{parquet_files[0].name!r} as its training baseline."
        )
    if preset.min_train_episodes is not None and train_episode_count < preset.min_train_episodes:
        raise ValueError(
            f"surface_preset={surface.preset_name!r} requires at least {preset.min_train_episodes} "
            f"training episodes, received {train_episode_count}."
        )


def ensure_safe_output_path(root: Path, path: Path, *, name: str) -> Path:
    canonical_root = root.resolve(strict=True)
    resolved = path.resolve()
    resolved_parent = resolved.parent.resolve(strict=True)
    resolved = resolved_parent / resolved.name
    try:
        resolved.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError(
            f"{name} must remain inside the repository root for the editable training surface: {resolved}"
        ) from exc
    return resolved


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_trajectory_datasets(
    *,
    parquet_files: Sequence[Path],
    context_length: int,
    state_dim: int,
    act_dim: int,
    discount: float,
) -> list[TrajectoryDataset]:
    datasets: list[TrajectoryDataset] = []
    for parquet_path in parquet_files:
        print(f"Loading dataset from {parquet_path}")
        ds = TrajectoryDataset(
            data_path=str(parquet_path),
            context_length=context_length,
            state_dim=state_dim,
            act_dim=act_dim,
            discount_factor=discount,
        )
        datasets.append(ds)
        print(f"  -> collected {len(ds)} sliding windows")
    return datasets


def merge_trajectory_datasets(datasets: list[TrajectoryDataset]) -> TrajectoryDataset:
    if not datasets:
        raise ValueError("datasets must be non-empty")
    first = datasets[0]
    merged_episodes: list[dict[str, Any]] = []
    for ds in datasets:
        merged_episodes.extend(ds.episodes)
    if not merged_episodes:
        raise ValueError("No episodes found in the provided datasets.")
    return TrajectoryDataset._from_episodes(
        merged_episodes,
        first.context_length,
        first.state_dim,
        first.act_dim,
        first.gamma,
    )


def dataset_dimensions(datasets: Sequence[TrajectoryDataset]) -> tuple[list[int], list[int]]:
    state_dims: set[int] = set()
    act_dims: set[int] = set()
    for ds in datasets:
        state_dims.add(int(ds.state_dim))
        act_dims.add(int(ds.act_dim))
        for episode in ds.episodes:
            states = np.asarray(episode["states"])
            actions = np.asarray(episode["actions"])
            if states.ndim != 2:
                raise ValueError("Each episode must contain 2D state arrays.")
            if actions.ndim != 2:
                raise ValueError("Each episode must contain 2D action arrays.")
            state_dims.add(int(states.shape[1]))
            act_dims.add(int(actions.shape[1]))
    return sorted(state_dims), sorted(act_dims)


def validate_dataset_dimensions(
    *,
    datasets: Sequence[TrajectoryDataset],
    expected_state_dim: int,
    expected_act_dim: int,
    action_mode: str | None,
    label: str,
) -> None:
    if not datasets:
        raise ValueError(f"{label} datasets must be non-empty.")
    state_dims, act_dims = dataset_dimensions(datasets)
    if state_dims != [expected_state_dim]:
        raise ValueError(
            f"{label} datasets must expose state_dim={expected_state_dim}, found {state_dims}."
        )
    if act_dims != [expected_act_dim]:
        raise ValueError(
            f"{label} datasets must expose act_dim={expected_act_dim}, found {act_dims}."
        )
    if action_mode is not None:
        expected_action_dim = ACTION_MODE_TO_ACT_DIM[action_mode]
        if expected_act_dim != expected_action_dim:
            raise ValueError(
                f"{label} datasets are incompatible with action_mode={action_mode!r}: "
                f"expected act_dim={expected_action_dim}, found {expected_act_dim}."
            )
    if expected_state_dim == AEMO_STATE_DIM and expected_act_dim not in VALID_AEMO_ACT_DIMS:
        raise ValueError(
            f"{label} datasets look AEMO-shaped but expose unsupported act_dim={expected_act_dim}."
        )


def build_surface_manifest(
    *,
    root: Path,
    args: argparse.Namespace,
    surface: ResolvedTrainingSurface,
    data_dir: Path,
    parquet_files: Sequence[Path],
    save_path: Path,
    checkpoint_path: Path,
    loss_csv_path: Path,
    progress_snapshot_path: Path,
    val_data_dir: Path | None,
    train_dataset: TrajectoryDataset,
    val_dataset: TrajectoryDataset,
    val_parquet_files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "energydecision.dt_training_surface.v1",
        "editable_training_surface_file": EDITABLE_TRAINING_SURFACE_FILE,
        "surface_preset": surface.preset_name,
        "surface_preset_description": surface.preset_description,
        "model_variant": surface.model_variant,
        "optimizer": surface.optimizer,
        "scheduler": surface.scheduler,
        "split_policy": surface.split_policy,
        "action_mode": surface.action_mode,
        "searchable_knobs": list(SEARCHABLE_KNOBS),
        "frozen_invariants": list(FROZEN_INVARIANTS),
        "surface_constraints": SURFACE_CONSTRAINTS,
        "model_kwargs": surface.model_kwargs,
        "training_kwargs": surface.training_kwargs,
        "paths": {
            "repo_root": str(root),
            "data_dir": str(data_dir),
            "save_path": str(save_path),
            "checkpoint_path": str(checkpoint_path),
            "loss_csv_path": str(loss_csv_path),
            "progress_snapshot_path": str(progress_snapshot_path),
            "val_data_dir": str(val_data_dir) if val_data_dir is not None else None,
        },
        "datasets": {
            "train_files": [str(path) for path in parquet_files],
            "val_files": [str(path) for path in (val_parquet_files or [])],
            "patterns": list(args.patterns),
            "val_patterns": list(args.val_patterns) if args.val_patterns is not None else None,
        },
        "dataset_summary": {
            "train": summarize_dataset_shape(train_dataset, file_count=len(parquet_files)),
            "val": summarize_dataset_shape(val_dataset, file_count=len(val_parquet_files or [])),
        },
        "canonical_command": [
            sys.executable,
            EDITABLE_TRAINING_SURFACE_FILE,
            "--surface-preset",
            surface.preset_name,
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root().resolve()

    data_dir = (args.data_dir or (root / "data" / "household" / "logs")).resolve()
    model_dir = root / "models" / "household" / "dt"
    config_path = (args.model_config or (model_dir / "decision_transformer_model_kwargs.json")).resolve()
    save_path = ensure_safe_output_path(
        root,
        (args.save_path or (model_dir / "dt_model.pt")),
        name="save_path",
    )
    checkpoint_path = ensure_safe_output_path(
        root,
        (args.checkpoint_path or (model_dir / "dt_model_checkpoint.pt")),
        name="checkpoint_path",
    )
    loss_csv_path = ensure_safe_output_path(
        root,
        (args.loss_csv_path or (model_dir / "dt_model_loss_history.csv")),
        name="loss_csv_path",
    )
    progress_snapshot_path = ensure_safe_output_path(
        root,
        (
            args.progress_snapshot_path
            or loss_csv_path.with_name(loss_csv_path.stem + "_progress.json")
        ),
        name="progress_snapshot_path",
    )
    surface_manifest_path = loss_csv_path.with_name(loss_csv_path.stem + "_surface_manifest.json")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = collect_parquet_files(data_dir, args.patterns)
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet trajectories matched {args.patterns} in {data_dir}"
        )

    base_kwargs = load_model_kwargs(config_path)
    surface = resolve_training_surface(args, base_kwargs=base_kwargs)
    model_kwargs = surface.model_kwargs

    print("Using resolved DT training surface:")
    print(
        json.dumps(
            {
                "editable_training_surface_file": EDITABLE_TRAINING_SURFACE_FILE,
                "surface_preset": surface.preset_name,
                "surface_preset_description": surface.preset_description,
                "model_variant": surface.model_variant,
                "optimizer": surface.optimizer,
                "scheduler": surface.scheduler,
                "split_policy": surface.split_policy,
                "action_mode": surface.action_mode,
                "model_kwargs": model_kwargs,
                "training_kwargs": surface.training_kwargs,
                "surface_constraints": SURFACE_CONSTRAINTS,
            },
            indent=2,
        )
    )

    context_length = model_kwargs["context_len"]
    state_dim = model_kwargs["state_dim"]
    act_dim = model_kwargs["act_dim"]

    datasets = load_trajectory_datasets(
        parquet_files=parquet_files,
        context_length=context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        discount=surface.training_kwargs["discount"],
    )
    validate_dataset_dimensions(
        datasets=datasets,
        expected_state_dim=state_dim,
        expected_act_dim=act_dim,
        action_mode=surface.action_mode,
        label="Training",
    )

    total_episodes = sum(len(ds.episodes) for ds in datasets)
    if total_episodes == 0:
        raise RuntimeError("Combined dataset has no episodes to train on.")
    print(f"Total episodes across all files: {total_episodes}")

    seed_everything(surface.training_kwargs["seed"])

    val_parquet_files: list[Path] | None = None
    val_data_dir: Path | None = None
    if surface.split_policy == "explicit_validation":
        val_data_dir = args.val_data_dir.resolve() if args.val_data_dir is not None else None
        if val_data_dir is None or not val_data_dir.is_dir():
            raise FileNotFoundError(f"Validation data directory not found: {val_data_dir}")
        val_patterns = args.val_patterns or args.patterns
        val_parquet_files = collect_parquet_files(val_data_dir, val_patterns)
        if not val_parquet_files:
            raise FileNotFoundError(
                f"No validation parquet trajectories matched {val_patterns} in {val_data_dir}"
            )
        val_datasets = load_trajectory_datasets(
            parquet_files=val_parquet_files,
            context_length=context_length,
            state_dim=state_dim,
            act_dim=act_dim,
            discount=surface.training_kwargs["discount"],
        )
        validate_dataset_dimensions(
            datasets=val_datasets,
            expected_state_dim=state_dim,
            expected_act_dim=act_dim,
            action_mode=surface.action_mode,
            label="Validation",
        )
        train_dataset = merge_trajectory_datasets(datasets)
        val_dataset = merge_trajectory_datasets(val_datasets)
        print("Using explicit validation parquet files; bypassing episode partitioning.")
    else:
        train_dataset, val_dataset = episode_train_val_split(
            datasets,
            val_split=surface.training_kwargs["val_split"],
            seed=surface.training_kwargs["seed"],
        )
    validate_preset_dataset_policy(
        surface=surface,
        parquet_files=parquet_files,
        train_episode_count=len(train_dataset.episodes),
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    surface_manifest = build_surface_manifest(
        root=root,
        args=args,
        surface=surface,
        data_dir=data_dir,
        parquet_files=parquet_files,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        loss_csv_path=loss_csv_path,
        progress_snapshot_path=progress_snapshot_path,
        val_data_dir=val_data_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_parquet_files=val_parquet_files,
    )
    write_json(surface_manifest_path, surface_manifest)

    model = DecisionTransformer(**model_kwargs)
    started_at = time.monotonic()
    _, train_losses, val_losses, history = train_decision_transformer(
        ds=train_dataset,
        model=model,
        batch_size=surface.training_kwargs["batch_size"],
        lr=surface.training_kwargs["lr"],
        epochs=surface.training_kwargs["epochs"],
        device=surface.training_kwargs["device"],
        save_path=str(save_path),
        checkpoint_path=str(checkpoint_path),
        checkpoint_interval=surface.training_kwargs["checkpoint_interval"],
        checkpoints_per_epoch=surface.training_kwargs["checkpoints_per_epoch"],
        val_ds=val_dataset if len(val_dataset) > 0 else None,
        resume=surface.training_kwargs["resume"],
        action_loss_weight=surface.training_kwargs["action_loss_weight"],
        state_loss_weight=surface.training_kwargs["state_loss_weight"],
        return_loss_weight=surface.training_kwargs["return_loss_weight"],
        weight_decay=surface.training_kwargs["weight_decay"],
        return_scale=surface.training_kwargs["return_scale"],
        amp_mode=surface.training_kwargs["amp_mode"],
        num_workers=surface.training_kwargs["num_workers"],
        persistent_workers=surface.training_kwargs["persistent_workers"],
        prefetch_factor=surface.training_kwargs["prefetch_factor"],
        progress_snapshot_path=str(progress_snapshot_path),
        return_history=True,
    )

    with open(loss_csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_total",
                "train_action",
                "train_state",
                "train_return",
                "val_total",
                "val_action",
                "val_state",
                "val_return",
            ]
        )
        train_action = history.get("train_action_losses", [])
        train_state = history.get("train_state_losses", [])
        train_return = history.get("train_return_losses", [])
        val_action = history.get("val_action_losses", [])
        val_state = history.get("val_state_losses", [])
        val_return = history.get("val_return_losses", [])

        max_epochs = len(train_losses)
        for idx in range(max_epochs):
            epoch_num = idx + 1
            row = [
                epoch_num,
                train_losses[idx],
                train_action[idx] if idx < len(train_action) else "",
                train_state[idx] if idx < len(train_state) else "",
                train_return[idx] if idx < len(train_return) else "",
                val_losses[idx] if idx < len(val_losses) else "",
                val_action[idx] if idx < len(val_action) else "",
                val_state[idx] if idx < len(val_state) else "",
                val_return[idx] if idx < len(val_return) else "",
            ]
            writer.writerow(row)

    checkpoints_csv = loss_csv_path.with_name(loss_csv_path.stem + "_checkpoints" + loss_csv_path.suffix)
    with open(checkpoints_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "epoch",
                "segment",
                "batch_idx",
                "train_total_avg",
                "train_total_ema",
                "train_action_avg",
                "train_state_avg",
                "train_return_avg",
                "train_valid",
                "val_total",
                "val_action",
                "val_state",
                "val_return",
                "val_valid",
            ]
        )
        for snap in history.get("loss_history", []):
            writer.writerow(
                [
                    snap.get("timestamp", ""),
                    snap.get("epoch", ""),
                    snap.get("segment", ""),
                    snap.get("batch_idx", ""),
                    snap.get("train_total_avg", ""),
                    snap.get("train_total_ema", ""),
                    snap.get("train_action_avg", ""),
                    snap.get("train_state_avg", ""),
                    snap.get("train_return_avg", ""),
                    snap.get("train_valid", ""),
                    snap.get("val_total", ""),
                    snap.get("val_action", ""),
                    snap.get("val_state", ""),
                    snap.get("val_return", ""),
                    snap.get("val_valid", ""),
                ]
            )

    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    total_windows_processed = int(len(train_dataset) * len(train_losses))
    final_val_total_loss = float(val_losses[-1]) if val_losses else None
    final_val_action_loss = float(val_action[-1]) if val_action else None
    final_val_state_loss = float(val_state[-1]) if val_state else None
    final_val_return_loss = float(val_return[-1]) if val_return else None
    best_val_total_loss = _best_loss_value(val_losses)
    best_val_action_loss = _best_loss_value(val_action)
    best_val_state_loss = _best_loss_value(val_state)
    best_val_return_loss = _best_loss_value(val_return)
    pilot_ranking = recommend_pilot_ranking(
        surface_preset=surface.preset_name,
        best_val_total_loss=best_val_total_loss,
        best_val_action_loss=best_val_action_loss,
    )
    surface_manifest["run_summary"] = {
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_count": int(len(history.get("loss_history", []))),
        "total_windows_processed": total_windows_processed,
        "effective_windows_per_second": (
            total_windows_processed / elapsed_seconds if elapsed_seconds > 0 else None
        ),
        "final_train_total_loss": float(train_losses[-1]),
        "final_val_total_loss": final_val_total_loss,
        "final_val_action_loss": final_val_action_loss,
        "final_val_state_loss": final_val_state_loss,
        "final_val_return_loss": final_val_return_loss,
        "best_val_total_loss": best_val_total_loss,
        "best_val_action_loss": best_val_action_loss,
        "best_val_state_loss": best_val_state_loss,
        "best_val_return_loss": best_val_return_loss,
        **pilot_ranking,
    }
    write_json(surface_manifest_path, surface_manifest)

    print(f"Training finished; final train total loss {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final validation total loss {val_losses[-1]:.6f}")
    print(f"Trained weights available at {save_path}")
    print(f"Loss history written to {loss_csv_path} and {checkpoints_csv}")
    print(f"Training surface manifest written to {surface_manifest_path}")


if __name__ == "__main__":
    main()
