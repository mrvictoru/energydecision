from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


DISTROBOX_REEXEC_ENV = "ENERGYDECISION_DISTROBOX_REEXEC"
DISTROBOX_SENTINEL_ENV_KEYS = (
    "DISTROBOX_ENTER_PATH",
    "DISTROBOX_PATH",
    "DISTROBOX_ID",
    "DISTROBOX_NAME",
    "CONTAINER_ID",
    "container",
)


@dataclass(frozen=True)
class RunTier:
    slug: str
    description: str
    surface_preset: str
    dataset_path: str
    val_dataset_path: str | None
    batch_size: int
    epochs: int
    lr: float
    checkpoints_per_epoch: int
    num_workers: int
    train_in_subsets: bool = False
    subset_episodes: int | None = None
    epochs_per_subset: int | None = None
    val_split: float = 0.1
    preferred_distrobox: str = "energydecision-gpu"
    require_cuda: bool = False
    model_variant: str | None = None


RUN_TIERS: dict[str, RunTier] = {
    "proxy-smoke": RunTier(
        slug="proxy_smoke",
        description="Fastest fixed-split smoke loop for checking the end-to-end training harness.",
        surface_preset="aemo_proxy",
        dataset_path="data/aemo_dt_fcas/autoresearch_pilot/aemo_dt_train_pilot.parquet",
        val_dataset_path="data/aemo_dt_fcas/autoresearch_pilot/aemo_dt_val_pilot.parquet",
        batch_size=32,
        epochs=1,
        lr=3e-5,
        checkpoints_per_epoch=2,
        num_workers=0,
        model_variant="compact",
    ),
    "proxy-baseline": RunTier(
        slug="proxy_baseline",
        description="Canonical fixed-split proxy baseline using the baked-in frontier AEMO DT defaults.",
        surface_preset="aemo_proxy_frontier",
        dataset_path="data/aemo_dt_fcas/autoresearch_pilot/aemo_dt_train_pilot.parquet",
        val_dataset_path="data/aemo_dt_fcas/autoresearch_pilot/aemo_dt_val_pilot.parquet",
        batch_size=16,
        epochs=2,
        lr=3e-5,
        checkpoints_per_epoch=4,
        num_workers=0,
    ),
    "learning-baseline": RunTier(
        slug="learning_baseline",
        description="Broader subset-based AEMO learning baseline for promotion checks.",
        surface_preset="aemo_learning_baseline",
        dataset_path="data/aemo_dt/aemo_dt_dataset.parquet",
        val_dataset_path=None,
        batch_size=16,
        epochs=2,
        lr=3e-5,
        checkpoints_per_epoch=4,
        num_workers=0,
        train_in_subsets=True,
        subset_episodes=24,
        epochs_per_subset=2,
        val_split=0.1,
        require_cuda=True,
        model_variant="deeper_wider",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_default_run_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch robust AEMO DT training runs with tier defaults, runtime checks, and a live tracker.",
    )
    parser.add_argument(
        "--run-tier",
        choices=sorted(RUN_TIERS),
        default="proxy-baseline",
        help="Canonical training tier that derives safe defaults and artifact layout.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=build_default_run_tag(),
        help="Artifact tag used under models/aemo/dt/<run-tag>/.",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=["auto", "require-distrobox", "allow-host"],
        default="auto",
        help="auto re-enters the preferred Distrobox when available; require-distrobox fails if that is impossible.",
    )
    parser.add_argument(
        "--distrobox-name",
        type=str,
        default=None,
        help="Override the preferred Distrobox name for the selected run tier.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional training parquet override. Defaults come from the selected run tier.",
    )
    parser.add_argument(
        "--val-dataset-path",
        type=Path,
        default=None,
        help="Optional validation parquet override for fixed explicit-validation tiers.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=repo_root() / "configs" / "aemo_decision_transformer_model_kwargs.json",
        help="Path to the AEMO DT model kwargs JSON.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoints-per-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--subset-episodes", type=int, default=None)
    parser.add_argument("--epochs-per-subset", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--act-dim", type=int, default=None)
    parser.add_argument("--n-block", type=int, default=None)
    parser.add_argument("--h-dim", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--drop-p", type=float, default=None)
    parser.add_argument("--max-timestep", type=int, default=None)
    parser.add_argument("--rope-enabled", action="store_true")
    parser.add_argument("--rope-max-position", type=int, default=None)
    parser.add_argument("--rope-base", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--optimizer-class-path", type=str, default=None)
    parser.add_argument("--optimizer-kwargs-json", type=str, default=None)
    parser.add_argument("--scheduler-class-path", type=str, default=None)
    parser.add_argument("--scheduler-kwargs-json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, help="Optional explicit torch device string.")
    parser.add_argument(
        "--amp-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="AMP mode forwarded to the trainer.",
    )
    parser.add_argument(
        "--tracker-ui",
        choices=["auto", "plain", "rich"],
        default="auto",
        help="UI mode forwarded to src/dt_progress_runner.py.",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        help="Optional model variant override forwarded to the AEMO wrapper.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the launch plan and print commands without starting training.",
    )
    return parser.parse_args(argv)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def detect_runtime() -> dict[str, Any]:
    env_hits = {
        key: value
        for key in DISTROBOX_SENTINEL_ENV_KEYS
        if (value := os.environ.get(key))
    }
    container_markers = {
        "containerenv": Path("/run/.containerenv").exists(),
        "dockerenv": Path("/.dockerenv").exists(),
    }
    return {
        "inside_container": bool(env_hits)
        or any(container_markers.values())
        or os.environ.get(DISTROBOX_REEXEC_ENV) == "1",
        "env": env_hits,
        "container_markers": container_markers,
    }


def strip_runtime_args(argv: Sequence[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--runtime-mode":
            skip_next = True
            continue
        if token.startswith("--runtime-mode="):
            continue
        if token == "--distrobox-name":
            skip_next = True
            continue
        if token.startswith("--distrobox-name="):
            continue
        stripped.append(token)
    return stripped


def has_cli_option(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def maybe_reenter_distrobox(args: argparse.Namespace, argv: Sequence[str]) -> int | None:
    runtime = detect_runtime()
    if args.runtime_mode == "allow-host" or runtime["inside_container"]:
        return None

    distrobox_name = args.distrobox_name or RUN_TIERS[args.run_tier].preferred_distrobox
    distrobox_binary = shutil.which("distrobox")
    if distrobox_binary is None:
        if args.runtime_mode == "require-distrobox":
            raise RuntimeError(
                "distrobox is not available on PATH, so this run cannot switch into the recommended "
                "container runtime."
            )
        return None

    forwarded = strip_runtime_args(argv)
    if not has_cli_option(forwarded, "--run-tag"):
        forwarded.extend(["--run-tag", args.run_tag])
    command = [
        distrobox_binary,
        "enter",
        distrobox_name,
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        *forwarded,
        "--runtime-mode",
        "allow-host",
    ]
    print(
        f"[launch_aemo_training] Re-entering Distrobox {distrobox_name!r} for the training run.",
        file=sys.stderr,
    )
    child_env = os.environ.copy()
    child_env[DISTROBOX_REEXEC_ENV] = "1"
    result = subprocess.run(command, cwd=repo_root(), env=child_env, check=False)
    return int(result.returncode)


def resolve_tier_value(explicit: Any, default: Any) -> Any:
    return default if explicit is None else explicit


def resolve_paths(args: argparse.Namespace, tier: RunTier, root: Path) -> dict[str, Path]:
    dataset_path = (args.dataset_path or (root / tier.dataset_path)).resolve()
    val_dataset_path = (
        (args.val_dataset_path or (root / tier.val_dataset_path)).resolve()
        if (args.val_dataset_path is not None or tier.val_dataset_path is not None)
        else None
    )
    run_dir = (root / "models" / "aemo" / "dt" / args.run_tag / tier.slug).resolve()
    return {
        "dataset_path": dataset_path,
        "val_dataset_path": val_dataset_path,
        "run_dir": run_dir,
        "save_path": run_dir / "aemo_dt_model.pt",
        "checkpoint_path": run_dir / "aemo_dt_checkpoint.pt",
        "loss_csv_path": run_dir / "aemo_dt_loss_history.csv",
        "progress_snapshot_path": run_dir / "aemo_dt_loss_history_progress.json",
        "surface_manifest_path": run_dir / "aemo_dt_loss_history_surface_manifest.json",
        "launch_plan_path": run_dir / "aemo_training_launch_plan.json",
    }


def validate_environment(
    *,
    tier: RunTier,
    dataset_path: Path,
    val_dataset_path: Path | None,
    model_config_path: Path,
    tracker_ui: str,
) -> dict[str, Any]:
    missing = [
        module_name
        for module_name in ("numpy", "polars", "torch")
        if not _module_available(module_name)
    ]
    rich_available = _module_available("rich")
    runtime = detect_runtime()
    diagnostics = {
        "runtime": runtime,
        "missing_modules": missing,
        "rich_available": rich_available,
        "torch_cuda_available": None,
    }
    if missing:
        raise RuntimeError(
            "Missing required Python modules for AEMO DT training: "
            f"{missing}. Use the documented Distrobox image or install the repo requirements first."
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
    if val_dataset_path is not None and not val_dataset_path.is_file():
        raise FileNotFoundError(f"Validation dataset not found: {val_dataset_path}")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    import torch

    diagnostics["torch_cuda_available"] = bool(torch.cuda.is_available())
    if tier.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            f"Run tier {tier.slug!r} expects CUDA but torch.cuda.is_available() is false in the current runtime."
        )
    if tracker_ui == "rich" and not rich_available:
        raise RuntimeError("Tracker UI 'rich' was requested but the rich package is unavailable.")
    return diagnostics


def build_training_command(
    *,
    root: Path,
    args: argparse.Namespace,
    tier: RunTier,
    paths: dict[str, Path],
) -> list[str]:
    command = [
        sys.executable,
        str(root / "src" / "pretrain_aemo_decision_transformer.py"),
        "--dataset-path",
        str(paths["dataset_path"]),
        "--model-config",
        str(args.model_config.resolve()),
        "--surface-preset",
        tier.surface_preset,
        "--save-path",
        str(paths["save_path"]),
        "--checkpoint-path",
        str(paths["checkpoint_path"]),
        "--loss-csv-path",
        str(paths["loss_csv_path"]),
        "--epochs",
        str(resolve_tier_value(args.epochs, tier.epochs)),
        "--batch-size",
        str(resolve_tier_value(args.batch_size, tier.batch_size)),
        "--lr",
        str(resolve_tier_value(args.lr, tier.lr)),
        "--val-split",
        str(resolve_tier_value(args.val_split, tier.val_split)),
        "--amp-mode",
        args.amp_mode,
        "--num-workers",
        str(resolve_tier_value(args.num_workers, tier.num_workers)),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--checkpoints-per-epoch",
        str(resolve_tier_value(args.checkpoints_per_epoch, tier.checkpoints_per_epoch)),
    ]

    resolved_model_variant = args.model_variant or tier.model_variant
    if resolved_model_variant is not None:
        command.extend(["--model-variant", resolved_model_variant])
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
    if args.rope_enabled:
        command.append("--rope-enabled")
    if args.rope_max_position is not None:
        command.extend(["--rope-max-position", str(args.rope_max_position)])
    if args.rope_base is not None:
        command.extend(["--rope-base", str(args.rope_base)])
    if args.optimizer is not None:
        command.extend(["--optimizer", args.optimizer])
    if args.scheduler is not None:
        command.extend(["--scheduler", args.scheduler])
    if args.optimizer_class_path is not None:
        command.extend(["--optimizer-class-path", args.optimizer_class_path])
    if args.optimizer_kwargs_json is not None:
        command.extend(["--optimizer-kwargs-json", args.optimizer_kwargs_json])
    if args.scheduler_class_path is not None:
        command.extend(["--scheduler-class-path", args.scheduler_class_path])
    if args.scheduler_kwargs_json is not None:
        command.extend(["--scheduler-kwargs-json", args.scheduler_kwargs_json])
    if args.device is not None:
        command.extend(["--device", args.device])
    if paths["val_dataset_path"] is not None:
        command.extend(["--val-dataset-path", str(paths["val_dataset_path"])])
    if tier.train_in_subsets:
        command.append("--train-in-subsets")
        command.extend(["--subset-episodes", str(resolve_tier_value(args.subset_episodes, tier.subset_episodes))])
        command.extend(
            [
                "--epochs-per-subset",
                str(resolve_tier_value(args.epochs_per_subset, tier.epochs_per_subset)),
            ]
        )
    return command


def build_tracker_command(
    *,
    root: Path,
    training_command: Sequence[str],
    paths: dict[str, Path],
    tracker_ui: str,
) -> list[str]:
    return [
        sys.executable,
        str(root / "src" / "dt_progress_runner.py"),
        "--ui",
        tracker_ui,
        "--progress-snapshot-path",
        str(paths["progress_snapshot_path"]),
        "--surface-manifest-path",
        str(paths["surface_manifest_path"]),
        "--",
        *training_command,
    ]


def build_launch_plan(
    *,
    args: argparse.Namespace,
    tier: RunTier,
    paths: dict[str, Path],
    diagnostics: dict[str, Any],
    training_command: Sequence[str],
    tracker_command: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "energydecision.aemo_training_launch_plan.v1",
        "run_tag": args.run_tag,
        "run_tier": tier.slug,
        "run_tier_description": tier.description,
        "runtime_mode": args.runtime_mode,
        "runtime_diagnostics": diagnostics,
        "paths": {key: str(value) for key, value in paths.items()},
        "training_command": list(training_command),
        "tracker_command": list(tracker_command),
        "recommended_evaluation_configs": {
            "pilot_screening": str((repo_root() / "configs" / "aemo_autoresearch_evaluator.mini.json").resolve()),
            "full_heldout": str((repo_root() / "configs" / "aemo_autoresearch_evaluator.example.json").resolve()),
        },
        "monitor_attach_command": [
            sys.executable,
            str(repo_root() / "src" / "dt_progress_runner.py"),
            "--attach",
            "--ui",
            args.tracker_ui,
            "--progress-snapshot-path",
            str(paths["progress_snapshot_path"]),
            "--surface-manifest-path",
            str(paths["surface_manifest_path"]),
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    reentered = maybe_reenter_distrobox(args, raw_argv)
    if reentered is not None:
        return reentered

    root = repo_root().resolve()
    tier = RUN_TIERS[args.run_tier]
    paths = resolve_paths(args, tier, root)
    diagnostics = validate_environment(
        tier=tier,
        dataset_path=paths["dataset_path"],
        val_dataset_path=paths["val_dataset_path"],
        model_config_path=args.model_config.resolve(),
        tracker_ui=args.tracker_ui,
    )
    training_command = build_training_command(root=root, args=args, tier=tier, paths=paths)
    tracker_command = build_tracker_command(
        root=root,
        training_command=training_command,
        paths=paths,
        tracker_ui=args.tracker_ui,
    )
    launch_plan = build_launch_plan(
        args=args,
        tier=tier,
        paths=paths,
        diagnostics=diagnostics,
        training_command=training_command,
        tracker_command=tracker_command,
    )
    write_json(paths["launch_plan_path"], launch_plan)
    print(json.dumps(launch_plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return subprocess.run(tracker_command, cwd=root, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
