import os
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pretrain_aemo_decision_transformer import build_training_commands, get_checkpoint_epoch  # noqa: E402


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        model_config=tmp_path / "config.json",
        surface_preset="aemo_learning_baseline",
        model_variant=None,
        save_path=tmp_path / "model.pt",
        checkpoint_path=tmp_path / "checkpoint.pt",
        loss_csv_path=tmp_path / "loss.csv",
        batch_size=6,
        lr=2e-5,
        val_split=0.1,
        seed=8964,
        amp_mode="auto",
        return_scale=1.0,
        action_loss_weight=1.0,
        state_loss_weight=0.01,
        return_loss_weight=0.002,
        weight_decay=1e-4,
        num_workers=0,
        prefetch_factor=2,
        checkpoint_interval=1,
        checkpoints_per_epoch=6,
        resume=False,
        device=None,
        persistent_workers=False,
        rope_enabled=False,
        rope_max_position=None,
        rope_base=None,
    )


def test_build_training_commands_resumes_after_first_subset(tmp_path: Path):
    args = _args(tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    dataset_paths = [
        tmp_path / "subset_001.parquet",
        tmp_path / "subset_002.parquet",
        tmp_path / "subset_003.parquet",
    ]

    commands = build_training_commands(
        root=root,
        args=args,
        dataset_paths=dataset_paths,
        epochs_per_stage=2,
    )

    assert len(commands) == 3
    assert "--resume" not in commands[0]
    assert "--resume" in commands[1]
    assert "--resume" in commands[2]
    assert commands[0][commands[0].index("--epochs") + 1] == "2"
    assert commands[1][commands[1].index("--epochs") + 1] == "4"
    assert commands[2][commands[2].index("--epochs") + 1] == "6"
    assert commands[0][0] == sys.executable
    assert commands[0][1] == str(root / "src" / "pretrain_decision_transformer.py")
    assert commands[0][commands[0].index("--surface-preset") + 1] == "aemo_learning_baseline"
    assert commands[0][commands[0].index("--patterns") + 1] == "subset_001"
    assert commands[1][commands[1].index("--patterns") + 1] == "subset_002"


def test_build_training_commands_preserves_explicit_resume_on_first_subset(tmp_path: Path):
    args = _args(tmp_path)
    args.resume = True
    root = tmp_path / "repo"
    root.mkdir()
    dataset_paths = [tmp_path / "subset_001.parquet"]

    commands = build_training_commands(
        root=root,
        args=args,
        dataset_paths=dataset_paths,
        epochs_per_stage=3,
    )

    assert len(commands) == 1
    assert "--resume" in commands[0]
    assert commands[0][commands[0].index("--epochs") + 1] == "3"


def test_build_training_commands_adds_explicit_validation_inputs(tmp_path: Path):
    args = _args(tmp_path)
    root = tmp_path / "repo"
    root.mkdir()
    dataset_paths = [tmp_path / "subset_001.parquet", tmp_path / "subset_002.parquet"]
    val_dataset_paths = [tmp_path / "val_subset_001.parquet", tmp_path / "val_subset_002.parquet"]

    commands = build_training_commands(
        root=root,
        args=args,
        dataset_paths=dataset_paths,
        epochs_per_stage=1,
        val_dataset_paths=val_dataset_paths,
    )

    assert commands[0][commands[0].index("--val-split") + 1] == "0.0"
    assert commands[0][commands[0].index("--val-data-dir") + 1] == str(tmp_path)
    patterns_index = commands[0].index("--val-patterns") + 1
    assert commands[0][patterns_index : patterns_index + 2] == ["val_subset_001", "val_subset_002"]


def test_build_training_command_passes_model_variant(tmp_path: Path):
    args = _args(tmp_path)
    args.surface_preset = "aemo_proxy"
    args.model_variant = "compact"
    root = tmp_path / "repo"
    root.mkdir()

    command = build_training_commands(
        root=root,
        args=args,
        dataset_paths=[tmp_path / "subset_001.parquet"],
        epochs_per_stage=1,
    )[0]

    assert command[command.index("--surface-preset") + 1] == "aemo_proxy"
    assert command[command.index("--model-variant") + 1] == "compact"


def test_get_checkpoint_epoch_reads_saved_epoch(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 7}, checkpoint_path)

    assert get_checkpoint_epoch(checkpoint_path) == 7
