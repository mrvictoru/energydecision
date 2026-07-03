import os
import sys
from argparse import Namespace
from importlib import import_module
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pretrain_aemo_decision_transformer import build_training_commands, get_checkpoint_epoch  # noqa: E402


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        model_config=tmp_path / "config.json",
        surface_preset="aemo_learning_baseline",
        model_variant=None,
        optimizer=None,
        scheduler=None,
        optimizer_class_path=None,
        optimizer_kwargs_json=None,
        scheduler_class_path=None,
        scheduler_kwargs_json=None,
        context_length=None,
        state_dim=None,
        act_dim=None,
        n_block=None,
        h_dim=None,
        n_heads=None,
        drop_p=None,
        max_timestep=None,
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
        state_loss_weight=0.002,
        return_loss_weight=0.002,
        weight_decay=1e-4,
        num_workers=0,
        prefetch_factor=2,
        checkpoint_interval=1,
        checkpoints_per_epoch=6,
        max_val_batches=1000,
        resume=False,
        device=None,
        persistent_workers=False,
        rope_enabled=False,
        rope_max_position=None,
        rope_base=None,
    )


def test_parse_args_bakes_in_frontier_aemo_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["pretrain_aemo_decision_transformer.py"])
    args = import_module("pretrain_aemo_decision_transformer").parse_args()

    assert args.surface_preset == "aemo_learning_baseline"
    assert args.batch_size == 16
    assert args.epochs == 2
    assert args.lr == 3e-5
    assert args.amp_mode == "auto"
    assert args.checkpoints_per_epoch == 4
    assert args.num_workers == 0


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


def test_build_training_command_forwards_direct_trainer_knobs(tmp_path: Path):
    args = _args(tmp_path)
    args.context_length = 576
    args.state_dim = 18
    args.act_dim = 3
    args.n_block = 4
    args.h_dim = 128
    args.n_heads = 8
    args.drop_p = 0.1
    args.max_timestep = 2016
    root = tmp_path / "repo"
    root.mkdir()

    command = build_training_commands(
        root=root,
        args=args,
        dataset_paths=[tmp_path / "subset_001.parquet"],
        epochs_per_stage=1,
    )[0]

    assert command[command.index("--context-length") + 1] == "576"
    assert command[command.index("--state-dim") + 1] == "18"
    assert command[command.index("--act-dim") + 1] == "3"
    assert command[command.index("--n-block") + 1] == "4"
    assert command[command.index("--h-dim") + 1] == "128"
    assert command[command.index("--n-heads") + 1] == "8"
    assert command[command.index("--drop-p") + 1] == "0.1"
    assert command[command.index("--max-timestep") + 1] == "2016"


def test_build_training_command_forwards_optimizer_surface_flags(tmp_path: Path):
    args = _args(tmp_path)
    args.optimizer = "custom"
    args.optimizer_class_path = "torch.optim:AdamW"
    args.optimizer_kwargs_json = '{"eps": 1e-7}'
    args.scheduler = "custom"
    args.scheduler_class_path = "torch.optim.lr_scheduler:StepLR"
    args.scheduler_kwargs_json = '{"step_size": 3, "gamma": 0.8}'
    root = tmp_path / "repo"
    root.mkdir()

    command = build_training_commands(
        root=root,
        args=args,
        dataset_paths=[tmp_path / "subset_001.parquet"],
        epochs_per_stage=1,
    )[0]

    assert command[command.index("--optimizer") + 1] == "custom"
    assert command[command.index("--optimizer-class-path") + 1] == "torch.optim:AdamW"
    assert command[command.index("--optimizer-kwargs-json") + 1] == '{"eps": 1e-7}'
    assert command[command.index("--scheduler") + 1] == "custom"
    assert command[command.index("--scheduler-class-path") + 1] == "torch.optim.lr_scheduler:StepLR"
    assert command[command.index("--scheduler-kwargs-json") + 1] == '{"step_size": 3, "gamma": 0.8}'


def test_main_forwards_explicit_validation_dataset(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    dataset_path = tmp_path / "train.parquet"
    val_dataset_path = tmp_path / "val.parquet"
    model_config = tmp_path / "config.json"
    dataset_path.write_bytes(b"")
    val_dataset_path.write_bytes(b"")
    model_config.write_text("{}", encoding="utf-8")

    args = _args(tmp_path)
    args.surface_preset = "aemo_proxy"
    args.dataset_path = dataset_path
    args.val_dataset_path = val_dataset_path
    args.train_in_subsets = False
    args.subset_episodes = None
    args.subset_output_dir = None
    args.epochs = 1
    args.epochs_per_subset = None

    captured: list[list[str]] = []

    monkeypatch.setattr("pretrain_aemo_decision_transformer.parse_args", lambda: args)
    monkeypatch.setattr("pretrain_aemo_decision_transformer.repo_root", lambda: root)
    monkeypatch.setattr(
        "pretrain_aemo_decision_transformer.subprocess.run",
        lambda command, check: captured.append(command),
    )

    from pretrain_aemo_decision_transformer import main  # noqa: WPS433

    main()

    assert len(captured) == 1
    command = captured[0]
    assert command[command.index("--val-data-dir") + 1] == str(tmp_path)
    patterns_index = command.index("--val-patterns") + 1
    assert command[patterns_index] == "val"


def test_get_checkpoint_epoch_reads_saved_epoch(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 7}, checkpoint_path)

    assert get_checkpoint_epoch(checkpoint_path) == 7
