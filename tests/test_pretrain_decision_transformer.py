import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pretrain_decision_transformer as pretrain_dt  # noqa: E402
from transformer_training import TrajectoryDataset  # noqa: E402


def _write_dataset(path: Path, *, state_dim: int = 12, act_dim: int = 1) -> None:
    rows = {
        "episode_id": [0, 0, 1, 1],
        "step": [0, 1, 0, 1],
        "norm_observation": [
            [float(i) for i in range(state_dim)],
            [float(i + 1) for i in range(state_dim)],
            [float(i + 2) for i in range(state_dim)],
            [float(i + 3) for i in range(state_dim)],
        ],
        "action": [
            [0.1] * act_dim,
            [0.2] * act_dim,
            [0.3] * act_dim,
            [0.4] * act_dim,
        ],
        "reward": [1.0, 0.5, 0.25, -0.5],
    }
    pl.DataFrame(rows).write_parquet(path)


def test_merge_trajectory_datasets_preserves_requested_stride():
    episode = {
        "states": np.zeros((10, 12), dtype=np.float32),
        "actions": np.zeros((10, 1), dtype=np.float32),
        "rtgs": np.zeros(10, dtype=np.float32),
        "timesteps": np.arange(10, dtype=np.int64),
        "length": 10,
    }
    dataset = TrajectoryDataset._from_episodes(
        [episode], context_length=2, state_dim=12, act_dim=1, stride=1
    )
    merged = pretrain_dt.merge_trajectory_datasets([dataset], stride=3)
    assert len(merged) == 3


def test_parse_args_accepts_legacy_cli_contract():
    args = pretrain_dt.parse_args(
        [
            "--data-dir",
            "/tmp/data",
            "--patterns",
            "train_episode_01",
            "train_episode_02",
            "--model-config",
            "/tmp/config.json",
            "--epochs",
            "3",
            "--batch-size",
            "8",
            "--lr",
            "0.0002",
            "--val-split",
            "0.2",
            "--save-path",
            "/tmp/model.pt",
            "--checkpoint-path",
            "/tmp/checkpoint.pt",
            "--loss-csv-path",
            "/tmp/loss.csv",
            "--amp-mode",
            "off",
        ]
    )

    assert args.data_dir == Path("/tmp/data")
    assert args.patterns == ["train_episode_01", "train_episode_02"]
    assert args.model_config == Path("/tmp/config.json")
    assert args.epochs == 3
    assert args.batch_size == 8
    assert args.lr == pytest.approx(0.0002)
    assert args.surface_preset == "legacy"
    assert args.optimizer == "adamw"
    assert args.scheduler == "steplr"


def test_training_surface_records_window_stride():
    args = pretrain_dt.parse_args(["--stride", "288"])

    training_kwargs = pretrain_dt.assemble_training_kwargs(args)

    assert training_kwargs["stride"] == 288
    assert "stride" in pretrain_dt.SEARCHABLE_KNOBS


def test_parse_args_accepts_custom_optimizer_and_scheduler_contract():
    args = pretrain_dt.parse_args(
        [
            "--optimizer",
            "custom",
            "--optimizer-class-path",
            "torch.optim:AdamW",
            "--optimizer-kwargs-json",
            '{"eps": 1e-7}',
            "--scheduler",
            "custom",
            "--scheduler-class-path",
            "torch.optim.lr_scheduler:StepLR",
            "--scheduler-kwargs-json",
            '{"step_size": 3, "gamma": 0.8}',
        ]
    )

    assert args.optimizer == "custom"
    assert args.optimizer_class_path == "torch.optim:AdamW"
    assert args.optimizer_kwargs_json == '{"eps": 1e-7}'
    assert args.scheduler == "custom"
    assert args.scheduler_class_path == "torch.optim.lr_scheduler:StepLR"
    assert args.scheduler_kwargs_json == '{"step_size": 3, "gamma": 0.8}'


def test_load_model_kwargs_rejects_unknown_keys(tmp_path: Path):
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(json.dumps({"state_dim": 12, "unsupported_knob": 7}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported model config keys"):
        pretrain_dt.load_model_kwargs(config_path)


def test_resolve_training_surface_rejects_action_mode_mismatch():
    args = pretrain_dt.parse_args(
        [
            "--surface-preset",
            "legacy",
            "--state-dim",
            "18",
            "--act-dim",
            "3",
            "--action-mode",
            "simple",
        ]
    )

    with pytest.raises(ValueError, match="requires act_dim=1"):
        pretrain_dt.resolve_training_surface(args, base_kwargs={})


def test_resolve_training_surface_exposes_aemo_learning_baseline_defaults():
    args = pretrain_dt.parse_args(
        [
            "--surface-preset",
            "aemo_learning_baseline",
        ]
    )

    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})

    assert surface.model_variant == "deeper_wider"
    assert surface.model_kwargs["context_len"] == 180
    assert surface.model_kwargs["act_dim"] == 3
    assert surface.model_kwargs["n_block"] == 8
    assert surface.model_kwargs["h_dim"] == 384
    assert surface.training_kwargs["lr"] == pytest.approx(3e-5)
    assert surface.training_kwargs["batch_size"] == 16
    assert surface.training_kwargs["epochs"] == 2
    assert surface.training_kwargs["checkpoints_per_epoch"] == 4


def test_resolve_training_surface_respects_explicit_learning_baseline_overrides():
    args = pretrain_dt.parse_args(
        [
            "--surface-preset",
            "aemo_learning_baseline",
            "--epochs",
            "4",
            "--lr",
            "4e-5",
            "--batch-size",
            "16",
        ]
    )

    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})

    assert surface.training_kwargs["epochs"] == 4
    assert surface.training_kwargs["lr"] == pytest.approx(4e-5)
    assert surface.training_kwargs["batch_size"] == 16


def test_resolve_training_surface_requires_custom_optimizer_path():
    args = pretrain_dt.parse_args(
        [
            "--optimizer",
            "custom",
        ]
    )

    with pytest.raises(ValueError, match="requires --optimizer-class-path"):
        pretrain_dt.resolve_training_surface(args, base_kwargs={})


def test_resolve_training_surface_parses_custom_optimizer_surface_kwargs():
    args = pretrain_dt.parse_args(
        [
            "--optimizer",
            "custom",
            "--optimizer-class-path",
            "torch.optim:AdamW",
            "--optimizer-kwargs-json",
            '{"eps": 1e-7}',
            "--scheduler",
            "none",
        ]
    )

    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})

    assert surface.training_kwargs["optimizer"] == "custom"
    assert surface.training_kwargs["optimizer_class_path"] == "torch.optim:AdamW"
    assert surface.training_kwargs["optimizer_kwargs"] == {"eps": 1e-7}
    assert surface.training_kwargs["scheduler"] == "none"


def test_recommend_pilot_ranking_prefers_val_action_for_aemo_proxy():
    ranking = pretrain_dt.recommend_pilot_ranking(
        surface_preset="aemo_proxy",
        best_val_total_loss=0.62,
        best_val_action_loss=0.48,
    )

    assert ranking["pilot_ranking_metric"] == "best_val_action_loss"
    assert ranking["pilot_ranking_value"] == pytest.approx(0.48)
    assert ranking["pilot_ranking_guardrail_metric"] == "best_val_total_loss"
    assert ranking["pilot_ranking_guardrail_value"] == pytest.approx(0.62)


def test_aemo_proxy_context_override_auto_scales_rope_cache():
    args = pretrain_dt.parse_args(
        [
            "--surface-preset",
            "aemo_proxy",
            "--context-length",
            "120",
        ]
    )

    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})

    assert surface.model_kwargs["context_len"] == 120
    assert surface.model_kwargs["rope_max_position"] == 360


def test_aemo_proxy_context_override_rejects_explicit_small_rope_cache():
    args = pretrain_dt.parse_args(
        [
            "--surface-preset",
            "aemo_proxy",
            "--context-length",
            "120",
            "--rope-max-position",
            "180",
        ]
    )

    with pytest.raises(ValueError, match="rope_max_position is too small"):
        pretrain_dt.resolve_training_surface(args, base_kwargs={})


def test_aemo_proxy_frontier_defaults_bake_in_frontier_shape():
    args = pretrain_dt.parse_args(["--surface-preset", "aemo_proxy_frontier"])

    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})

    assert surface.model_variant == "deeper_wider"
    assert surface.model_kwargs["context_len"] == 180
    assert surface.model_kwargs["rope_max_position"] == 540
    assert surface.model_kwargs["n_block"] == 8
    assert surface.model_kwargs["h_dim"] == 384
    assert surface.training_kwargs["batch_size"] == 16
    assert surface.training_kwargs["epochs"] == 2
    assert surface.training_kwargs["checkpoints_per_epoch"] == 4


def test_validate_preset_dataset_policy_requires_explicit_validation_for_learning_baseline(tmp_path: Path):
    args = pretrain_dt.parse_args(["--surface-preset", "aemo_learning_baseline"])
    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})
    train_file = tmp_path / "aemo_dt_dataset_train_subset_001.parquet"
    train_file.write_bytes(b"")

    with pytest.raises(ValueError, match="requires explicit validation parquet files"):
        pretrain_dt.validate_preset_dataset_policy(
            surface=surface,
            parquet_files=[train_file],
            train_episode_count=24,
        )


def test_validate_preset_dataset_policy_rejects_narrow_proxy_subset_for_learning_baseline(tmp_path: Path):
    args = pretrain_dt.parse_args(
        [
            "--surface-preset",
            "aemo_learning_baseline",
            "--split-policy",
            "explicit_validation",
            "--val-data-dir",
            str(tmp_path),
        ]
    )
    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})
    train_file = tmp_path / "aemo_dt_dataset_train_subset_007.parquet"
    train_file.write_bytes(b"")

    with pytest.raises(ValueError, match="cannot use the narrow proxy slice"):
        pretrain_dt.validate_preset_dataset_policy(
            surface=surface,
            parquet_files=[train_file],
            train_episode_count=24,
        )


def test_validate_preset_dataset_policy_rejects_too_few_learning_baseline_episodes(tmp_path: Path):
    args = pretrain_dt.parse_args(
        [
            "--surface-preset",
            "aemo_learning_baseline",
            "--split-policy",
            "explicit_validation",
            "--val-data-dir",
            str(tmp_path),
        ]
    )
    surface = pretrain_dt.resolve_training_surface(args, base_kwargs={})
    train_file = tmp_path / "aemo_dt_dataset_train_subset_001.parquet"
    train_file.write_bytes(b"")

    with pytest.raises(ValueError, match="requires at least 8 training episodes"):
        pretrain_dt.validate_preset_dataset_policy(
            surface=surface,
            parquet_files=[train_file],
            train_episode_count=2,
        )


def test_validate_dataset_dimensions_rejects_mismatch(tmp_path: Path):
    dataset_path = tmp_path / "episodes.parquet"
    _write_dataset(dataset_path, state_dim=18, act_dim=3)
    datasets = pretrain_dt.load_trajectory_datasets(
        parquet_files=[dataset_path],
        context_length=4,
        state_dim=18,
        act_dim=3,
        discount=0.99,
    )

    with pytest.raises(ValueError, match="action_mode='simple'"):
        pretrain_dt.validate_dataset_dimensions(
            datasets=datasets,
            expected_state_dim=18,
            expected_act_dim=3,
            action_mode="simple",
            label="Training",
        )


def test_main_writes_backward_compatible_artifacts_and_surface_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.json"
    save_path = output_dir / "model.pt"
    checkpoint_path = output_dir / "checkpoint.pt"
    loss_csv_path = output_dir / "loss.csv"
    progress_snapshot_path = output_dir / "loss_progress.json"
    data_dir.mkdir()
    output_dir.mkdir()
    _write_dataset(data_dir / "train_episode_01.parquet")
    # The 2-step episodes need a context length <= 2 to produce sliding windows;
    # otherwise the default context_len=60 yields zero training samples.
    config_path.write_text(json.dumps({"state_dim": 12, "act_dim": 1, "context_len": 2}), encoding="utf-8")

    monkeypatch.setattr(pretrain_dt, "repo_root", lambda: tmp_path)

    def fake_train_decision_transformer(**kwargs):
        assert kwargs["optimizer_name"] == "adamw"
        assert kwargs["scheduler_name"] == "steplr"
        assert kwargs["optimizer_kwargs"] == {}
        assert kwargs["scheduler_kwargs"] == {}
        Path(kwargs["save_path"]).write_bytes(b"weights")
        Path(kwargs["checkpoint_path"]).write_bytes(b"checkpoint")
        Path(kwargs["progress_snapshot_path"]).write_text(
            json.dumps(
                {
                    "schema": "energydecision.dt_progress_snapshot.v1",
                    "status": "finished",
                    "epoch": 2,
                    "epochs": 2,
                    "progress_fraction": 1.0,
                }
            ),
            encoding="utf-8",
        )
        return (
            kwargs["model"],
            [1.0, 0.5],
            [0.8, 0.4],
            {
                "train_action_losses": [0.9, 0.4],
                "train_state_losses": [0.08, 0.04],
                "train_return_losses": [0.02, 0.01],
                "val_action_losses": [0.7, 0.35],
                "val_state_losses": [0.07, 0.03],
                "val_return_losses": [0.01, 0.01],
                "loss_history": [
                    {
                        "timestamp": "2026-04-28T16:00:00",
                        "epoch": 1,
                        "segment": 1,
                        "batch_idx": 1,
                        "train_total_avg": 1.0,
                        "train_total_ema": 1.0,
                        "train_action_avg": 0.9,
                        "train_state_avg": 0.08,
                        "train_return_avg": 0.02,
                        "train_valid": True,
                        "val_total": 0.8,
                        "val_action": 0.7,
                        "val_state": 0.07,
                        "val_return": 0.01,
                        "val_valid": True,
                    }
                ],
            },
        )

    monkeypatch.setattr(pretrain_dt, "train_decision_transformer", fake_train_decision_transformer)

    pretrain_dt.main(
        [
            "--data-dir",
            str(data_dir),
            "--patterns",
            "train_episode_01",
            "--model-config",
            str(config_path),
            "--save-path",
            str(save_path),
            "--checkpoint-path",
            str(checkpoint_path),
            "--loss-csv-path",
            str(loss_csv_path),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--val-split",
            "0.5",
        ]
    )

    with loss_csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
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
    assert rows[1][0] == "1"
    assert rows[2][0] == "2"

    checkpoints_csv = loss_csv_path.with_name("loss_checkpoints.csv")
    assert checkpoints_csv.exists()
    assert progress_snapshot_path.exists()
    manifest_path = loss_csv_path.with_name("loss_surface_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["editable_training_surface_file"].endswith("pretrain_decision_transformer.py")
    assert manifest["surface_preset"] == "legacy"
    assert manifest["optimizer"] == "adamw"
    assert manifest["scheduler"] == "steplr"
    assert "searchable_knobs" in manifest
    assert "frozen_invariants" in manifest
    assert manifest["paths"]["progress_snapshot_path"] == str(progress_snapshot_path)
    assert manifest["dataset_summary"]["train"]["file_count"] == 1
    assert manifest["dataset_summary"]["train"]["window_count"] >= 1
    assert manifest["run_summary"]["checkpoint_count"] == 1
    assert manifest["run_summary"]["total_windows_processed"] >= 1
    assert manifest["run_summary"]["final_val_action_loss"] == pytest.approx(0.35)
    assert manifest["run_summary"]["best_val_action_loss"] == pytest.approx(0.35)
    assert manifest["run_summary"]["pilot_ranking_metric"] == "best_val_total_loss"
