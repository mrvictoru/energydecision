import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoresearch.config_utils import (  # noqa: E402
    build_training_cli_args,
    diff_configs,
    validate_mutable_surface,
)


def test_diff_configs_handles_added_removed_and_changed():
    baseline = {"a": 1, "b": 2, "c": 3}
    candidate = {"a": 1, "b": 4, "d": 9}
    diff = diff_configs(baseline, candidate)
    assert diff["b"] == {"old": 2, "new": 4}
    assert diff["c"] == {"old": 3, "new": None}
    assert diff["d"] == {"old": None, "new": 9}


def test_validate_mutable_surface_rejects_disallowed_key():
    with pytest.raises(ValueError):
        validate_mutable_surface({"state_dim": 99})


def test_build_training_cli_args_household_and_override(tmp_path: Path):
    config = {"epochs": 7, "batch_size": 8, "lr": 1e-4}
    benchmark = {
        "environment": "household",
        "data_dir": str(tmp_path),
        "state_dim": 12,
        "act_dim": 1,
        "max_timestep": 100,
        "discount": 0.99,
        "train_patterns": ["train_ep_*.parquet"],
    }
    args = build_training_cli_args(config, benchmark, str(tmp_path / "out"), str(tmp_path / "train.py"), epochs_override=1)

    assert "--data-dir" in args
    assert "--patterns" in args
    assert args[args.index("--epochs") + 1] == "1"


def test_build_training_cli_args_aemo(tmp_path: Path):
    config = {"epochs": 3}
    benchmark = {
        "environment": "aemo",
        "dataset_path": str(tmp_path / "dataset.parquet"),
        "state_dim": 18,
        "act_dim": 3,
        "max_timestep": 100,
    }
    args = build_training_cli_args(config, benchmark, str(tmp_path / "out"), str(tmp_path / "train.py"))

    assert "--dataset-path" in args
    assert args[args.index("--epochs") + 1] == "3"
