import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import launch_aemo_training as launcher  # noqa: E402


def test_build_training_command_uses_proxy_baseline_defaults(tmp_path: Path):
    args = launcher.parse_args(["--run-tier", "proxy-baseline", "--run-tag", "demo", "--runtime-mode", "allow-host"])
    paths = {
        "dataset_path": tmp_path / "train.parquet",
        "val_dataset_path": tmp_path / "val.parquet",
        "save_path": tmp_path / "model.pt",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "loss_csv_path": tmp_path / "loss.csv",
    }

    command = launcher.build_training_command(
        root=tmp_path,
        args=args,
        tier=launcher.RUN_TIERS["proxy-baseline"],
        paths=paths,
    )

    assert command[0] == sys.executable
    assert command[command.index("--surface-preset") + 1] == "aemo_proxy"
    assert command[command.index("--batch-size") + 1] == "128"
    assert command[command.index("--checkpoints-per-epoch") + 1] == "4"
    assert command[command.index("--val-dataset-path") + 1] == str(paths["val_dataset_path"])


def test_build_training_command_enables_subset_mode_for_learning_baseline(tmp_path: Path):
    args = launcher.parse_args(["--run-tier", "learning-baseline", "--run-tag", "demo", "--runtime-mode", "allow-host"])
    paths = {
        "dataset_path": tmp_path / "train.parquet",
        "val_dataset_path": None,
        "save_path": tmp_path / "model.pt",
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "loss_csv_path": tmp_path / "loss.csv",
    }

    command = launcher.build_training_command(
        root=tmp_path,
        args=args,
        tier=launcher.RUN_TIERS["learning-baseline"],
        paths=paths,
    )

    assert "--train-in-subsets" in command
    assert command[command.index("--subset-episodes") + 1] == "24"
    assert command[command.index("--epochs-per-subset") + 1] == "1"


def test_maybe_reenter_distrobox_invokes_distrobox_when_requested(monkeypatch: pytest.MonkeyPatch):
    args = launcher.parse_args(["--run-tier", "proxy-baseline", "--runtime-mode", "require-distrobox"])
    monkeypatch.setattr(launcher, "detect_runtime", lambda: {"inside_container": False})
    monkeypatch.setattr(launcher.shutil, "which", lambda _: "/usr/bin/distrobox")
    captured: list[list[str]] = []

    class Result:
        returncode = 0

    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, cwd, env, check: captured.append(command) or Result(),
    )

    exit_code = launcher.maybe_reenter_distrobox(args, ["--run-tier", "proxy-baseline"])

    assert exit_code == 0
    assert captured
    assert captured[0][:3] == ["/usr/bin/distrobox", "enter", "energydecision-gpu"]


def test_main_writes_launch_plan_on_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    dataset_path = tmp_path / "train.parquet"
    val_dataset_path = tmp_path / "val.parquet"
    config_path = tmp_path / "config.json"
    dataset_path.write_bytes(b"")
    val_dataset_path.write_bytes(b"")
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(launcher, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(launcher, "detect_runtime", lambda: {"inside_container": True})
    monkeypatch.setattr(launcher, "_module_available", lambda name: True)

    import types

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)))

    exit_code = launcher.main(
        [
            "--run-tier",
            "proxy-baseline",
            "--run-tag",
            "demo",
            "--runtime-mode",
            "allow-host",
            "--dataset-path",
            str(dataset_path),
            "--val-dataset-path",
            str(val_dataset_path),
            "--model-config",
            str(config_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    plan_path = tmp_path / "models" / "aemo" / "dt" / "demo" / "proxy_baseline" / "aemo_training_launch_plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["run_tier"] == "proxy_baseline"
    assert plan["paths"]["dataset_path"] == str(dataset_path)
    stdout = capsys.readouterr().out
    assert '"run_tag": "demo"' in stdout
