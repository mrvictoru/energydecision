import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dt_progress_runner as progress_runner  # noqa: E402


def test_render_dashboard_includes_high_signal_progress_fields(tmp_path: Path):
    log_path = tmp_path / "training.log"
    log_path.write_text("line one\nline two\n", encoding="utf-8")

    snapshot = {
        "status": "running",
        "epoch": 2,
        "epochs": 4,
        "segment": 1,
        "checkpoints_per_epoch": 6,
        "progress_fraction": 0.5,
        "current_train": {
            "train_total_avg": 0.123,
            "train_action_avg": 0.100,
            "train_state_avg": 0.020,
            "train_return_avg": 0.003,
            "train_total_ema": 0.111,
        },
        "validation": {"val_total": 0.456},
        "best": {"score": 0.456, "val_loss": 0.456, "train_loss_est": 0.111},
        "resources": {"cpu": "42%", "ram": "4.0/8.0G", "pcpu": "77%", "prss": "1.2G"},
        "latest_history": [
            {
                "epoch": 2,
                "segment": 1,
                "train_total_avg": 0.123,
                "val_total": 0.456,
                "train_total_ema": 0.111,
            }
        ],
    }
    manifest = {
        "surface_preset": "autoresearch_safe",
        "model_variant": "baseline",
        "optimizer": "adamw",
        "scheduler": "steplr",
    }

    dashboard = progress_runner.render_dashboard(
        snapshot=snapshot,
        manifest=manifest,
        log_path=log_path,
        log_tail=log_path.read_text(encoding="utf-8").splitlines()[-2:],
        command=["python3", "src/pretrain_decision_transformer.py"],
        child=None,
        started_at=0.0,
    )

    assert "progress: 50.0%" in dashboard
    assert "surface: preset=autoresearch_safe" in dashboard
    assert "variant=baseline" in dashboard
    assert "train_total_avg=0.123" in dashboard
    assert "val_total=0.456" in dashboard
    assert "pcpu=77%" in dashboard
    assert "line two" in dashboard


def test_build_dashboard_state_preserves_structured_fields(tmp_path: Path):
    log_path = tmp_path / "training.log"
    state = progress_runner.build_dashboard_state(
        snapshot={
            "status": "running",
            "progress_fraction": 0.25,
            "current_train": {"train_total_avg": 0.2},
            "validation": {"val_total": 0.4},
            "best": {"val_loss": 0.4},
            "resources": {"cpu": "50%"},
        },
        manifest={"surface_preset": "aemo_proxy", "model_variant": "compact"},
        log_path=log_path,
        log_tail=["hello"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
    )

    assert state.status == "running"
    assert state.progress_percent == 25.0
    assert "preset=aemo_proxy" in (state.surface_text or "")
    assert state.log_tail == ["hello"]


def test_normalize_command_supports_attach_mode_without_child_command():
    assert progress_runner.normalize_command([], attach=True) == []
    assert progress_runner.normalize_command(["--"], attach=True) == []


def test_normalize_command_rejects_invalid_combinations():
    try:
        progress_runner.normalize_command([], attach=False)
    except ValueError as exc:
        assert "must be provided" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected normalize_command to reject missing command")

    try:
        progress_runner.normalize_command(["python3"], attach=True)
    except ValueError as exc:
        assert "No command" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected normalize_command to reject attach commands")


def test_build_dashboard_state_marks_attach_mode(tmp_path: Path):
    state = progress_runner.build_dashboard_state(
        snapshot=None,
        manifest=None,
        log_path=tmp_path / "monitor.log",
        log_tail=[],
        command=[],
        child=None,
        started_at=0.0,
    )

    assert state.command_text == "(attach mode)"


def test_resolve_ui_mode_falls_back_to_plain_without_tty(monkeypatch):
    monkeypatch.setattr(progress_runner, "RICH_AVAILABLE", True)
    assert progress_runner.resolve_ui_mode("auto", is_tty=False) == "plain"
    assert progress_runner.resolve_ui_mode("rich", is_tty=False) == "plain"


def test_resolve_ui_mode_prefers_rich_when_available():
    assert progress_runner.resolve_ui_mode("plain", is_tty=True) == "plain"
    assert progress_runner.resolve_ui_mode("auto", is_tty=True) == ("rich" if progress_runner.RICH_AVAILABLE else "plain")


def test_build_rich_dashboard_contains_high_signal_labels(tmp_path: Path):
    if not progress_runner.RICH_AVAILABLE:
        return

    state = progress_runner.build_dashboard_state(
        snapshot={
            "status": "running",
            "progress_fraction": 0.5,
            "current_train": {"train_total_avg": 0.123},
            "validation": {"val_total": 0.456},
            "best": {"val_loss": 0.456},
            "resources": {"cpu": "42%", "pcpu": "77%"},
        },
        manifest={"surface_preset": "autoresearch_safe", "model_variant": "baseline"},
        log_path=tmp_path / "monitor.log",
        log_tail=["line one", "line two"],
        command=["python3", "src/pretrain_decision_transformer.py"],
        child=None,
        started_at=0.0,
    )

    console = progress_runner.Console(record=True, force_terminal=False, width=120)
    console.print(progress_runner.build_rich_dashboard(state))
    rendered = console.export_text()

    assert "DT progress runner" in rendered
    assert "metrics" in rendered
    assert "resources" in rendered
    assert "line two" in rendered
