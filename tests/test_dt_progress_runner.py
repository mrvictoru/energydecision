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
        "dataset_summary": {
            "train": {"file_count": 1, "episode_count": 2, "window_count": 32},
            "val": {"file_count": 1, "episode_count": 1, "window_count": 16},
        },
        "run_summary": {"effective_windows_per_second": 12.5, "checkpoint_count": 1},
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
    assert "datasets: train_files=1" in dashboard
    assert "variant=baseline" in dashboard
    assert "train_total_avg=0.123" in dashboard
    assert "val_total=0.456" in dashboard
    assert "run summary: wins_per_s=12.5" in dashboard
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
    assert state.dataset_text is None
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
    assert "training progress" in rendered
    assert "gpu" in rendered
    assert "line two" in rendered


def test_sparkline_renders_unicode_blocks():
    assert progress_runner.sparkline([]) == ""
    result = progress_runner.sparkline([1.0, 2.0, 3.0, 4.0])
    assert len(result) == 4
    assert all(ch in progress_runner.SPARKLINE_CHARS for ch in result)
    assert result[0] == progress_runner.SPARKLINE_CHARS[0]
    assert result[-1] == progress_runner.SPARKLINE_CHARS[-1]


def test_sparkline_constant_values():
    result = progress_runner.sparkline([5.0, 5.0, 5.0])
    assert len(result) == 3
    n = len(progress_runner.SPARKLINE_CHARS) - 1
    mid = progress_runner.SPARKLINE_CHARS[n // 2]
    assert all(ch == mid for ch in result)


def test_sparkline_width_truncation():
    values = list(range(100))
    result = progress_runner.sparkline(values, width=20)
    assert len(result) == 20


def test_bar_renders_filled_and_empty():
    assert progress_runner.bar(0.0, width=10) == "░" * 10
    assert progress_runner.bar(1.0, width=10) == "█" * 10
    result = progress_runner.bar(0.5, width=10)
    assert "█" in result and "░" in result
    assert len(result) == 10


def test_bar_clamps_fraction():
    assert progress_runner.bar(-0.5, width=10) == "░" * 10
    assert progress_runner.bar(1.5, width=10) == "█" * 10


def test_system_monitor_returns_dict():
    monitor = progress_runner.SystemMonitor(refresh_seconds=0.0)
    metrics = monitor.poll(force=True)
    assert isinstance(metrics, dict)


def test_system_monitor_caches_within_refresh_window():
    monitor = progress_runner.SystemMonitor(refresh_seconds=60.0)
    first = monitor.poll(force=True)
    second = monitor.poll()
    assert first == second


def test_metrics_history_records_and_retrieves(tmp_path: Path):
    csv_path = tmp_path / "metrics.csv"
    history = progress_runner.MetricsHistory(max_samples=10, csv_path=csv_path)
    history.record(
        snapshot={"epoch": 1, "segment": 0, "progress_fraction": 0.1, "current_train": {"train_total_avg": 0.5}},
        system={"gpu_temp_c": 70, "gpu_util_pct": 95},
    )
    history.record(
        snapshot={"epoch": 1, "segment": 1, "progress_fraction": 0.2, "current_train": {"train_total_avg": 0.4}},
        system={"gpu_temp_c": 72, "gpu_util_pct": 98},
    )
    temps = history.history("gpu_temp_c")
    assert temps == [70.0, 72.0]
    losses = history.history("train_loss")
    assert losses == [0.5, 0.4]
    history.close()
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "gpu_temp_c" in content
    assert "70" in content


def test_metrics_history_rolling_buffer(tmp_path: Path):
    history = progress_runner.MetricsHistory(max_samples=3, csv_path=tmp_path / "m.csv")
    for i in range(5):
        history.record(snapshot=None, system={"val": float(i)})
    vals = history.history("val")
    assert len(vals) == 3
    assert vals == [2.0, 3.0, 4.0]
    history.close()


def test_metrics_history_no_csv(tmp_path: Path):
    history = progress_runner.MetricsHistory(max_samples=5, csv_path=None)
    history.record(snapshot=None, system={"x": 1.0})
    assert history.history("x") == [1.0]
    history.close()


def test_plain_dashboard_includes_system_metrics(tmp_path: Path):
    log_path = tmp_path / "training.log"
    log_path.write_text("log line\n", encoding="utf-8")
    state = progress_runner.build_dashboard_state(
        snapshot={"status": "running", "progress_fraction": 0.5, "current_train": {"train_total_avg": 0.1}},
        manifest=None,
        log_path=log_path,
        log_tail=["log line"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
        system_metrics={"gpu_temp_c": 75, "gpu_power_w": 180.0, "gpu_util_pct": 99, "sys_cpu_pct": 42.0},
        loss_history=[0.5, 0.4, 0.3, 0.2],
        gpu_util_history=[90.0, 95.0, 99.0],
    )
    rendered = progress_runner.render_plain_dashboard(state)
    assert "75°C" in rendered
    assert "180" in rendered or "180.0" in rendered
    assert "99%" in rendered
    assert "train loss:" in rendered


def test_rich_dashboard_shows_gpu_and_system_panels(tmp_path: Path):
    if not progress_runner.RICH_AVAILABLE:
        return
    state = progress_runner.build_dashboard_state(
        snapshot={"status": "running", "progress_fraction": 0.5, "current_train": {"train_total_avg": 0.1}},
        manifest=None,
        log_path=tmp_path / "monitor.log",
        log_tail=["log line"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
        system_metrics={"gpu_name": "Test GPU", "gpu_temp_c": 75, "gpu_power_w": 180.0, "gpu_util_pct": 99, "sys_cpu_pct": 42.0},
        loss_history=[0.5, 0.4, 0.3, 0.2],
        gpu_util_history=[90.0, 95.0, 99.0],
        gpu_temp_history=[70.0, 73.0, 75.0],
    )
    console = progress_runner.Console(record=True, force_terminal=False, width=120, height=40)
    console.print(progress_runner.build_rich_dashboard(state))
    rendered = console.export_text()
    assert "Test GPU" in rendered
    assert "System" in rendered
    assert "training progress" in rendered
    assert "config" in rendered


SAMPLE_TQDM_LINE = (
    "Epoch 1/2:   4%|▍         | 10099/227086 [1:05:36<23:42:21,  2.54batch/s, "
    "loss=0.0175, avg=0.0176, ema=0.0106, lr=3.00e-05, skip=0, seg=1/4, "
    "cpu=7%, ram=7.3/62.7G, gpu=100%, vram=5.4/8.0G, vpeak=4.6G, "
    "pcpu=111%, prss=4.1G, pvms=25.3G, pth=118]"
)


def test_parse_tqdm_line_basic():
    result = progress_runner.parse_tqdm_line(SAMPLE_TQDM_LINE)
    assert result is not None
    assert result["epoch"] == 1
    assert result["epochs"] == 2
    assert abs(result["progress_pct"] - 4.0) < 0.1
    assert result["batch"] == 10099
    assert result["total_batches"] == 227086
    assert result["elapsed"] == "1:05:36"
    assert result["remaining"] == "23:42:21"
    assert abs(result["batch_per_s"] - 2.54) < 0.01
    assert abs(result["loss"] - 0.0175) < 0.0001
    assert abs(result["avg"] - 0.0176) < 0.0001
    assert abs(result["ema"] - 0.0106) < 0.0001
    assert abs(result["lr"] - 3e-05) < 1e-08
    assert result["seg"] == 1
    assert result["gpu"] == 100
    assert result["cpu"] == 7


def test_parse_tqdm_line_returns_none_for_non_matching():
    assert progress_runner.parse_tqdm_line("some random log line") is None
    assert progress_runner.parse_tqdm_line("") is None
    assert progress_runner.parse_tqdm_line("Partitioned aemo_dt_dataset.parquet") is None


def test_parse_tqdm_line_embedded_in_text():
    text = "prefix text " + SAMPLE_TQDM_LINE + " suffix"
    result = progress_runner.parse_tqdm_line(text)
    assert result is not None
    assert result["epoch"] == 1
    assert result["batch"] == 10099


def test_parse_last_tqdm_line_finds_last_match():
    lines = [
        "some header line",
        "Epoch 1/2:   2%|  | 5000/227086 [0:30:00<24:00:00,  2.50batch/s, loss=0.05, avg=0.06, ema=0.04, lr=3.00e-05, skip=0, seg=1/4, cpu=5%, ram=7.0/62.7G, gpu=95%, vram=5.0/8.0G, vpeak=4.0G, pcpu=100%, prss=4.0G, pvms=25.0G, pth=118]",
        "more text",
        SAMPLE_TQDM_LINE,
    ]
    result = progress_runner._parse_last_tqdm_line(lines)
    assert result is not None
    assert result["batch"] == 10099


def test_parse_last_tqdm_line_returns_none_when_no_match():
    assert progress_runner._parse_last_tqdm_line(["no match", "also no match"]) is None
    assert progress_runner._parse_last_tqdm_line([]) is None


def test_dashboard_shows_parsed_progress_when_no_snapshot(tmp_path: Path):
    log_path = tmp_path / "monitor.log"
    state = progress_runner.build_dashboard_state(
        snapshot=None,
        manifest=None,
        log_path=log_path,
        log_tail=[SAMPLE_TQDM_LINE],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
    )
    assert state.status == "running"
    assert state.progress_percent is not None
    assert abs(state.progress_percent - 4.0) < 0.1
    assert "loss=0.0175" in state.train_text
    assert "epoch 1/2" in state.progress_text
    assert "batch 10099/227086" in state.progress_text


def test_dashboard_shows_waiting_when_no_snapshot_and_no_log(tmp_path: Path):
    log_path = tmp_path / "monitor.log"
    state = progress_runner.build_dashboard_state(
        snapshot=None,
        manifest=None,
        log_path=log_path,
        log_tail=["some random log line"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
    )
    assert state.status == "waiting for training output"
    assert state.progress_percent is None
    assert state.train_text == "n/a"


def test_dashboard_uses_parsed_loss_for_history(tmp_path: Path):
    log_path = tmp_path / "monitor.log"
    state = progress_runner.build_dashboard_state(
        snapshot=None,
        manifest=None,
        log_path=log_path,
        log_tail=[SAMPLE_TQDM_LINE],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
        loss_history=[0.05, 0.04, 0.03],
    )
    assert len(state.loss_history) == 4
    assert abs(state.loss_history[-1] - 0.0175) < 0.0001


def test_dashboard_parsed_overrides_snapshot_train_text(tmp_path: Path):
    log_path = tmp_path / "monitor.log"
    state = progress_runner.build_dashboard_state(
        snapshot={"status": "running", "progress_fraction": 0.5, "current_train": {"train_total_avg": 0.1}},
        manifest=None,
        log_path=log_path,
        log_tail=[SAMPLE_TQDM_LINE],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
    )
    assert "loss=0.0175" in state.train_text
    assert "epoch 1/2" in state.progress_text


def test_config_extracted_from_manifest(tmp_path: Path):
    log_path = tmp_path / "monitor.log"
    log_path.write_text("log line\n", encoding="utf-8")
    state = progress_runner.build_dashboard_state(
        snapshot={"status": "running", "progress_fraction": 0.5, "current_train": {"train_total_avg": 0.1}},
        manifest={
            "model_variant": "deeper_wider",
            "optimizer": "adamw",
            "scheduler": "steplr",
            "model_kwargs": {"context_len": 180, "n_block": 8, "h_dim": 512, "n_heads": 8, "drop_p": 0.15, "act_dim": 3, "state_dim": 18},
            "dataset_summary": {"train": {"window_count": 3633361}},
        },
        log_path=log_path,
        log_tail=["log line"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
    )
    assert state.config_text is not None
    assert "context=180" in state.config_text
    assert "n_block=8" in state.config_text
    assert "h_dim=512" in state.config_text
    assert "n_heads=8" in state.config_text
    assert "drop_p=0.15" in state.config_text
    assert "variant=deeper_wider" in state.config_text


def test_val_loss_tracked_from_snapshot(tmp_path: Path):
    log_path = tmp_path / "monitor.log"
    log_path.write_text("log\n", encoding="utf-8")
    state = progress_runner.build_dashboard_state(
        snapshot={"status": "running", "progress_fraction": 0.5, "current_train": {"train_total_avg": 0.1}, "validation": {"val_total": 0.456}},
        manifest=None,
        log_path=log_path,
        log_tail=["log"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
        val_loss_history=[0.5],
    )
    assert len(state.val_loss_history) == 2
    assert abs(state.val_loss_history[-1] - 0.456) < 0.001


def test_plain_dashboard_shows_config(tmp_path: Path):
    log_path = tmp_path / "training.log"
    log_path.write_text("log\n", encoding="utf-8")
    state = progress_runner.build_dashboard_state(
        snapshot={"status": "running", "progress_fraction": 0.5, "current_train": {"train_total_avg": 0.1}},
        manifest={"model_variant": "compact", "model_kwargs": {"context_len": 120, "n_block": 6}},
        log_path=log_path,
        log_tail=["log"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
    )
    rendered = progress_runner.render_plain_dashboard(state)
    assert "config:" in rendered
    assert "context=120" in rendered
    assert "variant=compact" in rendered


def test_plain_dashboard_shows_val_loss_sparkline(tmp_path: Path):
    log_path = tmp_path / "training.log"
    log_path.write_text("log\n", encoding="utf-8")
    state = progress_runner.build_dashboard_state(
        snapshot={"status": "running", "progress_fraction": 0.5, "current_train": {"train_total_avg": 0.1}},
        manifest=None,
        log_path=log_path,
        log_tail=["log"],
        command=["python3", "train.py"],
        child=None,
        started_at=0.0,
        val_loss_history=[0.5, 0.4, 0.45],
    )
    rendered = progress_runner.render_plain_dashboard(state)
    assert "val loss:" in rendered
