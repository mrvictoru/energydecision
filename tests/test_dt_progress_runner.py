import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dt_progress_runner import render_dashboard  # noqa: E402


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

    dashboard = render_dashboard(
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
