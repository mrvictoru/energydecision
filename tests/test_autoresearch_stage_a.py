import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoresearch.stage_a import StageAScreen  # noqa: E402


def test_stage_a_passes_on_valid_summary(tmp_path: Path):
    model = tmp_path / "model.pt"
    ckpt = tmp_path / "ckpt.pt"
    model.write_bytes(b"m")
    ckpt.write_bytes(b"c")

    screen = StageAScreen(max_divergence_ratio=4.0, max_final_val_loss=10.0)
    ok, reason = screen.screen(
        {
            "crashed": False,
            "model_path": str(model),
            "checkpoint_path": str(ckpt),
            "divergence_ratio": 1.5,
            "final_val_loss": 2.0,
        }
    )
    assert ok is True
    assert reason == "ok"


def test_stage_a_rejects_diverged_or_invalid(tmp_path: Path):
    model = tmp_path / "model.pt"
    ckpt = tmp_path / "ckpt.pt"
    model.write_bytes(b"m")
    ckpt.write_bytes(b"c")

    screen = StageAScreen(max_divergence_ratio=2.0)
    ok, reason = screen.screen(
        {
            "crashed": False,
            "model_path": str(model),
            "checkpoint_path": str(ckpt),
            "divergence_ratio": 3.0,
            "final_val_loss": 1.0,
        }
    )
    assert ok is False
    assert "diverged" in reason
