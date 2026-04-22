from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class StageAScreen:
    def __init__(
        self,
        max_divergence_ratio: float = 4.0,
        max_final_val_loss: float | None = None,
        require_checkpoint: bool = True,
    ):
        self.max_divergence_ratio = float(max_divergence_ratio)
        self.max_final_val_loss = max_final_val_loss
        self.require_checkpoint = bool(require_checkpoint)

    def screen(self, training_summary: dict[str, Any]) -> tuple[bool, str]:
        if bool(training_summary.get("crashed", False)):
            return False, "training crashed"

        model_path = training_summary.get("model_path")
        if not model_path or not Path(str(model_path)).is_file():
            return False, "model artifact missing"

        checkpoint_path = training_summary.get("checkpoint_path")
        if self.require_checkpoint and (not checkpoint_path or not Path(str(checkpoint_path)).is_file()):
            return False, "checkpoint missing"

        divergence_ratio = training_summary.get("divergence_ratio")
        if divergence_ratio is None:
            return False, "missing divergence ratio"
        ratio = float(divergence_ratio)
        if math.isnan(ratio) or math.isinf(ratio):
            return False, "invalid divergence ratio"
        if ratio > self.max_divergence_ratio:
            return False, f"diverged: ratio {ratio:.4f} > {self.max_divergence_ratio:.4f}"

        final_val_loss = training_summary.get("final_val_loss")
        if final_val_loss is not None:
            val = float(final_val_loss)
            if math.isnan(val) or math.isinf(val):
                return False, "invalid final validation loss"
            if self.max_final_val_loss is not None and val > float(self.max_final_val_loss):
                return False, f"validation loss too high: {val:.6f}"

        return True, "ok"
