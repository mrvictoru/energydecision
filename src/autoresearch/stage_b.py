from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

try:
    from ..eval_common import load_benchmark
except ImportError:
    from eval_common import load_benchmark


class StageBEvaluator:
    def __init__(self, benchmark_path: str, environment: str):
        self.benchmark_path = str(Path(benchmark_path).resolve())
        self.environment = environment
        self.benchmark = load_benchmark(self.benchmark_path)

    def evaluate(
        self,
        model_path: str,
        model_config: dict[str, Any],
        rtg_value: float,
        return_scale: float,
        output_dir: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        model_cfg_path = out_dir / "model_kwargs.json"
        with model_cfg_path.open("w", encoding="utf-8") as fh:
            json.dump(model_config, fh, indent=2)

        argv = [
            "--benchmark",
            self.benchmark_path,
            "--model-path",
            str(Path(model_path).resolve()),
            "--model-config",
            str(model_cfg_path),
            "--rtg-value",
            str(float(rtg_value)),
            "--return-scale",
            str(float(return_scale)),
            "--output-dir",
            str(out_dir),
            "--device",
            str(device),
        ]

        module_name = "src.eval_household" if self.environment == "household" else "src.eval_aemo"
        eval_module = importlib.import_module(module_name)
        exit_code = eval_module.main(argv)
        if exit_code != 0:
            raise RuntimeError(f"Evaluation failed with exit code {exit_code}")

        metrics_path = out_dir / "eval_metrics.json"
        summary_path = out_dir / "eval_summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise RuntimeError("Evaluation outputs missing")

        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)

        return {"evaluation_summary": metrics, "eval_summary": summary}

    def compare(
        self,
        candidate_summary: dict[str, Any],
        baseline_summary: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if baseline_summary is None:
            return "keep", "first run, no baseline to compare"

        if not bool(candidate_summary.get("guardrails_passed", False)):
            return "discard", "guardrail violation"

        metric_name = str(candidate_summary.get("primary_metric_name", self.benchmark.get("primary_metric", "metric")))
        new_value = candidate_summary.get("primary_metric_value")
        old_value = baseline_summary.get("primary_metric_value") if baseline_summary else None

        if new_value is None or old_value is None:
            return "discard", f"missing metric value for comparison: {metric_name}"

        higher_is_better = bool(self.benchmark.get("higher_is_better", True))
        improved = float(new_value) > float(old_value) if higher_is_better else float(new_value) < float(old_value)

        if improved:
            return "keep", f"improved {metric_name} from {old_value} to {new_value}"
        return "discard", f"no improvement: {metric_name} {new_value} vs {old_value}"
