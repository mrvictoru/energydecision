import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_common import EvalSummary, check_guardrails, load_benchmark, read_return_scale, write_eval_outputs


def test_load_benchmark_resolves_relative_paths(tmp_path: Path):
    cfg = {
        "environment": "household",
        "data_dir": "data/household/logs",
        "guardrails": {},
    }
    cfg_path = tmp_path / "bench.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_benchmark(str(cfg_path))

    assert loaded["benchmark_path"] == str(cfg_path.resolve())
    assert loaded["data_dir"] == str((tmp_path / "data/household/logs").resolve())


def test_read_return_scale_prefers_cli_override(tmp_path: Path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"placeholder")
    sidecar = Path(str(model) + ".meta.json")
    sidecar.write_text(json.dumps({"return_scale": 123.0}), encoding="utf-8")

    assert read_return_scale(str(model), 3.5) == 3.5


def test_read_return_scale_uses_sidecar_or_default(tmp_path: Path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"placeholder")

    assert read_return_scale(str(model), None) == 1.0

    sidecar = Path(str(model) + ".meta.json")
    sidecar.write_text(json.dumps({"return_scale": 7.25}), encoding="utf-8")
    assert read_return_scale(str(model), None) == 7.25


def test_check_guardrails_marks_missing_metrics_as_failure():
    metrics = {"mean_reward": 1.0, "var_5": -4000.0}
    guardrails = {
        "max_var_5": -3000.0,
        "max_avg_degradation_per_episode": 0.05,
    }

    result = check_guardrails(metrics, guardrails)

    assert result["passed"] is False
    assert result["details"]["max_var_5"]["passed"] is True
    assert result["details"]["max_avg_degradation_per_episode"]["passed"] is False


def test_check_guardrails_passes_when_all_ok():
    metrics = {"mean_reward": 10.0, "var_5": -1200.0}
    guardrails = {
        "max_var_5": -1000.0,
        "min_mean_reward": 9.0,
    }
    result = check_guardrails(metrics, guardrails)
    assert result["passed"] is True
    assert result["details"]["max_var_5"]["passed"] is True
    assert result["details"]["min_mean_reward"]["passed"] is True


def test_write_eval_outputs_creates_json_files(tmp_path: Path):
    summary = EvalSummary(
        primary_metric_name="mean_reward",
        primary_metric_value=12.3,
        guardrails_passed=True,
        guardrail_details={"max_var_5": {"passed": True}},
        model_path="/tmp/model.pt",
        benchmark_path="/tmp/benchmark.json",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    metrics = {"mean_reward": 12.3, "var_5": -1000.0}

    write_eval_outputs(str(tmp_path), metrics, summary)

    metrics_path = tmp_path / "eval_metrics.json"
    summary_path = tmp_path / "eval_summary.json"
    assert metrics_path.is_file()
    assert summary_path.is_file()

    loaded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert loaded_metrics["mean_reward"] == 12.3
    assert loaded_summary["primary_metric_name"] == "mean_reward"
