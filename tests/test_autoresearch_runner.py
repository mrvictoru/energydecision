import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoresearch.runner import AutoresearchRunner  # noqa: E402


def _write_household_benchmark(path: Path) -> None:
    benchmark = {
        "environment": "household",
        "data_dir": "data/household/logs",
        "state_dim": 12,
        "act_dim": 1,
        "max_timestep": 100,
        "discount": 0.99,
        "primary_metric": "mean_reward",
        "higher_is_better": True,
        "guardrails": {},
    }
    path.write_text(json.dumps(benchmark), encoding="utf-8")


def test_runner_skip_training_evaluates_model_and_records_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bench_path = tmp_path / "benchmark.json"
    _write_household_benchmark(bench_path)

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"placeholder")

    runner = AutoresearchRunner(
        environment="household",
        benchmark_path=str(bench_path),
        output_dir=str(tmp_path / "out"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
    )

    def _fail_training(*_args, **_kwargs):
        raise AssertionError("_run_training should not be called when skip_training=True")

    monkeypatch.setattr(runner, "_run_training", _fail_training)

    fake_eval = {
        "evaluation_summary": {"mean_reward": 1.23},
        "eval_summary": {
            "primary_metric_name": "mean_reward",
            "primary_metric_value": 1.23,
            "guardrails_passed": True,
        },
    }
    monkeypatch.setattr(runner.stage_b, "evaluate", lambda **_kwargs: fake_eval)

    entry = runner.run_candidate(
        candidate_config={"lr": 1e-4},
        baseline_config={"lr": 1e-4},
        skip_training=True,
        model_path=str(model_path),
    )

    assert entry.stage_a_passed is True
    assert entry.stage_a_reason == "skipped training"
    assert entry.training_summary["skipped_training"] is True
    assert entry.training_summary["epochs_completed"] == 0
    assert entry.training_summary["model_path"] == str(model_path.resolve())
    assert entry.eval_summary["primary_metric_value"] == 1.23


def test_runner_skip_training_requires_existing_model(tmp_path: Path):
    bench_path = tmp_path / "benchmark.json"
    _write_household_benchmark(bench_path)

    runner = AutoresearchRunner(
        environment="household",
        benchmark_path=str(bench_path),
        output_dir=str(tmp_path / "out"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
    )

    with pytest.raises(FileNotFoundError):
        runner.run_candidate(
            candidate_config={"lr": 1e-4},
            baseline_config={"lr": 1e-4},
            skip_training=True,
            model_path=str(tmp_path / "missing_model.pt"),
        )
