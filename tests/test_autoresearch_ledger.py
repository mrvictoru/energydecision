import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoresearch.ledger import ExperimentLedger, LedgerEntry  # noqa: E402


def _entry(run_id: str) -> LedgerEntry:
    return LedgerEntry(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        environment="household",
        benchmark_path="/tmp/bench.json",
        benchmark_sha256="abc",
        candidate_config={"lr": 1e-4},
        baseline_config={"lr": 2e-5},
        diff_from_baseline={"lr": {"old": 2e-5, "new": 1e-4}},
        training_summary={"crashed": False},
        stage_a_passed=True,
        stage_a_reason="ok",
        evaluation_summary={"mean_reward": 1.0},
        eval_summary={"primary_metric_name": "mean_reward", "primary_metric_value": 1.0},
        stage_b_passed=True,
        stage_b_reason="improved",
        decision="keep",
        artifact_dir="/tmp/artifacts",
    )


def test_ledger_entry_serialization_roundtrip():
    e = _entry("r1")
    payload = json.loads(json.dumps(e.__dict__))
    e2 = LedgerEntry.from_dict(payload)
    assert e2.run_id == "r1"
    assert e2.decision == "keep"


def test_ledger_append_and_reload(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = ExperimentLedger(path)
    ledger.append(_entry("r1"))
    ledger.append(_entry("r2"))

    reloaded = ExperimentLedger(path)
    assert len(reloaded.last_n(10)) == 2
    assert reloaded.last_n(1)[0].run_id == "r2"


def test_ledger_summary_dataframe_not_empty(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = ExperimentLedger(path)
    ledger.append(_entry("r1"))
    df = ledger.summary_dataframe()
    assert df.height == 1
    assert "decision" in df.columns


def test_ledger_current_best_none_when_empty(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = ExperimentLedger(path)
    assert ledger.current_best("household") is None
