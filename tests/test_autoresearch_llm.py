import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoresearch.agent import AutoresearchAgent, AutoresearchError  # noqa: E402
from autoresearch.ledger import ExperimentLedger, LedgerEntry  # noqa: E402
from autoresearch.llm_backend import LLMBackend  # noqa: E402
from autoresearch.prompts import build_system_prompt, build_user_prompt, parse_llm_response  # noqa: E402


class DummyBackend(LLMBackend):
    def __init__(self, text: str):
        self.text = text

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.text


class DummyRunner:
    def __init__(self):
        self.called = False

    def run_candidate(self, candidate_config, baseline_config):
        self.called = True
        return LedgerEntry(
            run_id="x",
            timestamp="t",
            environment="household",
            benchmark_path="b",
            benchmark_sha256="s",
            candidate_config=candidate_config,
            baseline_config=baseline_config,
            diff_from_baseline={},
            training_summary={},
            stage_a_passed=True,
            stage_a_reason="ok",
            evaluation_summary=None,
            eval_summary={"primary_metric_name": "mean_reward", "primary_metric_value": 1.0, "guardrails_passed": True},
            stage_b_passed=True,
            stage_b_reason="ok",
            decision="keep",
            artifact_dir="a",
        )


def _seed_ledger(path: Path):
    ledger = ExperimentLedger(path)
    ledger.append(
        LedgerEntry(
            run_id="seed",
            timestamp="t",
            environment="household",
            benchmark_path="b",
            benchmark_sha256="s",
            candidate_config={"lr": 1e-4},
            baseline_config={"lr": 1e-4},
            diff_from_baseline={},
            training_summary={},
            stage_a_passed=True,
            stage_a_reason="ok",
            evaluation_summary=None,
            eval_summary={"primary_metric_name": "mean_reward", "primary_metric_value": 1.0, "guardrails_passed": True},
            stage_b_passed=True,
            stage_b_reason="ok",
            decision="keep",
            artifact_dir="a",
        )
    )
    return ledger


def test_parse_llm_response_extracts_first_json_block():
    raw = "noise before {\"lr\": 0.0002, \"unknown\": 1} noise after"
    parsed = parse_llm_response(raw, frozenset({"lr"}))
    assert parsed == {"lr": 0.0002}


def test_prompt_builders_smoke():
    s = build_system_prompt(frozenset({"lr"}), {"lr": {"type": "float", "description": "learning rate"}})
    u = build_user_prompt({"lr": 1e-4}, 1.2, "mean_reward", "- Run x", "guardrails")
    assert "Allowed keys" in s
    assert "Current best config" in u


def test_agent_step_runs_with_valid_diff(tmp_path: Path):
    ledger = _seed_ledger(tmp_path / "ledger.jsonl")
    backend = DummyBackend('{"lr": 0.0003}')
    runner = DummyRunner()

    agent = AutoresearchAgent(
        backend=backend,
        runner=runner,
        ledger=ledger,
        environment="household",
    )
    entry = agent.step()
    assert runner.called is True
    assert entry.candidate_config["lr"] == 0.0003


def test_agent_propose_fails_without_valid_json(tmp_path: Path):
    ledger = _seed_ledger(tmp_path / "ledger.jsonl")
    backend = DummyBackend("not-json")
    runner = DummyRunner()
    agent = AutoresearchAgent(
        backend=backend,
        runner=runner,
        ledger=ledger,
        environment="household",
        max_llm_retries=1,
    )

    with pytest.raises(AutoresearchError):
        agent.propose()
