from __future__ import annotations

import argparse
import os
from pathlib import Path

from autoresearch.agent import AutoresearchAgent
from autoresearch.config_utils import load_config
from autoresearch.ledger import ExperimentLedger, LedgerEntry
from autoresearch.llm_backend import LlamaCppBackend, OllamaBackend, OpenAIBackend
from autoresearch.runner import AutoresearchRunner


def _build_backend(args: argparse.Namespace):
    backend_name = str(args.llm_backend).strip().lower()
    common = {
        "endpoint": args.llm_endpoint,
        "model": args.llm_model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if backend_name == "llamacpp":
        if not common["endpoint"]:
            common["endpoint"] = "http://localhost:8080/v1"
        return LlamaCppBackend(**common)
    if backend_name == "ollama":
        if not common["endpoint"]:
            common["endpoint"] = "http://localhost:11434/v1"
        return OllamaBackend(**common)
    if backend_name == "openai":
        if not common["endpoint"]:
            common["endpoint"] = "https://api.openai.com/v1"
        return OpenAIBackend(api_key=args.llm_api_key, **common)

    raise ValueError(f"Unsupported llm backend: {args.llm_backend}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autoresearch CLI")
    parser.add_argument("--mode", default="agent", choices=["agent", "manual"])
    parser.add_argument("--environment", required=True, choices=["household", "aemo"])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--llm-backend", default="llamacpp", choices=["llamacpp", "ollama", "openai"])
    parser.add_argument("--llm-endpoint", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--primary-metric", default="")
    parser.add_argument("--constraint", default="")
    parser.add_argument("--ledger-path", default="eval_output/autoresearch/ledger.jsonl")
    parser.add_argument("--output-dir", default="eval_output/autoresearch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--docker", action="store_true")

    args = parser.parse_args(argv)

    runner = AutoresearchRunner(
        environment=args.environment,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        ledger_path=args.ledger_path,
        device=args.device,
        use_docker=args.docker,
    )

    if args.mode == "manual":
        if not args.candidate_config:
            raise ValueError("--candidate-config is required in manual mode")
        baseline = load_config(args.baseline_config)
        candidate = load_config(args.candidate_config)
        entry = runner.run_candidate(candidate_config=candidate, baseline_config=baseline)
        print(entry)
        return 0

    ledger = ExperimentLedger(args.ledger_path)

    if ledger.current_best(args.environment) is None:
        baseline = load_config(args.baseline_config)
        bootstrap_entry = LedgerEntry(
            run_id=ledger.next_run_id(),
            timestamp="bootstrap",
            environment=args.environment,
            benchmark_path=str(Path(args.benchmark).resolve()),
            benchmark_sha256="",
            candidate_config=baseline,
            baseline_config=baseline,
            diff_from_baseline={},
            training_summary={},
            stage_a_passed=True,
            stage_a_reason="bootstrap",
            evaluation_summary=None,
            eval_summary={
                "primary_metric_name": args.primary_metric or "metric",
                "primary_metric_value": 0.0,
                "guardrails_passed": True,
            },
            stage_b_passed=True,
            stage_b_reason="bootstrap",
            decision="keep",
            artifact_dir=str(Path(args.output_dir).resolve()),
        )
        ledger.append(bootstrap_entry)

    backend = _build_backend(args)
    agent = AutoresearchAgent(
        backend=backend,
        runner=runner,
        ledger=ledger,
        environment=args.environment,
        primary_metric=args.primary_metric or "mean_reward",
        constraint_clause=args.constraint,
    )
    agent.run(args.iterations)
    return 0
