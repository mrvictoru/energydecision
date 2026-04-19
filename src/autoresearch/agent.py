from __future__ import annotations

from dataclasses import dataclass

from .config_utils import ALLOWED_MUTABLE_KEYS_V1, diff_configs
from .ledger import ExperimentLedger, LedgerEntry
from .llm_backend import LLMBackend
from .prompts import ParseError, build_system_prompt, build_user_prompt, parse_llm_response
from .runner import AutoresearchRunner


class AutoresearchError(RuntimeError):
    pass


@dataclass
class _MetricContext:
    metric_name: str
    metric_value: float | str


class AutoresearchAgent:
    def __init__(
        self,
        backend: LLMBackend,
        runner: AutoresearchRunner,
        ledger: ExperimentLedger,
        environment: str,
        primary_metric: str = "mean_reward",
        higher_is_better: bool = True,
        max_llm_retries: int = 3,
        history_window: int = 10,
        constraint_clause: str = "",
    ):
        self.backend = backend
        self.runner = runner
        self.ledger = ledger
        self.environment = environment
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better
        self.max_llm_retries = int(max_llm_retries)
        self.history_window = int(history_window)
        self.constraint_clause = constraint_clause

        self.mutable_params = {k: {"type": "any"} for k in ALLOWED_MUTABLE_KEYS_V1}

    def _best_entry(self) -> LedgerEntry | None:
        return self.ledger.current_best(self.environment)

    def _best_context(self) -> tuple[dict, _MetricContext]:
        best = self._best_entry()
        if best is None:
            raise AutoresearchError("No kept baseline in ledger; run manual baseline first")
        metric_name = self.primary_metric
        metric_value: float | str = "n/a"
        if best.eval_summary:
            metric_name = str(best.eval_summary.get("primary_metric_name", self.primary_metric))
            metric_value = best.eval_summary.get("primary_metric_value", "n/a")
        return best.candidate_config, _MetricContext(metric_name=metric_name, metric_value=metric_value)

    def propose(self) -> dict:
        best_config, metric_ctx = self._best_context()
        history = self.ledger.format_history(self.history_window)
        system_prompt = build_system_prompt(ALLOWED_MUTABLE_KEYS_V1, self.mutable_params)
        user_prompt = build_user_prompt(
            best_config=best_config,
            best_metric=metric_ctx.metric_value,
            metric_name=metric_ctx.metric_name,
            history_lines=history,
            constraint_clause=self.constraint_clause,
        )

        last_error: Exception | None = None
        for _ in range(self.max_llm_retries):
            raw = self.backend.complete(system_prompt, user_prompt)
            try:
                return parse_llm_response(raw, ALLOWED_MUTABLE_KEYS_V1)
            except ParseError as exc:
                last_error = exc
                continue

        raise AutoresearchError(f"Failed to parse LLM response after retries: {last_error}")

    def step(self) -> LedgerEntry:
        best = self._best_entry()
        if best is None:
            raise AutoresearchError("No baseline keep-entry available in ledger")

        candidate_diff = self.propose()
        candidate_config = dict(best.candidate_config)
        candidate_config.update(candidate_diff)

        _ = diff_configs(best.candidate_config, candidate_config)
        return self.runner.run_candidate(candidate_config=candidate_config, baseline_config=best.candidate_config)

    def run(self, iterations: int) -> None:
        for i in range(int(iterations)):
            entry = self.step()
            print(f"[{i+1}/{iterations}] decision={entry.decision} reason={entry.stage_b_reason}")
