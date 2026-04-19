from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class LedgerEntry:
    run_id: str
    timestamp: str
    environment: str
    benchmark_path: str
    benchmark_sha256: str
    candidate_config: dict[str, Any]
    baseline_config: dict[str, Any]
    diff_from_baseline: dict[str, Any]
    training_summary: dict[str, Any]
    stage_a_passed: bool
    stage_a_reason: str
    evaluation_summary: dict[str, Any] | None
    eval_summary: dict[str, Any] | None
    stage_b_passed: bool | None
    stage_b_reason: str
    decision: str
    artifact_dir: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LedgerEntry":
        return cls(**payload)


class ExperimentLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._entries: list[LedgerEntry] = []
        self._load()

    def _load(self) -> None:
        self._entries = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                self._entries.append(LedgerEntry.from_dict(json.loads(stripped)))

    def append(self, entry: LedgerEntry) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), ensure_ascii=True) + "\n")
        self._entries.append(entry)

    def last_n(self, n: int) -> list[LedgerEntry]:
        if n <= 0:
            return []
        return self._entries[-n:]

    def current_best(self, environment: str) -> LedgerEntry | None:
        for entry in reversed(self._entries):
            if entry.environment == environment and entry.decision == "keep":
                return entry
        return None

    def format_history(self, last_n: int = 10) -> str:
        lines: list[str] = []
        for entry in self.last_n(last_n):
            metric_name = None
            metric_value = None
            if entry.eval_summary:
                metric_name = entry.eval_summary.get("primary_metric_name")
                metric_value = entry.eval_summary.get("primary_metric_value")
            diff_keys = sorted(entry.diff_from_baseline.keys())
            changed = ", ".join(diff_keys[:5]) if diff_keys else "no-change"
            lines.append(
                f"- Run {entry.run_id}: {changed} -> {metric_name}={metric_value}, {entry.decision.upper()}"
            )
        return "\n".join(lines)

    def next_run_id(self) -> str:
        return str(uuid.uuid4())

    def summary_dataframe(self) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for e in self._entries:
            primary_name = None
            primary_value = None
            if e.eval_summary:
                primary_name = e.eval_summary.get("primary_metric_name")
                primary_value = e.eval_summary.get("primary_metric_value")
            rows.append(
                {
                    "run_id": e.run_id,
                    "timestamp": e.timestamp,
                    "environment": e.environment,
                    "decision": e.decision,
                    "stage_a_passed": e.stage_a_passed,
                    "stage_b_passed": e.stage_b_passed,
                    "primary_metric": primary_name,
                    "primary_value": primary_value,
                }
            )
        return pl.DataFrame(rows) if rows else pl.DataFrame(schema={
            "run_id": pl.String,
            "timestamp": pl.String,
            "environment": pl.String,
            "decision": pl.String,
            "stage_a_passed": pl.Boolean,
            "stage_b_passed": pl.Boolean,
            "primary_metric": pl.String,
            "primary_value": pl.Float64,
        })

    def to_tsv(self, path: str) -> None:
        self.summary_dataframe().write_csv(path, separator="\t")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experiment ledger utility")
    parser.add_argument("--summary", type=str, default=None, help="Path to ledger JSONL")
    args = parser.parse_args(argv)

    if not args.summary:
        parser.print_help()
        return 0

    ledger = ExperimentLedger(args.summary)
    df = ledger.summary_dataframe()
    if df.height == 0:
        print("Ledger is empty")
        return 0

    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
