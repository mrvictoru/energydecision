from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from ..eval_common import load_benchmark
except ImportError:
    from eval_common import load_benchmark
from .config_utils import (
    build_training_cli_args,
    diff_configs,
    load_config,
    validate_mutable_surface,
    write_model_kwargs,
)
from .ledger import ExperimentLedger, LedgerEntry
from .stage_a import StageAScreen
from .stage_b import StageBEvaluator


class AutoresearchRunner:
    def __init__(
        self,
        environment: str,
        benchmark_path: str,
        output_dir: str = "eval_output/autoresearch",
        ledger_path: str = "eval_output/autoresearch/ledger.jsonl",
        device: str = "cpu",
        use_docker: bool = False,
        stage_a_screen: StageAScreen | None = None,
    ):
        self.environment = environment
        self.benchmark_path = str(Path(benchmark_path).resolve())
        self.benchmark = load_benchmark(self.benchmark_path)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ExperimentLedger(ledger_path)
        self.device = device
        self.use_docker = use_docker
        self.stage_a_screen = stage_a_screen or StageAScreen()
        self.stage_b = StageBEvaluator(self.benchmark_path, environment)
        self.repo_root = Path(__file__).resolve().parents[2]

    def _sha256_file(self, path: str | Path) -> str:
        file_path = Path(path).resolve()
        digest = hashlib.sha256()
        with file_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _training_script(self) -> Path:
        if self.environment == "household":
            return self.repo_root / "src" / "pretrain_decision_transformer.py"
        if self.environment == "aemo":
            return self.repo_root / "src" / "pretrain_aemo_decision_transformer.py"
        raise ValueError(f"Unsupported environment: {self.environment}")

    def _run_training(self, args: list[str], timeout: int) -> bool:
        command = args
        if self.use_docker:
            command = ["docker", "compose", "run", "--rm", "autoresearch-train", *args]
        result = subprocess.run(command, cwd=str(self.repo_root), timeout=timeout, check=False)
        return result.returncode != 0

    def _parse_training_summary(self, stage_dir: Path, crashed: bool) -> dict[str, Any]:
        loss_csv = stage_dir / "loss.csv"
        model_path = stage_dir / "model_final.pt"
        checkpoint_path = stage_dir / "checkpoint.pt"

        initial_train_loss = None
        final_train_loss = None
        final_val_loss = None
        epochs_completed = 0

        if loss_csv.is_file():
            with loss_csv.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                epochs_completed = len(rows)
                try:
                    initial_train_loss = float(rows[0].get("train_total", "nan"))
                except ValueError:
                    initial_train_loss = float("nan")
                try:
                    final_train_loss = float(rows[-1].get("train_total", "nan"))
                except ValueError:
                    final_train_loss = float("nan")
                val_raw = rows[-1].get("val_total")
                if val_raw not in (None, ""):
                    try:
                        final_val_loss = float(val_raw)
                    except ValueError:
                        final_val_loss = float("nan")

        divergence_ratio = float("nan")
        if initial_train_loss is not None and final_train_loss is not None and initial_train_loss != 0:
            divergence_ratio = final_train_loss / initial_train_loss

        return {
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "epochs_completed": epochs_completed,
            "divergence_ratio": divergence_ratio,
            "crashed": crashed,
            "checkpoint_path": str(checkpoint_path),
            "model_path": str(model_path),
            "loss_csv_path": str(loss_csv),
        }

    def _build_eval_model_config(self, candidate_config: dict[str, Any]) -> dict[str, Any]:
        stage_tmp = self.output_dir / "_tmp_eval"
        stage_tmp.mkdir(parents=True, exist_ok=True)
        path = write_model_kwargs(candidate_config, self.benchmark, stage_tmp)
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def run_candidate(
        self,
        candidate_config: dict[str, Any],
        baseline_config: dict[str, Any],
        skip_training: bool = False,
        model_path: str | None = None,
    ) -> LedgerEntry:
        diff = diff_configs(baseline_config, candidate_config)
        validate_mutable_surface({k: v["new"] for k, v in diff.items()})

        run_id = self.ledger.next_run_id()
        artifact_dir = self.output_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with (artifact_dir / "candidate_config.json").open("w", encoding="utf-8") as fh:
            json.dump(candidate_config, fh, indent=2)

        timestamp = datetime.now(timezone.utc).isoformat()
        benchmark_sha256 = self._sha256_file(self.benchmark_path)

        if skip_training:
            if not model_path:
                raise ValueError("model_path is required when skip_training=True")
            model_file = Path(model_path).resolve()
            if not model_file.is_file():
                raise FileNotFoundError(f"Model file not found: {model_file}")

            stage_b_dir = artifact_dir / "stage_b"
            stage_b_dir.mkdir(parents=True, exist_ok=True)
            write_model_kwargs(candidate_config, self.benchmark, stage_b_dir)

            eval_result = self.stage_b.evaluate(
                model_path=str(model_file),
                model_config=self._build_eval_model_config(candidate_config),
                rtg_value=float(candidate_config.get("rtg_value", 0.0)),
                return_scale=float(candidate_config.get("return_scale", 1.0)),
                output_dir=str(stage_b_dir),
                device=self.device,
            )

            best_kept = self.ledger.current_best(self.environment)
            baseline_eval_summary = best_kept.eval_summary if best_kept else None
            decision, reason = self.stage_b.compare(eval_result["eval_summary"], baseline_eval_summary)

            entry = LedgerEntry(
                run_id=run_id,
                timestamp=timestamp,
                environment=self.environment,
                benchmark_path=self.benchmark_path,
                benchmark_sha256=benchmark_sha256,
                candidate_config=candidate_config,
                baseline_config=baseline_config,
                diff_from_baseline=diff,
                training_summary={
                    "crashed": False,
                    "epochs_completed": 0,
                    "initial_train_loss": None,
                    "final_train_loss": None,
                    "final_val_loss": None,
                    "divergence_ratio": None,
                    "checkpoint_path": None,
                    "model_path": str(model_file),
                    "loss_csv_path": None,
                    "skipped_training": True,
                },
                stage_a_passed=True,
                stage_a_reason="skipped training",
                evaluation_summary=eval_result["evaluation_summary"],
                eval_summary=eval_result["eval_summary"],
                stage_b_passed=(decision == "keep"),
                stage_b_reason=reason,
                decision=decision,
                artifact_dir=str(artifact_dir),
            )
            self.ledger.append(entry)
            with (artifact_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
                json.dump(entry.__dict__, fh, indent=2)
            return entry

        script_path = self._training_script()

        stage_a_dir = artifact_dir / "stage_a"
        stage_a_dir.mkdir(parents=True, exist_ok=True)
        write_model_kwargs(candidate_config, self.benchmark, stage_a_dir)
        stage_a_args = build_training_cli_args(
            candidate_config,
            self.benchmark,
            str(stage_a_dir),
            str(script_path),
            epochs_override=1,
        )
        stage_a_crashed = self._run_training(stage_a_args, int(self.benchmark.get("stage_a_timeout", 600)))
        stage_a_summary = self._parse_training_summary(stage_a_dir, stage_a_crashed)
        stage_a_passed, stage_a_reason = self.stage_a_screen.screen(stage_a_summary)

        if not stage_a_passed:
            entry = LedgerEntry(
                run_id=run_id,
                timestamp=timestamp,
                environment=self.environment,
                benchmark_path=self.benchmark_path,
                benchmark_sha256=benchmark_sha256,
                candidate_config=candidate_config,
                baseline_config=baseline_config,
                diff_from_baseline=diff,
                training_summary=stage_a_summary,
                stage_a_passed=False,
                stage_a_reason=stage_a_reason,
                evaluation_summary=None,
                eval_summary=None,
                stage_b_passed=None,
                stage_b_reason="stage A rejected",
                decision="stage_a_reject",
                artifact_dir=str(artifact_dir),
            )
            self.ledger.append(entry)
            with (artifact_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
                json.dump(entry.__dict__, fh, indent=2)
            return entry

        stage_b_dir = artifact_dir / "stage_b"
        stage_b_dir.mkdir(parents=True, exist_ok=True)
        write_model_kwargs(candidate_config, self.benchmark, stage_b_dir)
        stage_b_args = build_training_cli_args(
            candidate_config,
            self.benchmark,
            str(stage_b_dir),
            str(script_path),
            epochs_override=None,
        )
        stage_b_crashed = self._run_training(stage_b_args, int(self.benchmark.get("stage_b_timeout", 3600)))
        stage_b_summary = self._parse_training_summary(stage_b_dir, stage_b_crashed)
        if stage_b_crashed:
            entry = LedgerEntry(
                run_id=run_id,
                timestamp=timestamp,
                environment=self.environment,
                benchmark_path=self.benchmark_path,
                benchmark_sha256=benchmark_sha256,
                candidate_config=candidate_config,
                baseline_config=baseline_config,
                diff_from_baseline=diff,
                training_summary=stage_b_summary,
                stage_a_passed=True,
                stage_a_reason=stage_a_reason,
                evaluation_summary=None,
                eval_summary=None,
                stage_b_passed=False,
                stage_b_reason="stage B training crashed",
                decision="crash",
                artifact_dir=str(artifact_dir),
            )
            self.ledger.append(entry)
            with (artifact_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
                json.dump(entry.__dict__, fh, indent=2)
            return entry

        eval_result = self.stage_b.evaluate(
            model_path=stage_b_summary["model_path"],
            model_config=self._build_eval_model_config(candidate_config),
            rtg_value=float(candidate_config.get("rtg_value", 0.0)),
            return_scale=float(candidate_config.get("return_scale", 1.0)),
            output_dir=str(stage_b_dir),
            device=self.device,
        )

        best_kept = self.ledger.current_best(self.environment)
        baseline_eval_summary = best_kept.eval_summary if best_kept else None
        decision, reason = self.stage_b.compare(eval_result["eval_summary"], baseline_eval_summary)

        entry = LedgerEntry(
            run_id=run_id,
            timestamp=timestamp,
            environment=self.environment,
            benchmark_path=self.benchmark_path,
            benchmark_sha256=benchmark_sha256,
            candidate_config=candidate_config,
            baseline_config=baseline_config,
            diff_from_baseline=diff,
            training_summary=stage_b_summary,
            stage_a_passed=True,
            stage_a_reason=stage_a_reason,
            evaluation_summary=eval_result["evaluation_summary"],
            eval_summary=eval_result["eval_summary"],
            stage_b_passed=(decision == "keep"),
            stage_b_reason=reason,
            decision=decision,
            artifact_dir=str(artifact_dir),
        )
        self.ledger.append(entry)

        with (artifact_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(entry.__dict__, fh, indent=2)

        return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one autoresearch candidate cycle")
    parser.add_argument("--environment", required=True, choices=["household", "aemo"])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output-dir", default="eval_output/autoresearch")
    parser.add_argument("--ledger-path", default="eval_output/autoresearch/ledger.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--stage-a-max-divergence", type=float, default=4.0)
    args = parser.parse_args(argv)

    if args.skip_training and not args.model_path:
        raise ValueError("--model-path is required when using --skip-training")

    baseline_config = load_config(args.baseline_config)
    candidate_config = load_config(args.candidate_config)

    runner = AutoresearchRunner(
        environment=args.environment,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        ledger_path=args.ledger_path,
        device=args.device,
        use_docker=args.docker,
        stage_a_screen=StageAScreen(max_divergence_ratio=float(args.stage_a_max_divergence)),
    )

    entry = runner.run_candidate(
        candidate_config=candidate_config,
        baseline_config=baseline_config,
        skip_training=bool(args.skip_training),
        model_path=args.model_path,
    )
    print(json.dumps(entry.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
