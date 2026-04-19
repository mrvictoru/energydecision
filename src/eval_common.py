from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import torch

try:
    from .decision_transformer import DecisionTransformer
except ImportError:
    from decision_transformer import DecisionTransformer


@dataclass
class EvalSummary:
    primary_metric_name: str
    primary_metric_value: float | None
    guardrails_passed: bool
    guardrail_details: dict[str, dict[str, Any]]
    model_path: str
    benchmark_path: str
    timestamp: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value


def _resolve_path(base_dir: Path, maybe_path: Any) -> Any:
    if not isinstance(maybe_path, str):
        return maybe_path
    candidate = Path(maybe_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def load_benchmark(path: str) -> dict[str, Any]:
    benchmark_path = Path(path).resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark not found: {benchmark_path}")

    with benchmark_path.open("r", encoding="utf-8") as fh:
        benchmark = json.load(fh)

    benchmark["benchmark_path"] = str(benchmark_path)

    base_dir = benchmark_path.parent
    for key in ("data_dir", "dataset_path", "manifest_path"):
        if key in benchmark:
            benchmark[key] = _resolve_path(base_dir, benchmark[key])

    if isinstance(benchmark.get("scenario_kwargs"), dict) and "cache_dir" in benchmark["scenario_kwargs"]:
        benchmark["scenario_kwargs"]["cache_dir"] = _resolve_path(base_dir, benchmark["scenario_kwargs"]["cache_dir"])

    return benchmark


def load_dt_model(model_path: str, model_config: dict[str, Any], device: str):
    model_file = Path(model_path).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model = DecisionTransformer(**model_config)
    model.load_from_checkpoint(str(model_file), map_location=device, strict=False)
    model.to(torch.device(device))
    model.eval()
    return model


def read_return_scale(model_path: str, cli_override: float | None) -> float:
    if cli_override is not None:
        return float(cli_override)

    sidecar = Path(str(model_path) + ".meta.json")
    if sidecar.is_file():
        with sidecar.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        value = payload.get("return_scale")
        if value is not None:
            return float(value)

    return 1.0


def check_guardrails(metrics: dict[str, Any], guardrails: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, dict[str, Any]] = {}
    passed = True

    for key, threshold in guardrails.items():
        metric_key = key
        if key.startswith("max_"):
            metric_key = key[len("max_") :]
        elif key.startswith("min_"):
            metric_key = key[len("min_") :]

        value = metrics.get(metric_key)
        item_passed = False

        if value is not None:
            if key.startswith("max_"):
                item_passed = float(value) <= float(threshold)
            elif key.startswith("min_"):
                item_passed = float(value) >= float(threshold)
            else:
                item_passed = float(value) <= float(threshold)

        if not item_passed:
            passed = False

        details[key] = {
            "metric": metric_key,
            "value": value,
            "threshold": threshold,
            "passed": item_passed,
        }

    return {"passed": passed, "details": details}


def write_eval_outputs(output_dir: str, metrics: dict[str, Any], summary: EvalSummary) -> None:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "eval_metrics.json"
    summary_path = out_dir / "eval_summary.json"

    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(metrics), fh, indent=2)

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(asdict(summary)), fh, indent=2)


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_parquet_by_patterns(data_dir: str, patterns: list[str]) -> list[Path]:
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    all_parquet = sorted(root.glob("*.parquet"))
    seen: set[Path] = set()
    ordered: list[Path] = []

    for pattern in patterns:
        current: list[Path]
        if any(ch in pattern for ch in "*?[]"):
            current = sorted(root.glob(pattern))
        else:
            current = [p for p in all_parquet if pattern in p.name]

        for path in current:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)

    return ordered


def split_episode_logs(df: pl.DataFrame) -> list[pl.DataFrame]:
    if "episode_id" in df.columns:
        ids = df.get_column("episode_id").unique().to_list()
        return [
            df.filter(pl.col("episode_id") == ep_id).sort("step") if "step" in df.columns else df.filter(pl.col("episode_id") == ep_id)
            for ep_id in ids
        ]
    if "step" in df.columns:
        return [df.sort("step")]
    return [df]
