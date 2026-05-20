from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from aemo_notebook_utils import (
    fetch_and_preprocess_aemo_scenarios,
    fit_aemo_global_stats,
    preflight_processed_cache_paths,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prewarm and validate AEMO processed cache files for an evaluator config before training comparisons.",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        required=True,
        help="JSON config describing the held-out AEMO scenarios to prewarm.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON output path for the prewarm manifest.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force cache refresh even when processed parquet files already exist.",
    )
    return parser.parse_args(argv)


def _parse_datetime(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromisoformat(f"{text}T00:00:00")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.evaluation_config.resolve()
    config = _load_json(config_path)
    heldout_cfg = dict(config.get("heldout", {}))
    scenarios_raw = heldout_cfg.get("scenarios")
    if not scenarios_raw:
        raise ValueError("evaluation_config.heldout.scenarios must be non-empty.")

    scenarios = [
        {
            **scenario,
            "start_date": _parse_datetime(scenario["start_date"]),
            "end_date": _parse_datetime(scenario["end_date"]),
        }
        for scenario in scenarios_raw
    ]
    cache_dir = (repo_root() / str(heldout_cfg.get("cache_dir", "data/aemo"))).resolve()
    step_duration = float(heldout_cfg.get("step_duration", 0.5))
    refresh = bool(args.refresh or heldout_cfg.get("refresh", False))
    fixed_stats = heldout_cfg.get("fixed_stats")
    if fixed_stats is None and bool(heldout_cfg.get("fit_global_stats", True)):
        fixed_stats, _ = fit_aemo_global_stats(
            scenarios=scenarios,
            cache_dir=cache_dir,
            step_duration=step_duration,
            refresh=refresh,
        )

    cache_preflight = preflight_processed_cache_paths(
        scenarios=scenarios,
        cache_dir=cache_dir,
        step_duration=step_duration,
        refresh=refresh,
        fixed_stats=fixed_stats,
    )
    processed_by_label, scenario_manifest = fetch_and_preprocess_aemo_scenarios(
        scenarios=scenarios,
        cache_dir=cache_dir,
        step_duration=step_duration,
        refresh=refresh,
        fixed_stats=fixed_stats,
    )
    output_path = (
        args.output_path.resolve()
        if args.output_path is not None
        else repo_root()
        / "eval_output"
        / "aemo_cache_prewarm"
        / f"{config_path.stem}_cache_manifest.json"
    )
    manifest = {
        "schema": "energydecision.aemo_cache_prewarm.v1",
        "evaluation_config_path": str(config_path),
        "cache_dir": str(cache_dir),
        "step_duration": step_duration,
        "refresh": refresh,
        "cache_preflight": cache_preflight,
        "scenario_manifest": [
            {
                **entry,
                "start_date": str(entry["start_date"]),
                "end_date": str(entry["end_date"]),
                "row_count": int(processed_by_label[entry["label"]].height),
            }
            for entry in scenario_manifest
        ],
    }
    _write_json(output_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
