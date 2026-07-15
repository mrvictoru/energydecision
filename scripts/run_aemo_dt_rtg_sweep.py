from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_rtg_values(text: str) -> list[float]:
    values = [segment.strip() for segment in str(text).split(",")]
    parsed = [float(value) for value in values if value]
    if not parsed:
        raise ValueError("At least one RTG value is required.")
    return parsed


def _rtg_slug(value: float) -> str:
    return format(value, "g").replace("-", "neg_").replace(".", "_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def set_dt_rtg_value(
    evaluation_config: dict[str, Any],
    *,
    rtg_value: float,
    candidate_policy_name: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(evaluation_config))
    matched = False
    for policy in updated.get("policies", []):
        if policy.get("kind") == "dt" and policy.get("name") == candidate_policy_name:
            policy["rtg_value"] = rtg_value
            matched = True
    if not matched:
        raise ValueError(f"Did not find DT policy named {candidate_policy_name!r} in evaluation config.")
    return updated


def extract_candidate_metrics(summary: dict[str, Any], candidate_policy_name: str) -> dict[str, Any]:
    heldout = summary.get("heldout_evaluation", {})
    metrics = next(
        (
            item for item in heldout.get("aggregate_metrics", [])
            if item.get("experiment") == candidate_policy_name
        ),
        None,
    )
    if metrics is None:
        raise ValueError(f"Summary did not include aggregate metrics for {candidate_policy_name!r}.")
    paired = heldout.get("paired_comparisons_vs_reference", {}).get(candidate_policy_name, {})
    return {
        "avg_reward_per_episode": metrics.get("avg_reward_per_episode"),
        "avg_profit_per_episode": metrics.get("avg_profit_per_episode"),
        "avg_fcas_revenue_per_episode": metrics.get("avg_fcas_revenue_per_episode"),
        "avg_degradation_cost_per_episode": metrics.get("avg_degradation_cost_per_episode"),
        "paired_mean_diff_vs_reference": paired.get("mean_diff"),
        "paired_p_value_vs_reference": paired.get("p_value"),
        "reference_policy": heldout.get("reference_policy"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a candidate DT RTG calibration sweep with the autoresearch evaluator.")
    parser.add_argument("--surface-manifest-path", type=Path, required=True, help="DT surface manifest to evaluate.")
    parser.add_argument("--evaluation-config", type=Path, required=True, help="Base evaluator config JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-RTG evaluator outputs and sweep summary files.")
    parser.add_argument(
        "--rtg-values",
        default="0.0,0.5,1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated RTG values to evaluate.",
    )
    parser.add_argument("--candidate-policy-name", default="candidate_dt", help="DT policy name to mutate in the evaluation config.")
    parser.add_argument("--device", default="auto", help="Device forwarded to src/autoresearch_evaluator.py.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    evaluator_script = root / "src" / "autoresearch_evaluator.py"
    base_config = _load_json(args.evaluation_config.resolve())
    rtg_values = parse_rtg_values(args.rtg_values)

    output_dir = args.output_dir.resolve()
    configs_dir = output_dir / "configs"
    rows: list[dict[str, Any]] = []

    for rtg_value in rtg_values:
        rtg_config = set_dt_rtg_value(
            base_config,
            rtg_value=rtg_value,
            candidate_policy_name=args.candidate_policy_name,
        )
        rtg_slug = _rtg_slug(rtg_value)
        rtg_config_path = configs_dir / f"rtg_{rtg_slug}.json"
        _write_json(rtg_config_path, rtg_config)

        rtg_output_dir = output_dir / rtg_slug
        command = [
            sys.executable,
            str(evaluator_script),
            "--surface-manifest-path",
            str(args.surface_manifest_path.resolve()),
            "--evaluation-config",
            str(rtg_config_path),
            "--output-dir",
            str(rtg_output_dir),
            "--device",
            args.device,
        ]
        subprocess.run(command, check=True)

        summary = _load_json(rtg_output_dir / "evaluation_summary.json")
        row = {
            "rtg_value": rtg_value,
            "config_path": str(rtg_config_path),
            "output_dir": str(rtg_output_dir),
            **extract_candidate_metrics(summary, args.candidate_policy_name),
        }
        rows.append(row)
        print(
            "[RTG sweep] "
            f"rtg={rtg_value:g} profit={row.get('avg_profit_per_episode')} "
            f"paired_mean_diff={row.get('paired_mean_diff_vs_reference')}"
        )

    summary_json_path = output_dir / "rtg_sweep_summary.json"
    _write_json(
        summary_json_path,
        {
            "surface_manifest_path": str(args.surface_manifest_path.resolve()),
            "evaluation_config_path": str(args.evaluation_config.resolve()),
            "candidate_policy_name": args.candidate_policy_name,
            "rows": rows,
            "best_by_profit": max(rows, key=lambda item: float(item.get("avg_profit_per_episode") or float("-inf"))),
        },
    )

    summary_csv_path = output_dir / "rtg_sweep_summary.csv"
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[RTG sweep] Summary JSON: {summary_json_path}")
    print(f"[RTG sweep] Summary CSV:  {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
