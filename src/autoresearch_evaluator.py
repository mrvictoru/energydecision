from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl
import torch

from aemo_notebook_utils import (
    build_dispatch_selection,
    create_aemo_env,
    fetch_and_preprocess_aemo_scenarios,
    fit_aemo_global_stats,
    preflight_processed_cache_paths,
    resolve_battery_variants,
    resolve_dispatch_battery_life_cost,
    run_rule_episodes,
    run_sb3_episodes,
    should_run_dispatch_for_scenario,
)
from decision import AEMOAgent
from decision_transformer import DecisionTransformer
from dispatch_utils import run_dispatch_replay
from helper import (
    bootstrap_confidence_intervals,
    evaluate_by_conditions,
    evaluate_experiments,
    paired_comparison,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DT autoresearch checkpoints with validation-loss summaries and held-out AEMO rollouts.",
    )
    parser.add_argument(
        "--surface-manifest-path",
        type=Path,
        required=True,
        help="Path to the DT training surface manifest written next to the loss CSV.",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        required=True,
        help="JSON config describing held-out scenarios, battery variants, and baseline policies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluator summaries, metrics tables, plots, and logs will be written.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional override for the DT weights/checkpoint path. Defaults to the surface manifest save_path.",
    )
    parser.add_argument(
        "--loss-csv-path",
        type=Path,
        default=None,
        help="Optional override for the DT loss CSV path. Defaults to the surface manifest loss_csv_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device for DT rollouts: auto, cpu, cuda, cuda:0, ...",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _parse_datetime(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("Datetime value must be non-empty.")
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.fromisoformat(f"{normalized}T00:00:00")


def _resolve_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def summarize_loss_history(loss_csv_path: Path) -> dict[str, Any]:
    if not loss_csv_path.is_file():
        raise FileNotFoundError(f"Loss CSV not found: {loss_csv_path}")

    with loss_csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        raise ValueError(f"Loss CSV has no rows: {loss_csv_path}")

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed_rows.append(
            {
                "epoch": int(row["epoch"]),
                "train_total": _maybe_float(row.get("train_total")),
                "train_action": _maybe_float(row.get("train_action")),
                "train_state": _maybe_float(row.get("train_state")),
                "train_return": _maybe_float(row.get("train_return")),
                "val_total": _maybe_float(row.get("val_total")),
                "val_action": _maybe_float(row.get("val_action")),
                "val_state": _maybe_float(row.get("val_state")),
                "val_return": _maybe_float(row.get("val_return")),
            }
        )

    final_row = parsed_rows[-1]
    val_rows = [row for row in parsed_rows if row["val_total"] is not None]
    best_val_row = min(val_rows, key=lambda row: row["val_total"]) if val_rows else None

    return {
        "loss_csv_path": str(loss_csv_path),
        "epochs_recorded": len(parsed_rows),
        "final_epoch": final_row["epoch"],
        "final_train_total_loss": final_row["train_total"],
        "final_val_total_loss": final_row["val_total"],
        "best_val_epoch": best_val_row["epoch"] if best_val_row is not None else None,
        "best_val_total_loss": best_val_row["val_total"] if best_val_row is not None else None,
        "best_val_action_loss": best_val_row["val_action"] if best_val_row is not None else None,
        "best_val_state_loss": best_val_row["val_state"] if best_val_row is not None else None,
        "best_val_return_loss": best_val_row["val_return"] if best_val_row is not None else None,
    }


def load_dt_model(surface_manifest: dict[str, Any], *, model_path: Path | None, device: str) -> tuple[DecisionTransformer, Path, str]:
    resolved_device = _resolve_device(device)
    model_kwargs = dict(surface_manifest["model_kwargs"])
    weights_path = Path(model_path or surface_manifest["paths"]["save_path"]).resolve()
    model = DecisionTransformer(**model_kwargs)
    model.load_from_checkpoint(str(weights_path), map_location=resolved_device)
    model.to(resolved_device)
    model.eval()
    return model, weights_path, resolved_device


def _coerce_info_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def compute_info_signal_metrics(logs: list[pl.DataFrame]) -> dict[str, Any]:
    violation_counts: dict[str, int] = {}
    clip_counts: dict[str, int] = {}
    violation_episode_count = 0
    deg_incident_episode_count = 0
    total_steps = 0

    for episode in logs:
        episode_had_violation = False
        episode_had_deg_incident = False
        infos = episode.get_column("info").to_list() if "info" in episode.columns else []
        total_steps += int(episode.height)
        for raw_info in infos:
            info = _coerce_info_dict(raw_info)
            for key, raw_value in info.items():
                key_lower = str(key).lower()
                value_bool = bool(raw_value)
                if "violation" in key_lower and value_bool:
                    violation_counts[key_lower] = violation_counts.get(key_lower, 0) + 1
                    episode_had_violation = True
                if ("clip" in key_lower or "clamp" in key_lower) and value_bool:
                    clip_counts[key_lower] = clip_counts.get(key_lower, 0) + 1
                if key_lower == "deg_incident" and value_bool:
                    episode_had_deg_incident = True
        if episode_had_violation:
            violation_episode_count += 1
        if episode_had_deg_incident:
            deg_incident_episode_count += 1

    violation_step_count = int(sum(violation_counts.values()))
    clip_step_count = int(sum(clip_counts.values()))
    episode_count = len(logs)
    return {
        "episodes_evaluated": episode_count,
        "total_steps": total_steps,
        "violation_step_count": violation_step_count,
        "violation_step_rate": (violation_step_count / total_steps) if total_steps > 0 else 0.0,
        "violation_episode_count": violation_episode_count,
        "violation_episode_rate": (violation_episode_count / episode_count) if episode_count > 0 else 0.0,
        "deg_incident_episode_count": deg_incident_episode_count,
        "deg_incident_episode_rate": (deg_incident_episode_count / episode_count) if episode_count > 0 else 0.0,
        "clip_step_count": clip_step_count,
        "clip_step_rate": (clip_step_count / total_steps) if total_steps > 0 else 0.0,
        "violation_counts": violation_counts,
        "clip_counts": clip_counts,
    }


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        den = float(denominator)
        if abs(den) < 1e-12:
            return None
        return float(numerator) / den
    except (TypeError, ValueError):
        return None


def default_aemo_conditions(config: dict[str, Any] | None = None) -> dict[str, Callable[..., bool]]:
    cfg = config or {}
    negative_price_threshold = float(cfg.get("negative_price_threshold", 0.0))
    price_spike_threshold = float(cfg.get("price_spike_threshold", 150.0))
    low_soc_ratio = float(cfg.get("low_soc_ratio", 0.2))
    high_soc_ratio = float(cfg.get("high_soc_ratio", 0.8))

    return {
        "negative_price": lambda obs, info=None: info is not None and float(info.get("energy_price", 0.0)) < negative_price_threshold,
        "price_spike": lambda obs, info=None: info is not None and float(info.get("energy_price", 0.0)) >= price_spike_threshold,
        "low_soc": lambda obs, info=None: (
            lambda ratio: ratio is not None and ratio <= low_soc_ratio
        )(_safe_ratio((info or {}).get("battery_soc"), (info or {}).get("capacity_mwh"))),
        "high_soc": lambda obs, info=None: (
            lambda ratio: ratio is not None and ratio >= high_soc_ratio
        )(_safe_ratio((info or {}).get("battery_soc"), (info or {}).get("capacity_mwh"))),
    }


def _resolve_max_step(*, processed_rows: int, step_duration: float, max_step: int | None, episode_hours: float | None) -> int:
    if max_step is not None:
        return max(1, int(max_step))
    if episode_hours is None:
        return max(1, int(processed_rows))
    return max(1, int(round(float(episode_hours) / float(step_duration))))


def _with_episode_metadata(
    episodes: Sequence[pl.DataFrame],
    *,
    policy_name: str,
    scenario_label: str,
    battery_label: str,
) -> list[pl.DataFrame]:
    tagged: list[pl.DataFrame] = []
    for idx, episode in enumerate(episodes):
        tagged.append(
            episode.with_columns(
                [
                    pl.lit(policy_name).alias("policy_name"),
                    pl.lit(scenario_label).alias("scenario_label"),
                    pl.lit(battery_label).alias("battery_label"),
                    pl.lit(idx).alias("heldout_episode_index"),
                ]
            )
        )
    return tagged


def _combine_episode_logs(episodes: Sequence[pl.DataFrame]) -> pl.DataFrame:
    if not episodes:
        return pl.DataFrame({"episode_id": []})
    return pl.concat(
        [episode.with_columns(pl.lit(idx).alias("episode_id")) for idx, episode in enumerate(episodes)],
        how="diagonal_relaxed",
    )


def _dt_rtg_value(policy_cfg: dict[str, Any], training_summary: dict[str, Any]) -> float:
    if "rtg_value" in policy_cfg:
        return float(policy_cfg["rtg_value"])
    best_val = training_summary.get("best_val_total_loss")
    return 0.0 if best_val is None else float(best_val)


def run_dt_episodes(
    *,
    processed_data: pl.DataFrame,
    battery_variant: dict[str, Any],
    model: DecisionTransformer,
    num_episodes: int,
    max_step: int,
    step_duration: float,
    action_mode: str,
    degradation_mode: str,
    degradation_chemistry: str,
    degradation_temperature: float,
    random_episode_start: bool,
    rtg_value: float,
    base_seed: int,
) -> list[pl.DataFrame]:
    episodes: list[pl.DataFrame] = []
    for episode_idx in range(num_episodes):
        env = create_aemo_env(
            processed_data=processed_data,
            battery_variant=battery_variant,
            max_step=max_step,
            step_duration=step_duration,
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
            random_episode_start=random_episode_start,
        )
        agent = AEMOAgent(
            env,
            algorithm="dt",
            model=model,
            rtg_value=rtg_value,
            reset_seed=base_seed + episode_idx if random_episode_start else None,
        )
        episode_df, _ = agent.run_episode()
        episodes.append(episode_df)
    return episodes


def run_policy_episodes(
    *,
    policy_cfg: dict[str, Any],
    processed_data: pl.DataFrame,
    scenario: dict[str, Any],
    battery_variant: dict[str, Any],
    heldout_cfg: dict[str, Any],
    training_summary: dict[str, Any],
    dt_model: DecisionTransformer | None,
) -> list[pl.DataFrame]:
    policy_kind = str(policy_cfg["kind"]).lower()
    episodes_per_variant = int(policy_cfg.get("episodes_per_variant", heldout_cfg.get("episodes_per_variant", 1)))
    step_duration = float(heldout_cfg["step_duration"])
    max_step = _resolve_max_step(
        processed_rows=processed_data.height,
        step_duration=step_duration,
        max_step=heldout_cfg.get("max_step"),
        episode_hours=heldout_cfg.get("episode_hours"),
    )
    action_mode = str(heldout_cfg.get("action_mode", "multi_market"))
    degradation_mode = str(heldout_cfg.get("degradation_mode", "real_world"))
    degradation_chemistry = str(heldout_cfg.get("degradation_chemistry", "LFP"))
    degradation_temperature = float(heldout_cfg.get("degradation_temperature", 30.0))
    random_episode_start = bool(heldout_cfg.get("random_episode_start", False))
    base_seed = int(policy_cfg.get("seed", heldout_cfg.get("seed", 42)))

    if policy_kind == "dt":
        if dt_model is None:
            raise ValueError("DT policy requested but no model was loaded.")
        return run_dt_episodes(
            processed_data=processed_data,
            battery_variant=battery_variant,
            model=dt_model,
            num_episodes=episodes_per_variant,
            max_step=max_step,
            step_duration=step_duration,
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
            random_episode_start=random_episode_start,
            rtg_value=_dt_rtg_value(policy_cfg, training_summary),
            base_seed=base_seed,
        )

    if policy_kind == "rule":
        return run_rule_episodes(
            processed_data=processed_data,
            num_episodes=episodes_per_variant,
            battery_capacity=float(battery_variant["battery_capacity"]),
            max_battery_flow=float(battery_variant["max_battery_flow"]),
            init_soc=float(battery_variant["init_soc"]),
            step_duration=step_duration,
            battery_life_cost=float(battery_variant["battery_life_cost"]),
            max_step=max_step,
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
            random_episode_start=random_episode_start,
            base_seed=base_seed,
        )

    if policy_kind == "sb3":
        return run_sb3_episodes(
            processed_data=processed_data,
            battery_variant=battery_variant,
            model_path=Path(policy_cfg["model_path"]),
            algorithm=str(policy_cfg["algorithm"]),
            num_episodes=episodes_per_variant,
            max_step=max_step,
            step_duration=step_duration,
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
            random_episode_start=random_episode_start,
            deterministic=bool(policy_cfg.get("deterministic", True)),
            device=str(policy_cfg.get("device", "auto")),
        )

    if policy_kind == "dispatch":
        should_run, dispatch_region = should_run_dispatch_for_scenario(
            scenario_region=str(scenario["region"]),
            dispatch_station=policy_cfg.get("station_name"),
            dispatch_duid=policy_cfg.get("dispatch_duid"),
            start_date=scenario["start_date"],
            end_date=scenario["end_date"],
        )
        if not should_run:
            return []
        selection = build_dispatch_selection(
            region=dispatch_region or str(scenario["region"]),
            start_date=scenario["start_date"],
            end_date=scenario["end_date"],
            cache_dir=(repo_root() / str(heldout_cfg.get("cache_dir", "data/aemo"))).resolve(),
            dispatch_station=policy_cfg.get("station_name"),
            dispatch_duid=policy_cfg.get("dispatch_duid"),
            dispatch_index=int(policy_cfg.get("dispatch_index", 0)),
            battery_capacity=float(battery_variant["battery_capacity"]),
            max_battery_flow=float(battery_variant["max_battery_flow"]),
            init_soc=float(battery_variant["init_soc"]),
            init_soc_ratio=policy_cfg.get("init_soc_ratio"),
        )
        battery_life_cost = resolve_dispatch_battery_life_cost(
            dispatch_run=policy_cfg,
            station_capacity_mwh=float(selection["battery_capacity"]),
        )
        logs, _, _ = run_dispatch_replay(
            processed_data=processed_data,
            selection=selection,
            start_date=scenario["start_date"],
            end_date=scenario["end_date"],
            region=dispatch_region or str(scenario["region"]),
            cache_dir=str((repo_root() / str(heldout_cfg.get("cache_dir", "data/aemo"))).resolve()),
            num_episodes=episodes_per_variant,
            step_duration=step_duration,
            battery_life_cost=battery_life_cost,
            max_step=max_step,
            output_dir=None,
            run_tag=str(policy_cfg.get("name", "dispatch")),
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
        )
        return logs

    raise ValueError(f"Unsupported policy kind: {policy_kind!r}")


def _flatten_safety_rows(metrics_by_experiment: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    for name, metrics in metrics_by_experiment.items():
        flattened[name] = {
            "violation_step_rate": float(metrics["violation_step_rate"]),
            "violation_episode_rate": float(metrics["violation_episode_rate"]),
            "deg_incident_episode_rate": float(metrics["deg_incident_episode_rate"]),
            "clip_step_rate": float(metrics["clip_step_rate"]),
        }
    return flattened


def evaluate_aemo_heldout(
    *,
    surface_manifest: dict[str, Any],
    training_summary: dict[str, Any],
    evaluation_config: dict[str, Any],
    output_dir: Path,
    dt_model: DecisionTransformer | None,
) -> dict[str, Any]:
    heldout_cfg = dict(evaluation_config.get("heldout", {}))
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
    refresh = bool(heldout_cfg.get("refresh", False))

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

    battery_variants = resolve_battery_variants(heldout_cfg.get("battery_variants", []))
    if not battery_variants:
        raise ValueError("evaluation_config.heldout.battery_variants must be non-empty.")

    policies = evaluation_config.get("policies") or [{"name": "candidate_dt", "kind": "dt"}, {"name": "rule", "kind": "rule"}]
    aggregate_logs: dict[str, list[pl.DataFrame]] = {str(policy["name"]): [] for policy in policies}
    cohort_logs: dict[str, list[pl.DataFrame]] = {}

    heldout_logs_dir = output_dir / "heldout_logs"
    heldout_logs_dir.mkdir(parents=True, exist_ok=True)

    for scenario in scenario_manifest:
        scenario_label = str(scenario["label"])
        processed_data = processed_by_label[scenario_label]
        runtime_scenario = {
            **scenario,
            "start_date": _parse_datetime(str(scenario["start_date"])),
            "end_date": _parse_datetime(str(scenario["end_date"])),
        }
        for battery_variant in battery_variants:
            battery_label = str(battery_variant["label"])
            for policy in policies:
                policy_name = str(policy["name"])
                episodes = run_policy_episodes(
                    policy_cfg=policy,
                    processed_data=processed_data,
                    scenario=runtime_scenario,
                    battery_variant=battery_variant,
                    heldout_cfg=heldout_cfg,
                    training_summary=training_summary,
                    dt_model=dt_model,
                )
                tagged = _with_episode_metadata(
                    episodes,
                    policy_name=policy_name,
                    scenario_label=scenario_label,
                    battery_label=battery_label,
                )
                aggregate_logs.setdefault(policy_name, []).extend(tagged)
                cohort_key = f"{policy_name}::{scenario_label}::{battery_label}"
                cohort_logs[cohort_key] = tagged

    aggregate_logs = {name: logs for name, logs in aggregate_logs.items() if logs}
    if not aggregate_logs:
        raise RuntimeError("Held-out evaluation produced no logs.")

    for policy_name, logs in aggregate_logs.items():
        combined = _combine_episode_logs(logs)
        combined.write_parquet(heldout_logs_dir / f"{policy_name}_heldout_logs.parquet")

    plots_dir = output_dir / "plots"
    metrics_df = evaluate_experiments(
        aggregate_logs,
        target_return=float(evaluation_config.get("target_return", 0.0)),
        make_plots=False,
        save_dir=str(plots_dir),
    )
    metrics_rows = {row["experiment"]: dict(row) for row in metrics_df.to_dicts()}

    cohort_metrics_df = evaluate_experiments(
        cohort_logs,
        target_return=float(evaluation_config.get("target_return", 0.0)),
        make_plots=False,
        save_dir=None,
    )

    safety_metrics = {name: compute_info_signal_metrics(logs) for name, logs in aggregate_logs.items()}
    safety_rows = _flatten_safety_rows(safety_metrics)
    for name, row in metrics_rows.items():
        row.update(safety_rows.get(name, {}))

    aggregate_metrics_df = pl.DataFrame(list(metrics_rows.values())).sort("experiment")
    aggregate_metrics_df.write_csv(output_dir / "heldout_metrics.csv")

    by_cohort_rows: list[dict[str, Any]] = []
    for row in cohort_metrics_df.to_dicts():
        experiment = str(row["experiment"])
        policy_name, scenario_label, battery_label = experiment.split("::", 2)
        safety = compute_info_signal_metrics(cohort_logs[experiment])
        by_cohort_rows.append(
            {
                **row,
                "policy_name": policy_name,
                "scenario_label": scenario_label,
                "battery_label": battery_label,
                "violation_step_rate": float(safety["violation_step_rate"]),
                "violation_episode_rate": float(safety["violation_episode_rate"]),
                "deg_incident_episode_rate": float(safety["deg_incident_episode_rate"]),
                "clip_step_rate": float(safety["clip_step_rate"]),
            }
        )
    by_cohort_df = pl.DataFrame(by_cohort_rows).sort(["policy_name", "scenario_label", "battery_label"])
    by_cohort_df.write_csv(output_dir / "heldout_metrics_by_scenario.csv")

    condition_metrics = {
        name: evaluate_by_conditions(logs, default_aemo_conditions(evaluation_config.get("condition_thresholds")))
        for name, logs in aggregate_logs.items()
    }
    bootstrap = bootstrap_confidence_intervals(
        aggregate_logs,
        n_bootstrap=int(evaluation_config.get("bootstrap_iterations", 1000)),
        seed=int(evaluation_config.get("bootstrap_seed", 42)),
    )

    reference_policy = evaluation_config.get("reference_policy")
    paired: dict[str, dict[str, float]] = {}
    if reference_policy is not None and str(reference_policy) in aggregate_logs:
        ref_name = str(reference_policy)
        for name, logs in aggregate_logs.items():
            if name == ref_name:
                continue
            paired[name] = paired_comparison(logs, aggregate_logs[ref_name])

    return {
        "scenario_manifest": [
            {
                **scenario,
                "start_date": str(scenario["start_date"]),
                "end_date": str(scenario["end_date"]),
            }
            for scenario in scenario_manifest
        ],
        "cache_preflight": cache_preflight,
        "battery_variants": battery_variants,
        "aggregate_metrics": aggregate_metrics_df.to_dicts(),
        "metrics_by_scenario": by_cohort_df.to_dicts(),
        "safety_metrics": safety_metrics,
        "condition_metrics": condition_metrics,
        "bootstrap_confidence_intervals": bootstrap,
        "paired_comparisons_vs_reference": paired,
        "reference_policy": reference_policy,
        "plots_dir": str(plots_dir),
        "heldout_logs_dir": str(heldout_logs_dir),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    surface_manifest_path = args.surface_manifest_path.resolve()
    evaluation_config_path = args.evaluation_config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    surface_manifest = _load_json(surface_manifest_path)
    evaluation_config = _load_json(evaluation_config_path)

    loss_csv_path = Path(args.loss_csv_path or surface_manifest["paths"]["loss_csv_path"]).resolve()
    training_summary = summarize_loss_history(loss_csv_path)

    policies = evaluation_config.get("policies") or [{"name": "candidate_dt", "kind": "dt"}, {"name": "rule", "kind": "rule"}]

    dt_model = None
    model_path = None
    model_device = None
    if any(str(policy.get("kind", "")).lower() == "dt" for policy in policies):
        dt_model, model_path, model_device = load_dt_model(
            surface_manifest,
            model_path=args.model_path.resolve() if args.model_path is not None else None,
            device=args.device,
        )

    summary: dict[str, Any] = {
        "schema": "energydecision.autoresearch_evaluation.v1",
        "surface_manifest_path": str(surface_manifest_path),
        "evaluation_config_path": str(evaluation_config_path),
        "training_summary": training_summary,
        "dt_model_path": str(model_path) if model_path is not None else None,
        "dt_model_device": model_device,
    }

    track = str(evaluation_config.get("track", "aemo")).lower()
    if track != "aemo":
        raise ValueError(f"Unsupported evaluation track: {track!r}. Only 'aemo' is currently implemented.")

    summary["heldout_evaluation"] = evaluate_aemo_heldout(
        surface_manifest=surface_manifest,
        training_summary=training_summary,
        evaluation_config=evaluation_config,
        output_dir=output_dir,
        dt_model=dt_model,
    )

    _write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
