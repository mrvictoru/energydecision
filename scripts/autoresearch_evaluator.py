from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
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
from decision_transformer import DecisionTransformer, LegacyDecisionTransformer
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
    parser.add_argument(
        "--forecast-npz-path",
        type=Path,
        default=None,
        help="Path to TTM forecast npz for ForecastDecisionTransformer evaluation. "
             "Defaults to data/aemo_dt_forecast/ttm_forecasts.npz relative to repo root.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


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


def _best_row_by_metric(rows: Sequence[dict[str, Any]], metric_key: str) -> dict[str, Any] | None:
    metric_rows = [row for row in rows if row.get(metric_key) is not None]
    return min(metric_rows, key=lambda row: row[metric_key]) if metric_rows else None


def resolve_pilot_ranking(surface_manifest: dict[str, Any], training_summary: dict[str, Any]) -> dict[str, Any]:
    surface_preset = str(surface_manifest.get("surface_preset", "")).lower()
    best_val_total = training_summary.get("best_val_total_loss")
    best_val_action = training_summary.get("best_val_action_loss")
    if surface_preset == "aemo_proxy" and best_val_action is not None:
        return {
            "surface_preset": surface_preset,
            "pilot_ranking_metric": "best_val_action_loss",
            "pilot_ranking_value": best_val_action,
            "pilot_ranking_guardrail_metric": "best_val_total_loss",
            "pilot_ranking_guardrail_value": best_val_total,
        }
    return {
        "surface_preset": surface_preset,
        "pilot_ranking_metric": "best_val_total_loss",
        "pilot_ranking_value": best_val_total,
        "pilot_ranking_guardrail_metric": None,
        "pilot_ranking_guardrail_value": None,
    }


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
    best_val_total_row = _best_row_by_metric(parsed_rows, "val_total")
    best_val_action_row = _best_row_by_metric(parsed_rows, "val_action")
    best_val_state_row = _best_row_by_metric(parsed_rows, "val_state")
    best_val_return_row = _best_row_by_metric(parsed_rows, "val_return")

    return {
        "loss_csv_path": str(loss_csv_path),
        "epochs_recorded": len(parsed_rows),
        "final_epoch": final_row["epoch"],
        "final_train_total_loss": final_row["train_total"],
        "final_val_total_loss": final_row["val_total"],
        "final_val_action_loss": final_row["val_action"],
        "final_val_state_loss": final_row["val_state"],
        "final_val_return_loss": final_row["val_return"],
        "best_val_epoch": best_val_total_row["epoch"] if best_val_total_row is not None else None,
        "best_val_total_loss": best_val_total_row["val_total"] if best_val_total_row is not None else None,
        "best_val_action_epoch": best_val_action_row["epoch"] if best_val_action_row is not None else None,
        "best_val_action_loss": best_val_action_row["val_action"] if best_val_action_row is not None else None,
        "best_val_state_epoch": best_val_state_row["epoch"] if best_val_state_row is not None else None,
        "best_val_state_loss": best_val_state_row["val_state"] if best_val_state_row is not None else None,
        "best_val_return_epoch": best_val_return_row["epoch"] if best_val_return_row is not None else None,
        "best_val_return_loss": best_val_return_row["val_return"] if best_val_return_row is not None else None,
    }


def load_dt_model(surface_manifest: dict[str, Any], *, model_path: Path | None, device: str) -> tuple[Any, Path, str]:
    resolved_device = _resolve_device(device)
    model_kwargs = dict(surface_manifest["model_kwargs"])
    model_class = str(model_kwargs.pop("model_class", "DecisionTransformer"))
    weights_path = Path(model_path or surface_manifest["paths"]["save_path"]).resolve()

    if model_class == "ForecastDecisionTransformer":
        from forecast_decision_transformer import ForecastDecisionTransformer as DTClass
    else:
        from decision_transformer import DecisionTransformer as DTClass

    model = DTClass(**model_kwargs)
    model.load_from_checkpoint(str(weights_path), map_location=resolved_device)
    model.to(resolved_device)
    model.eval()

    # Apply model_meta (e.g., return_scale) from the surface manifest
    meta = surface_manifest.get("model_meta")
    if isinstance(meta, dict) and "return_scale" in meta:
        try:
            rs = float(meta["return_scale"])
            if rs == rs and abs(rs) >= 1e-12:
                model.return_scale = rs
        except Exception:
            pass

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


def _split_episode_logs(combined: pl.DataFrame) -> list[pl.DataFrame]:
    if combined.height == 0:
        return []
    if "episode_id" not in combined.columns:
        return [combined]
    episode_ids = sorted(int(episode_id) for episode_id in combined["episode_id"].unique().to_list())
    return [combined.filter(pl.col("episode_id") == episode_id).sort("step") for episode_id in episode_ids]


def _resolve_reference_cache_dir(evaluation_config: dict[str, Any]) -> Path | None:
    raw_path = evaluation_config.get("reference_cache_dir")
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = (repo_root() / path).resolve()
    return path


def _cacheable_policy(policy_cfg: dict[str, Any], reference_cache_dir: Path | None) -> bool:
    if reference_cache_dir is None:
        return False
    return str(policy_cfg.get("kind", "")).lower() != "dt" and bool(policy_cfg.get("cache_rollouts", True))


def _cache_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in str(value).strip()]
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "cache"


def _reference_rollout_cache_path(
    *,
    reference_cache_dir: Path,
    policy_cfg: dict[str, Any],
    scenario: dict[str, Any],
    battery_variant: dict[str, Any],
    heldout_cfg: dict[str, Any],
) -> tuple[Path, str]:
    payload = {
        "schema": "energydecision.autoresearch_rollout_cache.v1",
        "policy": {key: value for key, value in policy_cfg.items() if key != "cache_rollouts"},
        "scenario": {
            "label": scenario["label"],
            "region": scenario["region"],
            "start_date": str(scenario["start_date"]),
            "end_date": str(scenario["end_date"]),
        },
        "battery_variant": {
            "label": battery_variant.get("label"),
            "battery_capacity": battery_variant.get("battery_capacity"),
            "max_battery_flow": battery_variant.get("max_battery_flow"),
            "init_soc": battery_variant.get("init_soc"),
            "battery_life_cost": battery_variant.get("battery_life_cost"),
        },
        "heldout": {
            "step_duration": heldout_cfg.get("step_duration"),
            "episode_hours": heldout_cfg.get("episode_hours"),
            "max_step": heldout_cfg.get("max_step"),
            "episodes_per_variant": heldout_cfg.get("episodes_per_variant"),
            "random_episode_start": heldout_cfg.get("random_episode_start"),
            "action_mode": heldout_cfg.get("action_mode"),
            "degradation_mode": heldout_cfg.get("degradation_mode"),
            "degradation_chemistry": heldout_cfg.get("degradation_chemistry"),
            "degradation_temperature": heldout_cfg.get("degradation_temperature"),
        },
    }
    cache_key = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    stem = "__".join(
        [
            _cache_slug(str(policy_cfg["name"])),
            _cache_slug(str(scenario["label"])),
            _cache_slug(str(battery_variant["label"])),
            cache_key,
        ]
    )
    return reference_cache_dir / f"{stem}.parquet", cache_key


def _dt_rtg_value(policy_cfg: dict[str, Any], training_summary: dict[str, Any]) -> float:
    if "rtg_value" in policy_cfg:
        return float(policy_cfg["rtg_value"])
    best_val = training_summary.get("best_val_total_loss")
    return 0.0 if best_val is None else float(best_val)


def run_dt_episodes(
    *,
    processed_data: pl.DataFrame,
    battery_variant: dict[str, Any],
    model: Any,
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
    forecast_npz_path: str | None = None,
    dt_gamma: float = 0.99,
) -> list[pl.DataFrame]:
    """Roll out the DT candidate across ``num_episodes`` episodes in lockstep,
    batching the transformer forward pass per step over all active episodes.

    This fills the GPU much better than the previous batch-1-per-step loop
    (which left the 22 GB GPU at ~1.7 GB util), speeding up the candidate-DT
    leg of an evaluation by several times.
    """
    from decision import _build_dt_inference_context, stable_rtg_update

    # Forecast models keep the serial agent path (forecast window construction
    # is per-episode stateful).
    is_forecast = hasattr(model, "forecast_len") and int(getattr(model, "forecast_len", 0) or 0) > 0
    if is_forecast:
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
                dt_gamma=dt_gamma,
                reset_seed=base_seed + episode_idx if random_episode_start else None,
                forecast_npz_path=forecast_npz_path,
            )
            episode_df, _ = agent.run_episode()
            episodes.append(episode_df)
        return episodes

    envs = [
        create_aemo_env(
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
        for _ in range(num_episodes)
    ]

    buffers: list[dict[str, Any]] = []
    for i, env in enumerate(envs):
        seed = base_seed + i if random_episode_start else None
        obs, _ = env.reset(seed=seed, options={"random_episode_start": random_episode_start})
        buffers.append(
            {
                "states": [np.asarray(obs, dtype=np.float32).copy()],
                "actions": [np.zeros(model.act_dim, dtype=np.float32)],
                "rtgs": [float(rtg_value)],
                "timesteps": [int(env.current_step)],
                "done": False,
                "logs": [],
            }
        )

    model.eval()
    device = next(model.parameters()).device
    context_len = model.context_len
    step = 0
    while step < max_step and any(not b["done"] for b in buffers):
        active = [i for i, b in enumerate(buffers) if not b["done"]]
        ctx = [
            _build_dt_inference_context(
                model,
                buffers[i]["states"],
                buffers[i]["actions"],
                buffers[i]["rtgs"],
                buffers[i]["timesteps"],
            )
            for i in active
        ]
        states = torch.stack([torch.tensor(c[0], dtype=torch.float32, device=device) for c in ctx], dim=0)
        actions = torch.stack([torch.tensor(c[1], dtype=torch.float32, device=device) for c in ctx], dim=0)
        rtgs = torch.stack([torch.tensor(c[2], dtype=torch.float32, device=device).unsqueeze(-1) for c in ctx], dim=0)
        timesteps = torch.stack([torch.tensor(c[3], dtype=torch.long, device=device) for c in ctx], dim=0)
        attn = torch.stack([torch.tensor(c[4], dtype=torch.bool, device=device) for c in ctx], dim=0)

        with torch.no_grad():
            if isinstance(model, LegacyDecisionTransformer):
                _, _, act_pred = model.forward(states, actions, rtgs, timesteps, attention_mask=attn)
            else:
                _, _, act_pred = model.forward(states, rtgs, timesteps, actions, attention_mask=attn)
            act = act_pred[:, -1]
        act = torch.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy()
        if act.ndim == 3 and act.shape[1] == 1:
            act = act[:, 0, :]

        for j, i in enumerate(active):
            b = buffers[i]
            env = envs[i]
            action_vec = np.asarray(act[j], dtype=np.float32)
            if str(action_mode).lower() == "full_fcas" and action_vec.shape[0] >= 2:
                action_vec = action_vec.copy()
                action_vec[1:] = np.clip(action_vec[1:], 0.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action_vec.tolist())
            b["logs"].append(
                {
                    "step": step,
                    "norm_observation": b["states"][-1].tolist(),
                    "action": action_vec.tolist(),
                    "reward": float(reward),
                    "info": info,
                }
            )
            b["actions"][-1] = action_vec
            next_rtg = stable_rtg_update(
                b["rtgs"][-1], float(reward),
                dt_gamma=dt_gamma, initial_rtg=float(rtg_value),
            )
            b["states"].append(np.asarray(obs, dtype=np.float32).copy())
            b["actions"].append(np.zeros(model.act_dim, dtype=np.float32))
            b["rtgs"].append(next_rtg)
            b["timesteps"].append(int(env.current_step))
            if len(b["states"]) > context_len:
                b["states"] = b["states"][-context_len:]
                b["actions"] = b["actions"][-context_len:]
                b["rtgs"] = b["rtgs"][-context_len:]
                b["timesteps"] = b["timesteps"][-context_len:]
            if terminated or truncated:
                b["done"] = True
        step += 1

    return [pl.DataFrame(b["logs"]) for b in buffers]


def run_policy_episodes(
    *,
    policy_cfg: dict[str, Any],
    processed_data: pl.DataFrame,
    scenario: dict[str, Any],
    battery_variant: dict[str, Any],
    heldout_cfg: dict[str, Any],
    training_summary: dict[str, Any],
    dt_model: Any | None,
    comparison_cfg: dict[str, Any] | None = None,
    forecast_npz_path: str | None = None,
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
            forecast_npz_path=forecast_npz_path,
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
            algorithm=str(policy_cfg.get("algorithm", "rule")),
            fcas_pctile=float(policy_cfg.get("fcas_pctile", 0.80)),
            fcas_raise_threshold=policy_cfg.get("fcas_raise_threshold"),
            fcas_lower_threshold=policy_cfg.get("fcas_lower_threshold"),
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

    if policy_kind == "oracle":
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
            algorithm=str(policy_cfg.get("algorithm", "aemo_oracle")),
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
        # When use_dispatch_asset_sizing is disabled, override the selection's
        # battery with the template battery so all policies compete on the same asset.
        if comparison_cfg is not None and not bool(comparison_cfg.get("use_dispatch_asset_sizing", True)):
            selection["battery_capacity"] = float(battery_variant["battery_capacity"])
            selection["max_battery_flow"] = float(battery_variant["max_battery_flow"])
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


def _sum_count_maps(metrics: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    combined: dict[str, int] = {}
    for metric in metrics:
        for key, value in dict(metric.get(field, {})).items():
            combined[str(key)] = combined.get(str(key), 0) + int(value)
    return combined


def _aggregate_safety_metrics_by_policy(cohort_safety: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cohort_key, metrics in cohort_safety.items():
        policy_name = str(cohort_key).split("::", 1)[0]
        grouped.setdefault(policy_name, []).append(metrics)

    aggregate: dict[str, dict[str, Any]] = {}
    for policy_name, metrics_list in grouped.items():
        total_steps = int(sum(int(metrics.get("total_steps", 0)) for metrics in metrics_list))
        episodes_evaluated = int(sum(int(metrics.get("episodes_evaluated", 0)) for metrics in metrics_list))
        violation_step_count = int(sum(int(metrics.get("violation_step_count", 0)) for metrics in metrics_list))
        clip_step_count = int(sum(int(metrics.get("clip_step_count", 0)) for metrics in metrics_list))
        violation_episode_count = int(sum(int(metrics.get("violation_episode_count", 0)) for metrics in metrics_list))
        deg_incident_episode_count = int(sum(int(metrics.get("deg_incident_episode_count", 0)) for metrics in metrics_list))

        aggregate[policy_name] = {
            "episodes_evaluated": episodes_evaluated,
            "total_steps": total_steps,
            "violation_step_count": violation_step_count,
            "violation_step_rate": (violation_step_count / total_steps) if total_steps > 0 else 0.0,
            "violation_episode_count": violation_episode_count,
            "violation_episode_rate": (violation_episode_count / episodes_evaluated) if episodes_evaluated > 0 else 0.0,
            "deg_incident_episode_count": deg_incident_episode_count,
            "deg_incident_episode_rate": (deg_incident_episode_count / episodes_evaluated) if episodes_evaluated > 0 else 0.0,
            "clip_step_count": clip_step_count,
            "clip_step_rate": (clip_step_count / total_steps) if total_steps > 0 else 0.0,
            "violation_counts": _sum_count_maps(metrics_list, "violation_counts"),
            "clip_counts": _sum_count_maps(metrics_list, "clip_counts"),
        }
    return aggregate


def _comparison_config(evaluation_config: dict[str, Any]) -> dict[str, Any]:
    raw = evaluation_config.get("comparison", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _policy_names(policies: Sequence[dict[str, Any]]) -> list[str]:
    return [str(policy["name"]) for policy in policies]


def _find_policy(policies: Sequence[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    for policy in policies:
        if str(policy["name"]) == policy_name:
            return policy
    raise ValueError(f"Policy {policy_name!r} is not defined in evaluation_config.policies.")


def _strict_required_policies(
    *,
    policies: Sequence[dict[str, Any]],
    comparison_cfg: dict[str, Any],
) -> list[str]:
    configured = comparison_cfg.get("required_policy_names")
    if configured:
        return [str(name) for name in configured]
    return _policy_names(policies)


def _validate_dispatch_comparison_config(
    *,
    heldout_cfg: dict[str, Any],
    comparison_cfg: dict[str, Any],
    policies: Sequence[dict[str, Any]],
) -> None:
    action_mode = str(heldout_cfg.get("action_mode", "multi_market"))
    has_dispatch = any(str(policy.get("kind", "")).lower() == "dispatch" for policy in policies)

    if has_dispatch and bool(comparison_cfg.get("require_full_fcas_dispatch", False)) and action_mode != "full_fcas":
        raise ValueError(
            "comparison.require_full_fcas_dispatch=true requires heldout.action_mode='full_fcas'."
        )

    if not bool(comparison_cfg.get("use_dispatch_asset_sizing", False)):
        return

    dispatch_ref_name = comparison_cfg.get("dispatch_reference_policy_name")
    if dispatch_ref_name is None:
        dispatch_policies = [policy for policy in policies if str(policy.get("kind", "")).lower() == "dispatch"]
        if len(dispatch_policies) != 1:
            raise ValueError(
                "comparison.use_dispatch_asset_sizing=true requires comparison.dispatch_reference_policy_name "
                "when there is not exactly one dispatch policy."
            )
        dispatch_ref_name = str(dispatch_policies[0]["name"])

    dispatch_policy = _find_policy(policies, str(dispatch_ref_name))
    if str(dispatch_policy.get("kind", "")).lower() != "dispatch":
        raise ValueError(
            "comparison.dispatch_reference_policy_name must point to a policy with kind='dispatch'."
        )


def _resolve_runtime_battery_variants(
    *,
    scenario: dict[str, Any],
    heldout_cfg: dict[str, Any],
    comparison_cfg: dict[str, Any],
    policies: Sequence[dict[str, Any]],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    base_variants = resolve_battery_variants(heldout_cfg.get("battery_variants", []))
    if not base_variants:
        raise ValueError("evaluation_config.heldout.battery_variants must be non-empty.")

    if not bool(comparison_cfg.get("use_dispatch_asset_sizing", False)):
        return base_variants

    dispatch_ref_name = comparison_cfg.get("dispatch_reference_policy_name")
    if dispatch_ref_name is None:
        dispatch_policies = [policy for policy in policies if str(policy.get("kind", "")).lower() == "dispatch"]
        dispatch_ref_name = str(dispatch_policies[0]["name"])
    dispatch_policy = _find_policy(policies, str(dispatch_ref_name))

    template_variant = base_variants[0]
    should_run, dispatch_region = should_run_dispatch_for_scenario(
        scenario_region=str(scenario["region"]),
        dispatch_station=dispatch_policy.get("station_name"),
        dispatch_duid=dispatch_policy.get("dispatch_duid"),
        start_date=scenario["start_date"],
        end_date=scenario["end_date"],
    )
    if not should_run:
        return []

    selection = build_dispatch_selection(
        region=dispatch_region or str(scenario["region"]),
        start_date=scenario["start_date"],
        end_date=scenario["end_date"],
        cache_dir=cache_dir,
        dispatch_station=dispatch_policy.get("station_name"),
        dispatch_duid=dispatch_policy.get("dispatch_duid"),
        dispatch_index=int(dispatch_policy.get("dispatch_index", 0)),
        battery_capacity=float(template_variant["battery_capacity"]),
        max_battery_flow=float(template_variant["max_battery_flow"]),
        init_soc=float(template_variant["init_soc"]),
        init_soc_ratio=template_variant.get("init_soc_ratio"),
    )
    battery_life_cost = resolve_dispatch_battery_life_cost(
        dispatch_run=dispatch_policy,
        station_capacity_mwh=float(selection["battery_capacity"]),
    )
    station_name = str(selection.get("station_name") or dispatch_policy.get("station_name") or dispatch_ref_name)
    return [
        {
            "label": f"asset_{station_name}",
            "name": f"asset_{station_name}",
            "battery_capacity": float(selection["battery_capacity"]),
            "max_battery_flow": float(selection["max_battery_flow"]),
            "init_soc": float(selection["init_battery_level"]),
            "init_soc_ratio": float(
                np.clip(
                    float(selection["init_battery_level"]) / float(selection["battery_capacity"])
                    if float(selection["battery_capacity"]) > 0
                    else 0.5,
                    0.0,
                    1.0,
                )
            ),
            "battery_life_cost": float(battery_life_cost),
            "dispatch_station_name": station_name,
            "dispatch_region": str(selection.get("region") or dispatch_region or scenario["region"]),
        }
    ]


def _cohort_suffix(cohort_key: str) -> str:
    return str(cohort_key).split("::", 1)[1]


def _apply_strict_policy_intersection(
    *,
    aggregate_logs: dict[str, list[pl.DataFrame]],
    cohort_logs: dict[str, list[pl.DataFrame]],
    required_policy_names: Sequence[str],
) -> tuple[dict[str, list[pl.DataFrame]], dict[str, list[pl.DataFrame]], dict[str, Any]]:
    policy_sets_by_cohort: dict[str, set[str]] = {}
    for cohort_key in cohort_logs:
        policy_name, cohort_suffix = str(cohort_key).split("::", 1)
        policy_sets_by_cohort.setdefault(cohort_suffix, set()).add(policy_name)

    required = {str(name) for name in required_policy_names}
    matched_suffixes = sorted(
        cohort_suffix
        for cohort_suffix, present_policies in policy_sets_by_cohort.items()
        if required.issubset(present_policies)
    )
    if not matched_suffixes:
        raise RuntimeError(
            "Strict policy intersection produced no comparable cohorts. "
            "Check scenario coverage, dispatch availability, and required_policy_names."
        )

    matched_suffix_set = set(matched_suffixes)
    filtered_cohort_logs = {
        cohort_key: logs
        for cohort_key, logs in cohort_logs.items()
        if _cohort_suffix(cohort_key) in matched_suffix_set
    }
    filtered_aggregate_logs: dict[str, list[pl.DataFrame]] = {}
    for cohort_key, logs in filtered_cohort_logs.items():
        policy_name = str(cohort_key).split("::", 1)[0]
        filtered_aggregate_logs.setdefault(policy_name, []).extend(logs)

    dropped_suffixes = sorted(set(policy_sets_by_cohort) - matched_suffix_set)
    return filtered_aggregate_logs, filtered_cohort_logs, {
        "required_policy_names": sorted(required),
        "matched_cohort_count": len(matched_suffixes),
        "matched_cohorts": matched_suffixes,
        "dropped_cohort_count": len(dropped_suffixes),
        "dropped_cohorts": dropped_suffixes,
        "policy_coverage_by_cohort": {
            cohort_suffix: sorted(policy_names)
            for cohort_suffix, policy_names in sorted(policy_sets_by_cohort.items())
        },
    }


def evaluate_aemo_heldout(
    *,
    surface_manifest: dict[str, Any],
    training_summary: dict[str, Any],
    evaluation_config: dict[str, Any],
    output_dir: Path,
    dt_model: Any | None,
    forecast_npz_path: str | None = None,
) -> dict[str, Any]:
    heldout_cfg = dict(evaluation_config.get("heldout", {}))
    comparison_cfg = _comparison_config(evaluation_config)
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

    policies = evaluation_config.get("policies") or [{"name": "candidate_dt", "kind": "dt"}, {"name": "rule", "kind": "rule"}]
    _validate_dispatch_comparison_config(
        heldout_cfg=heldout_cfg,
        comparison_cfg=comparison_cfg,
        policies=policies,
    )
    aggregate_logs: dict[str, list[pl.DataFrame]] = {str(policy["name"]): [] for policy in policies}
    cohort_logs: dict[str, list[pl.DataFrame]] = {}

    heldout_logs_dir = output_dir / "heldout_logs"
    heldout_logs_dir.mkdir(parents=True, exist_ok=True)
    reference_cache_dir = _resolve_reference_cache_dir(evaluation_config)
    if reference_cache_dir is not None:
        reference_cache_dir.mkdir(parents=True, exist_ok=True)
    reference_cache_hits: list[dict[str, Any]] = []
    reference_cache_misses: list[dict[str, Any]] = []
    parallel_workers = max(1, int(heldout_cfg.get("parallel_workers", 8)))
    parallelize_candidate_dt = bool(heldout_cfg.get("parallelize_candidate_dt", True))
    battery_variants_seen: dict[str, dict[str, Any]] = {}

    work_items: list[dict[str, Any]] = []
    for scenario in scenario_manifest:
        scenario_label = str(scenario["label"])
        processed_data = processed_by_label[scenario_label]
        runtime_scenario = {
            **scenario,
            "start_date": _parse_datetime(str(scenario["start_date"])),
            "end_date": _parse_datetime(str(scenario["end_date"])),
        }
        runtime_battery_variants = _resolve_runtime_battery_variants(
            scenario=runtime_scenario,
            heldout_cfg=heldout_cfg,
            comparison_cfg=comparison_cfg,
            policies=policies,
            cache_dir=cache_dir,
        )
        for battery_variant in runtime_battery_variants:
            battery_label = str(battery_variant["label"])
            battery_variants_seen[battery_label] = battery_variant
            for policy in policies:
                work_items.append(
                    {
                        "policy": policy,
                        "policy_name": str(policy["name"]),
                        "policy_kind": str(policy["kind"]).lower(),
                        "scenario_label": scenario_label,
                        "runtime_scenario": runtime_scenario,
                        "battery_variant": battery_variant,
                        "battery_label": battery_label,
                        "processed_data": processed_data,
                    }
                )

    battery_variants = list(battery_variants_seen.values())
    if not battery_variants:
        raise RuntimeError("No battery variants were resolved for the requested scenarios.")

    def _execute_rollout(item: dict[str, Any]) -> dict[str, Any]:
        policy = dict(item["policy"])
        policy_name = str(item["policy_name"])
        scenario_label = str(item["scenario_label"])
        battery_label = str(item["battery_label"])
        tagged: list[pl.DataFrame] | None = None
        cache_path: Path | None = None
        cache_key: str | None = None
        cache_event: str | None = None

        if _cacheable_policy(policy, reference_cache_dir):
            assert reference_cache_dir is not None
            cache_path, cache_key = _reference_rollout_cache_path(
                reference_cache_dir=reference_cache_dir,
                policy_cfg=policy,
                scenario=item["runtime_scenario"],
                battery_variant=item["battery_variant"],
                heldout_cfg=heldout_cfg,
            )
            if cache_path.is_file():
                tagged = _split_episode_logs(pl.read_parquet(cache_path))
                cache_event = "hit"
            else:
                cache_event = "miss"

        if tagged is None:
            episodes = run_policy_episodes(
                policy_cfg=policy,
                processed_data=item["processed_data"],
                scenario=item["runtime_scenario"],
                battery_variant=item["battery_variant"],
                heldout_cfg=heldout_cfg,
                training_summary=training_summary,
                dt_model=dt_model,
                comparison_cfg=comparison_cfg,
                forecast_npz_path=forecast_npz_path,
            )
            tagged = _with_episode_metadata(
                episodes,
                policy_name=policy_name,
                scenario_label=scenario_label,
                battery_label=battery_label,
            )
            if cache_path is not None and cache_event == "miss":
                cached = _combine_episode_logs(tagged)
                cached.write_parquet(cache_path)

        return {
            "policy_name": policy_name,
            "scenario_label": scenario_label,
            "battery_label": battery_label,
            "tagged": tagged,
            "cache_path": str(cache_path) if cache_path is not None else None,
            "cache_key": cache_key,
            "cache_event": cache_event,
        }

    results_by_index: list[dict[str, Any] | None] = [None] * len(work_items)
    parallel_slots: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(work_items):
        is_candidate_dt = item["policy_kind"] == "dt"
        can_parallel = parallel_workers > 1 and (parallelize_candidate_dt or not is_candidate_dt)
        if can_parallel:
            parallel_slots.append((idx, item))
        else:
            results_by_index[idx] = _execute_rollout(item)

    if parallel_slots:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            ordered_inputs = [item for _, item in parallel_slots]
            ordered_indices = [idx for idx, _ in parallel_slots]
            ordered_results = list(executor.map(_execute_rollout, ordered_inputs))
        for idx, result in zip(ordered_indices, ordered_results):
            results_by_index[idx] = result

    for result in results_by_index:
        assert result is not None
        policy_name = str(result["policy_name"])
        scenario_label = str(result["scenario_label"])
        battery_label = str(result["battery_label"])
        tagged = list(result["tagged"])
        aggregate_logs.setdefault(policy_name, []).extend(tagged)
        cohort_key = f"{policy_name}::{scenario_label}::{battery_label}"
        cohort_logs[cohort_key] = tagged
        cache_event = result.get("cache_event")
        cache_path = result.get("cache_path")
        cache_key = result.get("cache_key")
        if cache_event == "hit":
            reference_cache_hits.append(
                {
                    "policy_name": policy_name,
                    "scenario_label": scenario_label,
                    "battery_label": battery_label,
                    "cache_key": cache_key,
                    "path": cache_path,
                }
            )
        elif cache_event == "miss":
            reference_cache_misses.append(
                {
                    "policy_name": policy_name,
                    "scenario_label": scenario_label,
                    "battery_label": battery_label,
                    "cache_key": cache_key,
                    "path": cache_path,
                }
            )

    comparison_scope: dict[str, Any] = {
        "strict_policy_intersection": bool(comparison_cfg.get("strict_policy_intersection", False)),
        "require_full_fcas_dispatch": bool(comparison_cfg.get("require_full_fcas_dispatch", False)),
        "use_dispatch_asset_sizing": bool(comparison_cfg.get("use_dispatch_asset_sizing", False)),
        "dispatch_reference_policy_name": comparison_cfg.get("dispatch_reference_policy_name"),
    }
    if bool(comparison_cfg.get("strict_policy_intersection", False)):
        aggregate_logs, cohort_logs, intersection_meta = _apply_strict_policy_intersection(
            aggregate_logs=aggregate_logs,
            cohort_logs=cohort_logs,
            required_policy_names=_strict_required_policies(
                policies=policies,
                comparison_cfg=comparison_cfg,
            ),
        )
        comparison_scope.update(intersection_meta)

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

    cohort_safety_metrics = {name: compute_info_signal_metrics(logs) for name, logs in cohort_logs.items()}
    safety_metrics = _aggregate_safety_metrics_by_policy(cohort_safety_metrics)
    safety_rows = _flatten_safety_rows(safety_metrics)
    for name, row in metrics_rows.items():
        row.update(safety_rows.get(name, {}))

    aggregate_metrics_df = pl.DataFrame(list(metrics_rows.values())).sort("experiment")
    aggregate_metrics_df.write_csv(output_dir / "heldout_metrics.csv")

    by_cohort_rows: list[dict[str, Any]] = []
    for row in cohort_metrics_df.to_dicts():
        experiment = str(row["experiment"])
        policy_name, scenario_label, battery_label = experiment.split("::", 2)
        safety = cohort_safety_metrics[experiment]
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

    if by_cohort_rows:
        comparison_scope["matched_scenarios"] = sorted({row["scenario_label"] for row in by_cohort_rows})
        comparison_scope["matched_battery_labels"] = sorted({row["battery_label"] for row in by_cohort_rows})
    else:
        comparison_scope["matched_scenarios"] = []
        comparison_scope["matched_battery_labels"] = []

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
        "comparison_scope": comparison_scope,
        "reference_rollout_cache": {
            "enabled": reference_cache_dir is not None,
            "cache_dir": str(reference_cache_dir) if reference_cache_dir is not None else None,
            "hits": reference_cache_hits,
            "misses": reference_cache_misses,
        },
        "rollout_execution": {
            "parallel_workers": parallel_workers,
            "parallelize_candidate_dt": parallelize_candidate_dt,
            "work_item_count": len(work_items),
        },
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
        "pilot_ranking": resolve_pilot_ranking(surface_manifest, training_summary),
        "dt_model_path": str(model_path) if model_path is not None else None,
        "dt_model_device": model_device,
    }

    track = str(evaluation_config.get("track", "aemo")).lower()
    if track != "aemo":
        raise ValueError(f"Unsupported evaluation track: {track!r}. Only 'aemo' is currently implemented.")

    forecast_npz_path = args.forecast_npz_path
    if forecast_npz_path is None:
        default_npz = repo_root() / "data" / "aemo_dt_forecast" / "ttm_forecasts.npz"
        forecast_npz_path = default_npz if default_npz.exists() else None
    else:
        forecast_npz_path = forecast_npz_path.resolve()

    summary["heldout_evaluation"] = evaluate_aemo_heldout(
        surface_manifest=surface_manifest,
        training_summary=training_summary,
        evaluation_config=evaluation_config,
        output_dir=output_dir,
        dt_model=dt_model,
        forecast_npz_path=str(forecast_npz_path) if forecast_npz_path is not None else None,
    )

    _write_json(output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
