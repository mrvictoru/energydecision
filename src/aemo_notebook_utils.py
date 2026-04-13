from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import polars as pl
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from AEMOBatteryEnv import AEMOBatteryTradingEnv, AEMODataPreprocessor
from aemo_data import fetch_aemo_data_bundle, resolve_battery_duids
from decision import AEMOAgent, run_sb3_model_on_vec_env
from dispatch_utils import (
    list_dispatch_candidates,
    resolve_dispatch_selection,
    run_dispatch_replay,
)
from helper import flatten_episode_data
from sb3train import train_model

DEFAULT_FCAS_SERVICES = [
    "RAISEREG",
    "LOWERREG",
    "RAISE6SEC",
    "LOWER6SEC",
    "RAISE60SEC",
    "LOWER60SEC",
    "RAISE5MIN",
    "LOWER5MIN",
]
DEFAULT_FUEL_TYPES = ["solar", "wind"]
DEFAULT_BATTERY_CORE_COST_PER_KWH = 75.0  # battery pack cost only
DEFAULT_BATTERY_FIXED_COST_BASE_PER_KWH = 50.0  # fixed installation/grid cost at 10 MWh reference size
DEFAULT_BATTERY_COST_SCALE_EXPONENT = 0.15  # economy of scale factor for fixed costs
DEFAULT_BATTERY_COST_PREMIUM = 1.05  # small margin for contingencies and BOP
DEFAULT_BATTERY_COST_PER_KWH = DEFAULT_BATTERY_CORE_COST_PER_KWH
SB3_MODEL_CLASSES = {
    "A2C": A2C,
    "DDPG": DDPG,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_float_list(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32).reshape(-1)
        return arr.tolist()
    if isinstance(value, pl.Series):
        return _to_float_list(value.to_list())
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if value is None:
        raise ValueError("Encountered null value while normalizing vector column.")
    return [float(value)]


def default_aemo_dt_model_kwargs(
    *,
    action_mode: str = "multi_market",
    context_len: int = 288,
    max_timestep: int = 2016,
) -> dict[str, Any]:
    act_dim = 3 if action_mode == "multi_market" else 1
    return {
        "state_dim": 18,
        "act_dim": act_dim,
        "n_block": 4,
        "h_dim": 128,
        "context_len": int(context_len),
        "n_heads": 8,
        "drop_p": 0.1,
        "max_timestep": int(max_timestep),
        "rope_enabled": True,
        "rope_max_position": int(max(context_len * 3, 512)),
        "rope_base": 10000.0,
    }


def _episode_steps(
    *,
    processed_rows: int,
    step_duration: float,
    episode_hours: float | None,
) -> int:
    if episode_hours is None:
        return int(processed_rows)
    return max(1, int(round(float(episode_hours) / float(step_duration))))


def _processed_cache_path(
    *,
    cache_dir: Path,
    region: str,
    start_date: datetime,
    end_date: datetime,
    step_duration: float,
) -> Path:
    return cache_dir / (
        f"processed_{region}_{start_date.date()}_{end_date.date()}_{step_duration:.4f}h.parquet"
    )


def _scenario_label(scenario: dict[str, Any], index: int) -> str:
    label = scenario.get("label") or scenario.get("name")
    if label:
        return str(label)
    region = scenario["region"]
    start_date = scenario["start_date"]
    end_date = scenario["end_date"]
    return f"{region}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_{index}"


def _scenario_entry(scenario: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "label": _scenario_label(scenario, index),
        "region": str(scenario["region"]),
        "start_date": scenario["start_date"],
        "end_date": scenario["end_date"],
        "index": index,
    }


def _update_bounds(
    stats: dict[str, dict[str, float]],
    key: str,
    *,
    min_value: float | None,
    max_value: float | None,
) -> None:
    if min_value is None or max_value is None:
        return
    stats[key]["min"] = min(float(stats[key]["min"]), float(min_value))
    stats[key]["max"] = max(float(stats[key]["max"]), float(max_value))


def _fit_global_stats_from_frames(frames: Sequence[pl.DataFrame]) -> dict[str, dict[str, float]]:
    stats = {
        "RRP": {"min": float("inf"), "max": float("-inf")},
        "FCAS_PRICE": {"min": float("inf"), "max": float("-inf")},
        "TOTALDEMAND": {"min": float("inf"), "max": float("-inf")},
        "GENERATION": {"min": float("inf"), "max": float("-inf")},
    }
    for frame in frames:
        if frame.height == 0:
            continue

        if "RRP" in frame.columns:
            _update_bounds(
                stats,
                "RRP",
                min_value=frame.select(pl.col("RRP").min()).item(),
                max_value=frame.select(pl.col("RRP").max()).item(),
            )

        if "TOTALDEMAND" in frame.columns:
            _update_bounds(
                stats,
                "TOTALDEMAND",
                min_value=frame.select(pl.col("TOTALDEMAND").min()).item(),
                max_value=frame.select(pl.col("TOTALDEMAND").max()).item(),
            )

        fcas_cols = [c for c in frame.columns if c.startswith("FCAS_") and not c.endswith("_normalized")]
        if fcas_cols:
            fcas_mins = frame.select([pl.col(c).min() for c in fcas_cols]).row(0)
            fcas_maxs = frame.select([pl.col(c).max() for c in fcas_cols]).row(0)
            valid_mins = [v for v in fcas_mins if v is not None]
            valid_maxs = [v for v in fcas_maxs if v is not None]
            if valid_mins and valid_maxs:
                _update_bounds(
                    stats,
                    "FCAS_PRICE",
                    min_value=min(valid_mins),
                    max_value=max(valid_maxs),
                )

        gen_cols = [c for c in frame.columns if c.startswith("GEN_") and not c.endswith("_pct")]
        if gen_cols:
            gen_mins = frame.select([pl.col(c).min() for c in gen_cols]).row(0)
            gen_maxs = frame.select([pl.col(c).max() for c in gen_cols]).row(0)
            valid_mins = [v for v in gen_mins if v is not None]
            valid_maxs = [v for v in gen_maxs if v is not None]
            if valid_mins and valid_maxs:
                _update_bounds(
                    stats,
                    "GENERATION",
                    min_value=min(valid_mins),
                    max_value=max(valid_maxs),
                )

    defaults = AEMODataPreprocessor.default_stats()
    for key, value in stats.items():
        if value["min"] == float("inf") or value["max"] == float("-inf"):
            stats[key] = dict(defaults[key])

    return stats


def fetch_and_preprocess_aemo_data(
    *,
    region: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: Path,
    step_duration: float,
    refresh: bool = False,
    fixed_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pl.DataFrame, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_cache = _processed_cache_path(
        cache_dir=cache_dir,
        region=region,
        start_date=start_date,
        end_date=end_date,
        step_duration=step_duration,
    )
    if processed_cache.exists() and not refresh and fixed_stats is None:
        return pl.read_parquet(str(processed_cache)), processed_cache

    raw_data = fetch_aemo_data_bundle(
        start_date=start_date,
        end_date=end_date,
        region=region,
        fcas_services=DEFAULT_FCAS_SERVICES,
        fuel_types=DEFAULT_FUEL_TYPES,
        cache_dir=str(cache_dir),
    )
    preprocessor = AEMODataPreprocessor(
        step_duration_hours=step_duration,
        fixed_stats=fixed_stats,
    )
    processed_data = preprocessor.preprocess_aemo_data(
        prices=raw_data["prices"],
        fcas=raw_data["fcas"],
        generation=raw_data["generation"],
    )
    processed_data.write_parquet(str(processed_cache))
    return processed_data, processed_cache


def fit_aemo_global_stats(
    *,
    scenarios: Sequence[dict[str, Any]],
    cache_dir: Path,
    step_duration: float,
    refresh: bool = False,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    scenario_manifest: list[dict[str, Any]] = []
    prepared_frames: list[pl.DataFrame] = []

    for index, scenario in enumerate(scenarios):
        entry = _scenario_entry(scenario, index)
        scenario_manifest.append(entry)
        raw_data = fetch_aemo_data_bundle(
            start_date=entry["start_date"],
            end_date=entry["end_date"],
            region=entry["region"],
            fcas_services=DEFAULT_FCAS_SERVICES,
            fuel_types=DEFAULT_FUEL_TYPES,
            cache_dir=str(cache_dir),
            refresh=refresh,
        )
        preprocessor = AEMODataPreprocessor(
            step_duration_hours=step_duration,
            add_normalized_features=False,
            update_stats_from_data=False,
        )
        prepared_frames.append(
            preprocessor.prepare_aemo_data(
                prices=raw_data["prices"],
                fcas=raw_data["fcas"],
                generation=raw_data["generation"],
            )
        )

    return _fit_global_stats_from_frames(prepared_frames), scenario_manifest


def fetch_and_preprocess_aemo_scenarios(
    *,
    scenarios: Sequence[dict[str, Any]],
    cache_dir: Path,
    step_duration: float,
    refresh: bool = False,
    fixed_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, pl.DataFrame], list[dict[str, Any]]]:
    processed_by_label: dict[str, pl.DataFrame] = {}
    scenario_manifest: list[dict[str, Any]] = []

    for index, scenario in enumerate(scenarios):
        entry = _scenario_entry(scenario, index)
        processed, _ = fetch_and_preprocess_aemo_data(
            region=entry["region"],
            start_date=entry["start_date"],
            end_date=entry["end_date"],
            cache_dir=cache_dir,
            step_duration=step_duration,
            refresh=refresh,
            fixed_stats=fixed_stats,
        )
        processed_by_label[entry["label"]] = processed
        scenario_manifest.append(entry)

    return processed_by_label, scenario_manifest


def _battery_fixed_cost_per_kwh(
    capacity_mwh: float,
    base_fixed_cost_per_kwh: float = DEFAULT_BATTERY_FIXED_COST_BASE_PER_KWH,
    reference_capacity_mwh: float = 10.0,
    scale_exponent: float = DEFAULT_BATTERY_COST_SCALE_EXPONENT,
) -> float:
    capacity_mwh = float(capacity_mwh)
    return float(base_fixed_cost_per_kwh) * (
        reference_capacity_mwh / max(capacity_mwh, 1e-6)
    ) ** float(scale_exponent)


def _battery_life_cost(capacity_mwh: float, battery_cost_per_kwh: float) -> float:
    core_cost = float(battery_cost_per_kwh)
    fixed_cost = _battery_fixed_cost_per_kwh(capacity_mwh)
    total_cost_per_kwh = core_cost + fixed_cost
    return total_cost_per_kwh * DEFAULT_BATTERY_COST_PREMIUM * float(capacity_mwh) * 1000.0


def resolve_battery_variants(
    variants: Sequence[dict[str, Any]],
    *,
    default_cost_per_kwh: float = DEFAULT_BATTERY_COST_PER_KWH,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        label = str(variant.get("name") or variant.get("label") or f"battery_{idx}")
        battery_capacity = float(
            variant.get("battery_capacity", variant.get("capacity_mwh"))
        )
        max_battery_flow = float(
            variant.get(
                "max_battery_flow",
                variant.get("max_power_mw", variant.get("max_flow_mw")),
            )
        )
        init_soc = variant.get("init_soc")
        if init_soc is None:
            init_soc_ratio = float(variant.get("init_soc_ratio", 0.5))
            init_soc = float(np.clip(battery_capacity * init_soc_ratio, 0.0, battery_capacity))
        battery_life_cost = variant.get("battery_life_cost")
        if battery_life_cost is None:
            battery_life_cost = _battery_life_cost(
                battery_capacity,
                float(variant.get("battery_cost_per_kwh", default_cost_per_kwh)),
            )
        resolved.append(
            {
                **variant,
                "label": label,
                "battery_capacity": battery_capacity,
                "max_battery_flow": max_battery_flow,
                "init_soc": float(init_soc),
                "battery_life_cost": float(battery_life_cost),
            }
        )
    return resolved


def create_aemo_env(
    *,
    processed_data: pl.DataFrame,
    battery_variant: dict[str, Any],
    max_step: int,
    step_duration: float,
    action_mode: str = "multi_market",
    degradation_mode: str = "real_world",
    degradation_chemistry: str = "LFP",
    degradation_temperature: float = 30.0,
    random_episode_start: bool = False,
) -> AEMOBatteryTradingEnv:
    return AEMOBatteryTradingEnv(
        aemo_data=processed_data,
        battery_capacity=battery_variant["battery_capacity"],
        max_battery_flow=battery_variant["max_battery_flow"],
        init_battery_level=battery_variant["init_soc"],
        max_step=max_step,
        step_duration=step_duration,
        battery_life_cost=battery_variant["battery_life_cost"],
        action_mode=action_mode,
        random_episode_start=random_episode_start,
        degradation_mode=degradation_mode,
        degradation_chemistry=degradation_chemistry,
        degradation_temperature=degradation_temperature,
    )


def make_aemo_env_fns(
    *,
    processed_data: pl.DataFrame,
    battery_variants: Sequence[dict[str, Any]],
    episodes_per_variant: int,
    max_step: int,
    step_duration: float,
    action_mode: str = "multi_market",
    degradation_mode: str = "real_world",
    degradation_chemistry: str = "LFP",
    degradation_temperature: float = 30.0,
    random_episode_start: bool = True,
) -> list[Callable[[], AEMOBatteryTradingEnv]]:
    def _make_env_factory(
        battery_variant: dict[str, Any],
    ) -> Callable[[], AEMOBatteryTradingEnv]:
        def _factory() -> AEMOBatteryTradingEnv:
            return create_aemo_env(
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

        return _factory

    env_fns: list[Callable[[], AEMOBatteryTradingEnv]] = []
    for battery_variant in resolve_battery_variants(battery_variants):
        for _ in range(episodes_per_variant):
            env_fns.append(_make_env_factory(battery_variant))
    return env_fns


def make_multi_scenario_aemo_env_fns(
    *,
    scenario_data: Sequence[tuple[dict[str, Any], pl.DataFrame]],
    battery_variants: Sequence[dict[str, Any]],
    episodes_per_variant: int,
    max_step: int,
    step_duration: float,
    action_mode: str = "multi_market",
    degradation_mode: str = "real_world",
    degradation_chemistry: str = "LFP",
    degradation_temperature: float = 30.0,
    random_episode_start: bool = True,
) -> list[Callable[[], AEMOBatteryTradingEnv]]:
    env_fns: list[Callable[[], AEMOBatteryTradingEnv]] = []
    for scenario, processed_data in scenario_data:
        for env_fn in make_aemo_env_fns(
            processed_data=processed_data,
            battery_variants=battery_variants,
            episodes_per_variant=episodes_per_variant,
            max_step=max_step,
            step_duration=step_duration,
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
            random_episode_start=random_episode_start,
        ):
            env_fns.append(env_fn)
    return env_fns


def run_rule_episodes(
    *,
    processed_data: pl.DataFrame,
    num_episodes: int,
    battery_capacity: float,
    max_battery_flow: float,
    init_soc: float,
    step_duration: float,
    battery_life_cost: float,
    max_step: int,
    action_mode: str,
    degradation_mode: str,
    degradation_chemistry: str,
    degradation_temperature: float,
    random_episode_start: bool,
    base_seed: int,
) -> list[pl.DataFrame]:
    episodes: list[pl.DataFrame] = []
    for episode_idx in range(num_episodes):
        env = AEMOBatteryTradingEnv(
            aemo_data=processed_data,
            battery_capacity=battery_capacity,
            max_battery_flow=max_battery_flow,
            init_battery_level=init_soc,
            max_step=max_step,
            step_duration=step_duration,
            battery_life_cost=battery_life_cost,
            action_mode=action_mode,
            random_episode_start=random_episode_start,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
        )
        agent = AEMOAgent(
            env,
            algorithm="rule",
            reset_seed=base_seed + episode_idx if random_episode_start else None,
        )
        episode_df, _ = agent.run_episode()
        episodes.append(episode_df)
    return episodes


def get_sb3_model_class(name: str):
    key = str(name).strip().upper()
    if key not in SB3_MODEL_CLASSES:
        raise ValueError(f"Unsupported SB3 algorithm {name!r}. Expected one of {sorted(SB3_MODEL_CLASSES)}.")
    return SB3_MODEL_CLASSES[key]


def load_sb3_model(
    *,
    algorithm: str,
    model_path: str | Path,
    env=None,
    device: str = "auto",
):
    model_cls = get_sb3_model_class(algorithm)
    return model_cls.load(str(Path(model_path).resolve()), env=env, device=device)


def run_sb3_episodes(
    *,
    processed_data: pl.DataFrame,
    battery_variant: dict[str, Any],
    model_path: str | Path,
    algorithm: str,
    num_episodes: int,
    max_step: int,
    step_duration: float,
    action_mode: str = "multi_market",
    degradation_mode: str = "real_world",
    degradation_chemistry: str = "LFP",
    degradation_temperature: float = 30.0,
    random_episode_start: bool = True,
    deterministic: bool = True,
    device: str = "auto",
) -> list[pl.DataFrame]:
    resolved_variant = resolve_battery_variants([battery_variant])[0]
    env_fns = make_aemo_env_fns(
        processed_data=processed_data,
        battery_variants=[resolved_variant],
        episodes_per_variant=num_episodes,
        max_step=max_step,
        step_duration=step_duration,
        action_mode=action_mode,
        degradation_mode=degradation_mode,
        degradation_chemistry=degradation_chemistry,
        degradation_temperature=degradation_temperature,
        random_episode_start=random_episode_start,
    )
    vec_env = DummyVecEnv(env_fns)
    try:
        model = load_sb3_model(
            algorithm=algorithm,
            model_path=model_path,
            env=vec_env,
            device=device,
        )
        episode_data = run_sb3_model_on_vec_env(
            model,
            vec_env,
            deterministic=deterministic,
            max_steps=max_step,
        )
        flat = flatten_episode_data(episode_data)
    finally:
        vec_env.close()
    return [
        flat.filter(pl.col("episode_id") == episode_id)
        for episode_id in sorted(flat["episode_id"].unique().to_list())
    ]


def write_combined_episode_logs(
    *,
    episodes: Sequence[pl.DataFrame],
    output_path: Path,
) -> pl.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not episodes:
        combined = pl.DataFrame(schema={"episode_id": pl.Int64})
    else:
        combined = pl.concat(
            [
                episode.with_columns(pl.lit(idx).alias("episode_id"))
                for idx, episode in enumerate(episodes)
            ],
            how="diagonal_relaxed",
        )
    combined.write_parquet(str(output_path))
    return combined


def load_episode_logs_from_parquet(path: str | Path) -> list[pl.DataFrame]:
    parquet_path = Path(path)
    df = pl.read_parquet(str(parquet_path))
    if df.height == 0:
        return []
    if "episode_id" not in df.columns:
        return [df]
    return [
        df.filter(pl.col("episode_id") == episode_id)
        for episode_id in sorted(df["episode_id"].unique().to_list())
    ]


def _normalize_episode_dataframe(
    episode_df: pl.DataFrame,
    *,
    source_policy: str,
    episode_id: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if episode_df.height == 0:
        raise ValueError(f"{source_policy} episode {episode_id} is empty.")
    required = {"norm_observation", "action", "reward"}
    missing = required - set(episode_df.columns)
    if missing:
        raise ValueError(
            f"{source_policy} episode {episode_id} is missing required columns: {sorted(missing)}"
        )

    working = episode_df
    if "step" not in working.columns:
        working = working.with_row_index("step")
    else:
        working = working.sort("step")

    working = working.select(["step", "norm_observation", "action", "reward"]).with_columns(
        [
            pl.col("norm_observation")
            .map_elements(_to_float_list, return_dtype=pl.List(pl.Float32))
            .alias("norm_observation"),
            pl.col("action")
            .map_elements(_to_float_list, return_dtype=pl.List(pl.Float32))
            .alias("action"),
            pl.col("reward").cast(pl.Float32, strict=False).alias("reward"),
            pl.lit(episode_id).alias("episode_id"),
            pl.lit(source_policy).alias("source_policy"),
        ]
    )

    first_state = working["norm_observation"][0]
    first_action = working["action"][0]
    state_dim = len(first_state)
    act_dim = len(first_action)
    if state_dim == 0 or act_dim == 0:
        raise ValueError(
            f"{source_policy} episode {episode_id} has invalid vector sizes: state={state_dim}, action={act_dim}"
        )

    manifest_row = {
        "episode_id": episode_id,
        "source_policy": source_policy,
        "rows": int(working.height),
        "state_dim": int(state_dim),
        "act_dim": int(act_dim),
    }
    return working, manifest_row


def build_dt_dataset_from_logs(
    log_groups: dict[str, Sequence[pl.DataFrame]],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    dataset_frames: list[pl.DataFrame] = []
    episode_index: list[dict[str, Any]] = []
    source_summary: dict[str, dict[str, int]] = {}
    next_episode_id = 0

    for source_policy, episodes in log_groups.items():
        source_rows = 0
        source_episodes = 0
        for episode in episodes:
            normalized, manifest_row = _normalize_episode_dataframe(
                episode,
                source_policy=source_policy,
                episode_id=next_episode_id,
            )
            dataset_frames.append(normalized)
            episode_index.append(manifest_row)
            source_rows += int(normalized.height)
            source_episodes += 1
            next_episode_id += 1
        source_summary[source_policy] = {
            "episodes": source_episodes,
            "rows": source_rows,
        }

    if not dataset_frames:
        raise ValueError("No episodes were collected; dataset would be empty.")

    dataset = pl.concat(dataset_frames, how="diagonal_relaxed")
    state_dims = sorted({row["state_dim"] for row in episode_index})
    act_dims = sorted({row["act_dim"] for row in episode_index})
    manifest = {
        "episode_count": len(episode_index),
        "row_count": int(dataset.height),
        "state_dims": state_dims,
        "act_dims": act_dims,
        "sources": source_summary,
        "episode_index": episode_index,
    }
    return dataset, manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_model_config(
    *,
    action_mode: str,
    context_len: int,
    max_timestep: int,
    output_path: Path,
) -> dict[str, Any]:
    model_kwargs = default_aemo_dt_model_kwargs(
        action_mode=action_mode,
        context_len=context_len,
        max_timestep=max_timestep,
    )
    write_json(output_path, model_kwargs)
    return model_kwargs


def launch_dt_training(
    *,
    dataset_path: Path,
    model_config_path: Path,
    save_path: Path,
    checkpoint_path: Path,
    loss_csv_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    seed: int,
    device: str | None,
    amp_mode: str,
    return_scale: float,
    action_loss_weight: float,
    state_loss_weight: float,
    return_loss_weight: float,
    weight_decay: float,
    num_workers: int,
    prefetch_factor: int,
) -> list[str]:
    script_path = repo_root() / "src" / "pretrain_decision_transformer.py"
    command = [
        sys.executable,
        str(script_path),
        "--data-dir",
        str(dataset_path.parent),
        "--patterns",
        dataset_path.stem,
        "--model-config",
        str(model_config_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--val-split",
        str(val_split),
        "--seed",
        str(seed),
        "--save-path",
        str(save_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--loss-csv-path",
        str(loss_csv_path),
        "--amp-mode",
        amp_mode,
        "--return-scale",
        str(return_scale),
        "--action-loss-weight",
        str(action_loss_weight),
        "--state-loss-weight",
        str(state_loss_weight),
        "--return-loss-weight",
        str(return_loss_weight),
        "--weight-decay",
        str(weight_decay),
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        str(prefetch_factor),
    ]
    if device:
        command.extend(["--device", device])
    subprocess.run(command, check=True)
    return command

def prepare_run_paths(
    *,
    output_dir: str | Path,
    dataset_tag: str,
    dataset_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Path]:
    """Prepare output paths for an AEMO notebook run.

    Args:
        output_dir: Base directory for this run.
        dataset_tag: Prefix used to name dataset and manifest files.
        dataset_path: Optional explicit dataset parquet path.
        manifest_path: Optional explicit manifest JSON path.

    Returns:
        Dict with keys:
        - ``output_dir``: resolved base directory
        - ``raw_dir``: directory for raw policy logs
        - ``dataset_path``: resolved DT dataset parquet path
        - ``manifest_path``: resolved manifest JSON path
    """
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = resolved_output_dir / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolved_dataset_path = (
        Path(dataset_path).resolve()
        if dataset_path is not None
        else (resolved_output_dir / f"{dataset_tag}_dataset.parquet").resolve()
    )
    resolved_manifest_path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else (resolved_output_dir / f"{dataset_tag}_manifest.json").resolve()
    )
    return {
        "output_dir": resolved_output_dir,
        "raw_dir": raw_dir,
        "dataset_path": resolved_dataset_path,
        "manifest_path": resolved_manifest_path,
    }


def build_dispatch_selection(
    *,
    region: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: Path,
    dispatch_station: str | None,
    dispatch_duid: str | None,
    dispatch_index: int,
    battery_capacity: float,
    max_battery_flow: float,
    init_soc: float,
) -> dict[str, Any]:
    battery_units, active_battery_units = list_dispatch_candidates(
        region=region,
        start_date=start_date,
        end_date=end_date,
        station_name=dispatch_station,
        cache_dir=str(cache_dir),
    )
    return resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active_battery_units,
        selected_duid=dispatch_duid,
        selected_index=dispatch_index,
        battery_capacity=battery_capacity,
        max_battery_flow=max_battery_flow,
        init_soc=init_soc,
        apply_unit_sizing=True,
        start_date=start_date,
        end_date=end_date,
        cache_dir=str(cache_dir),
    )


def resolve_dispatch_run_region(
    *,
    dispatch_station: str | None,
    dispatch_duid: str | None,
    start_date: datetime,
    end_date: datetime,
) -> str | None:
    dispatch_target = dispatch_station or dispatch_duid
    if dispatch_target is None:
        return None

    resolution = resolve_battery_duids(dispatch_target, start_date, end_date)
    if not resolution["found"]:
        return None
    return resolution["region"]


def should_run_dispatch_for_scenario(
    *,
    scenario_region: str,
    dispatch_station: str | None,
    dispatch_duid: str | None,
    start_date: datetime,
    end_date: datetime,
) -> tuple[bool, str | None]:
    dispatch_region = resolve_dispatch_run_region(
        dispatch_station=dispatch_station,
        dispatch_duid=dispatch_duid,
        start_date=start_date,
        end_date=end_date,
    )
    if dispatch_region is None:
        return True, None
    return dispatch_region == scenario_region, dispatch_region


def validate_aemo_dt_dimensions(manifest: dict[str, Any], *, action_mode: str) -> None:
    """Validate that an AEMO DT dataset manifest matches expected dimensions.

    Validation rules:
    - ``state_dims`` must be exactly ``[18]``
    - ``act_dims`` must be ``[3]`` for ``multi_market`` and ``[1]`` for ``simple``

    Raises:
        ValueError: If the manifest dimensions do not match the selected action mode.
    """
    expected_act_dim = 3 if action_mode == "multi_market" else 1
    if manifest["state_dims"] != [18]:
        raise ValueError(
            f"AEMO DT dataset expected state_dim=18, found {manifest['state_dims']}."
        )
    if manifest["act_dims"] != [expected_act_dim]:
        raise ValueError(
            f"AEMO DT dataset expected act_dim={expected_act_dim} for action_mode={action_mode!r}, "
            f"found {manifest['act_dims']}."
        )


def train_sb3_model_on_aemo(
    *,
    processed_data: pl.DataFrame,
    algorithm: str,
    battery_variants: Sequence[dict[str, Any]],
    episodes_per_variant: int,
    max_step: int,
    step_duration: float,
    action_mode: str = "multi_market",
    degradation_mode: str = "real_world",
    degradation_chemistry: str = "LFP",
    degradation_temperature: float = 30.0,
    random_episode_start: bool = True,
    eval_battery_variant: dict[str, Any] | None = None,
    test_timesteps: int = 40_000,
    total_timesteps: int = 400_000,
    n_trials: int = 10,
    n_jobs: int = 10,
    default_model: bool = True,
):
    """Train an SB3 model on AEMO environments assembled from battery variants.

    The helper resolves the requested battery variants, builds a list of AEMO
    environment factories for training, creates a separate evaluation
    environment, and then delegates to ``sb3train.train_model``.

    Returns:
        Tuple ``(model, eval_result)`` matching ``sb3train.train_model``.
    """
    resolved_variants = resolve_battery_variants(battery_variants)
    eval_variant = resolve_battery_variants(
        [eval_battery_variant or resolved_variants[0]]
    )[0]
    env_fns = make_aemo_env_fns(
        processed_data=processed_data,
        battery_variants=resolved_variants,
        episodes_per_variant=episodes_per_variant,
        max_step=max_step,
        step_duration=step_duration,
        action_mode=action_mode,
        degradation_mode=degradation_mode,
        degradation_chemistry=degradation_chemistry,
        degradation_temperature=degradation_temperature,
        random_episode_start=random_episode_start,
    )

    def eval_env_fn() -> AEMOBatteryTradingEnv:
        return create_aemo_env(
            processed_data=processed_data,
            battery_variant=eval_variant,
            max_step=max_step,
            step_duration=step_duration,
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
            random_episode_start=random_episode_start,
        )

    return train_model(
        model_class=get_sb3_model_class(algorithm),
        vec_env=env_fns,
        eval_env_fn=eval_env_fn,
        test_timesteps=test_timesteps,
        total_timesteps=total_timesteps,
        n_trials=n_trials,
        n_jobs=n_jobs,
        default_model=default_model,
    )
