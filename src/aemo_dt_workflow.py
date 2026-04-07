from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from AEMOBatteryEnv import AEMOBatteryTradingEnv, AEMODataPreprocessor
from aemo_data import fetch_aemo_data_bundle
from decision import AEMOAgent
from dispatch_utils import (
    list_dispatch_candidates,
    resolve_dispatch_selection,
    run_dispatch_replay,
)

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
DEFAULT_BATTERY_COST_PER_KWH = 350.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime {value!r}. Use ISO-8601, e.g. 2024-01-01 or 2024-01-01T00:00:00."
        ) from exc


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return parsed


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


def fetch_and_preprocess_aemo_data(
    *,
    region: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: Path,
    step_duration: float,
    refresh: bool = False,
) -> tuple[pl.DataFrame, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    processed_cache = _processed_cache_path(
        cache_dir=cache_dir,
        region=region,
        start_date=start_date,
        end_date=end_date,
        step_duration=step_duration,
    )
    if processed_cache.exists() and not refresh:
        return pl.read_parquet(str(processed_cache)), processed_cache

    raw_data = fetch_aemo_data_bundle(
        start_date=start_date,
        end_date=end_date,
        region=region,
        fcas_services=DEFAULT_FCAS_SERVICES,
        fuel_types=DEFAULT_FUEL_TYPES,
        cache_dir=str(cache_dir),
    )
    preprocessor = AEMODataPreprocessor(step_duration_hours=step_duration)
    processed_data = preprocessor.preprocess_aemo_data(
        prices=raw_data["prices"],
        fcas=raw_data["fcas"],
        generation=raw_data["generation"],
    )
    processed_data.write_parquet(str(processed_cache))
    return processed_data, processed_cache


def _battery_life_cost(capacity_mwh: float, battery_cost_per_kwh: float) -> float:
    return float(battery_cost_per_kwh) * float(capacity_mwh) * 1000.0


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


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Collect AEMO offline trajectories and optionally train an AEMO-specific Decision Transformer.",
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "both"],
        default="both",
        help="collect = build dataset only, train = train on an existing dataset, both = collect then train.",
    )
    parser.add_argument("--region", type=str, default="SA1")
    parser.add_argument("--start-date", type=parse_datetime, required=True)
    parser.add_argument("--end-date", type=parse_datetime, required=True)
    parser.add_argument(
        "--episode-hours",
        type=positive_float,
        default=None,
        help="Optional fixed episode horizon in hours. If omitted, each episode spans the full date range.",
    )
    parser.add_argument("--step-duration", type=positive_float, default=5 / 60)
    parser.add_argument("--battery-capacity", type=positive_float, default=10.0)
    parser.add_argument("--max-battery-flow", type=positive_float, default=5.0)
    parser.add_argument("--init-soc", type=positive_float, default=5.0)
    parser.add_argument(
        "--battery-cost-per-kwh",
        type=positive_float,
        default=DEFAULT_BATTERY_COST_PER_KWH,
    )
    parser.add_argument(
        "--action-mode",
        choices=["simple", "multi_market"],
        default="multi_market",
    )
    parser.add_argument(
        "--degradation-mode",
        choices=["rainflow", "real_world", "simple"],
        default="real_world",
    )
    parser.add_argument("--degradation-chemistry", choices=["NMC", "LFP"], default="LFP")
    parser.add_argument("--degradation-temperature", type=float, default=30.0)
    parser.add_argument("--num-rule-episodes", type=int, default=8)
    parser.add_argument("--num-dispatch-episodes", type=int, default=4)
    parser.add_argument("--dispatch-station", type=str, default=None)
    parser.add_argument("--dispatch-duid", type=str, default=None)
    parser.add_argument("--dispatch-index", type=int, default=0)
    parser.add_argument("--random-episode-start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--include-log", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "aemo_dt")
    parser.add_argument("--cache-dir", type=Path, default=root / "data" / "aemo")
    parser.add_argument("--dataset-tag", type=str, default="aemo_dt")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional explicit parquet path. Defaults to <output-dir>/<dataset-tag>_dataset.parquet.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit manifest path. Defaults to <output-dir>/<dataset-tag>_manifest.json.",
    )
    parser.add_argument("--context-length", type=positive_int, default=288)
    parser.add_argument("--max-timestep", type=positive_int, default=None)
    parser.add_argument(
        "--model-config-path",
        type=Path,
        default=root / "configs" / "aemo_decision_transformer_model_kwargs.json",
    )
    parser.add_argument("--train-epochs", type=positive_int, default=5)
    parser.add_argument("--train-batch-size", type=positive_int, default=8)
    parser.add_argument("--train-lr", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp-mode", choices=["auto", "on", "off"], default="off")
    parser.add_argument("--return-scale", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.01)
    parser.add_argument("--return-loss-weight", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return parser.parse_args()


def _resolve_dataset_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = args.output_dir.resolve()
    dataset_path = (
        args.dataset_path.resolve()
        if args.dataset_path is not None
        else (output_dir / f"{args.dataset_tag}_dataset.parquet").resolve()
    )
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else (output_dir / f"{args.dataset_tag}_manifest.json").resolve()
    )
    return dataset_path, manifest_path


def _build_dispatch_selection(
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


def _collect_logs(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, manifest_path = _resolve_dataset_paths(args)

    processed_data, processed_cache = fetch_and_preprocess_aemo_data(
        region=args.region,
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir.resolve(),
        step_duration=args.step_duration,
        refresh=args.refresh_cache,
    )
    max_step = _episode_steps(
        processed_rows=processed_data.height,
        step_duration=args.step_duration,
        episode_hours=args.episode_hours,
    )
    max_timestep = args.max_timestep or max_step
    model_kwargs = build_model_config(
        action_mode=args.action_mode,
        context_len=args.context_length,
        max_timestep=max_timestep,
        output_path=args.model_config_path.resolve(),
    )
    battery_life_cost = _battery_life_cost(
        capacity_mwh=args.battery_capacity,
        battery_cost_per_kwh=args.battery_cost_per_kwh,
    )

    log_groups: dict[str, list[pl.DataFrame]] = {}
    raw_outputs: dict[str, str] = {}
    if args.num_rule_episodes > 0:
        rule_episodes = run_rule_episodes(
            processed_data=processed_data,
            num_episodes=args.num_rule_episodes,
            battery_capacity=args.battery_capacity,
            max_battery_flow=args.max_battery_flow,
            init_soc=args.init_soc,
            step_duration=args.step_duration,
            battery_life_cost=battery_life_cost,
            max_step=max_step,
            action_mode=args.action_mode,
            degradation_mode=args.degradation_mode,
            degradation_chemistry=args.degradation_chemistry,
            degradation_temperature=args.degradation_temperature,
            random_episode_start=args.random_episode_start,
            base_seed=args.seed,
        )
        rule_logs_path = raw_dir / f"{args.dataset_tag}_rule_logs.parquet"
        write_combined_episode_logs(episodes=rule_episodes, output_path=rule_logs_path)
        log_groups["rule"] = rule_episodes
        raw_outputs["rule_logs"] = str(rule_logs_path.resolve())

    if args.num_dispatch_episodes > 0:
        if not args.dispatch_station and not args.dispatch_duid:
            raise ValueError(
                "Dispatch replay collection requires --dispatch-station or --dispatch-duid."
            )
        selection = _build_dispatch_selection(
            region=args.region,
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=args.cache_dir.resolve(),
            dispatch_station=args.dispatch_station,
            dispatch_duid=args.dispatch_duid,
            dispatch_index=args.dispatch_index,
            battery_capacity=args.battery_capacity,
            max_battery_flow=args.max_battery_flow,
            init_soc=args.init_soc,
        )
        dispatch_episodes, dispatch_incidents, _ = run_dispatch_replay(
            processed_data=processed_data,
            selection=selection,
            start_date=args.start_date,
            end_date=args.end_date,
            region=args.region,
            cache_dir=str(args.cache_dir.resolve()),
            num_episodes=args.num_dispatch_episodes,
            step_duration=args.step_duration,
            battery_life_cost=battery_life_cost,
            max_step=max_step,
            output_dir=str(raw_dir),
            run_tag=f"{args.dataset_tag}_dispatch",
            action_mode=args.action_mode,
            degradation_mode=args.degradation_mode,
            degradation_chemistry=args.degradation_chemistry,
            degradation_temperature=args.degradation_temperature,
        )
        log_groups["dispatch"] = dispatch_episodes
        raw_outputs["dispatch_logs"] = str(
            (raw_dir / f"{args.dataset_tag}_dispatch_dispatch_logs.parquet").resolve()
        )
        if dispatch_incidents:
            raw_outputs["dispatch_incident_logs"] = str(
                (raw_dir / f"{args.dataset_tag}_dispatch_dispatch_incident_logs.parquet").resolve()
            )

    for extra_log in args.include_log:
        extra_path = Path(extra_log).resolve()
        log_groups[extra_path.stem] = load_episode_logs_from_parquet(extra_path)
        raw_outputs[f"extra::{extra_path.stem}"] = str(extra_path)

    dataset, manifest = build_dt_dataset_from_logs(log_groups)
    expected_act_dim = 3 if args.action_mode == "multi_market" else 1
    if manifest["state_dims"] != [18]:
        raise ValueError(
            f"AEMO DT dataset expected state_dim=18, found {manifest['state_dims']}."
        )
    if manifest["act_dims"] != [expected_act_dim]:
        raise ValueError(
            f"AEMO DT dataset expected act_dim={expected_act_dim} for action_mode={args.action_mode!r}, "
            f"found {manifest['act_dims']}."
        )
    dataset.write_parquet(str(dataset_path))
    manifest.update(
        {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_path": str(dataset_path),
            "processed_cache": str(processed_cache.resolve()),
            "region": args.region,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "episode_hours": args.episode_hours,
            "step_duration": args.step_duration,
            "max_step": max_step,
            "action_mode": args.action_mode,
            "degradation_mode": args.degradation_mode,
            "degradation_chemistry": args.degradation_chemistry,
            "degradation_temperature": args.degradation_temperature,
            "battery_capacity": args.battery_capacity,
            "max_battery_flow": args.max_battery_flow,
            "init_soc": args.init_soc,
            "battery_life_cost": battery_life_cost,
            "random_episode_start": args.random_episode_start,
            "seed": args.seed,
            "model_config_path": str(args.model_config_path.resolve()),
            "model_kwargs": model_kwargs,
            "raw_outputs": raw_outputs,
        }
    )
    write_json(manifest_path, manifest)
    return dataset_path, manifest_path, manifest


def main() -> None:
    args = parse_args()
    dataset_path, manifest_path = _resolve_dataset_paths(args)
    manifest: dict[str, Any] | None = None

    if args.mode in {"collect", "both"}:
        dataset_path, manifest_path, manifest = _collect_logs(args)
        print(f"[OK] Wrote dataset: {dataset_path}")
        print(f"[OK] Wrote manifest: {manifest_path}")

    if args.mode in {"train", "both"}:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset parquet not found: {dataset_path}. Run with --mode collect or --mode both first."
            )
        if not args.model_config_path.resolve().exists():
            inferred_max_timestep = args.max_timestep or (
                manifest["max_step"] if manifest is not None else args.context_length
            )
            build_model_config(
                action_mode=args.action_mode,
                context_len=args.context_length,
                max_timestep=inferred_max_timestep,
                output_path=args.model_config_path.resolve(),
            )
        save_path = (args.output_dir.resolve() / f"{args.dataset_tag}_dt_model.pt").resolve()
        checkpoint_path = (
            args.output_dir.resolve() / f"{args.dataset_tag}_dt_checkpoint.pt"
        ).resolve()
        loss_csv_path = (
            args.output_dir.resolve() / f"{args.dataset_tag}_dt_loss_history.csv"
        ).resolve()
        command = launch_dt_training(
            dataset_path=dataset_path,
            model_config_path=args.model_config_path.resolve(),
            save_path=save_path,
            checkpoint_path=checkpoint_path,
            loss_csv_path=loss_csv_path,
            epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            lr=args.train_lr,
            val_split=args.val_split,
            seed=args.seed,
            device=args.device,
            amp_mode=args.amp_mode,
            return_scale=args.return_scale,
            action_loss_weight=args.action_loss_weight,
            state_loss_weight=args.state_loss_weight,
            return_loss_weight=args.return_loss_weight,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        print("[OK] Training command completed:")
        print(" ".join(command))


if __name__ == "__main__":
    main()
