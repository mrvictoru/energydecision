from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from EnergySimEnv import SolarBatteryEnv
from decision import Agent, run_episodes_parallel
from helper import evaluate_experiment_logs
from eval_common import (
    EvalSummary,
    check_guardrails,
    collect_parquet_by_patterns,
    iso_timestamp,
    load_benchmark,
    load_dt_model,
    read_return_scale,
    split_episode_logs,
    write_eval_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained household Decision Transformer.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--rtg-value", type=float, default=None)
    parser.add_argument("--return-scale", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--save-episodes", action="store_true")
    return parser.parse_args(argv)


def _resolve_model_config(path: str | None, benchmark: dict) -> dict:
    if path is not None:
        with Path(path).resolve().open("r", encoding="utf-8") as fh:
            return json.load(fh)

    return {
        "state_dim": int(benchmark["state_dim"]),
        "act_dim": int(benchmark["act_dim"]),
        "n_block": 2,
        "h_dim": 128,
        "context_len": 60,
        "n_heads": 8,
        "drop_p": 0.1,
        "max_timestep": int(benchmark["max_timestep"]),
        "rope_enabled": False,
        "rope_max_position": 180,
        "rope_base": 10000.0,
    }


def _build_household_df(log_df: pl.DataFrame) -> pl.DataFrame:
    if "raw_observation" not in log_df.columns:
        raise ValueError("Expected raw_observation column in household parquet logs.")

    rows = []
    for idx, raw in enumerate(log_df.get_column("raw_observation").to_list()):
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size < 10:
            continue
        rows.append(
            {
                "Timestamp": idx,
                "Time": f"2021-01-01T{(idx // 2) % 24:02d}:{(idx % 2) * 30:02d}:00",
                "SolarGen": float(arr[4]),
                "HouseLoad": float(arr[5]),
                "FutureSolar": float(arr[6]),
                "FutureLoad": float(arr[7]),
                "ImportEnergyPrice": float(arr[8]),
                "ExportEnergyPrice": float(arr[9]),
            }
        )

    if not rows:
        raise ValueError("Could not reconstruct household environment dataframe from raw_observation.")
    return pl.DataFrame(rows)


def _run_sequential(envs: list[SolarBatteryEnv], agent_kwargs: dict) -> list[pl.DataFrame]:
    logs: list[pl.DataFrame] = []
    for env in envs:
        agent = Agent(env, **agent_kwargs)
        ep_df, _ = agent.run_episode(render=False, display_progress=False)
        logs.append(ep_df)
    return logs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark = load_benchmark(args.benchmark)

    model_config = _resolve_model_config(args.model_config, benchmark)
    model = load_dt_model(args.model_path, model_config, args.device)

    return_scale = read_return_scale(args.model_path, args.return_scale)
    _ = return_scale  # kept for compatibility with training sidecar conventions

    test_files = collect_parquet_by_patterns(benchmark["data_dir"], benchmark.get("test_patterns", []))
    if not test_files:
        raise FileNotFoundError("No test parquet files matched benchmark test_patterns.")

    episodes_target = int(benchmark.get("eval_episodes", 1))
    episode_seed = int(benchmark.get("eval_seed", 42))

    envs: list[SolarBatteryEnv] = []
    for parquet in test_files:
        raw_df = pl.read_parquet(str(parquet))
        for episode_df in split_episode_logs(raw_df):
            if len(envs) >= episodes_target:
                break
            env_df = _build_household_df(episode_df)
            env_kwargs = dict(benchmark.get("env_kwargs", {}))
            allowed_env_kwargs = {
                "battery_capacity",
                "max_battery_flow",
                "max_grid_flow",
                "render_mode",
                "battery_life_cost",
                "base_deg_DoD",
                "step_duration",
                "degradation_temperature",
            }
            env = SolarBatteryEnv(
                df=env_df,
                max_step=max(1, env_df.height - 1),
                init_battery_level=float(env_kwargs.get("battery_capacity", 7.0)) * 0.5,
                **{k: v for k, v in env_kwargs.items() if k in allowed_env_kwargs},
            )
            envs.append(env)
        if len(envs) >= episodes_target:
            break

    if not envs:
        raise RuntimeError("Failed to construct any household test environments.")

    rtg_value = args.rtg_value
    if rtg_value is None:
        rtg_value = float(benchmark.get("rtg_value", 0.0))

    agent_kwargs = {
        "algorithm": "dt",
        "model": model,
        "rtg_value": float(rtg_value),
        "dt_gamma": float(benchmark.get("discount", 0.99)),
        "reset_seed": episode_seed,
    }

    if int(args.num_workers) <= 1:
        episode_logs = _run_sequential(envs, agent_kwargs)
    else:
        episode_logs, _ = run_episodes_parallel(
            Agent,
            envs,
            agent_kwargs=agent_kwargs,
            render=False,
            max_workers=int(args.num_workers),
            use_notebook_tqdm=False,
            display_indi_prog=False,
        )

    metrics = evaluate_experiment_logs(episode_logs)
    guardrail_result = check_guardrails(metrics, benchmark.get("guardrails", {}))

    primary_name = str(benchmark.get("primary_metric", "mean_reward"))
    summary = EvalSummary(
        primary_metric_name=primary_name,
        primary_metric_value=metrics.get(primary_name),
        guardrails_passed=bool(guardrail_result["passed"]),
        guardrail_details=guardrail_result["details"],
        model_path=str(Path(args.model_path).resolve()),
        benchmark_path=str(Path(args.benchmark).resolve()),
        timestamp=iso_timestamp(),
    )

    write_eval_outputs(args.output_dir, metrics, summary)

    if args.save_episodes:
        out = Path(args.output_dir).resolve() / "episode_logs"
        out.mkdir(parents=True, exist_ok=True)
        for idx, df in enumerate(episode_logs):
            df.write_parquet(str(out / f"episode_{idx:03d}.parquet"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
