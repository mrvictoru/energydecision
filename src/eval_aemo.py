from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
import json
from datetime import datetime
try:
    from .aemo_notebook_utils import (
        fetch_and_preprocess_aemo_data,
        make_aemo_env_fns,
    )
    from .decision import AEMOAgent
    from .helper import evaluate_experiment_logs
    from .eval_common import (
        EvalSummary,
        check_guardrails,
        iso_timestamp,
        load_benchmark,
        load_dt_model,
        read_return_scale,
        write_eval_outputs,
    )
except ImportError:
    from aemo_notebook_utils import (
        fetch_and_preprocess_aemo_data,
        make_aemo_env_fns,
    )
    from decision import AEMOAgent
    from helper import evaluate_experiment_logs
    from eval_common import (
        EvalSummary,
        check_guardrails,
        iso_timestamp,
        load_benchmark,
        load_dt_model,
        read_return_scale,
        write_eval_outputs,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained AEMO Decision Transformer.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--rtg-value", type=float, default=None)
    parser.add_argument("--return-scale", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--save-episodes", action="store_true")
    return parser.parse_args(argv)


def _resolve_model_config(path: str | None, benchmark: dict) -> dict:
    if path is not None:
        with Path(path).resolve().open("r", encoding="utf-8") as fh:
            return json.load(fh)

    return {
        "state_dim": int(benchmark["state_dim"]),
        "act_dim": int(benchmark["act_dim"]),
        "n_block": 4,
        "h_dim": 128,
        "context_len": 288,
        "n_heads": 8,
        "drop_p": 0.1,
        "max_timestep": int(benchmark["max_timestep"]),
        "rope_enabled": True,
        "rope_max_position": 864,
        "rope_base": 10000.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark = load_benchmark(args.benchmark)

    model_config = _resolve_model_config(args.model_config, benchmark)
    model = load_dt_model(args.model_path, model_config, args.device)

    return_scale = read_return_scale(args.model_path, args.return_scale)
    _ = return_scale

    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = benchmark.get("scenario_kwargs", {}).get("cache_dir", "data/aemo_cache")

    test_window = benchmark["test_window"]
    step_duration_hours = float(benchmark.get("step_duration", 30.0)) / 60.0
    processed_data, _ = fetch_and_preprocess_aemo_data(
        region=str(benchmark.get("region", "SA1")),
        start_date=datetime.fromisoformat(test_window["start"]),
        end_date=datetime.fromisoformat(test_window["end"]),
        cache_dir=Path(cache_dir).resolve(),
        step_duration=step_duration_hours,
        refresh=False,
    )

    battery_variants = benchmark.get("battery_variants", [{"capacity": 100.0, "power": 50.0, "initial_soc": 0.5}])
    normalized_variants = []
    for variant in battery_variants:
        normalized_variants.append(
            {
                "capacity_mwh": float(variant.get("capacity", variant.get("capacity_mwh", 100.0))),
                "max_power_mw": float(variant.get("power", variant.get("max_power_mw", 50.0))),
                "init_soc_ratio": float(variant.get("initial_soc", variant.get("init_soc_ratio", 0.5))),
            }
        )

    episode_hours = float(benchmark.get("episode_hours", 24.0))
    max_step = max(1, int(round(episode_hours / step_duration_hours)))

    env_fns = make_aemo_env_fns(
        processed_data=processed_data,
        battery_variants=normalized_variants,
        episodes_per_variant=max(1, int(benchmark.get("eval_episodes", 1))),
        max_step=max_step,
        step_duration=step_duration_hours,
        action_mode=str(benchmark.get("action_mode", "multi_market")),
        degradation_mode=str(benchmark.get("degradation_mode", "real_world")),
        degradation_chemistry=str(benchmark.get("degradation_chemistry", "LFP")),
        degradation_temperature=float(benchmark.get("degradation_temperature", 25.0)),
        random_episode_start=True,
    )

    rtg_value = args.rtg_value
    if rtg_value is None:
        rtg_value = float(benchmark.get("rtg_value", 0.0))

    episode_logs = []
    for idx, fn in enumerate(env_fns):
        env = fn()
        agent = AEMOAgent(
            env,
            algorithm="dt",
            model=model,
            rtg_value=float(rtg_value),
            dt_gamma=float(benchmark.get("discount", 0.99)),
            reset_seed=int(benchmark.get("eval_seed", 42)) + idx,
        )
        ep_df, _ = agent.run_episode(render=False, display_progress=False)
        episode_logs.append(ep_df)

    metrics = evaluate_experiment_logs(episode_logs)
    guardrail_result = check_guardrails(metrics, benchmark.get("guardrails", {}))

    primary_name = str(benchmark.get("primary_metric", "avg_profit_per_episode"))
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
