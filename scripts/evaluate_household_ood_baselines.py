#!/usr/bin/env python3
"""Evaluate legacy household policies on gap-separated real-data OOD segments."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from EnergySimEnv import SolarBatteryEnv
from decision import Agent, _build_dt_inference_context, stable_rtg_update
from decision_transformer import DecisionTransformer
from household_ingest import build_year_dataset, env_view, split_segments
from household_forecast import apply_forecast_sidecar
from household_optimization import (
    apply_tariff,
    bootstrap_mean_ci,
    build_j_t_soc_prompt_provider,
    optimize_dispatch,
)
from household_replay import Tariff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", type=Path, default=ROOT / "data/household/real/normalized")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval_output/household/ood_baselines")
    parser.add_argument("--ppo-path", type=Path, default=ROOT / "models/household/sb3/ppo_model.zip")
    parser.add_argument("--dt-path", type=Path, default=ROOT / "models/household/dt/dt_model_new_best.pt")
    parser.add_argument("--dt-config", type=Path, default=ROOT / "models/household/dt/decision_transformer_model_kwargs.json")
    parser.add_argument("--dt-rtg-mode", choices=("standard", "j_t_soc"), default="standard")
    parser.add_argument(
        "--dt-rtg-value",
        type=float,
        default=0.0,
        help="Fixed initial RTG prompt for --dt-rtg-mode standard.",
    )
    parser.add_argument(
        "--forecast-mode",
        choices=("persistence", "zero", "shuffle"),
        default="persistence",
        help="Transform FutureSolar/FutureLoad for forecast ablation.",
    )
    parser.add_argument(
        "--forecast-sidecar",
        type=Path,
        default=None,
        help="Timestamp-keyed parquet whose FutureSolar/FutureLoad replace ingested forecasts.",
    )
    parser.add_argument(
        "--additional-dt",
        action="append",
        default=[],
        metavar="NAME:CHECKPOINT:CONFIG:RTG_MODE",
        help="Additional DT evaluated in the same run; RTG_MODE is standard or j_t_soc.",
    )
    parser.add_argument("--tariff", choices=("legacy_flat", "realistic"), default="legacy_flat")
    parser.add_argument("--capacity-kwh", type=float, default=5.0)
    parser.add_argument("--max-flow-kw", type=float, default=3.3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-vram-fraction", type=float, default=0.90)
    parser.add_argument("--skip-dt", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Evaluate deterministic complete-day windows instead of entire segments.",
    )
    parser.add_argument(
        "--windows-per-segment",
        type=int,
        default=1,
        help="Evenly spaced windows selected from each segment when --window-days is set.",
    )
    parser.add_argument(
        "--skip-reference-policies",
        action="store_true",
        help="Skip invariant rule/oracle rollouts for policy ablations.",
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=None,
        help="Evaluate on synthetic corpus episodes (with per-episode battery "
             "config from the manifest) instead of the real normalized telemetry.",
    )
    parser.add_argument("--synth-split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--limit-windows",
        type=int,
        default=None,
        help="Deterministically subsample this many evenly spaced evaluation windows.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of parallel worker processes for CPU-bound evaluations (rule, oracle) (default: min(8, cpu_count)).",
    )
    parser.add_argument(
        "--batch-eval",
        action="store_true",
        help="Step active evaluation windows concurrently with batched DT forward passes on GPU.",
    )
    return parser.parse_args()


def _bill_from_logs(frame: pl.DataFrame, logs: pl.DataFrame) -> float:
    """Price env-physical grid energy with import/export prices from the frame."""
    rows = logs["info"].to_list()
    bill = 0.0
    for index, info in enumerate(rows):
        grid_energy = float(info["grid_energy"])
        if grid_energy >= 0:
            bill += grid_energy * float(frame["ImportEnergyPrice"][index])
        else:
            bill += grid_energy * float(frame["ExportEnergyPrice"][index])
    return bill


def _degradation_cost_from_logs(logs: pl.DataFrame, battery_life_cost: float) -> float:
    """Accumulate the environment's realized per-step degradation cost."""
    if "info" not in logs.columns:
        return 0.0
    return float(sum(
        float(info.get("step_degradation", 0.0) or 0.0) * battery_life_cost
        for info in logs["info"].to_list()
    ))


def tariff_for_name(name: str) -> Tariff:
    if name == "realistic":
        return Tariff(31.042, 1.0, 11, 14)
    return Tariff(30.0, 5.0, 24, 24)


def _run_agent(
    frame: pl.DataFrame, algorithm: str, model, capacity: float, flow: float,
    tariff: Tariff, rtg_mode: str = "standard", rtg_value: float = 0.0,
) -> tuple[float, float]:
    env = SolarBatteryEnv(
        frame, battery_capacity=capacity, max_battery_flow=flow,
        init_battery_level=capacity / 2.0, max_step=len(frame),
    )
    prompt_provider = (
        build_j_t_soc_prompt_provider(
            frame, tariff=tariff, capacity_kwh=capacity, max_flow_kw=flow
        )
        if algorithm == "dt" and rtg_mode == "j_t_soc" else None
    )
    logs, _ = Agent(env, algorithm=algorithm, model=model, horizon=288,
                    soc_resolution=31, action_resolution=21,
                    rtg_value=rtg_value,
                    rtg_prompt_provider=prompt_provider).run_episode()
    return (
        _bill_from_logs(frame, logs),
        _degradation_cost_from_logs(logs, env.battery_life_cost),
    )


class _BatchedEnvState:
    def __init__(
        self,
        idx: int,
        frame: pl.DataFrame,
        capacity: float,
        flow: float,
        tariff: Tariff,
        model: DecisionTransformer,
        rtg_mode: str,
        rtg_value: float,
    ) -> None:
        self.idx = idx
        self.frame = frame
        self.env = SolarBatteryEnv(
            frame,
            battery_capacity=capacity,
            max_battery_flow=flow,
            init_battery_level=capacity / 2.0,
            max_step=len(frame),
        )
        self.prompt_provider = (
            build_j_t_soc_prompt_provider(
                frame, tariff=tariff, capacity_kwh=capacity, max_flow_kw=flow
            )
            if rtg_mode == "j_t_soc"
            else None
        )
        obs, _ = self.env.reset()
        self.obs = obs
        self.dt_states_buffer = [obs.copy()]
        self.dt_actions_buffer = [np.zeros(model.act_dim)]
        init_rtg = (
            self.prompt_provider(float(self.env.battery_level), int(self.env.current_step))
            if self.prompt_provider is not None
            else rtg_value
        )
        self.dt_rtgs_buffer = [float(init_rtg)]
        self.dt_timesteps_buffer = [self.env.current_step]
        self.logs: list[dict[str, object]] = []


def _run_batched_dt(
    segments: list[pl.DataFrame],
    model: DecisionTransformer,
    window_batteries: list[dict[str, float]],
    tariff: Tariff,
    rtg_mode: str = "standard",
    rtg_value: float = 0.0,
    device: str = "cuda",
) -> tuple[list[float], list[float]]:
    model.eval()
    dev = next(model.parameters()).device
    active = [
        _BatchedEnvState(
            idx=i,
            frame=segments[i],
            capacity=window_batteries[i]["capacity_kwh"],
            flow=window_batteries[i]["max_flow_kw"],
            tariff=tariff,
            model=model,
            rtg_mode=rtg_mode,
            rtg_value=rtg_value,
        )
        for i in range(len(segments))
    ]
    results: list[float] = [0.0] * len(segments)
    degradation_costs: list[float] = [0.0] * len(segments)

    step = 0
    while active:
        for item in active:
            if item.prompt_provider is not None:
                item.dt_rtgs_buffer[-1] = float(
                    item.prompt_provider(float(item.env.battery_level), int(item.env.current_step))
                )

        s_list, a_list, r_list, t_list, m_list = [], [], [], [], []
        for item in active:
            s, a, r, t, m = _build_dt_inference_context(
                model, item.dt_states_buffer, item.dt_actions_buffer,
                item.dt_rtgs_buffer, item.dt_timesteps_buffer,
            )
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            t_list.append(t)
            m_list.append(m)

        b_s = torch.tensor(np.stack(s_list), dtype=torch.float32, device=dev)
        b_a = torch.tensor(np.stack(a_list), dtype=torch.float32, device=dev)
        b_r = torch.tensor(np.stack(r_list), dtype=torch.float32, device=dev).unsqueeze(-1)
        b_t = torch.tensor(np.stack(t_list), dtype=torch.long, device=dev)
        b_m = torch.tensor(np.stack(m_list), dtype=torch.bool, device=dev)

        with torch.no_grad():
            preds = model.get_action(b_s, b_a, b_r, b_t, attention_mask=b_m)
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
        preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy()
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        if preds.ndim == 1:
            preds = preds.reshape(-1, model.act_dim)

        still_active = []
        for i, item in enumerate(active):
            act = preds[i].tolist()
            if not isinstance(act, list):
                act = [act]
            next_obs, reward, term, trunc, info = item.env.step(act)
            item.logs.append({
                "step": step,
                "norm_observation": item.obs.tolist(),
                "raw_observation": item.env.get_raw_obs().tolist(),
                "action": act,
                "reward": reward,
                "info": info,
            })

            item.dt_actions_buffer[-1] = np.array(act, dtype=np.float32)
            next_rtg = stable_rtg_update(item.dt_rtgs_buffer[-1], reward, dt_gamma=0.99, initial_rtg=rtg_value)
            item.dt_states_buffer.append(next_obs.copy())
            item.dt_actions_buffer.append(np.zeros(model.act_dim))
            item.dt_rtgs_buffer.append(next_rtg)
            item.dt_timesteps_buffer.append(item.env.current_step)
            if len(item.dt_states_buffer) > model.context_len:
                item.dt_states_buffer = item.dt_states_buffer[-model.context_len:]
                item.dt_actions_buffer = item.dt_actions_buffer[-model.context_len:]
                item.dt_rtgs_buffer = item.dt_rtgs_buffer[-model.context_len:]
                item.dt_timesteps_buffer = item.dt_timesteps_buffer[-model.context_len:]
            item.obs = next_obs

            if term or trunc:
                ep_df = pl.DataFrame(item.logs)
                results[item.idx] = _bill_from_logs(item.frame, ep_df)
                degradation_costs[item.idx] = _degradation_cost_from_logs(
                    ep_df, item.env.battery_life_cost
                )
            else:
                still_active.append(item)
        active = still_active
        step += 1

    return results, degradation_costs


def _oracle_bill(frame: pl.DataFrame, capacity: float, flow: float, tariff: Tariff) -> float:
    """Perfect-foresight daily DP, retaining the segment boundary discipline."""
    raw_kw = frame.with_columns([
        (pl.col("HouseLoad") * 12.0).alias("HouseLoad"),
        (pl.col("SolarGen") * 12.0).alias("SolarGen"),
    ])
    costs = []
    with_date = raw_kw.with_columns(pl.col("Timestamp").dt.date().alias("_date"))
    for day in with_date.partition_by("_date", maintain_order=True):
        if len(day) == 288:
            costs.append(optimize_dispatch(
                day.drop("_date"), tariff=tariff, capacity_kwh=capacity,
                max_flow_kw=flow, roundtrip_eff=1.0,
            ).bill_aud)
    return float(sum(costs))


def _oracle_worker(args: tuple[pl.DataFrame, float, float, Tariff]) -> float:
    frame, capacity, flow, tariff = args
    return _oracle_bill(frame, capacity, flow, tariff)


def _rule_worker(args: tuple[pl.DataFrame, float, float, Tariff]) -> tuple[float, float]:
    frame, capacity, flow, tariff = args
    return _run_agent(frame, "rule", None, capacity, flow, tariff)


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return name


def _apply_forecast_mode(frame: pl.DataFrame, mode: str, seed: int) -> pl.DataFrame:
    if mode == "persistence":
        return frame
    if mode == "zero":
        return frame.with_columns([
            pl.lit(0.0).alias("FutureSolar"),
            pl.lit(0.0).alias("FutureLoad"),
        ])
    rng = np.random.default_rng(seed)
    return frame.with_columns([
        pl.Series("FutureSolar", rng.permutation(frame["FutureSolar"].to_numpy())),
        pl.Series("FutureLoad", rng.permutation(frame["FutureLoad"].to_numpy())),
    ])


def _bounded_windows(
    segments: list[pl.DataFrame],
    window_days: int | None,
    windows_per_segment: int,
) -> tuple[list[pl.DataFrame], list[dict[str, object]]]:
    if window_days is None:
        return segments, [
            {
                "source_segment": index,
                "start": str(segment["Timestamp"][0]),
                "end": str(segment["Timestamp"][-1]),
                "days": _duration_days(segment),
            }
            for index, segment in enumerate(segments)
        ]
    if window_days < 1 or windows_per_segment < 1:
        raise ValueError("--window-days and --windows-per-segment must be positive")
    windows: list[pl.DataFrame] = []
    provenance: list[dict[str, object]] = []
    for segment_index, segment in enumerate(segments):
        dated = segment.with_columns(pl.col("Timestamp").dt.date().alias("_window_date"))
        days = [
            day.drop("_window_date")
            for day in dated.partition_by("_window_date", maintain_order=True)
        ]
        eligible = []
        for start in range(max(0, len(days) - window_days + 1)):
            dates = [day["Timestamp"][0].date() for day in days[start:start + window_days]]
            if all(
                dates[index] + dt.timedelta(days=1) == dates[index + 1]
                for index in range(len(dates) - 1)
            ):
                eligible.append(start)
        if not eligible:
            continue
        selected = np.linspace(0, len(eligible) - 1, min(windows_per_segment, len(eligible)))
        for selected_index in sorted({int(round(value)) for value in selected}):
            start = eligible[selected_index]
            window = pl.concat(days[start:start + window_days])
            windows.append(window)
            provenance.append({
                "source_segment": segment_index,
                "start": str(window["Timestamp"][0]),
                "end": str(window["Timestamp"][-1]),
                "days": window_days,
            })
    if not windows:
        raise ValueError("No complete bounded windows fit the requested surface")
    return windows, provenance


def _subsample_windows(
    segments: list[pl.DataFrame],
    provenance: list[dict[str, object]],
    batteries: list[dict[str, float]],
    limit: int | None,
) -> tuple[list[pl.DataFrame], list[dict[str, object]], list[dict[str, float]]]:
    if limit is None:
        return segments, provenance, batteries
    if limit < 1:
        raise ValueError("--limit-windows must be positive")
    selected = sorted({
        int(round(value))
        for value in np.linspace(0, len(segments) - 1, min(limit, len(segments)))
    })
    return (
        [segments[index] for index in selected],
        [provenance[index] for index in selected],
        [batteries[index] for index in selected],
    )


def _duration_days(frame: pl.DataFrame) -> float:
    timestamps = frame["Timestamp"]
    if len(timestamps) < 2:
        raise ValueError("Evaluation windows require at least two timestamps")
    deltas = np.diff(timestamps.to_numpy()).astype("timedelta64[ns]").astype(np.int64)
    positive = deltas[deltas > 0]
    if len(positive) == 0:
        raise ValueError("Evaluation timestamps must increase")
    step_ns = int(np.median(positive))
    duration_ns = int(
        (timestamps[-1] - timestamps[0]).total_seconds() * 1_000_000_000
    ) + step_ns
    return duration_ns / (86_400 * 1_000_000_000)


def _load_dt(path: Path, config: Path, device: str) -> DecisionTransformer:
    model = DecisionTransformer(**json.loads(config.read_text()))
    model.load_from_checkpoint(str(path), map_location=device)
    model.to(device)
    model.eval()
    return model


def _parse_additional_dt(spec: str) -> tuple[str, Path, Path, str]:
    parts = spec.split(":", 3)
    if len(parts) != 4 or parts[3] not in {"standard", "j_t_soc"}:
        raise ValueError("--additional-dt must be NAME:CHECKPOINT:CONFIG:RTG_MODE")
    return parts[0], Path(parts[1]), Path(parts[2]), parts[3]


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if free_bytes < total_bytes * (1.0 - args.max_vram_fraction):
            raise RuntimeError(
                f"Insufficient free CUDA memory: {free_bytes / 1024**3:.2f} GiB free "
                f"of {total_bytes / 1024**3:.2f} GiB; lower --max-vram-fraction or clear the GPU."
            )
        print(
            f"Using CUDA for policy inference: {torch.cuda.get_device_name(0)}; "
            f"{free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB free",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tariff = tariff_for_name(args.tariff)
    window_batteries: list[dict[str, float]] = []
    if args.synth_dir is not None:
        if args.forecast_sidecar is not None:
            raise ValueError("--forecast-sidecar applies to the real OOD surface only")
        corpus_manifest = json.loads((args.synth_dir / "manifest.json").read_text())
        corpus_entries = [
            entry for entry in corpus_manifest["episodes"]
            if entry["split"] == args.synth_split
        ]
        base_segments = [
            apply_tariff(pl.read_parquet(args.synth_dir / entry["path"]), tariff)
            for entry in corpus_entries
        ]
        corpus_batteries = [
            {
                "capacity_kwh": float(entry["battery"]["capacity_kwh"]),
                "max_flow_kw": float(entry["battery"]["max_flow_kw"]),
            }
            for entry in corpus_entries
        ]
    else:
        base_segments = [
            apply_tariff(env_view(segment), tariff)
            for segment in split_segments(build_year_dataset(args.normalized_dir))
        ]
        corpus_batteries = []
        if args.forecast_sidecar is not None:
            sidecar = pl.read_parquet(args.forecast_sidecar)
            base_segments = [
                apply_forecast_sidecar(segment, sidecar)
                for segment in base_segments
            ]
    segments, window_provenance = _bounded_windows(
        base_segments, args.window_days, args.windows_per_segment
    )
    window_batteries = [
        corpus_batteries[provenance["source_segment"]]
        if corpus_batteries
        else {
            "capacity_kwh": args.capacity_kwh,
            "max_flow_kw": args.max_flow_kw,
        }
        for provenance in window_provenance
    ]
    segments, window_provenance, window_batteries = _subsample_windows(
        segments, window_provenance, window_batteries, args.limit_windows
    )
    if args.synth_dir is not None:
        for provenance in window_provenance:
            entry = corpus_entries[provenance["source_segment"]]
            provenance["horizon"] = entry.get("horizon")
            provenance["archetype"] = entry.get("archetype")
            provenance["episode_path"] = entry["path"]
    segments = [
        _apply_forecast_mode(segment, args.forecast_mode, args.seed + index)
        for index, segment in enumerate(segments)
    ]
    if not segments:
        raise ValueError("No contiguous evaluation segments found")

    models: dict[str, tuple[object, str]] = {}
    if not args.skip_ppo:
        models["ppo"] = (PPO.load(str(args.ppo_path), device=device), "standard")
    if not args.skip_dt:
        models["dt"] = (_load_dt(args.dt_path, args.dt_config, device), args.dt_rtg_mode)
    for spec in args.additional_dt:
        name, checkpoint, config, rtg_mode = _parse_additional_dt(spec)
        if name in models or name in {"rule", "oracle", "no_battery"}:
            raise ValueError(f"Duplicate or reserved policy name: {name}")
        models[name] = (_load_dt(checkpoint, config, device), rtg_mode)

    bills: dict[str, list[float]] = {}
    degradation_costs: dict[str, list[float]] = {}
    if not args.skip_reference_policies:
        bills.update({"rule": [], "oracle": []})
        print(
            f"Evaluating reference policies across {len(segments)} windows (workers={args.workers})...",
            flush=True,
        )
        if args.workers > 1 and len(segments) > 1:
            ref_args = [
                (segment, window_batteries[idx]["capacity_kwh"], window_batteries[idx]["max_flow_kw"], tariff)
                for idx, segment in enumerate(segments)
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                bills["oracle"] = list(executor.map(_oracle_worker, ref_args))
                rule_results = list(executor.map(_rule_worker, ref_args))
                bills["rule"] = [result[0] for result in rule_results]
                degradation_costs["rule"] = [result[1] for result in rule_results]
        else:
            for idx, segment in enumerate(segments):
                battery = window_batteries[idx]
                rule_bill, rule_deg_cost = _run_agent(
                    segment, "rule", None, battery["capacity_kwh"], battery["max_flow_kw"], tariff
                )
                bills["rule"].append(rule_bill)
                degradation_costs.setdefault("rule", []).append(rule_deg_cost)
                bills["oracle"].append(
                    _oracle_bill(segment, battery["capacity_kwh"], battery["max_flow_kw"], tariff)
                )

    for name, (model, rtg_mode) in models.items():
        bills[name] = []
        degradation_costs[name] = []
        is_dt = isinstance(model, DecisionTransformer)
        if is_dt and args.batch_eval and len(segments) > 1:
            print(
                f"Evaluating policy '{name}' across {len(segments)} windows with batched DT stepping...",
                flush=True,
            )
            bills[name], degradation_costs[name] = _run_batched_dt(
                segments, model, window_batteries, tariff,
                rtg_mode=rtg_mode, rtg_value=args.dt_rtg_value, device=device,
            )
        else:
            print(f"Evaluating policy '{name}' across {len(segments)} windows...", flush=True)
            for segment_index, segment in enumerate(segments, start=1):
                forecast_label = "sidecar" if args.forecast_sidecar is not None else args.forecast_mode
                battery = window_batteries[segment_index - 1]
                print(
                    f"  [{name}] Window {segment_index}/{len(segments)} "
                    f"({_duration_days(segment):.0f} days, forecast={forecast_label}, "
                    f"capacity={battery['capacity_kwh']}kWh/{battery['max_flow_kw']}kW)",
                    flush=True,
                )
                bill, deg_cost = _run_agent(
                    segment, "rl" if name == "ppo" else "dt", model,
                    battery["capacity_kwh"], battery["max_flow_kw"], tariff, rtg_mode,
                    args.dt_rtg_value,
                )
                bills[name].append(bill)
                degradation_costs[name].append(deg_cost)

    no_battery = []
    for segment in segments:
        grid = segment["HouseLoad"] - segment["SolarGen"]
        no_battery.append(float(
            (grid.clip(lower_bound=0.0) * segment["ImportEnergyPrice"]).sum()
            - ((-grid).clip(lower_bound=0.0) * segment["ExportEnergyPrice"]).sum()
        ))
    segment_days = [_duration_days(segment) for segment in segments]
    output = {
        "surface": (
            f"synthetic corpus {args.synth_dir} split={args.synth_split}; "
            "per-episode battery configuration; deterministic bounded windows"
            if args.synth_dir is not None
            else "real normalized telemetry; held-out OOD; each environment instantiated per contiguous segment"
        ),
        "hardware": (
            {"per_window": window_batteries}
            if args.synth_dir is not None
            else {"capacity_kwh": args.capacity_kwh, "max_flow_kw": args.max_flow_kw}
        ),
        "tariff": args.tariff,
        "forecast_mode": args.forecast_mode,
        "dt_rtg_value": args.dt_rtg_value,
        "forecast_sidecar": (
            str(args.forecast_sidecar.resolve()) if args.forecast_sidecar is not None else None
        ),
        "segments": len(segments),
        "window_days": args.window_days,
        "windows_per_segment": args.windows_per_segment,
        "windows": window_provenance,
        "results": {},
    }
    for name, values in {**bills, "no_battery": no_battery}.items():
        annualized = [bill / days * 365.0 for bill, days in zip(values, segment_days)]
        result = {
            "bill_aud_per_year": bootstrap_mean_ci(annualized, n_bootstrap=args.bootstrap, seed=args.seed),
            "segment_bills_aud": values,
        }
        if name in degradation_costs:
            deg_values = degradation_costs[name]
            net_bills = [bill + deg for bill, deg in zip(values, deg_values)]
            net_annualized = [
                net_bill / days * 365.0
                for net_bill, days in zip(net_bills, segment_days)
            ]
            net_savings = [
                (base_bill - net_bill) / days * 365.0
                for base_bill, net_bill, days in zip(no_battery, net_bills, segment_days)
            ]
            result.update({
                "segment_deg_cost_aud": deg_values,
                "net_bill_aud": net_bills,
                "net_bill_aud_per_year": bootstrap_mean_ci(
                    net_annualized, n_bootstrap=args.bootstrap, seed=args.seed
                ),
                "net_savings_vs_no_battery_aud_per_year": float(np.mean(net_savings)),
                "net_savings_segment_aud_per_year": net_savings,
            })
        output["results"][name] = result
    base = np.asarray(output["results"]["no_battery"]["bill_aud_per_year"]["mean"])
    for name in bills:
        output["results"][name]["savings_vs_no_battery_aud_per_year"] = float(
            base - output["results"][name]["bill_aud_per_year"]["mean"]
        )
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
