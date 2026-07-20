#!/usr/bin/env python3
"""
Generate SDP-optimal energy-arbitrage trajectories for the AEMO environment.

Output is a Parquet file compatible with the existing FCAS-rich dataset
(data/aemo_dt_fcas/aemo_fcas_dataset.parquet) — same column schema:
step, norm_observation, action, reward, source_policy, episode_id.

Actions are padded to 9D (full_fcas) with FCAS dims = 0 so they can be
trained alongside the FCAS-rich dataset.

Usage:
    python3 scripts/generate_sdp_aemo_trajectories.py \
        --regions NSW1,SA1,QLD1 \
        --start-date 2024-01-01 --end-date 2024-07-01 \
        --episode-hours 48 --episodes-per-region 10 \
        --output data/aemo_dt_sdp/aemo_sdp_trajectories.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from AEMOBatteryEnv import AEMOBatteryTradingEnv
from aemo_sdp_solver import AEMOSDPSolver


# ---------------------------------------------------------------------------
# Battery configs matching the existing FCAS dataset
# ---------------------------------------------------------------------------
BATTERY_CONFIGS: dict[str, dict[str, float]] = {
    "medium_1c":   {"capacity": 10.0, "flow": 10.0},   # 1C (Torrens Island)
    "large_07c":   {"capacity": 30.0, "flow": 21.0},   # ~0.7C (Hornsdale)
    "small_05c":   {"capacity": 10.0, "flow": 5.0},    # 0.5C (Kennedy)
    "fast_375c":   {"capacity": 8.0,  "flow": 30.0},   # 3.75C (Dalrymple North)
}

# ---------------------------------------------------------------------------
# SDP resolution (matching the household default)
# ---------------------------------------------------------------------------
SOC_RESOLUTION = 20
ACTION_RESOLUTION = 41

# ---------------------------------------------------------------------------
# One episode: solve SDP → roll out → collect trajectory
# ---------------------------------------------------------------------------

def _find_aemo_cache(
    region: str, start_date: str, end_date: str, step_duration: float = 0.5
) -> Path:
    """Find a cached preprocessed AEMO file that covers the requested range."""
    # First try exact match on step_duration
    for dur in [f"{step_duration:.4f}h", f"{step_duration}h", "0.5000h", "0.0833h"]:
        p = Path(f"data/aemo/processed_{region}_{start_date}_{end_date}_{dur}.parquet")
        if p.exists():
            return p
    # Scan all files for this region — pick one covering the date range
    for f in sorted(Path("data/aemo").glob(f"processed_{region}_*.parquet")):
        parts = str(f.stem).split("_")
        if len(parts) < 5:
            continue
        f_start = parts[2]
        f_end = parts[3]
        if f_start <= start_date and f_end >= end_date:
            return f
    raise FileNotFoundError(
        f"No cached AEMO data covering {region} {start_date}..{end_date}. "
        f"Available: {[str(p) for p in sorted(Path('data/aemo').glob(f'processed_{region}_*.parquet'))]}"
    )


def _run_one_episode(
    region: str,
    start_date: str,
    end_date: str,
    episode_hours: int,
    battery_key: str,
    seed: int,
    step_duration: float = 0.5,
) -> pl.DataFrame:
    """Run SDP on one region/date/battery combo and return a trajectory DataFrame.

    Returns a DataFrame with columns: step, norm_observation, action, reward,
    source_policy, episode_id.
    """
    episode_id = hash(f"{region}_{start_date}_{end_date}_{battery_key}_{seed}") % (2**31 - 1)
    max_step = int(episode_hours / step_duration)

    # 1. Load preprocessed AEMO data
    cache_path = _find_aemo_cache(region, start_date, end_date, step_duration)
    df = pl.read_parquet(cache_path)

    battery_cap = BATTERY_CONFIGS[battery_key]["capacity"]
    battery_flow = BATTERY_CONFIGS[battery_key]["flow"]

    env = AEMOBatteryTradingEnv(
        aemo_data=df,
        battery_capacity=battery_cap,
        max_battery_flow=battery_flow,
        init_battery_level=battery_cap / 2.0,
        max_step=max_step,
        step_duration=step_duration,
        action_mode="simple",
        normalize_obs=True,
        random_episode_start=True,
        degradation_mode="rainflow",
    )

    # 3. Build forecast dicts (RRP for each timestep)
    # The env picks a random start index internally on reset
    obs, info = env.reset(seed=seed)
    start_idx = env.episode_start_idx

    forecast_rows = []
    for t in range(max_step):
        idx = start_idx + t
        if idx < len(df):
            row = df.row(idx, named=True)
            forecast_rows.append({"RRP": row["RRP"]})
        else:
            forecast_rows.append({"RRP": 0.0})

    # 4. Solve SDP
    solver = AEMOSDPSolver(
        env=env,
        horizon=len(forecast_rows),
        soc_resolution=SOC_RESOLUTION,
        action_resolution=ACTION_RESOLUTION,
        use_monte_carlo=False,
    )
    policy = solver.solve(forecast_rows)

    # 5. Roll out the optimal policy
    sos_levels = solver.soc_levels_kwh  # kWh arrays
    action_levels_norm = solver.action_levels_norm
    actions_norm = action_levels_norm  # [-1, 1]
    battery_flow_energies = solver.battery_flow_energies  # kWh per action

    steps: list[dict[str, Any]] = []
    env.reset(seed=seed)
    done = False
    step_count = 0

    while not done and step_count < max_step:
        # SDP policy lookup: nearest SoC level
        soc = env.battery_soc  # MWh
        soc_idx = int(np.argmin(np.abs(sos_levels - soc * 1000.0)))  # sos_levels is in kWh
        # Clamp soc_idx to valid range
        soc_idx = min(soc_idx, policy.shape[1] - 1)

        t_step = min(step_count, policy.shape[0] - 1)
        best_action_idx = int(policy[t_step, soc_idx])
        if best_action_idx < 0:
            best_action_idx = 0
        best_action_idx = min(best_action_idx, len(actions_norm) - 1)

        energy_action = float(actions_norm[best_action_idx])

        # Pad to 9D full_fcas action
        full_action = np.zeros(9, dtype=np.float32)
        full_action[0] = energy_action  # battery dispatch
        # FCAS dims 1-8 stay 0

        obs, reward, terminated, truncated, info = env.step(full_action)
        done = terminated or truncated

        steps.append({
            "step": step_count,
            "norm_observation": obs.tolist() if isinstance(obs, np.ndarray) else list(obs),
            "action": full_action.tolist(),
            "reward": float(reward),
            "source_policy": "sdp_energy",
            "episode_id": episode_id,
        })
        step_count += 1

    return pl.DataFrame(steps)


# ---------------------------------------------------------------------------
# Main launcher
# ---------------------------------------------------------------------------

def _build_scenarios(
    regions: list[str],
    start_date: str,
    end_date: str,
    episode_hours: int,
    episodes_per_region: int,
    batteries: list[str],
    step_duration: float = 0.5,
) -> list[dict[str, Any]]:
    """Build a list of scenario dicts, one per episode to generate."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_hours = (end - start).total_seconds() / 3600.0
    n_windows = max(1, int(total_hours // episode_hours))

    scenarios: list[dict[str, Any]] = []
    for region in regions:
        for wi in range(n_windows):
            ws = start + timedelta(hours=wi * episode_hours)
            we = ws + timedelta(hours=episode_hours)
            if we > end:
                we = end
            for bat in batteries:
                for ep in range(episodes_per_region):
                    scenarios.append({
                        "region": region,
                        "start_date": ws.strftime("%Y-%m-%d"),
                        "end_date": we.strftime("%Y-%m-%d"),
                        "episode_hours": int((we - ws).total_seconds() // 3600),
                        "battery_key": bat,
                        "seed": hash(f"{region}_{ws}_{bat}_{ep}") % (2**31 - 1),
                        "step_duration": step_duration,
                    })
    return scenarios


def _run_one(args: dict[str, Any]) -> pl.DataFrame:
    """Wrapper for multiprocessing."""
    return _run_one_episode(**args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SDP-optimal energy-arbitrage trajectories for AEMO."
    )
    parser.add_argument("--regions", default="NSW1,SA1,QLD1",
                        help="Comma-separated AEMO regions")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-07-01")
    parser.add_argument("--episode-hours", type=int, default=48,
                        help="Episode length in hours (default 48)")
    parser.add_argument("--episodes-per-region", type=int, default=5,
                        help="Episodes per region per time window (default 5)")
    parser.add_argument("--batteries", default="medium_1c,fast_375c",
                        help="Comma-separated battery configs (default medium_1c,fast_375c)")
    parser.add_argument("--step-duration", type=float, default=0.5,
                        help="Step duration in hours (default 0.5 = 30 min)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default = os.cpu_count / 2)")
    parser.add_argument("--output", default="data/aemo_dt_sdp/aemo_sdp_trajectories.parquet",
                        help="Output Parquet path")
    args = parser.parse_args()

    regions = [r.strip() for r in args.regions.split(",")]
    batteries = [b.strip() for b in args.batteries.split(",")]

    scenarios = _build_scenarios(
        regions=regions,
        start_date=args.start_date,
        end_date=args.end_date,
        episode_hours=args.episode_hours,
        episodes_per_region=args.episodes_per_region,
        batteries=batteries,
        step_duration=args.step_duration,
    )

    print(f"[SDP] Generating {len(scenarios)} SDP trajectories across "
          f"{len(regions)} regions × {len(batteries)} batteries")
    print(f"[SDP] Scenario range: {args.start_date} → {args.end_date}, "
          f"{args.episode_hours}h/episode")
    print(f"[SDP] Cached AEMO data needed: {len(set(
        (s['region'], s['start_date'][:7]) for s in scenarios
    ))} region-month combos")

    workers = args.workers or max(1, os.cpu_count() // 2)
    print(f"[SDP] Using {workers} parallel workers")

    # Generate trajectories
    t0 = time.time()
    all_dfs: list[pl.DataFrame] = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, s): s for s in scenarios}
        for future in as_completed(futures):
            scenario = futures[future]
            try:
                df = future.result()
                all_dfs.append(df)
            except Exception as e:
                print(f"[SDP] ERROR: {scenario.get('region')} "
                      f"{scenario.get('start_date')} "
                      f"{scenario.get('battery_key')}: {e}")
            completed += 1
            if completed % max(1, len(scenarios) // 10) == 0:
                print(f"[SDP] Progress: {completed}/{len(scenarios)} ({time.time()-t0:.0f}s)")

    if not all_dfs:
        print("[SDP] No trajectories generated!")
        sys.exit(1)

    combined = pl.concat(all_dfs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)

    elapsed = time.time() - t0
    print(f"[SDP] Done! {len(all_dfs)} episodes, {len(combined)} steps "
          f"→ {output_path} ({elapsed:.0f}s)")
    print(f"[SDP] Schema: {combined.schema}")
    adim = len(combined["action"][0])
    print(f"[SDP] Actions shape: 9 (action[0] len = {adim})")
    print(f"[SDP] Source policies: {combined['source_policy'].unique().to_list()}")


if __name__ == "__main__":
    main()
