#!/usr/bin/env python3
"""Generate Oracle-SOC waypoint targets for hierarchical DT training (Exp 3).

For each episode in a training corpus, reconstruct the per-step price frame
from the raw observations, run the free (perfect-foresight) Oracle LP to get
the optimal SOC trajectory, downsample it to K evenly-spaced waypoints, and
write a parquet whose `action` column is the K-dim normalized target-SOC
waypoint vector. This is the DT's regression target: given context + RTG,
predict the target-SOC schedule; the SOC-waypoint-pinned LP executor then
co-optimizes energy + FCAS within each segment (see AEMOAgent dt_soc_oracle).

Usage:
  python3 scripts/generate_soc_waypoint_targets.py \
    --raw-dir data/aemo_dt_fcas_v2/raw_logs \
    --policies a2c td3 sac ddpg \
    --horizons short medium long \
    --batteries medium_1c large_07c small_05c fast_375c \
    --K 8 \
    --out data/aemo_dt_soc_oracle/aemo_soc_waypoints.parquet

Observations: 18-dim, indices [0:4]=time, 4=is_peak, 5=RRP, 6=TOTALDEMAND,
7:15=8 FCAS prices, 15:17=GEN, 17=SOC (normalized soc/capacity).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_oracle_algo import AEMOOracleSolver  # noqa: E402

FCAS_ORDER = ["RAISEREG", "LOWERREG", "RAISE6SEC", "LOWER6SEC",
              "RAISE60SEC", "LOWER60SEC", "RAISE5MIN", "LOWER5MIN"]
# Oracle solver order
ORACLE_FCAS = ["RAISE6SEC", "RAISE60SEC", "RAISE5MIN", "RAISEREG",
               "LOWER6SEC", "LOWER60SEC", "LOWER5MIN", "LOWERREG"]


def battery_params(name: str) -> dict:
    specs = {
        "medium_1c": {"capacity_mwh": 10.0, "max_power_mw": 10.0},
        "large_07c": {"capacity_mwh": 50.0, "max_power_mw": 35.0},
        "small_05c": {"capacity_mwh": 2.0, "max_power_mw": 1.0},
        "fast_375c": {"capacity_mwh": 8.0, "max_power_mw": 30.0},
    }
    if name not in specs:
        raise ValueError(f"Unknown battery {name!r}; known: {sorted(specs)}")
    return specs[name]


def raw_obs_to_price_frame(raw_obs: np.ndarray, n_steps: int) -> pl.DataFrame:
    """Build the Oracle price DataFrame from raw observations (N x 18)."""
    rrp = raw_obs[:, 5].astype(float)
    fcas = raw_obs[:, 7:15].astype(float)  # env order
    cols = {"RRP": rrp}
    for i, svc in enumerate(ORACLE_FCAS):
        # env index for this service in FCAS_ORDER
        env_i = FCAS_ORDER.index(svc)
        cols[f"FCAS_{svc}"] = fcas[:, env_i]
    cols["SETTLEMENTDATE"] = [str(i) for i in range(n_steps)]
    return pl.DataFrame(cols)


def downsample_price_frame(prices: pl.DataFrame, factor: int) -> pl.DataFrame:
    """Average consecutive 5-min rows into coarser steps (e.g. factor=12 -> hourly).

    Only the first `n - (n % factor)` rows are used so every coarse step is full.
    """
    n = prices.height
    usable = n - (n % factor)
    if usable < factor:
        return prices
    sub = prices.head(usable)
    cols = [c for c in prices.columns if c != "SETTLEMENTDATE"]
    out = {}
    for c in cols:
        arr = sub[c].to_numpy().astype(float)
        out[c] = arr.reshape(-1, factor).mean(axis=1)
    out["SETTLEMENTDATE"] = [str(i) for i in range(len(out[cols[0]]))]
    return pl.DataFrame(out)


def waypoint_indices(n_steps: int, K: int) -> list[int]:
    """K evenly-spaced interval indices in [0, n_steps] (last = terminal SOC)."""
    if K <= 1:
        return [n_steps]
    return [int(round(i * n_steps / (K - 1))) for i in range(K)]


def generate_waypoints(raw_log: Path, K: int, coarse_factor: int = 12) -> pl.DataFrame:
    df = pl.read_parquet(raw_log)
    rows = df.height
    raw_obs = np.stack(df["raw_observation"].to_list())  # (rows, 18)
    battery = battery_params(raw_log.stem.split("__")[3])
    cap = battery["capacity_mwh"]
    init_soc = cap * 0.5

    step_h = 5.0 / 60.0
    solve_factor = 1
    prices = raw_obs_to_price_frame(raw_obs, rows)
    # Long episodes make the full 5-min LP infeasible; coarsen the price grid
    # (hourly by default) and interpolate the optimal SOC back to 5-min.
    if rows > 12000 and coarse_factor > 1:
        prices = downsample_price_frame(prices, coarse_factor)
        solve_factor = coarse_factor
        step_h = solve_factor * 5.0 / 60.0

    solver = AEMOOracleSolver(
        battery_capacity=cap,
        max_battery_flow=battery["max_power_mw"],
        step_duration=step_h,
        init_soc=init_soc,
        min_soc=0.0,
        max_soc=cap,
    )
    result = solver.solve(prices, verbose=False)
    if result.total_profit < -1e11:
        raise RuntimeError(f"Oracle solve failed for {raw_log.name}: {result.solver_message}")

    # Optimal SOC at START of each coarse interval. Upsample back to 5-min by
    # linear interpolation so waypoints sample the natural 5-min trajectory.
    coarse_soc = result.optimal_soc  # length C+1 (soc at start of each coarse step)
    C = prices.height
    coarse_t = np.arange(C + 1) * solve_factor  # 5-min index of each coarse boundary
    full_t = np.arange(rows + 1)
    soc_5min = np.interp(full_t, coarse_t, coarse_soc)

    idxs = waypoint_indices(rows, K)
    wp_mwh = np.array([float(soc_5min[t]) for t in idxs])
    wp_norm = np.clip(wp_mwh / cap, 0.0, 1.0)

    out = pl.DataFrame({
        "episode_id": [raw_log.stem] * rows,
        "step": df["step"].to_list(),
        "norm_observation": df["norm_observation"].to_list(),
        "action": [wp_norm.tolist()] * rows,
        "reward": df["reward"].to_list(),
        "source_policy": [raw_log.stem.split("__")[1]] * rows,
        "oracle_profit": [float(result.total_profit)] * rows,
        "waypoint_soc_mwh": [wp_mwh.tolist()] * rows,
        "coarse_factor": [solve_factor] * rows,
    })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--policies", nargs="+", default=["a2c", "td3", "sac", "ddpg"])
    p.add_argument("--horizons", nargs="+", default=["short", "medium", "long"])
    p.add_argument("--batteries", nargs="+",
                   default=["medium_1c", "large_07c", "small_05c", "fast_375c"])
    p.add_argument("--K", type=int, default=8, help="Number of SOC waypoints (DT act_dim).")
    p.add_argument("--coarse-factor", type=int, default=12,
                   help="Downsample 5-min prices to this-many-5min steps (12=hourly) for long episodes.")
    p.add_argument("--out", type=Path, default="data/aemo_dt_soc_oracle/aemo_soc_waypoints.parquet")
    p.add_argument("--limit", type=int, default=None, help="Cap episodes processed (smoke test).")
    args = p.parse_args(argv)

    raw_files: list[Path] = []
    for region_dir in sorted(args.raw_dir.iterdir()):
        if not region_dir.is_dir():
            continue
        for f in sorted(region_dir.glob("*.parquet")):
            stem = f.stem
            parts = stem.split("__")
            if len(parts) < 5:
                continue
            _, pol, horizon, battery, _ = parts[0], parts[1], parts[2], parts[3], parts[4]
            if pol in args.policies and horizon in args.horizons and battery in args.batteries:
                raw_files.append(f)
    if args.limit is not None:
        raw_files = raw_files[: args.limit]
    print(f"Matched {len(raw_files)} episodes (K={args.K})")

    frames = []
    for i, f in enumerate(raw_files):
        frames.append(generate_waypoints(f, args.K, coarse_factor=args.coarse_factor))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(raw_files)} processed")
    combined = pl.concat(frames) if frames else pl.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(args.out)
    print(f"Wrote {args.out} ({combined.height} rows, {combined['episode_id'].n_unique()} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
