#!/usr/bin/env python3
"""
Enrich a trajectory parquet with TTM forecast columns — vectorized.

Reads trajectory data, looks up forecasts by step index, and appends
a 'forecast' column to each row (shape [F, 6] per row).

Usage:
    python3 scripts/enrich_trajectory_with_forecasts.py \
        --input data/aemo_dt_sdp/aemo_sdp_trajectories.parquet \
        --forecast-npz data/aemo_dt_forecast/ttm_forecasts.npz \
        --output data/aemo_dt_forecast/aemo_sdp_forecast.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--forecast-npz", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.time()

    # Load trajectory
    df = pl.read_parquet(args.input, columns=["episode_id", "step", "norm_observation", "action", "reward"])
    steps = df["step"].to_numpy()
    n_rows = len(df)
    n_eps = df["episode_id"].n_unique()
    print(f"[Enrich] {n_rows:,} rows, {n_eps} episodes")

    # Load forecast lookup
    fc = np.load(args.forecast_npz)
    forecast_map = fc["forecast_map"]  # [N_global, 48, 6]
    n_global = len(forecast_map)
    F = forecast_map.shape[1]
    n_chan = forecast_map.shape[2]
    print(f"[Enrich] Forecast map: {forecast_map.shape}")

    # Vectorized: for each step, look up forecast_map[step]
    # Steps that exceed the forecast map get zeros
    valid = steps < n_global - F
    n_valid = valid.sum()
    n_missing = n_rows - n_valid
    print(f"[Enrich] Valid: {n_valid}/{n_rows} ({100*n_valid//n_rows}%)")

    # Build result array [N, F, 6] using vectorized indexing
    result_forecast = np.zeros((n_rows, F, n_chan), dtype=np.float32)
    valid_indices = np.where(valid)[0]
    valid_steps = steps[valid_indices]
    result_forecast[valid_indices] = forecast_map[valid_steps]

    # Store as flat list per row [F * n_chan] — reshape in dataset
    forecast_flat = result_forecast.reshape(n_rows, F * n_chan).tolist()

    result = df.with_columns(
        pl.Series("forecast", forecast_flat, dtype=pl.List(pl.Float64))
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path)

    elapsed = time.time() - t0
    print(f"[Enrich] Done in {elapsed:.1f}s")
    print(f"[Enrich] Output: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    print(f"[Enrich] Schema: {result.schema}")


if __name__ == "__main__":
    main()
