#!/usr/bin/env python3
"""Measure TTM forecast quality by comparing npz values against actual normalized AEMO data.

Computes per-channel MAE, RMSE, and correlation between the TTM forecasts in
ttm_forecasts.npz and the actual normalized observations from the processed
AEMO parquet files.  The npz stores the *first* step of each 48-step TTM
forecast at every position; this script compares that first forecast step
against the actual normalized value at the same position.

Usage:
    python3 scripts/measure_forecast_quality.py [--sample-every N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# The 6 TTM forecast channels, same order as the npz and TARGET_COLUMNS
FORECAST_CHANNELS = [
    "RRP",
    "TOTALDEMAND",
    "FCAS_RAISEREG",
    "FCAS_LOWERREG",
    "FCAS_RAISE5MIN",
    "FCAS_LOWER5MIN",
]

# Global normalization stats (from fit_aemo_global_stats, verified July 2026)
NORM_STATS = {
    "RRP": {"min": -1000.0, "max": 17500.0},
    "TOTALDEMAND": {"min": -37.14, "max": 13763.96},
    "FCAS_RAISEREG": {"min": 0.0, "max": 17500.0},
    "FCAS_LOWERREG": {"min": 0.0, "max": 17500.0},
    "FCAS_RAISE5MIN": {"min": 0.0, "max": 17500.0},
    "FCAS_LOWER5MIN": {"min": 0.0, "max": 17500.0},
}


def load_all_processed_data(data_dir: Path) -> pl.DataFrame:
    """Load all 5-min processed AEMO files, deduplicate by timestamp, and return raw + normalized columns."""
    files = sorted(data_dir.glob("processed_*_0.0833h.parquet"))
    dfs = []
    for f in files:
        cols = ["SETTLEMENTDATE"] + FORECAST_CHANNELS
        norm_cols = [f"{c}_normalized" for c in FORECAST_CHANNELS]
        available_norm = [c for c in norm_cols if c not in {"TOTALDEMAND_normalized"}]
        # For TOTALDEMAND, the normalized column is named DEMAND_normalized
        if "TOTALDEMAND" in FORECAST_CHANNELS and "DEMAND_normalized" not in cols:
            available_norm.append("DEMAND_normalized") if "DEMAND_normalized" not in available_norm else None
        try:
            df = pl.read_parquet(f, columns=cols + available_norm)
            df = df.with_columns(pl.col("SETTLEMENTDATE").cast(pl.Datetime("us")))
            dfs.append(df)
        except Exception:
            pass
    concat = pl.concat(dfs).unique(subset=["SETTLEMENTDATE"]).sort("SETTLEMENTDATE")
    return concat


def normalize_channel(values: np.ndarray, channel: str) -> np.ndarray:
    """Apply the same normalization the environment uses."""
    s = NORM_STATS[channel]
    denom = max(s["max"] - s["min"], 1e-9)
    return np.clip((values - s["min"]) / denom, 0.0, 1.0)


def compute_metrics(
    forecast_npz_path: Path,
    sample_every: int = 1,
    data_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """
    Returns per-channel dict of {mae, rmse, corr, n_samples}.
    """
    data_dir = data_dir or (REPO_ROOT / "data" / "aemo")

    print("Loading npz...")
    npz = np.load(forecast_npz_path, allow_pickle=True)
    fmap = npz["forecast_map"]  # [N, 48, 6]
    timestamps = npz["timestamps"]

    print("Loading processed data...")
    df = load_all_processed_data(data_dir)
    print(f"  {len(df)} rows, {df['SETTLEMENTDATE'].min()} → {df['SETTLEMENTDATE'].max()}")

    # Build lookup: unix timestamp → npz row index
    ts_to_npz_idx = {int(t): i for i, t in enumerate(timestamps)}

    results: dict[str, dict[str, float]] = {}
    for ci, channel in enumerate(FORECAST_CHANNELS):
        forecasts = []
        actuals = []
        matched = 0
        missed = 0

        # Get the actual raw values
        raw_col = channel if channel in df.columns else None
        norm_col = f"{channel}_normalized"
        if norm_col == "TOTALDEMAND_normalized":
            norm_col = "DEMAND_normalized"

        if raw_col is None or norm_col not in df.columns:
            print(f"  {channel}: skipped (missing columns)")
            continue

        raw_vals = df[raw_col].to_numpy()
        sd_col = df["SETTLEMENTDATE"].to_list()

        for i in range(0, len(df), sample_every):
            sd = sd_col[i]
            if hasattr(sd, "timestamp"):
                ts = int(sd.timestamp())
            elif isinstance(sd, (int, float)):
                ts = int(sd)
            else:
                continue

            npz_idx = ts_to_npz_idx.get(ts)
            if npz_idx is None:
                missed += 1
                continue

            # npz forecast for this position (first step of 48-step forecast)
            forecast_val = float(fmap[npz_idx, 0, ci])

            # Actual normalized value
            raw_val = float(raw_vals[i])
            actual_val = float(np.clip((raw_val - NORM_STATS[channel]["min"]) / max(NORM_STATS[channel]["max"] - NORM_STATS[channel]["min"], 1e-9), 0.0, 1.0))

            forecasts.append(forecast_val)
            actuals.append(actual_val)
            matched += 1

        forecasts = np.array(forecasts)
        actuals = np.array(actuals)

        if len(forecasts) < 2:
            results[channel] = {"mae": 0.0, "rmse": 0.0, "corr": 0.0, "n": 0}
            continue

        mae = float(np.mean(np.abs(forecasts - actuals)))
        rmse = float(np.sqrt(np.mean((forecasts - actuals) ** 2)))
        corr = float(np.corrcoef(forecasts, actuals)[0, 1]) if np.std(forecasts) > 0 and np.std(actuals) > 0 else 0.0

        results[channel] = {"mae": mae, "rmse": rmse, "corr": corr, "n": matched}
        print(f"  {channel:20s} MAE={mae:.4f}  RMSE={rmse:.4f}  corr={corr:+.3f}  n={matched}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure TTM forecast quality")
    parser.add_argument("--npz-path", default=str(REPO_ROOT / "data" / "aemo_dt_forecast" / "ttm_forecasts.npz"))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "aemo"))
    parser.add_argument("--sample-every", type=int, default=1, help="Subsample factor (1 = all rows)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = compute_metrics(
        Path(args.npz_path),
        sample_every=args.sample_every,
        data_dir=Path(args.data_dir),
    )

    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print("\n=== Summary ===")
        avg_mae = np.mean([r["mae"] for r in results.values() if r["n"] > 0])
        avg_rmse = np.mean([r["rmse"] for r in results.values() if r["n"] > 0])
        print(f"Average MAE:  {avg_mae:.4f}")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Samples:      {sum(r['n'] for r in results.values())} total")


if __name__ == "__main__":
    main()
