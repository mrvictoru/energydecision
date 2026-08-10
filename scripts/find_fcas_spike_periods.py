#!/usr/bin/env python3
"""Find the best date windows for generating FCAS-heavy episodes (Option A).

Scans cached AEMO processed data and ranks sliding windows by FCAS activity
(spike bars, total spike magnitude, max price) so we can target episode
generation during known FCAS-spike periods — the "generate during FCAS-spike
periods" sub-step of the FCAS-focused data generation plan.

Usage:
    python3 scripts/find_fcas_spike_periods.py \
        --region NSW1 --start 2021-01-01 --end 2023-04-01 \
        --window-days 12 --top-k 10

    # All regions (whatever is cached), 2024:
    python3 scripts/find_fcas_spike_periods.py --year 2024 --window-days 12 --top-k 10
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fcas_generator_eval import FCAS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", action="append", default=[], help="Region(s) to scan; default = all cached.")
    parser.add_argument("--year", type=int, default=None, help="Restrict to a calendar year (e.g. 2024).")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: earliest cached).")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: latest cached).")
    parser.add_argument("--window-days", type=int, default=12, help="Episode length in days (DT short horizon = 12).")
    parser.add_argument("--stride-days", type=int, default=1, help="Sliding-window stride in days.")
    parser.add_argument("--price-threshold", type=float, default=50.0, help="FCAS price level considered a 'spike' ($).")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top windows to report.")
    return parser.parse_args()


def _cached_regions(data_dir: Path) -> list[str]:
    regions = sorted({p.name.split("_")[1] for p in data_dir.glob("processed_*_*.parquet")})
    return regions


def _bounds_for_region(data_dir: Path, region: str) -> tuple[datetime, datetime]:
    starts, ends = [], []
    for p in data_dir.glob(f"processed_{region}_*.parquet"):
        parts = p.stem.split("_")
        try:
            starts.append(datetime.fromisoformat(parts[2]))
            ends.append(datetime.fromisoformat(parts[3]))
        except (ValueError, IndexError):
            continue
    if not starts:
        raise ValueError(f"no cached processed data for {region}")
    return min(starts), max(ends)


def _find_cache(data_dir: Path, region: str, start: datetime, end: datetime) -> list[tuple[Path, datetime]]:
    """Return (path, file_start_dt) for every cached parquet overlapping [start, end], sorted."""
    out = []
    for p in data_dir.glob(f"processed_{region}_*_*_0.0833h.parquet"):
        parts = p.stem.split("_")
        try:
            ps, pe = datetime.fromisoformat(parts[2]), datetime.fromisoformat(parts[3])
        except (ValueError, IndexError):
            continue
        if pe > start and ps < end:
            out.append((p, ps))
    out.sort(key=lambda c: c[1])
    return out


def _window_metrics(frame: pl.DataFrame, threshold: float) -> dict[str, float]:
    maxes = []
    spike_count = 0
    intensity = 0.0
    for s in FCAS:
        x = frame[f"FCAS_{s}"].to_numpy().astype(np.float64)
        maxes.append(float(x.max()))
        spike_count += int(np.sum(x >= threshold))
        intensity += float(np.sum(np.maximum(x - threshold, 0.0)))
    return {
        "spike_bars": float(spike_count),
        "intensity": intensity,
        "max_price": max(maxes),
    }


def main() -> int:
    args = parse_args()
    data_dir = ROOT / "data" / "aemo"
    regions = args.region or _cached_regions(data_dir)

    rows = []
    for region in regions:
        if not list(data_dir.glob(f"processed_{region}_*.parquet")):
            print(f"  [skip] no cached data for {region}")
            continue
        try:
            lo, hi = _bounds_for_region(data_dir, region)
        except ValueError as exc:
            print(f"  [skip] {exc}")
            continue
        if args.year:
            lo = max(lo, datetime(args.year, 1, 1))
            hi = min(hi, datetime(args.year + 1, 1, 1))
        if args.start:
            lo = max(lo, datetime.fromisoformat(args.start))
        if args.end:
            hi = min(hi, datetime.fromisoformat(args.end))
        if lo >= hi:
            continue

        files = _find_cache(data_dir, region, lo, hi)
        if not files:
            print(f"  [skip] no cached parquet overlaps {region} {lo.date()}..{hi.date()}")
            continue
        print(f"Scanning {region} {lo.date()}..{hi.date()} ({len(files)} cached file(s))")
        file_start = files[0][1]

        def _read(fp: Path) -> pl.DataFrame:
            df = pl.read_parquet(str(fp))
            if "SETTLEMENTDATE" in df.columns:
                df = df.with_columns(pl.col("SETTLEMENTDATE").cast(pl.Datetime("us")))
            return df

        frame = pl.concat([_read(p) for p, _ in files])
        # Processed frames are 5-min rows with no timestamp column; infer time
        # from the file's start date + row offset.
        start_row = max(0, int(round((lo - file_start).total_seconds() / 300)))
        end_row = min(frame.height, int(round((hi - file_start).total_seconds() / 300)))
        if start_row >= end_row:
            continue
        frame = frame.slice(start_row, end_row - start_row)

        window_steps = int(round(args.window_days * 24 * 60 / 5))
        step = int(round(args.stride_days * 24 * 60 / 5))
        if window_steps > frame.height:
            continue
        for start_idx in range(0, frame.height - window_steps + 1, step):
            win = frame.slice(start_idx, window_steps)
            m = _window_metrics(win, args.price_threshold)
            win_start = file_start + timedelta(minutes=5 * (start_row + start_idx))
            rows.append(
                {
                    "region": region,
                    "window_start": win_start.strftime("%Y-%m-%d"),
                    "spike_bars": int(m["spike_bars"]),
                    "intensity": float(m["intensity"]),
                    "max_price": float(m["max_price"]),
                }
            )

    if not rows:
        print("no windows found")
        return 1

    out = pl.DataFrame(rows).sort("intensity", descending=True).head(args.top_k * 4)
    print("\n=== TOP FCAS-SPIKE WINDOWS (best generation targets) ===")
    print(f"{'region':>7s} {'window_start':>12s} {'spike_bars':>10s} {'intensity':>12s} {'max_price':>10s}")
    for r in out.head(args.top_k).iter_rows(named=True):
        print(f"{r['region']:>7s} {r['window_start']:>12s} {r['spike_bars']:>10d} {r['intensity']:>12,.0f} {r['max_price']:>10.1f}")

    out_dir = ROOT / "eval_output" / "final" / "fcas_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.head(args.top_k * 4).write_csv(out_dir / "fcas_spike_windows.csv")
    print(f"\nSaved (top {args.top_k * 4}) -> {out_dir / 'fcas_spike_windows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
