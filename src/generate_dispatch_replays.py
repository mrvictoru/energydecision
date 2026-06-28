"""Generate dispatch replay episodes for all active battery stations across
2023, 2024, and 2025 windows, with correct real-life battery sizing.

Usage:
    python3 src/generate_dispatch_replays.py                     # full run
    python3 src/generate_dispatch_replays.py --window 2024       # single window
    python3 src/generate_dispatch_replays.py --station hornsdale # single station
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ── Target windows (6-month blocks matching processed data cache) ─────────
WINDOWS: dict[str, tuple[str, str]] = {
    "2023-h1": ("2023-01-01", "2023-07-01"),
    "2023-h2": ("2023-07-01", "2024-01-01"),
    "2024-h1": ("2024-01-01", "2024-07-01"),
    "2024-h2": ("2024-07-01", "2025-01-01"),
    "2025-h1": ("2025-01-01", "2025-07-01"),
}

# ── Processed data mapping: region -> list of (start, end, filepath) ──────
# Auto-discovered from data/aemo/ at startup
PROCESSED_CACHE: dict[str, list[dict[str, Any]]] = {}


def discover_processed_data(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan data/aemo/ for processed parquet files and group by region."""
    result: dict[str, list[dict[str, Any]]] = {}
    for fpath in sorted(data_dir.glob("processed_*.parquet")):
        parts = fpath.stem.split("_")
        # Format: processed_{REGION}_{START}_{END}_{STEP}h.parquet
        if len(parts) >= 5:
            region = parts[1]
            start_str = parts[2]
            end_str = parts[3]
        else:
            continue
        result.setdefault(region, []).append({
            "start": datetime.fromisoformat(start_str),
            "end": datetime.fromisoformat(end_str),
            "path": fpath,
        })
    return result


def find_processed_data(
    region: str,
    window_start: datetime,
    window_end: datetime,
    cache: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    """Find processed data files covering the window, returning a list.
    
    Returns multiple files when the window spans multiple processed-data
    blocks (e.g. 2024-01→2025-01 spans 2024-H1 and 2024-H2 6-month blocks).
    Only returns 5-min resolution (0.0833h) files since the replay needs
    sub-hourly data.
    """
    entries = cache.get(region, [])
    if not entries:
        return []
    entries_5min = [e for e in entries if "0.0833" in str(e["path"])]
    if not entries_5min:
        return []
    
    # Sort by start date
    entries_5min.sort(key=lambda e: e["start"])
    
    # Find files whose ranges overlap with our window
    matching = []
    covered_until = window_start
    for e in entries_5min:
        if e["start"] <= covered_until < e["end"]:
            matching.append(e["path"])
            covered_until = e["end"]
            if covered_until >= window_end:
                break
    
    if matching and covered_until >= window_end:
        return matching
    # If we can't fully cover, try the longest single file
    best = max(entries_5min, key=lambda e: (e["end"] - e["start"]).total_seconds())
    if best["start"] <= window_start and best["end"] >= window_start:
        print(f"  [WARN] Partial coverage: using {best['path'].name} "
              f"({best['start'].date()}→{best['end'].date()}) for "
              f"window ({window_start.date()}→{window_end.date()})")
        return [best["path"]]
    return []


def generate_window_replays(
    window_name: str,
    window_start: datetime,
    window_end: datetime,
    output_dir: Path,
    cache_dir: Path,
    station_filter: str | None = None,
) -> dict[str, Any]:
    """Generate dispatch replays for all active stations in a time window."""
    from aemo_data import BATTERY_REGISTRY
    from dispatch_utils import (
        list_dispatch_candidates,
        resolve_dispatch_selection,
        run_dispatch_replay,
    )
    import polars as pl

    results: dict[str, Any] = {}

    for station_name in sorted(BATTERY_REGISTRY.keys()):
        if station_filter and station_name != station_filter:
            continue

        info = BATTERY_REGISTRY[station_name]
        region = info.get("region", "?")

        # Skip if battery wasn't commissioned yet in this window
        duids = info.get("duids", [])
        min_valid = min(
            d.get("valid_from", datetime(2000, 1, 1))
            for d in duids if d.get("valid_from")
        )
        if min_valid > window_end:
            print(f"  [{window_name}] {station_name}: not yet commissioned (valid_from={min_valid.date()})")
            continue

        print(f"\n  [{window_name}] {station_name} ({region}) — checking availability...")

        # Step 1: Check if this station has dispatch data in the window
        try:
            battery_units, active_units = list_dispatch_candidates(
                region=region,
                start_date=window_start,
                end_date=window_end,
                station_name=station_name,
                cache_dir=str(cache_dir),
            )
        except Exception as e:
            print(f"  ERROR checking {station_name}: {e}")
            results[station_name] = {"status": "error", "error": str(e)}
            continue

        if active_units.height == 0:
            print(f"  [{window_name}] {station_name}: no dispatch data in this window — skipping")
            results[station_name] = {"status": "no_data"}
            continue

        nonzero = active_units["NonZeroIntervalCount"].to_list()
        print(f"  [{window_name}] {station_name}: ACTIVE — non-zero intervals={nonzero}")

        # Step 2: Resolve selection with correct real-life battery sizing
        try:
            selection = resolve_dispatch_selection(
                battery_units=battery_units,
                active_battery_units=active_units,
                selected_index=0,  # first (most active) DUID
                apply_unit_sizing=True,  # uses real registry sizing
                start_date=window_start,
                end_date=window_end,
                cache_dir=str(cache_dir),
            )
        except Exception as e:
            print(f"  ERROR resolving {station_name}: {e}")
            results[station_name] = {"status": "error", "error": str(e)}
            continue

        # Step 3: Load processed market data for the region
        processed_paths = find_processed_data(region, window_start, window_end, PROCESSED_CACHE)
        if not processed_paths:
            print(f"  [{window_name}] {station_name}: no processed market data for {region} — skipping")
            results[station_name] = {"status": "no_market_data"}
            continue

        # Load processed data (may be multiple files, concatenate)
        import polars as pl
        processed_dfs = [pl.read_parquet(str(p)) for p in processed_paths]
        if len(processed_dfs) > 1:
            processed_df = pl.concat(processed_dfs).unique(subset=["SETTLEMENTDATE"]).sort("SETTLEMENTDATE")
            print(f"  Loaded {len(processed_paths)} processed files → {processed_df.height} rows")
        else:
            processed_df = processed_dfs[0]
            print(f"  Loaded {processed_paths[0].name}: {processed_df.height} rows")

        # Step 4: Calculate max steps for the window
        window_hours = (window_end - window_start).total_seconds() / 3600
        step_duration = 0.5  # 30 min steps
        max_step = int(window_hours / step_duration)

        # Step 5: Run dispatch replay
        station_tag = f"{window_name}_{station_name}".replace(" ", "_")
        print(f"  [{window_name}] {station_name}: running dispatch replay ({max_step} steps)...")

        try:
            ep_logs, inc_logs, all_logs = run_dispatch_replay(
                processed_data=processed_df,
                selection=selection,
                start_date=window_start,
                end_date=window_end,
                region=region,
                cache_dir=str(cache_dir),
                num_episodes=1,
                step_duration=step_duration,
                max_step=max_step,
                output_dir=str(output_dir),
                run_tag=station_tag,
                action_mode="multi_market",
                degradation_mode="rainflow",
                degradation_chemistry="LFP",
                degradation_temperature=30.0,
            )
            total_reward = float(all_logs["reward"].sum())
            n_steps = all_logs.height
            print(f"  [{window_name}] {station_name}: done — {n_steps} steps, reward={total_reward:.2f}")

            results[station_name] = {
                "status": "success",
                "region": region,
                "steps": n_steps,
                "total_reward": total_reward,
                "battery_capacity_mwh": selection["battery_capacity"],
                "max_power_mw": selection["max_battery_flow"],
                "init_soc_mwh": selection["init_battery_level"],
                "station_name": selection.get("station_name"),
            }

        except Exception as e:
            print(f"  ERROR replaying {station_name}: {e}")
            results[station_name] = {"status": "error", "error": str(e)}

    return results


def main() -> None:
    root = repo_root()
    sys.path.insert(0, str(root / "src"))

    parser = argparse.ArgumentParser(description="Generate dispatch replay episodes")
    parser.add_argument("--window", type=str, default=None,
                        help="Target window: 2023-h1, 2023-h2, 2024-h1, 2024-h2, 2025-h1 (default: all)")
    parser.add_argument("--station", type=str, default=None,
                        help="Single station name (default: all)")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data" / "aemo_dispatch_replays",
                        help="Output directory for replay logs")
    parser.add_argument("--cache-dir", type=Path,
                        default=root / "data" / "aemo",
                        help="AEMO data cache directory")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available processed market data
    global PROCESSED_CACHE
    PROCESSED_CACHE = discover_processed_data(cache_dir)
    print(f"Discovered processed data for {len(PROCESSED_CACHE)} regions:")
    for region, entries in sorted(PROCESSED_CACHE.items()):
        for e in entries:
            print(f"  {region}: {e['start'].date()} → {e['end'].date()} ({e['path'].name})")

    # Determine which windows to process
    windows_to_run = [args.window] if args.window else list(WINDOWS.keys())

    all_results: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "windows": {},
        "total_stations": 0,
        "successful": 0,
    }

    for win_name in windows_to_run:
        if win_name not in WINDOWS:
            print(f"Unknown window: {win_name}. Choose from: {list(WINDOWS.keys())}")
            continue

        ws_str, we_str = WINDOWS[win_name]
        ws = datetime.fromisoformat(ws_str)
        we = datetime.fromisoformat(we_str)

        print(f"\n{'='*70}")
        print(f"Window: {win_name} ({ws_str} → {we_str})")
        print(f"{'='*70}")

        win_results = generate_window_replays(
            window_name=win_name,
            window_start=ws,
            window_end=we,
            output_dir=output_dir,
            cache_dir=cache_dir,
            station_filter=args.station,
        )

        successes = {k: v for k, v in win_results.items() if v.get("status") == "success"}
        all_results["windows"][win_name] = {
            "stations": win_results,
            "total_attempted": len(win_results),
            "successful": len(successes),
        }
        all_results["total_stations"] += len(win_results)
        all_results["successful"] += len(successes)

    # Save results manifest
    manifest_path = output_dir / "dispatch_replay_manifest.json"
    manifest_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nManifest saved: {manifest_path}")
    print(f"Total successful: {all_results['successful']}/{all_results['total_stations']}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for win_name, win_data in all_results["windows"].items():
        successes = {k: v for k, v in win_data["stations"].items() if v.get("status") == "success"}
        print(f"\n{win_name}: {len(successes)}/{win_data['total_attempted']} successful:")
        for s, info in sorted(successes.items()):
            print(f"  {s:30s} reward={info['total_reward']:.2f}  "
                  f"{info['battery_capacity_mwh']}MWh/{info['max_power_mw']}MW  "
                  f"{info['steps']} steps")


if __name__ == "__main__":
    main()
