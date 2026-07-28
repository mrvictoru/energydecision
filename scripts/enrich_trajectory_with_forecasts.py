#!/usr/bin/env python3
"""
Add episode_start to trajectories for TTM forecast alignment.

Instead of embedding full forecasts per row (84 GB), this script adds
a compact `episode_start` column. The ForecastTrajectoryDataset then
reads forecasts from the shared ttm_forecasts.npz at training time.

For rule episodes: seed = 42 + local_idx
For DT/GRPO episodes: seed = 8964 + local_idx
For SB3 episodes: seed = 0 + local_idx (approximate, still better than nothing)
SDP episodes: episode_start already present — passes through.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

REGION_MAP = {
    "nsw1": "NSW1", "qld1": "QLD1", "sa1": "SA1",
    "tas1": "TAS1", "vic1": "VIC1",
}


def load_region_length(region: str, aemo_dir: str) -> int:
    """Get the length of a region's processed AEMO data."""
    files = sorted(Path(aemo_dir).glob(f"processed_{region}_*_0.0833h.parquet"))
    if not files:
        return 500000
    best = max(files, key=lambda x: x.stat().st_size)
    return len(pl.read_parquet(best))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aemo-dir", default="data/aemo")
    args = parser.parse_args()

    t0 = time.time()
    df = pl.read_parquet(args.input)
    n_rows = len(df)
    n_eps = df["episode_id"].n_unique()
    print(f"[Enrich] {n_rows:,} rows, {n_eps} episodes")

    if "episode_start" in df.columns:
        print("[Enrich] Already has episode_start — copying")
        df.write_parquet(args.output)
        print(f"[Enrich] Done in {time.time()-t0:.1f}s")
        return

    # Build episode metadata
    eps_df = df.group_by("episode_id").agg([
        pl.col("source_policy").first(),
        pl.col("step").len().alias("ep_len"),
        pl.col("step").max().alias("max_step"),
    ])

    # Compute episode_start for each episode
    policy_counters: dict[str, int] = {}
    ep_starts_list: list[int] = []

    for row in eps_df.iter_rows(named=True):
        eid = row["episode_id"]
        sp = row["source_policy"]
        max_step = row["max_step"]
        parts = sp.split("__") if "__" in sp else []
        region_label = parts[0] if parts else "nsw1"
        region = REGION_MAP.get(region_label, "NSW1")
        policy_name = parts[1] if len(parts) > 1 else sp
        region_len = load_region_length(region, args.aemo_dir)

        is_rule = "rule" in policy_name.lower()
        is_dt = "dt" in policy_name or "grpo" in policy_name
        is_sdp = "sdp" in policy_name

        if is_sdp:
            ep_starts_list.append(0)  # SDP already has it; this is a fallback
        elif is_rule or is_dt:
            base = 42 if is_rule else 8964
            local_idx = policy_counters.get(sp, 0)
            policy_counters[sp] = local_idx + 1
            rng = np.random.default_rng(base + local_idx)
            ep_starts_list.append(int(rng.integers(0, max(1, region_len - max_step - 1))))
        else:
            # SB3: approximate via cumulative offset within region
            local_idx = policy_counters.get(sp, 0)
            policy_counters[sp] = local_idx + 1
            rng = np.random.default_rng(0 + local_idx)
            ep_starts_list.append(int(rng.integers(0, max(1, region_len - max_step - 1))))

    # Map episode_id → episode_start
    ep_start_map = dict(zip(eps_df["episode_id"].to_list(), ep_starts_list))

    # Add column
    result = df.with_columns(
        pl.col("episode_id").replace_strict(
            pl.Series(list(ep_start_map.keys()), dtype=pl.Int64),
            pl.Series(list(ep_start_map.values()), dtype=pl.Int64),
        ).alias("episode_start")
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out)

    n_det = sum(1 for s in ep_starts_list if s > 0)
    elapsed = time.time() - t0
    print(f"[Enrich] Done in {elapsed:.1f}s")
    print(f"[Enrich] {n_eps} episodes processed, {n_det} deterministic starts")
    print(f"[Enrich] Output: {out}")


if __name__ == "__main__":
    main()
