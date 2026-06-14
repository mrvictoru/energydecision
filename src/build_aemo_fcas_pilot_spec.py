"""Build stratified pilot spec from the FCAS-rich dataset for autoresearch.

This script:
1. Reconstructs episode_id → metadata from raw log file order
2. Stratifies by source_policy, horizon, region, battery
3. Generates a spec JSON for build_aemo_autoresearch_pilot.py
4. Optionally runs the builder to produce the final parquet files

Usage:
    # Generate spec only
    python3 src/build_aemo_fcas_pilot_spec.py

    # Generate spec and build pilot parquet files
    python3 src/build_aemo_fcas_pilot_spec.py --build-pilot

    # Custom paths
    python3 src/build_aemo_fcas_pilot_spec.py \
        --dataset-path data/aemo_dt_fcas/aemo_fcas_dataset.parquet \
        --output-path data/aemo_dt_fcas/autoresearch_pilot_spec.json \
        --pilot-output-dir data/aemo_dt_fcas/autoresearch_pilot \
        --build-pilot
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


TARGET_COMPOSITION: dict[str, tuple[int, int]] = {
    "ppo": (3, 1),
    "a2c": (1, 1),
    "ddpg": (1, 0),
    "sac": (1, 0),
    "td3": (1, 1),
    "fcas_rule": (1, 1),
}

PILOT_SLICE_LENGTH = 2016
PILOT_START_OFFSET = 288
MIN_EPISODE_LENGTH = PILOT_SLICE_LENGTH + PILOT_START_OFFSET
# Exclude short horizon (3456) — medium/long only for sufficient data diversity
SHORT_HORIZON_LENGTH = 3456

SCENARIO_TO_REGION = {
    "nsw1_2021_2023": "NSW1",
    "qld1_2021_2023": "QLD1",
    "sa1_2022_2023": "SA1",
    "tas1_2021_2023": "TAS1",
    "vic1_2021_2023": "VIC1",
}


def reconstruct_episode_metadata(raw_logs_dir: Path) -> pl.DataFrame:
    """Reconstruct episode_id → metadata from raw log file order.

    Episode IDs were assigned sequentially in alphabetical file order
    by memory_safe_assemble.py.
    """
    files = sorted(raw_logs_dir.rglob("*.parquet"))
    rows = []
    for idx, path in enumerate(files):
        parts = path.stem.split("__")
        if len(parts) < 5:
            continue
        scenario = parts[0]
        policy = parts[1]
        horizon = parts[2]
        battery = parts[3]
        region = SCENARIO_TO_REGION.get(scenario, scenario.split("_")[0].upper())
        rows.append({
            "episode_id": idx,
            "source_policy": policy,
            "region": region,
            "horizon": horizon,
            "battery": battery,
        })
    return pl.DataFrame(rows)


def get_episode_lengths(dataset_path: Path) -> pl.DataFrame:
    """Get episode lengths from the assembled dataset (episode_id + count)."""
    df = pl.read_parquet(str(dataset_path))
    return df.group_by("episode_id").agg(pl.len().alias("length"))


def select_episodes(
    metadata: pl.DataFrame,
    lengths: pl.DataFrame,
    composition: dict[str, tuple[int, int]],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Select stratified episodes for pilot train/val.

    Returns (train_selections, val_selections).
    """
    import random
    rng = random.Random(seed)

    metadata = metadata.join(lengths, on="episode_id", how="inner")

    # Exclude short horizon (3456) — medium (16128) or long (74880) only
    filtered = metadata.filter(
        (pl.col("length") >= MIN_EPISODE_LENGTH) &
        (pl.col("length") > SHORT_HORIZON_LENGTH)
    )

    train_selections = []
    val_selections = []

    for source_policy, (n_train, n_val) in composition.items():
        candidates = filtered.filter(pl.col("source_policy") == source_policy)
        if candidates.height < n_train + n_val:
            print(f"  WARNING: {source_policy} has only {candidates.height} suitable episodes, "
                  f"need {n_train + n_val}. Adjusting.")
            available = candidates.height
            n_train = min(n_train, available)
            n_val = min(n_val, max(0, available - n_train))

        indices = list(range(candidates.height))
        rng.shuffle(indices)

        for i in range(n_train):
            ep = candidates.row(indices[i])
            train_selections.append({
                "episode_id": int(ep[0]),
                "source_policy": str(ep[1]),
                "region": str(ep[2]),
                "horizon": str(ep[3]),
                "battery": str(ep[4]),
                "start_step": PILOT_START_OFFSET,
                "step_count": PILOT_SLICE_LENGTH,
            })

        for i in range(n_train, n_train + n_val):
            ep = candidates.row(indices[i])
            val_selections.append({
                "episode_id": int(ep[0]),
                "source_policy": str(ep[1]),
                "region": str(ep[2]),
                "horizon": str(ep[3]),
                "battery": str(ep[4]),
                "start_step": PILOT_START_OFFSET,
                "step_count": PILOT_SLICE_LENGTH,
            })

    return train_selections, val_selections


def build_spec(
    train_selections: list[dict],
    val_selections: list[dict],
    composition: dict[str, tuple[int, int]],
) -> dict:
    """Build spec in the format expected by build_aemo_autoresearch_pilot.py."""
    return {
        "description": (
            "FCAS-rich pilot split for autoresearch. Stratified by source policy from "
            "the 2,425-episode FCAS dataset. Medium/long horizon only. "
            "All slices are 2,016 steps starting at step 288."
        ),
        "train": [
            {
                "episode_id": s["episode_id"],
                "start_step": s["start_step"],
                "step_count": s["step_count"],
            }
            for s in train_selections
        ],
        "val": [
            {
                "episode_id": s["episode_id"],
                "start_step": s["start_step"],
                "step_count": s["step_count"],
            }
            for s in val_selections
        ],
        "composition": composition,
    }


def main() -> None:
    root = repo_root()
    sys.path.insert(0, str(root / "src"))

    default_dataset = root / "data" / "aemo_dt_fcas" / "aemo_fcas_dataset.parquet"
    default_raw_logs = root / "data" / "aemo_dt_fcas" / "raw_logs"
    default_output = root / "data" / "aemo_dt_fcas" / "autoresearch_pilot_spec.json"
    default_pilot_dir = root / "data" / "aemo_dt_fcas" / "autoresearch_pilot"

    parser = argparse.ArgumentParser(
        description="Build stratified pilot spec from FCAS-rich dataset",
    )
    parser.add_argument("--raw-logs-dir", type=Path, default=default_raw_logs,
                        help="Directory with raw episode parquet files")
    parser.add_argument("--dataset-path", type=Path, default=default_dataset,
                        help="Assembled FCAS dataset parquet")
    parser.add_argument("--output-path", type=Path, default=default_output,
                        help="Output path for the pilot spec JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible selection")
    parser.add_argument("--build-pilot", action="store_true",
                        help="Also run build_aemo_autoresearch_pilot.py to generate parquet files")
    parser.add_argument("--pilot-output-dir", type=Path, default=default_pilot_dir,
                        help="Where to write the pilot parquet files (only with --build-pilot)")
    args = parser.parse_args()

    print("Reconstructing episode metadata from raw log files...")
    metadata = reconstruct_episode_metadata(args.raw_logs_dir)
    print(f"  {metadata.height} raw log files mapped to episode_ids")

    print("Fetching episode lengths from assembled dataset...")
    lengths = get_episode_lengths(args.dataset_path)
    print(f"  {lengths.height} episode lengths retrieved")

    print(f"Selecting stratified episodes (train=8, val=4)...")
    train_sel, val_sel = select_episodes(
        metadata, lengths, TARGET_COMPOSITION, seed=args.seed,
    )

    print(f"\nTrain selections ({len(train_sel)}):")
    for s in train_sel:
        print(f"  ep={s['episode_id']:>4d} | {s['source_policy']:>10s} | {s['region']:>4s} | "
              f"{s['horizon']:>6s} | {s['battery']:>6s} | steps={s['start_step']}+{s['step_count']}")

    print(f"\nVal selections ({len(val_sel)}):")
    for s in val_sel:
        print(f"  ep={s['episode_id']:>4d} | {s['source_policy']:>10s} | {s['region']:>4s} | "
              f"{s['horizon']:>6s} | {s['battery']:>6s} | steps={s['start_step']}+{s['step_count']}")

    spec = build_spec(train_sel, val_sel, TARGET_COMPOSITION)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"\nSpec saved: {args.output_path}")

    if args.build_pilot:
        print(f"\nBuilding pilot dataset into {args.pilot_output_dir}...")
        builder = root / "src" / "build_aemo_autoresearch_pilot.py"
        cmd = [
            sys.executable, str(builder),
            "--dataset-path", str(args.dataset_path.resolve()),
            "--output-dir", str(args.pilot_output_dir.resolve()),
            "--spec-path", str(args.output_path.resolve()),
        ]
        subprocess.run(cmd, check=True)
        print(f"\nDone. Pilot dataset in: {args.pilot_output_dir}")


if __name__ == "__main__":
    main()
