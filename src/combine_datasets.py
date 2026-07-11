"""Combine v2, v3 (short), and v3_medium (medium) datasets for MoLab retraining.

Usage:
    python3 src/combine_datasets.py --output-dir data/aemo_dt_fcas_combined
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine v2 + v3 + v3_medium datasets")
    parser.add_argument("--output-dir", type=Path, default=repo_root() / "data" / "aemo_dt_fcas_combined",
                        help="Output directory for combined dataset")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("v2 (SB3)", root / "data" / "aemo_dt_fcas_v2" / "aemo_fcas_dataset.parquet"),
        ("v3 short (GRPO)", root / "data" / "aemo_dt_fcas_v3" / "aemo_fcas_dataset.parquet"),
        ("v3 medium (GRPO)", root / "data" / "aemo_dt_fcas_v3_medium" / "aemo_fcas_dataset.parquet"),
    ]

    all_frames: list[pl.DataFrame] = []
    total_eps = 0
    total_rows = 0
    source_summary: dict[str, dict[str, int]] = {}
    next_episode_id = 0

    for label, path in sources:
        if not path.exists():
            print(f"  [SKIP] {label} — file not found: {path}")
            continue

        df = pl.read_parquet(path)
        n_eps = df["episode_id"].n_unique()
        n_rows = df.height

        # Reassign episode_ids to avoid collisions
        old_ids = df["episode_id"].unique().to_list()
        id_map = {old: new for new, old in enumerate(sorted(old_ids), start=next_episode_id)}
        mapped = df.with_columns(
            pl.col("episode_id").replace_strict(id_map).cast(pl.Int32)
        )
        all_frames.append(mapped)

        source_summary[label] = {"episodes": n_eps, "rows": n_rows}
        total_eps += n_eps
        total_rows += n_rows
        next_episode_id += n_eps
        print(f"  [OK]   {label}: {n_eps} episodes, {n_rows:,} rows")

    if not all_frames:
        print("ERROR: no dataset files found")
        return 1

    combined = pl.concat(all_frames, how="diagonal_relaxed")
    out_path = output_dir / "aemo_fcas_dataset.parquet"
    combined.write_parquet(out_path)

    source_summary["total"] = {"episodes": total_eps, "rows": total_rows}

    manifest = {
        "episode_count": total_eps,
        "total_rows": total_rows,
        "source_summary": source_summary,
        "output_path": str(out_path),
    }
    manifest_path = output_dir / "aemo_fcas_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*60}")
    print(f"  Combined dataset: {total_eps} episodes, {total_rows:,} rows")
    print(f"  Output: {out_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
