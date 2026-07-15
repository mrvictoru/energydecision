from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

NEEDED_COLUMNS = ["episode_id", "step", "norm_observation", "action", "reward", "source_policy"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_spec() -> dict[str, Any]:
    return {
        "description": (
            "Curated fixed AEMO pilot split for autoresearch. Each selection is a contiguous week-long "
            "slice, the train set mixes regions and policy families, and validation stays rule-heavy to "
            "keep comparisons stable."
        ),
        "train": [
            {"episode_id": 0, "start_step": 0, "step_count": 2016},
            {"episode_id": 45, "start_step": 2048, "step_count": 2016},
            {"episode_id": 64, "start_step": 4096, "step_count": 2016},
            {"episode_id": 96, "start_step": 6144, "step_count": 2016},
            {"episode_id": 114, "start_step": 8192, "step_count": 2016},
            {"episode_id": 160, "start_step": 10240, "step_count": 2016},
        ],
        "val": [
            {"episode_id": 7, "start_step": 1024, "step_count": 2016},
            {"episode_id": 37, "start_step": 3072, "step_count": 2016},
            {"episode_id": 105, "start_step": 5120, "step_count": 2016},
            {"episode_id": 133, "start_step": 7168, "step_count": 2016},
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small fixed AEMO DT pilot train/validation split for autoresearch.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=repo_root() / "data" / "aemo_dt" / "aemo_dt_dataset.parquet",
        help="Source AEMO DT dataset parquet with episode_id and source_policy columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "data" / "aemo_dt" / "autoresearch_pilot",
        help="Directory where the pilot train/val parquet files and manifest will be written.",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=None,
        help="Optional JSON file describing the pilot episode slices to extract.",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_spec(spec_path: Path | None) -> dict[str, Any]:
    if spec_path is None:
        return default_spec()
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _normalize_selections(raw: Sequence[dict[str, Any]]) -> list[dict[str, int]]:
    selections: list[dict[str, int]] = []
    for idx, entry in enumerate(raw):
        try:
            episode_id = int(entry["episode_id"])
            start_step = int(entry.get("start_step", 0))
            step_count = int(entry["step_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid pilot selection at index {idx}: {entry!r}") from exc
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0 for episode_id={episode_id}.")
        if step_count < 1:
            raise ValueError(f"step_count must be >= 1 for episode_id={episode_id}.")
        selections.append(
            {
                "episode_id": episode_id,
                "start_step": start_step,
                "step_count": step_count,
            }
        )
    if not selections:
        raise ValueError("Pilot split selections must be non-empty.")
    return selections


def _load_selected_episodes(dataset_path: Path, episode_ids: Sequence[int]) -> pl.DataFrame:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"AEMO DT dataset not found: {dataset_path}")
    return (
        pl.scan_parquet(str(dataset_path))
        .filter(pl.col("episode_id").is_in([int(episode_id) for episode_id in episode_ids]))
        .select(NEEDED_COLUMNS)
        .collect()
        .sort(["episode_id", "step"])
    )


def _slice_split(
    source_df: pl.DataFrame,
    *,
    split_name: str,
    selections: Sequence[dict[str, int]],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    rows: list[pl.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for local_episode_id, selection in enumerate(selections):
        episode_id = int(selection["episode_id"])
        start_step = int(selection["start_step"])
        step_count = int(selection["step_count"])
        episode_df = source_df.filter(pl.col("episode_id") == episode_id).sort("step")
        if episode_df.height == 0:
            raise ValueError(f"{split_name} pilot selection references missing episode_id={episode_id}.")
        if start_step + step_count > episode_df.height:
            raise ValueError(
                f"{split_name} pilot selection episode_id={episode_id} requires rows "
                f"{start_step}:{start_step + step_count}, but only {episode_df.height} rows exist."
            )
        source_policy_values = episode_df.get_column("source_policy").unique().to_list()
        if len(source_policy_values) != 1:
            raise ValueError(
                f"{split_name} pilot selection episode_id={episode_id} has multiple source policies: "
                f"{sorted(str(value) for value in source_policy_values)}"
            )
        sliced = episode_df.slice(start_step, step_count).with_columns(pl.lit(local_episode_id).alias("episode_id"))
        rows.append(sliced)
        manifest_rows.append(
            {
                "local_episode_id": local_episode_id,
                "source_episode_id": episode_id,
                "source_policy": str(source_policy_values[0]),
                "start_step": start_step,
                "step_count": step_count,
            }
        )
    return pl.concat(rows, how="vertical"), manifest_rows


def build_pilot_split(
    *,
    dataset_path: Path,
    output_dir: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    train_selections = _normalize_selections(spec.get("train", []))
    val_selections = _normalize_selections(spec.get("val", []))
    selected_episode_ids = sorted(
        {
            int(selection["episode_id"])
            for selection in [*train_selections, *val_selections]
        }
    )
    source_df = _load_selected_episodes(dataset_path, selected_episode_ids)
    train_df, train_manifest = _slice_split(source_df, split_name="train", selections=train_selections)
    val_df, val_manifest = _slice_split(source_df, split_name="val", selections=val_selections)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "aemo_dt_train_pilot.parquet"
    val_path = output_dir / "aemo_dt_val_pilot.parquet"
    manifest_path = output_dir / "aemo_dt_autoresearch_pilot_manifest.json"
    train_df.write_parquet(str(train_path))
    val_df.write_parquet(str(val_path))

    manifest = {
        "schema": "energydecision.aemo_autoresearch_pilot.v1",
        "description": spec.get("description"),
        "dataset_path": str(dataset_path.resolve()),
        "train_path": str(train_path.resolve()),
        "val_path": str(val_path.resolve()),
        "train_row_count": int(train_df.height),
        "val_row_count": int(val_df.height),
        "train_episode_count": len(train_manifest),
        "val_episode_count": len(val_manifest),
        "train": train_manifest,
        "val": val_manifest,
    }
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_pilot_split(
        dataset_path=args.dataset_path.resolve(),
        output_dir=args.output_dir.resolve(),
        spec=_load_spec(args.spec_path.resolve() if args.spec_path is not None else None),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
