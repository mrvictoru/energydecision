import json
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import build_aemo_autoresearch_pilot as pilot_builder  # noqa: E402


def _episode_rows(episode_id: int, source_policy: str, *, steps: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(steps):
        rows.append(
            {
                "episode_id": episode_id,
                "step": step,
                "norm_observation": [float(step), float(step + 1)],
                "action": [float(step) / 10.0],
                "reward": float(step) / 100.0,
                "source_policy": source_policy,
            }
        )
    return rows


def test_build_pilot_split_writes_curated_train_and_val(tmp_path: Path):
    dataset_path = tmp_path / "aemo_dt_dataset.parquet"
    pl.DataFrame(
        _episode_rows(10, "nsw_rule", steps=8)
        + _episode_rows(20, "qld_a2c", steps=8)
        + _episode_rows(30, "vic_td3", steps=8)
    ).write_parquet(dataset_path)

    spec = {
        "description": "test pilot",
        "train": [
            {"episode_id": 10, "start_step": 1, "step_count": 3},
            {"episode_id": 20, "start_step": 2, "step_count": 3},
        ],
        "val": [
            {"episode_id": 30, "start_step": 0, "step_count": 4},
        ],
    }

    manifest = pilot_builder.build_pilot_split(
        dataset_path=dataset_path,
        output_dir=tmp_path / "pilot",
        spec=spec,
    )

    train_df = pl.read_parquet(tmp_path / "pilot" / "aemo_dt_train_pilot.parquet")
    val_df = pl.read_parquet(tmp_path / "pilot" / "aemo_dt_val_pilot.parquet")
    saved_manifest = json.loads((tmp_path / "pilot" / "aemo_dt_autoresearch_pilot_manifest.json").read_text(encoding="utf-8"))

    assert train_df.height == 6
    assert val_df.height == 4
    assert sorted(train_df["episode_id"].unique().to_list()) == [0, 1]
    assert sorted(val_df["episode_id"].unique().to_list()) == [0]
    assert manifest["train_episode_count"] == 2
    assert manifest["val_episode_count"] == 1
    assert saved_manifest["train"][0]["source_episode_id"] == 10
    assert saved_manifest["val"][0]["source_policy"] == "vic_td3"
