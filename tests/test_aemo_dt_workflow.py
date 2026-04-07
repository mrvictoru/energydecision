import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aemo_dt_workflow import (  # noqa: E402
    build_dt_dataset_from_logs,
    default_aemo_dt_model_kwargs,
    load_episode_logs_from_parquet,
    write_combined_episode_logs,
)


def _episode_df(offset: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "step": [0, 1],
            "norm_observation": [
                [float(offset + i) for i in range(18)],
                [float(offset + i + 1) for i in range(18)],
            ],
            "action": [
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ],
            "reward": [1.0, -0.5],
        }
    )


def test_default_aemo_dt_model_kwargs_matches_multimarket_aemo():
    kwargs = default_aemo_dt_model_kwargs()
    assert kwargs["state_dim"] == 18
    assert kwargs["act_dim"] == 3
    assert kwargs["context_len"] == 288
    assert kwargs["max_timestep"] == 2016
    assert kwargs["rope_enabled"] is True


def test_build_dt_dataset_from_logs_tracks_sources_and_episode_ids():
    dataset, manifest = build_dt_dataset_from_logs(
        {
            "rule": [_episode_df(0)],
            "dispatch": [_episode_df(10)],
        }
    )

    assert dataset.height == 4
    assert sorted(dataset["episode_id"].unique().to_list()) == [0, 1]
    assert sorted(dataset["source_policy"].unique().to_list()) == ["dispatch", "rule"]
    assert manifest["episode_count"] == 2
    assert manifest["row_count"] == 4
    assert manifest["state_dims"] == [18]
    assert manifest["act_dims"] == [3]
    assert manifest["sources"]["rule"]["episodes"] == 1
    assert manifest["sources"]["dispatch"]["rows"] == 2


def test_write_and_load_combined_episode_logs_round_trip(tmp_path: Path):
    output_path = tmp_path / "episodes.parquet"
    original = [_episode_df(0), _episode_df(100)]

    combined = write_combined_episode_logs(episodes=original, output_path=output_path)
    loaded = load_episode_logs_from_parquet(output_path)

    assert combined.height == 4
    assert output_path.exists()
    assert len(loaded) == 2
    assert loaded[0].height == 2
    assert loaded[1].height == 2
