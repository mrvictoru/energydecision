import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aemo_notebook_utils import (  # noqa: E402
    build_dt_dataset_from_logs,
    default_aemo_dt_model_kwargs,
    get_sb3_model_class,
    load_episode_logs_from_parquet,
    resolve_battery_variants,
    validate_aemo_dt_dimensions,
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
    validate_aemo_dt_dimensions(manifest, action_mode="multi_market")


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


def test_resolve_battery_variants_derives_label_soc_and_cost():
    resolved = resolve_battery_variants(
        [
            {
                "name": "small",
                "capacity_mwh": 2.0,
                "max_power_mw": 1.0,
                "init_soc_ratio": 0.25,
            }
        ]
    )

    assert resolved[0]["label"] == "small"
    assert resolved[0]["battery_capacity"] == 2.0
    assert resolved[0]["max_battery_flow"] == 1.0
    assert resolved[0]["init_soc"] == 0.5
    assert resolved[0]["battery_life_cost"] == 700000.0


def test_get_sb3_model_class_supports_expected_algorithms():
    assert get_sb3_model_class("ppo").__name__ == "PPO"
    assert get_sb3_model_class("sac").__name__ == "SAC"


def test_validate_aemo_dt_dimensions_rejects_bad_state_dim():
    with pytest.raises(ValueError, match="state_dim=18"):
        validate_aemo_dt_dimensions(
            {"state_dims": [12], "act_dims": [3]},
            action_mode="multi_market",
        )


def test_validate_aemo_dt_dimensions_rejects_bad_action_dim():
    with pytest.raises(ValueError, match="act_dim=1"):
        validate_aemo_dt_dimensions(
            {"state_dims": [18], "act_dims": [3]},
            action_mode="simple",
        )


def test_new_notebooks_exist_and_expose_config_cells():
    repo_root = Path(__file__).resolve().parents[1]
    sim_nb = json.loads((repo_root / "aemo_simrun.ipynb").read_text())
    sb3_nb = json.loads((repo_root / "aemo_sb3train.ipynb").read_text())

    sim_code = "\n".join("".join(cell.get("source", [])) for cell in sim_nb["cells"] if cell["cell_type"] == "code")
    sb3_code = "\n".join("".join(cell.get("source", [])) for cell in sb3_nb["cells"] if cell["cell_type"] == "code")

    assert "BATTERY_VARIANTS" in sim_code
    assert "BEHAVIOR_RUNS" in sim_code
    assert "run_dispatch_replay" in sim_code or "build_dispatch_selection" in sim_code
    assert "train_sb3_model_on_aemo" in sb3_code
    assert "SB3_ALGORITHM" in sb3_code
