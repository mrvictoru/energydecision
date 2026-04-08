import json
import os
import sys
from pathlib import Path
from datetime import datetime

import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aemo_notebook_utils import (  # noqa: E402
    build_dt_dataset_from_logs,
    default_aemo_dt_model_kwargs,
    fetch_and_preprocess_aemo_data,
    get_sb3_model_class,
    load_episode_logs_from_parquet,
    fetch_and_preprocess_aemo_scenarios,
    fit_aemo_global_stats,
    make_multi_scenario_aemo_env_fns,
    resolve_battery_variants,
    validate_aemo_dt_dimensions,
    write_combined_episode_logs,
)
from AEMOBatteryEnv import AEMODataPreprocessor  # noqa: E402


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


def _raw_bundle(rrp_values: list[float], demand_values: list[float], generation_values: list[float]) -> dict[str, pl.DataFrame]:
    settlement_dates = [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 5)]
    prices = pl.DataFrame(
        {
            "SETTLEMENTDATE": settlement_dates,
            "RRP": rrp_values,
            "TOTALDEMAND": demand_values,
        }
    )
    fcas_rows = []
    for settlement_date, price in zip(settlement_dates, generation_values):
        for service in [
            "RAISEREG",
            "LOWERREG",
            "RAISE6SEC",
            "LOWER6SEC",
            "RAISE60SEC",
            "LOWER60SEC",
            "RAISE5MIN",
            "LOWER5MIN",
        ]:
            fcas_rows.append(
                {
                    "SETTLEMENTDATE": settlement_date,
                    "SERVICE": service,
                    "PRICE": float(price),
                }
            )
    generation_rows = []
    for settlement_date, generation in zip(settlement_dates, generation_values):
        for fuel in ["solar", "wind"]:
            generation_rows.append(
                {
                    "SETTLEMENTDATE": settlement_date,
                    "FUEL_TYPE": fuel,
                    "GENERATION": float(generation),
                }
            )
    return {
        "prices": prices,
        "fcas": pl.DataFrame(fcas_rows),
        "generation": pl.DataFrame(generation_rows),
    }


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
    assert resolved[0]["battery_life_cost"] == pytest.approx(291150.0, rel=1e-3)


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


def test_fit_aemo_global_stats_and_locked_preprocessing(monkeypatch, tmp_path: Path):
    scenarios = [
        {
            "label": "sa1_window",
            "region": "SA1",
            "start_date": datetime(2024, 1, 1),
            "end_date": datetime(2024, 1, 2),
        },
        {
            "label": "vic1_window",
            "region": "VIC1",
            "start_date": datetime(2024, 2, 1),
            "end_date": datetime(2024, 2, 2),
        },
    ]
    bundles = {
        ("SA1", datetime(2024, 1, 1), datetime(2024, 1, 2)): _raw_bundle([10.0, 20.0], [100.0, 120.0], [5.0, 7.0]),
        ("VIC1", datetime(2024, 2, 1), datetime(2024, 2, 2)): _raw_bundle([100.0, 200.0], [300.0, 500.0], [9.0, 11.0]),
    }

    def fake_fetch_aemo_data_bundle(*, start_date, end_date, region, **kwargs):
        return bundles[(region, start_date, end_date)]

    monkeypatch.setattr("aemo_notebook_utils.fetch_aemo_data_bundle", fake_fetch_aemo_data_bundle)

    stats, manifest = fit_aemo_global_stats(
        scenarios=scenarios,
        cache_dir=tmp_path,
        step_duration=5 / 60,
        refresh=False,
    )

    assert manifest[0]["label"] == "sa1_window"
    assert stats["RRP"]["min"] == 10.0
    assert stats["RRP"]["max"] == 200.0
    assert stats["TOTALDEMAND"]["min"] == 100.0
    assert stats["TOTALDEMAND"]["max"] == 500.0
    assert stats["FCAS_PRICE"]["min"] == 5.0
    assert stats["FCAS_PRICE"]["max"] == 11.0

    processed_by_label, _ = fetch_and_preprocess_aemo_scenarios(
        scenarios=scenarios,
        cache_dir=tmp_path,
        step_duration=5 / 60,
        refresh=False,
        fixed_stats=stats,
    )

    assert set(processed_by_label) == {"sa1_window", "vic1_window"}
    for frame in processed_by_label.values():
        assert "RRP_normalized" in frame.columns
        assert "DEMAND_normalized" in frame.columns
        assert frame.select(pl.col("RRP_normalized").min()).item() >= 0.0
        assert frame.select(pl.col("RRP_normalized").max()).item() <= 1.0


def test_fixed_stats_ignore_stale_processed_cache(monkeypatch, tmp_path: Path):
    raw = _raw_bundle([10.0, 20.0], [100.0, 120.0], [5.0, 7.0])

    def fake_fetch_aemo_data_bundle(*args, **kwargs):
        return raw

    monkeypatch.setattr("aemo_notebook_utils.fetch_aemo_data_bundle", fake_fetch_aemo_data_bundle)

    baseline, cache_path = fetch_and_preprocess_aemo_data(
        region="SA1",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        cache_dir=tmp_path,
        step_duration=5 / 60,
        refresh=False,
    )
    assert cache_path.exists()

    locked, _ = fetch_and_preprocess_aemo_data(
        region="SA1",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 2),
        cache_dir=tmp_path,
        step_duration=5 / 60,
        refresh=False,
        fixed_stats={
            "RRP": {"min": 0.0, "max": 100.0},
            "FCAS_PRICE": {"min": 0.0, "max": 20.0},
            "TOTALDEMAND": {"min": 0.0, "max": 200.0},
            "GENERATION": {"min": 0.0, "max": 10.0},
        },
    )

    assert baseline.select(pl.col("RRP_normalized").max()).item() == 1.0
    assert locked.select(pl.col("RRP_normalized").max()).item() < 1.0


def test_fixed_stats_handle_zero_span_normalization():
    raw = _raw_bundle([50.0, 50.0], [100.0, 100.0], [5.0, 5.0])
    preprocessor = AEMODataPreprocessor(
        step_duration_hours=5 / 60,
        fixed_stats={
            "RRP": {"min": 50.0, "max": 50.0},
            "FCAS_PRICE": {"min": 5.0, "max": 5.0},
            "TOTALDEMAND": {"min": 100.0, "max": 100.0},
            "GENERATION": {"min": 0.0, "max": 10.0},
        },
    )

    processed = preprocessor.preprocess_aemo_data(
        prices=raw["prices"],
        fcas=raw["fcas"],
        generation=raw["generation"],
    )

    assert processed.select(pl.col("RRP_normalized").unique()).to_series().to_list() == [0.0]
    assert processed.select(pl.col("DEMAND_normalized").unique()).to_series().to_list() == [0.0]


def test_make_multi_scenario_aemo_env_fns_counts_all_combinations():
    scenario_data = [
        (
            {"label": "sa1_window", "region": "SA1"},
            pl.DataFrame(
                {
                    "SETTLEMENTDATE": [datetime(2024, 1, 1)],
                    "RRP": [100.0],
                    "TOTALDEMAND": [1000.0],
                    "FCAS_RAISEREG": [10.0],
                    "FCAS_LOWERREG": [10.0],
                    "GEN_solar": [5.0],
                    "GEN_wind": [5.0],
                    "hour_sin": [0.0],
                    "hour_cos": [1.0],
                    "day_sin": [0.0],
                    "day_cos": [1.0],
                    "is_peak": [1.0],
                    "RRP_normalized": [0.5],
                    "DEMAND_normalized": [0.5],
                    "FCAS_RAISEREG_normalized": [0.5],
                    "FCAS_LOWERREG_normalized": [0.5],
                    "GEN_solar_pct": [0.5],
                    "GEN_wind_pct": [0.5],
                }
            ),
        ),
        (
            {"label": "vic1_window", "region": "VIC1"},
            pl.DataFrame(
                {
                    "SETTLEMENTDATE": [datetime(2024, 2, 1)],
                    "RRP": [120.0],
                    "TOTALDEMAND": [1200.0],
                    "FCAS_RAISEREG": [12.0],
                    "FCAS_LOWERREG": [12.0],
                    "GEN_solar": [6.0],
                    "GEN_wind": [6.0],
                    "hour_sin": [0.0],
                    "hour_cos": [1.0],
                    "day_sin": [0.0],
                    "day_cos": [1.0],
                    "is_peak": [1.0],
                    "RRP_normalized": [0.6],
                    "DEMAND_normalized": [0.6],
                    "FCAS_RAISEREG_normalized": [0.6],
                    "FCAS_LOWERREG_normalized": [0.6],
                    "GEN_solar_pct": [0.5],
                    "GEN_wind_pct": [0.5],
                }
            ),
        ),
    ]

    env_fns = make_multi_scenario_aemo_env_fns(
        scenario_data=scenario_data,
        battery_variants=[{"name": "small", "capacity_mwh": 2.0, "max_power_mw": 1.0, "init_soc_ratio": 0.5}],
        episodes_per_variant=2,
        max_step=1,
        step_duration=5 / 60,
    )

    assert len(env_fns) == 4
