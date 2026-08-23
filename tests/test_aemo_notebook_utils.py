import json
import os
import sys
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import polars as pl
import pytest
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aemo_notebook_utils import (  # noqa: E402
    build_dt_dataset_from_logs,
    default_aemo_dt_model_kwargs,
    ensure_processed_cache_writable,
    fetch_and_preprocess_aemo_data,
    get_sb3_model_class,
    load_episode_logs_from_parquet,
    fine_tune_ppo_from_dt_on_aemo,
    fetch_and_preprocess_aemo_scenarios,
    fit_aemo_global_stats,
    make_multi_scenario_aemo_env_fns,
    partition_dt_dataset_by_episode,
    partition_dt_dataset_for_subset_training,
    resolve_dispatch_battery_life_cost,
    resolve_dispatch_replay_runs,
    resolve_dispatch_run_region,
    resolve_battery_variants,
    run_dt_episodes,
    should_run_dispatch_for_scenario,
    train_sb3_model_on_aemo,
    validate_aemo_dt_dimensions,
    warm_start_ppo_from_dt_episodes,
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
    # Legacy 3-dim multi_market actions are padded to 9-dim full_fcas on ingest.
    assert manifest["act_dims"] == [9]
    assert manifest["sources"]["rule"]["episodes"] == 1
    assert manifest["sources"]["dispatch"]["rows"] == 2
    validate_aemo_dt_dimensions(manifest, action_mode="full_fcas")


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


def test_partition_dt_dataset_by_episode_preserves_episode_boundaries(tmp_path: Path):
    dataset, manifest = build_dt_dataset_from_logs(
        {
            "rule": [_episode_df(0), _episode_df(10), _episode_df(20)],
            "dispatch": [_episode_df(30), _episode_df(40)],
        }
    )
    dataset_path = tmp_path / "aemo_dt_dataset.parquet"
    dataset.write_parquet(str(dataset_path))

    subset_manifest = partition_dt_dataset_by_episode(
        dataset_path=dataset_path,
        output_dir=tmp_path / "subsets",
        subset_episode_count=2,
    )

    assert subset_manifest["total_episode_count"] == manifest["episode_count"]
    assert subset_manifest["total_row_count"] == manifest["row_count"]
    assert subset_manifest["subset_count"] == 3

    combined_subset_rows = 0
    combined_episode_ids: list[int] = []
    for subset in subset_manifest["subsets"]:
        subset_df = pl.read_parquet(subset["path"])
        subset_episode_ids = sorted(subset_df["episode_id"].unique().to_list())
        assert subset_episode_ids == subset["episode_ids"]
        assert len(subset_episode_ids) <= 2
        combined_subset_rows += subset_df.height
        combined_episode_ids.extend(subset_episode_ids)

    assert combined_subset_rows == dataset.height
    assert sorted(combined_episode_ids) == sorted(dataset["episode_id"].unique().to_list())


def test_partition_dt_dataset_for_subset_training_creates_global_train_val_split(tmp_path: Path):
    dataset, manifest = build_dt_dataset_from_logs(
        {
            "rule": [_episode_df(0), _episode_df(10), _episode_df(20), _episode_df(30)],
            "dispatch": [_episode_df(40), _episode_df(50)],
        }
    )
    dataset_path = tmp_path / "aemo_dt_dataset.parquet"
    dataset.write_parquet(str(dataset_path))

    subset_manifest = partition_dt_dataset_for_subset_training(
        dataset_path=dataset_path,
        output_dir=tmp_path / "subset_training",
        subset_episode_count=2,
        val_split=0.34,
        seed=123,
    )

    assert subset_manifest["total_episode_count"] == manifest["episode_count"]
    assert subset_manifest["train_episode_count"] + subset_manifest["val_episode_count"] == manifest["episode_count"]
    assert subset_manifest["train_episode_count"] > 0

    train_ids = sorted(
        episode_id
        for subset in subset_manifest["train_subsets"]
        for episode_id in subset["episode_ids"]
    )
    val_ids = sorted(
        episode_id
        for subset in subset_manifest["val_subsets"]
        for episode_id in subset["episode_ids"]
    )
    assert not (set(train_ids) & set(val_ids))
    assert sorted(train_ids + val_ids) == sorted(dataset["episode_id"].unique().to_list())


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


def test_run_dt_episodes_loads_model_and_collects_logs(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, env, **kwargs):
            captured.setdefault("envs", []).append(env)
            captured.setdefault("kwargs", []).append(kwargs)

        def run_episode(self):
            return _episode_df(0), pl.DataFrame()

    def fake_load_dt_model(**kwargs):
        captured["load_kwargs"] = kwargs
        return object()

    monkeypatch.setattr("aemo_notebook_utils.load_dt_model", fake_load_dt_model)
    monkeypatch.setattr("aemo_notebook_utils.AEMOAgent", FakeAgent)
    monkeypatch.setattr("aemo_notebook_utils.create_aemo_env", lambda **kwargs: {"env_kwargs": kwargs})

    episodes = run_dt_episodes(
        processed_data=pl.DataFrame({"x": [1]}),
        battery_variant={"name": "medium", "capacity_mwh": 4.0, "max_power_mw": 2.0, "init_soc_ratio": 0.5},
        model_path="model.pt",
        model_config_path="model.json",
        num_episodes=2,
        max_step=16,
        step_duration=0.5,
        rtg_value=10.0,
        dt_gamma=0.95,
        base_seed=100,
    )

    assert len(episodes) == 2
    assert captured["load_kwargs"] == {
        "model_path": "model.pt",
        "model_config_path": "model.json",
        "device": "auto",
    }
    assert [kwargs["reset_seed"] for kwargs in captured["kwargs"]] == [100, 101]
    assert all(kwargs["algorithm"] == "dt" for kwargs in captured["kwargs"])
    assert all(kwargs["rtg_value"] == 10.0 for kwargs in captured["kwargs"])
    assert all(kwargs["dt_gamma"] == 0.95 for kwargs in captured["kwargs"])


def test_train_sb3_model_on_aemo_forwards_model_hooks(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "aemo_notebook_utils.make_aemo_env_fns",
        lambda **kwargs: [lambda: "env"],
    )
    monkeypatch.setattr(
        "aemo_notebook_utils.create_aemo_env",
        lambda **kwargs: "eval_env",
    )

    def fake_train_model(**kwargs):
        captured.update(kwargs)
        return "model", {"Post_training": {"mean_reward": 1.0, "std_reward": 0.0}}

    monkeypatch.setattr("aemo_notebook_utils.train_model", fake_train_model)

    model, eval_result = train_sb3_model_on_aemo(
        processed_data=pl.DataFrame({"x": [1]}),
        algorithm="ppo",
        battery_variants=[{"name": "medium", "capacity_mwh": 4.0, "max_power_mw": 2.0, "init_soc_ratio": 0.5}],
        episodes_per_variant=1,
        max_step=12,
        step_duration=0.5,
        model_kwargs_override={"n_steps": 32},
        model_post_create_fn=lambda model: model,
    )

    assert model == "model"
    assert eval_result["Post_training"]["mean_reward"] == 1.0
    assert captured["model_class"].__name__ == "PPO"
    assert captured["model_kwargs_override"] == {"n_steps": 32}
    assert callable(captured["model_post_create_fn"])


class _TinyBoxEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=float)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=float)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def test_warm_start_ppo_from_dt_episodes_updates_policy():
    model = PPO("MlpPolicy", _TinyBoxEnv(), n_steps=8, batch_size=4, policy_kwargs={"net_arch": [32, 32]})
    initial_weights = model.policy.action_net.weight.detach().clone()
    episodes = [_episode_df(0), _episode_df(20)]

    summary = warm_start_ppo_from_dt_episodes(
        model=model,
        episodes=episodes,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        max_batches=4,
    )

    assert summary["episode_count"] == 2.0
    assert summary["sample_count"] == 4.0
    assert summary["batch_count"] > 0
    assert not torch.allclose(initial_weights, model.policy.action_net.weight.detach())


def test_fine_tune_ppo_from_dt_on_aemo_uses_dt_seed_rollouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seed_path = tmp_path / "dt_seed_logs.parquet"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "aemo_notebook_utils.run_dt_episodes",
        lambda **kwargs: [_episode_df(0)],
    )
    monkeypatch.setattr(
        "aemo_notebook_utils.warm_start_ppo_from_dt_episodes",
        lambda **kwargs: {"actor_loss": 0.1, "value_loss": 0.2, "batch_count": 1.0},
    )

    def fake_train_sb3_model_on_aemo(**kwargs):
        captured.update(kwargs)
        kwargs["model_post_create_fn"]("fake_model")
        return "trained_model", {"Post_training": {"mean_reward": 2.0, "std_reward": 0.1}}

    monkeypatch.setattr("aemo_notebook_utils.train_sb3_model_on_aemo", fake_train_sb3_model_on_aemo)

    model, eval_result, manifest = fine_tune_ppo_from_dt_on_aemo(
        processed_data=pl.DataFrame({"x": [1]}),
        dt_model_path="dt.pt",
        dt_model_config_path="dt.json",
        battery_variants=[{"name": "medium", "capacity_mwh": 4.0, "max_power_mw": 2.0, "init_soc_ratio": 0.5}],
        seed_episodes_per_variant=1,
        max_step=12,
        step_duration=0.5,
        seed_logs_output_path=seed_path,
    )

    assert model == "trained_model"
    assert eval_result["Post_training"]["mean_reward"] == 2.0
    assert captured["algorithm"] == "ppo"
    assert manifest["seed_episode_count"] == 1
    assert manifest["warm_start"]["actor_loss"] == 0.1
    assert seed_path.exists()


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
    notebooks_dir = repo_root / "notebooks"
    sim_nb = json.loads((notebooks_dir / "aemo_simrun.ipynb").read_text())
    sb3_nb = json.loads((notebooks_dir / "aemo_sb3train.ipynb").read_text())

    sim_code = "\n".join("".join(cell.get("source", [])) for cell in sim_nb["cells"] if cell["cell_type"] == "code")
    sb3_code = "\n".join("".join(cell.get("source", [])) for cell in sb3_nb["cells"] if cell["cell_type"] == "code")

    assert "BATTERY_VARIANTS" in sim_code
    assert "BEHAVIOR_RUNS" in sim_code
    assert "DISPATCH_RUNS" in sim_code
    assert "run_dispatch_replay" in sim_code or "build_dispatch_selection" in sim_code
    assert "train_sb3_model_on_aemo" in sb3_code
    assert "SB3_ALGORITHM" in sb3_code


def test_resolve_dispatch_replay_runs_assigns_labels_and_defaults():
    resolved = resolve_dispatch_replay_runs(
        [
            {
                "station_name": "hornsdale",
                "episodes": 2,
            }
        ]
    )

    assert resolved[0]["label"] == "hornsdale"
    assert resolved[0]["episodes"] == 2
    assert resolved[0]["init_soc_ratio"] == 0.5
    assert resolved[0]["dispatch_index"] == 0


def test_resolve_dispatch_battery_life_cost_uses_station_capacity():
    cost = resolve_dispatch_battery_life_cost(
        dispatch_run={"battery_cost_per_kwh": 75.0},
        station_capacity_mwh=193.5,
    )

    assert cost > 0.0
    assert cost > 193.5 * 1000.0 * 75.0


def test_resolve_dispatch_run_region_returns_registry_region():
    region = resolve_dispatch_run_region(
        dispatch_station="hornsdale",
        dispatch_duid=None,
        start_date=datetime(2022, 4, 1),
        end_date=datetime(2023, 12, 1),
    )

    assert region == "SA1"


def test_should_run_dispatch_for_scenario_skips_region_mismatch():
    should_run, dispatch_region = should_run_dispatch_for_scenario(
        scenario_region="NSW1",
        dispatch_station="hornsdale",
        dispatch_duid=None,
        start_date=datetime(2022, 4, 1),
        end_date=datetime(2023, 12, 1),
    )

    assert should_run is False
    assert dispatch_region == "SA1"


def test_should_run_dispatch_for_scenario_allows_region_match():
    should_run, dispatch_region = should_run_dispatch_for_scenario(
        scenario_region="SA1",
        dispatch_station="hornsdale",
        dispatch_duid=None,
        start_date=datetime(2022, 4, 1),
        end_date=datetime(2023, 12, 1),
    )

    assert should_run is True
    assert dispatch_region == "SA1"


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


def test_ensure_processed_cache_writable_rejects_locked_file(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "processed_SA1_2024-01-01_2024-01-02_0.0833h.parquet"
    cache_path.write_bytes(b"stub")

    # A stale cache file that is readable but not writable must be rejected when a
    # (re)write is requested (refresh / fixed-stats path).
    monkeypatch.setattr(
        "aemo_notebook_utils.os.access",
        lambda path, mode: not (path == cache_path and mode == os.W_OK),
    )

    with pytest.raises(PermissionError, match="not writable"):
        ensure_processed_cache_writable(
            cache_dir=tmp_path,
            region="SA1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            step_duration=5 / 60,
            needs_write=True,
        )


def test_ensure_processed_cache_writable_rejects_unreadable_file(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "processed_SA1_2024-01-01_2024-01-02_0.0833h.parquet"
    cache_path.write_bytes(b"stub")

    # A cache file the runtime cannot read at all must be rejected regardless of
    # whether a write is requested.
    monkeypatch.setattr(
        "aemo_notebook_utils.os.access",
        lambda path, mode: path != cache_path,
    )

    with pytest.raises(PermissionError, match="not readable"):
        ensure_processed_cache_writable(
            cache_dir=tmp_path,
            region="SA1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2),
            step_duration=5 / 60,
            needs_write=True,
        )


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
