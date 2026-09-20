"""Tests for the H1.5 whole-day synthetic household generator."""

import datetime as dt

import numpy as np
import polars as pl

from EnergySimEnv import SolarBatteryEnv
from household_synthetic import (
    DayLibrary,
    assemble_episode,
    cluster_purity,
    inject_appliances,
    validate_gates,
)


def _day_frame(date, profile_kind=0):
    slots = np.arange(288)
    if profile_kind == 0:
        load = 0.45 + 2.2 * np.exp(-((slots - 12 * 8) / 16) ** 2)
    else:
        load = 0.45 + 2.2 * np.exp(-((slots - 12 * 19) / 16) ** 2)
    solar = np.maximum(0.0, 3.5 * np.sin(np.pi * (slots - 72) / 144))
    timestamps = [dt.datetime.combine(date, dt.time()) + dt.timedelta(minutes=5 * int(i)) for i in slots]
    return pl.DataFrame({
        "Timestamp": timestamps,
        "Time": timestamps,
        "HouseLoad": load,
        "SolarGen": solar,
        "ImportEnergyPrice": np.full(288, 0.30),
        "ExportEnergyPrice": np.full(288, 0.05),
        "BatterySOC": np.full(288, 0.5),
        "BatteryPower": np.zeros(288),
    })


def test_day_library_clusters_normalized_profiles_with_high_purity(tmp_path):
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    # Mondays only keep all examples in the winter/weekday k-means stratum.
    first = dt.date(2025, 6, 2)
    labels = {}
    for index in range(20):
        date = first + dt.timedelta(days=index * 7)
        kind = index % 2
        _day_frame(date, kind).write_parquet(normalized / f"sma_{date}_normalized.parquet")
        labels[date] = kind

    library = DayLibrary.from_normalized_dir(normalized, n_clusters=2, random_seed=7)
    members = library.group_days("winter", "weekday")
    assert len(members) == 13  # June-August examples
    purity = cluster_purity([day.cluster for day in members], [labels[day.source_date] for day in members])
    assert purity > 0.8

    sampled = library.sample("winter", "weekday", "family-ev", n_days=3, scale=1.2)
    assert len(sampled) == 3
    assert all(len(day.frame) == 288 for day in sampled)
    assert all(day.scale == 1.2 for day in sampled)


def test_g1_to_g6_harness_accepts_realistic_day_and_rejects_physical_violations():
    days = [_day_frame(dt.date(2025, 6, 2) + dt.timedelta(days=index)) for index in range(7)]
    result = validate_gates(days, days)
    assert result.passed
    assert all(result.gates.values())

    invalid = days[0].with_columns(pl.lit(-0.1).alias("SolarGen"))
    result = validate_gates([invalid, *days[1:]], days)
    assert not result.gates["G6"]

    idle = days[0].with_columns([
        pl.lit(0.0).alias("HouseLoad"),
        pl.lit(0.0).alias("SolarGen"),
    ])
    result = validate_gates([idle, *days[1:]], days)
    assert not result.gates["G5"]


def test_appliance_injection_is_additive_and_capped():
    base = _day_frame(dt.date(2025, 6, 2))
    generated, params = inject_appliances(
        base,
        season="summer",
        archetype="family-ev",
        day_type="weekday",
        rng=np.random.default_rng(1),
    )
    assert (generated["HouseLoad"] >= base["HouseLoad"]).all()
    assert params["injection_applied_kwh"] <= params["base_load_energy_kwh"] * 0.60 + 1e-9


def test_assembled_episode_instantiates_twelve_observation_environment():
    days = [_day_frame(dt.date(2025, 6, 2) + dt.timedelta(days=index)) for index in range(7)]
    episode = assemble_episode(days, episode_start=dt.date(2025, 6, 2))
    assert episode.height == 7 * 288
    assert set(episode.columns) == {
        "Timestamp", "Time", "SolarGen", "HouseLoad",
        "FutureSolar", "FutureLoad", "ImportEnergyPrice", "ExportEnergyPrice",
    }
    env = SolarBatteryEnv(episode, battery_capacity=10.0, max_battery_flow=5.0, max_step=episode.height)
    observation, _ = env.reset()
    assert observation.shape == (12,)
    assert env.step_duration == 5.0 / 60.0
