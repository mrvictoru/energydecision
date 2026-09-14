"""A2 tests: symmetric round-trip efficiency in SolarBatteryEnv."""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from EnergySimEnv import SolarBatteryEnv


def _df(n=48):
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=30 * i) for i in range(n)]  # 0.5 h steps
    zeros = [0.0] * n
    return pl.DataFrame({
        "Timestamp": ts, "Time": ts,
        "SolarGen": zeros, "HouseLoad": zeros,
        "FutureSolar": zeros, "FutureLoad": zeros,
        "ImportEnergyPrice": [0.30] * n, "ExportEnergyPrice": [0.05] * n,
    })


def _env(rte):
    return SolarBatteryEnv(
        _df(), battery_capacity=4.0, max_battery_flow=4.0,
        init_battery_level=2.0, max_step=48, roundtrip_eff=rte,
    )


def test_charge_stores_efficiency_adjusted_energy():
    rte = 0.80
    eff = rte ** 0.5
    env = _env(rte)
    env.reset()
    level0 = env.battery_level
    _, _, _, _, info = env.step(np.array([1.0]))
    grid_energy = info["battery_flow_energy"]
    assert grid_energy > 0
    assert np.isclose(env.battery_level - level0, grid_energy * eff)


def test_discharge_draws_more_than_grid_delivery():
    rte = 0.80
    eff = rte ** 0.5
    env = _env(rte)
    env.reset()
    level0 = env.battery_level
    _, _, _, _, info = env.step(np.array([-1.0]))
    grid_energy = info["battery_flow_energy"]
    assert grid_energy < 0
    assert np.isclose(env.battery_level - level0, grid_energy / eff)


def test_roundtrip_eff_one_is_lossless():
    env = _env(1.0)
    env.reset()
    level0 = env.battery_level
    _, _, _, _, info = env.step(np.array([1.0]))
    assert np.isclose(env.battery_level - level0, info["battery_flow_energy"])
