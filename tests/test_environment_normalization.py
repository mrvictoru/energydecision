"""A7/A8 tests: observation-normalization de-confounding."""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from EnergySimEnv import SolarBatteryEnv


def _df(n=8):
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=30 * i) for i in range(n)]
    zeros = [0.0] * n
    return pl.DataFrame({
        "Timestamp": ts, "Time": ts,
        "SolarGen": zeros, "HouseLoad": zeros,
        "FutureSolar": zeros, "FutureLoad": zeros,
        "ImportEnergyPrice": [0.30] * n, "ExportEnergyPrice": [0.05] * n,
    })


def test_deg_cost_normalizer_is_life_cost_independent():
    e1 = SolarBatteryEnv(_df(), battery_life_cost=1000.0, roundtrip_eff=1.0)
    e2 = SolarBatteryEnv(_df(), battery_life_cost=10000.0, roundtrip_eff=1.0)
    # A7: the normalized deg-cost scale is a fixed reference (0.001 * 5000).
    assert e1.battery_deg_cost_max_for_norm == 5.0
    assert e2.battery_deg_cost_max_for_norm == 5.0


def test_battery_level_normalized_by_current_capacity():
    env = SolarBatteryEnv(_df(), battery_capacity=4.0, max_battery_flow=2.0,
                          init_battery_level=2.0, roundtrip_eff=1.0)
    env.reset()
    # Simulate capacity fade: 1.5 kWh / 3.0 kWh = 0.5 SOC fraction.
    env.battery_capacity = 3.0
    env.battery_level = 1.5
    _, _, _, _, normalized_extra = env._get_observation_components()
    assert np.isclose(normalized_extra[0], 0.5, atol=1e-6)
