"""A1 tests: calendar aging is included in household 'full' mode."""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from EnergySimEnv import SolarBatteryEnv


def _idle_df(n=48):
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=30 * i) for i in range(n)]  # 0.5 h steps
    zeros = [0.0] * n
    return pl.DataFrame({
        "Timestamp": ts, "Time": ts,
        "SolarGen": zeros, "HouseLoad": zeros,
        "FutureSolar": zeros, "FutureLoad": zeros,
        "ImportEnergyPrice": [0.30] * n, "ExportEnergyPrice": [0.05] * n,
    })


def _idle_episode(mode):
    env = SolarBatteryEnv(
        _idle_df(), battery_capacity=4.0, max_battery_flow=4.0,
        init_battery_level=2.0, max_step=48, degradation_mode=mode,
        degradation_chemistry="LFP", roundtrip_eff=1.0,
    )
    env.reset()
    last = {}
    for _ in range(48):
        _, _, term, trunc, last = env.step(np.array([0.0]))
        if term or trunc:
            break
    return env, last


def test_full_includes_calendar_aging_while_idle():
    env_cycle, _ = _idle_episode("cycle_only")
    env_full, info_full = _idle_episode("full")
    # No cycling while idle, so cycle-only degradation is zero.
    assert np.isclose(env_cycle.total_degradation, 0.0, atol=1e-12)
    # "full" accrues calendar aging every step even when idle.
    assert env_full.total_degradation > 0.0
    assert info_full["calendar_degradation"] > 0.0


def test_disabled_has_no_degradation():
    env, _ = _idle_episode("disabled")
    assert env.total_degradation == 0.0
