"""Regression tests for Oracle -> env FCAS bid ordering (known_issues B2).

The Oracle solves raise and lower services as separate grouped vectors while
the environment interleaves all 8 services. ``oracle_fcas_bids_to_env_order``
must place each service in the env's action slot.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_oracle_algo import oracle_fcas_bids_to_env_order


ENV_FCAS_ORDER = [
    "RAISEREG", "LOWERREG", "RAISE6SEC", "LOWER6SEC",
    "RAISE60SEC", "LOWER60SEC", "RAISE5MIN", "LOWER5MIN",
]


def _make_full_fcas_env():
    from AEMOBatteryEnv import AEMOBatteryTradingEnv

    n_steps = 20
    timestamps = [
        datetime(2024, 1, 1) + timedelta(minutes=5 * i) for i in range(n_steps)
    ]
    rng = np.random.default_rng(0)
    data = pl.DataFrame({
        "SETTLEMENTDATE": timestamps,
        "RRP": rng.uniform(0, 500, n_steps),
        "TOTALDEMAND": rng.uniform(3000, 8000, n_steps),
        **{f"FCAS_{svc}": rng.uniform(0, 30, n_steps) for svc in ENV_FCAS_ORDER},
        **{f"FCAS_{svc}_normalized": rng.uniform(0, 1, n_steps) for svc in ENV_FCAS_ORDER},
        "GEN_solar": rng.uniform(0, 0.5, n_steps),
        "GEN_wind": rng.uniform(0, 0.5, n_steps),
    })
    return AEMOBatteryTradingEnv(
        aemo_data=data,
        battery_capacity=10.0,
        max_battery_flow=5.0,
        step_duration=5.0 / 60.0,
        action_mode="full_fcas",
    )


def test_env_fcas_order_matches_canonical():
    env = _make_full_fcas_env()
    assert list(env._fcas_services) == ENV_FCAS_ORDER


def test_oracle_bids_remapped_to_env_order():
    # Oracle grouped raise [6SEC, 60SEC, 5MIN, REG] = [1, 2, 3, 4] MW
    # Oracle grouped lower [6SEC, 60SEC, 5MIN, REG] = [5, 6, 7, 8] MW
    raise_bids = np.array([[1.0, 2.0, 3.0, 4.0]])
    lower_bids = np.array([[5.0, 6.0, 7.0, 8.0]])
    out = oracle_fcas_bids_to_env_order(
        raise_bids, lower_bids, ENV_FCAS_ORDER, max_flow=10.0
    )
    # env order: RAISEREG(4), LOWERREG(8), RAISE6SEC(1), LOWER6SEC(5),
    #            RAISE60SEC(2), LOWER60SEC(6), RAISE5MIN(3), LOWER5MIN(7)
    assert out[0].tolist() == [0.4, 0.8, 0.1, 0.5, 0.2, 0.6, 0.3, 0.7]


def test_oracle_bids_normalised_and_clipped_by_max_flow():
    raise_bids = np.full((1, 4), 5.0)
    lower_bids = np.full((1, 4), 10.0)
    out = oracle_fcas_bids_to_env_order(
        raise_bids, lower_bids, ENV_FCAS_ORDER, max_flow=10.0
    )
    assert np.allclose(out[0], [0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0])
