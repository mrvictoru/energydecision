import datetime as dt

import numpy as np
import polars as pl

from household_optimization import bill_for_actions, bootstrap_mean_ci, optimize_dispatch
from household_replay import Tariff


def _frame():
    start = dt.datetime(2025, 6, 1)
    timestamps = [start + dt.timedelta(minutes=5 * index) for index in range(288)]
    solar = np.zeros(288)
    solar[132:168] = 4.0  # free-window solar surplus
    return pl.DataFrame({
        "Timestamp": timestamps,
        "HouseLoad": np.full(288, 1.0),
        "SolarGen": solar,
    })


def test_optimizer_reduces_bill_and_respects_hardware_limits():
    result = optimize_dispatch(
        _frame(),
        tariff=Tariff(),
        capacity_kwh=5.0,
        max_flow_kw=3.3,
        roundtrip_eff=0.8,
    )
    no_battery = bill_for_actions(_frame(), np.zeros(288), Tariff())
    assert result.bill_aud < no_battery
    assert np.abs(result.actions_kw).max() <= 3.3 + 1e-9
    assert result.soc_kwh.min() >= -1e-9
    assert result.soc_kwh.max() <= 5.0 + 1e-9


def test_bootstrap_ci_is_reproducible():
    assert bootstrap_mean_ci([1.0, 2.0, 3.0], seed=4) == bootstrap_mean_ci([1.0, 2.0, 3.0], seed=4)
