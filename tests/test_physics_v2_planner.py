"""P2 tests: state-dependent SDP degradation (A5) and wear-aware teacher (A6)."""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sdp_algorithm import SDPSolver
from household_optimization import optimize_dispatch
from household_replay import Tariff


class _FakeEnv:
    battery_capacity = 10.0
    max_battery_flow = 5.0
    step_duration = 1.0 / 12.0
    max_grid_energy = 1e9
    battery_life_cost = 50_000.0
    degradation_temperature = 25.0


def test_sdp_deg_grid_is_state_dependent():
    solver = SDPSolver(_FakeEnv(), horizon=4, soc_resolution=6, action_resolution=5)
    grid = solver._deg_cost_grid
    assert grid.shape == (6, 5)
    assert np.isfinite(grid).all() and (grid >= 0).all()
    # Charging is clipped to zero at full SoC; discharging at empty SoC.
    assert grid[-1, -1] == 0.0
    assert grid[0, 0] == 0.0
    # A meaningful DoD transition costs something...
    assert grid[0, -1] > 0.0
    # ...and the cost varies with the starting SoC (not a fixed midpoint).
    assert len(set(np.round(grid[:, -1], 12))) > 1


def _arbitrage_frame(n=96):
    t0 = datetime(2024, 1, 1)
    ts = [t0 + timedelta(minutes=5 * i) for i in range(n)]
    solar = [3.0 if i < n // 2 else 0.0 for i in range(n)]
    load = [0.5 if i < n // 2 else 3.0 for i in range(n)]
    return pl.DataFrame({"Timestamp": ts, "HouseLoad": load, "SolarGen": solar})


def test_teacher_lambda_deg_reduces_throughput():
    frame = _arbitrage_frame()
    tariff = Tariff(import_cents_per_kwh=31.042, feed_in_cents_per_kwh=1.0,
                    free_window_start_hour=24, free_window_end_hour=24)
    common = dict(tariff=tariff, capacity_kwh=5.0, max_flow_kw=3.3,
                  roundtrip_eff=0.80, initial_soc=0.5)
    r0 = optimize_dispatch(frame, deg_cost_per_mwh=0.0, **common)
    r_hi = optimize_dispatch(frame, deg_cost_per_mwh=1000.0, **common)
    tp0 = float(np.abs(r0.actions_kw).sum())
    tp_hi = float(np.abs(r_hi.actions_kw).sum())
    assert tp0 > 0.0
    assert tp_hi < tp0, f"wear-aware plan should trade less ({tp_hi} vs {tp0})"


def test_teacher_default_is_degradation_blind():
    frame = _arbitrage_frame()
    tariff = Tariff(import_cents_per_kwh=31.042, feed_in_cents_per_kwh=1.0,
                    free_window_start_hour=24, free_window_end_hour=24)
    common = dict(tariff=tariff, capacity_kwh=5.0, max_flow_kw=3.3,
                  roundtrip_eff=0.80, initial_soc=0.5)
    default = optimize_dispatch(frame, **common)
    explicit = optimize_dispatch(frame, deg_cost_per_mwh=0.0, **common)
    assert np.allclose(default.actions_kw, explicit.actions_kw)


def test_planner_degradation_calibration_scales_grid():
    s1 = SDPSolver(_FakeEnv(), horizon=4, soc_resolution=6, action_resolution=5,
                   degradation_calibration=1.0)
    s2 = SDPSolver(_FakeEnv(), horizon=4, soc_resolution=6, action_resolution=5,
                   degradation_calibration=0.12)
    assert np.allclose(s2._deg_cost_grid, s1._deg_cost_grid * 0.12)


def test_household_step_wear_reduces_throughput_and_calibration_monotone():
    frame = _arbitrage_frame()
    tariff = Tariff(import_cents_per_kwh=31.042, feed_in_cents_per_kwh=1.0,
                    free_window_start_hour=24, free_window_end_hour=24)
    common = dict(tariff=tariff, capacity_kwh=5.0, max_flow_kw=3.3,
                  roundtrip_eff=0.80, initial_soc=0.5)
    none = optimize_dispatch(frame, deg_mode="none", **common)
    low = optimize_dispatch(frame, deg_mode="step", deg_calibration=0.1, **common)
    high = optimize_dispatch(frame, deg_mode="step", deg_calibration=1.0, **common)
    tp = [float(np.abs(r.actions_kw).sum()) for r in (none, low, high)]
    assert tp[0] > 0.0
    # More wear -> less throughput; both step plans trade less than the blind plan.
    assert tp[1] <= tp[0] and tp[2] <= tp[1]
