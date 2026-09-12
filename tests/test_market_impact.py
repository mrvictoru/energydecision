"""Tests for the market-impact model and its cost-to-go integration.

Regression coverage for the dispatch-sign convention: the environment passes
``actual_power`` where positive = charging, and the cost-to-go repricing in
``aemo_sdp_executor.compute_cost_to_go_table`` must use the same sign.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from aemo_data import aggregate_fcas_market_depth
from aemo_sdp_executor import compute_cost_to_go_table
from market_impact import IdentityImpact, PiecewiseMeritOrderImpact


_TS = datetime(2024, 1, 1, 0, 0, 0)


def _supply_curve() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "SETTLEMENTDATE": [_TS, _TS, _TS],
            "MARGINAL_COST": [20.0, 50.0, 100.0],
            "CUMULATIVE_MW": [100.0, 200.0, 300.0],
        }
    )


def test_piecewise_impact_charging_raises_price_discharging_lowers_it():
    impact = PiecewiseMeritOrderImpact(supply_curves=_supply_curve(), impact_intensity=1.0)
    state = {"SETTLEMENTDATE": _TS, "TOTALDEMAND": 200.0}
    base = 50.0
    charging = impact.realized_energy_price(base, 50.0, base, state)
    discharging = impact.realized_energy_price(base, -50.0, base, state)
    assert charging > base > discharging


def test_identity_impact_is_price_taking():
    impact = IdentityImpact()
    state = {"SETTLEMENTDATE": _TS, "TOTALDEMAND": 200.0}
    assert impact.realized_energy_price(50.0, 100.0, 50.0, state) == 50.0
    assert impact.realized_energy_price(50.0, -100.0, 50.0, state) == 50.0


def test_aggregate_fcas_market_depth_schema_and_values():
    demand = pl.DataFrame({"SETTLEMENTDATE": [_TS], "TOTALDEMAND": [8000.0]})
    depth = aggregate_fcas_market_depth(
        "NSW1", datetime(2024, 1, 1), datetime(2024, 1, 2), demand_series=demand
    )
    assert "SETTLEMENTDATE" in depth.columns
    for service, value in {
        "RAISE6SEC": max(50.0, 8000.0 * 0.10),
        "LOWERREG": max(30.0, 8000.0 * 0.03),
    }.items():
        assert f"FCAS_DEPTH_{service}_MW" in depth.columns
        assert depth[f"FCAS_DEPTH_{service}_MW"][0] == value


class _FakeEnv:
    battery_capacity = 10.0
    max_battery_flow = 10.0
    step_duration = 1.0 / 12.0
    max_grid_energy = 1e9
    battery_life_cost = 1_000_000.0
    degradation_temperature = 25.0


class _RecordingImpact:
    """Non-identity impact model that records the dispatch passed to it."""

    def __init__(self) -> None:
        self.dispatch_calls: list[float] = []

    def realized_energy_price(
        self, base_price, battery_dispatch_mw, energy_price, market_state
    ):
        self.dispatch_calls.append(float(battery_dispatch_mw))
        return float(energy_price)


def test_cost_to_go_repricing_uses_env_dispatch_sign():
    env = _FakeEnv()
    forecast = [{"RRP": 50.0, "SETTLEMENTDATE": _TS, "TOTALDEMAND": 200.0}]
    impact = _RecordingImpact()
    compute_cost_to_go_table(
        env,
        forecast,
        soc_resolution=5,
        action_resolution=5,
        deg_cost_per_mwh=0.0,
        impact_model=impact,
    )

    # `dispatch * dt` must equal the action energy (positive = charging). The
    # regression flipped this sign, so charging and dispatch disagreed.
    energies = sorted(round(d * env.step_duration, 6) for d in impact.dispatch_calls)
    expected = sorted(
        round(float(e), 6)
        for e in np.linspace(-1.0, 1.0, 5) * env.max_battery_flow * env.step_duration
    )
    assert energies == expected
    assert max(energies) > 0
