import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from AEMOBatteryEnv import AEMOBatteryTradingEnv


def _make_env(*, init_soc: float, battery_capacity: float = 10.0, max_battery_flow: float = 5.0,
              step_duration: float = 0.083333, raise_price: float = 100.0, lower_price: float = 20.0) -> AEMOBatteryTradingEnv:
    data = pl.DataFrame({
        'RRP': [0.0, 0.0],
        'TOTALDEMAND': [0.0, 0.0],
        'FCAS_RAISEREG': [raise_price, raise_price],
        'FCAS_LOWERREG': [lower_price, lower_price],
    })
    return AEMOBatteryTradingEnv(
        aemo_data=data,
        battery_capacity=battery_capacity,
        max_battery_flow=max_battery_flow,
        init_battery_level=init_soc,
        max_step=1,
        step_duration=step_duration,
        action_mode='full_fcas',
    )


def test_fcas_raise_revenue_uses_power_capability_not_energy_capacity():
    env = _make_env(init_soc=10.0, battery_capacity=10.0, max_battery_flow=5.0, raise_price=100.0, lower_price=20.0)
    env.reset()

    # full_fcas: [energy, RAISEREG, LOWERREG, RAISE6SEC, LOWER6SEC, RAISE60SEC, LOWER60SEC, RAISE5MIN, LOWER5MIN]
    _, _, _, _, info = env.step(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    expected_raise_mw = 5.0
    expected_lower_mw = 0.0
    expected_revenue = expected_raise_mw * 100.0 * 0.083333 + expected_lower_mw * 20.0 * 0.083333
    assert info['fcas_revenue'] == expected_revenue


def test_fcas_raise_enablement_is_limited_by_available_soc_over_step():
    # With 5-min steps, SOC headroom limits enablement only when
    # soc / step_duration < max_battery_flow, i.e. soc < 5.0 * 0.083333 ≈ 0.42.
    env = _make_env(init_soc=0.3, battery_capacity=10.0, max_battery_flow=5.0, raise_price=100.0, lower_price=20.0)
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    # 0.3 MWh available over a 0.0833 h (5-min) interval allows at most 3.6 MW of raise service.
    expected_raise_mw = 0.3 / 0.083333
    expected_revenue = expected_raise_mw * 100.0 * 0.083333
    assert info['fcas_revenue'] == pytest.approx(expected_revenue)


def test_fcas_lower_enablement_is_limited_by_charge_headroom_over_step():
    # With 5-min steps, charge headroom limits enablement only when
    # (capacity - soc) / step_duration < max_battery_flow, i.e. soc > 9.58.
    env = _make_env(init_soc=9.8, battery_capacity=10.0, max_battery_flow=5.0, raise_price=100.0, lower_price=40.0)
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    # 0.2 MWh of spare headroom over a 0.0833 h (5-min) interval allows at most 2.4 MW of lower service.
    expected_lower_mw = 0.2 / 0.083333
    expected_revenue = expected_lower_mw * 40.0 * 0.083333
    assert info['fcas_revenue'] == pytest.approx(expected_revenue)