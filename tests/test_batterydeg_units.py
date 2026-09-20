"""Unit tests for rainflow C-rate units and the reset cap (known_issues A3/A4)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from batterydeg import RainflowCounter


def _closed_cycles(step_duration, max_c_rate, profile):
    counter = RainflowCounter(step_duration=step_duration, max_c_rate=max_c_rate)
    cycles = []
    for soc in profile:
        cycles += counter.update(float(soc))
    return cycles


def test_rainflow_current_is_c_rate_not_percent_per_hour():
    # 1-hour step, 10 percentage-point move => 0.10 C-rate (not 10 %/h)
    cycles = _closed_cycles(1.0, 1e9, [50, 60, 50, 60, 50, 60])
    assert cycles, "expected at least one closed cycle"
    soc_avg, dod, Id, Ich = cycles[0]
    assert abs(dod - 10.0) < 1e-9
    assert abs(Ich - 0.10) < 1e-9, f"Ich should be 0.10 C, got {Ich}"
    assert Id == 0.0


def test_rainflow_clamps_to_max_c_rate():
    # 20pp over a 0.5 h step => 0.4 C true, clamped to 0.1 C
    cycles = _closed_cycles(0.5, 0.1, [50, 70, 50, 70, 50, 70])
    assert cycles, "expected at least one closed cycle"
    _, _, Id, Ich = cycles[0]
    assert abs(Ich - 0.1) < 1e-9, f"Ich should clamp to 0.1 C, got {Ich}"


def test_solar_env_reset_preserves_max_c_rate(small_env_df):
    from EnergySimEnv import SolarBatteryEnv

    env = SolarBatteryEnv(
        small_env_df,
        battery_capacity=4.0,
        max_battery_flow=2.0,
        max_grid_flow=4.0,
        init_battery_level=2.0,
        max_step=4,
        battery_life_cost=1000.0,
    )
    expected = env.max_battery_flow / env.initial_battery_capacity
    assert env._rainflow_counter.max_c_rate == expected
    env.reset()
    assert env._rainflow_counter.max_c_rate == expected
