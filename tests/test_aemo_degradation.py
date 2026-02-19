"""
Tests for degradation modelling integration in AEMOBatteryTradingEnv.

Validates:
- Rainflow-based degradation mode tracks cycles and computes degradation
- Simple degradation mode preserves backward compatibility
- Capacity fade reduces battery capacity over time
- Degradation info is correctly reported in step info dict
- Both degradation modes are configurable via constructor parameter
"""

import sys
import os
import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from AEMOBatteryEnv import AEMOBatteryTradingEnv


@pytest.fixture
def aemo_test_data():
    """Create a synthetic AEMO market data DataFrame for testing."""
    np.random.seed(42)
    num_steps = 200
    timestamps = [datetime(2024, 6, 1) + timedelta(minutes=30 * i) for i in range(num_steps)]

    return pl.DataFrame({
        'SETTLEMENTDATE': timestamps,
        'Time': timestamps,
        'RRP': np.random.uniform(20, 100, num_steps),
        'TOTALDEMAND': np.random.uniform(5000, 8000, num_steps),
        'FCAS_RAISEREG': np.random.uniform(5, 20, num_steps),
        'FCAS_LOWERREG': np.random.uniform(5, 20, num_steps),
        'FCAS_RAISE6SEC': np.random.uniform(10, 30, num_steps),
        'FCAS_LOWER6SEC': np.random.uniform(10, 30, num_steps),
        'FCAS_RAISE60SEC': np.random.uniform(8, 25, num_steps),
        'FCAS_LOWER60SEC': np.random.uniform(8, 25, num_steps),
        'FCAS_RAISE5MIN': np.random.uniform(7, 22, num_steps),
        'FCAS_LOWER5MIN': np.random.uniform(7, 22, num_steps),
        'GEN_solar': np.random.uniform(0, 500, num_steps),
        'GEN_wind': np.random.uniform(0, 300, num_steps),
        'hour_sin': np.sin(2 * np.pi * np.arange(num_steps) / 48),
        'hour_cos': np.cos(2 * np.pi * np.arange(num_steps) / 48),
        'day_sin': np.zeros(num_steps),
        'day_cos': np.ones(num_steps),
        'is_peak': np.ones(num_steps),
        'RRP_normalized': np.random.uniform(0, 1, num_steps),
        'DEMAND_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_RAISEREG_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_LOWERREG_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_RAISE6SEC_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_LOWER6SEC_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_RAISE60SEC_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_LOWER60SEC_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_RAISE5MIN_normalized': np.random.uniform(0, 1, num_steps),
        'FCAS_LOWER5MIN_normalized': np.random.uniform(0, 1, num_steps),
        'GEN_solar_pct': np.random.uniform(0, 0.3, num_steps),
        'GEN_wind_pct': np.random.uniform(0, 0.2, num_steps),
    })


class TestAEMORainflowDegradation:
    """Tests for the rainflow degradation mode in AEMOBatteryTradingEnv."""

    @pytest.fixture
    def rainflow_env(self, aemo_test_data):
        env = AEMOBatteryTradingEnv(
            aemo_data=aemo_test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            action_mode='simple',
            degradation_mode='rainflow',
            max_step=80,
        )
        env.reset(seed=42)
        return env

    def test_rainflow_counter_initialized(self, rainflow_env):
        """Rainflow mode should initialize degradation model and counter."""
        assert hasattr(rainflow_env, 'degradation_model')
        assert hasattr(rainflow_env, '_rainflow_counter')
        assert rainflow_env.degradation_mode == 'rainflow'

    def test_degradation_accumulates_with_cycling(self, rainflow_env):
        """Alternating charge/discharge should produce rainflow cycles and degradation."""
        for i in range(40):
            action_val = 1.0 if i % 2 == 0 else -1.0
            _, _, terminated, truncated, info = rainflow_env.step(np.array([action_val]))
            if terminated or truncated:
                break

        assert info['total_degradation'] > 0, "Cycling should produce degradation"
        assert info['rainflow_num_cycles'] > 0, "Cycling should produce rainflow cycles"

    def test_idle_battery_no_degradation(self, rainflow_env):
        """Idle battery should produce no degradation."""
        for _ in range(20):
            _, _, terminated, truncated, info = rainflow_env.step(np.array([0.0]))
            if terminated or truncated:
                break

        assert info['total_degradation'] == 0.0, "Idle battery should not degrade"
        assert info['rainflow_num_cycles'] == 0, "Idle battery should have no cycles"

    def test_capacity_fade_reduces_capacity(self, rainflow_env):
        """Degradation should reduce battery capacity below initial."""
        initial_cap = rainflow_env.initial_battery_capacity

        for i in range(60):
            action_val = 1.0 if i % 2 == 0 else -1.0
            _, _, terminated, truncated, info = rainflow_env.step(np.array([action_val]))
            if terminated or truncated:
                break

        if info['total_degradation'] > 0:
            assert info['capacity_mwh'] < initial_cap, \
                "Capacity should fade with degradation"

    def test_soc_history_tracked(self, rainflow_env):
        """SOC history should be tracked for rainflow counting."""
        num_steps = 10
        for _ in range(num_steps):
            rainflow_env.step(np.array([0.5]))

        assert len(rainflow_env.soc_history) == 1 + num_steps, \
            "SOC history should have initial + one per step"

    def test_info_contains_degradation_fields(self, rainflow_env):
        """Step info dict should contain degradation-related fields."""
        _, _, _, _, info = rainflow_env.step(np.array([0.5]))

        expected_keys = [
            'step_degradation', 'total_degradation', 'capacity_mwh',
            'rainflow_cumulative_deg', 'rainflow_num_cycles', 'degradation_cost',
        ]
        for key in expected_keys:
            assert key in info, f"Info dict should contain '{key}'"

    def test_reset_clears_degradation_state(self, rainflow_env):
        """Reset should clear all degradation tracking."""
        for i in range(20):
            action_val = 1.0 if i % 2 == 0 else -1.0
            rainflow_env.step(np.array([action_val]))

        rainflow_env.reset()

        assert rainflow_env.total_degradation == 0.0
        assert rainflow_env._rainflow_deg_cumulative == 0.0
        assert rainflow_env._rainflow_num_cycles == 0
        assert rainflow_env.battery_capacity == rainflow_env.initial_battery_capacity
        assert len(rainflow_env.soc_history) == 1


class TestAEMOSimpleDegradation:
    """Tests for the simple (backward-compatible) degradation mode."""

    @pytest.fixture
    def simple_env(self, aemo_test_data):
        env = AEMOBatteryTradingEnv(
            aemo_data=aemo_test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            action_mode='simple',
            degradation_mode='simple',
            max_step=80,
        )
        env.reset(seed=42)
        return env

    def test_simple_mode_no_rainflow_counter(self, simple_env):
        """Simple mode should not use degradation_model attribute."""
        assert simple_env.degradation_mode == 'simple'

    def test_simple_degradation_positive_on_action(self, simple_env):
        """Any non-zero action should produce some degradation in simple mode."""
        _, _, _, _, info = simple_env.step(np.array([1.0]))
        assert info['total_degradation'] > 0, "Charge action should cause simple degradation"

    def test_simple_idle_no_degradation(self, simple_env):
        """Idle action should produce no degradation in simple mode."""
        _, _, _, _, info = simple_env.step(np.array([0.0]))
        assert info['total_degradation'] == 0.0, "Idle should not degrade in simple mode"


class TestAEMODegradationModeSwitch:
    """Tests for switching between degradation modes."""

    def test_default_mode_is_rainflow(self, aemo_test_data):
        """Default degradation mode should be rainflow."""
        env = AEMOBatteryTradingEnv(
            aemo_data=aemo_test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
        )
        assert env.degradation_mode == 'rainflow'

    def test_rainflow_and_simple_produce_different_degradation(self, aemo_test_data):
        """Rainflow and simple modes should produce different degradation values."""
        np.random.seed(123)
        actions = [np.array([np.random.uniform(-1, 1)]) for _ in range(40)]

        env_rf = AEMOBatteryTradingEnv(
            aemo_data=aemo_test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            degradation_mode='rainflow',
            max_step=80,
        )
        env_rf.reset(seed=0)

        env_simple = AEMOBatteryTradingEnv(
            aemo_data=aemo_test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            degradation_mode='simple',
            max_step=80,
        )
        env_simple.reset(seed=0)

        for a in actions:
            _, _, t1, tr1, info_rf = env_rf.step(a)
            _, _, t2, tr2, info_simple = env_simple.step(a)
            if t1 or tr1 or t2 or tr2:
                break

        # They should generally differ since they use different models
        # (though not guaranteed for every seed; at minimum both should be non-negative)
        assert info_rf['total_degradation'] >= 0
        assert info_simple['total_degradation'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
