"""
Unit tests for the SolarBatteryEnv environment.

This module tests the core environment functionality including:
- Reward calculation and energy price selection
- Battery degradation costs
- Grid energy constraints and violations
- Dynamic correction factor behavior
"""

import pytest
import numpy as np
import polars as pl
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from EnergySimEnv import SolarBatteryEnv


class TestSolarBatteryEnv:
    """Test cases for SolarBatteryEnv."""
    
    @pytest.fixture
    def env(self, small_env_df):
        """Create a test environment."""
        env = SolarBatteryEnv(
            small_env_df,
            battery_capacity=4.0,
            max_battery_flow=2.0,
            max_grid_flow=4.0,
            init_battery_level=2.0,
            max_step=2,
            battery_life_cost=1000.0
        )
        env.reset()
        return env

    def test_export_price_selection(self, env):
        """Test that export price is correctly applied when exporting energy."""
        env.reset()
        # Charge battery fully
        env.battery_level = env.battery_capacity
        # Set solar high and load low for export
        env.df = env.df.with_columns([
            pl.lit(3.0).alias("SolarGen"),
            pl.lit(1.0).alias("HouseLoad"),
        ])
        action = np.array([0.0])  # no battery flow, all surplus solar should export
        obs, reward, terminated, truncated, info = env.step(action)

        export_price = float(env.df[0, "ExportEnergyPrice"])
        expected_grid_reward = -(info["grid_energy"] * export_price)
        expected_reward = expected_grid_reward - (info["step_degradation"] * env.battery_life_cost)

        assert info['grid_energy'] < 0, "Should be exporting to grid"
        assert np.isclose(expected_grid_reward, -(info["grid_energy"] * 0.05)), "Should use export price"
        assert np.isclose(reward, expected_reward), "Reward should reflect export price and degradation cost"

    def test_degradation_cost_sensible(self, env):
        """Test that battery degradation cost is positive and reasonable."""
        env.reset()
        action = np.array([-1.0])  # discharge
        obs, reward, terminated, truncated, info = env.step(action)

        deg_cost = info['step_degradation'] * env.battery_life_cost
        assert deg_cost >= 0, "Degradation cost should be non-negative"
        assert deg_cost < env.battery_life_cost * 0.01, "Degradation cost should be much less than battery_life_cost per step"

    def test_reward_is_cost_based(self, env):
        """Test that reward reflects costs (should be negative or zero)."""
        env.reset()
        action = np.array([1.0])  # charge
        obs, reward, terminated, truncated, info = env.step(action)

        import_price = float(env.df[0, "ImportEnergyPrice"])
        grid_reward = -(info["grid_energy"] * import_price)
        expected_reward = grid_reward - (info["step_degradation"] * env.battery_life_cost)

        # Grid reward should typically be negative (cost) or zero
        assert grid_reward <= 0, "Grid reward should be non-positive (represents cost)"
        assert np.isclose(reward, expected_reward), "Reward should combine grid cost and degradation cost"

    def test_battery_capacity_constraints(self, env):
        """Test that battery level stays within capacity constraints."""
        env.reset()
        env.battery_level = 0.5  # Start near empty
        
        # Try to discharge more than available
        action = np.array([-1.0])  # full discharge
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert env.battery_level >= 0, "Battery level should not go negative"
        assert env.battery_level <= env.battery_capacity, "Battery level should not exceed capacity"

    def test_soc_max_clips_overcharge_and_logs_warning(self, small_env_df, caplog):
        """Charging beyond the configured maximum should be clipped."""
        env = SolarBatteryEnv(
            small_env_df,
            battery_capacity=4.0,
            max_battery_flow=4.0,
            init_battery_level=3.0,
            soc_max=0.75,
            max_step=2,
        )
        env.reset()

        with caplog.at_level(logging.WARNING):
            _, _, _, _, info = env.step(np.array([1.0]))

        assert np.isclose(env.battery_level, 3.0)
        assert np.isclose(info["battery_flow_energy"], 0.0)
        assert info["soc_limit_clipped"] is True
        assert np.isclose(
            info["requested_battery_flow_energy"],
            env.battery_capacity - env.init_battery_level,
        )
        assert np.isclose(info["safety_penalty"], env.soc_limit_penalty)
        assert "SOC maximum" in caplog.text

    def test_soc_min_clips_overdischarge_and_logs_warning(self, small_env_df, caplog):
        """Discharging beyond the configured minimum should be clipped."""
        env = SolarBatteryEnv(
            small_env_df,
            battery_capacity=4.0,
            max_battery_flow=4.0,
            init_battery_level=1.0,
            soc_min=0.25,
            max_step=2,
        )
        env.reset()

        with caplog.at_level(logging.WARNING):
            _, _, _, _, info = env.step(np.array([-1.0]))

        assert np.isclose(env.battery_level, 1.0)
        assert np.isclose(info["battery_flow_energy"], 0.0)
        assert info["soc_limit_clipped"] is True
        assert np.isclose(
            info["requested_battery_flow_energy"],
            -env.init_battery_level,
        )
        assert np.isclose(info["safety_penalty"], env.soc_limit_penalty)
        assert "SOC minimum" in caplog.text

    def test_soc_limits_can_be_disabled(self, small_env_df, caplog):
        """Disabling enforcement should preserve the prior physical-limit behavior."""
        env = SolarBatteryEnv(
            small_env_df,
            battery_capacity=4.0,
            max_battery_flow=4.0,
            init_battery_level=3.0,
            soc_max=0.5,
            enforce_soc_limits=False,
            max_step=2,
        )
        env.reset()

        with caplog.at_level(logging.WARNING):
            _, _, _, _, info = env.step(np.array([1.0]))

        assert np.isclose(env.battery_level, 4.0)
        assert np.isclose(info["battery_flow_energy"], 1.0)
        assert info["soc_limit_clipped"] is False
        assert np.isclose(info["requested_battery_flow_energy"], 1.0)
        assert np.isclose(info["safety_penalty"], 0.0)
        assert "SOC maximum" not in caplog.text

    def test_soc_limit_penalty_reduces_reward(self, small_env_df):
        """Clipped actions should receive the configured learning penalty."""
        common_kwargs = {
            "battery_capacity": 4.0,
            "max_battery_flow": 4.0,
            "init_battery_level": 3.0,
            "soc_max": 0.75,
            "max_step": 2,
        }
        unpenalized_env = SolarBatteryEnv(
            small_env_df, soc_limit_penalty=0.0, **common_kwargs
        )
        penalized_env = SolarBatteryEnv(
            small_env_df, soc_limit_penalty=2.0, **common_kwargs
        )
        unpenalized_env.reset()
        penalized_env.reset()

        _, unpenalized_reward, _, _, _ = unpenalized_env.step(np.array([1.0]))
        _, penalized_reward, _, _, _ = penalized_env.step(np.array([1.0]))

        assert np.isclose(unpenalized_reward - penalized_reward, 2.0)

    def test_observation_shape(self, env):
        """Test that observations have the expected shape."""
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        
        assert isinstance(obs, np.ndarray), "Observation should be numpy array"
        assert obs.shape == env.observation_space.shape, "Observation shape should match observation space"

    def test_episode_termination(self, env):
        """Test that episode terminates when max_step is reached."""
        env.reset()
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            step_count += 1
            if step_count > env.max_step + 1:
                break
        
        assert terminated or truncated, "Episode should terminate at max_step"


class TestSolarBatteryEnvPerformance:
    """Performance-related tests for SolarBatteryEnv."""
    
    @pytest.fixture
    def large_env(self, sample_energy_df):
        """Create an environment with more data for performance testing."""
        return SolarBatteryEnv(
            sample_energy_df,
            battery_capacity=7.0,
            max_battery_flow=3.3,
            init_battery_level=3.5,
            max_step=50
        )
    
    def test_observation_computation_vectorized(self, large_env):
        """Test that observation computation uses vectorized operations."""
        large_env.reset()
        
        # The presence of _norm_by_capacity_mask indicates vectorized normalization
        assert hasattr(large_env, '_norm_by_capacity_mask'), \
            "Environment should have pre-computed normalization mask"
        assert isinstance(large_env._norm_by_capacity_mask, np.ndarray), \
            "Normalization mask should be numpy array"
    
    def test_min_max_precomputed(self, large_env):
        """Test that min/max values are precomputed for efficiency."""
        assert hasattr(large_env, 'df_mins_for_obs'), "Min values should be precomputed"
        assert hasattr(large_env, 'df_maxs_for_obs'), "Max values should be precomputed"
        assert isinstance(large_env.df_mins_for_obs, np.ndarray), "Min values should be numpy array"
        assert isinstance(large_env.df_maxs_for_obs, np.ndarray), "Max values should be numpy array"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
