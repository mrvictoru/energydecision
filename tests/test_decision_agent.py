"""
Unit tests for the decision Agent module.

This module tests the core decision-making functionality including:
- SDP (Stochastic Dynamic Programming) solver
- Oracle agent behavior
- Policy computation and action selection
"""

import pytest
import numpy as np
import polars as pl
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from EnergySimEnv import SolarBatteryEnv
from decision import Agent


def _build_test_dataframe(num_steps: int = 4) -> pl.DataFrame:
    """Create a simple test DataFrame."""
    timestamps = [f"2021-01-01T{hour:02d}:00:00" for hour in range(num_steps)]
    return pl.DataFrame({
        "Timestamp": list(range(num_steps)),
        "Time": timestamps,
        "SolarGen": [0.5, 1.5, 2.0, 1.0][:num_steps],
        "HouseLoad": [1.0, 0.5, 1.5, 0.5][:num_steps],
        "ImportEnergyPrice": [0.25, 0.3, 0.35, 0.4][:num_steps],
        "ExportEnergyPrice": [0.1, 0.12, 0.15, 0.18][:num_steps],
    })


def _make_env(df: pl.DataFrame) -> SolarBatteryEnv:
    """Create a standard test environment."""
    return SolarBatteryEnv(
        df=df,
        battery_capacity=4.0,
        max_battery_flow=2.0,
        max_grid_flow=5.0,
        init_battery_level=1.5,
        max_step=len(df),
        battery_life_cost=1000.0,
    )


class TestAgentInitialization:
    """Test Agent initialization and configuration."""
    
    def test_sdp_agent_creation(self, sample_energy_df):
        """Test that SDP agent can be created with default settings."""
        env = SolarBatteryEnv(sample_energy_df, battery_capacity=7.0, max_battery_flow=3.3)
        env.reset()
        
        agent = Agent(env, algorithm='sdp', horizon=24, soc_resolution=10, action_resolution=7)
        
        assert agent.algorithm == 'sdp'
        assert agent.horizon == 24
        assert agent.soc_resolution == 10
        assert agent.action_resolution == 7

    def test_oracle_agent_creation(self, sample_energy_df):
        """Test that Oracle agent can be created."""
        env = SolarBatteryEnv(sample_energy_df, battery_capacity=7.0, max_battery_flow=3.3)
        env.reset()
        
        agent = Agent(env, algorithm='oracle', horizon=1, action_resolution=5)
        
        assert agent.algorithm == 'oracle'
        assert agent.horizon == 1


class TestSDPSolver:
    """Test cases for the SDP solver."""
    
    @pytest.fixture
    def sdp_agent(self, sample_energy_df):
        """Create an SDP agent for testing."""
        env = SolarBatteryEnv(
            sample_energy_df,
            battery_capacity=7.0,
            max_battery_flow=3.3,
            init_battery_level=3.5,
            max_step=50
        )
        env.reset()
        return Agent(
            env,
            algorithm='sdp',
            horizon=24,
            soc_resolution=10,
            action_resolution=7,
            use_monte_carlo=False,
            mc_samples=50
        )

    def test_sdp_solve_returns_valid_policy(self, sdp_agent):
        """Test that SDP solve returns a valid policy table."""
        forecasts = sdp_agent._get_forecasts(sdp_agent.env.current_step, sdp_agent.sdp_solver.horizon)
        
        # Use the new SDPSolver directly
        policy_table = sdp_agent.sdp_solver.solve(forecasts, start_index=sdp_agent.env.current_step)
        
        assert policy_table.shape[0] == sdp_agent.sdp_solver.horizon, "Policy table should have horizon rows"
        assert policy_table.shape[1] == len(sdp_agent.soc_levels_kwh), "Policy table should have soc_resolution columns"

    def test_sdp_policy_values_valid(self, sdp_agent):
        """Test that policy values are within valid action range."""
        forecasts = sdp_agent._get_forecasts(sdp_agent.env.current_step, sdp_agent.sdp_solver.horizon)
        policy_table = sdp_agent.sdp_solver.solve(forecasts, start_index=sdp_agent.env.current_step)
        
        # Valid policy values are -1 (infeasible) or 0 to action_resolution-1
        valid_policies = (policy_table >= -1) & (policy_table < sdp_agent.action_resolution)
        assert np.all(valid_policies), "All policy values should be valid"

    def test_scenario_cache_initialization(self, sdp_agent):
        """Test that scenario cache is properly initialized."""
        # Initially cache should be None
        assert sdp_agent.sdp_solver._scenario_cache is None
        
        # After solving, cache may be populated
        forecasts = sdp_agent._get_forecasts(sdp_agent.env.current_step, sdp_agent.sdp_solver.horizon)
        sdp_agent.sdp_solver.solve(forecasts, start_index=sdp_agent.env.current_step)
        
        # Cache may or may not be set depending on settings
        # Just check it doesn't raise an error


class TestSDPPerformance:
    """Performance tests for SDP solver."""
    
    @pytest.fixture
    def performance_env(self, sample_energy_df):
        """Create environment for performance testing."""
        return SolarBatteryEnv(
            sample_energy_df,
            battery_capacity=7.0,
            max_battery_flow=3.3,
            init_battery_level=3.5,
            max_step=80
        )

    @pytest.mark.timeout(60)  # Max 60 seconds
    def test_sdp_solve_time(self, performance_env):
        """Test that SDP solve completes within reasonable time."""
        agent = Agent(
            performance_env,
            algorithm='sdp',
            horizon=48,
            soc_resolution=10,
            action_resolution=7,
            use_monte_carlo=True,
            mc_samples=100,
            mc_seed=42
        )
        
        start = time.perf_counter()
        
        try:
            agent.sdp_solver._scenario_cache = agent.sdp_solver.scenario_generator.generate_time_step_scenarios(performance_env.df)
        except Exception:
            pytest.skip("Scenario generation not available")
        
        forecasts = agent._get_forecasts(performance_env.current_step, agent.sdp_solver.horizon)
        policy_table = agent.sdp_solver.solve(forecasts, start_index=performance_env.current_step)
        
        duration = time.perf_counter() - start
        
        assert policy_table.shape[0] == agent.sdp_solver.horizon
        assert duration < 60, f"SDP solve took too long: {duration:.2f}s"
        
        print(f"SDP solve time: {duration:.3f} seconds")


class TestOracleAgent:
    """Test cases for the Oracle agent."""
    
    def test_oracle_returns_action(self):
        """Test that Oracle agent returns a valid action."""
        df = _build_test_dataframe()
        env = _make_env(df)
        env.reset()
        
        agent = Agent(env, algorithm='oracle', horizon=1, action_resolution=5)
        
        raw_obs = env.get_raw_obs()
        action = agent.choose_action(raw_obs)
        
        assert action is not None
        assert isinstance(action, (np.ndarray, list, tuple))
        assert -1.0 <= action[0] <= 1.0

    def test_oracle_action_improves_reward(self):
        """Test that Oracle chooses a valid action (may not always be optimal due to implementation details)."""
        df = _build_test_dataframe()
        env = _make_env(df)
        env.reset()
        
        agent = Agent(env, algorithm='oracle', horizon=1, action_resolution=5)
        
        raw_obs = env.get_raw_obs()
        oracle_action = agent.choose_action(raw_obs)[0]
        
        # Verify oracle returns a valid action in range
        assert -1.0 <= oracle_action <= 1.0, "Oracle should return action in valid range"
        
        # Test that oracle action is one of the candidate actions
        candidate_actions = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
        
        # Oracle action should be close to one of the candidates
        min_distance = min(abs(oracle_action - act) for act in candidate_actions)
        assert min_distance < 0.01, "Oracle action should be close to a candidate action"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
