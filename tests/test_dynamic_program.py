"""
Unit tests for DynamicProgram and related components.

These tests verify:
- Basic interpolation utilities work correctly
- DynamicProgram API compatibility and correctness
- Small deterministic problems with known analytic solutions
- Integration with Agent adapter interface

The tests are focused on correctness rather than performance, using small
problem sizes to ensure fast execution.
"""

import sys
import os
import numpy as np
import pytest

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamic_program import DynamicProgram, StateGrid, interp_vectorized
from dp_adapter import run_dp_for_agent, AgentAdapter


class TestInterpolation:
    """Test the vectorized interpolation utility."""
    
    def test_interp_vectorized_simple(self):
        """Test basic linear interpolation."""
        x_grid = np.array([0.0, 1.0, 2.0, 3.0])
        y_values = np.array([0.0, 10.0, 20.0, 30.0])
        
        # Test interpolation at grid points
        query = np.array([0.0, 1.0, 2.0, 3.0])
        result = interp_vectorized(x_grid, y_values, query)
        np.testing.assert_array_almost_equal(result, y_values)
        
        # Test interpolation between grid points
        query = np.array([0.5, 1.5, 2.5])
        expected = np.array([5.0, 15.0, 25.0])
        result = interp_vectorized(x_grid, y_values, query)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_interp_vectorized_clamping(self):
        """Test that interpolation clamps to grid bounds."""
        x_grid = np.array([1.0, 2.0, 3.0])
        y_values = np.array([10.0, 20.0, 30.0])
        
        # Test extrapolation (should clamp)
        query = np.array([0.0, 0.5, 3.5, 4.0])
        result = interp_vectorized(x_grid, y_values, query)
        
        # Should clamp to boundary values
        assert result[0] == 10.0  # Clamp to first value
        assert result[1] == 10.0  # Clamp to first value
        assert result[2] == 30.0  # Clamp to last value
        assert result[3] == 30.0  # Clamp to last value
    
    def test_interp_vectorized_2d(self):
        """Test interpolation with 2D query arrays."""
        x_grid = np.array([0.0, 1.0, 2.0])
        y_values = np.array([0.0, 10.0, 20.0])
        
        query = np.array([[0.0, 0.5], [1.0, 1.5]])
        result = interp_vectorized(x_grid, y_values, query)
        
        expected = np.array([[0.0, 5.0], [10.0, 15.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestStateGrid:
    """Test the StateGrid helper class."""
    
    def test_state_grid_init(self):
        """Test StateGrid initialization."""
        levels = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        grid = StateGrid(levels)
        
        assert grid.size == 5
        assert grid.min_val == 0.0
        assert grid.max_val == 10.0
        np.testing.assert_array_equal(grid.levels, levels)
    
    def test_to_index(self):
        """Test mapping continuous values to grid indices."""
        levels = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        grid = StateGrid(levels)
        
        # Test exact matches
        assert grid.to_index(0.0) == 0
        assert grid.to_index(5.0) == 2
        assert grid.to_index(10.0) == 4
        
        # Test nearest neighbor selection
        assert grid.to_index(1.0) == 0    # Closer to 0.0
        assert grid.to_index(3.0) == 1    # Closer to 2.5
        assert grid.to_index(6.0) == 2    # Closer to 5.0
    
    def test_interpolate(self):
        """Test StateGrid interpolation method."""
        levels = np.array([0.0, 1.0, 2.0])
        grid = StateGrid(levels)
        values = np.array([0.0, 10.0, 20.0])
        
        query = np.array([0.5, 1.5])
        result = grid.interpolate(values, query)
        expected = np.array([5.0, 15.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestDynamicProgram:
    """Test the DynamicProgram class with simple deterministic problems."""
    
    def test_dp_initialization(self):
        """Test DynamicProgram initialization."""
        state_grid = np.array([0.0, 5.0, 10.0])
        action_levels = np.array([-1.0, 0.0, 1.0])
        
        def dummy_transition(state, action, scenario):
            return state + action
        
        def dummy_cost(state, action, scenario):
            return abs(action)
        
        def dummy_scenarios(stage_index):
            return {'dummy': np.array([0.0])}, np.array([1.0])
        
        stage_times = [0, 1]
        
        dp = DynamicProgram(
            state_grid, action_levels, dummy_transition, 
            dummy_cost, dummy_scenarios, stage_times
        )
        
        assert dp.horizon == 2
        assert dp.num_states == 3
        assert dp.num_actions == 3
        assert len(dp.stages) == 2
    
    def test_dp_simple_two_stage(self):
        """Test a simple two-stage problem with known optimal solution."""
        # Simple problem: minimize action cost over 2 stages
        # State: inventory level [0, 1, 2]
        # Action: change in inventory [-1, 0, 1]
        # Cost: action^2 (prefer no action)
        # Transition: new_state = clip(old_state + action, 0, 2)
        
        state_grid = np.array([0.0, 1.0, 2.0])
        action_levels = np.array([-1.0, 0.0, 1.0])
        
        def transition(state, action, scenario):
            return np.clip(state + action, 0.0, 2.0)
        
        def cost(state, action, scenario):
            return action ** 2  # Quadratic cost favors zero action
        
        def scenarios(stage_index):
            # Single deterministic scenario
            return {'dummy': np.array([0.0])}, np.array([1.0])
        
        stage_times = [0, 1]
        
        dp = DynamicProgram(
            state_grid, action_levels, transition, cost, scenarios, stage_times
        )
        dp.initialize_states()
        dp.set_final_ctg(np.zeros(3))  # Zero terminal cost
        
        ctg_matrix, policy_table = dp.solve()
        
        # Check dimensions
        assert ctg_matrix.shape == (3, 3)  # (horizon+1, num_states)
        assert policy_table.shape == (2, 3)  # (horizon, num_states)
        
        # For this simple problem, optimal action should be 0 (no change)
        # since action cost is quadratic and there's no other incentive
        expected_action_idx = 1  # Index of action 0.0
        
        # At least some states should have optimal action = 0
        assert np.any(policy_table == expected_action_idx)
        
        # Final CTG should be zeros (as set)
        np.testing.assert_array_equal(ctg_matrix[2, :], [0.0, 0.0, 0.0])
        
        # All policies should be valid (>= 0)
        assert np.all(policy_table >= -1)  # -1 is allowed for infeasible
    
    def test_dp_infeasible_actions(self):
        """Test handling of infeasible actions."""
        state_grid = np.array([0.0, 1.0])
        action_levels = np.array([-2.0, 0.0, 2.0])  # Large actions
        
        def transition(state, action, scenario):
            next_state = state + action
            # Return inf for infeasible transitions
            if next_state < 0 or next_state > 1:
                return np.inf
            return next_state
        
        def cost(state, action, scenario):
            next_state = transition(state, action, scenario)
            if np.isinf(next_state):
                return np.inf  # Infeasible
            return 0.0  # Zero cost if feasible
        
        def scenarios(stage_index):
            return {'dummy': np.array([0.0])}, np.array([1.0])
        
        dp = DynamicProgram(
            state_grid, action_levels, transition, cost, scenarios, [0]
        )
        dp.initialize_states()
        dp.set_final_ctg(np.zeros(2))
        
        ctg_matrix, policy_table = dp.solve()
        
        # Check that infeasible actions are marked as -1
        # From state 0, action -2 should be infeasible
        # From state 1, action +2 should be infeasible
        # But action 0 should be feasible from both states
        
        expected_feasible_action = 1  # Index of action 0.0
        assert policy_table[0, 0] == expected_feasible_action  # State 0
        assert policy_table[0, 1] == expected_feasible_action  # State 1


class MockAgent:
    """Mock Agent class for testing the adapter."""
    
    def __init__(self):
        self.soc_levels_kwh = np.array([0.0, 5.0, 10.0])
        self.action_levels_norm = np.array([-1.0, 0.0, 1.0])
        self.battery_capacity = 10.0
        self.max_battery_flow = 5.0
        self.step_duration = 0.5
        self._scenario_cache = None
    
    def _calculate_sdp_stage_cost(self, row_idx, soc_kwh, battery_flow_rate, 
                                 battery_flow_energy, forecast_step):
        """Mock cost calculation - simple quadratic cost."""
        return battery_flow_rate ** 2


class TestAgentAdapter:
    """Test the AgentAdapter for integration with existing Agent code."""
    
    def test_adapter_initialization(self):
        """Test adapter initialization with mock agent."""
        agent = MockAgent()
        forecasts = [
            {'SolarGen': 2.0, 'HouseLoad': 3.0, 'ImportEnergyPrice': 0.2, 'ExportEnergyPrice': 0.1},
            {'SolarGen': 1.5, 'HouseLoad': 2.5, 'ImportEnergyPrice': 0.25, 'ExportEnergyPrice': 0.12}
        ]
        
        adapter = AgentAdapter(agent, forecasts, start_index=0)
        
        np.testing.assert_array_equal(adapter.soc_levels_kwh, agent.soc_levels_kwh)
        np.testing.assert_array_equal(adapter.action_levels_norm, agent.action_levels_norm)
    
    def test_adapter_transition_function(self):
        """Test adapter's transition function."""
        agent = MockAgent()
        adapter = AgentAdapter(agent, [], 0)
        
        transition_fn = adapter.create_transition_fn()
        
        # Test transition: state=5.0, action=0.5 (charge)
        # battery_energy = 0.5 * 5.0 * 0.5 = 1.25 kWh
        # next_state = 5.0 + 1.25 = 6.25 kWh
        result = transition_fn(5.0, 0.5, {})
        assert abs(result - 6.25) < 1e-6
        
        # Test boundary clamping
        result = transition_fn(9.0, 1.0, {})  # Should clamp to 10.0
        assert abs(result - 10.0) < 1e-6
        
        result = transition_fn(1.0, -1.0, {})  # Should clamp to 0.0
        assert abs(result - 0.0) < 1e-6
    
    def test_adapter_cost_function(self):
        """Test adapter's cost function."""
        agent = MockAgent()
        adapter = AgentAdapter(agent, [], 0)
        
        cost_fn = adapter.create_cost_fn()
        
        scenario_values = {
            'solar': 2.0,
            'load': 3.0, 
            'import_price': 0.2,
            'export_price': 0.1
        }
        
        # Test cost calculation (should use agent's mock quadratic cost)
        result = cost_fn(5.0, 0.5, scenario_values)
        # battery_flow_rate = 0.5 * 5.0 = 2.5
        # cost = 2.5^2 = 6.25
        assert abs(result - 6.25) < 1e-6
    
    def test_adapter_scenario_provider(self):
        """Test adapter's scenario provider."""
        agent = MockAgent()
        forecasts = [
            {'SolarGen': 2.0, 'HouseLoad': 3.0, 'ImportEnergyPrice': 0.2, 'ExportEnergyPrice': 0.1}
        ]
        adapter = AgentAdapter(agent, forecasts, start_index=10)
        
        provider = adapter.create_scenario_provider(forecasts, 10)
        
        # Test deterministic scenarios (no cache)
        scenario_values, scenario_probs = provider(10)  # stage_index=10, forecast_idx=0
        
        assert len(scenario_probs) == 1
        assert scenario_probs[0] == 1.0
        assert scenario_values['solar'][0] == 2.0
        assert scenario_values['load'][0] == 3.0
        assert scenario_values['import_price'][0] == 0.2
        assert scenario_values['export_price'][0] == 0.1


class TestEndToEnd:
    """End-to-end tests for the complete adapter workflow."""
    
    def test_run_dp_for_agent_basic(self):
        """Test run_dp_for_agent with a simple case."""
        agent = MockAgent()
        forecasts = [
            {'SolarGen': 2.0, 'HouseLoad': 3.0, 'ImportEnergyPrice': 0.2, 'ExportEnergyPrice': 0.1},
            {'SolarGen': 1.5, 'HouseLoad': 2.5, 'ImportEnergyPrice': 0.25, 'ExportEnergyPrice': 0.12}
        ]
        
        policy_table = run_dp_for_agent(agent, forecasts, start_index=0, chunk_length=2)
        
        # Check output format matches Agent._solve_sdp
        assert policy_table.shape == (2, 3)  # (horizon, num_soc_levels)
        assert policy_table.dtype == int
        
        # All policies should be valid action indices or -1
        valid_actions = set(range(len(agent.action_levels_norm))) | {-1}
        assert all(action in valid_actions for action in policy_table.flat)
    
    def test_run_dp_for_agent_chunking(self):
        """Test run_dp_for_agent with chunking for longer horizons."""
        agent = MockAgent()
        # Create longer forecast horizon
        forecasts = [
            {'SolarGen': 2.0, 'HouseLoad': 3.0, 'ImportEnergyPrice': 0.2, 'ExportEnergyPrice': 0.1}
        ] * 6  # 6 time steps
        
        policy_table = run_dp_for_agent(agent, forecasts, start_index=0, chunk_length=3)
        
        # Should handle chunking properly
        assert policy_table.shape == (6, 3)  # (horizon, num_soc_levels)
        
        # All policies should be valid
        valid_actions = set(range(len(agent.action_levels_norm))) | {-1}
        assert all(action in valid_actions for action in policy_table.flat)


if __name__ == '__main__':
    pytest.main([__file__])