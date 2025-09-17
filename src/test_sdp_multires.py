"""
Test suite for the sdp_multires module
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from sdp_multires import (
    compute_unique_energy_costs_vectorized,
    DynamicProgram,
    clear_stage_cost_cache,
    get_cache_stats,
    _compute_degradation_costs_vectorized
)


class TestComputeUniqueEnergyCostsVectorized:
    """Test vectorized energy cost computation."""
    
    def test_basic_functionality(self):
        """Test basic vectorized computation."""
        # Setup test data
        np.random.seed(42)
        n_samples = 100
        sampled_solar = np.random.uniform(1.0, 3.0, n_samples)
        sampled_load = np.random.uniform(3.0, 5.0, n_samples)
        sampled_imp = np.random.uniform(0.15, 0.25, n_samples)
        sampled_exp = np.random.uniform(0.08, 0.12, n_samples)
        monte_samples = (sampled_solar, sampled_load, sampled_imp, sampled_exp)
        
        unique_energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        step_duration = 0.5
        max_grid_energy = 10.0
        degradation_params = {
            'model': 'linear',
            'linear_deg_cost_per_kwh': 0.01
        }
        
        # Compute costs
        costs = compute_unique_energy_costs_vectorized(
            monte_samples, unique_energies, step_duration, 
            max_grid_energy, degradation_params
        )
        
        # Basic checks
        assert len(costs) == len(unique_energies)
        assert all(np.isfinite(costs))
        assert all(costs >= 0)  # Costs should be non-negative for this test case
    
    def test_grid_limit_violation(self):
        """Test that grid limit violations return infinite cost."""
        # Setup with very high load to exceed grid limits
        n_samples = 10
        sampled_solar = np.zeros(n_samples)
        sampled_load = np.full(n_samples, 15.0)  # Very high load
        sampled_imp = np.full(n_samples, 0.2)
        sampled_exp = np.full(n_samples, 0.1)
        monte_samples = (sampled_solar, sampled_load, sampled_imp, sampled_exp)
        
        unique_energies = np.array([0.0])  # No battery action
        step_duration = 1.0
        max_grid_energy = 5.0  # Low limit to trigger violation
        degradation_params = {'model': 'linear', 'linear_deg_cost_per_kwh': 0.0}
        
        costs = compute_unique_energy_costs_vectorized(
            monte_samples, unique_energies, step_duration,
            max_grid_energy, degradation_params
        )
        
        assert costs[0] == np.inf
    
    def test_degradation_models(self):
        """Test both linear and static degradation models."""
        # Simple test setup
        n_samples = 10
        monte_samples = (
            np.ones(n_samples),  # solar
            np.ones(n_samples) * 2,  # load  
            np.ones(n_samples) * 0.2,  # import price
            np.ones(n_samples) * 0.1   # export price
        )
        
        unique_energies = np.array([1.0])
        step_duration = 1.0
        max_grid_energy = 10.0
        
        # Test linear degradation
        linear_params = {
            'model': 'linear',
            'linear_deg_cost_per_kwh': 0.05
        }
        costs_linear = compute_unique_energy_costs_vectorized(
            monte_samples, unique_energies, step_duration,
            max_grid_energy, linear_params
        )
        
        # Test static degradation
        static_params = {
            'model': 'static',
            'battery_capacity': 10.0,
            'battery_life_cost': 5000.0,
            'static_deg_correction_factor': 0.01
        }
        costs_static = compute_unique_energy_costs_vectorized(
            monte_samples, unique_energies, step_duration,
            max_grid_energy, static_params
        )
        
        assert np.isfinite(costs_linear[0])
        assert np.isfinite(costs_static[0])
        assert costs_linear[0] != costs_static[0]  # Should be different


class TestDynamicProgram:
    """Test DynamicProgram class functionality."""
    
    def test_initialization(self):
        """Test DP initialization."""
        soc_levels = np.linspace(0, 10, 11)
        action_levels = np.linspace(-1, 1, 5)
        step_duration = 0.5
        max_battery_flow = 5.0
        battery_capacity = 10.0
        
        dp = DynamicProgram(
            soc_levels, action_levels, step_duration, 
            max_battery_flow, battery_capacity
        )
        
        assert len(dp.soc_levels_kwh) == 11
        assert len(dp.action_levels_norm) == 5
        assert dp.step_duration == 0.5
        assert len(dp.battery_flow_energies) == 5
    
    def test_mc_samples_schedule(self):
        """Test MC samples schedule creation."""
        soc_levels = np.linspace(0, 10, 6)
        action_levels = np.linspace(-1, 1, 3)
        dp = DynamicProgram(soc_levels, action_levels, 0.5, 5.0, 10.0)
        
        # Test with subhorizon specs
        subhorizon_specs = [
            {'horizon': 5, 'mc_samples': 200},
            {'horizon': 10, 'mc_samples': 50}
        ]
        
        schedule = dp._create_mc_samples_schedule(15, subhorizon_specs)
        
        assert len(schedule) == 15
        assert schedule[0] == 200  # First 5 steps
        assert schedule[4] == 200
        assert schedule[5] == 50   # Next 10 steps
        assert schedule[14] == 50
    
    def test_solve_basic(self):
        """Test basic solve functionality."""
        # Small problem for testing
        soc_levels = np.array([0.0, 5.0, 10.0])
        action_levels = np.array([-1.0, 0.0, 1.0])
        dp = DynamicProgram(soc_levels, action_levels, 1.0, 2.0, 10.0)
        
        # Simple forecasts
        forecasts = [
            {'SolarGen': 1.0, 'HouseLoad': 2.0, 'ImportEnergyPrice': 0.2, 'ExportEnergyPrice': 0.1},
            {'SolarGen': 1.5, 'HouseLoad': 2.5, 'ImportEnergyPrice': 0.25, 'ExportEnergyPrice': 0.12}
        ]
        
        # Simple stage cost function
        def simple_stage_cost(timestep, unique_energies, monte_samples, step_duration, 
                            max_grid_energy, degradation_params):
            return np.abs(unique_energies) * 0.1  # Simple cost proportional to energy
        
        # Solve
        policy = dp.solve(forecasts, simple_stage_cost, use_cache=False)
        
        assert policy.shape == (2, 3)  # (horizon, num_soc_levels)
        assert np.all((policy >= -1) & (policy <= 2))  # Valid action indices or -1


class TestCaching:
    """Test caching functionality."""
    
    def test_cache_operations(self):
        """Test cache clearing and stats."""
        clear_stage_cost_cache()
        
        initial_stats = get_cache_stats()
        assert initial_stats['cache_size'] == 0
        
        # The cache is populated internally by the module functions
        # This is a basic test of the API
        stats = get_cache_stats()
        assert 'cache_size' in stats
        assert 'memory_usage_bytes' in stats


class TestDegradationCosts:
    """Test degradation cost computation."""
    
    def test_linear_degradation(self):
        """Test linear degradation cost calculation."""
        energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        step_duration = 1.0
        params = {
            'model': 'linear',
            'linear_deg_cost_per_kwh': 0.05
        }
        
        costs = _compute_degradation_costs_vectorized(energies, step_duration, params)
        
        expected = 0.05 * np.abs(energies)
        np.testing.assert_array_almost_equal(costs, expected)
    
    def test_static_degradation(self):
        """Test static degradation cost calculation."""
        energies = np.array([0.0, 1.0, 2.0])
        step_duration = 1.0
        params = {
            'model': 'static',
            'battery_capacity': 10.0,
            'battery_life_cost': 5000.0,
            'static_deg_correction_factor': 0.01
        }
        
        costs = _compute_degradation_costs_vectorized(energies, step_duration, params)
        
        assert len(costs) == len(energies)
        assert costs[0] == 0.0  # No energy, no degradation
        assert all(costs >= 0.0)  # Non-negative costs


def test_module_import():
    """Test that the module imports correctly."""
    import sdp_multires
    
    # Check main functions are available
    assert hasattr(sdp_multires, 'compute_unique_energy_costs_vectorized')
    assert hasattr(sdp_multires, 'DynamicProgram')
    assert hasattr(sdp_multires, 'clear_stage_cost_cache')
    assert hasattr(sdp_multires, 'get_cache_stats')


if __name__ == "__main__":
    # Run basic tests
    test_module_import()
    
    # Test basic functionality
    test_compute = TestComputeUniqueEnergyCostsVectorized()
    test_compute.test_basic_functionality()
    test_compute.test_degradation_models()
    
    test_dp = TestDynamicProgram()
    test_dp.test_initialization()
    test_dp.test_mc_samples_schedule()
    
    test_deg = TestDegradationCosts()
    test_deg.test_linear_degradation()
    
    print("All basic tests passed!")