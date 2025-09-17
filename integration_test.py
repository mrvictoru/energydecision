"""
Integration test showing how to use the MRDP module with the existing Agent class.

This demonstrates the integration pattern described in the module docstring.
"""

import sys
import os
import numpy as np
import polars as pl

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from decision import Agent
from EnergySimEnv import SolarBatteryEnv
from sdp_multires import DynamicProgram, compute_unique_energy_costs_vectorized


def create_test_environment():
    """Create a simple test environment."""
    import datetime as dt
    
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(100)]
    
    np.random.seed(42)
    df = pl.DataFrame({
        'Time': times,
        'SolarGen': np.abs(np.random.normal(2.0, 1.0, 100)),
        'HouseLoad': np.abs(np.random.normal(4.0, 1.0, 100)),
        'ImportEnergyPrice': np.random.uniform(0.1, 0.3, 100),
        'ExportEnergyPrice': np.random.uniform(0.05, 0.15, 100),
    })
    
    return SolarBatteryEnv(
        df, battery_capacity=7.0, max_battery_flow=3.3, 
        init_battery_level=3.5, max_step=1000
    )


def test_integration():
    """Test integration between Agent and MRDP module."""
    print("Testing MRDP integration with Agent...")
    
    # Create environment and agent
    env = create_test_environment()
    agent = Agent(
        env, algorithm='sdp', horizon=24, 
        soc_resolution=8, action_resolution=5,
        use_monte_carlo=True, mc_samples=100, mc_seed=42
    )
    
    # Initialize stage cost cache in agent
    agent._stage_cost_cache = {}
    
    # Define sub-horizon specifications
    subhorizon_specs = [
        {'horizon': 8, 'mc_samples': 150},   # Near: higher accuracy
        {'horizon': 16, 'mc_samples': 50}    # Far: lower accuracy
    ]
    
    # Create DynamicProgram instance
    dp = DynamicProgram(
        soc_levels=agent.soc_levels_kwh,
        action_levels=agent.action_levels_norm,
        step_duration=agent.step_duration,
        max_battery_flow=agent.max_battery_flow,
        battery_capacity=agent.battery_capacity
    )
    
    # Create cached vectorized stage cost function as shown in docstring
    def cached_vectorized_stage_cost(timestep, unique_energies, monte_samples, 
                                   step_duration, max_grid_energy, degradation_params):
        # Check cache first for each energy value
        cached_results = {}
        uncached_energies = []
        uncached_indices = []
        
        for i, energy in enumerate(unique_energies):
            rounded_energy = round(float(energy), 10)
            cache_key = (timestep, rounded_energy)
            if cache_key in agent._stage_cost_cache:
                cached_results[i] = agent._stage_cost_cache[cache_key]
            else:
                uncached_energies.append(energy)
                uncached_indices.append(i)
        
        # Compute uncached values using vectorized function
        if uncached_energies:
            uncached_costs = compute_unique_energy_costs_vectorized(
                monte_samples, np.array(uncached_energies), step_duration, 
                max_grid_energy, degradation_params
            )
            
            # Cache the results
            for idx, energy_idx in enumerate(uncached_indices):
                energy = unique_energies[energy_idx]
                rounded_energy = round(float(energy), 10)
                cache_key = (timestep, rounded_energy)
                agent._stage_cost_cache[cache_key] = uncached_costs[idx]
                cached_results[energy_idx] = uncached_costs[idx]
        
        # Reconstruct full results array
        results = np.array([cached_results[i] for i in range(len(unique_energies))])
        return results
    
    # Get forecasts for the full horizon
    total_horizon = sum(spec['horizon'] for spec in subhorizon_specs)
    forecasts = agent._get_forecasts(0, total_horizon)
    
    print(f"Solving SDP with {total_horizon} step horizon...")
    print(f"Sub-horizon specs: {subhorizon_specs}")
    
    # Solve using optimized MRDP
    policy = dp.solve(
        forecasts=forecasts, 
        stage_cost_function=cached_vectorized_stage_cost,
        subhorizon_specs=subhorizon_specs,
        use_cache=True,
        start_index=0
    )
    
    # Validate results
    assert policy.shape == (total_horizon, len(agent.soc_levels_kwh))
    assert np.all((policy >= -1) & (policy < len(agent.action_levels_norm)))
    
    # Check cache usage
    cache_size = len(agent._stage_cost_cache)
    print(f"Cache populated with {cache_size} entries")
    
    # Test that we can extract a decision for current SoC
    current_soc_idx = agent._soc_to_idx(env.battery_level)
    action_idx = policy[0, current_soc_idx]
    
    if action_idx >= 0:
        chosen_action = agent.action_levels_norm[action_idx]
        print(f"Current SoC: {env.battery_level:.2f} kWh (idx {current_soc_idx})")
        print(f"Chosen action: {chosen_action:.3f} (idx {action_idx})")
    else:
        print(f"No feasible action for current SoC: {env.battery_level:.2f} kWh")
    
    print("✅ Integration test passed!")
    return policy, cache_size


def test_comparison():
    """Compare original vs MRDP performance on same problem."""
    print("\nComparing original Agent vs MRDP...")
    
    env = create_test_environment()
    
    # Test original agent
    import time
    agent_orig = Agent(
        env, algorithm='sdp', horizon=24,
        soc_resolution=8, action_resolution=5,
        use_monte_carlo=True, mc_samples=100, mc_seed=42
    )
    
    forecasts_orig = agent_orig._get_forecasts(0, 24)
    
    start_time = time.perf_counter()
    policy_orig = agent_orig._solve_sdp(forecasts_orig, start_index=0)
    orig_time = time.perf_counter() - start_time
    
    # Test MRDP version
    agent_mrdp = Agent(
        env, algorithm='sdp', horizon=24,
        soc_resolution=8, action_resolution=5,
        use_monte_carlo=True, mc_samples=100, mc_seed=42
    )
    agent_mrdp._stage_cost_cache = {}
    
    dp = DynamicProgram(
        soc_levels=agent_mrdp.soc_levels_kwh,
        action_levels=agent_mrdp.action_levels_norm,
        step_duration=agent_mrdp.step_duration,
        max_battery_flow=agent_mrdp.max_battery_flow,
        battery_capacity=agent_mrdp.battery_capacity
    )
    
    def simple_stage_cost(timestep, unique_energies, monte_samples, 
                         step_duration, max_grid_energy, degradation_params):
        return compute_unique_energy_costs_vectorized(
            monte_samples, unique_energies, step_duration,
            max_grid_energy, degradation_params
        )
    
    forecasts_mrdp = agent_mrdp._get_forecasts(0, 24)
    
    start_time = time.perf_counter()
    policy_mrdp = dp.solve(
        forecasts=forecasts_mrdp,
        stage_cost_function=simple_stage_cost,
        use_cache=False  # Fair comparison without caching
    )
    mrdp_time = time.perf_counter() - start_time
    
    print(f"Original Agent time: {orig_time:.4f} seconds")
    print(f"MRDP time:          {mrdp_time:.4f} seconds")
    print(f"Speedup:            {orig_time/mrdp_time:.2f}x")
    
    # Policies should have same shape
    assert policy_orig.shape == policy_mrdp.shape
    print("✅ Comparison test passed!")


if __name__ == "__main__":
    try:
        # Run integration test
        policy, cache_size = test_integration()
        
        # Run comparison test  
        test_comparison()
        
        print(f"\nAll integration tests passed! 🎉")
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()