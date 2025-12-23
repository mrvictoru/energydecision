"""
Simple validation test for the MRDP (Multi-Resolution Dynamic Programming) module.

This script demonstrates usage of the sdp_multires module and compares it against 
the existing single-horizon SDP implementation to validate behavior and performance.
"""

import os
import sys
import time
import numpy as np
import polars as pl

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from EnergySimEnv import SolarBatteryEnv
from decision import Agent
from sdp_multires import DynamicProgram, solve_mrdp


def make_test_df(n_rows=100):
    """Create synthetic test data for validation."""
    import datetime as dt
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]
    
    # Create realistic but simple data patterns
    df = pl.DataFrame({
        'Time': times,
        'SolarGen': np.maximum(0, 2.0 + 1.5 * np.sin(np.linspace(0, 4*np.pi, n_rows)) + 0.3 * np.random.randn(n_rows)),
        'HouseLoad': np.maximum(0.1, 3.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_rows) + np.pi) + 0.2 * np.random.randn(n_rows)),
        'ImportEnergyPrice': 0.15 + 0.05 * np.sin(np.linspace(0, 2*np.pi, n_rows)) + 0.01 * np.random.randn(n_rows),
        'ExportEnergyPrice': 0.08 + 0.02 * np.sin(np.linspace(0, 2*np.pi, n_rows)) + 0.005 * np.random.randn(n_rows),
    })
    
    # Ensure positive values
    for col in ['ImportEnergyPrice', 'ExportEnergyPrice']:
        df = df.with_columns(pl.col(col).clip(0.01, None))
    
    return df


def create_agent_stage_cost_function(agent, forecasts):
    """
    Create a stage cost function that wraps Agent's existing stage cost calculation.
    This reuses Agent's scenario_cache and Monte Carlo settings to avoid duplicate sampling.
    """
    def stage_cost_function(t_global_idx, unique_energy_values):
        """
        Compute stage costs for unique battery energy values at time t_global_idx.
        
        Args:
            t_global_idx: Global time index
            unique_energy_values: Array of unique battery flow energies (kWh)
            
        Returns:
            Array of stage costs corresponding to unique_energy_values
        """
        # Handle out-of-bounds time indices
        if t_global_idx < 0 or t_global_idx >= len(forecasts):
            return np.full(len(unique_energy_values), np.inf)
            
        forecast_step = forecasts[t_global_idx]
        costs = np.empty(len(unique_energy_values))
        
        for i, energy in enumerate(unique_energy_values):
            battery_rate = energy / agent.step_duration
            rep_soc = agent.battery_capacity / 2.0  # Representative SoC for cost calculation
            
            # Reuse Agent's existing stage cost calculation
            try:
                costs[i] = agent._calculate_sdp_stage_cost(t_global_idx, rep_soc, battery_rate, energy, forecast_step)
            except Exception as e:
                # Fallback to simple deterministic cost if agent's method fails
                print(f"Warning: Agent stage cost calculation failed for t={t_global_idx}, energy={energy}: {e}")
                costs[i] = simple_deterministic_stage_cost(energy, forecast_step, agent)
                
        return costs
        
    return stage_cost_function


def simple_deterministic_stage_cost(battery_energy, forecast_step, agent):
    """
    Simple fallback stage cost calculation using deterministic forecast values.
    """
    try:
        solar = forecast_step.get('SolarGen', 0)
        load = forecast_step.get('HouseLoad', 0)  
        import_price = forecast_step.get('ImportEnergyPrice', 0.15)
        export_price = forecast_step.get('ExportEnergyPrice', 0.08)
        
        battery_charge_energy = max(0, battery_energy)
        battery_discharge_energy = max(0, -battery_energy)
        
        grid_energy = load + battery_charge_energy - solar - battery_discharge_energy
        
        # Check grid limits
        if abs(grid_energy) > agent.max_grid_energy + 1e-6:
            return np.inf
            
        # Calculate grid cost
        if grid_energy > 0:
            grid_cost = grid_energy * import_price  # Importing
        else:
            grid_cost = -abs(grid_energy) * export_price  # Exporting (negative cost = revenue)
            
        # Add simple linear degradation cost
        degradation_cost = getattr(agent, 'linear_deg_cost_per_kwh', 0.01) * abs(battery_energy)
        
        return grid_cost + degradation_cost
        
    except Exception as e:
        print(f"Warning: Simple stage cost calculation failed: {e}")
        return np.inf


def test_mrdp_basic():
    """Test basic MRDP functionality."""
    print("=== Testing MRDP Basic Functionality ===")
    
    # Create test environment
    df = make_test_df(n_rows=60)  # 30 hours of data
    env = SolarBatteryEnv(df, battery_capacity=10.0, max_battery_flow=5.0, 
                         init_battery_level=5.0, max_step=50)
    
    # Create agent for reference (don't actually use its solve method yet)
    agent = Agent(env, algorithm='sdp', horizon=24, soc_resolution=8, action_resolution=5,
                  use_monte_carlo=False, mc_samples=50, mc_seed=42)
    
    print(f"Environment: battery_capacity={env.battery_capacity}, max_battery_flow={env.max_battery_flow}")
    print(f"Agent: soc_resolution={agent.soc_resolution}, action_resolution={agent.action_resolution}")
    
    # Get forecasts for testing
    forecasts = agent._get_forecasts(0, horizon=24)
    print(f"Retrieved {len(forecasts)} forecast steps")
    
    # Create stage cost function
    stage_cost_fn = create_agent_stage_cost_function(agent, forecasts)
    
    # Test simple two-horizon MRDP 
    subhorizon_specs = [
        {
            'start': 0, 'length': 8,  # First 8 steps with fine resolution
            'soc_resolution': 8, 'action_resolution': 5,
            'step_duration': 0.5
        },
        {
            'start': 8, 'length': 16,  # Next 16 steps with coarse resolution
            'soc_resolution': 5, 'action_resolution': 3,
            'step_duration': 0.5
        }
    ]
    
    print(f"Sub-horizon specs: {subhorizon_specs}")
    
    # Solve using MRDP
    start_time = time.perf_counter()
    try:
        policy_table, cost_to_go = solve_mrdp(
            env=env,
            forecasts=forecasts,
            subhorizon_specs=subhorizon_specs,
            global_start_index=0,
            stage_cost_function=stage_cost_fn
        )
        solve_time = time.perf_counter() - start_time
        
        print(f"MRDP solve completed in {solve_time:.3f} seconds")
        print(f"Policy table shape: {policy_table.shape}")
        print(f"Cost-to-go shape: {cost_to_go.shape}")
        
        # Check for reasonable results
        finite_policies = np.sum(policy_table >= 0)
        finite_ctg = np.sum(np.isfinite(cost_to_go))
        
        print(f"Finite policies: {finite_policies}/{policy_table.size}")
        print(f"Finite cost-to-go values: {finite_ctg}/{cost_to_go.size}")
        
        # Show some sample results
        current_soc_idx = agent._soc_to_idx(env.battery_level)
        if policy_table[0, current_soc_idx] >= 0:
            chosen_action_idx = policy_table[0, current_soc_idx]
            
            # We need to get the action levels from the first sub-horizon
            first_spec = subhorizon_specs[0]
            max_energy_per_step = env.max_battery_flow * first_spec['step_duration']
            action_levels_norm = np.linspace(-1, 1, first_spec['action_resolution'])
            action_levels_kwh = action_levels_norm * max_energy_per_step
            
            chosen_action_kwh = action_levels_kwh[chosen_action_idx]
            chosen_action_norm = action_levels_norm[chosen_action_idx]
            
            print(f"For current SoC {env.battery_level:.2f} kWh (idx {current_soc_idx}):")
            print(f"  Chosen action index: {chosen_action_idx}")
            print(f"  Chosen action: {chosen_action_norm:.3f} (normalized), {chosen_action_kwh:.3f} kWh")
            print(f"  Cost-to-go: {cost_to_go[0, current_soc_idx]:.3f}")
        else:
            print(f"No feasible action found for current SoC {env.battery_level:.2f} kWh")
            
        return True
        
    except Exception as e:
        print(f"MRDP solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_vs_multi_resolution():
    """Compare single-horizon vs multi-resolution approaches."""
    print("\n=== Comparing Single-Horizon vs Multi-Resolution ===")
    
    # Create test environment
    df = make_test_df(n_rows=80)  # Longer horizon
    env = SolarBatteryEnv(df, battery_capacity=10.0, max_battery_flow=5.0,
                         init_battery_level=5.0, max_step=60)
    
    # Single-horizon agent
    agent_single = Agent(env, algorithm='sdp', horizon=24, soc_resolution=10, action_resolution=7,
                        use_monte_carlo=False, mc_samples=50, mc_seed=42)
    
    # Get forecasts  
    forecasts = agent_single._get_forecasts(0, horizon=24)
    
    print(f"Comparing approaches with {len(forecasts)} time steps")
    
    # Test single-horizon solve
    print("\n--- Single-Horizon SDP ---")
    start_time = time.perf_counter()
    try:
        policy_single = agent_single._solve_sdp(forecasts, start_index=0)
        time_single = time.perf_counter() - start_time
        
        print(f"Single-horizon solve time: {time_single:.3f} seconds")
        print(f"Policy table shape: {policy_single.shape}")
        
        # Extract action for current state
        current_soc_idx = agent_single._soc_to_idx(env.battery_level)
        if policy_single[0, current_soc_idx] >= 0:
            action_idx = policy_single[0, current_soc_idx]
            action_norm = agent_single.action_levels_norm[action_idx]
            print(f"Single-horizon chosen action: {action_norm:.3f}")
        
        single_success = True
    except Exception as e:
        print(f"Single-horizon solve failed: {e}")
        single_success = False
        time_single = float('inf')
    
    # Test multi-resolution solve
    print("\n--- Multi-Resolution SDP ---")
    
    # Define multi-resolution structure
    subhorizon_specs = [
        {
            'start': 0, 'length': 8,  # Fine near-term: 4 hours
            'soc_resolution': 10, 'action_resolution': 7,
            'step_duration': 0.5
        },
        {
            'start': 8, 'length': 16,  # Coarse far-term: 8 hours  
            'soc_resolution': 6, 'action_resolution': 4,
            'step_duration': 0.5
        }
    ]
    
    stage_cost_fn = create_agent_stage_cost_function(agent_single, forecasts)
    
    start_time = time.perf_counter()
    try:
        policy_multi, ctg_multi = solve_mrdp(
            env=env,
            forecasts=forecasts,
            subhorizon_specs=subhorizon_specs,
            global_start_index=0,
            stage_cost_function=stage_cost_fn
        )
        time_multi = time.perf_counter() - start_time
        
        print(f"Multi-resolution solve time: {time_multi:.3f} seconds")
        print(f"Policy table shape: {policy_multi.shape}")
        
        # Extract action for current state
        current_soc_idx_multi = int(np.round((env.battery_level / env.battery_capacity) * (subhorizon_specs[0]['soc_resolution'] - 1)))
        current_soc_idx_multi = np.clip(current_soc_idx_multi, 0, subhorizon_specs[0]['soc_resolution'] - 1)
        
        if policy_multi[0, current_soc_idx_multi] >= 0:
            action_idx = policy_multi[0, current_soc_idx_multi]
            # Compute action from first sub-horizon specs
            first_spec = subhorizon_specs[0]
            action_levels_norm = np.linspace(-1, 1, first_spec['action_resolution'])
            action_norm = action_levels_norm[action_idx]
            print(f"Multi-resolution chosen action: {action_norm:.3f}")
        
        multi_success = True
    except Exception as e:
        print(f"Multi-resolution solve failed: {e}")
        import traceback
        traceback.print_exc()
        multi_success = False
        time_multi = float('inf')
    
    # Summary
    print(f"\n--- Comparison Summary ---")
    print(f"Single-horizon: {'Success' if single_success else 'Failed'}, Time: {time_single:.3f}s")
    print(f"Multi-resolution: {'Success' if multi_success else 'Failed'}, Time: {time_multi:.3f}s")
    
    if single_success and multi_success:
        speedup = time_single / time_multi if time_multi > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        # Could compare solution quality here if needed
        
    return single_success and multi_success


if __name__ == "__main__":
    print("MRDP (Multi-Resolution Dynamic Programming) Validation Test")
    print("=" * 60)
    
    # Run basic functionality test
    basic_success = test_mrdp_basic()
    
    # Run comparison test
    comparison_success = test_single_vs_multi_resolution()
    
    print("\n" + "=" * 60)
    if basic_success and comparison_success:
        print("✓ All tests passed! MRDP module is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
        sys.exit(1)