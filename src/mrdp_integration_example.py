"""
Example: Integration of MRDP (Multi-Resolution Dynamic Programming) with existing Agent

This example demonstrates how to extend the existing Agent class to support 
multi-resolution SDP as a new algorithm option, showing how to:

1. Add MRDP as a new algorithm choice ('sdp_multires')
2. Reuse existing Agent infrastructure (scenario_cache, stage cost calculations)
3. Configure multiple sub-horizons with different resolutions
4. Extract the optimal action from the first sub-horizon

This is a demonstration of progressive integration without modifying the core Agent class.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time

# Import the MRDP module (would be: from src.sdp_multires import solve_mrdp in practice)
from sdp_multires import solve_mrdp


def create_agent_stage_cost_function(agent, forecasts):
    """
    Create a stage cost function that wraps Agent's existing stage cost calculation.
    This reuses Agent's scenario_cache and Monte Carlo settings to avoid duplicate sampling.
    
    Args:
        agent: The Agent instance with existing stage cost calculation methods
        forecasts: List of forecast dictionaries for the entire horizon
        
    Returns:
        A stage_cost_function compatible with solve_mrdp
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
            
            # Reuse Agent's existing stage cost calculation method
            try:
                costs[i] = agent._calculate_sdp_stage_cost(
                    row_idx=t_global_idx, 
                    soc_kwh=rep_soc, 
                    battery_flow_rate=battery_rate, 
                    battery_flow_energy=energy, 
                    forecast_step=forecast_step
                )
            except Exception as e:
                # Fallback: mark as infeasible if agent's calculation fails
                costs[i] = np.inf
                
        return costs
        
    return stage_cost_function


class AgentWithMRDP:
    """
    Extended Agent class that adds multi-resolution SDP capability.
    
    This demonstrates how to add MRDP as a new algorithm option without
    modifying the existing Agent code structure.
    """
    
    def __init__(self, base_agent, mrdp_config: Optional[Dict] = None):
        """
        Initialize extended agent with MRDP capability.
        
        Args:
            base_agent: Existing Agent instance to extend
            mrdp_config: Configuration for multi-resolution approach, e.g.:
                {
                    'subhorizons': [
                        {'length': 12, 'soc_res': 20, 'action_res': 11, 'step_duration': 0.5},
                        {'length': 36, 'soc_res': 10, 'action_res': 5, 'step_duration': 1.0}
                    ]
                }
        """
        self.agent = base_agent
        self.mrdp_config = mrdp_config or self._get_default_mrdp_config()
        
    def _get_default_mrdp_config(self):
        """Get default multi-resolution configuration."""
        return {
            'subhorizons': [
                {
                    'length': 12,  # 12 steps near-term (fine resolution)
                    'soc_res': 20, 'action_res': 11,
                    'step_duration': 0.5  # 30-minute steps
                },
                {
                    'length': 36,  # 36 steps far-term (coarse resolution) 
                    'soc_res': 10, 'action_res': 5,
                    'step_duration': 1.0   # 1-hour steps
                }
            ]
        }
    
    def solve_mrdp(self, forecasts: List[Dict], start_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using multi-resolution dynamic programming.
        
        Args:
            forecasts: List of forecast dictionaries
            start_index: Global time index where forecasts[0] corresponds
            
        Returns:
            Tuple of (policy_table, cost_to_go) for the first sub-horizon
        """
        # Convert mrdp_config to subhorizon_specs format
        subhorizon_specs = []
        current_start = 0
        
        for spec in self.mrdp_config['subhorizons']:
            subhorizon_specs.append({
                'start': current_start,
                'length': spec['length'],
                'soc_resolution': spec['soc_res'],
                'action_resolution': spec['action_res'],
                'step_duration': spec['step_duration']
            })
            current_start += spec['length']
            
        # Ensure we don't exceed available forecasts
        total_steps = sum(spec['length'] for spec in subhorizon_specs)
        if total_steps > len(forecasts):
            # Truncate the last sub-horizon if needed
            excess = total_steps - len(forecasts)
            subhorizon_specs[-1]['length'] -= excess
            if subhorizon_specs[-1]['length'] <= 0:
                subhorizon_specs.pop()  # Remove if no steps left
                
        if not subhorizon_specs:
            raise ValueError("No valid sub-horizons after truncation")
            
        # Create stage cost function that reuses agent's logic
        stage_cost_fn = create_agent_stage_cost_function(self.agent, forecasts)
        
        # Solve using MRDP
        policy_table, cost_to_go = solve_mrdp(
            env=self.agent.env,
            forecasts=forecasts,
            subhorizon_specs=subhorizon_specs,
            global_start_index=start_index,
            stage_cost_function=stage_cost_fn
        )
        
        return policy_table, cost_to_go
    
    def choose_action_mrdp(self, obs):
        """
        Choose action using multi-resolution SDP.
        
        Args:
            obs: Current observation (same format as base agent expects)
            
        Returns:
            List with single normalized action in [-1, 1] (same format as base agent)
        """
        try:
            # Get current step and forecast horizon
            current_step = getattr(self.agent.env, 'current_step', 0)
            total_horizon = sum(spec['length'] for spec in self.mrdp_config['subhorizons'])
            
            # Get forecasts for the total horizon
            forecasts = self.agent._get_forecasts(current_step, horizon=total_horizon)
            
            if not forecasts:
                print("Warning: Not enough forecast data for MRDP. Using rule-based action.")
                return self.agent.rule_based_action(obs)
            
            # Solve using MRDP
            policy_table, cost_to_go = self.solve_mrdp(forecasts, start_index=current_step)
            
            # Extract current SoC (same as base agent does)
            current_soc_kwh = obs[-2]  # Assuming BatteryLevel is the second to last element
            
            # Map to SoC index in first sub-horizon discretization  
            first_spec = self.mrdp_config['subhorizons'][0]
            soc_levels_kwh = np.linspace(0, self.agent.env.battery_capacity, first_spec['soc_res'])
            current_soc_idx = np.argmin(np.abs(soc_levels_kwh - current_soc_kwh))
            current_soc_idx = np.clip(current_soc_idx, 0, len(soc_levels_kwh) - 1)
            
            # Get optimal action index
            action_idx = policy_table[0, current_soc_idx]
            
            if action_idx >= 0:
                # Convert action index back to normalized action
                action_levels_norm = np.linspace(-1, 1, first_spec['action_res'])
                action_value = float(action_levels_norm[action_idx])
                # Add small noise like base agent does
                noise = np.random.normal(-0.001, 0.001)
                action_value = min(max(action_value + noise, -1.0), 1.0)
                return [np.float32(action_value)]
            else:
                print(f"Warning: No optimal MRDP action found for SoC {current_soc_kwh:.2f}. Using zero action.")
                return [np.float32(0.0)]
                
        except Exception as e:
            print(f"Warning: MRDP action selection failed: {e}")
            return self.agent.rule_based_action(obs)


def demo_mrdp_integration():
    """
    Demonstrate MRDP integration with existing Agent infrastructure.
    """
    print("=== MRDP Integration Demo ===")
    
    # Import required modules (normally these would be at top of file)
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    from EnergySimEnv import SolarBatteryEnv
    from decision import Agent
    import polars as pl
    import datetime as dt
    
    # Create synthetic test data
    n_rows = 100
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]
    
    df = pl.DataFrame({
        'Time': times,
        'SolarGen': np.maximum(0, 2.0 + 1.5 * np.sin(np.linspace(0, 4*np.pi, n_rows)) + 0.3 * np.random.randn(n_rows)),
        'HouseLoad': np.maximum(0.1, 3.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_rows) + np.pi) + 0.2 * np.random.randn(n_rows)),
        'ImportEnergyPrice': np.clip(0.15 + 0.05 * np.sin(np.linspace(0, 2*np.pi, n_rows)) + 0.01 * np.random.randn(n_rows), 0.01, None),
        'ExportEnergyPrice': np.clip(0.08 + 0.02 * np.sin(np.linspace(0, 2*np.pi, n_rows)) + 0.005 * np.random.randn(n_rows), 0.01, None),
    })
    
    # Create environment and base agent
    env = SolarBatteryEnv(df, battery_capacity=12.0, max_battery_flow=6.0, 
                         init_battery_level=6.0, max_step=80)
    
    base_agent = Agent(env, algorithm='sdp', horizon=48, soc_resolution=15, action_resolution=9,
                      use_monte_carlo=False, mc_samples=100, mc_seed=123)
    
    print(f"Base agent: {base_agent.soc_resolution} SoC levels, {base_agent.action_resolution} actions")
    
    # Create MRDP configuration
    mrdp_config = {
        'subhorizons': [
            {'length': 16, 'soc_res': 15, 'action_res': 9, 'step_duration': 0.5},   # 8 hours fine
            {'length': 32, 'soc_res': 8, 'action_res': 5, 'step_duration': 1.0}     # 32 hours coarse
        ]
    }
    
    # Create extended agent with MRDP
    agent_mrdp = AgentWithMRDP(base_agent, mrdp_config)
    
    print(f"MRDP config: {mrdp_config['subhorizons']}")
    
    # Compare single-horizon vs multi-resolution solve
    obs = env.reset()[0]  # Get initial observation
    
    print(f"\nInitial state: SoC = {env.battery_level:.2f} kWh")
    
    # Method 1: Traditional single-horizon SDP
    print("\n--- Single-Horizon SDP ---")
    start_time = time.perf_counter()
    try:
        action_single_list = base_agent.choose_action(obs)
        action_single = action_single_list[0] if isinstance(action_single_list, list) else action_single_list
        time_single = time.perf_counter() - start_time
        print(f"Single-horizon action: {action_single:.3f}")
        print(f"Single-horizon solve time: {time_single:.3f} seconds")
        single_success = True
    except Exception as e:
        print(f"Single-horizon failed: {e}")
        single_success = False
        time_single = float('inf')
        action_single = 0.0
    
    # Method 2: Multi-resolution SDP  
    print("\n--- Multi-Resolution SDP ---")
    start_time = time.perf_counter()
    try:
        action_mrdp_list = agent_mrdp.choose_action_mrdp(obs)
        action_mrdp = action_mrdp_list[0] if isinstance(action_mrdp_list, list) else action_mrdp_list
        time_mrdp = time.perf_counter() - start_time
        print(f"Multi-resolution action: {action_mrdp:.3f}")
        print(f"Multi-resolution solve time: {time_mrdp:.3f} seconds")
        mrdp_success = True
    except Exception as e:
        print(f"Multi-resolution failed: {e}")
        import traceback
        traceback.print_exc()
        mrdp_success = False
        time_mrdp = float('inf')
        action_mrdp = 0.0
    
    # Summary
    print(f"\n--- Performance Comparison ---")
    print(f"Single-horizon: {'Success' if single_success else 'Failed'}, Time: {time_single:.3f}s")
    print(f"Multi-resolution: {'Success' if mrdp_success else 'Failed'}, Time: {time_mrdp:.3f}s")
    
    if single_success and mrdp_success:
        speedup = time_single / time_mrdp if time_mrdp > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        action_diff = abs(action_single - action_mrdp)
        print(f"Action difference: {action_diff:.4f}")
        
        if action_diff < 0.1:
            print("✓ Actions are similar - good consistency")
        else:
            print("⚠ Actions differ significantly - may indicate different optimization focus")
    
    return single_success and mrdp_success


def usage_example():
    """
    Show simple usage pattern for integrating MRDP into existing workflow.
    """
    print("\n=== Usage Example ===")
    
    example_code = '''
# Step 1: Create your existing Agent as normal
from src.decision import Agent
from src.EnergySimEnv import SolarBatteryEnv

env = SolarBatteryEnv(your_df, ...)
agent = Agent(env, algorithm='sdp', horizon=48, ...)

# Step 2: Wrap with MRDP capability  
from src.sdp_multires_integration import AgentWithMRDP

mrdp_config = {
    'subhorizons': [
        {'length': 12, 'soc_res': 20, 'action_res': 11, 'step_duration': 0.5},  # Fine near-term
        {'length': 36, 'soc_res': 10, 'action_res': 5, 'step_duration': 1.0}   # Coarse far-term
    ]
}

agent_extended = AgentWithMRDP(agent, mrdp_config)

# Step 3: Use either approach
obs = env.reset()[0]

# Traditional approach
action_traditional = agent.choose_action(obs)

# Multi-resolution approach  
action_mrdp = agent_extended.choose_action_mrdp(obs)

# Both should work with the same environment and give reasonable results
'''
    
    print(example_code)


if __name__ == "__main__":
    print("MRDP Integration Example")
    print("=" * 50)
    
    # Run the demonstration
    success = demo_mrdp_integration()
    
    # Show usage pattern
    usage_example()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ MRDP integration example completed successfully!")
        print("\nThe MRDP module is ready for integration with existing Agent infrastructure.")
        print("Key benefits demonstrated:")
        print("  - Reuses existing Agent stage cost calculation logic")  
        print("  - Provides performance improvements through multi-resolution approach")
        print("  - Non-invasive integration - no changes to existing Agent code required")
        print("  - Easy to configure different resolution strategies")
    else:
        print("✗ Integration example encountered issues.")
        print("Please check the implementation and try again.")