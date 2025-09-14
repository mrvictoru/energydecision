"""
Integration example showing how to use the new DynamicProgram + SolverOrchestrator
with the existing Agent._solve_sdp workflow.

This script demonstrates the drop-in replacement functionality provided by
the dp_adapter module.
"""

import sys
import os
import numpy as np
import polars as pl

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from decision import Agent
from EnergySimEnv import SolarBatteryEnv
from dp_adapter import run_dp_for_agent


def create_test_environment():
    """Create a simple test environment with synthetic data."""
    # Create synthetic data for testing
    n_rows = 100
    import datetime as dt
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]
    
    df = pl.DataFrame({
        'Time': times,
        'SolarGen': np.abs(np.random.normal(2.0, 1.0, n_rows)),
        'HouseLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
        'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),
        'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),
    })
    
    env = SolarBatteryEnv(
        df, 
        battery_capacity=10.0, 
        max_battery_flow=5.0, 
        init_battery_level=5.0, 
        max_step=50
    )
    
    return env


def demonstrate_original_sdp(agent, forecasts, start_index):
    """Demonstrate the original Agent._solve_sdp method."""
    print("=== Original Agent._solve_sdp ===")
    
    try:
        policy_table_original = agent._solve_sdp(forecasts, start_index=start_index)
        print(f"Original policy table shape: {policy_table_original.shape}")
        print(f"First few policy decisions: {policy_table_original[0, :]}")
        return policy_table_original
    except Exception as e:
        print(f"Original SDP failed: {e}")
        return None


def demonstrate_new_dp_adapter(agent, forecasts, start_index):
    """Demonstrate the new DynamicProgram + SolverOrchestrator via adapter."""
    print("\n=== New DynamicProgram + SolverOrchestrator ===")
    
    try:
        policy_table_new = run_dp_for_agent(agent, forecasts, start_index)
        print(f"New adapter policy table shape: {policy_table_new.shape}")
        print(f"First few policy decisions: {policy_table_new[0, :]}")
        return policy_table_new
    except Exception as e:
        print(f"New DP adapter failed: {e}")
        return None


def compare_results(policy_original, policy_new):
    """Compare the results from both methods."""
    if policy_original is None or policy_new is None:
        print("\nCannot compare - one of the methods failed")
        return
    
    print(f"\n=== Comparison ===")
    print(f"Shape match: {policy_original.shape == policy_new.shape}")
    
    if policy_original.shape == policy_new.shape:
        # Compare first few decisions
        differences = np.sum(policy_original != policy_new)
        total_decisions = policy_original.size
        print(f"Different decisions: {differences} / {total_decisions}")
        print(f"Agreement rate: {(total_decisions - differences) / total_decisions * 100:.1f}%")
        
        # Show a few sample decisions
        print(f"\nSample comparison (first 3 time steps, all SoC levels):")
        for t in range(min(3, policy_original.shape[0])):
            orig = policy_original[t, :]
            new = policy_new[t, :]
            matches = (orig == new).sum()
            print(f"  t={t}: {matches}/{len(orig)} matches")


def main():
    """Main demonstration function."""
    print("DynamicProgram + SolverOrchestrator Integration Demo")
    print("=" * 50)
    
    # Create test environment
    env = create_test_environment()
    
    # Create Agent with SDP algorithm (small settings for fast demo)
    agent = Agent(
        env, 
        algorithm='sdp', 
        horizon=12,  # Small horizon for quick demo
        soc_resolution=5, 
        action_resolution=5,
        use_monte_carlo=False  # Deterministic for consistency
    )
    
    # Get forecasts for the horizon
    current_step = 0
    forecasts = agent._get_forecasts(current_step, agent.horizon)
    
    if not forecasts:
        print("No forecasts available - cannot demonstrate")
        return
    
    print(f"Testing with horizon: {len(forecasts)} steps")
    print(f"SoC grid: {agent.soc_levels_kwh}")
    print(f"Action grid: {agent.action_levels_norm}")
    
    # Demonstrate original method
    policy_original = demonstrate_original_sdp(agent, forecasts, current_step)
    
    # Demonstrate new adapter method
    policy_new = demonstrate_new_dp_adapter(agent, forecasts, current_step)
    
    # Compare results
    compare_results(policy_original, policy_new)
    
    # Show usage example
    print(f"\n=== Usage Example ===")
    print("To integrate with existing Agent code, replace:")
    print("  policy_table = self._solve_sdp(forecasts, start_index=current_step_env)")
    print("With:")
    print("  from src.dp_adapter import run_dp_for_agent")
    print("  policy_table = run_dp_for_agent(self, forecasts, current_step_env)")
    
    print(f"\nDemo completed successfully!")


if __name__ == '__main__':
    main()