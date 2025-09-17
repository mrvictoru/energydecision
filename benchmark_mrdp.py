"""
Benchmark script for comparing MRDP performance with and without optimizations.

This script measures per-step decision time for:
- Original implementation vs optimized MRDP module
- Two sub-horizons: near (12 steps, mc_samples=200) and far (36 steps, mc_samples=20)
- Averaged over 20 agent steps

Usage:
    python benchmark_mrdp.py
"""

import time
import numpy as np
import polars as pl
import sys
import os
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from decision import Agent
from EnergySimEnv import SolarBatteryEnv
from sdp_multires import (
    DynamicProgram, 
    compute_unique_energy_costs_vectorized,
    clear_stage_cost_cache
)


def create_synthetic_forecast(n_rows: int = 200) -> pl.DataFrame:
    """Create synthetic forecast data for benchmarking."""
    import datetime as dt
    
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]
    
    np.random.seed(42)  # For reproducible benchmarks
    df = pl.DataFrame({
        'Time': times,
        'SolarGen': np.abs(np.random.normal(2.0, 1.0, n_rows)),
        'HouseLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
        'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),
        'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),
    })
    return df


def benchmark_original_agent(
    env: SolarBatteryEnv, 
    num_steps: int = 20,
    horizon: int = 48
) -> Dict[str, float]:
    """Benchmark the original Agent SDP implementation."""
    agent = Agent(
        env, 
        algorithm='sdp', 
        horizon=horizon,
        soc_resolution=10, 
        action_resolution=7,
        use_monte_carlo=True, 
        mc_samples=200, 
        mc_seed=42
    )
    
    # Warm up
    try:
        agent._scenario_cache = agent.scenario_generator.generate_time_step_scenarios(env.df)
    except Exception:
        agent._scenario_cache = None
    
    step_times = []
    
    for step in range(num_steps):
        env.current_step = step
        forecasts = agent._get_forecasts(step, horizon)
        
        start_time = time.perf_counter()
        policy_table = agent._solve_sdp(forecasts, start_index=step)
        end_time = time.perf_counter()
        
        step_times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(step_times),
        'std_time': np.std(step_times),
        'min_time': np.min(step_times),
        'max_time': np.max(step_times),
        'total_time': np.sum(step_times)
    }


def benchmark_mrdp_optimized(
    env: SolarBatteryEnv,
    num_steps: int = 20,
    subhorizon_specs: List[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Benchmark the optimized MRDP implementation."""
    if subhorizon_specs is None:
        subhorizon_specs = [
            {'horizon': 12, 'mc_samples': 200},  # Near: high accuracy
            {'horizon': 36, 'mc_samples': 20}    # Far: lower accuracy  
        ]
    
    total_horizon = sum(spec['horizon'] for spec in subhorizon_specs)
    
    # Create agent for parameters
    agent = Agent(
        env, 
        algorithm='sdp',
        horizon=total_horizon,
        soc_resolution=10,
        action_resolution=7,
        use_monte_carlo=True,
        mc_samples=200,
        mc_seed=42
    )
    
    # Create optimized DP solver
    dp = DynamicProgram(
        soc_levels=agent.soc_levels_kwh,
        action_levels=agent.action_levels_norm,
        step_duration=agent.step_duration,
        max_battery_flow=agent.max_battery_flow,
        battery_capacity=agent.battery_capacity
    )
    
    # Create cached vectorized stage cost function
    stage_cost_cache = {}
    
    def cached_vectorized_stage_cost(timestep, unique_energies, monte_samples, 
                                   step_duration, max_grid_energy, degradation_params):
        cached_results = {}
        uncached_energies = []
        uncached_indices = []
        
        # Check cache
        for i, energy in enumerate(unique_energies):
            rounded_energy = round(float(energy), 10)
            cache_key = (timestep, rounded_energy)
            if cache_key in stage_cost_cache:
                cached_results[i] = stage_cost_cache[cache_key]
            else:
                uncached_energies.append(energy)
                uncached_indices.append(i)
        
        # Compute uncached values using vectorized function
        if uncached_energies:
            uncached_costs = compute_unique_energy_costs_vectorized(
                monte_samples, np.array(uncached_energies), step_duration,
                max_grid_energy, degradation_params
            )
            
            # Cache results
            for idx, energy_idx in enumerate(uncached_indices):
                energy = unique_energies[energy_idx]
                rounded_energy = round(float(energy), 10)
                cache_key = (timestep, rounded_energy)
                stage_cost_cache[cache_key] = uncached_costs[idx]
                cached_results[energy_idx] = uncached_costs[idx]
        
        # Reconstruct full results
        return np.array([cached_results[i] for i in range(len(unique_energies))])
    
    step_times = []
    
    for step in range(num_steps):
        env.current_step = step
        forecasts = agent._get_forecasts(step, total_horizon)
        
        start_time = time.perf_counter()
        policy_table = dp.solve(
            forecasts=forecasts,
            stage_cost_function=cached_vectorized_stage_cost,
            subhorizon_specs=subhorizon_specs,
            use_cache=True,
            start_index=step
        )
        end_time = time.perf_counter()
        
        step_times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(step_times),
        'std_time': np.std(step_times),
        'min_time': np.min(step_times),
        'max_time': np.max(step_times),
        'total_time': np.sum(step_times),
        'cache_size': len(stage_cost_cache)
    }


def run_benchmark():
    """Run complete benchmark comparison."""
    print("MRDP Performance Benchmark")
    print("=" * 50)
    
    # Create synthetic environment
    df = create_synthetic_forecast(n_rows=300)
    env = SolarBatteryEnv(
        df, 
        battery_capacity=7.0, 
        max_battery_flow=3.3, 
        init_battery_level=3.5, 
        max_step=1000
    )
    
    num_test_steps = 20
    print(f"Running {num_test_steps} decision steps for each method...\n")
    
    # Clear any existing cache
    clear_stage_cost_cache()
    
    # Benchmark original implementation
    print("Benchmarking Original SDP Agent...")
    original_results = benchmark_original_agent(env, num_steps=num_test_steps, horizon=48)
    
    # Benchmark optimized MRDP
    print("Benchmarking Optimized MRDP...")
    subhorizon_specs = [
        {'horizon': 12, 'mc_samples': 200},  # Near: 12 steps, high accuracy
        {'horizon': 36, 'mc_samples': 20}    # Far: 36 steps, lower accuracy
    ]
    mrdp_results = benchmark_mrdp_optimized(env, num_steps=num_test_steps, 
                                          subhorizon_specs=subhorizon_specs)
    
    # Print results
    print("\nBenchmark Results:")
    print("-" * 30)
    
    print("Original SDP Agent:")
    print(f"  Mean time per step: {original_results['mean_time']:.4f} seconds")
    print(f"  Std deviation:      {original_results['std_time']:.4f} seconds")
    print(f"  Min time:           {original_results['min_time']:.4f} seconds")
    print(f"  Max time:           {original_results['max_time']:.4f} seconds")
    print(f"  Total time:         {original_results['total_time']:.4f} seconds")
    
    print("\nOptimized MRDP:")
    print(f"  Mean time per step: {mrdp_results['mean_time']:.4f} seconds")
    print(f"  Std deviation:      {mrdp_results['std_time']:.4f} seconds")
    print(f"  Min time:           {mrdp_results['min_time']:.4f} seconds")
    print(f"  Max time:           {mrdp_results['max_time']:.4f} seconds")
    print(f"  Total time:         {mrdp_results['total_time']:.4f} seconds")
    print(f"  Cache entries:      {mrdp_results['cache_size']}")
    
    # Performance comparison
    speedup = original_results['mean_time'] / mrdp_results['mean_time']
    time_saved = original_results['total_time'] - mrdp_results['total_time']
    
    print(f"\nPerformance Improvement:")
    print(f"  Speedup factor:     {speedup:.2f}x")
    print(f"  Time saved:         {time_saved:.4f} seconds ({time_saved/original_results['total_time']*100:.1f}%)")
    
    # Test configuration details
    print(f"\nTest Configuration:")
    print(f"  Sub-horizon specs:  {subhorizon_specs}")
    print(f"  SoC resolution:     10 levels")
    print(f"  Action resolution:  7 levels")
    print(f"  Test steps:         {num_test_steps}")
    
    return original_results, mrdp_results


if __name__ == "__main__":
    try:
        original, optimized = run_benchmark()
        
        print("\nBenchmark completed successfully!")
        
        # Basic validation
        if optimized['mean_time'] < original['mean_time']:
            print("✅ MRDP optimization shows performance improvement!")
        else:
            print("⚠️  MRDP optimization may need further tuning.")
            
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()