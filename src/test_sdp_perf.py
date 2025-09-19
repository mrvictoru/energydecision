#!/usr/bin/env python3
"""
MRDP Performance Test Script

This script provides a micro-benchmark for the MRDP performance improvements.
It builds synthetic forecast windows and scenario marginals, then runs two
representative MRDP solves with different configurations to compare:

1. Runtime with and without caching enabled
2. Runtime with vectorized Monte Carlo vs fallback
3. Performance difference between near-term fine resolution and far-term coarse resolution

Run with: python src/test_sdp_perf.py

The script is designed to be runnable without heavy dependencies and provides
timing comparisons with sanity checks on the policy tables produced.
"""

import time
import numpy as np
from typing import Dict, List, Any

# Import the MRDP module
try:
    from sdp_multires import (
        solve_mrdp, 
        DynamicProgram, 
        clear_stage_cost_cache, 
        get_stage_cost_cache_stats,
        vectorized_monte_carlo_stage_cost,
        deterministic_stage_cost
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('.')
    from src.sdp_multires import (
        solve_mrdp, 
        DynamicProgram, 
        clear_stage_cost_cache, 
        get_stage_cost_cache_stats,
        vectorized_monte_carlo_stage_cost,
        deterministic_stage_cost
    )


class MockEnvironment:
    """Mock environment for testing MRDP without full dependencies."""
    
    def __init__(self):
        self.battery_capacity = 10.0  # kWh
        self.max_battery_flow = 5.0   # kW
        self.max_grid_energy = 20.0   # kWh
        self.battery_life_cost = 0.01 # $/kWh degradation cost
        self.use_monte_carlo = True
        self.mc_samples = 100


def create_synthetic_forecasts(horizon: int) -> List[Dict]:
    """
    Create synthetic forecast data for testing.
    
    Args:
        horizon: Number of time steps
        
    Returns:
        List of forecast dictionaries with solar, load, and price data
    """
    np.random.seed(42)  # Reproducible results
    
    forecasts = []
    for t in range(horizon):
        # Diurnal patterns for solar and load
        hour_of_day = (t * 0.5) % 24  # Assuming 30-min steps
        solar_pattern = max(0, np.sin(np.pi * (hour_of_day - 6) / 12)) if 6 <= hour_of_day <= 18 else 0
        load_pattern = 0.3 + 0.4 * (1 + np.sin(np.pi * (hour_of_day - 6) / 12))
        
        # Add some randomness
        solar = solar_pattern * 8.0 + np.random.normal(0, 0.5)  # kWh
        load = load_pattern * 5.0 + np.random.normal(0, 0.3)   # kWh
        
        # Time-of-use pricing
        import_price = 0.15 if 9 <= hour_of_day <= 17 else 0.10  # $/kWh
        export_price = 0.08  # $/kWh
        
        forecasts.append({
            'SolarGen': max(0, solar),
            'HouseLoad': max(0.1, load),
            'ImportEnergyPrice': import_price + np.random.normal(0, 0.01),
            'ExportEnergyPrice': export_price + np.random.normal(0, 0.005)
        })
    
    return forecasts


def create_synthetic_scenarios(forecasts: List[Dict], n_samples: int = 100) -> Dict[str, np.ndarray]:
    """
    Create synthetic scenario marginals for vectorized Monte Carlo testing.
    
    Args:
        forecasts: Base forecast sequence
        n_samples: Number of Monte Carlo samples per time step
        
    Returns:
        Dictionary with scenario arrays for vectorized MC
    """
    horizon = len(forecasts)
    np.random.seed(123)  # Different seed for scenarios
    
    scenarios = {
        'solar': np.zeros((horizon, n_samples)),
        'load': np.zeros((horizon, n_samples)),
        'import_price': np.zeros((horizon, n_samples)),
        'export_price': np.zeros((horizon, n_samples))
    }
    
    for t, forecast in enumerate(forecasts):
        # Generate samples around forecast values with realistic variance
        scenarios['solar'][t] = np.maximum(0, 
            np.random.normal(forecast['SolarGen'], forecast['SolarGen'] * 0.2, n_samples))
        scenarios['load'][t] = np.maximum(0.1,
            np.random.normal(forecast['HouseLoad'], forecast['HouseLoad'] * 0.15, n_samples))
        scenarios['import_price'][t] = np.maximum(0.05,
            np.random.normal(forecast['ImportEnergyPrice'], 0.02, n_samples))
        scenarios['export_price'][t] = np.maximum(0.02,
            np.random.normal(forecast['ExportEnergyPrice'], 0.01, n_samples))
    
    return scenarios


def simple_stage_cost_function(t_global_idx: int, unique_energy_values: np.ndarray, 
                              forecasts: List[Dict], env: MockEnvironment) -> np.ndarray:
    """
    Simple stage cost function for testing (fallback when vectorized MC not used).
    
    Args:
        t_global_idx: Global time index
        unique_energy_values: Array of battery flow energies
        forecasts: Forecast sequence
        env: Mock environment
        
    Returns:
        Array of stage costs
    """
    if t_global_idx < 0 or t_global_idx >= len(forecasts):
        return np.full(len(unique_energy_values), np.inf)
    
    forecast = forecasts[t_global_idx]
    
    # Use deterministic stage cost computation
    costs = deterministic_stage_cost(
        unique_energy_values,
        forecast['SolarGen'],
        forecast['HouseLoad'], 
        forecast['ImportEnergyPrice'],
        forecast['ExportEnergyPrice'],
        env.max_grid_energy,
        env.battery_life_cost
    )
    
    return costs


def run_mrdp_benchmark(use_cache: bool = True, use_vectorized_mc: bool = True) -> Dict[str, Any]:
    """
    Run MRDP benchmark with specified configuration.
    
    Args:
        use_cache: Whether to enable stage cost caching
        use_vectorized_mc: Whether to use vectorized Monte Carlo
        
    Returns:
        Dictionary with timing results and policy info
    """
    print(f"\n--- Running MRDP benchmark (cache={use_cache}, vectorized_mc={use_vectorized_mc}) ---")
    
    # Clear cache before each run for fair comparison
    clear_stage_cost_cache()
    
    # Create test environment and data
    env = MockEnvironment()
    horizon = 48  # 24 hours with 30-min steps
    forecasts = create_synthetic_forecasts(horizon)
    
    # Create representative MRDP spec: near-term fine, far-term coarse
    subhorizon_specs = [
        {
            'start': 0, 'length': 12,  # First 12 steps (6 hours)
            'soc_resolution': 20, 'action_resolution': 11,
            'step_duration': 0.5,
            'mc_samples': 200 if use_vectorized_mc else 20,  # High accuracy for near-term
            'use_monte_carlo': use_vectorized_mc
        },
        {
            'start': 12, 'length': 36,  # Next 36 steps (18 hours)
            'soc_resolution': 10, 'action_resolution': 5,
            'step_duration': 1.0,
            'mc_samples': 20 if use_vectorized_mc else 10,   # Lower accuracy for far-term
            'use_monte_carlo': use_vectorized_mc
        }
    ]
    
    # Create stage cost function
    def stage_cost_fn(t_idx, energies):
        return simple_stage_cost_function(t_idx, energies, forecasts, env)
    
    # Prepare scenarios for vectorized MC if enabled
    sampled_scenarios = None
    if use_vectorized_mc:
        sampled_scenarios = create_synthetic_scenarios(forecasts, n_samples=200)
    
    # Run the benchmark
    start_time = time.perf_counter()
    
    try:
        policy_table, cost_to_go = solve_mrdp(
            env=env,
            forecasts=forecasts,
            subhorizon_specs=subhorizon_specs,
            global_start_index=0,
            stage_cost_function=stage_cost_fn,
            sampled_scenarios=sampled_scenarios,
            use_float64=False,  # Use float32 for speed
            use_cache=use_cache
        )
        
        solve_time = time.perf_counter() - start_time
        success = True
        
        # Sanity checks
        finite_policies = np.sum(policy_table >= 0)
        finite_ctg = np.sum(np.isfinite(cost_to_go))
        policy_shape = policy_table.shape
        ctg_shape = cost_to_go.shape
        
    except Exception as e:
        solve_time = float('inf')
        success = False
        finite_policies = 0
        finite_ctg = 0
        policy_shape = (0, 0)
        ctg_shape = (0, 0)
        print(f"Error during solve: {e}")
    
    # Get cache statistics
    cache_stats = get_stage_cost_cache_stats()
    
    return {
        'success': success,
        'solve_time': solve_time,
        'policy_shape': policy_shape,
        'ctg_shape': ctg_shape,
        'finite_policies': finite_policies,
        'finite_ctg': finite_ctg,
        'cache_stats': cache_stats
    }


def test_vectorized_monte_carlo():
    """Test the vectorized Monte Carlo functions directly."""
    print("\n--- Testing Vectorized Monte Carlo Functions ---")
    
    # Test data
    unique_energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    n_samples = 1000
    
    # Synthetic scenario samples - more realistic values
    np.random.seed(456)
    sampled_solar = np.random.exponential(1.5, n_samples)  # Smaller solar values
    sampled_load = np.random.gamma(2.0, 1.0, n_samples)    # Smaller load values
    sampled_imp = np.random.uniform(0.10, 0.20, n_samples)
    sampled_exp = np.random.uniform(0.05, 0.10, n_samples)
    
    max_grid_energy = 15.0
    degradation_cost = 0.01
    
    print(f"Sample data ranges:")
    print(f"  Solar: {np.min(sampled_solar):.2f} to {np.max(sampled_solar):.2f}, mean: {np.mean(sampled_solar):.2f}")
    print(f"  Load: {np.min(sampled_load):.2f} to {np.max(sampled_load):.2f}, mean: {np.mean(sampled_load):.2f}")
    print(f"  Grid limit: {max_grid_energy}")
    
    # Test vectorized MC
    start_time = time.perf_counter()
    mc_costs = vectorized_monte_carlo_stage_cost(
        unique_energies, sampled_solar, sampled_load, sampled_imp, sampled_exp,
        max_grid_energy, degradation_cost
    )
    mc_time = time.perf_counter() - start_time
    
    # Test deterministic computation
    start_time = time.perf_counter()
    det_costs = deterministic_stage_cost(
        unique_energies, 
        np.mean(sampled_solar), np.mean(sampled_load),
        np.mean(sampled_imp), np.mean(sampled_exp),
        max_grid_energy, degradation_cost
    )
    det_time = time.perf_counter() - start_time
    
    print(f"Vectorized MC time: {mc_time:.6f}s")
    print(f"Deterministic time: {det_time:.6f}s")
    print(f"MC costs: {mc_costs}")
    print(f"Deterministic costs: {det_costs}")
    print(f"Cost differences: {mc_costs - det_costs}")
    
    # Check for feasibility issues
    for i, energy in enumerate(unique_energies):
        battery_charge = max(0, energy)
        battery_discharge = max(0, -energy)
        mean_grid = np.mean(sampled_load) + battery_charge - np.mean(sampled_solar) - battery_discharge
        print(f"Energy {energy}: mean grid demand = {mean_grid:.2f}")
    
    return mc_costs, det_costs


def main():
    """Main performance test routine."""
    print("MRDP Performance Test Script")
    print("=" * 50)
    
    # Test vectorized functions
    mc_costs, det_costs = test_vectorized_monte_carlo()
    
    # Run benchmarks with different configurations
    configurations = [
        {'use_cache': False, 'use_vectorized_mc': False},  # Baseline
        {'use_cache': True, 'use_vectorized_mc': False},   # Cache only
        {'use_cache': False, 'use_vectorized_mc': True},   # Vectorized MC only  
        {'use_cache': True, 'use_vectorized_mc': True},    # All optimizations
    ]
    
    results = {}
    for config in configurations:
        config_name = f"cache={config['use_cache']}, vec_mc={config['use_vectorized_mc']}"
        results[config_name] = run_mrdp_benchmark(**config)
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    baseline_time = None
    for config_name, result in results.items():
        if result['success']:
            time_str = f"{result['solve_time']:.3f}s"
            if baseline_time is None:
                baseline_time = result['solve_time']
                speedup = "1.0x (baseline)"
            else:
                speedup = f"{baseline_time / result['solve_time']:.1f}x"
            
            cache_stats = result['cache_stats']
            cache_ratio = cache_stats['hits'] / max(1, cache_stats['hits'] + cache_stats['misses'])
            
            print(f"{config_name:30} | {time_str:>8} | {speedup:>10} | "
                  f"cache_hit_rate: {cache_ratio:.1%} | "
                  f"policies: {result['finite_policies']:>4}/{result['policy_shape'][0] * result['policy_shape'][1]:>4}")
        else:
            print(f"{config_name:30} | {'FAILED':>8} | {'--':>10}")
    
    print("\nKey Findings:")
    print("- Caching should improve performance for repeated energy values")
    print("- Vectorized MC should be faster for high sample counts")
    print("- All optimizations combined should provide best performance")
    
    print(f"\nFinal cache statistics: {get_stage_cost_cache_stats()}")
    
    print("\nTo integrate with Agent:")
    print("1. Call clear_stage_cost_cache() between episodes for K-step recompute")
    print("2. Pass pre-sampled scenarios via sampled_scenarios parameter")
    print("3. Configure per-subhorizon mc_samples: high for near-term, low for far-term")
    print("4. Use use_float64=False for memory/speed improvements")


if __name__ == "__main__":
    main()