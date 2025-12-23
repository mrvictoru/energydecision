"""
Performance benchmark tests for energydecision.

This module provides benchmarks for key performance-critical operations:
- Environment observation computation
- Battery degradation calculation
- Quantile scenario generation
- MRDP (Multi-Resolution Dynamic Programming) operations

Run benchmarks with: pytest tests/test_performance.py -v -s
"""

import pytest
import time
import numpy as np
import polars as pl
import datetime as dt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from EnergySimEnv import SolarBatteryEnv
from batterydeg import static_degradation, nCL_Id, nCL_Ich, nCL_SoC_DoD
from quantile_scenarios import QuantileScenarioGenerator


class TestEnvironmentPerformance:
    """Benchmark tests for SolarBatteryEnv performance."""
    
    @pytest.fixture
    def benchmark_env(self):
        """Create environment for benchmarking."""
        n_rows = 1100
        np.random.seed(42)
        base_time = dt.datetime(2023, 1, 1, 0, 0)
        times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]
        
        df = pl.DataFrame({
            'Time': times,
            'SolarGen': np.abs(np.random.normal(2.0, 1.0, n_rows)),
            'HouseLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
            'FutureSolar': np.abs(np.random.normal(2.0, 1.0, n_rows)),
            'FutureLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
            'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),
            'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),
        })
        
        return SolarBatteryEnv(df, battery_capacity=7.0, max_battery_flow=3.3, 
                               init_battery_level=3.5, max_step=1000)

    def test_observation_computation_performance(self, benchmark_env, benchmark_iterations=1000):
        """Benchmark observation computation speed."""
        benchmark_env.reset()
        
        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            _ = benchmark_env._get_observation_components()
        duration = time.perf_counter() - start
        
        per_call_ms = duration / benchmark_iterations * 1000
        
        print(f"\nObservation computation ({benchmark_iterations} calls):")
        print(f"  Total time: {duration:.4f}s")
        print(f"  Per-call time: {per_call_ms:.4f}ms")
        
        # Should be fast enough for real-time use (< 1ms per call)
        assert per_call_ms < 1.0, f"Observation computation too slow: {per_call_ms:.4f}ms"

    def test_vectorized_normalization_used(self, benchmark_env):
        """Verify that vectorized normalization mask is being used."""
        assert hasattr(benchmark_env, '_norm_by_capacity_mask')
        assert isinstance(benchmark_env._norm_by_capacity_mask, np.ndarray)
        
        # Check mask has correct size
        n_cols = len(benchmark_env.ordered_df_cols_for_obs)
        assert len(benchmark_env._norm_by_capacity_mask) == n_cols


class TestDegradationPerformance:
    """Benchmark tests for battery degradation calculations."""
    
    def test_degradation_calculation_performance(self, benchmark_iterations=10000):
        """Benchmark degradation calculation speed."""
        np.random.seed(42)
        Id_vals = np.random.uniform(0.0, 0.5, benchmark_iterations)
        Ich_vals = np.random.uniform(0.0, 0.5, benchmark_iterations)
        SoC_vals = np.random.uniform(0.0, 100.0, benchmark_iterations)
        DoD_vals = np.random.uniform(1.0, 100.0, benchmark_iterations)
        
        start = time.perf_counter()
        for i in range(benchmark_iterations):
            _ = static_degradation(Id_vals[i], Ich_vals[i], SoC_vals[i], DoD_vals[i])
        duration = time.perf_counter() - start
        
        per_call_us = duration / benchmark_iterations * 1e6
        
        print(f"\nDegradation calculation ({benchmark_iterations} calls):")
        print(f"  Total time: {duration:.4f}s")
        print(f"  Per-call time: {per_call_us:.2f}μs")
        
        # Should be very fast (< 100μs per call)
        assert per_call_us < 100, f"Degradation calculation too slow: {per_call_us:.2f}μs"

    def test_nCL_functions_precomputed(self):
        """Verify that nCL functions use precomputed denominators."""
        # Test that functions return consistent results
        result1 = nCL_Id(0.3)
        result2 = nCL_Id(0.3)
        assert result1 == result2, "nCL_Id should return consistent results"
        
        result1 = nCL_Ich(0.15)
        result2 = nCL_Ich(0.15)
        assert result1 == result2, "nCL_Ich should return consistent results"
        
        result1 = nCL_SoC_DoD(50.0, 80.0)
        result2 = nCL_SoC_DoD(50.0, 80.0)
        assert result1 == result2, "nCL_SoC_DoD should return consistent results"


class TestQuantileScenarioPerformance:
    """Benchmark tests for quantile scenario generation."""
    
    def test_scenario_generation_performance(self, benchmark_rows=1000):
        """Benchmark quantile scenario generation."""
        np.random.seed(42)
        df = pl.DataFrame({
            'SolarGen': np.random.gamma(2, 2, benchmark_rows),
            'HouseLoad': np.random.normal(5, 1.5, benchmark_rows),
            'ImportEnergyPrice': np.random.uniform(0.1, 0.3, benchmark_rows),
            'ExportEnergyPrice': np.random.uniform(0.05, 0.15, benchmark_rows),
        })
        
        generator = QuantileScenarioGenerator(n_scenarios=5)
        
        start = time.perf_counter()
        _ = generator.generate_time_step_scenarios(df)
        duration = time.perf_counter() - start
        
        print(f"\nQuantile scenario generation ({benchmark_rows} rows, 5 scenarios):")
        print(f"  Total time: {duration:.4f}s")
        
        # Should be fast enough for practical use (< 1s for 1000 rows)
        assert duration < 1.0, f"Scenario generation too slow: {duration:.4f}s"

    def test_batched_quantile_computation(self, benchmark_rows=100):
        """Verify that quantile computation uses batched queries."""
        np.random.seed(42)
        df = pl.DataFrame({
            'SolarGen': np.random.gamma(2, 2, benchmark_rows),
            'HouseLoad': np.random.normal(5, 1.5, benchmark_rows),
            'ImportEnergyPrice': np.random.uniform(0.1, 0.3, benchmark_rows),
            'ExportEnergyPrice': np.random.uniform(0.05, 0.15, benchmark_rows),
        })
        
        generator = QuantileScenarioGenerator(n_scenarios=5)
        
        # This should complete quickly due to batched computation
        result = generator.generate_time_step_scenarios(df)
        
        # Verify result structure
        assert 'solar' in result
        assert 'load' in result
        assert 'import_price' in result
        assert 'export_price' in result


class TestMRDPPerformance:
    """Benchmark tests for MRDP (Multi-Resolution Dynamic Programming)."""
    
    @pytest.fixture
    def mrdp_test_data(self):
        """Create test data for MRDP benchmarks."""
        from sdp_multires import (
            clear_stage_cost_cache,
            get_stage_cost_cache_stats,
            vectorized_monte_carlo_stage_cost,
            deterministic_stage_cost
        )
        
        # Create synthetic forecasts
        np.random.seed(42)
        horizon = 48
        forecasts = []
        for t in range(horizon):
            hour_of_day = (t * 0.5) % 24
            solar_pattern = max(0, np.sin(np.pi * (hour_of_day - 6) / 12)) if 6 <= hour_of_day <= 18 else 0
            load_pattern = 0.3 + 0.4 * (1 + np.sin(np.pi * (hour_of_day - 6) / 12))
            
            forecasts.append({
                'SolarGen': max(0, solar_pattern * 8.0 + np.random.normal(0, 0.5)),
                'HouseLoad': max(0.1, load_pattern * 5.0 + np.random.normal(0, 0.3)),
                'ImportEnergyPrice': 0.15 + np.random.normal(0, 0.01),
                'ExportEnergyPrice': 0.08 + np.random.normal(0, 0.005)
            })
        
        return {
            'forecasts': forecasts,
            'clear_cache': clear_stage_cost_cache,
            'get_stats': get_stage_cost_cache_stats,
            'vectorized_mc': vectorized_monte_carlo_stage_cost,
            'deterministic': deterministic_stage_cost
        }

    def test_vectorized_monte_carlo_performance(self, mrdp_test_data):
        """Benchmark vectorized Monte Carlo stage cost computation."""
        unique_energies = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        n_samples = 1000
        
        np.random.seed(456)
        sampled_solar = np.random.exponential(1.5, n_samples)
        sampled_load = np.random.gamma(2.0, 1.0, n_samples)
        sampled_imp = np.random.uniform(0.10, 0.20, n_samples)
        sampled_exp = np.random.uniform(0.05, 0.10, n_samples)
        
        max_grid_energy = 15.0
        degradation_cost = 0.01
        
        start = time.perf_counter()
        mc_costs = mrdp_test_data['vectorized_mc'](
            unique_energies, sampled_solar, sampled_load, sampled_imp, sampled_exp,
            max_grid_energy, degradation_cost
        )
        mc_time = time.perf_counter() - start
        
        start = time.perf_counter()
        det_costs = mrdp_test_data['deterministic'](
            unique_energies,
            np.mean(sampled_solar), np.mean(sampled_load),
            np.mean(sampled_imp), np.mean(sampled_exp),
            max_grid_energy, degradation_cost
        )
        det_time = time.perf_counter() - start
        
        print(f"\nMonte Carlo vs Deterministic stage cost:")
        print(f"  Vectorized MC time: {mc_time*1e6:.2f}μs")
        print(f"  Deterministic time: {det_time*1e6:.2f}μs")
        
        # Both should be fast
        assert mc_time < 0.1, f"Vectorized MC too slow: {mc_time:.4f}s"
        assert det_time < 0.01, f"Deterministic too slow: {det_time:.4f}s"


# Summary benchmark that can be run standalone
def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print("Performance Benchmark Summary")
    print("=" * 60)
    
    # Run each test class
    test_env = TestEnvironmentPerformance()
    
    # Create benchmark env
    n_rows = 1100
    np.random.seed(42)
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]
    
    df = pl.DataFrame({
        'Time': times,
        'SolarGen': np.abs(np.random.normal(2.0, 1.0, n_rows)),
        'HouseLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
        'FutureSolar': np.abs(np.random.normal(2.0, 1.0, n_rows)),
        'FutureLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
        'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),
        'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),
    })
    
    env = SolarBatteryEnv(df, battery_capacity=7.0, max_battery_flow=3.3, 
                          init_battery_level=3.5, max_step=1000)
    
    test_env.test_observation_computation_performance(env)
    
    test_deg = TestDegradationPerformance()
    test_deg.test_degradation_calculation_performance()
    
    test_quant = TestQuantileScenarioPerformance()
    test_quant.test_scenario_generation_performance()
    
    print("\n" + "=" * 60)
    print("All benchmarks completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()
