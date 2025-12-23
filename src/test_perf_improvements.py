#!/usr/bin/env python3
"""
Simple benchmark script to verify performance improvements.
Run with: python src/test_perf_improvements.py
"""

import time
import numpy as np
import polars as pl
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from EnergySimEnv import SolarBatteryEnv
from batterydeg import static_degradation, nCL_Id, nCL_Ich, nCL_SoC_DoD
from quantile_scenarios import QuantileScenarioGenerator


def benchmark_env_observation(n_steps=1000):
    """Benchmark environment observation computation."""
    # Create a simple test DataFrame
    n_rows = n_steps + 100
    np.random.seed(42)
    import datetime as dt
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
                          init_battery_level=3.5, max_step=n_steps)
    
    # Benchmark observation generation
    env.reset()
    start = time.perf_counter()
    for _ in range(n_steps):
        obs = env._get_observation_components()
    duration = time.perf_counter() - start
    
    print(f"Environment observation benchmark ({n_steps} calls):")
    print(f"  Total time: {duration:.4f}s")
    print(f"  Per-call time: {duration/n_steps*1000:.4f}ms")
    return duration


def benchmark_degradation(n_calls=10000):
    """Benchmark battery degradation calculation."""
    np.random.seed(42)
    Id_vals = np.random.uniform(0.0, 0.5, n_calls)
    Ich_vals = np.random.uniform(0.0, 0.5, n_calls)
    SoC_vals = np.random.uniform(0.0, 100.0, n_calls)
    DoD_vals = np.random.uniform(1.0, 100.0, n_calls)
    
    start = time.perf_counter()
    for i in range(n_calls):
        _ = static_degradation(Id_vals[i], Ich_vals[i], SoC_vals[i], DoD_vals[i])
    duration = time.perf_counter() - start
    
    print(f"Degradation calculation benchmark ({n_calls} calls):")
    print(f"  Total time: {duration:.4f}s")
    print(f"  Per-call time: {duration/n_calls*1000:.4f}ms")
    return duration


def benchmark_quantile_scenarios(n_rows=1000):
    """Benchmark quantile scenario generation."""
    np.random.seed(42)
    df = pl.DataFrame({
        'SolarGen': np.random.gamma(2, 2, n_rows),
        'HouseLoad': np.random.normal(5, 1.5, n_rows),
        'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),
        'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),
    })
    
    generator = QuantileScenarioGenerator(n_scenarios=5)
    
    start = time.perf_counter()
    result = generator.generate_time_step_scenarios(df)
    duration = time.perf_counter() - start
    
    print(f"Quantile scenario generation benchmark ({n_rows} rows, 5 scenarios):")
    print(f"  Total time: {duration:.4f}s")
    return duration


def main():
    print("=" * 60)
    print("Performance Improvements Benchmark")
    print("=" * 60)
    print()
    
    benchmark_env_observation(n_steps=1000)
    print()
    
    benchmark_degradation(n_calls=10000)
    print()
    
    benchmark_quantile_scenarios(n_rows=1000)
    print()
    
    print("=" * 60)
    print("Benchmark complete!")
    print()
    print("Key optimizations applied:")
    print("1. EnergySimEnv: Batched min/max queries and vectorized normalization")
    print("2. batterydeg: Pre-computed constant denominators")
    print("3. quantile_scenarios: Batched quantile computation")
    print("=" * 60)


if __name__ == "__main__":
    main()
