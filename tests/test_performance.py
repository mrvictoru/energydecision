"""
Performance smoke tests for degradation (critical path).
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from batterydeg import static_degradation


def test_degradation_performance(benchmark_iterations=10000):
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

    assert per_call_us < 100, f"Degradation too slow: {per_call_us:.2f}μs"
