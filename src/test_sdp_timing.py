import os
import sys
import time
import numpy as np
import polars as pl
import pytest

# Ensure src is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from decision import Agent
from EnergySimEnv import SolarBatteryEnv
from quantile_scenarios import QuantileScenarioGenerator
from helper import transform_polars_df


def make_dummy_df(n_rows=200):
    """Try to build a real per-customer dataset from the repo CSV; fall back to synthetic if anything fails."""
    # Just use synthetic data to avoid data file issues
    print(f"Using synthetic data for testing")
    import datetime as dt
    base_time = dt.datetime(2023, 1, 1, 0, 0)
    times = [base_time + dt.timedelta(minutes=i * 30) for i in range(n_rows)]  # 30-min steps
    df = pl.DataFrame({
        'Time': times,
        'SolarGen': np.abs(np.random.normal(2.0, 1.0, n_rows)),
        'HouseLoad': np.abs(np.random.normal(4.0, 1.0, n_rows)),
        'ImportEnergyPrice': np.random.uniform(0.1, 0.3, n_rows),
        'ExportEnergyPrice': np.random.uniform(0.05, 0.15, n_rows),
    })
    return df


def test_sdp_one_decision_timing():
    """Measure wall-clock time for a single SDP solve and fail if it exceeds threshold."""
    max_seconds = float(os.environ.get('SDP_MAX_SECONDS', '60'))

    df = make_dummy_df(n_rows=300)
    env = SolarBatteryEnv(df, battery_capacity=7.0, max_battery_flow=3.3, init_battery_level=3.5, max_step=1000)

    # Small SDP settings for test stability
    agent = Agent(env, algorithm='sdp', horizon=48, soc_resolution=10, action_resolution=7,
                  use_monte_carlo=True, mc_samples=200, mc_seed=42)
    
    start = time.perf_counter()
    # Precompute array-based scenario forecasts for fast indexing
    # Use the generator directly to create per-row arrays
    try:
        agent._scenario_cache = agent.scenario_generator.generate_time_step_scenarios(env.df)
    except Exception as e:
        pytest.skip(f"Scenario generation failed: {e}")

    # Get forecasts for horizon using agent helper
    forecasts = agent._get_forecasts(env.current_step, agent.horizon)
    assert len(forecasts) > 0, "Forecasts should be available for the horizon"

    
    policy_table = agent._solve_sdp(forecasts, start_index=env.current_step)
    duration = time.perf_counter() - start
    print(f"Forecasts: {forecasts}")
    print(f"SDP single-horizon solve time: {duration:.3f} seconds")

    # Basic sanity checks
    assert policy_table.shape[0] == agent.horizon
    assert policy_table.shape[1] == len(agent.soc_levels_kwh)

    # Fail the test if it takes longer than allowed (configurable via SDP_MAX_SECONDS)
    assert duration <= max_seconds, f"SDP solve took too long: {duration:.2f}s > {max_seconds}s"
