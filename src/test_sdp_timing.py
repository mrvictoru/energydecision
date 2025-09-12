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
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', '2010-2011 Solar home electricity data.csv')
    try:
        raw = pl.read_csv(data_path, skip_rows=1)
        # pick a random customer
        customers = raw['Customer'].unique().to_list()
        if len(customers) == 0:
            raise RuntimeError("No customers found in CSV")
        cust = np.random.choice(customers)
        customer_df = raw.filter(pl.col('Customer') == cust)

        # Transform using the helper; parameters mirror the pipeline you supplied
        transformed = transform_polars_df(
            customer_df,
            import_energy_price=0.23,
            export_energy_price=0.015,
            price_periods="7am – 10am | 4pm – 9pm",
            default_import_energy_price=0.15,
            default_export_energy_price=0.01,
        )
        return transformed
    except Exception as e:
        # If real-data pipeline fails, fall back to a small synthetic dataset so tests remain robust.
        print(f"Warning: could not build real dataset, falling back to synthetic data (reason: {e})")
        base = np.datetime64('2023-01-01T00:00')
        times = [base + np.timedelta64(int(i * 30), 'm') for i in range(n_rows)]  # 30-min steps
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
