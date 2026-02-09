# Helper Utilities (src/helper.py)

This document explains the data transformation and evaluation helpers used across the project. The functions are designed to work with both SolarBatteryEnv and AEMOBatteryTradingEnv episode logs.

## Overview

The helper module provides:
- Data transformation for household solar data.
- Convenience utilities to build environments and flatten logs.
- Evaluation metrics for rewards, risk, grid flows, and degradation.
- AEMO trading metrics such as revenue, degradation cost, and dispatch energy.
- Action distribution comparisons and temporal analysis across algorithms.

## Log Assumptions

Episode logs are expected to be Polars DataFrames with these columns:
- reward: per-step reward values.
- action: per-step action values (scalar or list-like).
- raw_observation: raw observation vector (optional).
- info: per-step info dict or JSON string (optional).

The evaluation functions auto-detect the following info keys when available:

SolarBatteryEnv keys:
- grid_energy
- step_degradation
- battery_flow_energy

AEMOBatteryTradingEnv keys:
- energy_revenue
- fcas_revenue
- total_revenue
- degradation_cost
- battery_dispatch
- actual_energy
- battery_soc

Missing keys are handled safely and return 0.0 averages.

## Core Data Helpers

### transform_polars_df
Transforms Ausgrid household data into the SolarBatteryEnv format.

Key outputs:
- Timestamp
- SolarGen
- HouseLoad
- FutureSolar
- FutureLoad
- ImportEnergyPrice
- ExportEnergyPrice
- Time

### make_env
Returns an environment factory that creates SolarBatteryEnv instances.

### flatten_episode_data
Flattens a list of SB3 trajectories into a single Polars DataFrame.

### plot_48h_from_logs
Plots a 48-hour window of battery, solar, load, grid energy, and actions.

## Evaluation Functions

### evaluate_experiment_logs
Computes metrics for a single experiment (list of episode DataFrames).

Reward metrics:
- mean, median, std, percentiles, max
- sharpe_ratio, sortino_ratio
- recommended_rtg and recommended_return_scale

Operational metrics (if present in info):
- avg_grid_import, avg_grid_export, avg_grid_net
- avg_degradation_per_episode, avg_degradation_per_step
- avg_battery_flow_energy, avg_battery_flow_per_step

AEMO trading metrics (if present in info):
- avg_total_revenue_per_episode
- avg_total_degradation_cost_per_episode
- avg_profit_per_episode
- avg_energy_revenue_per_episode
- avg_fcas_revenue_per_episode
- avg_actual_energy_per_episode, avg_actual_energy_per_step
- avg_battery_dispatch_abs_per_episode, avg_battery_dispatch_abs_per_step

### evaluate_experiments
Runs evaluate_experiment_logs for multiple experiments and returns a Polars DataFrame. Optional diagnostic plots:
- Mean reward with std
- Grid energy with degradation overlay
- Risk-return scatter (mean vs std, color by Sharpe)
- Episode return distribution

### evaluate_by_conditions
Computes mean reward under custom conditions. Conditions can accept:
- (obs)
- (obs, info)
- (obs, info, action)
- (obs, info, action, reward)
- (obs, info, action, reward, step_idx)

### compute_decision_divergence
Compares two episode logs and returns action divergence statistics.

## Action Comparison and Temporal Analysis

### AlgorithmActionComparator
Provides action-level comparison across algorithms with:
- action histograms and CDFs
- per-step action profiles (median and IQR)
- pairwise divergence metrics
- action vs SOC scatter

### compare_actions_across_algorithms
Backward compatible wrapper for AlgorithmActionComparator.compare.

### analyze_temporal_actions
Backward compatible wrapper for AlgorithmActionComparator.analyze_temporal.

## Example Usage

```python
import polars as pl
from src.helper import evaluate_experiment_logs, evaluate_experiments, evaluate_by_conditions

ppo_logs = [
    pl.read_parquet("data/ppo_test_episode_01_logs.parquet"),
    pl.read_parquet("data/ppo_test_episode_02_logs.parquet"),
]

metrics = evaluate_experiment_logs(ppo_logs)
print(metrics["avg_profit_per_episode"])  # AEMO logs will populate this

comparison = evaluate_experiments(
    {"ppo": ppo_logs},
    save_dir="eval_output/figures",
    save_format="png",
)

conditions = {
    "high_price": lambda obs, info: info.get("energy_price", 0.0) > 200.0,
    "low_soc": lambda obs: obs[-2] < 0.2,
}
conditional = evaluate_by_conditions(ppo_logs, conditions)
print(conditional)
```
