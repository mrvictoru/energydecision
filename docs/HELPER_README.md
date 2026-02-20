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
(Household/SolarBatteryEnv only — for a unified visualiser that works with
both environments, see `EpisodeVisualizer` below.)

### EpisodeVisualizer

Unified class for inspecting agent behaviour during an episode.  Works with
logs from **both** `SolarBatteryEnv` (household) and `AEMOBatteryTradingEnv`
(AEMO).  The environment type is auto-detected from `info` keys
(`battery_level` → household, `battery_soc` → AEMO) or can be forced via
the `env_type` parameter.

#### How it works

`EpisodeVisualizer` takes a single-episode Polars (or pandas) DataFrame —
the same format returned by `Agent.run_episode()` / `AEMOAgent.run_episode()`
— and renders a multi-panel matplotlib figure over a configurable time window:

| Panel              | Household                          | AEMO (simple)          | AEMO (multi-market)         |
|--------------------|------------------------------------|------------------------|-----------------------------|
| **1 – SOC**        | Battery level (kWh) as line        | Battery SOC (MWh)      | Battery SOC (MWh)           |
| **2 – Actions**    | Bar chart (green=charge, red=discharge) | Same                   | Same (energy dispatch)      |
| **3 – Context**    | Solar generation, load & grid      | Energy price (RRP)     | FCAS raise / lower bids     |
| **4 – Price**      | Import / export prices             | *(not shown)*          | Energy price (RRP)          |

#### Constructor

```python
EpisodeVisualizer(
    logs_df,                    # single-episode DataFrame
    step_duration: float = 0.5, # hours per step (default 30 min)
    env_type: str | None = None # 'household', 'aemo', or None (auto-detect)
)
```

#### `.plot()` — single-episode view

```python
fig = vis.plot(
    start_step=0,       # first step to include
    num_hours=48.0,     # length of the time window in hours
    title=None,         # custom figure title
    save_path=None,     # save figure to file
    dpi=150,            # saved image resolution
    figsize=None,       # (width, height) in inches
    show=True,          # call plt.show()
)
```

Returns a `matplotlib.figure.Figure` for programmatic use.

#### `.compare()` — two-agent overlay (static method)

```python
fig = EpisodeVisualizer.compare(
    logs_df1, logs_df2,
    label1="Agent 1", label2="Agent 2",
    start_step=0, num_hours=48.0,
    step_duration=0.5,
    env_type=None,      # auto-detected from logs_df1
    title=None, save_path=None, dpi=150, figsize=None, show=True,
)
```

Overlays SOC traces (line) and action bars (side-by-side colour coding) for
two agents over the same time window.

#### Quick examples

```python
from helper import EpisodeVisualizer

# --- Single agent, 24-hour window starting at step 96 ---
vis = EpisodeVisualizer(episode_df, step_duration=0.5)
vis.plot(start_step=96, num_hours=24, save_path="day2.png")

# --- Compare two agents ---
EpisodeVisualizer.compare(
    rule_episode, rl_episode,
    label1="Rule", label2="SAC",
    num_hours=48, save_path="rule_vs_sac.png",
)
```

## Evaluation Functions

### evaluate_experiment_logs
Computes metrics for a single experiment (list of episode DataFrames).

Reward metrics:
- mean, median, std, percentiles, max
- sharpe_ratio, sortino_ratio
- var_5 (Value-at-Risk at 5% — the 5th percentile of episode returns)
- cvar_5 (Conditional VaR / Expected Shortfall at 5% — mean of returns at or below var_5)
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

### bootstrap_confidence_intervals
Computes bootstrap confidence intervals for a metric across experiments.
Episode logs are resampled with replacement `n_bootstrap` times and the
metric is evaluated on each resample. The default metric is mean total
episode reward.

Parameters:
- `all_logs`: dict mapping experiment name → list of episode DataFrames
- `metric_fn`: callable(logs) → float (default: mean episode reward)
- `n_bootstrap`: number of bootstrap iterations (default 1000)
- `confidence_level`: e.g. 0.95 for a 95% CI
- `seed`: random seed for reproducibility

Returns dict mapping experiment name → `{"mean", "ci_lower", "ci_upper", "std"}`.

```python
from helper import bootstrap_confidence_intervals

cis = bootstrap_confidence_intervals(all_logs, n_bootstrap=1000, confidence_level=0.95, seed=42)
for algo, ci in cis.items():
    print(f"{algo}: mean={ci['mean']:.2f}  95% CI=[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]")
```

### paired_comparison
Paired statistical comparison of two experiments on matched episodes
(same seed / customer index). Uses the Wilcoxon signed-rank test when
scipy is available and there are at least 10 paired episodes.

Parameters:
- `logs_a`, `logs_b`: lists of per-episode DataFrames (should be same length)
- `metric_fn`: callable(episode_df) → float (default: total episode reward)

Returns dict with keys: `mean_diff`, `median_diff`, `std_diff`,
`wilcoxon_stat`, `wilcoxon_p` (NaN when scipy is unavailable or the test
cannot be computed).

```python
from helper import paired_comparison

result = paired_comparison(ppo_logs, sac_logs)
print(f"mean diff = {result['mean_diff']:.4f}, p = {result['wilcoxon_p']:.4f}")
```

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

### Visualising an agent in action

```python
from src.helper import EpisodeVisualizer

# Works with both SolarBatteryEnv and AEMO episode logs
episode_df = ppo_logs[0]  # single episode DataFrame

# View a 24-hour window starting at the beginning
vis = EpisodeVisualizer(episode_df, step_duration=0.5)
vis.plot(start_step=0, num_hours=24)

# Compare two agents over 48 hours
rule_df = rule_logs[0]
EpisodeVisualizer.compare(
    rule_df, episode_df,
    label1="Rule", label2="PPO",
    num_hours=48,
    save_path="eval_output/rule_vs_ppo.png",
)
```
