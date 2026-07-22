# Helper Reference

This document is the focused reference for `src/helper.py`.

Use it when you need:

- the household data transformation contract
- the experiment-log evaluation helpers
- the log-flattening utilities used by SB3 and DT workflows
- the episode visualization helpers

For repo structure and workflow context, start with [README.md](README.md), [architecture.md](architecture.md), and [development.md](development.md).

## What This Module Owns

`src/helper.py` is mostly a shared utility module for:

- household dataset transformation
- environment factory helpers
- flattening rollout logs into DT-compatible parquet tables
- experiment evaluation and comparison metrics
- plotting and episode inspection utilities

It is not the source of truth for environment dynamics. For those details, use:

- [household/environment.md](household/environment.md)
- [aemo/environment.md](aemo/environment.md)

## Main Helpers

### `transform_polars_df`

Transforms Ausgrid-style household data into the `SolarBatteryEnv` schema.

Key output columns:

- `Timestamp`
- `SolarGen`
- `HouseLoad`
- `FutureSolar`
- `FutureLoad`
- `ImportEnergyPrice`
- `ExportEnergyPrice`
- `Time`

Use this when preparing raw household CSVs for environment or baseline runs.

### `make_env`

Returns an environment factory for creating household `SolarBatteryEnv` instances.

This is primarily useful for vectorized RL training or helper code that expects callables rather than instantiated environments.

### `flatten_episode_data`

Flattens rollout trajectories into a single Polars DataFrame.

This is one of the main bridges between rollout collection and offline DT training. It is typically used after SB3 or other batched rollouts when you want parquet logs with a consistent schema.

### `evaluate_experiment_logs`

Computes metrics for one experiment represented as a list of episode DataFrames.

It supports:

- reward summary statistics
- risk metrics such as Sharpe, Sortino, VaR, and CVaR
- recommended RTG and return-scale heuristics
- operational metrics extracted from `info`
- AEMO revenue and dispatch metrics when present

### `evaluate_experiments`

Runs evaluation across multiple named experiments and returns a comparison table.

Use this for benchmark summaries and cross-policy comparisons.

### `bootstrap_confidence_intervals`

Resamples episode logs to produce uncertainty intervals for a chosen metric.

Use this when you need error bars or a more formal comparison than raw mean values.

### `evaluate_by_conditions`

Computes conditional mean reward using custom predicates over observation, info, action, reward, and step index.

This is useful for targeted analysis such as peak-price behavior, high-SoC behavior, or dispatch windows.

## Episode Log Expectations

The evaluation helpers assume episode logs are Polars DataFrames with columns such as:

- `reward`
- `action`
- `raw_observation` optionally
- `norm_observation` optionally
- `info` optionally

### Common `info` keys for household runs

- `grid_energy`
- `step_degradation`
- `battery_flow_energy`
- `battery_level`

### Common `info` keys for AEMO runs

- `energy_revenue`
- `fcas_revenue`
- `total_revenue`
- `degradation_cost`
- `battery_dispatch`
- `actual_energy`
- `battery_soc`

Missing keys are handled defensively. Metrics that depend on unavailable fields fall back to zero or remain absent rather than crashing.

## Episode Visualizer

`EpisodeVisualizer` is the main plotting surface for inspecting a single episode or comparing two episodes.

It supports both environments and auto-detects the environment type from log fields when possible.

### Typical uses

- inspect a single policy over a 24 to 48 hour window
- compare two policies on the same episode slice
- aggregate longer episodes into daily or multi-day summaries

### Key methods

- `.plot(...)`: per-step or short-horizon view
- `.compare(...)`: overlay two agents on the same interval
- `.plot_long_horizon(...)`: aggregated multi-period economics and SoC view

## Notes On AEMO Revenue Interpretation

For corrected AEMO runs, `fcas_revenue` should reflect FCAS enablement in MW, not energy in MWh. Older logs or plots created before that correction can overstate gross and net revenue.

If you are validating historical figures, confirm which revenue model was in use before comparing across runs.

## Where This Fits In The Repo

- Household preprocessing starts here.
- AEMO evaluation often ends here.
- Offline RL dataset preparation often passes through `flatten_episode_data`.

For the broader codebase map, use [architecture.md](architecture.md).