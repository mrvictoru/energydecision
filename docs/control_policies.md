# Control policies reference

This document describes the software control-policy wrappers in `src/decision.py`: `Agent` and `AEMOAgent`. It is not a guide to AI coding assistants.

Use this document when you need:

- the role of `Agent` and `AEMOAgent`
- the supported algorithm modes
- the logging behaviour of episode execution
- dispatch replay and Decision Transformer inference entrypoints

For the repository-wide system view, see [architecture.md](architecture.md). For workflow entrypoints, see [development.md](development.md) and [aemo/workflow.md](aemo/workflow.md).

## Main control-policy wrappers

### `Agent`

`Agent` is the general control wrapper for the household track and shared logic.

Supported modes include:

- `rule`
- `sdp`
- `mrdp`
- `oracle`
- `rl`
- `dt`

Main responsibilities:

- choose actions based on the selected algorithm
- manage DT rollout buffers for inference
- run episodes and emit structured logs
- bridge solver-style and policy-style controllers into a common interface

### `AEMOAgent`

`AEMOAgent` is the grid-scale counterpart for `AEMOBatteryTradingEnv`.

Supported modes include:

- `rule`
- `fcas_rule`
- `dispatch`
- `rl`
- `dt`

Main responsibilities:

- interpret AEMO observations
- produce energy-only or FCAS-aware actions depending on `action_mode`
- replay historical dispatch trajectories
- support Decision Transformer inference on AEMO state/action shapes

## Common execution pattern

Both wrappers expose `run_episode(...)` as the main data-collection surface. The typical outcome is:

- an episode log DataFrame
- an incident or auxiliary DataFrame depending on the environment

Those logs are later consumed by helper functions in [HELPER_README.md](HELPER_README.md), saved to parquet, or converted into DT datasets.

## Decision Transformer inference

DT-backed modes maintain rolling buffers of:

- states
- actions
- RTG values
- timesteps

These are padded and forwarded to `DecisionTransformer.get_action(...)` during rollout.

Important operational details:

- DT dimensions must match the environment observation and action spaces
- AEMO runs must keep `act_dim` aligned with the configured `action_mode`
- RTG prompting behaviour depends on `rtg_value` and `dt_gamma`

## Dispatch replay

`AEMOAgent` owns the wrapper-side portion of dispatch replay.

Key supporting surfaces:

- `set_dispatch_data(...)`
- `_build_dispatch_actions(...)`
- `_dispatch_action()`

The usual workflow is:

1. resolve a station or DUID using AEMO data and dispatch helpers
2. build a replay action trace aligned to the environment timestep
3. run the episode through `AEMOAgent(algorithm='dispatch', ...)`

For the broader replay workflow, see [aemo/dispatch-replay.md](aemo/dispatch-replay.md).

## Algorithm notes

### Rule-based modes

- Household `rule` reacts to solar, load, and battery state.
- AEMO `rule` reacts mainly to price thresholds and SoC.
- AEMO `fcas_rule` extends this with percentile-based FCAS bidding.

### Planning modes

- `sdp`, `mrdp`, and `oracle` are exposed through `Agent`.
- They depend on environment forecasts or forecast-like slices and solver configuration.

### RL modes

- `rl` forwards normalised observations to a provided SB3 policy.
- The control-policy wrapper handles rollout and logging, not training.

## Logging shape

Typical episode logs include fields such as:

- `step`
- `norm_observation`
- `raw_observation` when available
- `action`
- `reward`
- `info`

This logging contract is important because downstream evaluation and DT dataset tooling assume it.

## Related docs

- [HELPER_README.md](HELPER_README.md)
- [DP_ALGORITHM_README.md](DP_ALGORITHM_README.md)
- [aemo/dispatch-replay.md](aemo/dispatch-replay.md)
- [architecture.md](architecture.md)
