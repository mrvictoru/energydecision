# Agent Guide

This document describes the agent-related modules that manage policy logic, optimization algorithms, and data collection for both the household (`SolarBatteryEnv`) and grid-scale (`AEMOBatteryTradingEnv`) environments.

## Agent Classes

### `Agent` (Solar battery)
- **Purpose**: Runs rule-based heuristics, SDP/MRDP/Oracle solvers, RL policies (SB3), and Decision Transformer inference on `SolarBatteryEnv` instances.
- **Key constructor arguments**:
  - `env`: `SolarBatteryEnv` instance
  - `algorithm`: `'rule' | 'sdp' | 'mrdp' | 'oracle' | 'rl' | 'dt'`
  - `model`: pre-trained RL or DT model
  - `horizon`, `soc_resolution`, `action_resolution`: configure DP solvers
  - `dt_gamma`, `rtg_value`: RTG handling for Decision Transformer inference
- **Mode behaviours**:
  - **rule**: solar/load heuristic uses normalized raw observations
  - **sdp/mrdp/oracle**: lazily instantiate solver classes, call `solve()` to obtain policy, index action by closest SoC
  - **rl**: forwards normalized observation to SB3 policy (`model.predict`)
  - **dt**: maintains rolling buffers of `(state, action, RTG, timestep)` and builds padded tensors for `model.get_action`
- **Episode execution**: `run_episode()` resets env, chooses observation type (raw vs norm), logs transitions, updates DT buffers, and returns episode/incident DataFrames.

### `AEMOAgent` (grid-scale)
- **Purpose**: Specializes in interacting with `AEMOBatteryTradingEnv`; added after AEMO env development.
- **Features**:
  - **rule**: price-based arbitrage action using FCAS-aware observation slices
  - **dispatch**: replays real AEMO dispatch data (via `fetch_aemo_unit_dispatch()`) by translating `TOTALCLEARED` into normalized actions
  - **rl/dt**: forwards obs to provided RL or Decision Transformer models similar to `Agent` but with AEMO observation layout
  - **DT buffering**: replicates `Agent`'s buffer logic to support `DecisionTransformer.get_action` (same context_len handling)
- **Dispatch mapping helper**: `_build_dispatch_actions()` resamples dispatch time series to env step duration and aligns to `SETTLEMENTDATE`, producing actions in `[-1, 1]` and optional FCAS bid fractions.

## Running Multiple Agents
- Use `run_episodes_parallel()` for `Agent` on `SolarBatteryEnv` with `algorithm` in `['rule','sdp','mrdp','dt','oracle','dispatch']` (new addition for `dispatch` replay).
- `AEMOAgent` is not yet included in parallel runner and should be run serially or via custom wrappers.

## Logging and Output
- Both agents return `(episode_df, incident_df)` with mean/per-step metrics.
- `Agent` collects `raw_observation` (if available) and normalized observation for RL/DT modes; `AEMOAgent` mirrors this for AEMO envs.
- Use the dictionaries in `info` (from `_make_reward_info`) to extract revenues, SOC, and degradation tracking.

## Usage Example
```python
from datetime import datetime
from src.aemo_data import fetch_aemo_data_bundle, fetch_aemo_unit_dispatch
from src.AEMOBatteryEnv import AEMODataPreprocessor, AEMOBatteryTradingEnv
from src.decision import AEMOAgent

bundle = fetch_aemo_data_bundle(start=datetime(2023,6,1), end=datetime(2023,6,2), region="NSW1")
dispatch = fetch_aemo_unit_dispatch(start=datetime(2023,6,1), end=datetime(2023,6,2), duid="LBBG1")

pre = AEMODataPreprocessor(step_duration_hours=0.5, add_normalized_features=True, update_stats_from_data=True)
processed = pre.preprocess_aemo_data(bundle['prices'], bundle['fcas'], bundle['generation'])
env = AEMOBatteryTradingEnv(
  aemo_data=processed,
  action_mode='simple',
  normalize_obs=True,
  return_raw_obs=True,
)

agent = AEMOAgent(env, algorithm='dispatch', dispatch_data=dispatch, dispatch_duid='LBBG1')
episode_df, incident_df = agent.run_episode()
```

## Agent Methods

### `Agent`

- `choose_action(obs)`
  - **Args**: `obs` is the current observation shipped from `SolarBatteryEnv.reset()`/`step()` (raw + cyclical features). The method is tolerant of `list` inputs for rule-based modes but prefers `numpy.ndarray` when passing into SB3 or Decision Transformer models.
  - **Returns**: a normalized action or list of actions in `[-1, 1]`. Rule/DP/oracle modes wrap the result in a 1-element list, RL returns the SB3 `model.predict` output (scalar or vector) and DT returns the action produced by `model.get_action` after padding its context.
  - **Behaviour**: routes to the heuristic, solver, RL, or DT helper according to `algorithm`. DT requires buffering of `(state, action, RTG, timestep)` and applies return-scale clipping before inference.

- `rule_based_action(obs)`
  - **Args**: raw observation containing cyclical time features, the `SolarBatteryEnv.df` columns (SolarGen, HouseLoad…), `battery_level_kwh`, and `battery_deg_cost`.
  - **Returns**: `[np.float32]` list with the normalized charger/discharger command. It compares solar vs. load to derive surplus/deficit, applies noise, respects SOC safety windows, and biases toward charge/discharge when SoC strays outside the degradation-safe band.

- `_get_forecasts(current_step, horizon)`
  - **Args**: `current_step` (int) and `horizon` (int) used by the SDP/MRDP solvers to fetch future observations from `self.env.df`.
  - **Returns**: List of dictionaries with forecast columns (`SolarGen`, `HouseLoad`, `ImportEnergyPrice`, `ExportEnergyPrice`). If `FutureGen`/`FutureLoad` columns exist, they are renamed for backwards compatibility; otherwise the base columns are used. Returns an empty list when insufficient data remains.

- `run_episode(render=False, display_progress=False)`
  - **Args**: optional `render` flag to call `env.render()` and `display_progress` to print step updates (notebook progress bars are commented out but placeholders exist).
  - **Returns**: Tuple `(episode_df, incident_df)`, both Polars DataFrames. `episode_df` collects `step`, `norm_observation`, `raw_observation`, `action`, `reward`, and `info`; `incident_df` mirrors `env.deg_incidents` or yields an empty schema when no degradation events were recorded.
  - **Behaviour**: resets the env, initializes DT buffers if needed, chooses `raw_obs` for rule/Solver/oracle modes and normalized obs for RL/DT, logs transitions, updates DT buffers, and optionally renders per step.

### `AEMOAgent` Specific Methods

- `set_dispatch_data(dispatch_data, dispatch_duid=None, dispatch_duid_gen=None, dispatch_duid_load=None, assume_single_duid_is_generator=True)`
  - **Args**: accepts the raw dispatch dataframe from `fetch_aemo_unit_dispatch()` plus optional DUID filters for separating generation/load streams.
  - **Returns**: `None`. Side effect is populating `self.dispatch_actions` with a resampled, normalized action trace used by the `dispatch` algorithm.

- `_build_dispatch_actions(dispatch_data, dispatch_duid=None, dispatch_duid_gen=None, dispatch_duid_load=None, assume_single_duid_is_generator=True)`
  - **Args**: same inputs as `set_dispatch_data`. Internally groups data by `SETTLEMENTDATE` to the env cadence, coalesces generator/load rows, and renames columns (NET_MW, RAISEREG_MW, LOWERREG_MW).
  - **Returns**: `np.ndarray` shaped `(steps, 1)` for simple mode or `(steps, 3)` when FCAS bids are requested; values are clipped to `[-1, 1]` relative to `env.max_battery_flow`. Returns `None` when there is no dispatch data or when the env lacks `aemo_data`.

- `_dispatch_action()`
  - **Args**: none; reads `self.env.current_step` internally.
  - **Returns**: one timestep of replayed action (scalar or 3-vector); zero-fills if dispatch data is unavailable or the env index falls outside the recorded range.

- `choose_action(obs)`
  - **Args**: `obs` is either raw AEMO observation (rule/dispatch) or normalized vector (RL/DT).
  - **Returns**: for `dispatch`, the replayed actions from `_dispatch_action`; for rule, a `[np.float32]` list derived from price/SOC thresholds; for RL/DT, same behaviour as `Agent.choose_action` albeit with the AEMO-specific observation layout (shorter state vector, extra FCAS fields).

- `rule_based_action(obs)`
  - **Args**: expects raw AEMO observation `[time⁵, RRP, TOTALDEMAND, FCAS×8, GEN×2, SOC]`; returns zero action when inputs are missing.
  - **Returns**: `[np.float32]` scaled action chosen by comparing the energy price to `charge_price`/`discharge_price` thresholds, enforcing SOC limits, and adding Gaussian noise for smoothing.

- `run_episode(render=False, display_progress=False)`
  - **Args**: same knobs as `Agent.run_episode` but also interprets `algorithm in ['rule','dispatch']` as using `raw_obs` for logs.
  - **Returns**: `(episode_df, incident_df)` where `episode_df` mirrors the same columns as `Agent` but logs both normalized and raw AEMO observations; `incident_df` is currently an empty `pl.DataFrame()` placeholder (no degradation tracking yet).
  - **Behaviour**: if `algorithm == 'dt'`, initializes DT buffers with the shorter AEMO state/action dims. The loop calls `choose_action`, steps the env, logs, updates DT buffers with the returned actions/RTGs, and tracks dispatch playback when relevant.
