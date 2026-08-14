import logging
import inspect
import numpy as np
import torch
import polars as pl
import warnings
from typing import Optional, Any, Dict, List, Tuple
from EnergySimEnv import SolarBatteryEnv, VIOLATION_PENALTY

from batterydeg import DegradationModel, RainflowCounter
from quantile_scenarios import QuantileScenarioGenerator
from grpo_posttraining import stable_rtg_update

# Import self-contained algorithm classes
from sdp_algorithm import SDPSolver
from mrdp_algorithm import MRDPSolver
from oracle_algorithm import OracleSolver
from aemo_oracle_algo import AEMOOracleSolver, OracleResult, FCAS_SERVICES

import concurrent.futures
from tqdm.notebook import tqdm

DEG_INCIDENT_FIELDS = [
    "episode_id",
    "step",
    "SOCav",
    "DOD",
    "Id",
    "Ich",
    "nCL_T",
    "nCL_Id",
    "nCL_Ich",
    "nCL_SOCav_DOD",
    "CL4_raw",
    "mult",
    "CL",
    "step_degradation",
]


def _reset_env(env, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
    """Reset an environment while supporting both Gymnasium-style and legacy reset signatures."""
    if options is None:
        return env.reset(seed=seed)

    reset_sig = inspect.signature(env.reset)
    if 'options' in reset_sig.parameters:
        return env.reset(seed=seed, options=options)
    return env.reset(seed=seed, **options)


def _safe_float32_array(data, default_shape=(0,), dtype=np.float32):
    """Convert a list of RTGs to a safe float32 numpy array."""
    if not data:
        return np.zeros(default_shape, dtype=dtype)
    arr = np.array(data, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, 0.0)
    clip_max = np.finfo(np.float32).max
    arr = np.clip(arr, -clip_max, clip_max)
    return arr.astype(dtype)


def _build_dt_inference_context(model, states_buffer, actions_buffer, rtgs_buffer, timesteps_buffer):
    """Build padded, scaled, sanitized numpy arrays for DT inference.

    Returns (states, actions, rtgs, timesteps, mask) as numpy arrays,
    all shaped [context_len, ...].
    """
    context_len = model.context_len
    state_dim = model.state_dim
    act_dim = model.act_dim
    buffer_len = len(states_buffer)

    buffer_states = (
        np.array(states_buffer, dtype=np.float32)
        if buffer_len > 0 else np.zeros((0, state_dim), dtype=np.float32)
    )
    if buffer_states.ndim == 1 and buffer_len > 0:
        buffer_states = buffer_states.reshape(buffer_len, state_dim)

    buffer_actions = (
        np.array(actions_buffer, dtype=np.float32)
        if buffer_len > 0 else np.zeros((0, act_dim), dtype=np.float32)
    )
    if buffer_actions.ndim == 1 and buffer_len > 0:
        buffer_actions = buffer_actions.reshape(buffer_len, act_dim)

    buffer_rtgs = _safe_float32_array(
        rtgs_buffer, default_shape=(0,), dtype=np.float32
    ) if buffer_len > 0 else np.zeros(0, dtype=np.float32)

    buffer_timesteps = (
        np.array(timesteps_buffer, dtype=np.int64)
        if buffer_len > 0 else np.zeros(0, dtype=np.int64)
    )

    if buffer_len < context_len:
        pad_len = context_len - buffer_len
        states = np.vstack([np.zeros((pad_len, state_dim), dtype=np.float32), buffer_states])
        actions = np.vstack([np.zeros((pad_len, act_dim), dtype=np.float32), buffer_actions])
        rtgs = np.concatenate([np.zeros(pad_len, dtype=np.float32), buffer_rtgs])
        timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), buffer_timesteps])
        mask = np.concatenate([np.zeros(pad_len, dtype=np.bool_), np.ones(buffer_len, dtype=np.bool_)])
    else:
        states = buffer_states[-context_len:]
        actions = buffer_actions[-context_len:]
        rtgs = buffer_rtgs[-context_len:]
        timesteps = buffer_timesteps[-context_len:]
        mask = np.ones(context_len, dtype=np.bool_)

    return_scale_attr = getattr(model, 'return_scale', 1.0)
    return_scale = float(return_scale_attr.detach().cpu().item()) if isinstance(return_scale_attr, torch.Tensor) else float(return_scale_attr)
    if not np.isfinite(return_scale) or abs(return_scale) < 1e-12:
        raise ValueError(f"Invalid Decision Transformer return_scale: {return_scale}")
    if return_scale != 1.0:
        rtgs = rtgs / return_scale

    states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
    actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    rtgs = np.nan_to_num(rtgs, nan=0.0, posinf=0.0, neginf=0.0)

    rtgs = np.clip(rtgs, -1e3, 1e3)

    max_time = getattr(model.embed_timestep, 'num_embeddings', None)
    if max_time is not None and max_time > 0:
        timesteps = np.clip(timesteps, 0, max_time - 1)

    return states, actions, rtgs, timesteps, mask


class Agent:
    def __init__(self, env: SolarBatteryEnv, algorithm='rule', model=None,
                 horizon=72, soc_resolution=20, action_resolution=41,
                 use_monte_carlo: bool = True, mc_samples: int = 200, mc_seed: Optional[int] = None,
                 subhorizon_specs=None, rtg_value: float = 0.0, dt_gamma: float = 0.99,
                 reset_seed: Optional[int] = None, reset_options: Optional[Dict[str, Any]] = None):
        """
        env: an instance of SolarBatteryEnv.
        algorithm: choose between 'rule', 'rl', 'dt', 'mrdp', 'sdp', or 'oracle'.
        model: For RL/DT algorithm, a trained model.
        horizon: Time horizon for SDP optimization (default: 48 steps = 24 hours).
        soc_resolution: Resolution of state-of-charge discretization (default: 20 levels).
        action_resolution: Resolution of action discretization (default: 41 levels, e.g., -1.0, -0.95, ..., 0.95, 1.0).
        dt_gamma: Discount factor for RTG updates in DT inference (default: 0.99, matching training).
        """
        self.env = env
        self.algorithm = algorithm.lower()
        self.model = model
        self.rule_presistence = False  # Preset for rule-based action persistence
        self.reset_seed = reset_seed
        self.reset_options = reset_options

        if self.algorithm in ('sdp', 'mrdp', 'oracle'):
            required_subhorizon_keys = {'start', 'length', 'soc_resolution', 'action_resolution', 'step_duration'}
            self.subhorizon_specs = subhorizon_specs
            # Normalize shorthand keys (soc_res/action_res) to soc_resolution/action_resolution
            if self.subhorizon_specs is not None:
                if not isinstance(self.subhorizon_specs, list):
                    raise ValueError("subhorizon_specs must be a list of dicts.")
                normalized = []
                for idx, spec in enumerate(self.subhorizon_specs):
                    if not isinstance(spec, dict):
                        raise ValueError(f"subhorizon_specs[{idx}] must be a dict.")
                    spec_copy = dict(spec)
                    if 'soc_res' in spec_copy and 'soc_resolution' not in spec_copy:
                        spec_copy['soc_resolution'] = spec_copy.pop('soc_res')
                    if 'action_res' in spec_copy and 'action_resolution' not in spec_copy:
                        spec_copy['action_resolution'] = spec_copy.pop('action_res')
                    missing = required_subhorizon_keys - set(spec_copy.keys())
                    if missing:
                        raise ValueError(f"subhorizon_specs[{idx}] is missing required keys: {missing}")
                    normalized.append(spec_copy)
                self.subhorizon_specs = normalized

            self.horizon = horizon
            self.soc_resolution = soc_resolution
            self.action_resolution = action_resolution

            # Shared DP parameters
            self.battery_capacity = env.battery_capacity
            self.max_battery_flow = env.max_battery_flow
            self.step_duration = env.step_duration
            self.max_grid_energy = env.max_grid_energy
            self.battery_life_cost = env.battery_life_cost
            self.soc_levels_kwh = np.linspace(0, self.battery_capacity, self.soc_resolution)

            if self.algorithm == 'oracle':
                self.oracle_horizon = horizon
                self.oracle_action_levels = np.linspace(-1.0, 1.0, action_resolution, dtype=np.float32)
                
                # Initialize self-contained Oracle solver (rainflow-only)
                self.oracle_solver = OracleSolver(
                    env=env,
                    horizon=horizon,
                    soc_resolution=soc_resolution,
                    action_resolution=action_resolution
                )

            if self.algorithm == 'sdp':
                # Initialize self-contained SDP solver (rainflow-only degradation)
                self.sdp_solver = SDPSolver(
                    env=env,
                    horizon=horizon,
                    soc_resolution=soc_resolution,
                    action_resolution=action_resolution,
                    use_monte_carlo=use_monte_carlo,
                    mc_samples=mc_samples,
                    mc_seed=mc_seed,
                    scenario_generator=QuantileScenarioGenerator(n_scenarios=5)
                    )
                    
                    # Keep these for backward compatibility
                self.soc_levels_kwh = self.sdp_solver.soc_levels_kwh
                self.action_levels_norm = self.sdp_solver.action_levels_norm
            
            elif self.algorithm == 'mrdp':
                # Initialize self-contained MRDP solver
                self.mrdp_solver = MRDPSolver(
                    env=env,
                    subhorizon_specs=subhorizon_specs,
                    use_monte_carlo=use_monte_carlo,
                    mc_samples=mc_samples,
                    mc_seed=mc_seed,
                    scenario_generator=QuantileScenarioGenerator(n_scenarios=5)
                )
                
                # Keep these for backward compatibility
                if subhorizon_specs and len(subhorizon_specs) > 0:
                    first_spec = subhorizon_specs[0]
                    self.soc_levels_kwh = np.linspace(0, env.battery_capacity, first_spec['soc_resolution'])
                    self.action_levels_norm = np.linspace(-1.0, 1.0, first_spec['action_resolution'])
                else:
                    self.soc_levels_kwh = np.linspace(0, env.battery_capacity, soc_resolution)
                    self.action_levels_norm = np.linspace(-1.0, 1.0, action_resolution)
        elif self.algorithm == 'dt':
            self.rtg_value = rtg_value  # Initial Return-to-go value for DT input
            self.dt_gamma = dt_gamma    # Discount factor for RTG updates
            # Initialize rolling context buffers (will be populated after reset)
            self.dt_states_buffer = []
            self.dt_actions_buffer = []
            self.dt_rtgs_buffer = []
            self.dt_timesteps_buffer = []
        
        

    def choose_action(self, obs):
        if self.algorithm == 'rule':
            return self.rule_based_action(obs)
        elif self.algorithm == 'rl':
            if self.model is None:
                raise ValueError("RL algorithm selected but no model provided.")
            if not isinstance(obs, np.ndarray):
                warnings.warn("Observation is not a numpy array. RL model expects a numpy array input.")
                # return specific values for action 
                return [0.0008964] # Example value, adjust as needed
            
            action, _ = self.model.predict(obs, deterministic=True)
            return action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action
        elif self.algorithm == 'dt':
            if self.model is None:
                raise ValueError("Decision Transformer selected but no model provided.")
            self.model.eval()
            device = next(self.model.parameters()).device
            
            states, actions, rtgs, timesteps, mask = _build_dt_inference_context(
                self.model, self.dt_states_buffer, self.dt_actions_buffer,
                self.dt_rtgs_buffer, self.dt_timesteps_buffer,
            )
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
            rtgs = torch.tensor(rtgs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            timesteps = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                action = self.model.get_action(states, actions, rtgs, timesteps, attention_mask=attention_mask)
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            action = action.detach().cpu().numpy()
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).tolist()
            return action

        
        elif self.algorithm == 'sdp':
            current_soc_kwh = obs[-2]  # Assuming BatteryLevel is the second to last element
            current_step_env = self.env.current_step
            
            # Get forecasts for horizon
            forecasts = self._get_forecasts(current_step_env, self.sdp_solver.horizon)
            if not forecasts:
                #print("Warning: Not enough forecast data for full horizon. Using rule-based action.")
                return self.rule_based_action(obs)
            
            # Solve using self-contained SDP solver
            policy_table = self.sdp_solver.solve(forecasts, start_index=current_step_env)
            
            # Get action for current state
            soc_idx = np.argmin(np.abs(self.soc_levels_kwh - current_soc_kwh))
            optimal_action_idx = policy_table[0, soc_idx]
            
            if optimal_action_idx == -1:
                #print(f"Warning: No optimal action found for SoC {current_soc_kwh:.2f}. Using zero action.")
                action_value = 0.0008964
            else:
                action_value = self.action_levels_norm[int(optimal_action_idx)]
            
            noise = np.random.normal(-0.001, 0.001)
            action_value = np.clip(action_value + noise, -1.0, 1.0)
            return [np.float32(action_value)]
        
        elif self.algorithm == 'mrdp':
            current_soc_kwh = obs[-2]
            current_step_env = self.env.current_step
            
            # Get forecasts for total horizon
            total_horizon = sum(int(spec.get('length', 0)) for spec in self.mrdp_solver.subhorizon_specs)
            forecasts = self._get_forecasts(current_step_env, total_horizon)
            if not forecasts:
                #print("Warning: Not enough forecast data for full horizon. Using rule-based action.")
                return self.rule_based_action(obs)
            
            # Solve using self-contained MRDP solver
            policy_table = self.mrdp_solver.solve(forecasts, start_index=current_step_env)
            
            # Get action for current state
            soc_idx = np.argmin(np.abs(self.soc_levels_kwh - current_soc_kwh))
            optimal_action_idx = policy_table[0, soc_idx]
            
            if optimal_action_idx == -1:
                #print(f"Warning: No optimal action found for SoC {current_soc_kwh:.2f}. Using zero action.")
                action_value = 0.0008964
            else:
                action_value = self.action_levels_norm[int(optimal_action_idx)]
            
            noise = np.random.normal(-0.001, 0.001)
            action_value = np.clip(action_value + noise, -1.0, 1.0)
            return [np.float32(action_value)]
        
        elif self.algorithm == 'oracle':
            # Determine horizon
            remaining_steps = max(0, len(self.env.df) - self.env.current_step)
            remaining_env_budget = max(0, self.env.max_step - self.env.current_step)
            horizon = min(self.oracle_solver.horizon, remaining_steps, remaining_env_budget)
            
            if horizon <= 0:
                raw_obs = self.env.get_raw_obs()
                return self.rule_based_action(raw_obs)
            
            # Solve using self-contained Oracle solver
            policy_table = self.oracle_solver.solve(self.env.current_step, horizon)
            
            if policy_table is None:
                raw_obs = self.env.get_raw_obs()
                return self.rule_based_action(raw_obs)
            
            # Get action for current state
            action_value = self.oracle_solver.get_action_for_current_state(
                policy_table, self.env.battery_level
            )
            
            return [np.float32(action_value)]
        
     # --- SDP Helper Methods ---

    def _get_forecasts(self, current_step, horizon):
        """Retrieves forecast data for the SDP horizon using FutureGen/FutureLoad if available."""
        end_step = current_step + horizon
        if end_step > len(self.env.df):
            return []

        # Use FutureGen/FutureLoad if available, otherwise fallback to SolarGen/HouseLoad
        df = self.env.df.slice(current_step, horizon)
        required_cols = ['ImportEnergyPrice', 'ExportEnergyPrice']

        # Check if FutureGen/FutureLoad exist in the DataFrame
        use_future = all(col in df.columns for col in ['FutureGen', 'FutureLoad'])
        if use_future:
            forecast_df = df.select(['FutureGen', 'FutureLoad'] + required_cols)
            forecast_df = forecast_df.rename({'FutureGen': 'SolarGen', 'FutureLoad': 'HouseLoad'})
        else:
            forecast_df = df.select(['SolarGen', 'HouseLoad'] + required_cols)

        forecast_list = forecast_df.to_dicts()
        return forecast_list

    def _soc_to_idx(self, soc_kwh):
        """Maps a continuous SoC value to the index of the nearest discrete level."""
        return np.argmin(np.abs(self.soc_levels_kwh - soc_kwh))

    # --- Oracle Helper Methods ---


    # --- SDP/MRDP/Oracle methods removed - now using self-contained algorithm classes ---
    # See sdp_algorithm.py, mrdp_algorithm.py, oracle_algorithm.py
    # Old methods (_solve_sdp, _solve_mrdp, _choose_oracle_action, _calculate_sdp_stage_cost)
    # have been deprecated and removed.

    def rule_based_action(self, obs):
        # raw_obs layout: 4 cyclical time features, then DF columns in env.ordered_df_cols_for_obs,
        # then [battery_level_kwh, battery_deg_cost]
        cyclical_len = 4
        try:
            solar_idx = cyclical_len + self.env.ordered_df_cols_for_obs.index('SolarGen')
            load_idx = cyclical_len + self.env.ordered_df_cols_for_obs.index('HouseLoad')
        except Exception:
            # Fallback: assume SolarGen/HouseLoad are the first two DF features
            solar_idx = cyclical_len
            load_idx = cyclical_len + 1

        diff = float(obs[solar_idx]) - float(obs[load_idx])  # surplus if positive
        max_flow = self.env.max_battery_flow
        battery_level = obs[-2]/self.env.battery_capacity  # normalize battery level to [0, 1] by dividing battery energy (obs[-2]) by capacity
        noise = np.random.normal(-0.01, 0.01)  # add small noise with standard deviation of 0.01
        safe_margin_pct = max(getattr(self.env, "base_deg_DoD", 80.0) / 2.0, 5.0)
        safe_lower_norm = min(safe_margin_pct / 100.0, 0.5)
        safe_upper_norm = max(1.0 - safe_lower_norm, safe_lower_norm)
        # this is to check
        if self.rule_presistence and battery_level < 0.9:  # battery is not full
            #continue charging.
            result = min(0.5 + noise, 0.5)
        elif self.rule_presistence and battery_level > 0.9:  # battery is not empty
            self.rule_presistence = False
            result = max(-0.1 + noise, -0.1)
        elif battery_level < 0.1:  # battery is empty
            self.rule_presistence = True
            result = min(0.5 + noise, 0.5)
        
        elif diff > 0 and battery_level < 0.9:  # surplus energy
            # Compute the recommended charging power as a fraction of the maximum battery flow;
            # ensure it does not exceed 1.0 (100% of max battery flow)
            result = min((diff / max_flow)+noise, 1.0)            
        elif diff < 0 and battery_level > 0.1:  # deficit energy
            # Compute the recommended discharging power as a fraction of the maximum battery flow;
            # ensure it does not exceed 1.0 (100% of max battery flow)
            result = max((diff / max_flow)+noise, -1.0)           
        else:
            # No action needed; add noise to zero action.
            result = 0.0 + noise

        if battery_level < safe_lower_norm:
            # Encourage charging when SoC is below the degradation-safe window
            result = max(result, 0.2)
        elif battery_level > safe_upper_norm:
            # Encourage discharging when SoC is above the degradation-safe window
            result = min(result, -0.2)

        result = float(np.clip(result, -1.0, 1.0))
    
        return [np.float32(result)]  # Return as a list to match expected action format
    
    def run_episode(self, render=False, display_progress=False):
        obs, info = _reset_env(self.env, seed=self.reset_seed, options=self.reset_options)
        raw_obs = self.env.get_raw_obs()  # Get raw observation if available
        max_possible_steps = len(self.env.df)

        # Initialize DT rolling buffers upon reset
        if self.algorithm == 'dt':
            self.dt_states_buffer = [obs.copy()]
            self.dt_actions_buffer = [np.zeros(self.model.act_dim)]  # Placeholder action for first step
            self.dt_rtgs_buffer = [self.rtg_value]
            self.dt_timesteps_buffer = [self.env.current_step]

        logs = []
        terminated, truncated = False, False
        step = 0  # Step counter
        # Decide which obs to use based on agent type
        if self.algorithm in ['rule', 'sdp', 'oracle']:
            # Use raw_obs if available, else fallback to obs
            current_obs = raw_obs
        else:  # 'rl', 'dt', etc.
            # Use norm_obs if available, else fallback to obs
            current_obs = obs

        #pbar = None
        if display_progress:
            print("Starting Simulation...")
            """
            try:
                from tqdm.notebook import tqdm as tqdm_bar
            except Exception:
                from tqdm import tqdm as tqdm_bar
            # Use DataFrame length as an upper bound for progress
            pbar = tqdm_bar(total=max_possible_steps, desc="Episode", leave=False)
            """
        try:
            while not (terminated or truncated):
                action = self.choose_action(current_obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                logs.append({
                    'step': step,
                    'norm_observation': obs.tolist() if isinstance(current_obs, np.ndarray) else current_obs,
                    'raw_observation': raw_obs.tolist() if isinstance(raw_obs, np.ndarray) else raw_obs,
                    'action': action,
                    'reward': reward,
                    'info': info
                })

                # Update DT rolling buffers after step
                if self.algorithm == 'dt':
                    context_len = self.model.context_len
                    # Update the last action slot with the actual action taken
                    if isinstance(action, list):
                        action_array = np.array(action, dtype=np.float32)
                    else:
                        action_array = np.array([action], dtype=np.float32) if np.isscalar(action) else action
                    self.dt_actions_buffer[-1] = action_array
                    
                    # Stable discounted RTG update (clamped to the trained envelope for
                    # gamma<1 so the 1/gamma recurrence cannot explode on long horizons).
                    next_rtg = stable_rtg_update(
                        self.dt_rtgs_buffer[-1], reward,
                        dt_gamma=self.dt_gamma, initial_rtg=self.rtg_value,
                    )
                    
                    # Append next state, action placeholder, RTG, and timestep
                    self.dt_states_buffer.append(next_obs.copy())
                    self.dt_actions_buffer.append(np.zeros(self.model.act_dim))  # Placeholder for next action
                    self.dt_rtgs_buffer.append(next_rtg)
                    self.dt_timesteps_buffer.append(self.env.current_step)
                    
                    # Clamp buffers to context_len
                    if len(self.dt_states_buffer) > context_len:
                        self.dt_states_buffer = self.dt_states_buffer[-context_len:]
                        self.dt_actions_buffer = self.dt_actions_buffer[-context_len:]
                        self.dt_rtgs_buffer = self.dt_rtgs_buffer[-context_len:]
                        self.dt_timesteps_buffer = self.dt_timesteps_buffer[-context_len:]

                # Select the correct next_obs for the next step
                raw_obs = self.env.get_raw_obs()  # Get raw observation if available
                obs = next_obs
                if self.algorithm in ['rule', 'sdp', 'oracle']:
                    current_obs = raw_obs
                else:
                    current_obs = obs
                if render:
                    self.env.render()
                step += 1  # Increment step counter
                if display_progress is not False:
                    #pbar.update(1)
                    print(f"Step {step}/{max_possible_steps}", end='\r')

        finally:
            #if pbar is not None:
                #pbar.close()
            print("Sim Complete")

        episode_df = pl.DataFrame(logs)

        incident_df = (
            pl.DataFrame(self.env.deg_incidents)
            if self.env.deg_incidents
            else pl.DataFrame(schema={
                "step": pl.Int64,
                "step_degradation": pl.Float64,
                **{k: pl.Float64 for k in DEG_INCIDENT_FIELDS},
            })
        )

        return episode_df, incident_df


class AEMOAgent:
    """
    Agent for AEMOBatteryTradingEnv supporting rule-based, RL, DT, and dispatch replay actions.
    """

    def __init__(self,
                 env: Any,
                 algorithm: str = 'rule',
                 model: Optional[Any] = None,
                 dispatch_data: Optional[pl.DataFrame] = None,
                 dispatch_duid: Optional[str] = None,
                 dispatch_duid_gen: Optional[str] = None,
                 dispatch_duid_load: Optional[str] = None,
                 assume_single_duid_is_generator: bool = True,
                 dispatch_type: Optional[str] = None,
                 dispatch_action_mode: Optional[str] = None,
                 rtg_value: float = 0.0,
                 dt_gamma: float = 0.99,
                 reset_seed: Optional[int] = None,
                 reset_options: Optional[Dict[str, Any]] = None,
                 fcas_raise_threshold: float | None = None,
                 fcas_lower_threshold: float | None = None,
                 fcas_pctile: float = 0.80,
                 forecast_npz_path: str | None = None):
        self.env = env
        self.algorithm = algorithm.lower()
        self.model = model
        self.rule_presistence = False
        self.reset_seed = reset_seed
        self.reset_options = reset_options

        self.rtg_value = rtg_value
        self.dt_gamma = dt_gamma
        self.dt_states_buffer = []
        self.dt_actions_buffer = []
        self.dt_rtgs_buffer = []
        self.dt_timesteps_buffer = []

        self.dispatch_action_mode = dispatch_action_mode or getattr(env, 'action_mode', 'simple')
        self.dispatch_actions = None
        if dispatch_data is not None:
            self.set_dispatch_data(
                dispatch_data,
                dispatch_duid=dispatch_duid,
                dispatch_duid_gen=dispatch_duid_gen,
                dispatch_duid_load=dispatch_duid_load,
                assume_single_duid_is_generator=assume_single_duid_is_generator,
                dispatch_type=dispatch_type,
            )

        # Oracle (perfect-foresight LP co-optimizer)
        self._oracle_result: OracleResult | None = None
        self._oracle_actions: np.ndarray | None = None
        if self.algorithm == 'aemo_oracle':
            self._init_oracle()

        # Hierarchical DT+LP executor: a DT predicts a coarse target-SOC
        # waypoint trajectory; the SOC-waypoint-pinned Oracle LP co-optimizes
        # energy + FCAS within each segment while tracking it.
        self._soc_oracle_actions: np.ndarray | None = None
        self._soc_oracle_waypoints: list[float] = []
        if self.algorithm == 'dt_soc_oracle':
            self._init_soc_oracle()

        # FCAS rule config
        self.fcas_raise_threshold = fcas_raise_threshold
        self.fcas_lower_threshold = fcas_lower_threshold
        self.fcas_pctile = float(fcas_pctile)
        self._fcas_norm_max = self._compute_fcas_norm_max()
        self._init_fcas_thresholds()

        # Forecast DT config
        self.forecast_npz_path = forecast_npz_path
        self._forecast_map: np.ndarray | None = None

    def set_dispatch_data(self,
                          dispatch_data: pl.DataFrame,
                          dispatch_duid: Optional[str] = None,
                          dispatch_duid_gen: Optional[str] = None,
                          dispatch_duid_load: Optional[str] = None,
                          assume_single_duid_is_generator: bool = True,
                          dispatch_type: Optional[str] = None) -> None:
        """Configure dispatch replay from a NEMOSIS DISPATCHLOAD DataFrame.

        Args:
            dispatch_data: Polars DataFrame with SETTLEMENTDATE, DUID, TOTALCLEARED, and
                FCAS columns (as returned by ``fetch_aemo_unit_dispatch``).
            dispatch_duid: Single DUID to replay.  Used when the battery is registered
                as a "Bidirectional Unit" or when only one DUID is available.
            dispatch_duid_gen: Generator (discharge) DUID for paired gen/load batteries.
                When provided together with ``dispatch_duid_load``, the net energy action
                is computed as ``LOAD_MW - GEN_MW``.
            dispatch_duid_load: Load (charge) DUID for paired gen/load batteries.
            assume_single_duid_is_generator: When ``dispatch_duid`` is set and neither
                ``dispatch_duid_gen`` nor ``dispatch_duid_load`` is set, this controls
                the sign of the energy action.  ``True`` (default) → battery is a
                generator, so ``NET_MW = -TOTALCLEARED`` (negative = discharging);
                ``False`` → battery is a load, so ``NET_MW = +TOTALCLEARED``
                (positive = charging).  Automatically overridden to ``False`` when
                ``dispatch_type`` is ``'Load'``.
            dispatch_type: Optional AEMO dispatch type string (e.g. ``'Load'``,
                ``'Generating Unit'``, ``'Bidirectional Unit'``).  When ``'Load'`` is
                passed, ``assume_single_duid_is_generator`` is forced to ``False``
                regardless of its explicit value.
        """
        # Auto-correct the sign convention when the DUID is explicitly a Load unit.
        if dispatch_type is not None:
            dt_lower = str(dispatch_type).strip().lower()
            if 'load' in dt_lower and 'generating' not in dt_lower:
                assume_single_duid_is_generator = False

        self.dispatch_actions = self._build_dispatch_actions(
            dispatch_data,
            dispatch_duid=dispatch_duid,
            dispatch_duid_gen=dispatch_duid_gen,
            dispatch_duid_load=dispatch_duid_load,
            assume_single_duid_is_generator=assume_single_duid_is_generator
        )

    def _build_dispatch_actions(self,
                                dispatch_data: pl.DataFrame,
                                dispatch_duid: Optional[str] = None,
                                dispatch_duid_gen: Optional[str] = None,
                                dispatch_duid_load: Optional[str] = None,
                                assume_single_duid_is_generator: bool = True) -> Optional[np.ndarray]:
        if dispatch_data is None or dispatch_data.height == 0:
            return None
        if not hasattr(self.env, 'aemo_data') or self.env.aemo_data.height == 0:
            return None
        if 'SETTLEMENTDATE' not in dispatch_data.columns:
            raise ValueError("dispatch_data must include SETTLEMENTDATE")

        every_minutes = int(round(float(self.env.step_duration) * 60))
        every = f"{every_minutes}m"

        def _prep(df: pl.DataFrame) -> pl.DataFrame:
            df = df.with_columns(pl.col('SETTLEMENTDATE').cast(pl.Datetime, strict=False))
            df = df.sort('SETTLEMENTDATE')
            numeric_cols = [c for c in df.columns if c not in {'SETTLEMENTDATE', 'DUID'}]
            aggs = [pl.col(c).mean().alias(c) for c in numeric_cols]
            return df.group_by_dynamic('SETTLEMENTDATE', every=every, label='left', closed='left').agg(aggs)

        df = dispatch_data
        # All 8 FCAS services, ordered to match env's _fcas_services
        FCAS_SERVICES_DISP = [
            'RAISEREG', 'LOWERREG', 'RAISE6SEC', 'LOWER6SEC',
            'RAISE60SEC', 'LOWER60SEC', 'RAISE5MIN', 'LOWER5MIN',
        ]

        if 'DUID' in df.columns and (dispatch_duid_gen or dispatch_duid_load):
            gen_df = None
            load_df = None
            if dispatch_duid_gen:
                gen_pre = _prep(df.filter(pl.col('DUID') == dispatch_duid_gen))
                gen_rename = {'TOTALCLEARED': 'GEN_MW'}
                gen_rename.update({svc: f'GEN_{svc}' for svc in FCAS_SERVICES_DISP if svc in gen_pre.columns})
                gen_df = gen_pre.rename(gen_rename)
            if dispatch_duid_load:
                load_pre = _prep(df.filter(pl.col('DUID') == dispatch_duid_load))
                load_rename = {'TOTALCLEARED': 'LOAD_MW'}
                load_rename.update({svc: f'LOAD_{svc}' for svc in FCAS_SERVICES_DISP if svc in load_pre.columns})
                load_df = load_pre.rename(load_rename)

            merged = gen_df if gen_df is not None else load_df
            if merged is None:
                return None
            if gen_df is not None and load_df is not None:
                merged = gen_df.join(load_df, on='SETTLEMENTDATE', how='full', coalesce=True)

            required_cols = {
                'GEN_MW': 0.0,
                'LOAD_MW': 0.0,
            }
            for svc in FCAS_SERVICES_DISP:
                required_cols[f'GEN_{svc}'] = 0.0
                required_cols[f'LOAD_{svc}'] = 0.0
            missing_exprs = [
                pl.lit(default_value).alias(col_name)
                for col_name, default_value in required_cols.items()
                if col_name not in merged.columns
            ]
            if missing_exprs:
                merged = merged.with_columns(missing_exprs)
            merged = merged.fill_null(0.0)

            sum_exprs = [
                (pl.col('LOAD_MW').fill_null(0.0) - pl.col('GEN_MW').fill_null(0.0)).alias('NET_MW'),
            ]
            for svc in FCAS_SERVICES_DISP:
                sum_exprs.append(
                    (pl.col(f'GEN_{svc}').fill_null(0.0) + pl.col(f'LOAD_{svc}').fill_null(0.0))
                    .alias(f'{svc}_MW')
                )
            select_cols = ['SETTLEMENTDATE', 'NET_MW'] + [f'{svc}_MW' for svc in FCAS_SERVICES_DISP]
            dispatch_res = merged.with_columns(sum_exprs).select(select_cols)
        else:
            if 'DUID' in df.columns and dispatch_duid:
                df = df.filter(pl.col('DUID') == dispatch_duid)
                if df.height == 0:
                    return None
            dispatch_res = _prep(df)
            if 'TOTALCLEARED' not in dispatch_res.columns:
                return None
            sign = -1.0 if assume_single_duid_is_generator else 1.0
            sum_exprs = [
                (pl.lit(sign) * pl.col('TOTALCLEARED')).alias('NET_MW'),
            ]
            for svc in FCAS_SERVICES_DISP:
                if svc in dispatch_res.columns:
                    sum_exprs.append(pl.col(svc).fill_null(0.0).alias(f'{svc}_MW'))
                else:
                    sum_exprs.append(pl.lit(0.0).alias(f'{svc}_MW'))
            select_cols = ['SETTLEMENTDATE', 'NET_MW'] + [f'{svc}_MW' for svc in FCAS_SERVICES_DISP]
            dispatch_res = dispatch_res.with_columns(sum_exprs).select(select_cols)

        dispatch_res = dispatch_res.with_columns(
            pl.col('SETTLEMENTDATE').cast(pl.Datetime('us'), strict=False)
        )
        grid = (
            self.env.aemo_data
            .select(['SETTLEMENTDATE'])
            .with_columns(pl.col('SETTLEMENTDATE').cast(pl.Datetime('us'), strict=False))
            .sort('SETTLEMENTDATE')
        )
        aligned = grid.join(dispatch_res, on='SETTLEMENTDATE', how='left').fill_null(0.0)

        total_nonzero = (
            aligned['NET_MW'].abs().sum()
            + sum(aligned[f'{svc}_MW'].abs().sum() for svc in FCAS_SERVICES_DISP)
        )
        if total_nonzero == 0.0:
            warnings.warn(
                "Dispatch actions are all zero after aligning with the environment grid. "
                "Possible causes:\n"
                "  1. TOTALCLEARED is zero for all intervals (battery was not dispatched for energy).\n"
                "  2. All 8 FCASenablement columns are zero (battery did not provide FCAS).\n"
                "  3. Timestamp mismatch between dispatch data and environment data.\n"
                "Check that the selected DUID was actively dispatched during the date range. "
                "For batteries that only provide contingency FCAS (RAISE6SEC, LOWER6SEC, etc.), "
                "consider using the paired gen/load DUID approach via dispatch_duid_gen and "
                "dispatch_duid_load arguments.",
                stacklevel=3,
            )

        net = aligned['NET_MW'].to_numpy()
        a0 = np.clip(net / float(self.env.max_battery_flow), -1.0, 1.0).astype(np.float32)

        if self.dispatch_action_mode == 'simple':
            return a0.reshape(-1, 1)

        env_action_mode = getattr(self.env, 'action_mode', 'multi_market')
        if env_action_mode == 'full_fcas':
            # Return all 8 FCAS service bid fractions
            fcas_cols = []
            for svc in FCAS_SERVICES_DISP:
                bid = np.clip(
                    aligned[f'{svc}_MW'].to_numpy() / float(self.env.max_battery_flow),
                    0.0, 1.0,
                ).astype(np.float32)
                fcas_cols.append(bid)
            return np.stack([a0] + fcas_cols, axis=1).astype(np.float32)
        else:
            # Legacy multi_market (3-dim): only RAISEREG / LOWERREG
            raise_bid = np.clip(
                aligned['RAISEREG_MW'].to_numpy() / float(self.env.max_battery_flow),
                0.0, 1.0,
            ).astype(np.float32)
            lower_bid = np.clip(
                aligned['LOWERREG_MW'].to_numpy() / float(self.env.max_battery_flow),
                0.0, 1.0,
            ).astype(np.float32)
            return np.stack([a0, raise_bid, lower_bid], axis=1).astype(np.float32)

    def _dispatch_action(self) -> np.ndarray:
        if self.dispatch_actions is None:
            return np.array([0.0], dtype=np.float32) if self.dispatch_action_mode == 'simple' else np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        episode_start = int(getattr(self.env, 'episode_start_idx', 0))
        idx = episode_start + int(self.env.current_step)
        if idx < 0 or idx >= len(self.dispatch_actions):
            return np.array([0.0], dtype=np.float32) if self.dispatch_action_mode == 'simple' else np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        return self.dispatch_actions[idx]

    # ── Forecast DT helpers ─────────────────────────────────────────────

    def _ensure_forecast_map(self) -> None:
        if self._forecast_map is not None:
            return
        if self.forecast_npz_path:
            try:
                fc = np.load(str(self.forecast_npz_path))
                self._forecast_map = fc["forecast_map"]
                self._forecast_timestamps = fc["timestamps"]
            except Exception:
                self._forecast_map = np.array([], dtype=np.float32)
                self._forecast_timestamps = np.array([], dtype=np.float32)
        else:
            self._forecast_map = np.array([], dtype=np.float32)
            self._forecast_timestamps = np.array([], dtype=np.float32)

    def _forecast_npz_offset(self) -> int:
        fts = getattr(self, '_forecast_timestamps', np.array([], dtype=np.float32))
        if len(fts) == 0:
            return 0
        df = getattr(self.env, 'aemo_data', None)
        if df is None or 'SETTLEMENTDATE' not in df.columns:
            return 0
        first_row = df.row(0)
        col_idx = df.columns.index('SETTLEMENTDATE')
        raw_ts = first_row[col_idx]
        if hasattr(raw_ts, 'timestamp'):
            target = int(raw_ts.timestamp())
        elif isinstance(raw_ts, (int, float)):
            target = int(raw_ts)
        else:
            try:
                target = int(raw_ts)
            except (ValueError, TypeError):
                return 0
        idx = int(np.searchsorted(fts, target))
        if idx >= len(fts):
            return 0
        return idx

    def _build_forecast_window(self, episode_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        F = int(self.model.forecast_len)
        f_states = np.zeros((F, self.model.state_dim), dtype=np.float32)
        if self._forecast_map is not None and len(self._forecast_map) > 0:
            offset = self._forecast_npz_offset()
            fmap = self._forecast_map
            for fi in range(F):
                g_idx = offset + episode_step + fi
                if 0 <= g_idx < len(fmap):
                    f_states[fi, 5:11] = fmap[g_idx, 0, :6]
        f_rtgs = np.zeros((F, 1), dtype=np.float32)
        f_timesteps = np.zeros(F, dtype=np.int64)
        return f_states, f_rtgs, f_timesteps

    def _compute_fcas_norm_max(self) -> float:
        if not hasattr(self.env, 'aemo_data') or self.env.aemo_data is None:
            return 100.0
        try:
            df = self.env.aemo_data
            fcas_cols = [c for c in df.columns if c.startswith('FCAS_') and not c.endswith('_normalized')]
            if not fcas_cols:
                return 100.0
            max_vals = []
            for c in fcas_cols:
                mv = df[c].max()
                if mv is not None and not (isinstance(mv, float) and (mv != mv)):
                    max_vals.append(float(mv))
            return max(max_vals) if max_vals else 100.0
        except Exception:
            return 100.0

    # All 8 FCAS services (same order as env._fcas_services)
    _FCAS_SERVICES = [
        'RAISEREG', 'LOWERREG', 'RAISE6SEC', 'LOWER6SEC',
        'RAISE60SEC', 'LOWER60SEC', 'RAISE5MIN', 'LOWER5MIN',
    ]

    def _init_fcas_thresholds(self) -> None:
        # Legacy thresholds for multi_market (RAISEREG / LOWERREG only)
        self._fcas_raise_thresh: float | None = self.fcas_raise_threshold
        self._fcas_lower_thresh: float | None = self.fcas_lower_threshold

        # Per-service thresholds for full_fcas (same p80 percentile)
        self._fcas_thresholds: dict[str, float] = {}
        for svc in self._FCAS_SERVICES:
            if svc == 'RAISEREG':
                explicit = self.fcas_raise_threshold
            elif svc == 'LOWERREG':
                explicit = self.fcas_lower_threshold
            else:
                explicit = None
            self._fcas_thresholds[svc] = float(explicit) if explicit is not None else 0.0

        if not hasattr(self.env, 'aemo_data') or self.env.aemo_data is None:
            if self._fcas_raise_thresh is None: self._fcas_raise_thresh = 30.0
            if self._fcas_lower_thresh is None: self._fcas_lower_thresh = 25.0
            for svc in self._FCAS_SERVICES:
                if self._fcas_thresholds[svc] == 0.0:
                    self._fcas_thresholds[svc] = 30.0 if svc.startswith('RAISE') else 25.0
            return
        try:
            import numpy as np
            df = self.env.aemo_data
            for svc in self._FCAS_SERVICES:
                col = f'FCAS_{svc}'
                if col in df.columns and self._fcas_thresholds[svc] == 0.0:
                    prices = df[col].to_numpy()
                    prices = prices[~np.isnan(prices)]
                    if len(prices) > 0:
                        self._fcas_thresholds[svc] = float(np.percentile(prices, self.fcas_pctile * 100))
                    else:
                        self._fcas_thresholds[svc] = 30.0 if svc.startswith('RAISE') else 25.0
                elif self._fcas_thresholds[svc] == 0.0:
                    self._fcas_thresholds[svc] = 30.0 if svc.startswith('RAISE') else 25.0

            # Set legacy thresholds from the per-service thresholds
            if self._fcas_raise_thresh is None:
                self._fcas_raise_thresh = self._fcas_thresholds.get('RAISEREG', 30.0)
            if self._fcas_lower_thresh is None:
                self._fcas_lower_thresh = self._fcas_thresholds.get('LOWERREG', 25.0)
        except Exception:
            if self._fcas_raise_thresh is None: self._fcas_raise_thresh = 30.0
            if self._fcas_lower_thresh is None: self._fcas_lower_thresh = 25.0
            for svc in self._FCAS_SERVICES:
                if self._fcas_thresholds[svc] == 0.0:
                    self._fcas_thresholds[svc] = 30.0 if svc.startswith('RAISE') else 25.0

    def _init_oracle(self):
        """Solve Oracle LP at agent init using the env's full market data."""
        try:
            aemo_data = self.env.aemo_data
            if aemo_data is None:
                return

            # Map env action mode -> which FCAS services the Oracle price-vector covers.
            # The solver always solves for all 8 services, but zero-bids services
            # whose price columns are missing from the env's market frame.
            fcas_cols_present = [f'FCAS_{s}' for s in FCAS_SERVICES
                                 if f'FCAS_{s}' in aemo_data.columns]
            if not fcas_cols_present:
                print("  [Oracle] Error: no FCAS price columns in aemo_data")
                self._oracle_actions = None
                return
            missing = [f'FCAS_{s}' for s in FCAS_SERVICES
                       if f'FCAS_{s}' not in aemo_data.columns]
            if missing:
                print(f"  [Oracle] Warning: missing FCAS columns (zero bids): {missing}")

            env = self.env
            solver = AEMOOracleSolver(
                battery_capacity=float(env.battery_capacity),
                max_battery_flow=float(env.max_battery_flow),
                step_duration=float(env.step_duration),
                init_soc=float(env.init_battery_level),
                min_soc=0.0,
                max_soc=float(env.battery_capacity),
            )
            self._oracle_result = solver.solve(aemo_data, verbose=False)
            T = self._oracle_result.n_intervals
            max_flow = float(env.max_battery_flow)

            # Pack per-interval actions NORMALISED to env's action space [-1,1] and [0,1].
            # Env interprets: dispatch_mw = action[0] * max_flow (positive = charge, negative = discharge).
            # Oracle's dispatch convention is the opposite: positive = discharge, negative = charge.
            # So we negate when mapping to env's convention.
            # FCAS bid order must match the env's self._fcas_services order:
            #   ['RAISEREG', 'LOWERREG', 'RAISE6SEC', 'LOWER6SEC',
            #    'RAISE60SEC', 'LOWER60SEC', 'RAISE5MIN', 'LOWER5MIN']
            env_fcas_order = self.env._fcas_services if hasattr(self.env, '_fcas_services') else None
            actions = np.zeros((T, 1 + 8))
            actions[:, 0] = np.clip(-self._oracle_result.optimal_dispatch / max_flow, -1.0, 1.0)
            if env_fcas_order is not None:
                # Map Oracle's RAISE bids (indexed 0-3: 6SEC, 60SEC, 5MIN, REG) and
                # LOWER bids (0-3: 6SEC, 60SEC, 5MIN, REG) into env's order.
                ORAISE = {'RAISE6SEC': 0, 'RAISE60SEC': 1, 'RAISE5MIN': 2, 'RAISEREG': 3}
                OLOWER = {'LOWER6SEC': 0, 'LOWER60SEC': 1, 'LOWER5MIN': 2, 'LOWERREG': 3}
                for i, svc in enumerate(env_fcas_order):
                    if svc.startswith('RAISE'):
                        acts = self._oracle_result.optimal_raise_bids
                        oidx = ORAISE.get(svc, 0)
                    else:  # LOWER
                        acts = self._oracle_result.optimal_lower_bids
                        oidx = OLOWER.get(svc, 0)
                    actions[:, 1 + i] = np.clip(acts[:, oidx] / max_flow, 0.0, 1.0)
            else:
                actions[:, 1:5] = np.clip(self._oracle_result.optimal_raise_bids / max_flow, 0.0, 1.0)
                actions[:, 5:9] = np.clip(self._oracle_result.optimal_lower_bids / max_flow, 0.0, 1.0)
            self._oracle_actions = actions
            print(f"  [Oracle] Solved: ${self._oracle_result.total_profit:,.0f}/ep "
                  f"(energy ${self._oracle_result.energy_revenue:,.0f} "
                  f"+ FCAS ${self._oracle_result.fcas_revenue:,.0f}), "
                  f"{T} intervals")
        except Exception as e:
            print(f"  [Oracle] Init failed: {e}")
            import traceback; traceback.print_exc()
            self._oracle_actions = None

    def _oracle_action(self) -> list[float]:
        """Return precomputed action for the current env step."""
        if self._oracle_actions is None:
            return [0.0] * (1 + 8)

        # The env tracks the step count (0-indexed) via current_step.
        # episode_start_idx is the row in aemo_data where this episode starts.
        step = int(getattr(self.env, 'current_step', 0))
        step = max(0, min(step, len(self._oracle_actions) - 1))

        action = self._oracle_actions[step].tolist()
        return action

    def _init_soc_oracle(self):
        """Hierarchical DT+LP: predict target-SOC waypoints, then solve the
        SOC-waypoint-pinned Oracle LP once and cache per-step actions.

        The DT is expected to be a SOC-waypoint model: its action output is a
        K-dim vector of normalized target SOC at K evenly-spaced checkpoints
        (K = model.act_dim). The executor denormalizes to MWh, pins the SOC at
        those intervals in the Oracle LP (which co-optimizes energy + all 8
        FCAS within each segment), and replays the LP's per-step actions.
        """
        try:
            if self.model is None:
                raise ValueError("dt_soc_oracle requires a DT model.")
            aemo_data = self.env.aemo_data
            if aemo_data is None:
                return
            T = max(1, len(aemo_data))
            K = int(getattr(self.model, 'act_dim', 0))
            if K < 2:
                raise ValueError(f"dt_soc_oracle requires act_dim>=2 (got {K}).")

            # Predict the K normalized waypoints from the initial context.
            self.model.eval()
            params = list(self.model.parameters())
            device = params[0].device if params else torch.device('cpu')
            states, actions, rtgs, timesteps, mask = _build_dt_inference_context(
                self.model, self.dt_states_buffer, self.dt_actions_buffer,
                self.dt_rtgs_buffer, self.dt_timesteps_buffer,
            )
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
            rtgs = torch.tensor(rtgs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            timesteps = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                wp = self.model.get_action(
                    states, actions, rtgs, timesteps, attention_mask=attention_mask,
                )
            waypoints_norm = torch.nan_to_num(wp, nan=0.0, posinf=0.0, neginf=0.0)
            waypoints_norm = waypoints_norm.detach().cpu().numpy().astype(float)
            # get_action returns [B, T=1, act_dim] for batched input; squeeze.
            if waypoints_norm.ndim == 3:
                waypoints_norm = waypoints_norm[0, -1]
            if waypoints_norm.ndim == 0:
                waypoints_norm = np.array([float(waypoints_norm)])

            # Denormalize to MWh. DT was trained with normalized SOC targets in
            # [0,1] (index 17 of the obs = soc/capacity).
            capacity = float(self.env.battery_capacity)
            waypoints_mwh = np.clip(waypoints_norm, 0.0, 1.0) * capacity
            # Waypoint 0 is the SOC at the START of interval 0, which the env
            # fixes to init_battery_level. Force it (and clamp to capacity) so
            # the LP is always feasible at the boundary.
            init_soc = float(getattr(self.env, 'init_battery_level', capacity * 0.5))
            waypoints_mwh[0] = float(np.clip(init_soc, 0.0, capacity))
            waypoints_mwh = np.clip(waypoints_mwh, 0.0, capacity)
            self._soc_oracle_waypoints = waypoints_mwh.tolist()

            # Map K waypoints to interval indices spread across the episode
            # (waypoint 0 == episode start, last == terminal SOC at t=T).
            # K=2 -> [0, T]; K>2 -> include interior points.
            idxs = [int(round(i * T / max(1, K - 1))) for i in range(K)]
            idxs = [max(0, min(T, i)) for i in idxs]
            soc_waypoints = {t: float(waypoints_mwh[i]) for i, t in enumerate(idxs)}

            env = self.env
            solver = AEMOOracleSolver(
                battery_capacity=capacity,
                max_battery_flow=float(env.max_battery_flow),
                step_duration=float(env.step_duration),
                init_soc=float(env.init_battery_level),
                min_soc=0.0,
                max_soc=capacity,
            )
            result = solver.solve(aemo_data, verbose=False, soc_waypoints=soc_waypoints)
            if result.total_profit < -1e11:
                # Full-episode pinned LP infeasible (DT trajectory not
                # physically trackable). Fall back to per-segment solves: pin
                # only the terminal SOC of each segment so each sub-LP is
                # feasible (a single segment only needs to reach one SOC).
                print(f"  [dt_soc_oracle] full-episode LP infeasible "
                      f"({result.solver_message}); falling back to per-segment solves")
                result = self._solve_soc_segments(aemo_data, solver, idxs, waypoints_mwh)
            max_flow = float(env.max_battery_flow)
            env_fcas_order = getattr(self.env, '_fcas_services', None)
            n = max(1, result.n_intervals)
            actions_out = np.zeros((n, 1 + 8))
            actions_out[:, 0] = np.clip(-result.optimal_dispatch / max_flow, -1.0, 1.0)
            if env_fcas_order is not None:
                ORAISE = {'RAISE6SEC': 0, 'RAISE60SEC': 1, 'RAISE5MIN': 2, 'RAISEREG': 3}
                OLOWER = {'LOWER6SEC': 0, 'LOWER60SEC': 1, 'LOWER5MIN': 2, 'LOWERREG': 3}
                for i, svc in enumerate(env_fcas_order):
                    if svc.startswith('RAISE'):
                        acts = result.optimal_raise_bids
                        oidx = ORAISE.get(svc, 0)
                    else:
                        acts = result.optimal_lower_bids
                        oidx = OLOWER.get(svc, 0)
                    actions_out[:, 1 + i] = np.clip(acts[:, oidx] / max_flow, 0.0, 1.0)
            else:
                actions_out[:, 1:5] = np.clip(result.optimal_raise_bids / max_flow, 0.0, 1.0)
                actions_out[:, 5:9] = np.clip(result.optimal_lower_bids / max_flow, 0.0, 1.0)
            self._soc_oracle_actions = actions_out
            print(f"  [dt_soc_oracle] waypoints={[round(w,1) for w in waypoints_mwh]} "
                  f"-> ${result.total_profit:,.0f}/ep (E ${result.energy_revenue:,.0f} "
                  f"+ F ${result.fcas_revenue:,.0f})")
        except Exception as e:
            print(f"  [dt_soc_oracle] Init failed: {e}")
            import traceback; traceback.print_exc()
            self._soc_oracle_actions = None

    def _soc_oracle_action(self) -> list[float]:
        if self._soc_oracle_actions is None:
            return [0.0] * (1 + 8)
        step = int(getattr(self.env, 'current_step', 0))
        step = max(0, min(step, len(self._soc_oracle_actions) - 1))
        return self._soc_oracle_actions[step].tolist()

    def _solve_soc_segments(
        self,
        aemo_data: pl.DataFrame,
        solver: AEMOOracleSolver,
        idxs: list[int],
        waypoints_mwh: np.ndarray,
    ) -> OracleResult:
        """Fallback executor: solve one SOC-pinned LP per segment.

        Each segment [t_i, t_{i+1}] is solved independently with its terminal
        SOC pinned, so every segment is feasible by construction (a single
        segment only needs to reach one SOC within ramp limits). Concatenates
        the per-segment actions into a full-episode action plan. Returns the
        best-effort OracleResult (revenue summed, dispatch concatenated).
        """
        env = self.env
        max_flow = float(env.max_battery_flow)
        env_fcas_order = getattr(self.env, '_fcas_services', None)
        T = max(1, len(aemo_data))
        n = max(1, T)
        actions_out = np.zeros((n, 1 + 8))
        ORAISE = {'RAISE6SEC': 0, 'RAISE60SEC': 1, 'RAISE5MIN': 2, 'RAISEREG': 3}
        OLOWER = {'LOWER6SEC': 0, 'LOWER60SEC': 1, 'LOWER5MIN': 2, 'LOWERREG': 3}

        total_profit = 0.0
        energy_rev = 0.0
        fcas_rev = 0.0
        dispatch_all: list[np.ndarray] = []
        raise_all: list[np.ndarray] = []
        lower_all: list[np.ndarray] = []
        soc_all: list[np.ndarray] = []
        last_soc = float(env.init_battery_level)
        last_idx = 0

        for seg in range(len(idxs) - 1):
            t0, t1 = idxs[seg], idxs[seg + 1]
            if t1 <= t0:
                continue
            seg_data = aemo_data.slice(t0, t1 - t0)
            seg_solver = AEMOOracleSolver(
                battery_capacity=float(env.battery_capacity),
                max_battery_flow=max_flow,
                step_duration=float(env.step_duration),
                init_soc=last_soc,
                min_soc=0.0,
                max_soc=float(env.battery_capacity),
            )
            seg_wp = {t1 - t0: float(waypoints_mwh[seg + 1])}
            res = seg_solver.solve(seg_data, verbose=False, soc_waypoints=seg_wp)
            if res.total_profit < -1e11:
                # Segment itself infeasible: pin the terminal SOC to the last
                # achievable value (clamp within ramp reach from current SOC).
                max_delta = max_flow * float(env.step_duration) * (t1 - t0)
                target = float(np.clip(waypoints_mwh[seg + 1],
                                       last_soc - max_delta, last_soc + max_delta))
                res = seg_solver.solve(seg_data, verbose=False,
                                       soc_waypoints={t1 - t0: target})
                if res.total_profit < -1e11:
                    # Last resort: flat SOC segment (no net dispatch).
                    flat = AEMOOracleSolver(
                        battery_capacity=float(env.battery_capacity),
                        max_battery_flow=max_flow,
                        step_duration=float(env.step_duration),
                        init_soc=last_soc,
                        min_soc=0.0,
                        max_soc=float(env.battery_capacity),
                    )
                    res = flat.solve(seg_data, verbose=False,
                                     soc_waypoints={t1 - t0: last_soc})
                    if res.total_profit < -1e11:
                        continue

            seg_len = res.n_intervals
            actions_out[last_idx:last_idx + seg_len, 0] = np.clip(
                -res.optimal_dispatch / max_flow, -1.0, 1.0)
            if env_fcas_order is not None:
                for i, svc in enumerate(env_fcas_order):
                    if svc.startswith('RAISE'):
                        acts = res.optimal_raise_bids
                        oidx = ORAISE.get(svc, 0)
                    else:
                        acts = res.optimal_lower_bids
                        oidx = OLOWER.get(svc, 0)
                    actions_out[last_idx:last_idx + seg_len, 1 + i] = np.clip(
                        acts[:, oidx] / max_flow, 0.0, 1.0)
            else:
                actions_out[last_idx:last_idx + seg_len, 1:5] = np.clip(
                    res.optimal_raise_bids / max_flow, 0.0, 1.0)
                actions_out[last_idx:last_idx + seg_len, 5:9] = np.clip(
                    res.optimal_lower_bids / max_flow, 0.0, 1.0)

            total_profit += res.total_profit
            energy_rev += res.energy_revenue
            fcas_rev += res.fcas_revenue
            dispatch_all.append(res.optimal_dispatch)
            raise_all.append(res.optimal_raise_bids)
            lower_all.append(res.optimal_lower_bids)
            soc_all.append(res.optimal_soc[:-1])
            last_soc = float(res.optimal_soc[-1])
            last_idx += seg_len

        dispatch = np.concatenate(dispatch_all) if dispatch_all else np.zeros(n)
        raise_bid = np.concatenate(raise_all) if raise_all else np.zeros((n, 4))
        lower_bid = np.concatenate(lower_all) if lower_all else np.zeros((n, 4))
        soc = np.concatenate(soc_all + [np.array([last_soc])]) if soc_all else np.full(n + 1, last_soc)
        return OracleResult(
            total_profit=float(total_profit),
            energy_revenue=float(energy_rev),
            fcas_revenue=float(fcas_rev),
            total_dispatch_mwh=float(np.clip(dispatch, 0, None).sum() * float(env.step_duration)),
            total_charge_mwh=float(np.clip(-dispatch, 0, None).sum() * float(env.step_duration)),
            n_intervals=int(dispatch.shape[0]),
            optimal_dispatch=dispatch,
            optimal_raise_bids=raise_bid,
            optimal_lower_bids=lower_bid,
            optimal_soc=soc,
            per_step_fcas_revenue=np.zeros(dispatch.shape[0]),
            per_step_energy_revenue=np.zeros(dispatch.shape[0]),
            solver_status=0,
            solver_message="segmented fallback",
        )

    def choose_action(self, obs):
        if self.algorithm == 'aemo_oracle':
            return self._oracle_action()
        if self.algorithm == 'dt_soc_oracle':
            return self._soc_oracle_action()
        if self.algorithm in ('rule', 'fcas_rule'):
            return self.fcas_rule_based_action(obs) if self.algorithm == 'fcas_rule' else self.rule_based_action(obs)
        if self.algorithm == 'dispatch':
            return self._dispatch_action()
        if self.algorithm == 'rl':
            if self.model is None:
                raise ValueError("RL algorithm selected but no model provided.")
            if not isinstance(obs, np.ndarray):
                warnings.warn("Observation is not a numpy array. RL model expects a numpy array input.")
                return [0.0]
            action, _ = self.model.predict(obs, deterministic=True)
            return action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action
        if self.algorithm == 'dt':
            if self.model is None:
                raise ValueError("Decision Transformer selected but no model provided.")
            self.model.eval()
            device = next(self.model.parameters()).device

            states, actions, rtgs, timesteps, mask = _build_dt_inference_context(
                self.model, self.dt_states_buffer, self.dt_actions_buffer,
                self.dt_rtgs_buffer, self.dt_timesteps_buffer,
            )
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)
            rtgs = torch.tensor(rtgs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            timesteps = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            is_forecast = hasattr(self.model, 'forecast_len') and self.model.forecast_len > 0
            if is_forecast:
                self._ensure_forecast_map()
                # Training sliding window: history at start_idx..start_idx+T-1,
                # forecast at start_idx+T..start_idx+T+F-1. At inference the
                # right-aligned buffer means start_idx = max(0, buffer_len-T),
                # so the forecast must be at max(T, buffer_len), not pinned
                # at context_len (which leaves a stale forecast for 88% of
                # a 1728-step episode).
                context_len = self.model.context_len
                buffer_len = len(self.dt_states_buffer)
                episode_start = int(getattr(self.env, 'episode_start_idx', 0))
                forecast_ep_step = episode_start + max(context_len, buffer_len)
                f_states, f_rtgs, f_timesteps = self._build_forecast_window(forecast_ep_step)
                f_states = torch.tensor(f_states, dtype=torch.float32, device=device).unsqueeze(0)
                f_rtgs = torch.tensor(f_rtgs, dtype=torch.float32, device=device).unsqueeze(0)
                f_timesteps = torch.tensor(f_timesteps, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad():
                    action = self.model.get_action(
                        states, actions, rtgs, timesteps, attention_mask=attention_mask,
                        forecast_states=f_states, forecast_rtgs=f_rtgs,
                        forecast_timesteps=f_timesteps,
                    )
            else:
                with torch.no_grad():
                    action = self.model.get_action(
                        states, actions, rtgs, timesteps, attention_mask=attention_mask,
                    )
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            action = action.detach().cpu().numpy()
            action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            if action.ndim > 1 and action.shape[0] == 1:
                action = action[0]
            if action.ndim == 0:
                action = np.array([float(action)], dtype=np.float32)
            if self._action_is_full_fcas():
                action = self._clip_fcas_dims(action)
            return action.tolist()

        raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _action_is_full_fcas(self) -> bool:
        action_mode = getattr(self.env, 'action_mode', 'multi_market')
        return action_mode == 'full_fcas' and getattr(self.model, 'act_dim', 0) >= 2

    def _clip_fcas_dims(self, action: np.ndarray) -> np.ndarray:
        """Clip the 8 FCAS bid dims (indices 1..8) of a 9-dim full_fcas action to [0, 1].

        Energy dim 0 stays in [-1, 1]. This is a no-op for the 'mixed' head (which
        already emits FCAS in [0, 1] via Sigmoid) but corrects Tanh-head models whose
        FCAS predictions can be negative (the env would otherwise treat them as 0 via
        max(0, ...) but leaving them negative wastes headroom signal in buffers).
        """
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1 and action.shape[0] >= 2:
            out = action.copy()
            out[1:] = np.clip(out[1:], 0.0, 1.0)
            return out
        return action

    def rule_based_action(self, obs):
        # AEMO raw obs layout: [time(5), RRP, TOTALDEMAND, FCAS(8), GEN(2), SOC]
        if obs is None or len(obs) < 6:
            return [np.float32(0.0)]

        price = float(obs[5])
        soc_kwh = float(obs[-1])
        cap = float(getattr(self.env, 'battery_capacity', 1.0))
        soc_norm = soc_kwh / cap if cap > 0 else 0.0

        charge_price = 30.0
        discharge_price = 120.0

        noise = np.random.normal(0.0, 0.01)
        action = 0.0

        safe_low, safe_high = 0.10, 0.90
        if soc_norm <= safe_low:
            action = 0.5
        elif soc_norm >= safe_high:
            action = -0.5
        else:
            if price <= charge_price:
                action = 0.6
            elif price >= discharge_price:
                action = -0.6
            else:
                action = 0.0

        action_val = float(np.clip(action + noise, -1.0, 1.0))
        action_mode = getattr(self.env, 'action_mode', 'simple')
        if action_mode == 'full_fcas':
            return np.array([np.float32(action_val)] + [np.float32(0.0)] * 8, dtype=np.float32)
        elif action_mode == 'multi_market':
            fcas_raise = np.float32(0.0)
            fcas_lower = np.float32(0.0)
            return np.array([np.float32(action_val), fcas_raise, fcas_lower], dtype=np.float32)
        return np.array([np.float32(action_val)], dtype=np.float32)

    def fcas_rule_based_action(self, obs):
        if obs is None or len(obs) < 17:
            return [np.float32(0.0)]
        # obs: [time(5), RRP, DEMAND, FCAS(8), GEN(2), SOC] — normalized [0,1]
        price_norm = float(obs[5])
        soc_norm = float(obs[-1])
        fcas_norm = [float(obs[7 + i]) for i in range(8)]
        _F = ['RAISEREG','LOWERREG','RAISE6SEC','LOWER6SEC','RAISE60SEC','LOWER60SEC','RAISE5MIN','LOWER5MIN']

        # Energy dispatch (normalized thresholds from raw $30/$120 ÷ RRP max ~16600)
        charge_thresh_n = 0.060  # ~$30 / 500
        discharge_thresh_n = 0.80  # ~$120 / 150 ... actually let me use raw values
        # Denormalize RRP: norm = raw / max, so raw = norm * max. Use obs[5] as normalized RRP
        # Actually the existing rule uses raw obs values, so let's denomralize
        rrp_max = getattr(self.env, '_raw_col_bounds', {}).get('RRP', (-100, 500))[1]
        rrp_min = getattr(self.env, '_raw_col_bounds', {}).get('RRP', (-100, 500))[0]
        price_raw = price_norm * (rrp_max - rrp_min) + rrp_min
        charge_price = 30.0
        discharge_price = 120.0

        noise = np.random.normal(0.0, 0.01)
        safe_low, safe_high = 0.10, 0.90
        if soc_norm <= safe_low:
            action = 0.5
        elif soc_norm >= safe_high:
            action = -0.5
        else:
            if price_raw <= charge_price:
                action = 0.6
            elif price_raw >= discharge_price:
                action = -0.6
            else:
                action = 0.0
        action_val = float(np.clip(action + noise, -1.0, 1.0))

        action_mode = getattr(self.env, 'action_mode', 'simple')
        if action_mode == 'full_fcas':
            # All 8 FCAS services with per-service p80 thresholds
            bids = []
            for i, svc in enumerate(self._FCAS_SERVICES):
                raw_price = fcas_norm[i] * self._fcas_norm_max
                thresh = self._fcas_thresholds.get(svc, 30.0)
                if svc.startswith('RAISE'):
                    bid = 1.0 if raw_price >= thresh and soc_norm > 0.15 else 0.0
                else:
                    bid = 1.0 if raw_price >= thresh and soc_norm < 0.85 else 0.0
                bids.append(np.float32(bid))
            return np.array([np.float32(action_val)] + bids, dtype=np.float32)

        if action_mode != 'multi_market':
            return np.array([np.float32(action_val)], dtype=np.float32)

        # FCAS bidding — denormalise FCAS prices using _fcas_norm_max
        # obs FCAS entries are normalised as raw / _fcas_norm_max
        fcas_raise_raw = fcas_norm[0] * self._fcas_norm_max  # RAISEREG
        fcas_lower_raw = fcas_norm[1] * self._fcas_norm_max  # LOWERREG

        fcas_raise = 1.0 if fcas_raise_raw >= self._fcas_raise_thresh and soc_norm > 0.15 else 0.0
        fcas_lower = 1.0 if fcas_lower_raw >= self._fcas_lower_thresh and soc_norm < 0.85 else 0.0
        return np.array([np.float32(action_val), np.float32(fcas_raise), np.float32(fcas_lower)], dtype=np.float32)

    def run_episode(self, render: bool = False, display_progress: bool = False):
        obs, info = _reset_env(self.env, seed=self.reset_seed, options=self.reset_options)
        raw_obs = self.env.get_raw_obs()
        max_possible_steps = len(self.env.aemo_data) if hasattr(self.env, 'aemo_data') else 0

        if self.algorithm == 'dt':
            self.dt_states_buffer = [obs.copy()]
            self.dt_actions_buffer = [np.zeros(self.model.act_dim)]
            self.dt_rtgs_buffer = [self.rtg_value]
            self.dt_timesteps_buffer = [self.env.current_step]

        logs = []
        terminated, truncated = False, False
        step = 0

        if self.algorithm in ['rule', 'dispatch']:
            current_obs = raw_obs
        else:
            current_obs = obs

        try:
            while not (terminated or truncated):
                action = self.choose_action(current_obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                logs.append({
                    'step': step,
                    'norm_observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
                    'raw_observation': raw_obs.tolist() if isinstance(raw_obs, np.ndarray) else raw_obs,
                    'action': action,
                    'reward': reward,
                    'info': info
                })

                if self.algorithm == 'dt':
                    context_len = self.model.context_len
                    if isinstance(action, list):
                        action_array = np.array(action, dtype=np.float32)
                    else:
                        action_array = np.array([action], dtype=np.float32) if np.isscalar(action) else action
                    self.dt_actions_buffer[-1] = action_array

                    next_rtg = stable_rtg_update(
                        self.dt_rtgs_buffer[-1], reward,
                        dt_gamma=self.dt_gamma, initial_rtg=self.rtg_value,
                    )

                    self.dt_states_buffer.append(next_obs.copy())
                    self.dt_actions_buffer.append(np.zeros(self.model.act_dim))
                    self.dt_rtgs_buffer.append(next_rtg)
                    self.dt_timesteps_buffer.append(self.env.current_step)

                    if len(self.dt_states_buffer) > context_len:
                        self.dt_states_buffer = self.dt_states_buffer[-context_len:]
                        self.dt_actions_buffer = self.dt_actions_buffer[-context_len:]
                        self.dt_rtgs_buffer = self.dt_rtgs_buffer[-context_len:]
                        self.dt_timesteps_buffer = self.dt_timesteps_buffer[-context_len:]

                raw_obs = self.env.get_raw_obs()
                obs = next_obs
                current_obs = raw_obs if self.algorithm in ['rule', 'dispatch'] else obs

                if render:
                    self.env.render()

                step += 1
                if display_progress:
                    print(f"Step {step}/{max_possible_steps}", end='\r')
        finally:
            print("Sim Complete")

        episode_df = pl.DataFrame(logs)
        incident_df = pl.DataFrame()
        return episode_df, incident_df



def run_single(agent_class, env, agent_kwargs, render, display_progress=False):
    agent = agent_class(env, **agent_kwargs)
    episode_df, incident_df = agent.run_episode(
        render=render,
        display_progress=display_progress
    )
    return episode_df, incident_df

def run_single_with_logging(agent_class, env, agent_kwargs, render, idx, display_progress=False):
    import time
    print(f"[START] Episode {idx}")
    start = time.time()
    episode_df, incident_df = run_single(agent_class, env, agent_kwargs, render, display_progress=display_progress)
    elapsed = time.time() - start
    print(f"[DONE]  Episode {idx} (Elapsed: {elapsed:.2f} sec)")
    return episode_df, incident_df


# this can be used to run multiple episodes in parallel
def run_episodes_parallel(agent_class, envs, agent_kwargs=None, render=False, max_workers=4, use_notebook_tqdm=False, display_indi_prog=False):
    """
    Runs one episode per environment in parallel.
    agent_class: The Agent class to instantiate. 
    envs: List of SolarBatteryEnv instances.
    agent_kwargs: Dict of kwargs for Agent constructor, or a list of per-environment
        kwargs dicts with the same length as envs.
    use_notebook_tqdm: If True, use tqdm.notebook.tqdm; else use tqdm.tqdm (for scripts)
    Returns: List of DataFrames (one per environment).
    """
    if agent_kwargs is None:
        agent_kwargs_list = [{} for _ in envs]
    elif isinstance(agent_kwargs, list):
        if len(agent_kwargs) != len(envs):
            raise ValueError("When agent_kwargs is a list, it must have the same length as envs.")
        agent_kwargs_list = [kwargs or {} for kwargs in agent_kwargs]
    else:
        shared_kwargs = agent_kwargs or {}
        agent_kwargs_list = [shared_kwargs for _ in envs]

    allowed_algorithms = {'rule', 'sdp', 'mrdp', 'dt', 'oracle', 'dispatch'}
    for kwargs in agent_kwargs_list:
        algorithm = kwargs.get('algorithm', 'rule').lower()
        if algorithm not in allowed_algorithms:
            raise ValueError("Parallel execution is only supported for 'rule', 'sdp', 'mrdp', 'dt', 'oracle', and 'dispatch' algorithms. ")

    if use_notebook_tqdm:
        from tqdm.notebook import tqdm as tqdm_bar
    else:
        from tqdm import tqdm as tqdm_bar

    episode_logs = []
    incident_logs = []

    print(f"[INFO] Starting {len(envs)} episodes with max_workers={max_workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_with_logging, agent_class, env, kwargs, render, idx, display_indi_prog)
            for idx, (env, kwargs) in enumerate(zip(envs, agent_kwargs_list))
        ]
        for f in tqdm_bar(concurrent.futures.as_completed(futures), total=len(futures), desc="Episodes"):
            ep_df, inc_df = f.result()
            episode_logs.append(ep_df)
            incident_logs.append(inc_df)
    print(f"[INFO] All episodes complete.")
    return episode_logs, incident_logs

# This function runs a trained SB3 model on a vectorized environment and collects episode trajectories.
def run_sb3_model_on_vec_env(model, vec_env, deterministic=False, max_steps=None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    import json
    """
    Runs a trained SB3 model on a vectorized environment and collects episode trajectories.
    Args:
        model: Trained SB3 model (e.g., PPO, A2C, etc.)
        vec_env: Vectorized environment (DummyVecEnv or SubprocVecEnv)
        deterministic: Whether to use deterministic actions
        max_steps: Optional maximum number of steps to run (default: run until all envs are done)
    Returns:
        List of dicts, one per environment, each containing lists of 'obs', 'actions', 'rewards', 'infos'
    """

    if not hasattr(model, 'predict'):
        raise ValueError("The provided model does not have a 'predict' method. Ensure it is a valid SB3 model.")
    if not isinstance(vec_env, (DummyVecEnv, SubprocVecEnv)):
        raise ValueError("The provided vec_env must be a DummyVecEnv or SubprocVecEnv instance.")
    num_envs = vec_env.num_envs
    obs = vec_env.reset()
    dones = np.zeros(num_envs, dtype=bool)
    episode_data = [
        {'norm_observation':[], 'raw_observation': [], 'actions': [], 'rewards': [], 'infos': []}
        for _ in range(num_envs)
    ]
    steps = 0

    while not dones.all():
        raw_obs_list = vec_env.env_method('get_raw_obs')
        actions, _ = model.predict(obs, deterministic=deterministic)
        next_obs, rewards, next_dones, infos = vec_env.step(actions)
        not_done_indices = np.where(dones == False)[0]
        for i in not_done_indices:
            # Convert data to Parquet-compatible types at collection time
            episode_data[i]['norm_observation'].append(obs[i].tolist())
            episode_data[i]['raw_observation'].append(raw_obs_list[i].tolist() if isinstance(raw_obs_list[i], np.ndarray) else raw_obs_list[i])
            episode_data[i]['actions'].append(actions[i].tolist() if isinstance(actions[i], np.ndarray) else actions[i])
            episode_data[i]['rewards'].append(float(rewards[i]))
            
            # The info dict from SB3 can contain non-serializable types.
            # We filter for basic types and convert the rest to strings before JSON serialization.
            serializable_info = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in infos[i].items()}
            episode_data[i]['infos'].append(json.dumps(serializable_info))

            if next_dones[i]:
                dones[i] = True
        obs = next_obs
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    return episode_data