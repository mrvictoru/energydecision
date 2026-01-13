import logging
import numpy as np
import torch
import polars as pl
import warnings
from typing import Optional
from EnergySimEnv import SolarBatteryEnv, VIOLATION_PENALTY

from batterydeg import DegradationModel, RainflowCounter
from quantile_scenarios import QuantileScenarioGenerator

# Import self-contained algorithm classes
from sdp_algorithm import SDPSolver
from mrdp_algorithm import MRDPSolver
from oracle_algorithm import OracleSolver

import concurrent.futures
from tqdm.notebook import tqdm


class Agent:
    def __init__(self, env: SolarBatteryEnv, algorithm='rule', model=None,
                 horizon=48, soc_resolution=20, action_resolution=41, static_deg_correction_factor=0.8,
                 degradation_model='rainflow', linear_deg_cost_p_kwh=None,
                 use_monte_carlo: bool = True, mc_samples: int = 200, mc_seed: Optional[int] = None,
                 subhorizon_specs=None, rtg_value: float = 0.0, dt_gamma: float = 0.99):
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
            self.static_deg_correction_factor = static_deg_correction_factor

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
                        static_deg_correction_factor=static_deg_correction_factor,
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
                    static_deg_correction_factor=static_deg_correction_factor,
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
        
        

    @staticmethod
    def _safe_float32_array(data, default_shape=(0,), dtype=np.float32):
        """Convert a list of RTGs to a safe float32 numpy array."""
        if not data:
            return np.zeros(default_shape, dtype=dtype)
        arr = np.array(data, dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.all():
            logging.warning(
                "Found %d non-finite RTG values; replacing with 0 before casting.",
                int((~finite).sum())
            )
            arr = np.where(finite, arr, 0.0)
        clip_max = np.finfo(np.float32).max
        arr = np.clip(arr, -clip_max, clip_max)
        return arr.astype(dtype)

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
            
            # Build inputs from rolling buffers
            context_len = self.model.context_len
            state_dim = self.model.state_dim
            act_dim = self.model.act_dim
            buffer_len = len(self.dt_states_buffer)

            buffer_states = (
                np.array(self.dt_states_buffer, dtype=np.float32)
                if buffer_len > 0 else np.zeros((0, state_dim), dtype=np.float32)
            )
            if buffer_states.ndim == 1 and buffer_len > 0:
                buffer_states = buffer_states.reshape(buffer_len, state_dim)

            buffer_actions = (
                np.array(self.dt_actions_buffer, dtype=np.float32)
                if buffer_len > 0 else np.zeros((0, act_dim), dtype=np.float32)
            )
            if buffer_actions.ndim == 1 and buffer_len > 0:
                buffer_actions = buffer_actions.reshape(buffer_len, act_dim)

            buffer_rtgs = self._safe_float32_array(
                self.dt_rtgs_buffer,
                default_shape=(0, ),
                dtype=np.float32
            ) if buffer_len > 0 else np.zeros(0, dtype=np.float32)

            buffer_timesteps = (
                np.array(self.dt_timesteps_buffer, dtype=np.int64)
                if buffer_len > 0 else np.zeros(0, dtype=np.int64)
            )
            
            # Prepare tensors with left-padding if buffer is shorter than context_len
            if buffer_len < context_len:
                # Left-pad with zeros
                pad_len = context_len - buffer_len
                states = np.vstack([
                    np.zeros((pad_len, state_dim), dtype=np.float32),
                    buffer_states
                ])
                actions = np.vstack([
                    np.zeros((pad_len, act_dim), dtype=np.float32),
                    buffer_actions
                ])
                rtgs = np.concatenate([
                    np.zeros(pad_len, dtype=np.float32),
                    buffer_rtgs
                ])
                timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), buffer_timesteps])
                # Attention mask: 0 for padding, 1 for valid
                mask = np.concatenate([
                    np.zeros(pad_len, dtype=np.bool_),
                    np.ones(buffer_len, dtype=np.bool_)
                ])
            else:
                # Use the last context_len items
                states = buffer_states[-context_len:]
                actions = buffer_actions[-context_len:]
                rtgs = buffer_rtgs[-context_len:]
                timesteps = buffer_timesteps[-context_len:]
                mask = np.ones(context_len, dtype=np.bool_)
            
            # Apply return_scale to RTGs (matching training behavior)
            return_scale_attr = getattr(self.model, 'return_scale', 1.0)
            if isinstance(return_scale_attr, torch.Tensor):
                return_scale = float(return_scale_attr.detach().cpu().item())
            else:
                return_scale = float(return_scale_attr)

            # Guard against invalid scaling factors that would introduce NaNs
            if not np.isfinite(return_scale) or abs(return_scale) < 1e-12:
                raise ValueError(f"Invalid Decision Transformer return_scale: {return_scale}")
            if return_scale != 1.0:
                rtgs = rtgs / return_scale

            # Sanitize numeric inputs
            states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
            actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            rtgs = np.nan_to_num(rtgs, nan=0.0, posinf=0.0, neginf=0.0)

            # Clip RTGs to stay within a safe numerical range (tuned to dataset scale if needed)
            rtg_clip = 1e3
            rtgs = np.clip(rtgs, -rtg_clip, rtg_clip)

            # Clamp timesteps to embedding range
            max_time = getattr(self.model.embed_timestep, 'num_embeddings', None)
            if max_time is not None and max_time > 0:
                timesteps = np.clip(timesteps, 0, max_time - 1)
            
            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, state_dim]
            actions = torch.tensor(actions, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, act_dim]
            rtgs = torch.tensor(rtgs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            timesteps = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
            attention_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)  # [1, T]
            
            # Get action prediction using get_action() helper
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
                print("Warning: Not enough forecast data for full horizon. Using rule-based action.")
                return self.rule_based_action(obs)
            
            # Solve using self-contained SDP solver
            policy_table = self.sdp_solver.solve(forecasts, start_index=current_step_env)
            
            # Get action for current state
            soc_idx = np.argmin(np.abs(self.soc_levels_kwh - current_soc_kwh))
            optimal_action_idx = policy_table[0, soc_idx]
            
            if optimal_action_idx == -1:
                print(f"Warning: No optimal action found for SoC {current_soc_kwh:.2f}. Using zero action.")
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
                print("Warning: Not enough forecast data for full horizon. Using rule-based action.")
                return self.rule_based_action(obs)
            
            # Solve using self-contained MRDP solver
            policy_table = self.mrdp_solver.solve(forecasts, start_index=current_step_env)
            
            # Get action for current state
            soc_idx = np.argmin(np.abs(self.soc_levels_kwh - current_soc_kwh))
            optimal_action_idx = policy_table[0, soc_idx]
            
            if optimal_action_idx == -1:
                print(f"Warning: No optimal action found for SoC {current_soc_kwh:.2f}. Using zero action.")
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
        obs, info = self.env.reset()
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
                    
                    # Update RTG using discounted recurrence: R_{t+1} = (R_t - r_t) / gamma
                    if self.dt_gamma == 1.0:
                        # Undiscounted: R_{t+1} = R_t - r_t
                        next_rtg = self.dt_rtgs_buffer[-1] - reward
                    else:
                        # Discounted: R_{t+1} = (R_t - r_t) / gamma
                        next_rtg = (self.dt_rtgs_buffer[-1] - reward) / self.dt_gamma
                    
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

        return pl.DataFrame(logs)


def run_single(agent_class, env, agent_kwargs, render, display_progress=False):
    agent = agent_class(env, **agent_kwargs)
    return agent.run_episode(render=render, display_progress=display_progress)

def run_single_with_logging(agent_class, env, agent_kwargs, render, idx, display_progress=False):
    import time
    print(f"[START] Episode {idx}")
    start = time.time()
    result = run_single(agent_class, env, agent_kwargs, render, display_progress=display_progress)
    elapsed = time.time() - start
    print(f"[DONE]  Episode {idx} (Elapsed: {elapsed:.2f} sec)")
    return result

# this can be used to run multiple episodes in parallel
def run_episodes_parallel(agent_class, envs, agent_kwargs=None, render=False, max_workers=4, use_notebook_tqdm=False, display_indi_prog=False):
    """
    Runs one episode per environment in parallel.
    agent_class: The Agent class to instantiate. 
    envs: List of SolarBatteryEnv instances.
    agent_kwargs: Dict of kwargs for Agent constructor. (only suitable for rule, sdp algorithms)
    use_notebook_tqdm: If True, use tqdm.notebook.tqdm; else use tqdm.tqdm (for scripts)
    Returns: List of DataFrames (one per environment).
    """
    agent_kwargs = agent_kwargs or {}
    if agent_kwargs.get('algorithm', 'rule').lower() not in ['rule', 'sdp', 'mrdp','dt', 'oracle']:
        raise ValueError("Parallel execution is only supported for 'rule', 'sdp', 'mrdp', 'dt', and 'oracle' algorithms. ")

    if use_notebook_tqdm:
        from tqdm.notebook import tqdm as tqdm_bar
    else:
        from tqdm import tqdm as tqdm_bar

    results = []
    print(f"[INFO] Starting {len(envs)} episodes with max_workers={max_workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_with_logging, agent_class, env, agent_kwargs, render, idx, display_indi_prog) for idx, env in enumerate(envs)]
        for f in tqdm_bar(concurrent.futures.as_completed(futures), total=len(futures), desc="Episodes"):
            results.append(f.result())
    print(f"[INFO] All episodes complete.")
    return results

# This function runs a trained SB3 model on a vectorized environment and collects episode trajectories.
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import json
def run_sb3_model_on_vec_env(model, vec_env, deterministic=False, max_steps=None):
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