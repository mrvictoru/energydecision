import numpy as np
import torch
import polars as pl
import warnings
from EnergySimEnv import SolarBatteryEnv

from batterydeg import static_degradation


# Helper functions for SDP implementation
def interpolate_ctg(soc_levels_kwh, ctg_array, soc_value):
    """
    Linearly interpolate cost-to-go values for a continuous SoC between discrete levels.
    Clamps at the ends if soc_value is outside the range.
    
    Args:
        soc_levels_kwh: Array of discrete SoC levels in kWh
        ctg_array: Array of cost-to-go values corresponding to soc_levels_kwh
        soc_value: Continuous SoC value to interpolate at
    
    Returns:
        Interpolated cost-to-go value
    """
    # Clamp soc_value to the bounds of soc_levels_kwh
    soc_value = np.clip(soc_value, soc_levels_kwh[0], soc_levels_kwh[-1])
    
    # Use numpy's interp function for linear interpolation
    return np.interp(soc_value, soc_levels_kwh, ctg_array)


def compute_grid_cost(grid_energy, import_price, export_price, max_grid_energy):
    """
    Compute grid cost with explicit import/export semantics and grid limit checking.
    
    Args:
        grid_energy: Grid energy (positive = import, negative = export)
        import_price: Price per kWh for importing energy
        export_price: Price per kWh for exporting energy (revenue)
        max_grid_energy: Maximum allowed grid energy (absolute value)
    
    Returns:
        Grid cost (positive = cost, negative = revenue, np.inf if limit exceeded)
    """
    # Check grid limits first
    if abs(grid_energy) > max_grid_energy + 1e-6:  # Add small tolerance
        return np.inf
    
    if grid_energy > 0:  # Importing energy
        return grid_energy * import_price
    else:  # Exporting energy (grid_energy is negative)
        # Export generates revenue, so cost is negative (revenue reduces total cost)
        export_revenue = abs(grid_energy) * export_price
        return -export_revenue


class Agent:
    def __init__(self, env: SolarBatteryEnv, algorithm='rule', model=None,
                horizon=48, soc_resolution=20, action_resolution=11, static_deg_correction_factor = 0.01,# Added SDP params
                degradation_model = 'linear', linear_deg_cost_p_kwh=None):
        """
        env: an instance of SolarBatteryEnv.
        algorithm: choose between 'rule', 'rl', 'dt', or 'sdp'.
        model: For RL/DT algorithm, a trained model.
        horizon: Time horizon for SDP optimization (default: 48 steps = 24 hours).
        soc_resolution: Resolution of state-of-charge discretization (default: 20 levels).
        action_resolution: Resolution of action discretization (default: 11 levels, e.g., -1.0, -0.8, ..., 0.8, 1.0).
        """
        self.env = env
        self.algorithm = algorithm.lower()
        self.model = model
        self.rule_presistence = False  # Preset for rule-based action persistence

        if self.algorithm == 'sdp': # Stochastic Dynamic Programming
        # The following algorithm is derived from the paper "Optimal Operation of Energy Storage Systems Considering Forecasts and Battery Degradation (2018)"
        # by K Abdulla, J De Hoog, V Muenzel, F Suits, K Steer, A Wirth, S Halgamuge 
            self.horizon = horizon
            self.soc_resolution = soc_resolution
            self.action_resolution = action_resolution
            self.static_deg_correction_factor = static_deg_correction_factor

            self.degradation_model = degradation_model
            if self.degradation_model == 'linear':
                # If not provided, default to battery_life_cost / (battery_capacity * cycle_life)
                if linear_deg_cost_p_kwh is not None:
                    self.linear_deg_cost_per_kwh = linear_deg_cost_p_kwh
                else:
                    # Example: assume 3650 cycles (10 years daily), adjust as needed
                    cycle_life = 3650
                    self.linear_deg_cost_per_kwh = env.battery_life_cost / (env.battery_capacity * cycle_life)

            # Store env parameters needed for SDP calculations
            self.battery_capacity = env.battery_capacity
            self.max_battery_flow = env.max_battery_flow
            self.step_duration = env.step_duration
            self.max_grid_energy = env.max_grid_energy
            self.battery_life_cost = env.battery_life_cost

            # Discretize state (SoC in kWh) and action (normalized flow) spaces
            self.soc_levels_kwh = np.linspace(0, self.battery_capacity, self.soc_resolution)
            self.action_levels_norm = np.linspace(-1.0, 1.0, self.action_resolution)

            # Cache for the policy table (optional, might recompute every step in receding horizon)
            # self.sdp_policy_cache = None
            # self.cache_step = -1

            # debugging log when using SDP, it tracks the steps solving SDP
            self.sdp_debug_log = []

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
            device = next(self.model.parameters()).device
            state = torch.tensor(obs, dtype=torch.float32, device=device).reshape(1, 1, -1)
            rtg = torch.tensor([[0.0]], dtype=torch.float32, device=device).reshape(1, 1, 1)
            timestep = torch.tensor([[0]], dtype=torch.long, device=device)
            actions = torch.zeros((1, 1, self.model.act_dim), dtype=torch.float32, device=device)
            _, _, act_preds = self.model(state, rtg, timestep, actions)
            action = act_preds[0, 0].detach().cpu().numpy().tolist()
            return action
        elif self.algorithm == 'sdp':
            current_soc_kwh = obs[-2] # Assuming BatteryLevel is the second to last element
            current_step_env = self.env.current_step # Get current step from env

            # --- Receding Horizon SDP ---
            # 1. Get Forecasts
            forecasts = self._get_forecasts(current_step_env, self.horizon)
            if not forecasts: # Handle case where horizon goes beyond data
                print("Warning: Not enough forecast data for full horizon. Using rule-based action.")
                return self.rule_based_action(obs) # Fallback action

            # 2. Solve SDP for the horizon
            policy_table = self._solve_sdp(forecasts)

            # 3. Determine Current State Index
            soc_idx = self._soc_to_idx(current_soc_kwh)

            # 4. Get Optimal Action for the *first* step
            optimal_action_idx = policy_table[0, soc_idx]
            if optimal_action_idx == -1: # Handle cases where no valid action was found (e.g., all lead to penalties)
                print(f"Warning: No optimal action found for SoC {current_soc_kwh:.2f} at step {current_step_env}. Using zero action.")
                action_value = 0.0008964
            else:
                action_value = self.action_levels_norm[optimal_action_idx]
            # add small noise to the action
            noise = np.random.normal(-0.01, 0.01)
            action_value = min(max(action_value + noise, -1.0), 1.0)
            return [np.float32(action_value)]
        else:
            raise NotImplementedError(f"Algorithm '{self.algorithm}' is not supported.")
        
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

    def _solve_sdp(self, forecasts):
        """
        The following algorithm is derived from the paper "Optimal Operation of Energy Storage Systems Considering Forecasts and Battery Degradation (2018)"
        by K Abdulla, J De Hoog, V Muenzel, F Suits, K Steer, A Wirth, S Halgamuge
        Implements the backward induction algorithm with vectorized feasibility checks and action effects.
        """
        num_soc_levels = len(self.soc_levels_kwh)
        horizon = len(forecasts)
    
        # Initialize cost-to-go (J) and policy tables
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)  # Store action indices
    
        # Terminal cost is zero
        cost_to_go[horizon, :] = 0.0
    
        # Precompute battery flow energies for all actions
        battery_flow_energies = self.action_levels_norm * self.max_battery_flow * self.step_duration
    
        # Backward induction: iterate over each time step in reverse (from horizon-1 to 0)
        for t in range(horizon - 1, -1, -1):
            forecast_step = forecasts[t]
            # For each discretized state-of-charge (SoC) level
            for soc_idx, soc_kwh in enumerate(self.soc_levels_kwh):
            
                # Feasibility Check: Compute all possible next SoC values for all actions
                potential_next_socs = soc_kwh + battery_flow_energies
                feasible_mask = (potential_next_socs >= -1e-6) & (potential_next_socs <= self.battery_capacity + 1e-6)
                feasible_action_indices = np.where(feasible_mask)[0]
            
                if feasible_action_indices.size == 0:
                    continue  # No feasible actions from this SoC
            
                # Clip battery flow energies to ensure SoC stays within bounds
                actual_battery_flow_energies = np.clip(
                    battery_flow_energies[feasible_action_indices],
                    -soc_kwh,
                    self.battery_capacity - soc_kwh
                )
                
                # Compute next SoC values (continuous) and interpolate future costs
                next_soc_kwhs = soc_kwh + actual_battery_flow_energies
                
                # Use linear interpolation for cost-to-go lookup instead of nearest-index mapping
                # This reduces discretization error by avoiding snapping to grid points
                future_costs = np.array([
                    interpolate_ctg(self.soc_levels_kwh, cost_to_go[t + 1, :], next_soc_kwh)
                    for next_soc_kwh in next_soc_kwhs
                ])
                
                # Assert shape alignment for safety
                assert len(future_costs) == len(feasible_action_indices), \
                    f"Shape mismatch: future_costs {len(future_costs)} vs feasible_actions {len(feasible_action_indices)}"
            
                # Compute battery flow rates for feasible actions
                battery_flow_rates = self.action_levels_norm[feasible_action_indices] * self.max_battery_flow
            
                # Calculate stage costs for all feasible actions at this step
                stage_costs = np.array([
                    self._calculate_sdp_stage_cost(
                        t, soc_kwh,
                        battery_flow_rates[i],
                        actual_battery_flow_energies[i],
                        forecast_step
                    )
                    for i in range(len(feasible_action_indices))
                ])
                
                total_costs = stage_costs + future_costs
            
                # Select the action with the minimum total cost (stage + future)
                valid_mask = (stage_costs != np.inf) & (future_costs != np.inf)
                if np.any(valid_mask):
                    min_idx = np.argmin(total_costs[valid_mask])
                    min_total_cost = total_costs[valid_mask][min_idx]
                    best_action_idx = feasible_action_indices[valid_mask][min_idx]
                    cost_to_go[t, soc_idx] = min_total_cost
                    policy_table[t, soc_idx] = best_action_idx
            
                    # Debug logging for the current actual SoC at t=0 (optional)
                    current_actual_soc_idx = self._soc_to_idx(self.env.battery_level)
                    if t == 0 and soc_idx == current_actual_soc_idx:
                        self.sdp_debug_log.append({
                            't': t,
                            'soc_idx': soc_idx,
                            'action_idx': best_action_idx,
                            'action_norm': self.action_levels_norm[best_action_idx],
                            'stage_cost': stage_costs[valid_mask][min_idx],
                            'future_cost': future_costs[valid_mask][min_idx],
                            'total_cost': min_total_cost
                        })
        
        # Return the computed policy table (action indices for each time/SoC)
        return policy_table


    def _calculate_sdp_stage_cost(self, t, soc_kwh, battery_flow_rate, battery_flow_energy, forecast_step):
        """Calculates the cost for a single step in the SDP."""
        solar = forecast_step['SolarGen']
        load = forecast_step['HouseLoad']
        import_price = forecast_step['ImportEnergyPrice']
        export_price = forecast_step['ExportEnergyPrice']

        # --- Grid Cost ---
        battery_charge_energy = max(0, battery_flow_energy)
        battery_discharge_energy = max(0, -battery_flow_energy)
        grid_energy = load + battery_charge_energy - solar - battery_discharge_energy

        # Use helper function for clear import/export semantics and grid limit checking
        grid_cost = compute_grid_cost(grid_energy, import_price, export_price, self.max_grid_energy)
        
        if grid_cost == np.inf:
            return np.inf  # Return early if grid violated

        # --- Degradation Cost ---
        if self.degradation_model == 'linear':
            # Linear throughput-based degradation (from the paper)
            degradation_cost = self.linear_deg_cost_per_kwh * abs(battery_flow_energy)
        else:
            # Use the *intended* flow rate for degradation calculation
            Id_crate = abs(max(0, -battery_flow_rate) / self.battery_capacity)
            Ich_crate = abs(max(0, battery_flow_rate) / self.battery_capacity)

            # DoD based on the *actual* energy moved
            DoD_percent = abs(battery_flow_energy / self.battery_capacity) * 100.0

            # Average SoC for the step
            # If discharging (flow_energy < 0), avg is current - half_discharged
            # If charging (flow_energy > 0), avg is current + half_charged
            SoC_avg_percent = (soc_kwh + 0.5 * battery_flow_energy) / self.battery_capacity * 100.0
            SoC_avg_percent = np.clip(SoC_avg_percent, 0, 100) # Ensure valid range

            # Handle zero DoD case (no degradation)
            if DoD_percent < 1e-6:
                degradation_fraction = 0.0
            else:
                # Calculate degradation fraction using the static degradation model and apply correction factor
                degradation_fraction = static_degradation(Id_crate, Ich_crate, SoC_avg_percent, DoD_percent)*self.static_deg_correction_factor

            degradation_cost = degradation_fraction * self.battery_life_cost

        return grid_cost + degradation_cost

    def rule_based_action(self, obs):
        diff = obs[1] - obs[2] # difference between solar generation (obs[1]) and house load (obs[2])
        max_flow = self.env.max_battery_flow
        battery_level = obs[-2]/self.env.battery_capacity  # normalize battery level to [0, 1] by dividing battery energy (obs[-2]) by capacity
        noise = np.random.normal(-0.01, 0.01)  # add small noise with standard deviation of 0.01
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
    
        return [np.float32(result)]  # Return as a list to match expected action format
    
    def run_episode(self, render=False):
        obs, info = self.env.reset()
        raw_obs = self.env.get_raw_obs()  # Get raw observation if available
        logs = []
        terminated, truncated = False, False
        step = 0  # Step counter
        # Decide which obs to use based on agent type
        if self.algorithm in ['rule', 'sdp']:
            # Use raw_obs if available, else fallback to obs
            current_obs = raw_obs
        else:  # 'rl', 'dt', etc.
            # Use norm_obs if available, else fallback to obs
            current_obs = obs

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

            # Select the correct next_obs for the next step
            raw_obs = self.env.get_raw_obs()  # Get raw observation if available
            obs = next_obs
            if self.algorithm in ['rule', 'sdp']:
                current_obs = raw_obs
            else:
                current_obs = obs
            if render:
                self.env.render()
            step += 1  # Increment step counter

        return pl.DataFrame(logs)

import concurrent.futures
from tqdm.notebook import tqdm

def run_single(agent_class, env, agent_kwargs, render):
    agent = agent_class(env, **agent_kwargs)
    return agent.run_episode(render=render)

# this can be used to run multiple episodes in parallel
def run_episodes_parallel(agent_class, envs, agent_kwargs=None, render=False, max_workers=4, use_notebook_tqdm=True):
    """
    Runs one episode per environment in parallel.
    agent_class: The Agent class to instantiate. 
    envs: List of SolarBatteryEnv instances.
    agent_kwargs: Dict of kwargs for Agent constructor. (only suitable for rule, sdp algorithms)
    use_notebook_tqdm: If True, use tqdm.notebook.tqdm; else use tqdm.tqdm (for scripts)
    Returns: List of DataFrames (one per environment).
    """
    agent_kwargs = agent_kwargs or {}
    # check if algorith is suitable for parallel execution
    if agent_kwargs.get('algorithm', 'rule').lower() not in ['rule', 'sdp']:
        raise ValueError("Parallel execution is only supported for 'rule' and 'sdp' algorithms. ")

    # Select tqdm version
    if use_notebook_tqdm:
        from tqdm.notebook import tqdm as tqdm_bar
    else:
        from tqdm import tqdm as tqdm_bar

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use as_completed to update progress as each finishes
        futures = [executor.submit(run_single, agent_class, env, agent_kwargs, render) for env in envs]
        for f in tqdm_bar(concurrent.futures.as_completed(futures), total=len(futures), desc="Episodes"):
            results.append(f.result())
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