import logging
import numpy as np
import torch
import polars as pl
import warnings
from typing import Optional
from EnergySimEnv import SolarBatteryEnv, VIOLATION_PENALTY

from batterydeg import DegradationModel, RainflowCounter
from quantile_scenarios import QuantileScenarioGenerator
from sdp_multires import DynamicProgram, solve_mrdp

import concurrent.futures
from tqdm.notebook import tqdm

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

        if self.algorithm == 'sdp' or self.algorithm == 'mrdp':  # Stochastic Dynamic Programming

            self.degradation_model = degradation_model
            if self.degradation_model == 'linear':
                # If not provided, default to battery_life_cost / (battery_capacity * cycle_life)
                if linear_deg_cost_p_kwh is not None:
                    self.linear_deg_cost_per_kwh = linear_deg_cost_p_kwh
                else:
                    # Example: assume 3650 cycles (10 years daily), adjust as needed
                    cycle_life = 3650
                    self.linear_deg_cost_per_kwh = env.battery_life_cost / (env.battery_capacity * cycle_life)
            elif self.degradation_model == 'rainflow':
                # initialize degradation model
                self.degradation_temperature = getattr(self.env, 'degradation_temperature', 25.0)
                self.cycle_degradation_model = DegradationModel()
            else:
                raise ValueError(f"Unsupported degradation_model: {self.degradation_model}")

            # Store env parameters needed for SDP calculations
            self.battery_capacity = env.battery_capacity
            self.max_battery_flow = env.max_battery_flow
            self.step_duration = env.step_duration
            self.max_grid_energy = env.max_grid_energy
            self.battery_life_cost = env.battery_life_cost

            # Discretize state (SoC in kWh) and action (normalized flow) spaces
            self.soc_levels_kwh = np.linspace(0, self.battery_capacity, self.soc_resolution)
            self.action_levels_norm = np.linspace(-1.0, 1.0, self.action_resolution)

            # Quantile scenario generator (used to compute expected grid cost)
            # Keep a small default scenario count to limit compute; caller can replace if needed.
            self.scenario_generator = QuantileScenarioGenerator(n_scenarios=5)
            # simple cache for per-row scenarios to avoid recomputing inside tight loops
            self._scenario_cache = None
            # Monte Carlo configuration for expected cost approximation
            self.use_monte_carlo = use_monte_carlo
            self.mc_samples = int(mc_samples) if mc_samples is not None else 100
            self.mc_seed = mc_seed

            # Cache for the policy table (optional, might recompute every step in receding horizon)
            # self.sdp_policy_cache = None
            # self.cache_step = -1

            # debugging log when using SDP, it tracks the steps solving SDP
            self.sdp_debug_log = []
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

    def _degradation_per_cycle(self, Id, Ich, soc_percent, DoD):
        return self.cycle_degradation_model.degradation_per_cycle(
            T=self.degradation_temperature,
            Id=Id,
            Ich=Ich,
            SOCav=soc_percent,
            DOD=DoD,
        )

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

        
        elif self.algorithm == 'sdp' or self.algorithm == 'mrdp':
            current_soc_kwh = obs[-2] # Assuming BatteryLevel is the second to last element
            current_step_env = self.env.current_step # Get current step from env

            # --- Receding Horizon SDP ---
            # 1. Get Forecasts
            if self.algorithm == 'mrdp' and self.subhorizon_specs is not None:
                total_horizon = sum(int(spec.get('length', 0)) for spec in self.subhorizon_specs)
                horizon_for_forecasts = max(1, total_horizon)
            else:
                horizon_for_forecasts = self.horizon
            forecasts = self._get_forecasts(current_step_env, horizon_for_forecasts)
            if not forecasts: # Handle case where horizon goes beyond data
                print("Warning: Not enough forecast data for full horizon. Using rule-based action.")
                return self.rule_based_action(obs) # Fallback action
            if self.algorithm == 'sdp':
                policy_table = self._solve_sdp(forecasts, start_index=current_step_env)
            elif self.algorithm == 'mrdp':
                policy_table = self._solve_mrdp(forecasts, start_index=current_step_env)

            # 3. Determine Current State Index
            if self.algorithm == 'mrdp' and self.subhorizon_specs is not None and len(self.subhorizon_specs) > 0:
                first_spec = self.subhorizon_specs[0]
                soc_res_first = int(first_spec.get('soc_resolution', self.soc_resolution))
                soc_levels_first = np.linspace(0, self.battery_capacity, soc_res_first)
                soc_idx = int(np.argmin(np.abs(soc_levels_first - current_soc_kwh)))
            else:
                soc_idx = self._soc_to_idx(current_soc_kwh)

            # 4. Get Optimal Action for the *first* step
            optimal_action_idx = policy_table[0, soc_idx]
            if optimal_action_idx == -1:
                print(f"Warning: No optimal action found for SoC {current_soc_kwh:.2f} at step {current_step_env}. Using zero action.")
                action_value = 0.0008964
            else:
                if self.algorithm == 'mrdp' and self.subhorizon_specs is not None and len(self.subhorizon_specs) > 0:
                    action_res_first = int(first_spec.get('action_resolution', self.action_resolution))
                    action_levels_first = np.linspace(-1.0, 1.0, action_res_first)
                    action_value = action_levels_first[int(optimal_action_idx)]
                else:
                    action_value = self.action_levels_norm[int(optimal_action_idx)]
            noise = np.random.normal(-0.001, 0.001)
            action_value = min(max(action_value + noise, -1.0), 1.0)
            return [np.float32(action_value)]
        elif self.algorithm == 'oracle':
            return self._choose_oracle_action()
        
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

    def _choose_oracle_action(self):
        """Determines the optimal first action by solving a DP using actual future values."""
        if not hasattr(self, 'oracle_action_levels'):
            raise ValueError("Oracle algorithm requested but oracle_action_levels not initialized.")

        remaining_steps = max(0, len(self.env.df) - self.env.current_step)
        remaining_env_budget = max(0, self.env.max_step - self.env.current_step)
        horizon = min(self.oracle_horizon, remaining_steps, remaining_env_budget)

        if horizon <= 0:
            raw_obs = self.env.get_raw_obs()
            return self.rule_based_action(raw_obs)

        policy_table = self._solve_oracle_dp(self.env.current_step, horizon)
        if policy_table is None:
            raw_obs = self.env.get_raw_obs()
            return self.rule_based_action(raw_obs)

        current_soc_idx = self._soc_to_idx(self.env.battery_level)
        action_idx = policy_table[0, current_soc_idx]
        if action_idx == -1:
            raw_obs = self.env.get_raw_obs()
            return self.rule_based_action(raw_obs)

        action_value = self.oracle_action_levels[int(action_idx)]
        noise = np.random.normal(-0.001, 0.001)
        action_value = min(max(action_value + noise, -1.0), 1.0)
        return [np.float32(action_value)]

    def _solve_oracle_dp(self, start_index: int, horizon: int):
        """DP-based oracle over the actual future rows (static degradation only)."""
        num_soc_levels = len(self.soc_levels_kwh)
        num_actions = len(self.oracle_action_levels)
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        cost_to_go[horizon, :] = 0.0

        socs = self.soc_levels_kwh[:, np.newaxis]  # (S,1)
        battery_energies = self.oracle_action_levels * self.max_battery_flow * self.step_duration  # (A,)

        for t in range(horizon - 1, -1, -1):
            row = self.env._get_row(start_index + t)
            solar = row['SolarGen']
            load = row['HouseLoad']
            imp_price = row['ImportEnergyPrice']
            exp_price = row['ExportEnergyPrice']

            battery_reshaped = battery_energies[np.newaxis, :]  # (1, A)
            clipped_energies = np.clip(
                battery_reshaped,
                -socs,
                self.battery_capacity - socs
            )

            battery_rates = clipped_energies / self.step_duration
            battery_charge = np.maximum(clipped_energies, 0.0)
            battery_discharge = np.maximum(-clipped_energies, 0.0)

            demand = load + battery_charge
            supply = solar + battery_discharge
            grid_energy = demand - supply
            grid_violation = np.abs(grid_energy) > (self.max_grid_energy + 1e-6)
            energy_price = np.where(grid_energy >= 0, imp_price, exp_price)
            grid_cost = np.where(grid_energy >= 0, grid_energy * imp_price, -np.abs(grid_energy) * exp_price)
            grid_cost[grid_violation] = np.inf

            # Compute per-(soc,action) degradation cost with linearized wear per kWh
            num_soc = num_soc_levels
            stage_costs = np.full((num_soc, num_actions), np.inf)
            for si in range(num_soc):
                soc_val = float(socs[si, 0])
                for ai in range(num_actions):
                    # Skip infeasible/masked entries
                    if grid_violation[si, ai]:
                        continue
                    energy = float(clipped_energies[si, ai])
                    battery_rate = energy / self.step_duration

                    if self.degradation_model == 'rainflow':
                        soc_next = soc_val + energy
                        deg_frac = self._compute_deg_fraction_rainflow(soc_val, soc_next)
                    else:
                        Id = abs(min(0.0, battery_rate)) / self.battery_capacity
                        Ich = abs(max(0.0, battery_rate)) / self.battery_capacity
                        DoD_percent = abs(energy) / self.battery_capacity * 100.0

                        avg_soc = (soc_val + 0.5 * energy) / self.battery_capacity * 100.0
                        avg_soc = float(np.clip(avg_soc, 0.0, 100.0))

                        energy_abs = abs(energy)
                        deg_frac = self._compute_deg_fraction_linearized(Id, Ich, avg_soc, energy_abs)

                    degradation_cost = deg_frac * self.battery_life_cost
                    stage_costs[si, ai] = float(grid_cost[si, ai]) + degradation_cost

            next_socs = socs + clipped_energies
            next_costs = np.interp(next_socs.ravel(), self.soc_levels_kwh, cost_to_go[t + 1, :]).reshape(next_socs.shape)

            total_costs = stage_costs + next_costs
            feasible_mask = np.isfinite(total_costs)

            row_min = np.min(np.where(feasible_mask, total_costs, np.inf), axis=1)
            best_actions = np.argmin(np.where(feasible_mask, total_costs, np.inf), axis=1)

            finite_mask = np.isfinite(row_min)
            cost_to_go[t, :] = np.where(finite_mask, row_min, np.inf)
            policy_table[t, :] = -1
            policy_table[t, finite_mask] = best_actions[finite_mask]

        if not np.any(np.isfinite(cost_to_go[0, :])):
            return None
        return policy_table

    def _solve_sdp(self, forecasts, start_index: int = 0):
        """
        Vectorized SDP backward induction: compute feasible next SoC and stage costs in
        array form across all SoC levels and actions. Evaluate expensive stage costs
        only once per unique clipped battery energy per timestep.
        """

        num_soc_levels = len(self.soc_levels_kwh)
        horizon = len(forecasts)

        # Initialize cost-to-go (J) and policy tables
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        cost_to_go[horizon, :] = 0.0

        # Precompute battery flow energies for all actions (kWh)
        battery_flow_energies = self.action_levels_norm * self.max_battery_flow * self.step_duration
        num_actions = len(self.action_levels_norm)
        socs = self.soc_levels_kwh  # shape (num_soc,)

        # Ensure scenario cache prepared once if possible
        if self._scenario_cache is None:
            try:
                self._scenario_cache = self.scenario_generator.generate_time_step_scenarios(self.env.df)
            except Exception:
                self._scenario_cache = None

        # Backward induction (vectorized per timestep)
        for t in range(horizon - 1, -1, -1):
            forecast_step = forecasts[t]
            row_idx = start_index + t

            # Prepare Monte Carlo pre-sampled arrays for this t (if possible)
            monte_samples = None
            if getattr(self, 'use_monte_carlo', False) and self._scenario_cache is not None:
                try:
                    vals_solar_arr, ps_solar_arr = self._scenario_cache['solar']
                    vals_load_arr, ps_load_arr = self._scenario_cache['load']
                    vals_imp_arr, ps_imp_arr = self._scenario_cache['import_price']
                    vals_exp_arr, ps_exp_arr = self._scenario_cache['export_price']

                    if (row_idx >= 0 and row_idx < vals_solar_arr.shape[0]):
                        # Extract per-timestep scenario marginals
                        vals_solar_t = vals_solar_arr[row_idx, :]
                        ps_solar_t = ps_solar_arr[row_idx, :]
                        vals_load_t = vals_load_arr[row_idx, :]
                        ps_load_t = ps_load_arr[row_idx, :]
                        vals_imp_t = vals_imp_arr[row_idx, :]
                        ps_imp_t = ps_imp_arr[row_idx, :]
                        vals_exp_t = vals_exp_arr[row_idx, :]
                        ps_exp_t = ps_exp_arr[row_idx, :]

                        # Sample joint marginal indices once per timestep
                        mc_seed = getattr(self, 'mc_seed', None)
                        mc_samples = getattr(self, 'mc_samples', 100)
                        rng_seed_t = None if mc_seed is None else (mc_seed + t)
                        rng = np.random.default_rng(rng_seed_t)

                        idx_s = rng.choice(len(vals_solar_t), size=mc_samples, p=ps_solar_t)
                        idx_l = rng.choice(len(vals_load_t), size=mc_samples, p=ps_load_t)
                        idx_i = rng.choice(len(vals_imp_t), size=mc_samples, p=ps_imp_t)
                        idx_e = rng.choice(len(vals_exp_t), size=mc_samples, p=ps_exp_t)

                        sampled_solar = vals_solar_t[idx_s]
                        sampled_load = vals_load_t[idx_l]
                        sampled_imp = vals_imp_t[idx_i]
                        sampled_exp = vals_exp_t[idx_e]

                        monte_samples = (sampled_solar, sampled_load, sampled_imp, sampled_exp)
                except Exception:
                    monte_samples = None

            # Vectorized feasibility: shape (num_soc, num_actions)
            socs_reshaped = socs[:, np.newaxis]  # (num_soc, 1)
            battery_energies_reshaped = battery_flow_energies[np.newaxis, :]  # (1, num_actions)
            potential_next_socs = socs_reshaped + battery_energies_reshaped  # (num_soc, num_actions)

            # Feasibility mask: within [0, battery_capacity] with tolerance
            feasible_mask = ((potential_next_socs >= -1e-6) & 
                           (potential_next_socs <= self.battery_capacity + 1e-6))

            # Clipped battery flow energies for each (soc, action) pair
            clipped_battery_energies = np.clip(
                battery_energies_reshaped,
                -socs_reshaped,  # minimum: discharge down to 0
                self.battery_capacity - socs_reshaped  # maximum: charge up to capacity
            )

            # Compute next SoCs and future costs via vectorized interpolation
            next_socs = socs_reshaped + clipped_battery_energies  # (num_soc, num_actions)
            # Vectorized interpolation over flattened array is much faster than per-element calls
            future_costs = np.interp(next_socs.ravel(), self.soc_levels_kwh, cost_to_go[t + 1, :]).reshape(next_socs.shape)

            # Build unique energy list from feasible clipped energies using numpy.unique with return_inverse
            rounded = np.round(clipped_battery_energies, 10)
            rounded_flat = rounded.ravel()
            feasible_flat = feasible_mask.ravel()

            stage_costs = np.full(rounded.shape, np.inf)

            if feasible_flat.any():
                values_flat = rounded_flat[feasible_flat]
                unique_vals, inverse = np.unique(values_flat, return_inverse=True)

                # Compute cost for each unique energy value (loop over unique values only)
                unique_costs = np.empty(unique_vals.shape, dtype=float)
                for ui, energy_key in enumerate(unique_vals):
                    energy = float(energy_key)
                    battery_rate = energy / self.step_duration

                    if monte_samples is not None:
                        sampled_solar, sampled_load, sampled_imp, sampled_exp = monte_samples
                        battery_charge_energy = max(0.0, energy)
                        battery_discharge_energy = max(0.0, -energy)

                        grid_energy = sampled_load + battery_charge_energy - sampled_solar - battery_discharge_energy
                        if np.any(np.abs(grid_energy) > (self.max_grid_energy + 1e-6)):
                            stage_cost = np.inf
                        else:
                            is_import = grid_energy > 0
                            costs = np.where(is_import, grid_energy * sampled_imp, -np.abs(grid_energy) * sampled_exp)
                            if np.any(np.isinf(costs)):
                                stage_cost = np.inf
                            else:
                                stage_cost = float(np.mean(costs))

                        if self.degradation_model == 'linear':
                            degradation_cost = self.linear_deg_cost_per_kwh * abs(energy)
                        else:
                            rep_soc = self.battery_capacity / 2.0
                            Id_crate = abs(max(0, -battery_rate) / self.battery_capacity)
                            Ich_crate = abs(max(0, battery_rate) / self.battery_capacity)
                            SoC_avg_percent = (rep_soc + 0.5 * energy) / self.battery_capacity * 100.0
                            SoC_avg_percent = np.clip(SoC_avg_percent, 0, 100)
                            deg_frac = self._compute_deg_fraction_linearized(Id_crate, Ich_crate, SoC_avg_percent, abs(energy))
                            deg_frac *= self.static_deg_correction_factor
                            degradation_cost = deg_frac * self.battery_life_cost

                        unique_costs[ui] = stage_cost + degradation_cost
                    else:
                        rep_soc = self.battery_capacity / 2.0
                        unique_costs[ui] = self._calculate_sdp_stage_cost(row_idx, rep_soc, battery_rate, energy, forecast_step)

                # Scatter unique_costs back into stage_costs using inverse indices
                costs_flat = np.full(rounded_flat.shape, np.inf)
                costs_flat[feasible_flat] = unique_costs[inverse]
                stage_costs = costs_flat.reshape(rounded.shape)

            # Vectorized total cost and optimal action selection
            total_costs = stage_costs + future_costs
            # Mask invalid entries
            total_costs_masked = np.where(feasible_mask & np.isfinite(stage_costs) & np.isfinite(future_costs), total_costs, np.inf)

            # Row-wise minima and argmin to choose best action for each SoC
            row_min = np.min(total_costs_masked, axis=1)
            best_actions = np.argmin(total_costs_masked, axis=1)

            # Update cost_to_go and policy_table; set policy to -1 where row_min is inf
            finite_mask = np.isfinite(row_min)
            cost_to_go[t, :] = row_min
            policy_table[t, :] = -1
            policy_table[t, finite_mask] = best_actions[finite_mask]

            # Optional debug logging for the current actual SoC at t=0
            if t == 0:
                current_actual_soc_idx = self._soc_to_idx(self.env.battery_level)
                if finite_mask[current_actual_soc_idx]:
                    aidx = policy_table[0, current_actual_soc_idx]
                    self.sdp_debug_log.append({
                        't': 0,
                        'soc_idx': current_actual_soc_idx,
                        'action_idx': int(aidx),
                        'action_norm': float(self.action_levels_norm[aidx]) if aidx >= 0 else None,
                        'stage_cost': float(stage_costs[current_actual_soc_idx, aidx]) if aidx >= 0 else np.inf,
                        'future_cost': float(future_costs[current_actual_soc_idx, aidx]) if aidx >= 0 else np.inf,
                        'total_cost': float(total_costs_masked[current_actual_soc_idx, aidx]) if aidx >= 0 else np.inf
                    })

        return policy_table


    def _calculate_sdp_stage_cost(self, row_idx, soc_kwh, battery_flow_rate, battery_flow_energy, forecast_step):
        """
        Vectorized stage cost calculation: compute expected grid cost using scenario arrays
        indexed by absolute row index, with degradation cost added deterministically.
        
        Note: Assumes self._scenario_cache is already initialized by _solve_sdp() before calling this method.
        """

        # Default fallback: use deterministic forecast values
        deterministic_solar = forecast_step.get('SolarGen')
        deterministic_load = forecast_step.get('HouseLoad')
        deterministic_imp = forecast_step.get('ImportEnergyPrice')
        deterministic_exp = forecast_step.get('ExportEnergyPrice')

        expected_grid_cost = None
        
        # Try to use scenario arrays indexed by absolute row index
        if self._scenario_cache is not None:
            try:
                vals_solar_arr, ps_solar_arr = self._scenario_cache['solar']
                vals_load_arr, ps_load_arr = self._scenario_cache['load']
                vals_imp_arr, ps_imp_arr = self._scenario_cache['import_price']
                vals_exp_arr, ps_exp_arr = self._scenario_cache['export_price']

                if (row_idx >= 0 and row_idx < vals_solar_arr.shape[0]):
                    # Extract scenario marginals for this row
                    vals_solar = vals_solar_arr[row_idx, :]
                    ps_solar = ps_solar_arr[row_idx, :]
                    vals_load = vals_load_arr[row_idx, :]
                    ps_load = ps_load_arr[row_idx, :]
                    vals_imp = vals_imp_arr[row_idx, :]
                    ps_imp = ps_imp_arr[row_idx, :]
                    vals_exp = vals_exp_arr[row_idx, :]
                    ps_exp = ps_exp_arr[row_idx, :]

                    # Vectorized stage cost function across scenarios
                    def _stage_cost_fn(solar_v, load_v, imp_v, exp_v):
                        battery_charge_energy = max(0, battery_flow_energy)
                        battery_discharge_energy = max(0, -battery_flow_energy)
                        grid_energy = load_v + battery_charge_energy - solar_v - battery_discharge_energy
                        return compute_grid_cost(grid_energy, imp_v, exp_v, self.max_grid_energy)

                    # Decide between Monte Carlo and exhaustive evaluation
                    n_comb = len(vals_solar) * len(vals_load) * len(vals_imp) * len(vals_exp)
                    use_mc = getattr(self, 'use_monte_carlo', False) or (n_comb > 200)

                    if use_mc:
                        expected_grid_cost = self.scenario_generator.expected_cost_monte_carlo(
                            vals_solar, ps_solar,
                            vals_load, ps_load,
                            vals_imp, ps_imp,
                            vals_exp, ps_exp,
                            _stage_cost_fn,
                            n_samples=getattr(self, 'mc_samples', 100),
                            rng_seed=getattr(self, 'mc_seed', None),
                        )
                    else:
                        expected_grid_cost = self.scenario_generator.expected_cost_cartesian(
                            vals_solar, ps_solar,
                            vals_load, ps_load,
                            vals_imp, ps_imp,
                            vals_exp, ps_exp,
                            _stage_cost_fn
                        )
            except Exception:
                expected_grid_cost = None

        # Fallback to deterministic evaluation if scenarios unavailable
        if expected_grid_cost is None:
            battery_charge_energy = max(0, battery_flow_energy)
            battery_discharge_energy = max(0, -battery_flow_energy)
            deterministic_grid_energy = (deterministic_load + battery_charge_energy - 
                                       deterministic_solar - battery_discharge_energy)
            deterministic_grid_cost = compute_grid_cost(
                deterministic_grid_energy, deterministic_imp, deterministic_exp, self.max_grid_energy
            )
            if deterministic_grid_cost == np.inf:
                return np.inf
            expected_grid_cost = deterministic_grid_cost

        # Add degradation cost deterministically
        if self.degradation_model == 'linear':
            degradation_cost = self.linear_deg_cost_per_kwh * abs(battery_flow_energy)
        elif self.degradation_model == 'rainflow':
            soc_next_kwh = soc_kwh + battery_flow_energy
            deg_frac = self._compute_deg_fraction_rainflow(soc_kwh, soc_next_kwh)
            degradation_cost = deg_frac * self.battery_life_cost
        else:
            Id_crate = abs(max(0, -battery_flow_rate) / self.battery_capacity)
            Ich_crate = abs(max(0, battery_flow_rate) / self.battery_capacity)
            SoC_avg_percent = (soc_kwh + 0.5 * battery_flow_energy) / self.battery_capacity * 100.0
            SoC_avg_percent = np.clip(SoC_avg_percent, 0, 100)
            deg_frac = self._compute_deg_fraction_linearized(Id_crate, Ich_crate, SoC_avg_percent, abs(battery_flow_energy))
            deg_frac *= self.static_deg_correction_factor
            degradation_cost = deg_frac * self.battery_life_cost

        return expected_grid_cost + degradation_cost

    def _compute_deg_fraction_linearized(self, Id, Ich, soc_percent, energy_kwh):
        """
        Convert a representative full-cycle wear into a per-kWh wear using a base DoD,
        then scale by the energy moved in this step.
        """
        base_DoD = getattr(self.env, "base_deg_DoD", 80.0)
        if energy_kwh <= 0:
            return 0.0
        energy_full_base_cycle = self.battery_capacity * (base_DoD / 100.0) * 2.0
        if energy_full_base_cycle <= 0:
            return 0.0
        cycle_wear = self._degradation_per_cycle(Id, Ich, soc_percent, base_DoD)
        wear_per_kwh = cycle_wear / energy_full_base_cycle
        frac = wear_per_kwh * energy_kwh
        return self.env._sanitize_deg_frac(frac)

    def _compute_deg_fraction_rainflow(self, soc_start_kwh: float, soc_end_kwh: float) -> float:
        """Estimate degradation for a single step using the rainflow counter."""
        if self.battery_capacity <= 0:
            return 0.0

        start_pct = np.clip((soc_start_kwh / self.battery_capacity) * 100.0, 0.0, 100.0)
        end_pct = np.clip((soc_end_kwh / self.battery_capacity) * 100.0, 0.0, 100.0)

        counter = RainflowCounter(step_duration=self.step_duration)
        cycles = []
        for val in (start_pct, end_pct, start_pct):
            cycles.extend(counter.update(val))

        deg_frac = 0.0
        for SoC_avg, DoD, Id_cycle, Ich_cycle in cycles:
            deg_frac += self._degradation_per_cycle(Id_cycle, Ich_cycle, SoC_avg, DoD)

        return self.env._sanitize_deg_frac(deg_frac)
    
    def _solve_mrdp(self, forecasts, start_index: int = 0):
        """
        MRDP-enabled SDP backward induction: Uses sdp_multires.solve_mrdp for multi-resolution DP.
        """
        # 1. Use subhorizon_specs from Agent argument, or default if not provided
        # Default step_duration values use the same time unit as env.step_duration
        subhorizon_specs = self.subhorizon_specs if self.subhorizon_specs is not None else [
            {'start': 0, 'length': 12, 'soc_resolution': 20, 'action_resolution': 41, 'step_duration': self.step_duration},
            {'start': 12, 'length': 36, 'soc_resolution': 8,  'action_resolution': 17, 'step_duration': max(self.step_duration, 2 * self.step_duration)},
        ]

        def stage_cost_function(t_global_idx, unique_energy_values):
            costs = np.empty(len(unique_energy_values), dtype=float)
            for i, energy in enumerate(unique_energy_values):
                soc_kwh = self.battery_capacity / 2.0
                battery_rate = energy / self.step_duration
                forecast_idx = t_global_idx
                forecast_step = forecasts[forecast_idx] if forecast_idx < len(forecasts) else forecasts[-1]
                costs[i] = self._calculate_sdp_stage_cost(
                    forecast_idx, soc_kwh, battery_rate, energy, forecast_step
                )
            return costs

        policy_table, cost_to_go = solve_mrdp(
            env=self.env,
            forecasts=forecasts,
            subhorizon_specs=subhorizon_specs,
            global_start_index=start_index,
            stage_cost_function=stage_cost_function
        )
        return policy_table

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