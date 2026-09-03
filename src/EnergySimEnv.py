# import the necessary packages
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces, error, utils

import numpy as np
import polars as pl

from batterydeg import DegradationModel, RainflowCounter, CycleOnlyDegradationModel, DisabledDegradationModel

# global variables
VIOLATION_PENALTY = -8964
MAX_RAW_BATTERY_DEG_COST_IN_OBS_FACTOR = 0.01  # 1% of battery_life_cost per step
MAX_PCT_BATTERY_LIFE_COST_PER_STEP_FOR_NORM = 0.001  # 0.1% of battery_life_cost per step

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

class SolarBatteryEnv(gym.Env):
    """
    A gym environment for home solar-battery-grid energy management.
    Action space: (battery_flow)
        battery_flow > 0 -> battery charge, < 0 -> battery discharge; normalized to [-1, 1]

    Observation (if normalized): 
    [
        hour_sin, hour_cos, day_sin, day_cos,  # Cyclical time features [-1, 1]
        NormalizedSolarGen, NormalizedHouseLoad, ... , # DF features [0, 1]
        NormalizedBatteryLevel, NormalizedBatteryDegCost # Extra features [0, 1]
    ]
    Observation (if not normalized):
    [
        hour_sin, hour_cos, day_sin, day_cos,  # Cyclical time features [-1, 1]
        RawSolarGen, RawHouseLoad, ... ,        # DF features (raw values)
        RawBatteryLevel, RawBatteryDegCost      # Extra features (raw values)
    ]
    """
    metadata = {'render.modes': ['human', 'file', 'None']}
    
    # The environment expects a DataFrame with columns:
    # - 'Timestamp': timestamp (as index)
    # - 'SolarGen': solar energy generation (kWh)
    # - 'HouseLoad': household energy consumption (kWh)
    # - 'FutureSolar': forecasted solar generation (kWh)
    # - 'FutureLoad': forecasted household load (kWh)
    # - 'ImportEnergyPrice': time-based energy price ($/kWh)
    # - 'ExportEnergyPrice': time-based energy price ($/kWh)
    # - 'Time': original time column (for reference in datetime format, should be dropped in obs)
    # - Additional columns can be included for custom observations

    def __init__(
        self,
        df: pl.DataFrame,
        battery_capacity=7.0, #kWh (default Tesla Powerwall 1)
        max_battery_flow=3.3, #kW
        max_grid_flow=10.0, #kW
        init_battery_level=5.0, #kWh
        max_step=1000,
        render_mode=None,
        battery_life_cost=5000.0,  # cost of the battery over its lifetime (USD), this is for calculating the battery degradation cost
        base_deg_DoD = 80.0,  # reference DoD for per-kWh wear linearization (%)
        step_duration = 0.5, # duration of each step in hours (default half an hour)
        degradation_temperature = 25.0,
        degradation_mode: str = "full",  # "full", "cycle_only", "disabled"
    ):
        super(SolarBatteryEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.max_step = max_step
        self.initial_battery_capacity = float(battery_capacity)
        self.battery_capacity = float(battery_capacity)  # effective capacity (fades)
        self.battery_level = init_battery_level
        self.init_battery_level = init_battery_level
        self.max_battery_flow = max_battery_flow
        self.max_grid_energy = max_grid_flow*step_duration
        self.render_mode = render_mode
        self.battery_life_cost = battery_life_cost
        self.base_deg_DoD = base_deg_DoD
        # Automatically determine step_duration from the DataFrame's 'Time' column
        # Assumes 'Time' is in a format compatible with numpy.datetime64 or pandas.Timestamp
        timestamps = self.df['Time'].to_numpy()
        if len(timestamps) >= 2:
            # Try to infer step duration in hours
            try:
                # Convert to numpy.datetime64 if not already
                t0 = np.datetime64(timestamps[0])
                t1 = np.datetime64(timestamps[1])
                delta_hours = (t1 - t0) / np.timedelta64(1, 'h')
                self.step_duration = float(delta_hours)
            except Exception:
                # Fallback to default if conversion fails
                self.step_duration = step_duration
        else:
            self.step_duration = step_duration

        self.degradation_temperature = float(degradation_temperature)
        self.degradation_mode = degradation_mode
        
        # Initialize degradation model based on mode
        if degradation_mode == "cycle_only":
            self.degradation_model = CycleOnlyDegradationModel()
        elif degradation_mode == "disabled":
            self.degradation_model = DisabledDegradationModel()
        else:  # "full"
            self.degradation_model = DegradationModel()
        
        self._rainflow_counter = RainflowCounter(step_duration=self.step_duration, max_c_rate=self.max_battery_flow / self.initial_battery_capacity)
        self._rainflow_num_cycles = 0

        # Empty list for debugging
        self.deg_incidents = []

        # Initialize state of charge history
        self.soc_history = []
        self.static_deg_history = []
        self.total_degradation = 0.0
        self._rainflow_deg_cumulative = 0.0
        
        # Action space (1D): battery_flow(normalized to [-1,1])
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # always use normalized observations
        self.normalize_obs = True

        # --- Normalization Parameters ---
        self.ordered_df_cols_for_obs = [col for col in self.df.columns if col not in ['Time', 'Timestamp']]
        norm_by_capacity_cols = {"SolarGen", "HouseLoad", "FutureSolar", "FutureLoad"}
        self._norm_by_capacity_mask = np.array([col in norm_by_capacity_cols for col in self.ordered_df_cols_for_obs], dtype=bool)
        
        self.df_mins_for_obs = np.array([self.df.select(pl.min(col)).item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
        self.df_maxs_for_obs = np.array([self.df.select(pl.max(col)).item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
        self.df_ranges_for_obs = self.df_maxs_for_obs - self.df_mins_for_obs
        self.df_ranges_for_obs[self.df_ranges_for_obs == 0] = 1.0

        self.battery_level_min_raw = 0.0
        self.battery_level_max_raw = self.initial_battery_capacity

        self.battery_deg_cost_min_raw = 0.0
        # Max raw degradation cost for observation space if not normalizing
        # (Used if raw obs is primary, or for the raw_obs in info dict)
        self.battery_deg_cost_max_raw_obs_bound = MAX_RAW_BATTERY_DEG_COST_IN_OBS_FACTOR * battery_life_cost
        if self.battery_deg_cost_max_raw_obs_bound == 0: self.battery_deg_cost_max_raw_obs_bound = 1.0

        # Max degradation cost for normalization purposes (used if norm_obs is primary)
        self.battery_deg_cost_max_for_norm = MAX_PCT_BATTERY_LIFE_COST_PER_STEP_FOR_NORM * battery_life_cost
        if self.battery_deg_cost_max_for_norm == 0: self.battery_deg_cost_max_for_norm = 1.0
        
        # --- Observation Space Definition (for the primary observation) ---
        num_cyclical_features = 4
        num_df_obs_features = len(self.ordered_df_cols_for_obs)
        num_extra_obs_features = 2 # battery_level, battery_deg_cost
        total_obs_features = num_cyclical_features + num_df_obs_features + num_extra_obs_features
        
        obs_space_low = np.zeros(total_obs_features, dtype=np.float32)
        obs_space_high = np.zeros(total_obs_features, dtype=np.float32)

        obs_space_low[0:num_cyclical_features] = -1.0
        obs_space_high[0:num_cyclical_features] = 1.0
        
        start_idx = num_cyclical_features
        end_idx = start_idx + num_df_obs_features
        if self.normalize_obs: # Primary observation will be normalized
            obs_space_low[start_idx:end_idx] = 0.0
            obs_space_high[start_idx:end_idx] = 1.0
            obs_space_low[end_idx:end_idx+num_extra_obs_features] = 0.0
            obs_space_high[end_idx:end_idx+num_extra_obs_features] = 1.0
        else: # Primary observation will be raw
            obs_space_low[start_idx:end_idx] = self.df_mins_for_obs
            obs_space_high[start_idx:end_idx] = self.df_maxs_for_obs
            obs_space_low[end_idx] = self.battery_level_min_raw
            obs_space_high[end_idx] = self.battery_level_max_raw
            obs_space_low[end_idx+1] = self.battery_deg_cost_min_raw
            obs_space_high[end_idx+1] = self.battery_deg_cost_max_raw_obs_bound

        self.observation_space = spaces.Box(
            low=obs_space_low,
            high=obs_space_high,
            shape=(total_obs_features,),
            dtype=np.float32
        )

    def _get_observation_components(self, current_step_actual_deg_cost=0.0):
        """
        Helper to compute all components for both raw and normalized observations.
        Returns:
            cyclical_time_features (np.array): Shape (4,)
            raw_df_values (np.array): Raw values from DF for obs.
            normalized_df_values (np.array): Normalized DF values.
            raw_extra_features (np.array): Raw [battery_level, deg_cost]. Shape (2,)
            normalized_extra_features (np.array): Normalized [battery_level, deg_cost]. Shape (2,)
        """
        # Gymnasium permits the terminal observation to duplicate the final
        # valid state.  ``step()`` advances before producing that observation,
        # so clamp only the lookup rather than indexing one past the frame.
        row_dict = self._get_row(min(self.current_step, len(self.df) - 1))
        time_str = row_dict.pop('Time', None)
        row_dict.pop('Timestamp', None)

        if time_str is not None:
            dt = np.datetime64(time_str)
            hour = dt.astype('datetime64[h]').astype(int) % 24
            day_of_year = (dt - dt.astype('datetime64[Y]')).astype('timedelta64[D]').astype(int) + 1
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_cos = np.cos(2 * np.pi * day_of_year / 365)
            cyclical_time_features = np.array([hour_sin, hour_cos, day_sin, day_cos], dtype=np.float32)
        else:
            cyclical_time_features = np.zeros(4, dtype=np.float32)

        raw_df_values = np.array([row_dict[col] for col in self.ordered_df_cols_for_obs], dtype=np.float32)
        # Normalize selected columns by battery_capacity, others by min-max
        normalized_df_values = np.empty_like(raw_df_values)
        # Vectorized normalization using pre-computed mask
        cap = self.initial_battery_capacity
        normalized_df_values[self._norm_by_capacity_mask] = raw_df_values[self._norm_by_capacity_mask] / (cap + 1e-9)
        normalized_df_values[~self._norm_by_capacity_mask] = (
            (raw_df_values[~self._norm_by_capacity_mask] - self.df_mins_for_obs[~self._norm_by_capacity_mask])
            / self.df_ranges_for_obs[~self._norm_by_capacity_mask]
        )
        normalized_df_values = normalized_df_values.astype(np.float32)
        normalized_df_values = np.clip(normalized_df_values, 0.0, 1.0)

        raw_battery_level = np.float32(self.battery_level)
        raw_battery_deg_cost = np.float32(current_step_actual_deg_cost)
        raw_extra_features = np.array([raw_battery_level, raw_battery_deg_cost], dtype=np.float32)

        norm_battery_level = (raw_battery_level - self.battery_level_min_raw) / (self.battery_level_max_raw - self.battery_level_min_raw + 1e-9)
        norm_battery_level = np.clip(norm_battery_level, 0.0, 1.0)
        
        norm_battery_deg_cost = (raw_battery_deg_cost - self.battery_deg_cost_min_raw) / (self.battery_deg_cost_max_for_norm - self.battery_deg_cost_min_raw + 1e-9)
        norm_battery_deg_cost = np.clip(norm_battery_deg_cost, 0.0, 1.0)
        normalized_extra_features = np.array([norm_battery_level, norm_battery_deg_cost], dtype=np.float32)
        
        return cyclical_time_features, raw_df_values, normalized_df_values, raw_extra_features, normalized_extra_features
    

    # Helper method to retrieve a row from the Polars DataFrame as a dictionary.
    def _get_row(self, index: int) -> dict:
        row_tuple = self.df.row(index)  # Polars returns a tuple for the row.
        return dict(zip(self.df.columns, row_tuple))
    
    def get_raw_obs(self):
        """
        Returns the raw observation components.
        This is useful for getting the raw observation without stepping the environment.
        """
        ctf, rdfv, ndfv, ref, nef = self._get_observation_components()
        return np.concatenate((ctf, rdfv, ref))

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed) # Important for Gymnasium compatibility
        self.current_step = 0
        self.battery_capacity = self.initial_battery_capacity
        self.total_degradation = 0.0
        self._rainflow_deg_cumulative = 0.0
        self.battery_level = np.clip(self.init_battery_level, 0, self.battery_capacity)
        # Store SoC at step boundaries (percent). Keep 1 more SoC point than actions.
        self.soc_history = [float((self.battery_level / self.battery_capacity) * 100.0)]
        self.static_deg_history = []
        self.last_dynamic_deg = 0.0
        self.last_num_cycles = 0
        self._rainflow_counter = RainflowCounter(step_duration=self.step_duration)
        self._rainflow_num_cycles = 0
        
        self.deg_incidents = []

        components = self._get_observation_components(current_step_actual_deg_cost=0.0)
        ctf, _, ndfv, _, nef = components

        info = {}

        primary_obs = np.concatenate((ctf, ndfv, nef))

        return primary_obs, info

    def _calculate_grid_reward(self, grid_energy, energy_price):
        # If grid energy exceeds limits, add a violation penalty.
        grid_violation_penalty = VIOLATION_PENALTY if abs(grid_energy) > self.max_grid_energy else 0
        # Grid reward: negative cost for importing energy (or reward for exporting)
        grid_reward = -(grid_energy * energy_price) + grid_violation_penalty
        return grid_reward

    def _make_reward_info(
        self,
        battery_flow_energy: float,
        battery_level: float,
        grid_energy: float,
        energy_conservation_violation: bool = False,
        dynamic_updated: bool = False,
        last_dynamic_deg: float = 0.0,
        last_num_cycles: int = 0,
        step_degradation: float = 0.0,
        total_degradation: float = 0.0,
        capacity_kwh: float = 0.0,
        current_step: int = 0,
        new_cycles_in_step: int = 0,
        rainflow_cumulative_deg: float = 0.0,
        deg_incident: bool = False,
        deg_error: str = ""
    ) -> dict:
        """Build a compact reward/info dict with key debugging signals about degradation and cycles."""
        return {
            "battery_flow_energy": float(battery_flow_energy),
            "battery_level": float(battery_level),
            "grid_energy": float(grid_energy),
            "dynamic_updated": bool(dynamic_updated),
            "last_dynamic_deg": float(last_dynamic_deg),
            "last_num_cycles": int(last_num_cycles),
            "new_cycles_in_step": int(new_cycles_in_step),
            "rainflow_cumulative_deg": float(rainflow_cumulative_deg),
            "energy_conservation_violation": bool(energy_conservation_violation),
            "step_degradation": float(step_degradation),
            "total_degradation": float(total_degradation),
            "capacity_kwh": float(capacity_kwh),
            "current_step": int(current_step),
            "deg_incident": deg_incident,
            "deg_error": deg_error
        }

    def _safe_degradation_per_cycle(self, Id: float, Ich: float, soc: float, DoD: float):
        """Compute degradation per cycle; return (value, error_message)."""
        try:
            value = self.degradation_model.degradation_per_cycle(
                T=self.degradation_temperature,
                Id=Id,
                Ich=Ich,
                SOCav=soc,
                DOD=DoD,
            )
            return float(value), None
        except ValueError as exc:
            return 0.0, str(exc)

    def step(self, action):
        # ----- Scale Actions -----
        battery_flow_rate = np.clip(
            action[0] * self.max_battery_flow,
            -self.max_battery_flow,
            self.max_battery_flow
        )

        # Convert power (kW) to energy (kWh) over the step duration.
        battery_flow_energy = battery_flow_rate * self.step_duration

        # ----- Check if battery level can support battery flow action -----
        if battery_flow_energy < 0:  # Discharging action: ensure sufficient battery level
            battery_flow_energy = max(battery_flow_energy, -self.battery_level)
        else:  # Charging action: ensure battery does not exceed its capacity
            battery_flow_energy = min(battery_flow_energy, self.battery_capacity - self.battery_level)

        # ----- Update Battery Level & Check Constraints -----
        new_battery_level = self.battery_level + battery_flow_energy
        soc_after = float(np.clip((new_battery_level / self.battery_capacity) * 100.0, 0.0, 100.0))

        # ----- Retrieve Current Data -----
        row = self._get_row(self.current_step)
        solar = row['SolarGen']
        load = row['HouseLoad']
        
        # ----- Determine battery charge and discharge -----
        battery_charge = max(0, battery_flow_energy)
        battery_discharge = max(0, -battery_flow_energy)

        # ----- Compute grid_flow automatically -----
        demand = load + battery_charge
        supply = solar + battery_discharge
        grid_energy_needed = demand - supply  # If positive, importing; if negative, exporting

        # Clip the grid flow if needed
        grid_energy = np.clip(grid_energy_needed, -self.max_grid_energy, self.max_grid_energy)

        # Determine energy price based on grid flow (import vs export).
        # grid_energy: positive => import (pay ImportEnergyPrice),
        #              negative => export (receive ExportEnergyPrice).
        energy_price = row['ImportEnergyPrice'] if grid_energy >= 0 else row['ExportEnergyPrice']

        # Check if this clipping breaks energy conservation
        actual_supply = supply + grid_energy
        tolerance = 1e-2  # Tolerance for energy conservation check
        if abs(actual_supply - demand) > tolerance:
            # Return a large negative reward and flag violation
            components = self._get_observation_components()
            ctf, rdfv, ndfv, ref, nef = components
            primary_obs = np.concatenate((ctf, ndfv, nef))
            reward_info = self._make_reward_info(
                battery_flow_energy=battery_flow_energy,
                battery_level=new_battery_level,
                grid_energy=grid_energy,
                energy_conservation_violation=False,
                last_dynamic_deg=self._rainflow_deg_cumulative,
                last_num_cycles=self._rainflow_num_cycles,
                new_cycles_in_step=0,
                rainflow_cumulative_deg=self._rainflow_deg_cumulative,
                current_step=self.current_step,
                step_degradation=0.0,
                total_degradation=self.total_degradation,
                capacity_kwh=self.battery_capacity,
            )

            return primary_obs, np.float64(VIOLATION_PENALTY), True, False, reward_info


        # Record SoC for degradation tracking (percent) and feed the counter
        self.soc_history.append(soc_after)
        new_cycles = self._rainflow_counter.update(soc_after)

        step_degradation = 0.0
        deg_error = ""
        for SoC_avg, DoD, Id_cycle, Ich_cycle in new_cycles:
            inc, err = self._safe_degradation_per_cycle(Id_cycle, Ich_cycle, SoC_avg, DoD)
            if err:
                deg_error = err
                break
            step_degradation += inc

        if deg_error:
            reward_info = self._make_reward_info(
                battery_flow_energy=battery_flow_energy,
                battery_level=new_battery_level,
                grid_energy=grid_energy,
                energy_conservation_violation=False,
                last_dynamic_deg=self._rainflow_deg_cumulative,
                last_num_cycles=self._rainflow_num_cycles,
                new_cycles_in_step=0,
                rainflow_cumulative_deg=self._rainflow_deg_cumulative,
                current_step=self.current_step,
                step_degradation=0.0,
                total_degradation=self.total_degradation,
                capacity_kwh=self.battery_capacity,
                deg_error=deg_error,
            )

            components = self._get_observation_components()
            ctf, rdfv, ndfv, ref, nef = components
            primary_obs = np.concatenate((ctf, ndfv, nef))
            return primary_obs, float(VIOLATION_PENALTY), True, False, reward_info

        self._rainflow_num_cycles += len(new_cycles)
        self._rainflow_deg_cumulative += step_degradation
        self.total_degradation = min(1.0, self.total_degradation + step_degradation)
        self.last_dynamic_deg = self._rainflow_deg_cumulative
        self.last_num_cycles = self._rainflow_num_cycles

        # Capacity fade (normalization remains fixed to initial capacity)
        self.battery_capacity = max(self.initial_battery_capacity * (1.0 - self.total_degradation), 1e-9)
        # Ensure stored energy respects faded capacity
        self.battery_level = min(new_battery_level, self.battery_capacity)

        current_step_deg_cost = step_degradation * self.battery_life_cost

        # ----- Compute Rewards (electricity cost and grid violation)-----
        grid_reward = self._calculate_grid_reward(grid_energy, energy_price)

        # Final reward: trade-off energy cost vs degradation cost
        reward = grid_reward - current_step_deg_cost
        # check if step degradation is abnormally large

        if step_degradation > 0.05:  # realistic explosion threshold
            debug = self.degradation_model.debug_degradation_per_cycle(
                T=self.degradation_temperature,
                Id=Id_cycle,
                Ich=Ich_cycle,
                SOCav=SoC_avg,
                DOD=DoD,
            )

            incident = {
                "episode_id": None,  # fill later
            }

            for k in DEG_INCIDENT_FIELDS :
                incident[k] = float(debug.get(k)) if debug.get(k) is not None else None
            incident["step"] = self.current_step
            incident["step_degradation"] = float(step_degradation)

            self.deg_incidents.append(incident)

        reward_info = self._make_reward_info(
            battery_flow_energy=battery_flow_energy,
            battery_level=self.battery_level,
            grid_energy=grid_energy,
            energy_conservation_violation=False,
            last_dynamic_deg=self._rainflow_deg_cumulative,
            last_num_cycles=self._rainflow_num_cycles,
            step_degradation=step_degradation,
            total_degradation=self.total_degradation,
            capacity_kwh=self.battery_capacity,
            current_step=self.current_step,
            new_cycles_in_step=len(new_cycles),
            rainflow_cumulative_deg=self._rainflow_deg_cumulative,
            deg_incident=len(self.deg_incidents) > 0
        )

        # ----- Advance Simulation Step -----
        self.current_step += 1
        truncated = (self.current_step >= min(self.max_step, len(self.df)))
        terminated = bool(self.total_degradation >= 1.0)

        components = self._get_observation_components(current_step_actual_deg_cost=current_step_deg_cost)
        ctf, rdfv, ndfv, ref, nef = components

        primary_obs = np.concatenate((ctf, ndfv, nef))

        return primary_obs, float(reward), terminated, truncated, reward_info

def render(self, **kwargs):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh, Solar: {self.df['SolarGen'][self.current_step]:.2f} kWh, Load: {self.df['HouseLoad'][self.current_step]:.2f} kWh")
