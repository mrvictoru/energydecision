# import the necessary packages
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces, error, utils

import numpy as np
import polars as pl

from batterydeg import static_degradation, dynamic_degradation

# global variables
VIOLATION_PENALTY = -1000
MAX_RAW_BATTERY_DEG_COST_IN_OBS_FACTOR = 0.01  # 1% of battery_life_cost per step
MAX_PCT_BATTERY_LIFE_COST_PER_STEP_FOR_NORM = 0.001  # 0.1% of battery_life_cost per step

class SolarBatteryEnv(gym.Env):
    """
    A gym environment for solar-battery-grid energy management.
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
        battery_capacity=13.5, #kWh (default Tesla Powerwall 2)
        max_battery_flow=5.0, #kW
        max_grid_flow=10.0, #kW
        init_battery_level=5.0, #kWh
        max_step=1000,
        render_mode=None,
        battery_life_cost=15300.0,  # cost of the battery over its lifetime (USD), this is for calculating the battery degradation cost
        correction_interval = 100, # steps before dynamic correction
        init_correction_steps = [10, 20, 40 ,70, 110, 160],
        step_duration = 0.5, # duration of each step in hours (default half an hour)
        normalize_obs: bool = True
    ):
        super(SolarBatteryEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.max_step = max_step
        self.battery_capacity = battery_capacity
        self.battery_level = init_battery_level
        self.max_battery_flow = max_battery_flow
        self.max_grid_energy = max_grid_flow*step_duration
        self.render_mode = render_mode
        self.battery_life_cost = battery_life_cost
        self.correction_interval = correction_interval
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
        self.init_correction_steps = init_correction_steps

        # Initialize state of charge history for dynamic correction
        self.soc_history = []
        self.static_deg_history = []
        self.correction_factor = 1.0
        
        # Action space (1D): battery_flow(normalized to [-1,1])
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # ... (rest of your __init__ assignments) ...
        self.normalize_obs = normalize_obs # Store the preference

        # --- Normalization Parameters ---
        self.ordered_df_cols_for_obs = [col for col in self.df.columns if col not in ['Time', 'Timestamp']]
        
        self.df_mins_for_obs = np.array([self.df.select(pl.min(col)).item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
        self.df_maxs_for_obs = np.array([self.df.select(pl.max(col)).item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
        self.df_ranges_for_obs = self.df_maxs_for_obs - self.df_mins_for_obs
        self.df_ranges_for_obs[self.df_ranges_for_obs == 0] = 1.0

        self.battery_level_min_raw = 0.0
        self.battery_level_max_raw = self.battery_capacity

        self.battery_deg_cost_min_raw = 0.0
        # Max raw degradation cost for observation space if not normalizing
        # (Used if raw obs is primary, or for the raw_obs in info dict)
        self.battery_deg_cost_max_raw_obs_bound = MAX_RAW_BATTERY_DEG_COST_IN_OBS_FACTOR * self.battery_life_cost
        if self.battery_deg_cost_max_raw_obs_bound == 0: self.battery_deg_cost_max_raw_obs_bound = 1.0

        # Max degradation cost for normalization purposes (used if norm_obs is primary)
        self.battery_deg_cost_max_for_norm = MAX_PCT_BATTERY_LIFE_COST_PER_STEP_FOR_NORM * self.battery_life_cost
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
        row_dict = self._get_row(self.current_step)
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
        normalized_df_values = (raw_df_values - self.df_mins_for_obs) / self.df_ranges_for_obs
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

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed) # Important for Gymnasium compatibility
        self.current_step = 0
        self.battery_level = np.clip(self.init_battery_level, 0, self.battery_capacity)
        self.soc_history = []
        self.static_deg_history = []
        self.correction_factor = 1.0
        
        components = self._get_observation_components(current_step_actual_deg_cost=0.0)
        ctf, rdfv, ndfv, ref, nef = components

        info = {}
        if self.normalize_obs:
            primary_obs = np.concatenate((ctf, ndfv, nef))
            info['raw_obs'] = np.concatenate((ctf, rdfv, ref))
        else:
            primary_obs = np.concatenate((ctf, rdfv, ref))
            info['norm_obs'] = np.concatenate((ctf, ndfv, nef))
            
        return primary_obs, info

    # ... (get_observation_header, _calculate_grid_reward, _calculate_battery_degradation, render) ...
    # Make sure get_observation_header also reflects the primary observation format
    def get_observation_header(self):
        header = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        prefix = "Norm_" if self.normalize_obs else "" # Add prefix if primary obs is normalized
        header.extend([f"{prefix}{col}" for col in self.ordered_df_cols_for_obs])
        header.extend([f'{prefix}BatteryLevel', f'{prefix}BatteryDegCost'])
        return header

    def _calculate_grid_reward(self, grid_energy, energy_price):
        # If grid energy exceeds limits, add a violation penalty.
        grid_violation_penalty = VIOLATION_PENALTY if abs(grid_energy) > self.max_grid_energy else 0
        # Grid reward: negative cost for importing energy (or reward for exporting)
        grid_reward = -(grid_energy * energy_price) + grid_violation_penalty
        return grid_reward, grid_violation_penalty

    def _calculate_battery_degradation(self, DoD, battery_flow_rate, soc):
        # battery flow rate is negative for discharge and positive for charge
        Id = abs(max(0, -battery_flow_rate) / self.battery_capacity) # discharge c rate
        Ich = abs(max(0, battery_flow_rate) / self.battery_capacity)
        battery_deg_penalty = static_degradation(Id, Ich, soc, DoD)
        return battery_deg_penalty*self.correction_factor, battery_deg_penalty # return degradation after correction factor and static degradation

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

        # ----- Retrieve Current Data -----
        row = self._get_row(self.current_step)
        solar = row['SolarGen']
        load = row['HouseLoad']
        energy_price = row['ImportEnergyPrice'] if battery_flow_energy >= 0 else row['ExportEnergyPrice']

        # ----- Determine battery charge and discharge -----
        battery_charge = max(0, battery_flow_energy)
        battery_discharge = max(0, -battery_flow_energy)

        # ----- Compute grid_flow automatically -----
        demand = load + battery_charge
        supply = solar + battery_discharge
        grid_energy_needed = demand - supply  # If positive, importing; if negative, exporting

        # Clip the grid flow if needed and flag a violation penalty if limits are exceeded
        grid_violation_penalty = VIOLATION_PENALTY if abs(grid_energy_needed) > self.max_grid_energy else 0
        grid_energy = np.clip(grid_energy_needed, -self.max_grid_energy, self.max_grid_energy)

        # Check if this clipping breaks energy conservation
        actual_supply = supply + grid_energy
        tolerance = 1e-2  # Tolerance for energy conservation check
        if abs(actual_supply - demand) > tolerance:
            # Return a large negative reward and flag violation
            obs = self._next_observation()
            return obs, VIOLATION_PENALTY, True, False, {"energy_conservation_violation": True}

        # ----- Compute Rewards -----
        grid_reward, grid_violation_penalty = self._calculate_grid_reward(grid_energy, energy_price)
        # SoC_avg = 100 · ((q_t) − 0.5 · (b_t) / B)
        # where
        # q_t is the battery energy (in kWh) before the operation,
        # b_t is the energy discharged (kWh) , and
        # B is the nominal battery capacity (kWh).
        avg_soc = (self.battery_level-0.5*(-battery_flow_energy))/self.battery_capacity * 100.0
        DoD = abs(battery_flow_energy / self.battery_capacity) * 100.0  # Depth of discharge in percentage
        battery_deg_penalty, static_deg = self._calculate_battery_degradation(DoD, battery_flow_rate, avg_soc)

        # Record SoC for dynamic degradation correction
        self.soc_history.append(self.battery_level/self.battery_capacity * 100.0)
        self.static_deg_history.append(battery_deg_penalty)
        num_cycles = 0 # Placeholder for number of cycles
        dynamic_deg = -1.0 # Placeholder for dynamic degradation percentage

        # ----- Dynamic Degradation Correction -----
        # Check if the current step is one of the initial frequent steps OR
        # if it's past the initial phase and meets the regular interval condition.
        dynamic_correct = False
        if self.current_step in self.init_correction_steps:
            dynamic_correct = True
        elif self.current_step > (self.init_correction_steps[-1] if self.init_correction_steps else 0) and \
            self.current_step % self.correction_interval == 0:
            dynamic_correct = True

        if dynamic_correct:
            dynamic_deg, num_cycles = dynamic_degradation(self.soc_history, self.step_duration)
            static_deg_sum = np.sum(self.static_deg_history, dtype=np.float64)
            # Avoid division by zero or near-zero
            if abs(static_deg_sum) > 1e-9: 
                self.correction_factor = dynamic_deg / static_deg_sum
            else:
                # Handle case where static sum is zero (e.g., set factor to 1 or log a warning)
                self.correction_factor = 1.0


        # ----- Compute Final Reward -----
        current_step_deg_cost = battery_deg_penalty * self.battery_life_cost
        reward = grid_reward - current_step_deg_cost

        # Log reward calculation details
        reward_info = {
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "battery_level": new_battery_level,
            "demand": demand,
            "supply": supply,
            "grid_energy": grid_energy,
            "energy_price": energy_price,
            "grid_violation_penalty": grid_violation_penalty,
            "grid_reward": grid_reward,
            "battery_deg_penalty": battery_deg_penalty,
            "dynamic_deg": dynamic_deg,
            "static_deg": static_deg,
            "num_cycles": num_cycles,
            "correction_factor": self.correction_factor,
            "final_reward": reward
        }

        # ----- Advance Simulation Step -----
        self.battery_level = new_battery_level
        self.current_step += 1
        truncated = (self.current_step >= self.max_step)
        terminated = False

        components = self._get_observation_components(current_step_actual_deg_cost=current_step_deg_cost)
        ctf, rdfv, ndfv, ref, nef = components

        info_dict_for_return = {"reward_info": reward_info} # Start with reward_info
        if self.normalize_obs:
            primary_obs = np.concatenate((ctf, ndfv, nef))
            info_dict_for_return['raw_obs'] = np.concatenate((ctf, rdfv, ref))
        else:
            primary_obs = np.concatenate((ctf, rdfv, ref))
            info_dict_for_return['norm_obs'] = np.concatenate((ctf, ndfv, nef))

        return primary_obs, float(reward), terminated, truncated, info_dict_for_return

    def render(self, **kwargs):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh, Solar: {self.df['SolarGen'][self.current_step]:.2f} kWh, Load: {self.df['HouseLoad'][self.current_step]:.2f} kWh")
            """
        elif self.render_mode == 'file':
            # Use a filename based on the dataset if possible
            # Auto-generate dataset name based on meta data columns if available
            try:
                customer = self.df.select("Customer").item() if "Customer" in self.df.columns else "unknown"
                postcode = self.df.select("Postcode").item() if "Postcode" in self.df.columns else "unknown"
                daterange = self.df.select("DateRange").item() if "DateRange" in self.df.columns else "unknown"
                dataset_name = f"{customer}_{postcode}_{daterange}"
            except Exception:
                dataset_name = kwargs.get('dataset_name', 'default_dataset')
            filename = kwargs.get('filename', f'render_{dataset_name}.txt')
            # Store the current observation as well
            obs = self._next_observation()
            with open(filename, 'a+') as f:
                f.write(
                    f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh, "
                    f"Solar: {self.df['SolarGen'][self.current_step]:.2f} kWh, "
                    f"Load: {self.df['HouseLoad'][self.current_step]:.2f} kWh, "
                    f"Obs: {obs.tolist()}\n"
            )
            """

