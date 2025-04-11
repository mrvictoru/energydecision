# import the necessary packages
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces, error, utils

import numpy as np
import polars as pl

from batterydeg import static_degradation, dynamic_degradation

# global variables
VIOLATION_PENALTY = -1000


class SolarBatteryEnv(gym.Env):
    """
    A gym environment for solar-battery-grid energy management.
    Action space: (battery_flow, grid_flow)
        battery_flow > 0 -> battery charge, < 0 -> battery discharge
        grid_flow > 0 -> import from grid,   < 0 -> export to grid
    Observation: [Time, SolarGen, HouseLoad, BatteryLevel, GridFlow, ...]
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
        battery_life_cost=15300,  # cost of the battery over its lifetime (USD), this is for calculating the battery degradation cost
        correction_interval = 100, # steps before dynamic correction
        step_duration = 0.5 # duration of each step in hours (default half an hour)
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
        self.step_duration = step_duration

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
        num_features = self.df.shape[1] -1 + 2  # adding battery level, battery degradation cost and removing 'Time'
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )

    # Helper method to retrieve a row from the Polars DataFrame as a dictionary.
    def _get_row(self, index: int) -> dict:
        row_tuple = self.df.row(index)  # Polars returns a tuple for the row.
        return dict(zip(self.df.columns, row_tuple))

    def reset(self, seed=None, **kwargs):
        self.current_step = 0
        self.battery_level = min(self.battery_capacity, self.battery_level)
        self.soc_history = []
        self.static_deg_history = []
        self.correction_factor = 1.0
        return self._next_observation(), {}

    def _next_observation(self, battery_deg_penalty=0.0):
        # Retrieve the current row as a dictionary using Polars.
        row = self._get_row(self.current_step)
        # Drop the 'Time' column and convert the remaining dictionary values to a numpy array.
        row.pop('Time', None)  # Remove the 'Time' column if it exists.
        row_array = np.array(list(row.values()), dtype=np.float32)
        extra_features = np.array([self.battery_level, battery_deg_penalty], dtype=np.float32)
        obs = np.concatenate((row_array, extra_features))
        # the observation is a 1D array with the following features:
        # [Timestamp, SolarGen, HouseLoad, FutureSolar, FutureLoad, ImportEnergyPrice, ExportEnergyPrice, BatteryLevel, BatteryDegCost]
        return obs
    
    def get_observation_header(self):
        row = self._get_row(0)
        # drop the 'Time' column
        row = {k: v for k, v in row.items() if k != 'Time'}
        return list(row.keys()) + ['BatteryLevel', 'BatteryDegCost']

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
        return battery_deg_penalty*self.correction_factor, battery_deg_penalty

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
        avg_soc = (self.battery_level-0.5*(-battery_flow_energy))/self.battery_capacity * 100
        DoD = abs(battery_flow_energy / self.battery_capacity) * 100  # Depth of discharge in percentage
        battery_deg_penalty, static_deg = self._calculate_battery_degradation(DoD, battery_flow_rate, avg_soc)

        # Record SoC for dynamic degradation correction
        self.soc_history.append(self.battery_level/self.battery_capacity * 100)
        self.static_deg_history.append(battery_deg_penalty)
        num_cycles = 0 # Placeholder for number of cycles
        dynamic_deg_percent = -99 # Placeholder for dynamic degradation percentage
        # ----- Dynamic Degradation Correction -----
        if self.current_step > 0 and self.current_step % self.correction_interval == 0:
            dynamic_deg_percent, num_cycles = dynamic_degradation(self.soc_history)
            # convert percentage to fraction
            dynamic_deg = dynamic_deg_percent/100.0
            static_deg_sum = np.sum(self.static_deg_history, dtype=np.float64)
            self.correction_factor = dynamic_deg / static_deg_sum if static_deg_sum > 0 else 1.0


        # ----- Compute Final Reward -----
        reward = grid_reward - battery_deg_penalty*self.battery_life_cost

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
            "dynamic_deg_percent": dynamic_deg_percent,
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

        obs = self._next_observation(battery_deg_penalty=battery_deg_penalty*self.battery_life_cost)
        return obs, reward, terminated, truncated, {"reward_info": reward_info}

    def render(self, **kwargs):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh, Solar: {self.df['SolarGen'][self.current_step]:.2f} kWh, Load: {self.df['HouseLoad'][self.current_step]:.2f} kWh")
        elif self.render_mode == 'file':
            filename = kwargs.get('filename', 'render.txt')
            with open(filename, 'a+') as f:
                f.write(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh, Solar: {self.df['SolarGen'][self.current_step]:.2f} kWh, Load: {self.df['HouseLoad'][self.current_step]:.2f} kWh\n")

