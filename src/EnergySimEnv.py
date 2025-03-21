# import the necessary packages
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces, error, utils

import numpy as np
import polars as pl

from batterydeg import static_degradation, dynamic_degradation

# global variables
VIOLATION_PENALTY = 100


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
        max_grid_flow=7.0, #kW
        init_battery_level=5.0, #kWh
        max_step=1000,
        render_mode=None,
        battery_life_cost=15300,  # cost of the battery over its lifetime (USD), this is for calculating the battery degradation cost
        correction_interval = 100 # steps before dynamic correction
    ):
        super(SolarBatteryEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.max_step = max_step
        self.battery_capacity = battery_capacity
        self.battery_level = init_battery_level
        self.max_battery_flow = max_battery_flow
        self.max_grid_flow = max_grid_flow
        self.render_mode = render_mode
        self.battery_life_cost = battery_life_cost
        self.correction_interval = correction_interval

        # Initialize state of charge history for dynamic correction
        self.soc_history = []
        self.correction_factor = 1.0
        
        # Action space (1D): battery_flow(normalized to [-1,1])
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        num_features = self.df.shape[1] -1 + 2  # adding grid flow and battery level and removing 'Time'
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
        self.correction_factor = 1.0
        return self._next_observation(), {}

    def _next_observation(self, grid_flow=0.0):
        # Retrieve the current row as a dictionary using Polars.
        row = self._get_row(self.current_step)
        # Drop the 'Time' column and convert the remaining dictionary values to a numpy array.
        row.pop('Time', None)  # Remove the 'Time' column if it exists.
        row_array = np.array(list(row.values()), dtype=np.float32)
        extra_features = np.array([self.battery_level, grid_flow], dtype=np.float32)
        obs = np.concatenate((row_array, extra_features))
        return obs

    def step(self, action):
        # ----- Scale Actions -----
        battery_flow = np.clip(
            action[0] * self.max_battery_flow,
            -self.max_battery_flow,
            self.max_battery_flow
        )

        # ----- Check if battery level can support battery flow action -----
        if battery_flow < 0:  # Discharging action: ensure sufficient battery level
            battery_flow = max(battery_flow, -self.battery_level)
        else:  # Charging action: ensure battery does not exceed its capacity
            battery_flow = min(battery_flow, self.battery_capacity - self.battery_level)

        # ----- Update Battery Level & Check Constraints -----
        new_battery_level = self.battery_level + battery_flow

        # ----- Retrieve Current Data -----
        row = self._get_row(self.current_step)
        solar = row['SolarGen']
        load = row['HouseLoad']
        energy_price = row['ImportEnergyPrice'] if battery_flow >= 0 else row['ExportEnergyPrice']

        # ----- Determine battery charge and discharge -----
        battery_charge = max(0, battery_flow)
        battery_discharge = max(0, -battery_flow)

        # ----- Compute grid_flow automatically -----
        demand = load + battery_charge
        supply = solar + battery_discharge
        grid_flow_needed = demand - supply  # If positive, importing; if negative, exporting

        # Clip the grid flow if needed and flag a violation penalty if limits are exceeded
        grid_violation_penalty = VIOLATION_PENALTY if abs(grid_flow_needed) > self.max_grid_flow else 0
        grid_flow = np.clip(grid_flow_needed, -self.max_grid_flow, self.max_grid_flow)

        # Check if this clipping breaks energy conservation
        actual_supply = solar + battery_discharge + grid_flow
        tolerance = 1e-2  # Tolerance for energy conservation check
        if abs(actual_supply - demand) > tolerance:
            # Return a large negative reward and flag violation
            large_penalty = 1000
            obs = self._next_observation(grid_flow=grid_flow)
            return obs, -large_penalty, True, False, {"energy_conservation_violation": True}
        
        # ----- Compute Grid Reward -----
        grid_reward = -(grid_flow * energy_price) - grid_violation_penalty

        # ----- Calculate Battery Degradation Penalty -----
        soc = (self.battery_level / self.battery_capacity) * 100  # State of Charge in %
        DoD = abs(battery_flow / self.battery_capacity) * 100    # Depth of Discharge in %
        Id = abs(battery_flow / self.battery_capacity)
        Ich = abs(battery_flow / self.battery_capacity)

        battery_deg_penalty = static_degradation(Id, Ich, soc, DoD, self.correction_factor)

        # Record SoC for dynamic degradation correction
        self.soc_history.append(soc)

        # ----- Dynamic Degradation Correction -----
        if self.current_step > 0 and self.current_step % self.correction_interval == 0:
            dynamic_deg = dynamic_degradation(self.soc_history)
            static_deg = sum(
                static_degradation(
                    abs(flow / self.battery_capacity),
                    abs(flow / self.battery_capacity),
                    soc_val,
                    abs(flow / self.battery_capacity) * 100,
                    self.correction_factor
                )
                for flow, soc_val in zip(self.soc_history, self.soc_history)
            )
            self.correction_factor = dynamic_deg / static_deg if static_deg > 0 else 1.0
            self.soc_history = []  # Reset history after correction

        # ----- Compute Final Reward -----
        reward = grid_reward - battery_deg_penalty*self.battery_life_cost

        # ----- Advance Simulation Step -----
        self.battery_level = new_battery_level
        self.current_step += 1
        truncated = (self.current_step >= self.max_step)
        terminated = False

        obs = self._next_observation(grid_flow)
        return obs, reward, terminated, truncated, {}

    def render(self, **kwargs):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh")
        elif self.render_mode == 'file':
            filename = kwargs.get('filename', 'render.txt')
            with open(filename, 'a+') as f:
                f.write(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh\n")

