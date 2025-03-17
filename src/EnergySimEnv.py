# import the necessary packages
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces, error, utils

import numpy as np
import pandas as pd

from batterydeg import static_degradation, dynamic_degradation


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
    # - 'Time': timestamp
    # - 'SolarGen': solar energy generation (kWh)
    # - 'HouseLoad': household energy consumption (kWh)
    # - 'EnergyPrice': time-based energy price ($/kWh)
    # - 'PredictedSolar': forecasted solar generation (kWh)
    # - 'PredictedLoad': forecasted household load (kWh)
    # - 'EnergyPrice': time-based energy price ($/kWh)

    def __init__(
        self,
        df,
        battery_capacity=10.0, #kWh
        max_battery_flow=2.0, #kW
        max_grid_flow=2.0, #kW
        init_battery_level=5.0, #kWh
        max_step=1000,
        render_mode=None,
        battery_deg_cost=0.02,  # cost per kWh cycled in degradation
        correction_interval = 100 # steps before dynamic correction
    ):
        super(SolarBatteryEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.max_step = max_step
        self.battery_capacity = battery_capacity
        self.battery_level = init_battery_level
        self.max_battery_flow = max_battery_flow
        self.max_grid_flow = max_grid_flow
        self.render_mode = render_mode
        self.battery_deg_cost = battery_deg_cost
        self.correction_interval = correction_interval

        # Initialize state of charge history for dynamic correction
        self.soc_history = []
        self.correction_factor = 1.0
        
        # Action space (2D): battery_flow, grid_flow (both normalized to [-1,1])
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        # Observation: [SolarGen, HouseLoad, BatteryLevel, EnergyPrice]
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

    def reset(self, seed=None, **kwargs):
        self.current_step = 0
        self.battery_level = min(self.battery_capacity, self.battery_level)
        self.soc_history = []
        self.correction_factor = 1.0
        return self._next_observation(), {}

    def _next_observation(self, grid_flow=0.0):
        row = self.df.iloc[self.current_step]
        # Example columns: 'SolarGen', 'HouseLoad'
        solar = row['SolarGen']
        load = row['HouseLoad']
        predicted_solar = row['PredictedSolar']
        predicted_load = row['PredictedLoad']

        obs = np.array([
            solar,
            load,
            predicted_solar,
            predicted_load,
            self.battery_level,
            grid_flow  # placeholder for net grid flow, if desired
        ], dtype=np.float32)
        return obs

    def step(self, action):
        # ----- Scale Actions -----
        battery_flow = np.clip(
            action[0] * self.max_battery_flow,
            -self.max_battery_flow,
            self.max_battery_flow
        )
        grid_flow = np.clip(
            action[1] * self.max_grid_flow,
            -self.max_grid_flow,
            self.max_grid_flow
        )

        # ----- Retrieve Current Data -----
        row = self.df.iloc[self.current_step]
        solar = row['SolarGen']
        load = row['HouseLoad']
        energy_price = row['EnergyPrice']

        # ----- Energy Conservation Check -----
        # Determine energy going into or out of battery
        battery_charge = max(0, battery_flow)
        battery_discharge = max(0, -battery_flow)
        supply = solar + grid_flow + battery_discharge
        demand = load + battery_charge
        tolerance = 1e-3

        if abs(supply - demand) > tolerance:
            # Return a large negative reward and flag violation
            large_penalty = 1000
            obs = self._next_observation()
            return obs, -large_penalty, True, False, {"energy_conservation_violation": True}

        # ----- Update Battery Level & Check Constraints -----
        new_battery_level = self.battery_level + battery_flow
        new_battery_level = np.clip(new_battery_level, 0, self.battery_capacity)

        is_constraint_violated = (
            (self.battery_level == self.battery_capacity and battery_flow > 0) or
            (self.battery_level == 0 and battery_flow < 0)
        )
        violation_penalty = 100 if is_constraint_violated else 0

        # ----- Compute Grid Reward -----
        grid_cost = grid_flow * energy_price  # cost if import, profit if export
        grid_reward = -grid_cost

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
        reward = grid_reward - battery_deg_penalty - violation_penalty

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

