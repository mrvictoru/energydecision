
class SolarBatteryEnv(gym.Env):
    """
    A gym environment for solar-battery-grid energy management.
    Action space: (battery_flow, grid_flow)
        battery_flow > 0 -> battery charge, < 0 -> battery discharge
        grid_flow > 0 -> import from grid,   < 0 -> export to grid
    Observation: [Time, SolarGen, HouseLoad, BatteryLevel, GridFlow, ...]
    """
    metadata = {'render.modes': ['human', 'file', 'None']}

    def __init__(
        self,
        df,
        battery_capacity=10.0,
        max_battery_flow=2.0,
        max_grid_flow=2.0,
        init_battery_level=5.0,
        max_step=1000,
        render_mode=None,
        battery_deg_cost=0.02  # cost per kWh cycled in degradation
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
        return self._next_observation(), {}

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        # Example columns: 'SolarGen', 'HouseLoad'
        solar = row['SolarGen']
        load = row['HouseLoad']

        obs = np.array([
            solar,
            load,
            self.battery_level,
            0.0  # placeholder for net grid flow, if desired
        ], dtype=np.float32)
        return obs

    def step(self, action):
        # Scale action values to actual flow rates
        battery_flow = np.clip(action[0] * self.max_battery_flow, -self.max_battery_flow, self.max_battery_flow)
        grid_flow = np.clip(action[1] * self.max_grid_flow, -self.max_grid_flow, self.max_grid_flow)

        # Get current environmental data
        row = self.df.iloc[self.current_step]
        solar = row['SolarGen']
        load = row['HouseLoad']
        energy_price = row['EnergyPrice']  # time-based energy price in $/kWh
        
        # Update battery with physical limits
        new_battery_level = self.battery_level + battery_flow
        new_battery_level = np.clip(new_battery_level, 0, self.battery_capacity)
        battery_constraint_violation = (
            (self.battery_level == self.battery_capacity and battery_flow > 0) or
            (self.battery_level == 0 and battery_flow < 0)
        )
        if battery_constraint_violation:
            # Severe penalty if physical constraint is violated
            violation_penalty = 100
        else:
            violation_penalty = 0

        # Compute grid cost/profit:
        # If grid_flow > 0 then energy is imported (cost incurred),
        # if grid_flow < 0 then energy is exported (profit earned)
        grid_cost = grid_flow * energy_price  
        # Negative grid_cost (from export) results in a reward boost.
        grid_reward = -grid_cost

        # Battery degradation penalty based on the absolute energy flow
        battery_deg_penalty = self.battery_deg_cost * abs(battery_flow)

        # Final reward accounts for grid energy cost/profit, battery degradation, and any violation penalty.
        reward = grid_reward - battery_deg_penalty - violation_penalty

        # Advance simulation
        self.battery_level = new_battery_level
        self.current_step += 1
        truncated = (self.current_step >= self.max_step)
        terminated = False

        # Next observation includes energy price, e.g.:
        obs = np.array([
            solar,
            load,
            self.battery_level,
            energy_price
        ], dtype=np.float32)

        return obs, reward, terminated, truncated, {}

    def render(self, **kwargs):
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh")
        elif self.render_mode == 'file':
            filename = kwargs.get('filename', 'render.txt')
            with open(filename, 'a+') as f:
                f.write(f"Step: {self.current_step}, Battery: {self.battery_level:.2f} kWh\n")

