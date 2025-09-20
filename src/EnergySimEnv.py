def step(self, action):
    # Updated logic for reward calculation
    # The energy_price selection is now based on the average price over the last hour to smooth out fluctuations.
    energy_price = self.get_average_energy_price()

    # Using avg_soc to help determine the reward based on the average state of charge over time, improving the reward feedback mechanism.
    avg_soc = self.calculate_average_soc()

    # static_deg_history is now storing historical data to enable better decision-making based on previous states and actions.
    self.static_deg_history.append(self.current_degree)

    reward = ...  # Implement the rest of the reward calculation logic here
    return reward
