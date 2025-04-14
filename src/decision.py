import numpy as np
import torch
import polars as pl
from EnergySimEnv import SolarBatteryEnv

class Agent:
    def __init__(self, env:SolarBatteryEnv , algorithm='rule', model=None, horizon=48, soc_resolution=20):
        """
        env: an instance of SolarBatteryEnv.
        algorithm: choose between 'rule', 'rl', 'dt', or 'sdp'.
        model: For RL algorithm, a trained model with a predict method (e.g., from stable_baselines3).
        horizon: Time horizon for SDP optimization (default: 48 steps).
        soc_resolution: Resolution of state-of-charge discretization (default: 20 levels).
        """
        self.env = env
        self.algorithm = algorithm.lower()
        self.model = model
        self.horizon = horizon
        self.soc_resolution = soc_resolution
        self.value_function = None
        self.policy = None
        self.rule_presistence = False  # Preset for rule-based action persistence

    def choose_action(self, obs):
        if self.algorithm == 'rule':
            return self.rule_based_action(obs)
        elif self.algorithm == 'rl':
            if self.model is None:
                raise ValueError("RL algorithm selected but no model provided.")
            obs_batch = obs[None, ...] if isinstance(obs, np.ndarray) else obs
            action, _ = self.model.predict(obs_batch, deterministic=True)
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
            if self.value_function is None or self.policy is None:
                self.value_function, self.policy = self._sdp_optimization()
            return self.sdp_action(obs)
        else:
            raise NotImplementedError(f"Algorithm '{self.algorithm}' is not supported.")

    def rule_based_action(self, obs):
        diff = obs[1] - obs[2] # difference between solar generation (obs[1]) and house load (obs[2])
        max_flow = self.env.max_battery_flow
        battery_level = obs[-2]/self.env.battery_capacity  # normalize battery level to [0, 1] by dividing battery energy (obs[-2]) by capacity
        noise = np.random.normal(-0.01, 0.01)  # add small noise with standard deviation of 0.01
        # this is to check
        if self.rule_presistence and battery_level < 0.9:  # battery is not full
            #continue charging.
            return [min((0.5 + noise, 0.5))]
        elif self.rule_presistence and battery_level > 0.9:  # battery is not empty
            self.rule_presistence = False
            return [max((-0.1 + noise, -0.1))]
        elif battery_level < 0.1:  # battery is empty
            self.rule_presistence = True
            return [min((0.5 + noise, 0.5))]
        
        elif diff > 0 and battery_level < 0.9:  # surplus energy
            # Compute the recommended charging power as a fraction of the maximum battery flow;
            # ensure it does not exceed 1.0 (100% of max battery flow)
            action_value = min((diff / max_flow)+noise, 1.0)
            return [action_value]
        elif diff < 0 and battery_level > 0.1:  # deficit energy
            # Compute the recommended discharging power as a fraction of the maximum battery flow;
            # ensure it does not exceed 1.0 (100% of max battery flow)
            action_value = max((diff / max_flow)+noise, -1.0)
            return [action_value]
        else:
            # No action needed; add noise to zero action.
            return [0.0 + noise]

    def sdp_action(self, obs):
        """
        Computes the optimal action using the precomputed SDP policy.
        Observation must contain the current battery level.
        """
        battery_level = obs[-2]
        soc_states = np.linspace(0, self.env.battery_capacity, self.soc_resolution)
        closest_state_idx = np.argmin(np.abs(soc_states - battery_level))
        return [self.policy[0, closest_state_idx]]

    def _sdp_optimization(self):
        """
        Computes the value function and policy using SDP optimization.
        """
        soc_states = np.linspace(0, self.env.battery_capacity, self.soc_resolution)
        value_function = np.zeros((self.horizon, len(soc_states)))
        policy = np.zeros((self.horizon, len(soc_states)))

        for t in reversed(range(self.horizon)):
            for i, soc in enumerate(soc_states):
                costs = []
                actions = []
                for action in np.linspace(-self.env.max_battery_flow, self.env.max_battery_flow, 20):
                    battery_flow_energy = action * self.env.step_duration
                    if action < 0 and abs(battery_flow_energy) > soc:
                        continue
                    if action > 0 and soc + battery_flow_energy > self.env.battery_capacity:
                        continue

                    soc_next = soc + battery_flow_energy
                    soc_next = np.clip(soc_next, 0, self.env.battery_capacity)
                    next_value = 0 if t == self.horizon - 1 else np.interp(soc_next, soc_states, value_function[t + 1])

                    avg_soc = (soc - 0.5 * (-battery_flow_energy)) / self.env.battery_capacity * 100
                    degradation_cost = self.env._calculate_battery_degradation(battery_flow_energy, avg_soc) * self.env.battery_life_cost
                    grid_energy = battery_flow_energy + self.env.df['HouseLoad'][t] - self.env.df['SolarGen'][t]
                    energy_price = self.env.df['ImportEnergyPrice'][t] if battery_flow_energy >= 0 else self.env.df['ExportEnergyPrice'][t]
                    grid_reward, _ = self.env._calculate_grid_reward(grid_energy, energy_price)
                    total_cost = -grid_reward + degradation_cost + next_value

                    costs.append(total_cost)
                    actions.append(action)

                if costs:
                    best_action_idx = np.argmin(costs)
                    value_function[t, i] = costs[best_action_idx]
                    policy[t, i] = actions[best_action_idx]

        return value_function, policy

    def run_episode(self, render=False):
        obs, _ = self.env.reset()
        logs = []
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = self.choose_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            logs.append({
                'observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
                'action': action,
                'reward': reward,
                'info': info
            })
            obs = next_obs
            if render:
                self.env.render()

        return pl.DataFrame(logs)
