import numpy as np
import torch
import polars as pl
from EnergySimEnv import SolarBatteryEnv

from batterydeg import static_degradation

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

        if self.algorithm == 'sdp':
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

            # debugging log
            self.sdp_debug_log = []

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
            return [action_value]
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
        """Implements the backward induction algorithm."""
        num_soc_levels = len(self.soc_levels_kwh)
        num_action_levels = len(self.action_levels_norm)
        horizon = len(forecasts)

        # Initialize cost-to-go (J) and policy tables
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int) # Store action indices, -1 for invalid/unreachable

        # Terminal cost is zero
        cost_to_go[horizon, :] = 0.0

        # Precompute battery flow energies for all actions
        battery_flow_energies = self.action_levels_norm * self.max_battery_flow * self.step_duration


        # Backward iteration
        for t in range(horizon - 1, -1, -1):
            forecast_step = forecasts[t]
            for soc_idx, soc_kwh in enumerate(self.soc_levels_kwh):
                min_total_cost = np.inf
                best_action_idx = -1

                # Vectorized feasibility check
                potential_next_socs = soc_kwh + battery_flow_energies
                feasible_mask = (potential_next_socs >= -1e-6) & (potential_next_socs <= self.battery_capacity + 1e-6)

                # Iterate over all feasible actions for this SoC
                for action_idx in np.where(feasible_mask)[0]:
                    battery_flow_energy = battery_flow_energies[action_idx]
                    # Clip battery flow to ensure SoC stays within bounds
                    actual_battery_flow_energy = np.clip(
                        battery_flow_energy, -soc_kwh, self.battery_capacity - soc_kwh
                    )
                    # Compute the battery flow rate (normalized action * max flow)
                    battery_flow_rate = self.action_levels_norm[action_idx] * self.max_battery_flow

                    # Compute next SoC after applying the action
                    next_soc_kwh = soc_kwh + actual_battery_flow_energy
                    next_soc_idx = self._soc_to_idx(next_soc_kwh)

                    # Calculate immediate (stage) cost for this action
                    stage_cost = self._calculate_sdp_stage_cost(
                        t, soc_kwh, battery_flow_rate, actual_battery_flow_energy, forecast_step
                    )
                    if stage_cost == np.inf:
                        continue  # Skip infeasible actions

                    # Get cost-to-go from the next state
                    future_cost = cost_to_go[t + 1, next_soc_idx]
                    if future_cost == np.inf:
                        continue  # Skip if next state is unreachable

                    # Total cost for this action (stage cost + future cost)
                    total_cost = stage_cost + future_cost

                    # Debug logging for the current actual SoC at t=0
                    current_actual_soc_idx = self._soc_to_idx(self.env.battery_level)
                    if t == 0 and soc_idx == current_actual_soc_idx:
                        self.sdp_debug_log.append({
                            't': t,
                            'soc_idx': soc_idx,
                            'action_idx': action_idx,
                            'action_norm': self.action_levels_norm[action_idx],
                            'stage_cost': stage_cost,
                            'future_cost': future_cost,
                            'total_cost': total_cost
                        })

                    # Update minimum cost and best action if this action is better
                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        best_action_idx = action_idx

                # Store the best action and its cost for this SoC at this time step
                if best_action_idx != -1:
                    cost_to_go[t, soc_idx] = min_total_cost
                    policy_table[t, soc_idx] = best_action_idx

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

        grid_cost = 0
        # Check grid limits (using energy directly)
        if abs(grid_energy) > self.max_grid_energy + 1e-6: # Add tolerance
            grid_cost = np.inf # Penalize grid violation heavily
        else:
            if grid_energy > 0: # Importing
                grid_cost = grid_energy * import_price
            else: # Exporting (grid_energy is negative)
                grid_cost = grid_energy * export_price # Export price might be lower

        if grid_cost == np.inf:
            return np.inf # Return early if grid violated

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
