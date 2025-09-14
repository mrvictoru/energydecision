"""import numpy as np
from typing import Callable, Optional, Tuple


class DynamicProgram:
    """A drop-in compatible Dynamic Program implementation that wraps the Agent's
    SDP logic into an object with a clear API.

    This class is designed to be constructed from an existing Agent instance via
    DynamicProgram.from_agent(...) and to return the same policy_table shape and
    semantics as the existing Agent._solve_sdp(...).

    The implementation intentionally re-uses the Agent's helper methods
    (e.g. _calculate_sdp_stage_cost, attributes like soc_levels_kwh, action_levels_norm)
    so it can be dropped in without changing the Agent internals.
    """

    def __init__(self, agent, forecasts: list, start_index: int = 0):
        # Keep a reference to the agent for cost/transition helpers and params
        self.agent = agent
        self.forecasts = forecasts
        self.start_index = int(start_index)

        # Horizon and grids
        self.horizon = len(forecasts)
        self.soc_levels_kwh = np.array(agent.soc_levels_kwh, dtype=float)
        self.action_levels_norm = np.array(agent.action_levels_norm, dtype=float)

        # Precompute action energies (kWh) for quick use
        # The agent stores action levels as normalized [-1,1], where positive is charge
        # Convert normalized level to energy over step_duration using max_battery_flow.
        self.step_duration = float(agent.step_duration)
        self.max_battery_flow = float(agent.max_battery_flow)
        self.battery_capacity = float(agent.battery_capacity)

        # energy change per normalized action level (kWh)
        self.battery_flow_energies = (self.action_levels_norm * self.max_battery_flow) * self.step_duration

        # Scenario cache reference (agent may have prepared this)
        self._scenario_cache = getattr(agent, '_scenario_cache', None)

        # Monte Carlo settings (used the same way as agent._solve_sdp)
        self.use_monte_carlo = getattr(agent, 'use_monte_carlo', False)
        self.mc_samples = int(getattr(agent, 'mc_samples', 100))
        self.mc_seed = getattr(agent, 'mc_seed', None)

        # Degradation settings forwarded
        self.degradation_model = getattr(agent, 'degradation_model', 'linear')
        self.linear_deg_cost_per_kwh = getattr(agent, 'linear_deg_cost_per_kwh', None)
        self.static_deg_correction_factor = getattr(agent, 'static_deg_correction_factor', 0.01)

        # Internal storage for solution
        self.cost_to_go = np.full((self.horizon + 1, len(self.soc_levels_kwh)), np.inf, dtype=float)
        self.policy_table = np.full((self.horizon, len(self.soc_levels_kwh)), -1, dtype=int)

    @classmethod
    def from_agent(cls, agent, forecasts: list, start_index: int = 0):
        """Convenience constructor that mirrors how the Agent would call the solver."""
        return cls(agent, forecasts, start_index)

    def set_final_ctg(self, ctg_vector: Optional[np.ndarray]):
        """Set the terminal cost-to-go vector for time T (must match soc grid length).

        If ctg_vector is None, defaults to zeros.
        """
        if ctg_vector is None:
            self.cost_to_go[self.horizon, :] = 0.0
        else:
            ctg = np.array(ctg_vector, dtype=float)
            if ctg.shape[0] != len(self.soc_levels_kwh):
                raise ValueError("Final CTG length must match number of SoC grid points")
            self.cost_to_go[self.horizon, :] = ctg

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Perform backward induction and return (policy_table, cost_to_go matrix).

        This implementation mirrors the vectorized structure of Agent._solve_sdp
        but is encapsulated in this class so it can be reused or tested separately.
        """
        horizon = self.horizon
        socs = self.soc_levels_kwh
        battery_energies = np.array(self.battery_flow_energies, dtype=float)

        # For safety if terminal not provided
        if not np.isfinite(self.cost_to_go[horizon, :]).all():
            self.cost_to_go[horizon, :] = 0.0

        # Backward induction
        for t in range(horizon - 1, -1, -1):
            forecast_step = self.forecasts[t]
            row_idx = self.start_index + t

            # Prepare Monte Carlo samples for this timestep if available
            monte_samples = None
            if self.use_monte_carlo and self._scenario_cache is not None:
                try:
                    vals_solar_arr, ps_solar_arr = self._scenario_cache['solar']
                    vals_load_arr, ps_load_arr = self._scenario_cache['load']
                    vals_imp_arr, ps_imp_arr = self._scenario_cache['import_price']
                    vals_exp_arr, ps_exp_arr = self._scenario_cache['export_price']

                    if 0 <= row_idx < vals_solar_arr.shape[0]:
                        vals_solar_t = vals_solar_arr[row_idx, :]
                        ps_solar_t = ps_solar_arr[row_idx, :]
                        vals_load_t = vals_load_arr[row_idx, :]
                        ps_load_t = ps_load_arr[row_idx, :]
                        vals_imp_t = vals_imp_arr[row_idx, :]
                        ps_imp_t = ps_imp_arr[row_idx, :]
                        vals_exp_t = vals_exp_arr[row_idx, :]
                        ps_exp_t = ps_exp_arr[row_idx, :]

                        rng_seed_t = None if self.mc_seed is None else (self.mc_seed + t)
                        rng = np.random.default_rng(rng_seed_t)

                        mc_s = rng.choice(len(vals_solar_t), size=self.mc_samples, p=ps_solar_t)
                        mc_l = rng.choice(len(vals_load_t), size=self.mc_samples, p=ps_load_t)
                        mc_i = rng.choice(len(vals_imp_t), size=self.mc_samples, p=ps_imp_t)
                        mc_e = rng.choice(len(vals_exp_t), size=self.mc_samples, p=ps_exp_t)

                        sampled_solar = vals_solar_t[mc_s]
                        sampled_load = vals_load_t[mc_l]
                        sampled_imp = vals_imp_t[mc_i]
                        sampled_exp = vals_exp_t[mc_e]

                        monte_samples = (sampled_solar, sampled_load, sampled_imp, sampled_exp)
                except Exception:
                    monte_samples = None

            # Vectorized per-state/action feasibility and next SoC
            socs_reshaped = socs[:, np.newaxis]  # (num_soc, 1)
            battery_energies_reshaped = battery_energies[np.newaxis, :]  # (1, num_actions)
            potential_next_socs = socs_reshaped + battery_energies_reshaped  # (num_soc, num_actions)

            feasible_mask = ((potential_next_socs >= -1e-6) & (potential_next_socs <= self.battery_capacity + 1e-6))

            clipped_battery_energies = np.clip(
                battery_energies_reshaped,
                -socs_reshaped,
                self.battery_capacity - socs_reshaped
            )

            next_socs = socs_reshaped + clipped_battery_energies

            # Interpolate future cost-to-go from next layer
            future_costs = np.interp(next_socs.ravel(), self.soc_levels_kwh, self.cost_to_go[t + 1, :]).reshape(next_socs.shape)

            # Build unique energy list to compute stage costs efficiently
            rounded = np.round(clipped_battery_energies, 10)
            rounded_flat = rounded.ravel()
            feasible_flat = feasible_mask.ravel()

            stage_costs = np.full(rounded.shape, np.inf)

            if feasible_flat.any():
                values_flat = rounded_flat[feasible_flat]
                unique_vals, inverse = np.unique(values_flat, return_inverse=True)

                unique_costs = np.empty(unique_vals.shape, dtype=float)
                for ui, energy in enumerate(unique_vals):
                    battery_rate = energy / self.step_duration
                    if monte_samples is not None:
                        sampled_solar, sampled_load, sampled_imp, sampled_exp = monte_samples
                        battery_charge_energy = max(0.0, energy)
                        battery_discharge_energy = max(0.0, -energy)

                        grid_energy = sampled_load + battery_charge_energy - sampled_solar - battery_discharge_energy
                        if np.any(np.abs(grid_energy) > (self.agent.max_grid_energy + 1e-6)):
                            stage_cost = np.inf
                        else:
                            is_import = grid_energy > 0
                            costs = np.where(is_import, grid_energy * sampled_imp, -np.abs(grid_energy) * sampled_exp)
                            if np.any(np.isinf(costs)):
                                stage_cost = np.inf
                            else:
                                stage_cost = float(np.mean(costs))
                    else:
                        # Delegate to agent method to compute deterministic expected cost / degradation
                        stage_cost = self.agent._calculate_sdp_stage_cost(row_idx, self.battery_capacity / 2.0, battery_rate, energy, forecast_step)

                    # Degradation handling if agent uses linear model and monte samples used elsewhere is not present
                    if self.degradation_model == 'linear' and monte_samples is None:
                        degradation_cost = self.linear_deg_cost_per_kwh * abs(energy) if self.linear_deg_cost_per_kwh is not None else 0.0
                    else:
                        # Use agent's degradation calculation path if necessary
                        degradation_cost = 0.0

                    unique_costs[ui] = stage_cost + degradation_cost

                costs_flat = np.full(rounded_flat.shape, np.inf)
                costs_flat[feasible_flat] = unique_costs[inverse]
                stage_costs = costs_flat.reshape(rounded.shape)

            total_costs = stage_costs + future_costs
            total_costs_masked = np.where(feasible_mask & np.isfinite(stage_costs) & np.isfinite(future_costs), total_costs, np.inf)

            row_min = np.min(total_costs_masked, axis=1)
            best_actions = np.argmin(total_costs_masked, axis=1)

            finite_mask = np.isfinite(row_min)
            self.cost_to_go[t, :] = row_min
            self.policy_table[t, :] = -1
            self.policy_table[t, finite_mask] = best_actions[finite_mask]

        return self.policy_table, self.cost_to_go


class DPSolver:
    """Orchestrator that can solve a long horizon by splitting into sub-horizons
    and passing interpolated terminal cost-to-go vectors between sub-problems.

    It uses DynamicProgram internally and exposes a compatible solve(...) method
    similar to the Agent._solve_sdp signature.
    """

    def __init__(self, agent, chunk_length: Optional[int] = None):
        self.agent = agent
        self.chunk_length = chunk_length

    def _partition_indices(self, horizon: int) -> list:
        if self.chunk_length is None or self.chunk_length >= horizon:
            return [(0, horizon)]
        parts = []
        t = 0
        while t < horizon:
            end = min(horizon, t + self.chunk_length)
            parts.append((t, end))
            t = end
        return parts

    def solve(self, forecasts: list, start_index: int = 0) -> np.ndarray:
        """Solve the full horizon by chunking and return a policy_table shaped (horizon, n_soc).

        The returned format matches the Agent._solve_sdp return value.
        """
        horizon = len(forecasts)
        soc_len = len(self.agent.soc_levels_kwh)
        full_policy = np.full((horizon, soc_len), -1, dtype=int)

        chunks = self._partition_indices(horizon)

        # Solve from last chunk to first, carrying CTG backwards
        next_ctg = None
        # iterate reversed with absolute indices
        for (chunk_start, chunk_end) in reversed(chunks):
            sub_forecasts = forecasts[chunk_start:chunk_end]
            dp = DynamicProgram.from_agent(self.agent, sub_forecasts, start_index + chunk_start)
            # set terminal CTG from next_ctg (interpolated) or zeros for final chunk
            if next_ctg is None:
                dp.set_final_ctg(None)
            else:
                # interpolate next_ctg (which lives on agent.soc_levels_kwh) onto dp.soc_levels_kwh
                interp_ctg = np.interp(dp.soc_levels_kwh, self.agent.soc_levels_kwh, next_ctg)
                dp.set_final_ctg(interp_ctg)

            policy_sub, ctg_sub = dp.solve()
            full_policy[chunk_start:chunk_end, :] = policy_sub

            # next_ctg becomes first-stage ctg of this solved chunk (time index 0 of ctg_sub)
            next_ctg = ctg_sub[0, :]

        return full_policy
"""