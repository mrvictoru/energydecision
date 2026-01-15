"""
Stochastic Dynamic Programming (SDP) Algorithm

This module implements a self-contained SDP solver for battery energy management.
All algorithm logic is in one place for easy understanding and debugging.

Algorithm Overview:
-------------------
1. Initialize: Set up state space (SoC levels), action space (battery flows), and terminal costs
2. Backward Induction: For each time step t from horizon-1 down to 0:
   a. Get forecast data (solar, load, prices)
   b. Generate uncertainty scenarios (optional Monte Carlo)
   c. For each state-action pair:
      - Compute feasibility (battery/grid constraints)
      - Compute stage cost (grid cost + degradation cost)
      - Compute future cost (interpolated from next time step)
      - Find optimal action that minimizes total cost
3. Extract Policy: Return optimal action for each state at each time step

Reference: Based on khalida/optimal-energy-storage dynamic programming approach
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from algorithm_helpers import DegradationCalculator, compute_grid_cost


class SDPSolver:
    """
    Self-contained Stochastic Dynamic Programming solver for battery control.
    
    This class encapsulates all SDP logic in one place, making it easy to understand
    the complete algorithm flow without jumping between multiple files.
    """
    
    def __init__(self, 
                 env: Any,
                 horizon: int,
                 soc_resolution: int,
                 action_resolution: int,
                 use_monte_carlo: bool = False,
                 mc_samples: int = 100,
                 mc_seed: Optional[int] = None,
                 scenario_generator: Optional[Any] = None):
        """
        Initialize SDP solver.
        
        Notes:
            - This solver uses **rainflow-based** degradation exclusively. The
              `DegradationCalculator` provides the `compute_rainflow_degradation`
              method which is used to estimate per-step degradation based on SoC
              transitions.

        Args:
            env: Environment with battery and grid parameters
            horizon: Number of time steps to optimize over
            soc_resolution: Number of discrete SoC levels
            action_resolution: Number of discrete action levels
            use_monte_carlo: Whether to use Monte Carlo for uncertainty
            mc_samples: Number of Monte Carlo samples
            mc_seed: Random seed for reproducibility
            scenario_generator: Optional scenario generator for uncertainty
        """
        # Environment parameters
        self.env = env
        self.battery_capacity = env.battery_capacity
        self.max_battery_flow = env.max_battery_flow
        self.step_duration = env.step_duration
        self.max_grid_energy = env.max_grid_energy
        self.battery_life_cost = env.battery_life_cost
        
        # Algorithm parameters
        self.horizon = horizon
        self.soc_resolution = soc_resolution
        self.action_resolution = action_resolution
        
        # Degradation configuration (rainflow-only)
        self.degradation_calc = DegradationCalculator(
            battery_capacity=self.battery_capacity,
            step_duration=self.step_duration,
            battery_life_cost=self.battery_life_cost,
            degradation_temperature=getattr(env, 'degradation_temperature', 25.0)
        )
        
        # Uncertainty handling
        self.use_monte_carlo = use_monte_carlo
        self.mc_samples = mc_samples
        self.mc_seed = mc_seed
        self.scenario_generator = scenario_generator
        self._scenario_cache = None
        
        # Discretization
        self.soc_levels_kwh = np.linspace(0, self.battery_capacity, self.soc_resolution)
        self.action_levels_norm = np.linspace(-1.0, 1.0, self.action_resolution)
        self.battery_flow_energies = self.action_levels_norm * self.max_battery_flow * self.step_duration
    
    def solve(self, forecasts: List[Dict], start_index: int = 0) -> np.ndarray:
        """
        Solve SDP problem using backward induction.
        
        Args:
            forecasts: List of forecast dicts with keys: SolarGen, HouseLoad, 
                      ImportEnergyPrice, ExportEnergyPrice
            start_index: Global time index for first forecast
        
        Returns:
            policy_table: Array of shape (horizon, num_soc_levels) where
                         policy_table[t, s] = optimal action index at time t, state s
        """
        num_soc_levels = len(self.soc_levels_kwh)
        horizon = len(forecasts)
        
        # STEP 1: Initialize cost-to-go and policy tables
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        cost_to_go[horizon, :] = 0.0  # Terminal cost is zero
        
        # STEP 2: Prepare scenario cache for uncertainty (if using Monte Carlo)
        if self._scenario_cache is None and self.scenario_generator is not None:
            try:
                self._scenario_cache = self.scenario_generator.generate_time_step_scenarios(self.env.df)
            except Exception:
                self._scenario_cache = None
        
        # STEP 3: Backward induction - solve from last time step to first
        for t in range(horizon - 1, -1, -1):
            forecast_step = forecasts[t]
            row_idx = start_index + t
            
            # STEP 3a: Prepare Monte Carlo samples for this time step (if enabled)
            monte_samples = self._prepare_monte_carlo_samples(row_idx, t)
            
            # STEP 3b: Compute stage costs and optimal actions for all states
            stage_costs = self._compute_stage_costs(
                forecast_step, row_idx, monte_samples
            )
            
            # STEP 3c: Compute future costs by interpolating next time step's cost-to-go
            future_costs = self._compute_future_costs(cost_to_go[t + 1, :])
            
            # STEP 3d: Find optimal action for each state
            self._update_policy(
                t, stage_costs, future_costs, cost_to_go, policy_table
            )
        
        return policy_table
    
    def _prepare_monte_carlo_samples(self, row_idx: int, t: int) -> Optional[Tuple]:
        """
        Prepare Monte Carlo samples for uncertainty modeling.
        
        Returns:
            Tuple of (sampled_solar, sampled_load, sampled_imp, sampled_exp) or None
        """
        if not self.use_monte_carlo or self._scenario_cache is None:
            return None
        
        try:
            vals_solar_arr, ps_solar_arr = self._scenario_cache['solar']
            vals_load_arr, ps_load_arr = self._scenario_cache['load']
            vals_imp_arr, ps_imp_arr = self._scenario_cache['import_price']
            vals_exp_arr, ps_exp_arr = self._scenario_cache['export_price']
            
            if row_idx < 0 or row_idx >= vals_solar_arr.shape[0]:
                return None
            
            # Extract scenario distributions for this time step
            vals_solar_t = vals_solar_arr[row_idx, :]
            ps_solar_t = ps_solar_arr[row_idx, :]
            vals_load_t = vals_load_arr[row_idx, :]
            ps_load_t = ps_load_arr[row_idx, :]
            vals_imp_t = vals_imp_arr[row_idx, :]
            ps_imp_t = ps_imp_arr[row_idx, :]
            vals_exp_t = vals_exp_arr[row_idx, :]
            ps_exp_t = ps_exp_arr[row_idx, :]
            
            # Sample from distributions
            rng_seed_t = None if self.mc_seed is None else (self.mc_seed + t)
            rng = np.random.default_rng(rng_seed_t)
            
            idx_s = rng.choice(len(vals_solar_t), size=self.mc_samples, p=ps_solar_t)
            idx_l = rng.choice(len(vals_load_t), size=self.mc_samples, p=ps_load_t)
            idx_i = rng.choice(len(vals_imp_t), size=self.mc_samples, p=ps_imp_t)
            idx_e = rng.choice(len(vals_exp_t), size=self.mc_samples, p=ps_exp_t)
            
            sampled_solar = vals_solar_t[idx_s]
            sampled_load = vals_load_t[idx_l]
            sampled_imp = vals_imp_t[idx_i]
            sampled_exp = vals_exp_t[idx_e]
            
            return (sampled_solar, sampled_load, sampled_imp, sampled_exp)
        except Exception:
            return None
    
    def _compute_stage_costs(self, forecast_step: Dict, row_idx: int, 
                            monte_samples: Optional[Tuple]) -> np.ndarray:
        """
        Compute stage cost for each (state, action) pair.
        
        Stage cost = Grid cost + Degradation cost
        
        Returns:
            Array of shape (num_soc_levels, num_actions) with stage costs
        """
        num_soc_levels = len(self.soc_levels_kwh)
        num_actions = len(self.action_levels_norm)
        socs = self.soc_levels_kwh
        
        # Compute feasibility and clipped battery energies
        socs_reshaped = socs[:, np.newaxis]  # (num_soc, 1)
        battery_energies_reshaped = self.battery_flow_energies[np.newaxis, :]  # (1, num_actions)
        
        # Clipped battery flow energies respect battery capacity constraints
        clipped_battery_energies = np.clip(
            battery_energies_reshaped,
            -socs_reshaped,  # Can't discharge more than current SoC
            self.battery_capacity - socs_reshaped  # Can't charge beyond capacity
        )
        
        # Feasibility mask
        potential_next_socs = socs_reshaped + battery_energies_reshaped
        feasible_mask = ((potential_next_socs >= -1e-6) & 
                        (potential_next_socs <= self.battery_capacity + 1e-6))
        
        # Compute costs only for unique energy values (optimization)
        rounded = np.round(clipped_battery_energies, 10)
        rounded_flat = rounded.ravel()
        feasible_flat = feasible_mask.ravel()
        
        stage_costs = np.full(rounded.shape, np.inf)
        
        if feasible_flat.any():
            values_flat = rounded_flat[feasible_flat]
            unique_vals, inverse = np.unique(values_flat, return_inverse=True)
            
            # Compute cost for each unique energy value
            unique_costs = np.empty(unique_vals.shape, dtype=float)
            for ui, energy in enumerate(unique_vals):
                unique_costs[ui] = self._compute_single_stage_cost(
                    energy, forecast_step, monte_samples
                )
            
            # Map unique costs back to full array
            costs_flat = np.full(rounded_flat.shape, np.inf)
            costs_flat[feasible_flat] = unique_costs[inverse]
            stage_costs = costs_flat.reshape(rounded.shape)
        
        return stage_costs
    
    def _compute_single_stage_cost(self, energy: float, forecast_step: Dict,
                                   monte_samples: Optional[Tuple]) -> float:
        """
        Compute stage cost for a single energy value.
        
        Cost = Grid cost + Degradation cost
        """
        battery_rate = energy / self.step_duration
        
        # Compute grid cost (with or without Monte Carlo)
        if monte_samples is not None:
            grid_cost = self._compute_grid_cost_monte_carlo(energy, monte_samples)
        else:
            grid_cost = self._compute_grid_cost_deterministic(energy, forecast_step)
        
        if grid_cost == np.inf:
            return np.inf
        
        # Compute degradation cost
        degradation_cost = self._compute_degradation_cost(energy, battery_rate)
        
        return grid_cost + degradation_cost
    
    def _compute_grid_cost_monte_carlo(self, energy: float, 
                                      monte_samples: Tuple) -> float:
        """Compute expected grid cost using Monte Carlo samples."""
        sampled_solar, sampled_load, sampled_imp, sampled_exp = monte_samples
        
        battery_charge_energy = max(0.0, energy)
        battery_discharge_energy = max(0.0, -energy)
        
        grid_energy = sampled_load + battery_charge_energy - sampled_solar - battery_discharge_energy
        
        # Check grid limits
        if np.any(np.abs(grid_energy) > (self.max_grid_energy + 1e-6)):
            return np.inf
        
        # Compute costs for all samples
        is_import = grid_energy > 0
        costs = np.where(is_import, 
                        grid_energy * sampled_imp, 
                        -np.abs(grid_energy) * sampled_exp)
        
        if np.any(np.isinf(costs)):
            return np.inf
        
        return float(np.mean(costs))
    
    def _compute_grid_cost_deterministic(self, energy: float, 
                                        forecast_step: Dict) -> float:
        """Compute grid cost using deterministic forecast."""
        battery_charge_energy = max(0.0, energy)
        battery_discharge_energy = max(0.0, -energy)
        
        solar = forecast_step.get('SolarGen', 0.0)
        load = forecast_step.get('HouseLoad', 0.0)
        import_price = forecast_step.get('ImportEnergyPrice', 0.0)
        export_price = forecast_step.get('ExportEnergyPrice', 0.0)
        
        grid_energy = load + battery_charge_energy - solar - battery_discharge_energy
        
        return compute_grid_cost(grid_energy, import_price, export_price, self.max_grid_energy)
    
    def _compute_degradation_cost(self, energy: float, battery_rate: float) -> float:
        """Compute battery degradation cost for energy throughput using rainflow counting.
        
        A representative SoC is used for the per-step estimation (midpoint of capacity)
        because `_compute_single_stage_cost` is evaluated on unique energy values only
        and does not have direct access to the current state's SoC. This keeps the
        computation efficient while using the rainflow-based estimator for wear.
        """
        if abs(energy) <= 0.0:
            return 0.0

        # Representative SoC (kWh) at midpoint of battery
        rep_soc = self.battery_capacity / 2.0
        soc_next = rep_soc + energy

        # Compute rainflow-based degradation fraction for this SoC transition
        deg_frac = self.degradation_calc.compute_rainflow_degradation(rep_soc, soc_next)
        return float(deg_frac * self.battery_life_cost)

    
    def _compute_future_costs(self, next_cost_to_go: np.ndarray) -> np.ndarray:
        """
        Compute future costs by interpolating next time step's cost-to-go.
        
        Returns:
            Array of shape (num_soc_levels, num_actions) with future costs
        """
        socs = self.soc_levels_kwh[:, np.newaxis]
        battery_energies = self.battery_flow_energies[np.newaxis, :]
        
        # Clipped energies
        clipped_energies = np.clip(
            battery_energies,
            -socs,
            self.battery_capacity - socs
        )
        
        # Next SoCs
        next_socs = socs + clipped_energies
        
        # Interpolate cost-to-go
        future_costs = np.interp(
            next_socs.ravel(), 
            self.soc_levels_kwh, 
            next_cost_to_go
        ).reshape(next_socs.shape)
        
        return future_costs
    
    def _update_policy(self, t: int, stage_costs: np.ndarray, 
                      future_costs: np.ndarray, cost_to_go: np.ndarray,
                      policy_table: np.ndarray):
        """
        Update policy and cost-to-go for time step t.
        
        For each state, find the action that minimizes total cost = stage cost + future cost
        """
        # Total cost = stage cost + future cost
        total_costs = stage_costs + future_costs
        
        # Mask invalid entries
        total_costs_masked = np.where(
            np.isfinite(stage_costs) & np.isfinite(future_costs),
            total_costs,
            np.inf
        )
        
        # Find best action for each state
        row_min = np.min(total_costs_masked, axis=1)
        best_actions = np.argmin(total_costs_masked, axis=1)
        
        # Update cost-to-go and policy
        finite_mask = np.isfinite(row_min)
        cost_to_go[t, :] = row_min
        policy_table[t, :] = -1
        policy_table[t, finite_mask] = best_actions[finite_mask]
