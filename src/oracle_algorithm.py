"""
Oracle Algorithm for Battery Control

This module implements a self-contained Oracle solver that uses perfect future information
to compute the optimal policy. This serves as an upper bound for comparison with other algorithms.

Algorithm Overview:
-------------------
1. Oracle has perfect knowledge of future solar generation, load, and prices
2. Uses dynamic programming (similar to SDP) but without uncertainty
3. Computes truly optimal policy given perfect information
4. Used as benchmark to evaluate performance of SDP/MRDP/RL algorithms

Key Difference from SDP:
- SDP: Uses forecasts/scenarios to handle uncertainty
- Oracle: Uses actual future values (no uncertainty)

Reference: Based on khalida/optimal-energy-storage oracle implementation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from algorithm_helpers import DegradationCalculator, compute_grid_cost


class OracleSolver:
    """
    Self-contained Oracle solver for battery control with perfect future information.
    
    This class implements dynamic programming with actual future values,
    providing an upper bound on achievable performance.
    """
    
    def __init__(self,
                 env: Any,
                 horizon: int,
                 action_resolution: int,
                 degradation_model: str = 'linear',
                 linear_deg_cost_p_kwh: Optional[float] = None):
        """
        Initialize Oracle solver.
        
        Args:
            env: Environment with battery parameters and data
            horizon: Number of steps to look ahead
            action_resolution: Number of discrete action levels
            degradation_model: 'linear', 'rainflow', or default (linearized)
            linear_deg_cost_p_kwh: Cost per kWh for linear model
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
        self.action_resolution = action_resolution
        
        # Degradation configuration
        self.degradation_model = degradation_model
        self.degradation_calc = DegradationCalculator(
            battery_capacity=self.battery_capacity,
            step_duration=self.step_duration,
            battery_life_cost=self.battery_life_cost,
            degradation_temperature=getattr(env, 'degradation_temperature', 25.0)
        )
        
        if self.degradation_model == 'linear':
            if linear_deg_cost_p_kwh is not None:
                self.linear_deg_cost_per_kwh = linear_deg_cost_p_kwh
            else:
                cycle_life = 3650
                self.linear_deg_cost_per_kwh = env.battery_life_cost / (env.battery_capacity * cycle_life)
        
        # Discretization - use same SoC levels as SDP for fair comparison
        self.soc_resolution = 20  # Standard resolution
        self.soc_levels_kwh = np.linspace(0, self.battery_capacity, self.soc_resolution)
        self.action_levels = np.linspace(-1.0, 1.0, self.action_resolution, dtype=np.float32)
        self.battery_energies = self.action_levels * self.max_battery_flow * self.step_duration
    
    def solve(self, start_index: int, horizon: int) -> Optional[np.ndarray]:
        """
        Solve Oracle DP problem using actual future values.
        
        Algorithm:
        1. Initialize cost-to-go with zero terminal cost
        2. For each time step (backward):
           a. Get actual future values from environment
           b. Compute stage cost (grid + degradation)
           c. Compute total cost (stage + future)
           d. Find optimal action
        3. Return policy table
        
        Args:
            start_index: Starting time index in environment data
            horizon: Number of steps to optimize
        
        Returns:
            policy_table: Array of shape (horizon, num_soc_levels) or None if infeasible
        """
        num_soc_levels = len(self.soc_levels_kwh)
        num_actions = len(self.action_levels)
        
        # STEP 1: Initialize cost-to-go and policy tables
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        cost_to_go[horizon, :] = 0.0  # Terminal cost is zero
        
        # Precompute for vectorization
        socs = self.soc_levels_kwh[:, np.newaxis]  # (num_soc, 1)
        battery_energies = self.battery_energies  # (num_actions,)
        
        # STEP 2: Backward induction with actual future values
        for t in range(horizon - 1, -1, -1):
            # STEP 2a: Get actual future values (this is what makes it "oracle")
            row = self.env._get_row(start_index + t)
            solar = row['SolarGen']
            load = row['HouseLoad']
            import_price = row['ImportEnergyPrice']
            export_price = row['ExportEnergyPrice']
            
            # STEP 2b: Compute feasible battery energies
            battery_reshaped = battery_energies[np.newaxis, :]  # (1, num_actions)
            clipped_energies = np.clip(
                battery_reshaped,
                -socs,  # Can't discharge more than current SoC
                self.battery_capacity - socs  # Can't charge beyond capacity
            )
            
            # STEP 2c: Compute grid energy and check violations
            battery_charge = np.maximum(clipped_energies, 0.0)
            battery_discharge = np.maximum(-clipped_energies, 0.0)
            
            demand = load + battery_charge
            supply = solar + battery_discharge
            grid_energy = demand - supply
            
            # Grid constraint violations
            grid_violation = np.abs(grid_energy) > (self.max_grid_energy + 1e-6)
            
            # STEP 2d: Compute grid cost
            grid_cost = np.where(
                grid_energy >= 0,
                grid_energy * import_price,  # Import cost
                -np.abs(grid_energy) * export_price  # Export revenue (negative cost)
            )
            grid_cost[grid_violation] = np.inf
            
            # STEP 2e: Compute degradation cost for each (soc, action) pair
            stage_costs = self._compute_stage_costs(
                socs, clipped_energies, grid_cost, grid_violation,
                num_soc_levels, num_actions
            )
            
            # STEP 2f: Compute future costs
            next_socs = socs + clipped_energies
            next_costs = np.interp(
                next_socs.ravel(),
                self.soc_levels_kwh,
                cost_to_go[t + 1, :]
            ).reshape(next_socs.shape)
            
            # STEP 2g: Find optimal actions
            total_costs = stage_costs + next_costs
            feasible_mask = np.isfinite(total_costs)
            
            row_min = np.min(np.where(feasible_mask, total_costs, np.inf), axis=1)
            best_actions = np.argmin(np.where(feasible_mask, total_costs, np.inf), axis=1)
            
            # Update cost-to-go and policy
            finite_mask = np.isfinite(row_min)
            cost_to_go[t, :] = np.where(finite_mask, row_min, np.inf)
            policy_table[t, :] = -1
            policy_table[t, finite_mask] = best_actions[finite_mask]
        
        # STEP 3: Check if solution is feasible
        if not np.any(np.isfinite(cost_to_go[0, :])):
            return None
        
        return policy_table
    
    def _compute_stage_costs(self, socs: np.ndarray, clipped_energies: np.ndarray,
                            grid_cost: np.ndarray, grid_violation: np.ndarray,
                            num_soc_levels: int, num_actions: int) -> np.ndarray:
        """
        Compute stage cost including degradation for each (soc, action) pair.
        
        Returns:
            Array of shape (num_soc_levels, num_actions) with stage costs
        """
        stage_costs = np.full((num_soc_levels, num_actions), np.inf)
        
        for si in range(num_soc_levels):
            soc_val = float(socs[si, 0])
            for ai in range(num_actions):
                # Skip infeasible states
                if grid_violation[si, ai]:
                    continue
                
                energy = float(clipped_energies[si, ai])
                battery_rate = energy / self.step_duration
                
                # Compute degradation cost
                if self.degradation_model == 'rainflow':
                    soc_next = soc_val + energy
                    deg_frac = self.degradation_calc.compute_rainflow_degradation(
                        soc_val, soc_next
                    )
                    degradation_cost = deg_frac * self.battery_life_cost
                elif self.degradation_model == 'linear':
                    degradation_cost = self.linear_deg_cost_per_kwh * abs(energy)
                else:
                    # Linearized class-based degradation
                    Id = abs(min(0.0, battery_rate)) / self.battery_capacity
                    Ich = abs(max(0.0, battery_rate)) / self.battery_capacity
                    avg_soc = (soc_val + 0.5 * energy) / self.battery_capacity * 100.0
                    avg_soc = float(np.clip(avg_soc, 0.0, 100.0))
                    
                    deg_frac = self.degradation_calc.compute_linearized_degradation(
                        Id, Ich, avg_soc, abs(energy)
                    )
                    degradation_cost = deg_frac * self.battery_life_cost
                
                stage_costs[si, ai] = float(grid_cost[si, ai]) + degradation_cost
        
        return stage_costs
    
    def get_action_for_current_state(self, policy_table: np.ndarray, 
                                     current_soc_kwh: float) -> float:
        """
        Extract action from policy table for current battery state.
        
        Args:
            policy_table: Solved policy table
            current_soc_kwh: Current state of charge in kWh
        
        Returns:
            Normalized action value in [-1, 1]
        """
        # Find nearest SoC level
        soc_idx = np.argmin(np.abs(self.soc_levels_kwh - current_soc_kwh))
        
        # Get action index from policy
        action_idx = policy_table[0, soc_idx]
        
        if action_idx == -1:
            return 0.0  # No feasible action, return zero
        
        # Convert to normalized action
        action_value = self.action_levels[int(action_idx)]
        
        # Add small noise for exploration
        noise = np.random.normal(-0.001, 0.001)
        action_value = np.clip(action_value + noise, -1.0, 1.0)
        
        return float(action_value)
