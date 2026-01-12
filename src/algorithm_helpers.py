"""
Helper classes for energy decision algorithms (SDP, MRDP, Oracle).

This module extracts algorithm-specific logic from the main Agent class to improve
readability and maintainability. Each algorithm helper encapsulates its own methods
while using shared degradation models from batterydeg.py.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from batterydeg import DegradationModel, RainflowCounter


class DegradationCalculator:
    """
    Centralized degradation calculation for all algorithms.
    
    Uses DegradationModel and RainflowCounter from batterydeg.py to ensure
    consistency with the multi-factor battery cycle life prediction methodology
    from Muenzel et al. (2015).
    """
    
    def __init__(self, battery_capacity: float, step_duration: float, 
                 battery_life_cost: float, degradation_temperature: float = 25.0):
        """
        Initialize degradation calculator.
        
        Args:
            battery_capacity: Battery capacity in kWh
            step_duration: Time step duration in hours
            battery_life_cost: Total cost of battery replacement in $
            degradation_temperature: Operating temperature in °C
        """
        self.battery_capacity = battery_capacity
        self.step_duration = step_duration
        self.battery_life_cost = battery_life_cost
        self.degradation_temperature = degradation_temperature
        
        # Initialize the class-based degradation model from batterydeg.py
        self.cycle_degradation_model = DegradationModel()
    
    def degradation_per_cycle(self, Id: float, Ich: float, soc_percent: float, DoD: float) -> float:
        """
        Calculate degradation fraction per cycle using the class-based model.
        
        Args:
            Id: Discharge current C-rate
            Ich: Charge current C-rate
            soc_percent: Average state of charge in percent (0-100)
            DoD: Depth of discharge in percent (0-100)
            
        Returns:
            Degradation fraction (0-1) for this cycle
        """
        return self.cycle_degradation_model.degradation_per_cycle(
            T=self.degradation_temperature,
            Id=Id,
            Ich=Ich,
            SOCav=soc_percent,
            DOD=DoD,
        )
    
    def compute_linearized_degradation(self, Id: float, Ich: float, soc_percent: float, 
                                      energy_kwh: float, base_DoD: float = 80.0,
                                      correction_factor: float = 1.0) -> float:
        """
        Compute degradation fraction using linearized per-kWh model.
        
        This method converts a representative full-cycle wear into a per-kWh wear 
        using a base DoD, then scales by the energy moved in this step.
        
        Args:
            Id: Discharge current C-rate
            Ich: Charge current C-rate
            soc_percent: Average state of charge in percent (0-100)
            energy_kwh: Energy throughput in kWh
            base_DoD: Base depth of discharge for reference cycle (default 80%)
            correction_factor: Static correction factor to apply
            
        Returns:
            Degradation fraction (0-1) for this energy throughput
        """
        if energy_kwh <= 0:
            return 0.0
        
        # Calculate energy for a full base cycle (charge + discharge)
        energy_full_base_cycle = self.battery_capacity * (base_DoD / 100.0) * 2.0
        if energy_full_base_cycle <= 0:
            return 0.0
        
        # Get degradation for one full cycle at base DoD
        cycle_wear = self.degradation_per_cycle(Id, Ich, soc_percent, base_DoD)
        
        # Convert to per-kWh wear and apply to actual energy
        wear_per_kwh = cycle_wear / energy_full_base_cycle
        frac = wear_per_kwh * energy_kwh * correction_factor
        
        return float(np.clip(frac, 0.0, 1.0))
    
    def compute_rainflow_degradation(self, soc_start_kwh: float, soc_end_kwh: float) -> float:
        """
        Estimate degradation for a single step using rainflow counting.
        
        Uses the RainflowCounter class from batterydeg.py to detect cycles
        and calculate degradation.
        
        Args:
            soc_start_kwh: Starting state of charge in kWh
            soc_end_kwh: Ending state of charge in kWh
            
        Returns:
            Degradation fraction (0-1) for this SoC transition
        """
        if self.battery_capacity <= 0:
            return 0.0
        
        # Convert to percentages
        start_pct = np.clip((soc_start_kwh / self.battery_capacity) * 100.0, 0.0, 100.0)
        end_pct = np.clip((soc_end_kwh / self.battery_capacity) * 100.0, 0.0, 100.0)
        
        # Use RainflowCounter to detect cycles
        counter = RainflowCounter(step_duration=self.step_duration)
        cycles = []
        for val in (start_pct, end_pct, start_pct):
            cycles.extend(counter.update(val))
        
        # Sum degradation from all detected cycles
        deg_frac = 0.0
        for SoC_avg, DoD, Id_cycle, Ich_cycle in cycles:
            deg_frac += self.degradation_per_cycle(Id_cycle, Ich_cycle, SoC_avg, DoD)
        
        return float(np.clip(deg_frac, 0.0, 1.0))


def interpolate_ctg(soc_levels_kwh: np.ndarray, ctg_array: np.ndarray, soc_value: float) -> float:
    """
    Linearly interpolate cost-to-go values for a continuous SoC between discrete levels.
    
    Clamps at the ends if soc_value is outside the range.
    
    Args:
        soc_levels_kwh: Array of discrete SoC levels in kWh
        ctg_array: Array of cost-to-go values corresponding to soc_levels_kwh
        soc_value: Continuous SoC value to interpolate at
    
    Returns:
        Interpolated cost-to-go value
    """
    soc_value = np.clip(soc_value, soc_levels_kwh[0], soc_levels_kwh[-1])
    return np.interp(soc_value, soc_levels_kwh, ctg_array)


def compute_grid_cost(grid_energy: float, import_price: float, export_price: float, 
                     max_grid_energy: float) -> float:
    """
    Compute grid cost with explicit import/export semantics and grid limit checking.
    
    Args:
        grid_energy: Grid energy (positive = import, negative = export)
        import_price: Price per kWh for importing energy
        export_price: Price per kWh for exporting energy (revenue)
        max_grid_energy: Maximum allowed grid energy (absolute value)
    
    Returns:
        Grid cost (positive = cost, negative = revenue, np.inf if limit exceeded)
    """
    # Check grid limits first
    if abs(grid_energy) > max_grid_energy + 1e-6:  # Add small tolerance
        return np.inf
    
    if grid_energy > 0:  # Importing energy
        return grid_energy * import_price
    else:  # Exporting energy (grid_energy is negative)
        # Export generates revenue, so cost is negative (revenue reduces total cost)
        export_revenue = abs(grid_energy) * export_price
        return -export_revenue


class OracleHelper:
    """
    Helper class for Oracle algorithm implementation.
    
    The Oracle agent uses perfect future information to compute the optimal
    policy via dynamic programming. This is used as an upper bound for comparison.
    """
    
    def __init__(self, env: Any, degradation_calc: DegradationCalculator,
                 degradation_model: str = 'linear', linear_deg_cost_p_kwh: Optional[float] = None):
        """
        Initialize Oracle helper.
        
        Args:
            env: Environment instance
            degradation_calc: DegradationCalculator instance
            degradation_model: Type of degradation model ('linear' or 'rainflow')
            linear_deg_cost_p_kwh: Degradation cost per kWh for linear model
        """
        self.env = env
        self.degradation_calc = degradation_calc
        self.degradation_model = degradation_model
        
        # Linear degradation cost (if using linear model)
        if self.degradation_model == 'linear':
            if linear_deg_cost_p_kwh is not None:
                self.linear_deg_cost_per_kwh = linear_deg_cost_p_kwh
            else:
                # Default: assume 3650 cycles (10 years daily)
                cycle_life = 3650
                self.linear_deg_cost_per_kwh = (
                    env.battery_life_cost / (env.battery_capacity * cycle_life)
                )
    
    def solve_oracle_dp(self, start_index: int, horizon: int, soc_levels_kwh: np.ndarray,
                       action_levels: np.ndarray, max_battery_flow: float, 
                       step_duration: float) -> Optional[np.ndarray]:
        """
        Solve dynamic programming problem using actual future values (oracle).
        
        This implements backward induction DP with perfect information about
        future solar generation, load, and prices.
        
        Args:
            start_index: Starting time index in environment
            horizon: Number of steps to look ahead
            soc_levels_kwh: Discretized SoC levels in kWh
            action_levels: Discretized action levels (normalized -1 to 1)
            max_battery_flow: Maximum battery power flow in kW
            step_duration: Time step duration in hours
            
        Returns:
            Policy table (horizon x num_soc_levels) or None if infeasible
        """
        num_soc_levels = len(soc_levels_kwh)
        num_actions = len(action_levels)
        
        # Initialize cost-to-go and policy tables
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        cost_to_go[horizon, :] = 0.0
        
        # Precompute battery energies for all actions
        socs = soc_levels_kwh[:, np.newaxis]  # (S, 1)
        battery_energies = action_levels * max_battery_flow * step_duration  # (A,)
        
        # Backward induction
        for t in range(horizon - 1, -1, -1):
            row = self.env._get_row(start_index + t)
            solar = row['SolarGen']
            load = row['HouseLoad']
            imp_price = row['ImportEnergyPrice']
            exp_price = row['ExportEnergyPrice']
            
            # Vectorized feasibility and energy calculation
            battery_reshaped = battery_energies[np.newaxis, :]  # (1, A)
            clipped_energies = np.clip(
                battery_reshaped,
                -socs,  # Can't discharge more than current SoC
                self.env.battery_capacity - socs  # Can't charge beyond capacity
            )
            
            battery_rates = clipped_energies / step_duration
            battery_charge = np.maximum(clipped_energies, 0.0)
            battery_discharge = np.maximum(-clipped_energies, 0.0)
            
            # Compute grid energy and costs
            demand = load + battery_charge
            supply = solar + battery_discharge
            grid_energy = demand - supply
            grid_violation = np.abs(grid_energy) > (self.env.max_grid_energy + 1e-6)
            grid_cost = np.where(
                grid_energy >= 0,
                grid_energy * imp_price,
                -np.abs(grid_energy) * exp_price
            )
            grid_cost[grid_violation] = np.inf
            
            # Compute degradation cost for each (soc, action) pair
            stage_costs = np.full((num_soc_levels, num_actions), np.inf)
            for si in range(num_soc_levels):
                soc_val = float(socs[si, 0])
                for ai in range(num_actions):
                    if grid_violation[si, ai]:
                        continue
                    
                    energy = float(clipped_energies[si, ai])
                    battery_rate = energy / step_duration
                    
                    # Calculate degradation cost using degradation calculator
                    if self.degradation_model == 'rainflow':
                        soc_next = soc_val + energy
                        deg_frac = self.degradation_calc.compute_rainflow_degradation(
                            soc_val, soc_next
                        )
                    else:  # linear
                        if self.degradation_model == 'linear':
                            degradation_cost = self.linear_deg_cost_per_kwh * abs(energy)
                            stage_costs[si, ai] = float(grid_cost[si, ai]) + degradation_cost
                            continue
                        
                        # Linearized model with class-based degradation
                        Id = abs(min(0.0, battery_rate)) / self.env.battery_capacity
                        Ich = abs(max(0.0, battery_rate)) / self.env.battery_capacity
                        avg_soc = (soc_val + 0.5 * energy) / self.env.battery_capacity * 100.0
                        avg_soc = float(np.clip(avg_soc, 0.0, 100.0))
                        energy_abs = abs(energy)
                        deg_frac = self.degradation_calc.compute_linearized_degradation(
                            Id, Ich, avg_soc, energy_abs
                        )
                    
                    degradation_cost = deg_frac * self.env.battery_life_cost
                    stage_costs[si, ai] = float(grid_cost[si, ai]) + degradation_cost
            
            # Compute next SoCs and future costs
            next_socs = socs + clipped_energies
            next_costs = np.interp(
                next_socs.ravel(),
                soc_levels_kwh,
                cost_to_go[t + 1, :]
            ).reshape(next_socs.shape)
            
            # Find optimal actions
            total_costs = stage_costs + next_costs
            feasible_mask = np.isfinite(total_costs)
            
            row_min = np.min(np.where(feasible_mask, total_costs, np.inf), axis=1)
            best_actions = np.argmin(np.where(feasible_mask, total_costs, np.inf), axis=1)
            
            finite_mask = np.isfinite(row_min)
            cost_to_go[t, :] = np.where(finite_mask, row_min, np.inf)
            policy_table[t, :] = -1
            policy_table[t, finite_mask] = best_actions[finite_mask]
        
        # Return None if no feasible solution found
        if not np.any(np.isfinite(cost_to_go[0, :])):
            return None
        
        return policy_table
