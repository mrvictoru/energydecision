"""
Multi-Resolution Dynamic Programming (MRDP) Module

This module provides infrastructure for multi-resolution/multi-sub-horizon dynamic programming
for the SDP agent. It allows splitting the optimization horizon into multiple sub-horizons
with different resolutions (state/action discretization, time steps) and solving them 
sequentially with proper terminal cost propagation.

The module is designed to be non-invasive and reusable, allowing progressive integration
with existing Agent._solve_sdp without modifying Agent internals.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any


class DynamicProgram:
    """
    A reusable dynamic programming class that implements vectorized backward induction
    for a single sub-horizon. Preserves the performance patterns from the original
    Agent._solve_sdp implementation.
    """
    
    def __init__(self, 
                 soc_levels_kwh: np.ndarray, 
                 action_levels_kwh: np.ndarray, 
                 step_duration: float,
                 env: Any,
                 use_monte_carlo: bool = False,
                 mc_samples: int = 100):
        """
        Initialize the DynamicProgram instance.
        
        Args:
            soc_levels_kwh: Array of discretized SoC levels in kWh
            action_levels_kwh: Array of battery flow energy levels in kWh
            step_duration: Duration of each time step in hours
            env: Environment instance (used for accessing battery/grid constraints)
            use_monte_carlo: Whether to use Monte Carlo sampling for cost evaluation
            mc_samples: Number of Monte Carlo samples to use
        """
        self.soc_levels_kwh = np.array(soc_levels_kwh)
        self.action_levels_kwh = np.array(action_levels_kwh) 
        self.step_duration = step_duration
        self.env = env
        self.use_monte_carlo = use_monte_carlo
        self.mc_samples = mc_samples
        
        # Extract environment constraints
        self.battery_capacity = env.battery_capacity
        self.max_battery_flow = env.max_battery_flow
        self.max_grid_energy = env.max_grid_energy
        self.battery_life_cost = env.battery_life_cost
        
        # Initialize terminal cost-to-go (will be set via set_final_ctg or defaults to zero)
        self.terminal_ctg = None
        
        # Storage for computed policy and cost-to-go
        self.policy_table = None
        self.cost_to_go = None
        
    def set_final_ctg(self, states_kwh: np.ndarray, values: np.ndarray):
        """
        Set the terminal cost-to-go values for the sub-horizon.
        
        Args:
            states_kwh: Array of state values (SoC levels) in kWh
            values: Array of corresponding cost-to-go values
        """
        # Store terminal boundary conditions - will be interpolated to match our grid
        self.terminal_states_kwh = np.array(states_kwh)
        self.terminal_values = np.array(values)
        
        # Interpolate terminal values to match our SoC discretization
        self.terminal_ctg = np.interp(self.soc_levels_kwh, states_kwh, values)
        
    def get_first_stage_states_and_ctg(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the first-stage states and cost-to-go values after solving.
        This is used to set terminal conditions for the previous sub-horizon.
        
        Returns:
            Tuple of (states_kwh, ctg_values) where states_kwh are the SoC levels
            and ctg_values are the corresponding cost-to-go values at t=0.
        """
        if self.cost_to_go is None:
            raise RuntimeError("Must call solve() before extracting first-stage values")
            
        return self.soc_levels_kwh.copy(), self.cost_to_go[0, :].copy()
        
    def solve(self, 
              forecasts_segment: List[Dict], 
              start_index: int, 
              stage_cost_function: Callable[[int, np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the dynamic program using vectorized backward induction.
        
        Args:
            forecasts_segment: List of forecast dictionaries for this sub-horizon
            start_index: Global time index where this sub-horizon starts
            stage_cost_function: Function that computes stage costs for unique battery energies.
                                 Signature: stage_cost_function(t_global_idx, unique_energy_values) -> costs_array
                                 
        Returns:
            Tuple of (policy_table, cost_to_go) arrays
        """
        num_soc_levels = len(self.soc_levels_kwh)
        horizon = len(forecasts_segment)
        
        # Initialize cost-to-go (J) and policy tables  
        self.cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        self.policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        
        # Set terminal conditions
        if self.terminal_ctg is not None:
            self.cost_to_go[horizon, :] = self.terminal_ctg
        else:
            # Default: zero terminal cost
            self.cost_to_go[horizon, :] = 0.0
            
        # Extract dimensions
        num_actions = len(self.action_levels_kwh)
        socs = self.soc_levels_kwh  # shape (num_soc,)
        
        # Backward induction (vectorized per timestep)
        for t in range(horizon - 1, -1, -1):
            t_global_idx = start_index + t
            
            # Vectorized feasibility computation: shape (num_soc, num_actions)
            socs_reshaped = socs[:, np.newaxis]  # (num_soc, 1)  
            battery_energies_reshaped = self.action_levels_kwh[np.newaxis, :]  # (1, num_actions)
            potential_next_socs = socs_reshaped + battery_energies_reshaped  # (num_soc, num_actions)
            
            # Feasibility mask: within [0, battery_capacity] with tolerance
            feasible_mask = ((potential_next_socs >= -1e-6) & 
                           (potential_next_socs <= self.battery_capacity + 1e-6))
            
            # Clipped battery flow energies for each (soc, action) pair
            clipped_battery_energies = np.clip(
                battery_energies_reshaped,
                -socs_reshaped,  # minimum: discharge down to 0
                self.battery_capacity - socs_reshaped  # maximum: charge up to capacity
            )
            
            # Compute next SoCs and future costs via vectorized interpolation
            next_socs = socs_reshaped + clipped_battery_energies  # (num_soc, num_actions)
            # Vectorized interpolation over flattened array is much faster than per-element calls
            future_costs = np.interp(next_socs.ravel(), self.soc_levels_kwh, self.cost_to_go[t + 1, :]).reshape(next_socs.shape)
            
            # Build unique energy list from feasible clipped energies using numpy.unique with return_inverse
            # This is the key optimization: compute stage costs only once per unique clipped battery energy
            rounded = np.round(clipped_battery_energies, 10)
            rounded_flat = rounded.ravel()
            feasible_flat = feasible_mask.ravel()
            
            stage_costs = np.full(rounded.shape, np.inf)
            
            if feasible_flat.any():
                values_flat = rounded_flat[feasible_flat]
                unique_vals, inverse = np.unique(values_flat, return_inverse=True)
                
                # Call the provided stage cost function for all unique energy values
                # This separates orchestration logic from stage-cost computation
                unique_costs = stage_cost_function(t_global_idx, unique_vals)
                
                # Scatter unique_costs back into stage_costs using inverse indices
                costs_flat = np.full(rounded_flat.shape, np.inf)
                costs_flat[feasible_flat] = unique_costs[inverse]
                stage_costs = costs_flat.reshape(rounded.shape)
            
            # Vectorized total cost and optimal action selection
            total_costs = stage_costs + future_costs
            # Mask invalid entries
            total_costs_masked = np.where(feasible_mask & np.isfinite(stage_costs) & np.isfinite(future_costs), 
                                        total_costs, np.inf)
            
            # Row-wise minima and argmin to choose best action for each SoC
            row_min = np.min(total_costs_masked, axis=1)
            best_actions = np.argmin(total_costs_masked, axis=1)
            
            # Update cost_to_go and policy_table; set policy to -1 where row_min is inf
            finite_mask = np.isfinite(row_min)
            self.cost_to_go[t, :] = row_min
            self.policy_table[t, :] = -1
            self.policy_table[t, finite_mask] = best_actions[finite_mask]
            
        return self.policy_table.copy(), self.cost_to_go.copy()


def solve_mrdp(env: Any,
               forecasts: List[Dict], 
               subhorizon_specs: List[Dict],
               global_start_index: int,
               stage_cost_function: Callable[[int, np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve multi-resolution dynamic programming across multiple sub-horizons.
    
    This orchestration function creates DynamicProgram instances for each sub-horizon
    and solves them sequentially from the last to the first, propagating terminal
    cost-to-go values between consecutive sub-horizons.
    
    Args:
        env: Environment instance (for accessing constraints and parameters)
        forecasts: Complete forecast sequence covering all sub-horizons
        subhorizon_specs: List of sub-horizon specifications, each containing:
            - 'start': Starting time index within forecasts (int)
            - 'length': Number of time steps for this sub-horizon (int)  
            - 'soc_resolution': Number of SoC discretization levels (int)
            - 'action_resolution': Number of action discretization levels (int)
            - 'step_duration': Time step duration in hours (float)
        global_start_index: Global time index corresponding to forecasts[0] 
        stage_cost_function: Function to compute stage costs for unique battery energies.
                           Signature: stage_cost_function(t_global_idx, unique_energy_values) -> costs_array
                           
    Returns:
        Tuple of (policy_table, cost_to_go) for the first sub-horizon, which contains
        the optimal actions for the immediate decision period.
        
    Example:
        # Define two sub-horizons: near-term fine-grained, far-term coarse-grained
        subhorizon_specs = [
            {
                'start': 0, 'length': 12, 
                'soc_resolution': 20, 'action_resolution': 11, 
                'step_duration': 0.5  # 30-min steps
            },
            {
                'start': 12, 'length': 36,
                'soc_resolution': 10, 'action_resolution': 5,
                'step_duration': 1.0  # 1-hour steps  
            }
        ]
        
        policy, ctg = solve_mrdp(env, forecasts, subhorizon_specs, 0, my_stage_cost_function)
    """
    if not subhorizon_specs:
        raise ValueError("Must provide at least one sub-horizon specification")
        
    # Validate sub-horizon specifications
    for i, spec in enumerate(subhorizon_specs):
        required_keys = ['start', 'length', 'soc_resolution', 'action_resolution', 'step_duration']
        for key in required_keys:
            if key not in spec:
                raise ValueError(f"Sub-horizon {i} missing required key: {key}")
    
    # Create DynamicProgram instances for each sub-horizon
    dynamic_programs = []
    for spec in subhorizon_specs:
        # Create SoC discretization levels
        soc_levels_kwh = np.linspace(0, env.battery_capacity, spec['soc_resolution'])
        
        # Create action discretization levels (battery flow energies in kWh)
        max_energy_per_step = env.max_battery_flow * spec['step_duration']
        action_levels_norm = np.linspace(-1, 1, spec['action_resolution'])
        action_levels_kwh = action_levels_norm * max_energy_per_step
        
        # Create DP instance
        dp = DynamicProgram(
            soc_levels_kwh=soc_levels_kwh,
            action_levels_kwh=action_levels_kwh,
            step_duration=spec['step_duration'],
            env=env,
            use_monte_carlo=getattr(env, 'use_monte_carlo', False),  # Use env settings if available
            mc_samples=getattr(env, 'mc_samples', 100)
        )
        
        dynamic_programs.append(dp)
    
    # Solve sub-horizons backward: last to first
    for i in range(len(subhorizon_specs) - 1, -1, -1):
        spec = subhorizon_specs[i]
        dp = dynamic_programs[i]
        
        # Extract forecast segment for this sub-horizon
        forecast_start = spec['start']
        forecast_end = forecast_start + spec['length']
        forecasts_segment = forecasts[forecast_start:forecast_end]
        
        if not forecasts_segment:
            raise ValueError(f"Sub-horizon {i} has empty forecast segment")
            
        # Global time index for this sub-horizon start
        sub_start_index = global_start_index + forecast_start
        
        # If not the last sub-horizon, set terminal cost-to-go from the next sub-horizon
        if i < len(subhorizon_specs) - 1:
            next_dp = dynamic_programs[i + 1]
            terminal_states, terminal_values = next_dp.get_first_stage_states_and_ctg()
            dp.set_final_ctg(terminal_states, terminal_values)
        # else: last sub-horizon uses default zero terminal cost
        
        # Solve this sub-horizon
        dp.solve(forecasts_segment, sub_start_index, stage_cost_function)
    
    # Return policy table and cost-to-go from the first sub-horizon
    first_dp = dynamic_programs[0]
    return first_dp.policy_table, first_dp.cost_to_go


# Example usage and integration guidance
"""
Integration Example: How to use MRDP with existing Agent

The following example shows how to create a stage_cost_function that wraps 
the existing Agent's stage cost calculation logic and how to call solve_mrdp
from within Agent._solve_sdp (or from Agent.choose_action).

# Step 1: Create a stage cost function that reuses Agent's logic
def create_agent_stage_cost_function(agent, forecasts):
    '''
    Create a stage cost function that wraps Agent's existing stage cost calculation.
    This function reuses Agent's scenario_cache and Monte Carlo settings.
    '''
    def stage_cost_function(t_global_idx, unique_energy_values):
        '''
        Compute stage costs for unique battery energy values at time t_global_idx.
        
        Args:
            t_global_idx: Global time index
            unique_energy_values: Array of unique battery flow energies (kWh)
            
        Returns:
            Array of stage costs corresponding to unique_energy_values
        '''
        if t_global_idx < 0 or t_global_idx >= len(forecasts):
            return np.full(len(unique_energy_values), np.inf)
            
        forecast_step = forecasts[t_global_idx]
        costs = np.empty(len(unique_energy_values))
        
        for i, energy in enumerate(unique_energy_values):
            battery_rate = energy / agent.step_duration
            rep_soc = agent.battery_capacity / 2.0  # Representative SoC for cost calculation
            
            # Reuse Agent's existing stage cost calculation
            costs[i] = agent._calculate_sdp_stage_cost(t_global_idx, rep_soc, battery_rate, energy, forecast_step)
            
        return costs
        
    return stage_cost_function

# Step 2: Example integration within Agent
class Agent:
    # ... existing methods ...
    
    def _solve_sdp_multires(self, forecasts, start_index: int = 0):
        '''
        Alternative SDP solve using multi-resolution approach.
        Can be called from choose_action() or replace _solve_sdp().
        '''
        # Define multi-resolution structure: near-term fine, far-term coarse
        subhorizon_specs = [
            {
                'start': 0, 'length': 12,  # First 12 steps (6 hours if 30-min steps)
                'soc_resolution': 20, 'action_resolution': 11,
                'step_duration': 0.5
            },
            {
                'start': 12, 'length': 36,  # Next 36 steps (36 hours if 1-hour steps)
                'soc_resolution': 10, 'action_resolution': 5,
                'step_duration': 1.0
            }
        ]
        
        # Create stage cost function that reuses existing Agent logic
        stage_cost_fn = create_agent_stage_cost_function(self, forecasts)
        
        # Solve using MRDP
        policy_table, cost_to_go = solve_mrdp(
            env=self.env,
            forecasts=forecasts,
            subhorizon_specs=subhorizon_specs,
            global_start_index=start_index,
            stage_cost_function=stage_cost_fn
        )
        
        return policy_table
        
    def choose_action(self, obs):
        # ... existing code ...
        
        if self.algorithm == 'sdp_multires':
            # Get forecasts
            forecasts = self._get_forecasts(current_step, horizon=48)  
            
            # Solve using MRDP
            policy_table = self._solve_sdp_multires(forecasts, start_index=current_step)
            
            # Extract action for current SoC
            current_soc_idx = self._soc_to_idx(self.env.battery_level)
            action_idx = policy_table[0, current_soc_idx]
            
            if action_idx >= 0:
                return self.action_levels_norm[action_idx]  # Use action levels from first sub-horizon
            else:
                return 0.0  # Fallback
                
        # ... rest of existing algorithm logic ...

# Step 3: Simple test/validation example
if __name__ == "__main__":
    # This would be run as a simple validation script
    import sys
    sys.path.append('.')  # Adjust as needed
    
    from EnergySimEnv import SolarBatteryEnv
    from decision import Agent
    
    # Create simple test environment with dummy data
    # ... create test environment and agent ...
    
    # Compare single-horizon vs multi-resolution solve times and solutions
    # ... validation code ...
"""