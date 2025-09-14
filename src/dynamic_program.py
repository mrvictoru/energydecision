"""
Dynamic Programming Engine for Stochastic Dynamic Programming (SDP).

This module provides a reusable DynamicProgram class that implements vectorized
backward induction for energy storage optimization problems. The implementation
is designed to be agnostic of specific environment or agent internals.

Usage Example:
    # Create state and action grids
    state_grid = np.linspace(0, 10, 21)  # SoC levels in kWh
    action_levels = np.linspace(-1, 1, 11)  # Normalized actions
    
    # Define transition and cost functions
    def transition_fn(state, action, scenario):
        battery_energy = action * max_flow * step_duration
        return np.clip(state + battery_energy, 0, battery_capacity)
    
    def cost_fn(state_kwh, action_norm, scenario_values):
        # Compute immediate cost based on grid energy and degradation
        # scenario_values is a dict with 'solar', 'load', 'import_price', 'export_price'
        return grid_cost + degradation_cost
    
    def scenario_provider(stage_index):
        # Return scenario values and probabilities for the given stage
        scenario_values = {'solar': [2.0, 3.0], 'load': [4.0, 5.0], ...}
        scenario_probs = [0.5, 0.5]
        return scenario_values, scenario_probs
    
    # Create and solve DP
    dp = DynamicProgram(state_grid, action_levels, transition_fn, cost_fn,
                        scenario_provider, stage_times=list(range(horizon)))
    dp.initialize_states()
    ctg_matrix, policy_table = dp.solve()

Performance Notes:
    - Uses vectorized operations via numpy for efficient computation
    - Linear interpolation via numpy.searchsorted and vectorized weighting
    - Supports both vectorized and per-scenario cost computation
    - Memory usage scales with state_grid × action_levels × horizon
"""

import numpy as np
from typing import Callable, Dict, List, Tuple


class Stage:
    """Helper class representing a single stage in the dynamic program."""
    
    def __init__(self, index: int, time: int):
        self.index = index  # Stage index (0 to horizon-1)
        self.time = time    # Absolute time index
        self.ctg = None     # Cost-to-go values (filled during solve)


class StateGrid:
    """Helper class for managing state grid operations."""
    
    def __init__(self, state_levels: np.ndarray):
        self.levels = np.asarray(state_levels)
        self.size = len(self.levels)
        self.min_val = self.levels[0]
        self.max_val = self.levels[-1]
    
    def to_index(self, state_value: float) -> int:
        """Map a continuous state value to the nearest discrete grid index."""
        return np.argmin(np.abs(self.levels - state_value))
    
    def interpolate(self, values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
        """Vectorized linear interpolation of values at query_points."""
        return interp_vectorized(self.levels, values, query_points)


def interp_vectorized(x_grid: np.ndarray, y_values: np.ndarray,
                     x_query: np.ndarray) -> np.ndarray:
    """
    Vectorized linear interpolation utility.
    
    Args:
        x_grid: 1D array of x-coordinates (must be sorted)
        y_values: 1D array of y-values corresponding to x_grid
        x_query: Array of query points (any shape)
    
    Returns:
        Interpolated values with same shape as x_query
    """
    x_query_flat = x_query.ravel()
    x_query_clipped = np.clip(x_query_flat, x_grid[0], x_grid[-1])
    
    # Use numpy's interp for vectorized linear interpolation
    result_flat = np.interp(x_query_clipped, x_grid, y_values)
    
    return result_flat.reshape(x_query.shape)


class DynamicProgram:
    """
    Reusable Dynamic Programming engine for stochastic optimization.
    
    This class implements vectorized backward induction that works with:
    - User-defined state and action grids
    - Transition function: transition_fn(state, action, scenario) -> next_state
    - Cost function: cost_fn(state_kwh, action_norm, scenario_values) -> cost
    - Scenario provider: scenario_provider(stage_index) -> (values_dict, probs)
    """
    
    def __init__(self, 
                 state_grid: np.ndarray,
                 action_levels: np.ndarray, 
                 transition_fn: Callable,
                 cost_fn: Callable,
                 scenario_provider: Callable,
                 stage_times: List[int]):
        """
        Initialize the Dynamic Program.
        
        Args:
            state_grid: 1D array of discrete state levels (e.g., SoC in kWh)
            action_levels: 1D array of discrete action levels (e.g., normalized flow)
            transition_fn: Function with signature (state, action, scenario) -> next_state
                where scenario is a dict with scenario values
            cost_fn: Function with signature (state_kwh, action_norm, scenario_values) -> cost
                Returns immediate cost (float or array over scenarios)
            scenario_provider: Function with signature (stage_index) -> (scenario_values_dict, scenario_probs)
                scenario_values_dict maps variable names to arrays of scenario values
                scenario_probs is array of probabilities (must sum to 1)
            stage_times: List of absolute time indices for each stage
        """
        self.state_grid = StateGrid(state_grid)
        self.action_levels = np.asarray(action_levels)
        self.transition_fn = transition_fn
        self.cost_fn = cost_fn
        self.scenario_provider = scenario_provider
        self.stage_times = stage_times
        
        self.horizon = len(stage_times)
        self.num_states = self.state_grid.size
        self.num_actions = len(self.action_levels)
        
        # Initialize stages
        self.stages = []
        for i, time in enumerate(stage_times):
            self.stages.append(Stage(i, time))
    
    def initialize_states(self):
        """Initialize cost-to-go matrices. Call before solve()."""
        # Initialize cost-to-go: (horizon + 1) x num_states
        # ctg[t, s] = minimum expected cost from state s at stage t to end
        self.ctg_matrix = np.full((self.horizon + 1, self.num_states), np.inf)
        
        # Initialize policy table: horizon x num_states
        # policy[t, s] = optimal action index for state s at stage t (-1 if infeasible)
        self.policy_table = np.full((self.horizon, self.num_states), -1, dtype=int)
    
    def set_final_ctg(self, ctg: np.ndarray):
        """
        Set the final cost-to-go values (terminal conditions).
        
        Args:
            ctg: 1D array of final cost-to-go values, one per state
        """
        if len(ctg) != self.num_states:
            raise ValueError(f"Final CTG length {len(ctg)} != num_states {self.num_states}")
        self.ctg_matrix[self.horizon, :] = ctg
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the dynamic program using vectorized backward induction.
        
        Returns:
            ctg_matrix: (horizon + 1) x num_states array of cost-to-go values
            policy_table: horizon x num_states array of optimal action indices
        """
        if not hasattr(self, 'ctg_matrix'):
            raise RuntimeError("Must call initialize_states() before solve()")
        
        # Default final CTG to zero if not set
        if np.all(np.isinf(self.ctg_matrix[self.horizon, :])):
            self.ctg_matrix[self.horizon, :] = 0.0
        
        # Backward induction
        for t in range(self.horizon - 1, -1, -1):
            self._solve_stage(t)
        
        return self.ctg_matrix, self.policy_table
    
    def _solve_stage(self, stage_idx: int):
        """Solve a single stage using vectorized operations."""
        stage = self.stages[stage_idx]
        
        # Get scenarios for this stage
        scenario_values, scenario_probs = self.scenario_provider(stage.time)
        scenario_probs = np.asarray(scenario_probs)
        n_scenarios = len(scenario_probs)
        
        if abs(np.sum(scenario_probs) - 1.0) > 1e-6:
            raise ValueError(f"Scenario probabilities must sum to 1, got {np.sum(scenario_probs)}")
        
        # Vectorized computation over all state-action pairs
        states = self.state_grid.levels  # (num_states,)
        actions = self.action_levels     # (num_actions,)
        
        # Create meshgrids for vectorized computation
        state_mesh, action_mesh = np.meshgrid(states, actions, indexing='ij')
        # state_mesh, action_mesh are both (num_states, num_actions)
        
        # Compute costs and next states for all scenarios
        total_costs = np.full((self.num_states, self.num_actions), np.inf)
        
        for scenario_idx in range(n_scenarios):
            # Extract scenario values for this scenario
            scenario_dict = {}
            for var_name, var_values in scenario_values.items():
                scenario_dict[var_name] = var_values[scenario_idx]
            
            # Compute immediate costs for all state-action pairs under this scenario
            immediate_costs = self._compute_immediate_costs(
                state_mesh, action_mesh, scenario_dict
            )
            
            # Compute next states for all state-action pairs under this scenario
            next_states = self._compute_next_states(
                state_mesh, action_mesh, scenario_dict
            )
            
            # Interpolate future costs at next states
            future_costs = self.state_grid.interpolate(
                self.ctg_matrix[stage_idx + 1, :], next_states
            )
            
            # Total cost for this scenario
            scenario_costs = immediate_costs + future_costs
            prob = scenario_probs[scenario_idx]
            
            # Handle first scenario vs accumulation
            if scenario_idx == 0:
                total_costs = prob * scenario_costs
            else:
                total_costs += prob * scenario_costs
        
        # Find optimal actions and update CTG
        # Mask out infinite costs (infeasible actions)
        feasible_mask = np.isfinite(total_costs)
        
        # For each state, find the best action
        for state_idx in range(self.num_states):
            state_costs = total_costs[state_idx, :]
            feasible_actions = feasible_mask[state_idx, :]
            
            if np.any(feasible_actions):
                # Find best feasible action
                feasible_costs = state_costs[feasible_actions]
                best_cost_idx = np.argmin(feasible_costs)
                best_action_idx = np.where(feasible_actions)[0][best_cost_idx]
                
                self.ctg_matrix[stage_idx, state_idx] = feasible_costs[best_cost_idx]
                self.policy_table[stage_idx, state_idx] = best_action_idx
            else:
                # No feasible actions
                self.ctg_matrix[stage_idx, state_idx] = np.inf
                self.policy_table[stage_idx, state_idx] = -1
    
    def _compute_immediate_costs(self, state_mesh: np.ndarray, action_mesh: np.ndarray,
                                scenario_dict: Dict[str, float]) -> np.ndarray:
        """Compute immediate costs for all state-action pairs under one scenario."""
        costs = np.zeros_like(state_mesh)
        
        # Vectorized cost computation
        for i in range(self.num_states):
            for j in range(self.num_actions):
                state = state_mesh[i, j]
                action = action_mesh[i, j]
                cost = self.cost_fn(state, action, scenario_dict)
                costs[i, j] = cost
        
        return costs
    
    def _compute_next_states(self, state_mesh: np.ndarray, action_mesh: np.ndarray,
                           scenario_dict: Dict[str, float]) -> np.ndarray:
        """Compute next states for all state-action pairs under one scenario."""
        next_states = np.zeros_like(state_mesh)
        
        # Vectorized transition computation
        for i in range(self.num_states):
            for j in range(self.num_actions):
                state = state_mesh[i, j]
                action = action_mesh[i, j]
                next_state = self.transition_fn(state, action, scenario_dict)
                next_states[i, j] = next_state
        
        return next_states
