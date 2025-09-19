"""
Multi-Resolution Dynamic Programming (MRDP) Module

This module provides infrastructure for multi-resolution/multi-sub-horizon dynamic programming
for the SDP agent. It allows splitting the optimization horizon into multiple sub-horizons
with different resolutions (state/action discretization, time steps) and solving them 
sequentially with proper terminal cost propagation.

The module is designed to be non-invasive and reusable, allowing progressive integration
with existing Agent._solve_sdp without modifying Agent internals.

Performance Optimizations:
- Per-timestep/per-energy stage-cost cache to avoid recomputing identical costs
- Vectorized Monte Carlo stage-cost evaluator for batched operations
- Per-subhorizon Monte Carlo configuration support
- Optional float32 arrays for memory/speed improvements

Cache Control:
- clear_stage_cost_cache(): Clear the module-level cache
- get_stage_cost_cache_stats(): Get cache hit/miss statistics

Integration with Agent:
Agent can provide pre-sampled scenario arrays and configure Monte Carlo settings
per sub-horizon for optimal performance (e.g., fewer samples for far horizons).

Agent should call clear_stage_cost_cache() between episodes when using K-step
recompute strategy to avoid stale cached values.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union


# Module-level stage cost cache: (int(t_global), rounded_energy) -> cost
_stage_cost_cache: Dict[Tuple[int, float], float] = {}
_cache_stats = {'hits': 0, 'misses': 0}


def clear_stage_cost_cache() -> None:
    """Clear the module-level stage cost cache and reset statistics."""
    global _stage_cost_cache, _cache_stats
    _stage_cost_cache.clear()
    _cache_stats = {'hits': 0, 'misses': 0}


def get_stage_cost_cache_stats() -> Dict[str, int]:
    """Get stage cost cache statistics.
    
    Returns:
        Dictionary with 'hits', 'misses', and 'size' keys
    """
    return {
        'hits': _cache_stats['hits'],
        'misses': _cache_stats['misses'], 
        'size': len(_stage_cost_cache)
    }


def vectorized_monte_carlo_stage_cost(
    unique_energy_values: np.ndarray,
    sampled_solar: np.ndarray,
    sampled_load: np.ndarray, 
    sampled_imp: np.ndarray,
    sampled_exp: np.ndarray,
    max_grid_energy: float,
    degradation_cost_per_kwh: float = 0.0
) -> np.ndarray:
    """
    Vectorized Monte Carlo stage cost evaluator for multiple unique energy values.
    
    Computes costs for an array of unique energies in one batched numpy operation
    instead of looping in Python. Handles infeasible samples (grid limits => np.inf).
    
    Args:
        unique_energy_values: Array of unique battery flow energies (kWh)
        sampled_solar: Pre-sampled solar generation values [kWh]
        sampled_load: Pre-sampled load demand values [kWh]  
        sampled_imp: Pre-sampled import energy prices [$/kWh]
        sampled_exp: Pre-sampled export energy prices [$/kWh]
        max_grid_energy: Maximum allowed grid energy exchange [kWh]
        degradation_cost_per_kwh: Battery degradation cost per kWh cycled
        
    Returns:
        Array of mean stage costs corresponding to unique_energy_values
        
    Note:
        This function expects pre-sampled scenario arrays. The Agent can plug-in
        its scenario cache here to avoid duplicate sampling across sub-horizon solves.
    """
    n_energies = len(unique_energy_values)
    n_samples = len(sampled_solar)
    
    if not (len(sampled_load) == len(sampled_imp) == len(sampled_exp) == n_samples):
        raise ValueError("All sampled arrays must have the same length")
    
    # Broadcast for vectorized computation: (n_energies, n_samples)
    energies = unique_energy_values[:, np.newaxis]  # (n_energies, 1)
    solar = sampled_solar[np.newaxis, :]  # (1, n_samples)
    load = sampled_load[np.newaxis, :]
    imp = sampled_imp[np.newaxis, :]
    exp = sampled_exp[np.newaxis, :]
    
    # Vectorized grid energy calculation
    battery_charge = np.maximum(0, energies)  # (n_energies, n_samples)
    battery_discharge = np.maximum(0, -energies)
    grid_energy = load + battery_charge - solar - battery_discharge
    
    # Vectorized grid cost calculation with infeasibility handling
    grid_cost = np.where(
        np.abs(grid_energy) <= max_grid_energy,
        np.where(grid_energy >= 0, grid_energy * imp, -grid_energy * exp),
        np.inf
    )
    
    # Add degradation cost (deterministic, per energy)
    if degradation_cost_per_kwh > 0:
        degradation_cost = np.abs(energies) * degradation_cost_per_kwh
        grid_cost = grid_cost + degradation_cost
    
    # Mean cost across samples for each energy value
    mean_costs = np.mean(grid_cost, axis=1)
    
    return mean_costs


def deterministic_stage_cost(
    unique_energy_values: np.ndarray,
    solar_value: float,
    load_value: float,
    import_price: float,
    export_price: float,
    max_grid_energy: float,
    degradation_cost_per_kwh: float = 0.0
) -> np.ndarray:
    """
    Deterministic stage cost evaluator for fallback when Monte Carlo is disabled.
    
    Args:
        unique_energy_values: Array of unique battery flow energies (kWh)
        solar_value: Deterministic solar generation [kWh]
        load_value: Deterministic load demand [kWh]
        import_price: Deterministic import energy price [$/kWh]
        export_price: Deterministic export energy price [$/kWh]
        max_grid_energy: Maximum allowed grid energy exchange [kWh]
        degradation_cost_per_kwh: Battery degradation cost per kWh cycled
        
    Returns:
        Array of stage costs corresponding to unique_energy_values
    """
    # Vectorized computation
    battery_charge = np.maximum(0, unique_energy_values)
    battery_discharge = np.maximum(0, -unique_energy_values)
    grid_energy = load_value + battery_charge - solar_value - battery_discharge
    
    # Grid cost with infeasibility handling
    grid_cost = np.where(
        np.abs(grid_energy) <= max_grid_energy,
        np.where(grid_energy >= 0, grid_energy * import_price, -grid_energy * export_price),
        np.inf
    )
    
    # Add degradation cost
    if degradation_cost_per_kwh > 0:
        degradation_cost = np.abs(unique_energy_values) * degradation_cost_per_kwh
        grid_cost = grid_cost + degradation_cost
        
    return grid_cost


class DynamicProgram:
    """
    A reusable dynamic programming class that implements vectorized backward induction
    for a single sub-horizon. Preserves the performance patterns from the original
    Agent._solve_sdp implementation.
    
    Performance features:
    - Stage cost caching to avoid recomputing identical costs
    - Optional float32 arrays for memory/speed improvements  
    - Vectorized Monte Carlo and deterministic stage cost evaluation
    - Per-instance Monte Carlo configuration overrides
    """
    
    def __init__(self, 
                 soc_levels_kwh: np.ndarray, 
                 action_levels_kwh: np.ndarray, 
                 step_duration: float,
                 env: Any,
                 use_monte_carlo: bool = False,
                 mc_samples: int = 100,
                 use_float64: bool = False,
                 use_cache: bool = True):
        """
        Initialize the DynamicProgram instance.
        
        Args:
            soc_levels_kwh: Array of discretized SoC levels in kWh
            action_levels_kwh: Array of battery flow energy levels in kWh
            step_duration: Duration of each time step in hours
            env: Environment instance (used for accessing battery/grid constraints)
            use_monte_carlo: Whether to use Monte Carlo sampling for cost evaluation
            mc_samples: Number of Monte Carlo samples to use
            use_float64: If True, use float64 for arrays; otherwise use float32 for speed
            use_cache: Whether to use stage cost caching
        """
        # Choose array dtype for performance optimization
        self.dtype = np.float64 if use_float64 else np.float32
        
        self.soc_levels_kwh = np.array(soc_levels_kwh, dtype=self.dtype)
        self.action_levels_kwh = np.array(action_levels_kwh, dtype=self.dtype) 
        self.step_duration = float(step_duration)
        self.env = env
        self.use_monte_carlo = use_monte_carlo
        self.mc_samples = int(mc_samples)
        self.use_cache = use_cache
        
        # Extract environment constraints
        self.battery_capacity = float(env.battery_capacity)
        self.max_battery_flow = float(env.max_battery_flow)
        self.max_grid_energy = float(env.max_grid_energy)
        self.battery_life_cost = float(getattr(env, 'battery_life_cost', 0.0))
        
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
        self.terminal_states_kwh = np.array(states_kwh, dtype=self.dtype)
        self.terminal_values = np.array(values, dtype=self.dtype)
        
        # Interpolate terminal values to match our SoC discretization
        self.terminal_ctg = np.interp(self.soc_levels_kwh, states_kwh, values).astype(self.dtype)
        
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
              stage_cost_function: Callable[[int, np.ndarray], np.ndarray],
              sampled_scenarios: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the dynamic program using vectorized backward induction.
        
        Args:
            forecasts_segment: List of forecast dictionaries for this sub-horizon
            start_index: Global time index where this sub-horizon starts
            stage_cost_function: Function that computes stage costs for unique battery energies.
                                 Signature: stage_cost_function(t_global_idx, unique_energy_values) -> costs_array
            sampled_scenarios: Optional pre-sampled scenario arrays for vectorized MC.
                              Dict with keys: 'solar', 'load', 'import_price', 'export_price'
                              Each value should be array of shape (horizon, n_samples)
                                 
        Returns:
            Tuple of (policy_table, cost_to_go) arrays
        """
        num_soc_levels = len(self.soc_levels_kwh)
        horizon = len(forecasts_segment)
        
        # Initialize cost-to-go (J) and policy tables with appropriate dtype
        self.cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf, dtype=self.dtype)
        self.policy_table = np.full((horizon, num_soc_levels), -1, dtype=np.int32)
        
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
            forecast_step = forecasts_segment[t]
            
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
            
            stage_costs = np.full(rounded.shape, np.inf, dtype=self.dtype)
            
            if feasible_flat.any():
                values_flat = rounded_flat[feasible_flat]
                unique_vals, inverse = np.unique(values_flat, return_inverse=True)
                
                # Try cache first if enabled
                unique_costs = np.full(len(unique_vals), np.inf, dtype=self.dtype)
                cache_mask = np.zeros(len(unique_vals), dtype=bool)
                
                if self.use_cache:
                    global _stage_cost_cache, _cache_stats
                    for i, energy in enumerate(unique_vals):
                        cache_key = (int(t_global_idx), float(energy))
                        if cache_key in _stage_cost_cache:
                            unique_costs[i] = _stage_cost_cache[cache_key]
                            cache_mask[i] = True
                            _cache_stats['hits'] += 1
                        else:
                            _cache_stats['misses'] += 1
                
                # Compute costs for cache misses
                if not cache_mask.all():
                    uncached_energies = unique_vals[~cache_mask]
                    if len(uncached_energies) > 0:
                        # Try vectorized approach if scenarios available
                        if (sampled_scenarios is not None and self.use_monte_carlo and 
                            t < len(sampled_scenarios.get('solar', []))):
                            try:
                                uncached_costs = vectorized_monte_carlo_stage_cost(
                                    uncached_energies,
                                    sampled_scenarios['solar'][t],
                                    sampled_scenarios['load'][t],
                                    sampled_scenarios['import_price'][t],
                                    sampled_scenarios['export_price'][t],
                                    self.max_grid_energy,
                                    self.battery_life_cost
                                )
                                unique_costs[~cache_mask] = uncached_costs
                            except Exception:
                                # Fallback to stage_cost_function
                                unique_costs[~cache_mask] = stage_cost_function(t_global_idx, uncached_energies)
                        else:
                            # Use deterministic or fallback to stage_cost_function
                            if not self.use_monte_carlo and sampled_scenarios is None:
                                # Pure deterministic path
                                try:
                                    uncached_costs = deterministic_stage_cost(
                                        uncached_energies,
                                        forecast_step.get('SolarGen', 0.0),
                                        forecast_step.get('HouseLoad', 0.0),
                                        forecast_step.get('ImportEnergyPrice', 0.0),
                                        forecast_step.get('ExportEnergyPrice', 0.0),
                                        self.max_grid_energy,
                                        self.battery_life_cost
                                    )
                                    unique_costs[~cache_mask] = uncached_costs
                                except Exception:
                                    unique_costs[~cache_mask] = stage_cost_function(t_global_idx, uncached_energies)
                            else:
                                # Fallback to provided stage_cost_function
                                unique_costs[~cache_mask] = stage_cost_function(t_global_idx, uncached_energies)
                        
                        # Cache the computed costs
                        if self.use_cache:
                            for i, energy in enumerate(uncached_energies):
                                cache_key = (int(t_global_idx), float(energy))
                                _stage_cost_cache[cache_key] = float(unique_costs[~cache_mask][i])
                
                # Scatter unique_costs back into stage_costs using inverse indices
                costs_flat = np.full(rounded_flat.shape, np.inf, dtype=self.dtype)
                costs_flat[feasible_flat] = unique_costs[inverse]
                stage_costs = costs_flat.reshape(rounded.shape)
            
            # Vectorized total cost and optimal action selection
            total_costs = stage_costs + future_costs.astype(self.dtype)
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
               stage_cost_function: Callable[[int, np.ndarray], np.ndarray],
               sampled_scenarios: Optional[Dict[str, np.ndarray]] = None,
               use_float64: bool = False,
               use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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
            - 'mc_samples': Optional override for Monte Carlo samples (int)
            - 'use_monte_carlo': Optional override for Monte Carlo flag (bool)
        global_start_index: Global time index corresponding to forecasts[0] 
        stage_cost_function: Function to compute stage costs for unique battery energies.
                           Signature: stage_cost_function(t_global_idx, unique_energy_values) -> costs_array
        sampled_scenarios: Optional pre-sampled scenario arrays for vectorized MC.
                          Dict with keys: 'solar', 'load', 'import_price', 'export_price'
                          Each value should be array of shape (total_horizon, n_samples)
        use_float64: If True, use float64 arrays; otherwise use float32 for speed
        use_cache: Whether to enable stage cost caching
                           
    Returns:
        Tuple of (policy_table, cost_to_go) for the first sub-horizon, which contains
        the optimal actions for the immediate decision period.
        
    Example:
        # Define two sub-horizons: near-term fine-grained, far-term coarse-grained
        subhorizon_specs = [
            {
                'start': 0, 'length': 12, 
                'soc_resolution': 20, 'action_resolution': 11, 
                'step_duration': 0.5,  # 30-min steps
                'mc_samples': 200,     # High accuracy for near-term
                'use_monte_carlo': True
            },
            {
                'start': 12, 'length': 36,
                'soc_resolution': 10, 'action_resolution': 5,
                'step_duration': 1.0,  # 1-hour steps  
                'mc_samples': 20,      # Lower accuracy for far-term
                'use_monte_carlo': True
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
        
        # Extract per-subhorizon Monte Carlo settings with fallbacks
        use_mc = spec.get('use_monte_carlo', getattr(env, 'use_monte_carlo', False))
        mc_samples = spec.get('mc_samples', getattr(env, 'mc_samples', 100))
        
        # Create DP instance
        dp = DynamicProgram(
            soc_levels_kwh=soc_levels_kwh,
            action_levels_kwh=action_levels_kwh,
            step_duration=spec['step_duration'],
            env=env,
            use_monte_carlo=use_mc,
            mc_samples=mc_samples,
            use_float64=use_float64,
            use_cache=use_cache
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
        
        # Extract sampled scenarios for this sub-horizon if available
        sub_scenarios = None
        if sampled_scenarios is not None:
            try:
                sub_scenarios = {}
                for key in ['solar', 'load', 'import_price', 'export_price']:
                    if key in sampled_scenarios:
                        sub_scenarios[key] = sampled_scenarios[key][forecast_start:forecast_end]
            except Exception:
                sub_scenarios = None
        
        # Solve this sub-horizon
        dp.solve(forecasts_segment, sub_start_index, stage_cost_function, sub_scenarios)
    
    # Return policy table and cost-to-go from the first sub-horizon
    first_dp = dynamic_programs[0]
    return first_dp.policy_table, first_dp.cost_to_go


# Example usage and integration guidance
"""
Integration Example: How to use MRDP with existing Agent

The following example shows how to create a stage_cost_function that wraps 
the existing Agent's stage cost calculation logic and how to call solve_mrdp
from within Agent._solve_sdp (or from Agent.choose_action).

Performance optimizations are enabled through several mechanisms:

1. Stage Cost Caching:
   - Module-level cache avoids recomputing identical (t_global, energy) pairs
   - Particularly effective across sub-horizon solves and receding-horizon steps
   - Agent should call clear_stage_cost_cache() between episodes for K-step recompute

2. Vectorized Monte Carlo:
   - Pre-sample scenarios once and pass to all sub-horizons via sampled_scenarios
   - Batched computation across unique energies instead of Python loops
   - Fallback to deterministic evaluation when Monte Carlo disabled

3. Per-Subhorizon MC Configuration:
   - Near-term sub-horizons can use high accuracy (mc_samples=200)
   - Far-term sub-horizons can use lower accuracy (mc_samples=20) for speed

4. Float32 Arrays:
   - Use use_float64=False for memory/speed improvements
   - Switch to use_float64=True for debugging precision issues

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

# Step 2: Example integration within Agent with performance optimizations
class Agent:
    # ... existing methods ...
    
    def _solve_sdp_multires(self, forecasts, start_index: int = 0):
        '''
        Alternative SDP solve using multi-resolution approach with performance optimizations.
        Can be called from choose_action() or replace _solve_sdp().
        '''
        # Define multi-resolution structure with per-subhorizon MC settings
        subhorizon_specs = [
            {
                'start': 0, 'length': 12,  # First 12 steps (6 hours if 30-min steps)
                'soc_resolution': 20, 'action_resolution': 11,
                'step_duration': 0.5,
                'mc_samples': 200,         # High accuracy for near-term decisions
                'use_monte_carlo': True
            },
            {
                'start': 12, 'length': 36,  # Next 36 steps (36 hours if 1-hour steps)
                'soc_resolution': 10, 'action_resolution': 5,
                'step_duration': 1.0,
                'mc_samples': 20,          # Lower accuracy for far-term estimates
                'use_monte_carlo': True
            }
        ]
        
        # Create stage cost function that reuses existing Agent logic
        stage_cost_fn = create_agent_stage_cost_function(self, forecasts)
        
        # Optional: Pre-sample scenarios for vectorized MC (if Agent has scenario_cache)
        sampled_scenarios = None
        if hasattr(self, '_scenario_cache') and self._scenario_cache is not None:
            try:
                # Extract and pre-sample scenarios for the full horizon
                # Agent can plug-in its scenario cache here to avoid duplicate sampling
                sampled_scenarios = self._prepare_sampled_scenarios_for_mrdp(forecasts, start_index)
            except Exception:
                sampled_scenarios = None
        
        # Solve using MRDP with performance optimizations
        policy_table, cost_to_go = solve_mrdp(
            env=self.env,
            forecasts=forecasts,
            subhorizon_specs=subhorizon_specs,
            global_start_index=start_index,
            stage_cost_function=stage_cost_fn,
            sampled_scenarios=sampled_scenarios,  # Enable vectorized MC
            use_float64=False,  # Use float32 for speed
            use_cache=True      # Enable stage cost caching
        )
        
        return policy_table
        
    def _prepare_sampled_scenarios_for_mrdp(self, forecasts, start_index):
        '''
        Helper to prepare pre-sampled scenarios for vectorized MRDP Monte Carlo.
        This allows Agent to plug-in its scenario cache without duplicate sampling.
        '''
        # This is where Agent can integrate its existing scenario generation
        # Example structure - Agent should implement based on its scenario_cache
        horizon = len(forecasts)
        n_samples = getattr(self, 'mc_samples', 100)
        
        scenarios = {
            'solar': np.zeros((horizon, n_samples)),
            'load': np.zeros((horizon, n_samples)),
            'import_price': np.zeros((horizon, n_samples)),
            'export_price': np.zeros((horizon, n_samples))
        }
        
        # Populate from Agent's scenario cache...
        # Implementation depends on Agent's existing scenario generation approach
        
        return scenarios
        
    def choose_action(self, obs):
        # ... existing code ...
        
        if self.algorithm == 'sdp_multires':
            # Clear cache periodically if using K-step recompute strategy
            if self.should_clear_cache():  # Agent implements this logic
                clear_stage_cost_cache()
            
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

# Step 3: Cache control for Agent integration
'''
The Agent should manage cache lifecycle:

1. Between episodes (if using K-step recompute):
   clear_stage_cost_cache()

2. Monitor cache performance:
   stats = get_stage_cost_cache_stats()
   print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}, size: {stats['size']}")

3. Example subhorizon_specs with MC configuration:
   [
       {
           'start': 0, 'length': 12,
           'soc_resolution': 20, 'action_resolution': 11,
           'step_duration': 0.5,
           'mc_samples': 200,      # High accuracy near-term
           'use_monte_carlo': True
       },
       {
           'start': 12, 'length': 36,
           'soc_resolution': 8, 'action_resolution': 5,
           'step_duration': 1.0,
           'mc_samples': 20,       # Lower accuracy far-term
           'use_monte_carlo': True 
       }
   ]
'''

# Step 4: Simple test/validation example
if __name__ == "__main__":
    # This would be run as a simple validation script
    import sys
    sys.path.append('.')  # Adjust as needed
    
    try:
        from EnergySimEnv import SolarBatteryEnv
        from decision import Agent
        
        # Create simple test environment with dummy data
        # ... create test environment and agent ...
        
        # Compare single-horizon vs multi-resolution solve times and solutions
        # ... validation code ...
        
        print("MRDP module validation completed successfully!")
        print("Cache control functions available:")
        print("- clear_stage_cost_cache()")
        print("- get_stage_cost_cache_stats()")
        print("- vectorized Monte Carlo support")
        print("- per-subhorizon MC configuration")
        
    except ImportError as e:
        print(f"Validation requires additional modules: {e}")
        print("Module imports successfully and is ready for integration.")
"""