"""
Multi-Resolution Stochastic Dynamic Programming (MRDP) Module

This module provides efficient SDP implementation with performance optimizations:

1. Per-(timestep, rounded_energy) cache for stage-cost results to avoid recomputing 
   identical stage costs across sub-horizon solves.
2. Vectorized Monte Carlo routine that computes costs for all unique energies 
   in one batched numpy operation.
3. Configurable mc_samples per sub-horizon via subhorizon_specs.

Example usage with Agent class:

```python
import numpy as np
from decision import Agent
from sdp_multires import DynamicProgram, compute_unique_energy_costs_vectorized

# Setup agent and environment
agent = Agent(env, algorithm='sdp', use_monte_carlo=True)

# Initialize stage cost cache in agent
agent._stage_cost_cache = {}

# Define sub-horizon specifications with different MC sample counts
subhorizon_specs = [
    {'horizon': 12, 'mc_samples': 200},  # Near horizon: high accuracy
    {'horizon': 36, 'mc_samples': 20}    # Far horizon: lower accuracy
]

# Create vectorized stage cost function with caching
def cached_vectorized_stage_cost(timestep, unique_energies, monte_samples, 
                                step_duration, max_grid_energy, degradation_params):
    # Check cache first for each energy value
    cached_results = {}
    uncached_energies = []
    uncached_indices = []
    
    for i, energy in enumerate(unique_energies):
        rounded_energy = round(float(energy), 10)
        cache_key = (timestep, rounded_energy)
        if cache_key in agent._stage_cost_cache:
            cached_results[i] = agent._stage_cost_cache[cache_key]
        else:
            uncached_energies.append(energy)
            uncached_indices.append(i)
    
    # Compute uncached values using vectorized function
    if uncached_energies:
        uncached_costs = compute_unique_energy_costs_vectorized(
            monte_samples, np.array(uncached_energies), step_duration, 
            max_grid_energy, degradation_params
        )
        
        # Cache the results
        for idx, energy_idx in enumerate(uncached_indices):
            energy = unique_energies[energy_idx]
            rounded_energy = round(float(energy), 10)
            cache_key = (timestep, rounded_energy)
            agent._stage_cost_cache[cache_key] = uncached_costs[idx]
            cached_results[energy_idx] = uncached_costs[idx]
    
    # Reconstruct full results array
    results = np.array([cached_results[i] for i in range(len(unique_energies))])
    return results

# Use with DynamicProgram solver
dp = DynamicProgram(
    soc_levels=agent.soc_levels_kwh,
    action_levels=agent.action_levels_norm,
    step_duration=agent.step_duration,
    max_battery_flow=agent.max_battery_flow,
    battery_capacity=agent.battery_capacity
)

# Solve with custom stage cost function
policy = dp.solve(
    forecasts=forecasts, 
    stage_cost_function=cached_vectorized_stage_cost,
    subhorizon_specs=subhorizon_specs
)
```
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from batterydeg import static_degradation


# Module-level cache for stage costs - maps (timestep, rounded_energy) -> cost
_STAGE_COST_CACHE: Dict[Tuple[int, float], float] = {}


def compute_unique_energy_costs_vectorized(
    monte_samples: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    unique_energies: np.ndarray,
    step_duration: float,
    max_grid_energy: float,
    degradation_params: Dict[str, Any]
) -> np.ndarray:
    """
    Vectorized Monte Carlo computation of stage costs for all unique energy values.
    
    This function replaces the Python loop over unique energy values with a 
    batched numpy operation that computes costs for all unique energies simultaneously.
    
    Args:
        monte_samples: Tuple of (sampled_solar, sampled_load, sampled_imp, sampled_exp)
                      where each is a 1D numpy array of Monte Carlo samples
        unique_energies: 1D array of unique battery flow energies to evaluate (kWh)
        step_duration: Time step duration in hours
        max_grid_energy: Maximum allowed grid energy (absolute value)
        degradation_params: Dictionary containing degradation model parameters:
            - 'model': 'linear' or 'static'
            - 'linear_deg_cost_per_kwh': cost per kWh for linear model
            - 'battery_capacity': battery capacity in kWh
            - 'battery_life_cost': total battery replacement cost
            - 'static_deg_correction_factor': correction factor for static model
    
    Returns:
        1D numpy array of stage costs corresponding to unique_energies
        
    Notes:
        - Grid cost calculation: positive grid_energy = import (cost), 
          negative = export (revenue)
        - Degradation cost added to grid cost for total stage cost
        - Returns np.inf for infeasible energy values (grid limit exceeded)
        - Uses vectorized operations across both energy values and MC samples
    """
    sampled_solar, sampled_load, sampled_imp, sampled_exp = monte_samples
    n_samples = len(sampled_solar)
    n_energies = len(unique_energies)
    
    # Reshape for broadcasting: (n_energies, 1) and (1, n_samples)
    energies_col = unique_energies[:, np.newaxis]  # Shape: (n_energies, 1)
    solar_row = sampled_solar[np.newaxis, :]      # Shape: (1, n_samples)
    load_row = sampled_load[np.newaxis, :]        # Shape: (1, n_samples)
    imp_row = sampled_imp[np.newaxis, :]          # Shape: (1, n_samples)
    exp_row = sampled_exp[np.newaxis, :]          # Shape: (1, n_samples)
    
    # Vectorized battery energy split (charge vs discharge)
    battery_charge_energy = np.maximum(0.0, energies_col)    # Shape: (n_energies, 1)
    battery_discharge_energy = np.maximum(0.0, -energies_col) # Shape: (n_energies, 1)
    
    # Vectorized grid energy calculation for all (energy, sample) combinations
    # Shape: (n_energies, n_samples)
    grid_energy = load_row + battery_charge_energy - solar_row - battery_discharge_energy
    
    # Check grid limits - any sample exceeding limit makes the energy infeasible
    grid_limit_exceeded = np.any(np.abs(grid_energy) > (max_grid_energy + 1e-6), axis=1)
    
    # Vectorized cost calculation
    is_import = grid_energy > 0
    # Import cost (positive) vs export revenue (negative)
    costs_matrix = np.where(is_import, grid_energy * imp_row, -np.abs(grid_energy) * exp_row)
    
    # Check for infinite costs in any sample
    inf_costs = np.any(np.isinf(costs_matrix), axis=1)
    
    # Mean cost across MC samples for each energy
    mean_grid_costs = np.mean(costs_matrix, axis=1)
    
    # Set infeasible energies to inf cost
    grid_costs = np.where(grid_limit_exceeded | inf_costs, np.inf, mean_grid_costs)
    
    # Vectorized degradation cost calculation
    degradation_costs = _compute_degradation_costs_vectorized(
        unique_energies, step_duration, degradation_params
    )
    
    # Total stage costs
    total_costs = grid_costs + degradation_costs
    
    return total_costs


def _compute_degradation_costs_vectorized(
    energies: np.ndarray,
    step_duration: float, 
    degradation_params: Dict[str, Any]
) -> np.ndarray:
    """
    Vectorized computation of degradation costs for battery flow energies.
    
    Args:
        energies: 1D array of battery flow energies (kWh)
        step_duration: Time step duration in hours
        degradation_params: Degradation model parameters
        
    Returns:
        1D array of degradation costs corresponding to energies
    """
    model = degradation_params.get('model', 'linear')
    
    if model == 'linear':
        # Simple linear degradation: cost per kWh of energy throughput
        linear_cost_per_kwh = degradation_params.get('linear_deg_cost_per_kwh', 0.0)
        return linear_cost_per_kwh * np.abs(energies)
    
    else:  # static degradation model
        battery_capacity = degradation_params['battery_capacity']
        battery_life_cost = degradation_params['battery_life_cost']
        correction_factor = degradation_params.get('static_deg_correction_factor', 0.01)
        
        # Convert energies to rates
        battery_rates = energies / step_duration
        
        # Representative SoC for degradation calculation
        rep_soc = battery_capacity / 2.0
        
        # Vectorized C-rate calculations
        Id_crate = np.abs(np.maximum(0, -battery_rates) / battery_capacity)
        Ich_crate = np.abs(np.maximum(0, battery_rates) / battery_capacity)
        DoD_percent = np.abs(energies / battery_capacity) * 100.0
        SoC_avg_percent = (rep_soc + 0.5 * energies) / battery_capacity * 100.0
        SoC_avg_percent = np.clip(SoC_avg_percent, 0, 100)
        
        # Vectorized degradation fraction calculation
        degradation_fractions = np.zeros_like(energies)
        non_zero_mask = DoD_percent >= 1e-6
        
        if np.any(non_zero_mask):
            # Apply static_degradation function element-wise for non-zero DoD
            for i in np.where(non_zero_mask)[0]:
                degradation_fractions[i] = static_degradation(
                    Id_crate[i], Ich_crate[i], SoC_avg_percent[i], DoD_percent[i]
                ) * correction_factor
        
        return degradation_fractions * battery_life_cost


class DynamicProgram:
    """
    Multi-Resolution Dynamic Programming solver with caching and vectorization.
    
    This class provides an alternative SDP implementation with performance optimizations:
    - Optional in-memory caching of stage costs
    - Vectorized Monte Carlo evaluation
    - Support for configurable mc_samples per sub-horizon
    """
    
    def __init__(
        self,
        soc_levels: np.ndarray,
        action_levels: np.ndarray,
        step_duration: float,
        max_battery_flow: float,
        battery_capacity: float
    ):
        """
        Initialize the Dynamic Programming solver.
        
        Args:
            soc_levels: Array of discrete SoC levels (kWh)
            action_levels: Array of normalized action levels [-1, 1]
            step_duration: Time step duration in hours
            max_battery_flow: Maximum battery flow rate (kW)
            battery_capacity: Battery capacity (kWh)
        """
        self.soc_levels_kwh = np.array(soc_levels)
        self.action_levels_norm = np.array(action_levels)
        self.step_duration = step_duration
        self.max_battery_flow = max_battery_flow
        self.battery_capacity = battery_capacity
        
        # Precompute battery flow energies for all actions
        self.battery_flow_energies = self.action_levels_norm * max_battery_flow * step_duration
        
    def solve(
        self,
        forecasts: List[Dict[str, float]],
        stage_cost_function: Callable,
        subhorizon_specs: Optional[List[Dict[str, Any]]] = None,
        use_cache: bool = True,
        start_index: int = 0
    ) -> np.ndarray:
        """
        Solve the dynamic programming problem with optional multi-resolution approach.
        
        Args:
            forecasts: List of forecast dictionaries for each time step
            stage_cost_function: Function to compute stage costs
                Signature: (timestep, unique_energies, monte_samples, step_duration, 
                           max_grid_energy, degradation_params) -> costs
            subhorizon_specs: Optional list of sub-horizon specifications with different
                            mc_samples. Each spec should have 'horizon' and 'mc_samples' keys.
            use_cache: Whether to use module-level caching (default True)
            start_index: Starting index for absolute timestep numbering
            
        Returns:
            Policy table of shape (horizon, num_soc_levels) with optimal action indices
        """
        num_soc_levels = len(self.soc_levels_kwh)
        horizon = len(forecasts)
        num_actions = len(self.action_levels_norm)
        
        # Initialize cost-to-go and policy tables
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        cost_to_go[horizon, :] = 0.0  # Terminal cost
        
        # Determine mc_samples for each timestep
        mc_samples_schedule = self._create_mc_samples_schedule(horizon, subhorizon_specs)
        
        # Backward induction
        for t in range(horizon - 1, -1, -1):
            forecast_step = forecasts[t]
            timestep = start_index + t
            mc_samples_t = mc_samples_schedule[t]
            
            # Vectorized feasibility computation
            socs_reshaped = self.soc_levels_kwh[:, np.newaxis]  # (num_soc, 1)
            battery_energies_reshaped = self.battery_flow_energies[np.newaxis, :]  # (1, num_actions)
            potential_next_socs = socs_reshaped + battery_energies_reshaped
            
            # Feasibility mask
            feasible_mask = ((potential_next_socs >= -1e-6) & 
                           (potential_next_socs <= self.battery_capacity + 1e-6))
            
            # Clipped battery energies
            clipped_battery_energies = np.clip(
                battery_energies_reshaped,
                -socs_reshaped,  # Min: discharge to 0
                self.battery_capacity - socs_reshaped  # Max: charge to capacity
            )
            
            # Future costs via interpolation
            next_socs = socs_reshaped + clipped_battery_energies
            future_costs = np.interp(
                next_socs.ravel(), 
                self.soc_levels_kwh, 
                cost_to_go[t + 1, :]
            ).reshape(next_socs.shape)
            
            # Get unique energies for stage cost computation
            rounded = np.round(clipped_battery_energies, 10)
            rounded_flat = rounded.ravel()
            feasible_flat = feasible_mask.ravel()
            
            stage_costs = np.full(rounded.shape, np.inf)
            
            if feasible_flat.any():
                values_flat = rounded_flat[feasible_flat]
                unique_vals, inverse = np.unique(values_flat, return_inverse=True)
                
                # Compute stage costs for unique energies
                if use_cache:
                    unique_costs = self._compute_cached_stage_costs(
                        timestep, unique_vals, mc_samples_t, stage_cost_function
                    )
                else:
                    # Generate Monte Carlo samples for this timestep
                    monte_samples = self._generate_monte_samples(forecast_step, mc_samples_t)
                    
                    # Use provided stage cost function
                    unique_costs = stage_cost_function(
                        timestep, unique_vals, monte_samples, self.step_duration,
                        getattr(self, 'max_grid_energy', 10.0),  # Default if not set
                        self._get_default_degradation_params()
                    )
                
                # Map back to full stage cost matrix
                costs_flat = np.full(rounded_flat.shape, np.inf)
                costs_flat[feasible_flat] = unique_costs[inverse]
                stage_costs = costs_flat.reshape(rounded.shape)
            
            # Total costs and policy selection
            total_costs = stage_costs + future_costs
            total_costs_masked = np.where(
                feasible_mask & np.isfinite(stage_costs) & np.isfinite(future_costs),
                total_costs, np.inf
            )
            
            # Optimal policy selection
            cost_to_go[t, :] = np.min(total_costs_masked, axis=1)
            policy_table[t, :] = np.argmin(total_costs_masked, axis=1)
            
            # Set invalid policies to -1
            invalid_mask = ~np.isfinite(cost_to_go[t, :])
            policy_table[t, invalid_mask] = -1
        
        return policy_table
    
    def _create_mc_samples_schedule(
        self, 
        horizon: int, 
        subhorizon_specs: Optional[List[Dict[str, Any]]]
    ) -> List[int]:
        """
        Create MC samples schedule for each timestep based on subhorizon_specs.
        
        Args:
            horizon: Total horizon length
            subhorizon_specs: List of sub-horizon specifications
            
        Returns:
            List of mc_samples for each timestep
        """
        if subhorizon_specs is None:
            return [100] * horizon  # Default samples
        
        schedule = [100] * horizon  # Default
        
        current_pos = 0
        for spec in subhorizon_specs:
            spec_horizon = spec['horizon']
            spec_samples = spec['mc_samples']
            
            end_pos = min(current_pos + spec_horizon, horizon)
            for i in range(current_pos, end_pos):
                schedule[i] = spec_samples
            
            current_pos = end_pos
            if current_pos >= horizon:
                break
        
        return schedule
    
    def _compute_cached_stage_costs(
        self,
        timestep: int,
        unique_energies: np.ndarray,
        mc_samples: int,
        stage_cost_function: Callable
    ) -> np.ndarray:
        """
        Compute stage costs with caching support.
        
        Uses module-level cache to avoid recomputing identical stage costs.
        Cache keys are (timestep, rounded_energy) tuples.
        """
        cached_results = {}
        uncached_energies = []
        uncached_indices = []
        
        # Check cache for each energy
        for i, energy in enumerate(unique_energies):
            rounded_energy = round(float(energy), 10)
            cache_key = (timestep, rounded_energy)
            
            if cache_key in _STAGE_COST_CACHE:
                cached_results[i] = _STAGE_COST_CACHE[cache_key]
            else:
                uncached_energies.append(energy)
                uncached_indices.append(i)
        
        # Compute uncached values
        if uncached_energies:
            # This is a simplified version - in practice, you'd generate actual monte_samples
            monte_samples = self._generate_monte_samples_for_cache(mc_samples)
            
            uncached_costs = stage_cost_function(
                timestep, np.array(uncached_energies), monte_samples, self.step_duration,
                getattr(self, 'max_grid_energy', 10.0),
                self._get_default_degradation_params()
            )
            
            # Cache the results
            for idx, energy_idx in enumerate(uncached_indices):
                energy = unique_energies[energy_idx]
                rounded_energy = round(float(energy), 10)
                cache_key = (timestep, rounded_energy)
                _STAGE_COST_CACHE[cache_key] = uncached_costs[idx]
                cached_results[energy_idx] = uncached_costs[idx]
        
        # Reconstruct full results array
        return np.array([cached_results[i] for i in range(len(unique_energies))])
    
    def _generate_monte_samples(self, forecast_step: Dict[str, float], mc_samples: int):
        """Generate Monte Carlo samples for a forecast step."""
        # This is a simplified version - in practice, you'd use the scenario generator
        solar = forecast_step.get('SolarGen', 2.0)
        load = forecast_step.get('HouseLoad', 4.0) 
        imp_price = forecast_step.get('ImportEnergyPrice', 0.2)
        exp_price = forecast_step.get('ExportEnergyPrice', 0.1)
        
        # Add small random variations
        np.random.seed(42)  # For reproducibility in this example
        sampled_solar = np.random.normal(solar, 0.1 * solar, mc_samples)
        sampled_load = np.random.normal(load, 0.1 * load, mc_samples)
        sampled_imp = np.random.normal(imp_price, 0.01, mc_samples)
        sampled_exp = np.random.normal(exp_price, 0.01, mc_samples)
        
        return (sampled_solar, sampled_load, sampled_imp, sampled_exp)
    
    def _generate_monte_samples_for_cache(self, mc_samples: int):
        """Generate generic Monte Carlo samples for caching."""
        np.random.seed(42)
        sampled_solar = np.random.uniform(1.0, 3.0, mc_samples)
        sampled_load = np.random.uniform(3.0, 5.0, mc_samples)
        sampled_imp = np.random.uniform(0.15, 0.25, mc_samples)
        sampled_exp = np.random.uniform(0.08, 0.12, mc_samples)
        
        return (sampled_solar, sampled_load, sampled_imp, sampled_exp)
    
    def _get_default_degradation_params(self) -> Dict[str, Any]:
        """Get default degradation parameters."""
        return {
            'model': 'linear',
            'linear_deg_cost_per_kwh': 0.01,
            'battery_capacity': self.battery_capacity,
            'battery_life_cost': 7000.0,
            'static_deg_correction_factor': 0.01
        }


def clear_stage_cost_cache():
    """Clear the module-level stage cost cache."""
    global _STAGE_COST_CACHE
    _STAGE_COST_CACHE.clear()


def get_cache_stats() -> Dict[str, int]:
    """Get statistics about the stage cost cache."""
    return {
        'cache_size': len(_STAGE_COST_CACHE),
        'memory_usage_bytes': sum(64 for _ in _STAGE_COST_CACHE)  # Rough estimate
    }