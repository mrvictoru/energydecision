"""
Drop-in adapter for integrating DynamicProgram with existing Agent code.

This module provides a thin adapter function that allows the existing Agent
to replace its internal _solve_sdp call with a single-line call to run_dp_for_agent
without refactoring the Agent internals.

The adapter handles the translation between Agent's existing interfaces and the
new DynamicProgram + SolverOrchestrator architecture.

Usage Example:
    # In Agent.choose_action(), replace:
    # policy_table = self._solve_sdp(forecasts, start_index=current_step_env)
    
    # With:
    from src.dp_adapter import run_dp_for_agent
    policy_table = run_dp_for_agent(self, forecasts, current_step_env)

The adapter automatically:
- Creates appropriate transition and cost functions from Agent methods
- Sets up scenario providers using Agent's _scenario_cache or scenario_generator
- Handles chunking for long horizons via SolverOrchestrator
- Returns policy_table in the same format as Agent._solve_sdp
"""

import numpy as np
from typing import Dict, List, Any, Optional
from solver_orchestrator import orchestrate_solve


class AgentAdapter:
    """
    Adapter that wraps an Agent to provide the AgentLikeInterface.
    
    This class takes an existing Agent instance and exposes the interface
    expected by the SolverOrchestrator, translating between the Agent's
    existing methods and the new DP framework.
    """
    
    def __init__(self, agent, forecasts: List[Dict], start_index: int):
        """
        Initialize the adapter with an Agent instance.
        
        Args:
            agent: Agent instance with _solve_sdp-compatible interface
            forecasts: List of forecast dictionaries  
            start_index: Absolute starting time index
        """
        self.agent = agent
        self.forecasts = forecasts
        self.start_index = start_index
        
        # Expose Agent's state and action grids
        self.soc_levels_kwh = agent.soc_levels_kwh
        self.action_levels_norm = agent.action_levels_norm
    
    def create_transition_fn(self):
        """Create transition function compatible with DynamicProgram."""
        def transition(state, action, scenario):
            """
            Transition function: (state, action, scenario) -> next_state
            
            Args:
                state: Current SoC in kWh
                action: Normalized action (-1 to 1)
                scenario: Dict with scenario values (not used in basic transition)
            
            Returns:
                Next SoC in kWh, clipped to battery capacity bounds
            """
            battery_flow_energy = action * self.agent.max_battery_flow * self.agent.step_duration
            next_soc = state + battery_flow_energy
            return np.clip(next_soc, 0.0, self.agent.battery_capacity)
        
        return transition
    
    def create_cost_fn(self):
        """Create cost function compatible with DynamicProgram."""
        def cost(state_kwh, action_norm, scenario_values):
            """
            Cost function: (state, action, scenario_values) -> immediate_cost
            
            Args:
                state_kwh: Current SoC in kWh
                action_norm: Normalized action (-1 to 1) 
                scenario_values: Dict with 'solar', 'load', 'import_price', 'export_price'
            
            Returns:
                Immediate cost (grid cost + degradation cost)
            """
            battery_flow_rate = action_norm * self.agent.max_battery_flow
            battery_flow_energy = battery_flow_rate * self.agent.step_duration
            
            # Create a forecast_step dict in the format expected by Agent's cost calculation
            forecast_step = {
                'SolarGen': scenario_values.get('solar', 0.0),
                'HouseLoad': scenario_values.get('load', 0.0), 
                'ImportEnergyPrice': scenario_values.get('import_price', 0.0),
                'ExportEnergyPrice': scenario_values.get('export_price', 0.0)
            }
            
            # Use Agent's existing stage cost calculation
            # Note: row_idx is not critical for the cost calculation in most cases
            row_idx = 0  # Placeholder, could be made more sophisticated if needed
            return self.agent._calculate_sdp_stage_cost(
                row_idx, state_kwh, battery_flow_rate, battery_flow_energy, forecast_step
            )
        
        return cost
    
    def create_scenario_provider(self, forecasts: List[Dict], start_index: int):
        """Create scenario provider compatible with DynamicProgram."""
        def scenario_provider(stage_index):
            """
            Scenario provider: (stage_index) -> (scenario_values_dict, scenario_probs)
            
            Args:
                stage_index: Absolute time index for the stage
            
            Returns:
                scenario_values_dict: Dict mapping variable names to scenario arrays
                scenario_probs: Array of scenario probabilities
            """
            # Try to use Agent's scenario cache first
            if hasattr(self.agent, '_scenario_cache') and self.agent._scenario_cache is not None:
                return self._get_scenarios_from_cache(stage_index)
            
            # Fallback to deterministic scenarios from forecasts
            return self._get_deterministic_scenarios(stage_index)
        
        return scenario_provider
    
    def _get_scenarios_from_cache(self, stage_index: int):
        """Extract scenarios from Agent's _scenario_cache."""
        try:
            cache = self.agent._scenario_cache
            
            # Extract scenario arrays for the given time index
            vals_solar_arr, ps_solar_arr = cache['solar']
            vals_load_arr, ps_load_arr = cache['load'] 
            vals_imp_arr, ps_imp_arr = cache['import_price']
            vals_exp_arr, ps_exp_arr = cache['export_price']
            
            # Check bounds
            if stage_index < 0 or stage_index >= vals_solar_arr.shape[0]:
                return self._get_deterministic_scenarios(stage_index)
            
            # Extract values and probabilities for this time step
            scenario_values = {
                'solar': vals_solar_arr[stage_index, :],
                'load': vals_load_arr[stage_index, :],
                'import_price': vals_imp_arr[stage_index, :],
                'export_price': vals_exp_arr[stage_index, :]
            }
            
            # For simplicity, assume uniform probabilities across scenarios
            # (More sophisticated: use the actual probabilities if available)
            n_scenarios = len(scenario_values['solar'])
            scenario_probs = np.ones(n_scenarios) / n_scenarios
            
            return scenario_values, scenario_probs
            
        except Exception:
            # Fallback if cache extraction fails
            return self._get_deterministic_scenarios(stage_index)
    
    def _get_deterministic_scenarios(self, stage_index: int):
        """Create single-scenario (deterministic) values from forecasts."""
        # Convert absolute stage_index back to relative forecast index
        forecast_idx = stage_index - self.start_index
        
        if forecast_idx < 0 or forecast_idx >= len(self.forecasts):
            # Use zeros as fallback for out-of-bounds indices
            scenario_values = {
                'solar': np.array([0.0]),
                'load': np.array([0.0]),
                'import_price': np.array([0.1]),  # Reasonable default price
                'export_price': np.array([0.05])  # Reasonable default price
            }
        else:
            forecast = self.forecasts[forecast_idx]
            scenario_values = {
                'solar': np.array([forecast.get('SolarGen', 0.0)]),
                'load': np.array([forecast.get('HouseLoad', 0.0)]),
                'import_price': np.array([forecast.get('ImportEnergyPrice', 0.1)]),
                'export_price': np.array([forecast.get('ExportEnergyPrice', 0.05)])
            }
        
        scenario_probs = np.array([1.0])  # Single scenario with probability 1
        return scenario_values, scenario_probs


def run_dp_for_agent(agent, 
                    forecasts: List[Dict[str, Any]], 
                    start_index: int,
                    chunk_length: Optional[int] = None) -> np.ndarray:
    """
    Drop-in replacement for Agent._solve_sdp using DynamicProgram + SolverOrchestrator.
    
    This function provides a seamless interface that matches Agent._solve_sdp's
    signature and return format, allowing existing Agent code to use the new
    DP framework without modification.
    
    Args:
        agent: Agent instance with _solve_sdp-compatible interface
        forecasts: List of forecast dictionaries for each time step
        start_index: Absolute time index for the first forecast
        chunk_length: Optional chunk length for long horizons (default: auto-select)
    
    Returns:
        policy_table: Policy table with shape (horizon, num_soc_levels)
                     policy_table[t, soc_idx] = action_idx (or -1 if infeasible)
                     Same format as Agent._solve_sdp output
    
    Example Usage:
        # In Agent.choose_action(), replace:
        # policy_table = self._solve_sdp(forecasts, start_index=current_step_env)
        
        # With:
        from src.dp_adapter import run_dp_for_agent
        policy_table = run_dp_for_agent(self, forecasts, current_step_env)
    """
    # Auto-select chunk length if not provided
    if chunk_length is None:
        horizon = len(forecasts)
        # Use chunking for horizons longer than 72 steps to manage memory
        chunk_length = 48 if horizon > 72 else horizon
    
    # Create adapter to bridge Agent and DynamicProgram interfaces
    adapter = AgentAdapter(agent, forecasts, start_index)
    
    # Use orchestrator to solve with chunking
    ctg_matrix, policy_table = orchestrate_solve(
        adapter, forecasts, chunk_length=chunk_length
    )
    
    # Return only the policy table to match Agent._solve_sdp interface
    return policy_table
