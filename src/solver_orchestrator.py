"""
Solver Orchestrator for managing long-horizon dynamic programming problems.

This module provides functionality to split long horizons into manageable chunks,
solve each chunk independently, and stitch the results together via interpolation.
This approach enables solving problems with very long time horizons that would
otherwise be computationally prohibitive.

Usage Example:
    from src.solver_orchestrator import SolverOrchestrator, orchestrate_solve
    
    # Using the orchestrator class directly
    orchestrator = SolverOrchestrator(chunk_length=48)
    ctg_matrix, policy_table = orchestrator.solve(
        agent_like_interface, forecasts, start_index=0
    )
    
    # Using the convenience function
    ctg_matrix, policy_table = orchestrate_solve(
        agent_like_interface, forecasts, chunk_length=48
    )

Performance Notes:
    - Optimal chunk_length depends on state/action resolution and available memory
    - Typical values: 24-96 for half-hourly data (12-48 hours)
    - Each chunk is solved independently, enabling potential parallelization
    - CTG stitching uses linear interpolation between different state grids
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Any, Protocol
from dynamic_program import DynamicProgram


class AgentLikeInterface(Protocol):
    """Protocol defining the interface expected from agent-like objects."""
    
    soc_levels_kwh: np.ndarray
    action_levels_norm: np.ndarray
    
    def create_transition_fn(self) -> Callable:
        """Return transition function with signature (state, action, scenario) -> next_state."""
        ...
    
    def create_cost_fn(self) -> Callable:
        """Return cost function with signature (state, action, scenario_dict) -> cost."""
        ...
    
    def create_scenario_provider(self, forecasts: List[Dict], start_index: int) -> Callable:
        """Return scenario provider with signature (stage_index) -> (scenario_values, probs)."""
        ...


class SolverOrchestrator:
    """
    Orchestrates solving long-horizon problems by chunking and stitching.
    
    The orchestrator splits a long horizon into smaller chunks, solves each chunk
    from last to first, and uses interpolation to connect cost-to-go values
    between chunks with potentially different state grids.
    """
    
    def __init__(self, chunk_length: int = 48):
        """
        Initialize the solver orchestrator.
        
        Args:
            chunk_length: Number of time steps per chunk (default: 48)
        """
        self.chunk_length = chunk_length
    
    def solve(self, 
              agent_like: AgentLikeInterface,
              forecasts: List[Dict[str, Any]],
              start_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a long-horizon problem by chunking.
        
        Args:
            agent_like: Object implementing AgentLikeInterface
            forecasts: List of forecast dictionaries for each time step
            start_index: Absolute time index for the first forecast
        
        Returns:
            ctg_matrix: Cost-to-go matrix for the full horizon
            policy_table: Policy table for the full horizon
        """
        horizon = len(forecasts)
        
        if horizon <= self.chunk_length:
            # Single chunk - solve directly
            return self._solve_single_chunk(
                agent_like, forecasts, start_index
            )
        
        # Split into chunks
        chunks = self._create_chunks(forecasts, start_index)
        
        # Solve from last chunk to first, stitching CTG values
        final_ctg = None
        all_ctg_matrices = {}
        all_policy_tables = {}
        
        for chunk_idx in reversed(range(len(chunks))):
            chunk_forecasts, chunk_start_index = chunks[chunk_idx]
            
            # Create DP for this chunk
            dp = self._create_dynamic_program(
                agent_like, chunk_forecasts, chunk_start_index
            )
            dp.initialize_states()
            
            # Set final CTG from next chunk (if any)
            if final_ctg is not None:
                # Interpolate final CTG from next chunk's state grid to current grid
                interpolated_ctg = self._interpolate_ctg_between_grids(
                    final_ctg, agent_like.soc_levels_kwh, agent_like.soc_levels_kwh
                )
                dp.set_final_ctg(interpolated_ctg)
            else:
                # Last chunk - use zero final CTG
                dp.set_final_ctg(np.zeros(len(agent_like.soc_levels_kwh)))
            
            # Solve this chunk
            ctg_matrix, policy_table = dp.solve()
            
            # Store results
            all_ctg_matrices[chunk_idx] = ctg_matrix
            all_policy_tables[chunk_idx] = policy_table
            
            # Update final CTG for next (previous) chunk
            final_ctg = ctg_matrix[0, :]  # Initial CTG of current chunk
        
        # Stitch results together
        return self._stitch_results(all_ctg_matrices, all_policy_tables, chunks)
    
    def _create_chunks(self, forecasts: List[Dict], start_index: int) -> List[Tuple[List[Dict], int]]:
        """Split forecasts into chunks with their absolute start indices."""
        chunks = []
        horizon = len(forecasts)
        
        for i in range(0, horizon, self.chunk_length):
            end_idx = min(i + self.chunk_length, horizon)
            chunk_forecasts = forecasts[i:end_idx]
            chunk_start_index = start_index + i
            chunks.append((chunk_forecasts, chunk_start_index))
        
        return chunks
    
    def _create_dynamic_program(self,
                               agent_like: AgentLikeInterface,
                               forecasts: List[Dict],
                               start_index: int) -> DynamicProgram:
        """Create a DynamicProgram instance for the given chunk."""
        horizon = len(forecasts)
        stage_times = list(range(start_index, start_index + horizon))
        
        transition_fn = agent_like.create_transition_fn()
        cost_fn = agent_like.create_cost_fn()
        scenario_provider = agent_like.create_scenario_provider(forecasts, start_index)
        
        return DynamicProgram(
            state_grid=agent_like.soc_levels_kwh,
            action_levels=agent_like.action_levels_norm,
            transition_fn=transition_fn,
            cost_fn=cost_fn,
            scenario_provider=scenario_provider,
            stage_times=stage_times
        )
    
    def _solve_single_chunk(self,
                           agent_like: AgentLikeInterface,
                           forecasts: List[Dict],
                           start_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Solve a single chunk directly."""
        dp = self._create_dynamic_program(agent_like, forecasts, start_index)
        dp.initialize_states()
        dp.set_final_ctg(np.zeros(len(agent_like.soc_levels_kwh)))
        return dp.solve()
    
    def _interpolate_ctg_between_grids(self,
                                     ctg_values: np.ndarray,
                                     source_grid: np.ndarray,
                                     target_grid: np.ndarray) -> np.ndarray:
        """Interpolate CTG values from source grid to target grid."""
        if np.array_equal(source_grid, target_grid):
            return ctg_values.copy()
        
        # Use linear interpolation
        return np.interp(target_grid, source_grid, ctg_values)
    
    def _stitch_results(self,
                       ctg_matrices: Dict[int, np.ndarray],
                       policy_tables: Dict[int, np.ndarray],
                       chunks: List[Tuple[List[Dict], int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Stitch chunk results into full horizon matrices."""
        # Calculate total horizon
        total_horizon = sum(len(chunk_forecasts) for chunk_forecasts, _ in chunks)
        num_states = ctg_matrices[0].shape[1]
        
        # Initialize full matrices
        full_ctg_matrix = np.full((total_horizon + 1, num_states), np.inf)
        full_policy_table = np.full((total_horizon, num_states), -1, dtype=int)
        
        # Copy chunk results into full matrices
        time_offset = 0
        for chunk_idx in range(len(chunks)):
            chunk_forecasts, _ = chunks[chunk_idx]
            chunk_horizon = len(chunk_forecasts)
            
            ctg_matrix = ctg_matrices[chunk_idx]
            policy_table = policy_tables[chunk_idx]
            
            # Copy CTG values (excluding final for all but last chunk)
            end_ctg_idx = chunk_horizon + 1 if chunk_idx == len(chunks) - 1 else chunk_horizon
            full_ctg_matrix[time_offset:time_offset + end_ctg_idx, :] = ctg_matrix[:end_ctg_idx, :]
            
            # Copy policy values
            full_policy_table[time_offset:time_offset + chunk_horizon, :] = policy_table
            
            time_offset += chunk_horizon
        
        return full_ctg_matrix, full_policy_table


def orchestrate_solve(agent_like: AgentLikeInterface,
                     forecasts: List[Dict[str, Any]],
                     chunk_length: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to orchestrate solving with chunking.
    
    This function demonstrates how Agent can call the orchestrator with minimal
    adaptation. The agent should provide methods to create transition functions,
    cost functions, and scenario providers that work with the forecasts.
    
    Args:
        agent_like: Object implementing AgentLikeInterface with:
            - soc_levels_kwh: State grid array
            - action_levels_norm: Action levels array  
            - create_transition_fn(): Returns transition function
            - create_cost_fn(): Returns cost function
            - create_scenario_provider(forecasts, start_index): Returns scenario provider
        forecasts: List of forecast dictionaries for each time step
        chunk_length: Number of time steps per chunk (default: 48)
    
    Returns:
        ctg_matrix: Cost-to-go matrix for the full horizon
        policy_table: Policy table for the full horizon
    
    Example Agent Integration:
        ```python
        # In Agent class, add these methods:
        def create_transition_fn(self):
            def transition(state, action, scenario):
                battery_energy = action * self.max_battery_flow * self.step_duration
                return np.clip(state + battery_energy, 0, self.battery_capacity)
            return transition
        
        def create_cost_fn(self):
            def cost(state_kwh, action_norm, scenario_values):
                # Compute grid cost + degradation cost using scenario_values
                # scenario_values: {'solar': float, 'load': float, 'import_price': float, 'export_price': float}
                return self._calculate_sdp_stage_cost(0, state_kwh, action_norm * self.max_battery_flow, ...)
            return cost
        
        def create_scenario_provider(self, forecasts, start_index):
            def provider(stage_index):
                # Use self._scenario_cache or self.scenario_generator
                # Return (scenario_values_dict, scenario_probs)
                return scenario_values, scenario_probs
            return provider
        
        # Replace _solve_sdp call with:
        from src.dp_adapter import run_dp_for_agent
        policy_table = run_dp_for_agent(self, forecasts, start_index, chunk_length=48)
        ```
    """
    orchestrator = SolverOrchestrator(chunk_length=chunk_length)
    return orchestrator.solve(agent_like, forecasts, start_index=0)
