"""
Multi-Resolution Dynamic Programming (MRDP) Algorithm

This module implements a self-contained MRDP solver that uses multiple resolution
levels to balance accuracy and computational efficiency.

Algorithm Overview:
-------------------
1. Divide horizon into sub-horizons with different resolutions:
   - Near-term: High resolution (many states/actions) for accurate immediate decisions
   - Far-term: Low resolution (fewer states/actions) for computational efficiency

2. Solve sub-horizons backward (last to first):
   - Last sub-horizon: Use zero terminal cost
   - Earlier sub-horizons: Use next sub-horizon's first-stage cost-to-go as terminal cost

3. Extract policy from first sub-horizon for immediate action

Key Advantage: Better computational efficiency while maintaining accuracy for near-term decisions

Reference: Based on khalida/optimal-energy-storage with multi-resolution extension
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from sdp_algorithm import SDPSolver


class MRDPSolver:
    """
    Self-contained Multi-Resolution Dynamic Programming solver.
    
    This class uses multiple SDPSolver instances with different resolutions
    to efficiently solve long-horizon problems.
    """
    
    def __init__(self,
                 env: Any,
                 subhorizon_specs: List[Dict],
                 use_monte_carlo: bool = False,
                 mc_samples: int = 100,
                 mc_seed: Optional[int] = None,
                 static_deg_correction_factor: float = 0.8,
                 scenario_generator: Optional[Any] = None):
        """
        Initialize MRDP solver.
        
        Args:
            env: Environment with battery and grid parameters
            subhorizon_specs: List of dicts, each specifying a sub-horizon:
                - 'start': Starting time index
                - 'length': Number of time steps
                - 'soc_resolution': Number of SoC levels
                - 'action_resolution': Number of action levels
                - 'step_duration': Time step duration
            use_monte_carlo: Whether to use Monte Carlo
            mc_samples: Number of Monte Carlo samples
            mc_seed: Random seed
            static_deg_correction_factor: Degradation correction factor
            scenario_generator: Optional scenario generator
        """
        self.env = env
        self.subhorizon_specs = subhorizon_specs or self._default_subhorizon_specs()
        self.use_monte_carlo = use_monte_carlo
        self.mc_samples = mc_samples
        self.mc_seed = mc_seed
        self.static_deg_correction_factor = static_deg_correction_factor
        self.scenario_generator = scenario_generator
        
        # Create SDP solver for each sub-horizon
        self.sub_solvers = []
        for spec in self.subhorizon_specs:
            solver = SDPSolver(
                env=env,
                horizon=spec['length'],
                soc_resolution=spec['soc_resolution'],
                action_resolution=spec['action_resolution'],
                use_monte_carlo=use_monte_carlo,
                mc_samples=mc_samples,
                mc_seed=mc_seed,
                static_deg_correction_factor=static_deg_correction_factor,
                scenario_generator=scenario_generator
            )
            # Override step duration for this sub-horizon
            solver.step_duration = spec['step_duration']
            solver.battery_flow_energies = solver.action_levels_norm * env.max_battery_flow * spec['step_duration']
            self.sub_solvers.append(solver)
    
    def _default_subhorizon_specs(self) -> List[Dict]:
        """Default sub-horizon specification: near-term fine, far-term coarse."""
        return [
            {
                'start': 0, 
                'length': 12, 
                'soc_resolution': 20, 
                'action_resolution': 41, 
                'step_duration': self.env.step_duration
            },
            {
                'start': 12, 
                'length': 36, 
                'soc_resolution': 8, 
                'action_resolution': 17, 
                'step_duration': max(self.env.step_duration, 2 * self.env.step_duration)
            },
        ]
    
    def solve(self, forecasts: List[Dict], start_index: int = 0) -> np.ndarray:
        """
        Solve MRDP problem.
        
        Algorithm:
        1. Solve sub-horizons backward (last to first)
        2. Propagate terminal costs between sub-horizons
        3. Return policy from first sub-horizon
        
        Args:
            forecasts: List of forecast dicts
            start_index: Global time index for first forecast
        
        Returns:
            policy_table: Optimal actions from first sub-horizon
        """
        num_subhorizons = len(self.subhorizon_specs)
        cost_to_go_tables = []
        policy_tables = []
        soc_levels_list = []
        
        # STEP 1: Solve sub-horizons backward (last to first)
        for i in range(num_subhorizons - 1, -1, -1):
            spec = self.subhorizon_specs[i]
            solver = self.sub_solvers[i]
            
            # Extract forecast segment for this sub-horizon
            forecast_start = spec['start']
            forecast_end = forecast_start + spec['length']
            forecasts_segment = forecasts[forecast_start:forecast_end]
            
            if not forecasts_segment:
                raise ValueError(f"Sub-horizon {i} has empty forecast segment")
            
            # Global time index for this sub-horizon
            sub_start_index = start_index + forecast_start
            
            # STEP 1a: Set terminal cost for this sub-horizon
            if i < num_subhorizons - 1:
                # Use next sub-horizon's first-stage cost-to-go as terminal cost
                next_soc_levels = soc_levels_list[-1]
                next_ctg = cost_to_go_tables[-1][0, :]
                
                # Interpolate to current sub-horizon's SoC discretization
                terminal_ctg = np.interp(
                    solver.soc_levels_kwh,
                    next_soc_levels,
                    next_ctg
                )
            else:
                # Last sub-horizon: zero terminal cost
                terminal_ctg = np.zeros(len(solver.soc_levels_kwh))
            
            # STEP 1b: Solve this sub-horizon with custom terminal cost
            policy_table = self._solve_subhorizon_with_terminal_cost(
                solver, forecasts_segment, sub_start_index, terminal_ctg
            )
            
            # Store results (in reverse order since we're going backward)
            cost_to_go_tables.insert(0, solver._last_cost_to_go)
            policy_tables.insert(0, policy_table)
            soc_levels_list.insert(0, solver.soc_levels_kwh)
        
        # STEP 2: Return policy from first sub-horizon
        return policy_tables[0]
    
    def _solve_subhorizon_with_terminal_cost(self, solver: SDPSolver, 
                                             forecasts: List[Dict],
                                             start_index: int,
                                             terminal_ctg: np.ndarray) -> np.ndarray:
        """
        Solve a sub-horizon with custom terminal cost.
        
        This is similar to standard SDP but uses provided terminal cost
        instead of zero.
        """
        num_soc_levels = len(solver.soc_levels_kwh)
        horizon = len(forecasts)
        
        # Initialize
        cost_to_go = np.full((horizon + 1, num_soc_levels), np.inf)
        policy_table = np.full((horizon, num_soc_levels), -1, dtype=int)
        cost_to_go[horizon, :] = terminal_ctg  # Use provided terminal cost
        
        # Prepare scenario cache
        if solver._scenario_cache is None and solver.scenario_generator is not None:
            try:
                solver._scenario_cache = solver.scenario_generator.generate_time_step_scenarios(solver.env.df)
            except Exception:
                solver._scenario_cache = None
        
        # Backward induction
        for t in range(horizon - 1, -1, -1):
            forecast_step = forecasts[t]
            row_idx = start_index + t
            
            monte_samples = solver._prepare_monte_carlo_samples(row_idx, t)
            stage_costs = solver._compute_stage_costs(forecast_step, row_idx, monte_samples)
            future_costs = solver._compute_future_costs(cost_to_go[t + 1, :])
            solver._update_policy(t, stage_costs, future_costs, cost_to_go, policy_table)
        
        # Store cost-to-go for terminal cost propagation
        solver._last_cost_to_go = cost_to_go
        
        return policy_table
