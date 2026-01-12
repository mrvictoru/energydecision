# Algorithm Implementation Guide

## Overview

This guide explains how the SDP, MRDP, and Oracle algorithms are implemented in a self-contained, easy-to-understand manner.

## Problem

**Before**: To understand how an algorithm works, you had to jump between multiple files:
- `decision.py` - Agent class and choose_action logic
- `algorithm_helpers.py` - Degradation calculations
- `sdp_multires.py` - Multi-resolution DP infrastructure
- `quantile_scenarios.py` - Uncertainty modeling

This made it difficult to:
- Understand the complete algorithm flow
- Debug issues
- Modify algorithm behavior
- Learn how the algorithms work

## Solution

**After**: Each algorithm is self-contained in its own file with all logic in one place.

## File Structure

```
src/
├── sdp_algorithm.py          # Complete SDP implementation
├── mrdp_algorithm.py         # Complete MRDP implementation
├── oracle_algorithm.py       # Complete Oracle implementation
├── decision.py               # Agent class (uses algorithm classes)
├── algorithm_helpers.py      # Shared utilities (DegradationCalculator)
├── quantile_scenarios.py     # Uncertainty modeling
└── batterydeg.py            # Battery degradation models
```

## How to Read Each Algorithm

### 1. Stochastic Dynamic Programming (SDP)

**File**: `src/sdp_algorithm.py`

**Read in this order**:

1. **Module docstring** (top of file) - Explains algorithm overview
2. **`SDPSolver.__init__()`** - See what parameters control the algorithm
3. **`SDPSolver.solve()`** - Main algorithm flow with clear steps:
   ```python
   # STEP 1: Initialize cost-to-go and policy tables
   # STEP 2: Prepare scenario cache for uncertainty
   # STEP 3: Backward induction - solve from last time step to first
   #   STEP 3a: Prepare Monte Carlo samples
   #   STEP 3b: Compute stage costs
   #   STEP 3c: Compute future costs
   #   STEP 3d: Find optimal action
   ```
4. **Helper methods** - Each does one clear task:
   - `_prepare_monte_carlo_samples()` - Uncertainty sampling
   - `_compute_stage_costs()` - Grid cost + degradation cost
   - `_compute_future_costs()` - Interpolate next time step
   - `_update_policy()` - Find optimal action for each state

**Algorithm Summary**:
```
For t = horizon-1 down to 0:
    For each battery state (SoC level):
        For each possible action (battery flow):
            1. Check feasibility (battery/grid constraints)
            2. Compute stage cost (grid cost + degradation cost)
            3. Compute future cost (from next time step)
            4. Total cost = stage cost + future cost
        Choose action that minimizes total cost
```

### 2. Multi-Resolution Dynamic Programming (MRDP)

**File**: `src/mrdp_algorithm.py`

**Read in this order**:

1. **Module docstring** - Explains multi-resolution strategy
2. **`MRDPSolver.__init__()`** - See sub-horizon specifications
3. **`MRDPSolver.solve()`** - Main flow:
   ```python
   # STEP 1: Solve sub-horizons backward (last to first)
   #   STEP 1a: Set terminal cost from next sub-horizon
   #   STEP 1b: Solve this sub-horizon
   # STEP 2: Return policy from first sub-horizon
   ```

**Algorithm Summary**:
```
Divide horizon into sub-horizons:
    Near-term: High resolution (20 SoC levels, 41 actions)
    Far-term: Low resolution (8 SoC levels, 17 actions)

Solve backward (last to first):
    Last sub-horizon: Use zero terminal cost
    Earlier sub-horizons: Use next sub-horizon's cost as terminal cost

Return: Policy from first sub-horizon (for immediate action)
```

**Key Advantage**: Faster computation while maintaining accuracy for near-term decisions.

### 3. Oracle Algorithm

**File**: `src/oracle_algorithm.py`

**Read in this order**:

1. **Module docstring** - Explains perfect information approach
2. **`OracleSolver.__init__()`** - See parameters
3. **`OracleSolver.solve()`** - Similar to SDP but uses actual future values:
   ```python
   # STEP 1: Initialize
   # STEP 2: Backward induction with actual future values
   #   STEP 2a: Get actual future values (not forecasts!)
   #   STEP 2b: Compute feasibility
   #   STEP 2c: Compute grid energy
   #   STEP 2d: Compute grid cost
   #   STEP 2e: Compute degradation cost
   #   STEP 2f: Compute future costs
   #   STEP 2g: Find optimal actions
   ```

**Algorithm Summary**:
```
Same as SDP, but:
- No uncertainty (uses actual future values)
- No Monte Carlo sampling
- Provides theoretical upper bound on performance
```

## How the Agent Uses These Algorithms

**File**: `src/decision.py`

The Agent class is now much simpler:

```python
class Agent:
    def __init__(self, env, algorithm='sdp', ...):
        if algorithm == 'sdp':
            self.sdp_solver = SDPSolver(env, horizon, ...)
        elif algorithm == 'mrdp':
            self.mrdp_solver = MRDPSolver(env, subhorizon_specs, ...)
        elif algorithm == 'oracle':
            self.oracle_solver = OracleSolver(env, horizon, ...)
    
    def choose_action(self, obs):
        if self.algorithm == 'sdp':
            forecasts = self._get_forecasts(...)
            policy_table = self.sdp_solver.solve(forecasts, ...)
            return self._extract_action(policy_table, current_soc)
        # Similar for MRDP and Oracle
```

## Common Algorithm Components

All three algorithms share:

1. **Dynamic Programming Structure**:
   - Initialize cost-to-go with terminal cost (usually zero)
   - Backward induction (solve from last time step to first)
   - At each step: minimize stage cost + future cost

2. **Stage Cost Computation**:
   ```
   Stage Cost = Grid Cost + Degradation Cost
   
   Grid Cost:
     - Import: positive cost (pay for energy)
     - Export: negative cost (receive payment)
     - Violation: infinite cost (exceeds grid limits)
   
   Degradation Cost:
     - Based on battery throughput (kWh)
     - Uses DegradationCalculator from algorithm_helpers.py
     - Accounts for C-rate, SoC, DoD, temperature
   ```

3. **Feasibility Constraints**:
   - Battery: Can't discharge below 0% or charge above 100%
   - Grid: Import/export must be within grid connection limits

## Differences Between Algorithms

| Aspect | SDP | MRDP | Oracle |
|--------|-----|------|--------|
| **Uncertainty** | Uses forecasts/scenarios | Uses forecasts/scenarios | Uses actual future values |
| **Resolution** | Single resolution | Multiple resolutions | Single resolution |
| **Computation** | Medium | Fast (coarse far-term) | Medium |
| **Accuracy** | Good | Good (near-term) | Perfect (theoretical bound) |
| **Use Case** | General purpose | Long horizons | Benchmarking |

## How to Debug

### SDP Issues

1. Check `sdp_algorithm.py:SDPSolver.solve()`
2. Add print statements in backward induction loop:
   ```python
   for t in range(horizon - 1, -1, -1):
       print(f"Solving time step {t}")
       # ... rest of code
   ```
3. Inspect stage costs:
   ```python
   stage_costs = self._compute_stage_costs(...)
   print(f"Stage costs range: [{stage_costs.min()}, {stage_costs.max()}]")
   ```

### MRDP Issues

1. Check `mrdp_algorithm.py:MRDPSolver.solve()`
2. Print sub-horizon info:
   ```python
   for i, spec in enumerate(self.subhorizon_specs):
       print(f"Sub-horizon {i}: {spec}")
   ```
3. Check terminal cost propagation:
   ```python
   if i < num_subhorizons - 1:
       print(f"Terminal cost from sub-horizon {i+1}")
   ```

### Oracle Issues

1. Check `oracle_algorithm.py:OracleSolver.solve()`
2. Verify actual values are being used:
   ```python
   row = self.env._get_row(start_index + t)
   print(f"Actual values at t={t}: solar={row['SolarGen']}, load={row['HouseLoad']}")
   ```

## How to Modify

### Change SDP Horizon

```python
agent = Agent(env, algorithm='sdp', horizon=96)  # 96 steps = 48 hours
```

### Change MRDP Resolution

```python
subhorizon_specs = [
    {
        'start': 0, 'length': 24,  # First 24 steps
        'soc_resolution': 30,      # High resolution
        'action_resolution': 51,   
        'step_duration': 0.5
    },
    {
        'start': 24, 'length': 72, # Next 72 steps
        'soc_resolution': 10,      # Low resolution
        'action_resolution': 11,
        'step_duration': 1.0
    }
]
agent = Agent(env, algorithm='mrdp', subhorizon_specs=subhorizon_specs)
```

### Change Degradation Model

```python
# Linear (fastest)
agent = Agent(env, algorithm='sdp', degradation_model='linear')

# Rainflow counting (most accurate)
agent = Agent(env, algorithm='sdp', degradation_model='rainflow')
```

## Testing

Each algorithm class can be tested independently:

```python
from sdp_algorithm import SDPSolver

# Create solver
solver = SDPSolver(env, horizon=48, soc_resolution=20, action_resolution=41)

# Prepare test forecasts
forecasts = [
    {'SolarGen': 1.0, 'HouseLoad': 1.5, 'ImportEnergyPrice': 0.25, 'ExportEnergyPrice': 0.1},
    # ... more forecasts
]

# Solve
policy_table = solver.solve(forecasts, start_index=0)

# Check results
print(f"Policy shape: {policy_table.shape}")
print(f"Infeasible states: {(policy_table == -1).sum()}")
```

## References

- **SDP**: Bertsekas, "Dynamic Programming and Optimal Control"
- **MRDP**: Multi-resolution extension for computational efficiency
- **Oracle**: Theoretical upper bound with perfect information
- **Battery Model**: Muenzel et al. (2015), "Multi-Factor Battery Cycle Life Prediction"
- **Original Implementation**: https://github.com/khalida/optimal-energy-storage

## Summary

The refactored code makes it easy to:
- ✅ Understand each algorithm by reading one file
- ✅ Debug by following execution flow in one place
- ✅ Modify algorithm parameters
- ✅ Test algorithms independently
- ✅ Learn how the algorithms work

No more jumping between files to understand the algorithm flow!
