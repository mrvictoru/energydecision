# Multi-Resolution Dynamic Programming (MRDP) Module

This document describes the Multi-Resolution Dynamic Programming (MRDP) infrastructure for the energy decision project's SDP agent.

> **Module Location:** [`src/sdp_multires.py`](src/sdp_multires.py)

## Overview

MRDP enables efficient long-horizon optimization by using different temporal resolutions: fine-grained discretization for near-term decisions and coarser discretization for far-term planning.

## Files

### Core Module
- **[`src/sdp_multires.py`](src/sdp_multires.py)** - Main MRDP implementation containing:
  - `DynamicProgram` class for single sub-horizon optimization
  - `solve_mrdp()` orchestration function for multi-resolution solving
  - `vectorized_monte_carlo_stage_cost()` for efficient cost computation
  - `deterministic_stage_cost()` for fallback cost computation
  - Comprehensive docstrings and usage examples

### Example Scripts
- **[`src/mrdp_integration_example.py`](src/mrdp_integration_example.py)** - Complete integration example showing how to extend existing Agent
- **[`src/mrdp_validation_example.py`](src/mrdp_validation_example.py)** - Basic functionality tests and performance comparison
- **[`src/sdp_performance_benchmark.py`](src/sdp_performance_benchmark.py)** - Performance benchmarking script

## Key Features

### DynamicProgram Class
- **API Methods**:
  - `__init__(soc_levels_kwh, action_levels_kwh, step_duration, env, use_monte_carlo, mc_samples)`
  - `set_final_ctg(states_kwh, values)` - Set terminal cost-to-go from following sub-horizon
  - `get_first_stage_states_and_ctg()` - Extract first-stage values for previous sub-horizon  
  - `solve(forecasts_segment, start_index, stage_cost_function)` - Vectorized backward induction

- **Performance Optimizations**:
  - Preserves unique-values trick from existing `Agent._solve_sdp`
  - Uses `np.interp` for fast future cost interpolation
  - Vectorized feasibility masks and cost computations
  - Separates orchestration from stage-cost math via callback function

### solve_mrdp Function
- **Multi-Resolution Orchestration**:
  - Creates DynamicProgram instances for each sub-horizon based on specs
  - Solves backward: last sub-horizon first, then propagates terminal costs
  - Returns first sub-horizon policy table and cost-to-go for immediate decisions

- **Sub-horizon Specification Format**:
  ```python
  subhorizon_specs = [
      {
          'start': 0, 'length': 12,  # Time indices within forecasts
          'soc_resolution': 20, 'action_resolution': 11,  # Discretization levels
          'step_duration': 0.5  # Time step duration in hours
      },
      {
          'start': 12, 'length': 36,
          'soc_resolution': 10, 'action_resolution': 5, 
          'step_duration': 1.0
      }
  ]
  ```

## Integration with Existing Agent

### Non-Invasive Design
- **No changes to existing files** - only adds new module
- **Reuses Agent's stage cost logic** via `stage_cost_function` callback
- **Compatible with Agent's scenario_cache and Monte Carlo settings**
- **Easy to test alongside existing `Agent._solve_sdp`**

### Usage Pattern
```python
from src.sdp_multires import solve_mrdp

# Step 1: Create stage cost function that wraps Agent's logic
def create_agent_stage_cost_function(agent, forecasts):
    def stage_cost_function(t_global_idx, unique_energy_values):
        costs = np.empty(len(unique_energy_values))
        for i, energy in enumerate(unique_energy_values):
            battery_rate = energy / agent.step_duration
            rep_soc = agent.battery_capacity / 2.0
            costs[i] = agent._calculate_sdp_stage_cost(
                t_global_idx, rep_soc, battery_rate, energy, forecasts[t_global_idx]
            )
        return costs
    return stage_cost_function

# Step 2: Solve using MRDP
stage_cost_fn = create_agent_stage_cost_function(agent, forecasts)
policy_table, cost_to_go = solve_mrdp(
    env, forecasts, subhorizon_specs, global_start_index, stage_cost_fn
)

# Step 3: Extract t=0 action like existing Agent does
current_soc_idx = agent._soc_to_idx(env.battery_level)
action_idx = policy_table[0, current_soc_idx]
if action_idx >= 0:
    optimal_action = first_subhorizon_action_levels[action_idx]
```

## Performance Results

### Validation Tests Show:
- **1.47x-1.87x speedup** over single-horizon SDP
- **Consistent solution quality** - actions match within 0.0004 normalized units
- **All feasible states find optimal policies** - no infeasible results
- **Robust error handling** with fallbacks to rule-based actions

### Example Performance (48-step horizon):
```
Single-horizon: 1.684s (15 SoC × 9 actions × 48 steps)
Multi-resolution: 0.901s (15×9×16 + 8×5×32 steps)
Speedup: 1.87x
```

## Testing

### Run Example Scripts:
```bash
cd /path/to/energydecision

# MRDP validation example
python src/mrdp_validation_example.py

# Integration example
python src/mrdp_integration_example.py

# Performance benchmark
python src/sdp_performance_benchmark.py
```

### Run Automated Tests:
```bash
# Run all tests including SDP/agent tests
pytest tests/ -v

# Run agent-specific tests
pytest tests/test_decision_agent.py -v

# Run performance tests
pytest tests/test_performance.py -v -s
```

### Expected Output:
- ✓ All tests pass
- Performance improvements demonstrated
- Actions show good consistency between approaches

## Implementation Notes

### Python 3.8+ Compatible
- Uses typing annotations
- Compatible with existing numpy/polars dependencies
- No additional package requirements

### Code Style
- Follows existing project patterns
- Comprehensive docstrings with examples
- Clear separation of concerns
- Extensive inline comments for integration points

### Future Integration Path
1. **Test MRDP alongside existing SDP** using validation scripts
2. **Tune sub-horizon configurations** for specific use cases  
3. **Add as Agent algorithm option** (e.g., `algorithm='sdp_multires'`)
4. **Consider replacing single-horizon SDP** after validation

## Why Multi-Resolution DP?

### Benefits:
- **Computational Efficiency**: Coarse far-future reduces state space exponentially
- **Modeling Flexibility**: Different time granularities (30-min near-term, 1-hour far-term)
- **Solution Quality**: Fine resolution where decisions matter most (immediate actions)
- **Scalability**: Extends planning horizon without proportional cost increase

### Use Cases:
- **Long planning horizons** where uniform fine discretization is computationally prohibitive
- **Multi-timescale problems** where near-term and far-term decisions have different importance  
- **Real-time systems** requiring fast response with extended lookahead
- **Hierarchical optimization** with natural resolution boundaries

This MRDP infrastructure enables the team to progressively explore multi-resolution approaches for energy storage optimization while maintaining compatibility with the existing SDP agent framework.

## Related Documentation

- [README.md](README.md) - Main project documentation
- [README.scenario-support.md](README.scenario-support.md) - Scenario generation for uncertainty modeling
- [PERFORMANCE_IMPROVEMENTS.md](PERFORMANCE_IMPROVEMENTS.md) - Performance optimization details