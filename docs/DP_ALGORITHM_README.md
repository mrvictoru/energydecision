# Dynamic Programming Algorithms: Technical Deep Dive & Guide

This document provides a comprehensive guide to the dynamic programming (DP) algorithms implemented in this project: **Stochastic Dynamic Programming (SDP)**, **Multi-Resolution Dynamic Programming (MRDP)**, and **Oracle** (Deterministic DP).

---

## 1. Overview

### The Problem
Previously, understanding how an algorithm worked required jumping between multiple files (`decision.py`, `algorithm_helpers.py`, Legacy `sdp_multires.py`, etc.). This made it difficult to debug, modify, or learn the complete algorithm flow.

### The Solution
Each algorithm is now self-contained in its own file within `src/`, with all logic localized and optimized.

### File Structure
```
src/
├── sdp_algorithm.py          # Complete SDP implementation
├── mrdp_algorithm.py         # Complete MRDP implementation
├── oracle_algorithm.py       # Complete Oracle implementation
├── decision.py               # Agent class (uses algorithm classes)
├── algorithm_helpers.py      # Shared utilities (DegradationCalculator)
├── quantile_scenarios.py     # Uncertainty modeling (Scenario Generator)
└── batterydeg.py            # Battery degradation models
```

---

## 2. Mathematical Foundation

All three algorithms solve a variation of the finite-horizon optimal control problem using the Bellman equation.

### The Objective
Minimize the total cost over a finite horizon $H$:
$$J = \sum_{t=0}^{H-1} C_t(s_t, u_t, w_t) + C_H(s_H)$$

Where:
*   $s_t$: State (Battery SoC) at time $t$
*   $u_t$: Action (Battery Charge/Discharge Power) at time $t$
*   $w_t$: Disturbance/Uncertainty (Solar Generation, Load, Prices)
*   $C_t$: Stage Cost (Grid Cost + Degradation Cost)
*   $C_H$: Terminal Cost (usually 0)

### The Bellman Equation (Backward Induction)
We compute the *Cost-to-Go* function $J_t(s_t)$ backwards from $t=H$ to $0$:

$$J_H(s) = 0$$
$$J_t(s) = \min_{u \in U(s)} \mathbb{E}_{w} \left[ C_t(s, u, w) + J_{t+1}(f(s, u)) \right]$$

Where $f(s, u)$ is the system dynamics (next SoC) and $\mathbb{E}_{w}$ is the expectation over uncertainty.

---

## 3. Algorithm Deep Dives

### A. Stochastic Dynamic Programming (SDP)
**File:** `src/sdp_algorithm.py`

Solves the control problem **under uncertainty**. It assumes we know the *distribution* of future values (via forecasts) but not the exact realization.

**How to Read/Logic:**
1.  **Initialize (`__init__`)**: Sets up discretization grids for SoC and Actions.
2.  **Backward Induction (`solve`)**: Iterates $t$ from $H-1 \to 0$. Use this order:
    *   **STEP 1**: Initialize cost-to-go and policy tables.
    *   **STEP 2**: Prepare scenario cache from `QuantileScenarioGenerator`.
    *   **STEP 3**: Backward induction loop.
        *   **STEP 3a**: Prepare Monte Carlo samples (if enabled).
        *   **STEP 3b**: Compute stage costs vectorized across all states and actions.
        *   **STEP 3c**: Interpolate future costs from the next step's value function.
        *   **STEP 3d**: Choose the optimal action index and update policy.

**Key Components:**
*   `_compute_stage_costs()`: Calculates Grid + Deg cost via NumPy broadcasting.
*   `_prepare_monte_carlo_samples()`: Uncertainty sampling from `QuantileScenarioGenerator`.

### B. Multi-Resolution Dynamic Programming (MRDP)
**File:** `src/mrdp_algorithm.py`

Addresses the **Curse of Dimensionality** by balancing accuracy and speed.

**Strategy:**
Divide horizon into sub-horizons:
*   **Near-term**: High resolution (e.g., 20 SoC levels, 41 actions) for accurate immediate decisions.
*   **Far-term**: Low resolution (e.g., 8 SoC levels, 17 actions) for computational efficiency.

**Logic Flow:**
1.  **Solve Backward**: Starts from the last sub-horizon (terminal cost = 0).
2.  **Propagate**: The first-stage Cost-to-Go of a sub-horizon becomes the **Terminal Cost** for the preceding one.
3.  **Return**: Returns the policy from the first sub-horizon for immediate action.

### C. Oracle (Perfect Foresight)
**File:** `src/oracle_algorithm.py`

Deterministic DP using **Perfect Information**. Identical to SDP but removes $\mathbb{E}_w$ by using actual historical data rows. Use this for benchmarking to see the maximum achievable performance ("theoretical upper bound").

---

## 4. Shared Implementation Details

### State/Action Discretization
*   **State Space ($S$)**: `soc_levels_kwh` grid. Clamped transitions.
*   **Action Space ($U$)**: `action_levels_norm` [-1, 1].

### Degradation Model (`src/algorithm_helpers.py`)
All solvers use the `DegradationCalculator` based on Muenzel et al. (2015).
*   **Nuance**: Standard Rainflow requires a full series; in DP, we treat transition $s_t \to s_{t+1}$ as a half-cycle event to estimate cost-to-go.

### Common Components
*   **Feasibility**: Invalid transitions (SoC < 0 or > Cap) are masked with `np.inf`.
*   **Interpolation**: `interpolate_ctg` estimates the future value between discrete SoC nodes.

| Aspect | SDP | MRDP | Oracle |
|--------|-----|------|--------|
| **Uncertainty** | Forecasts/Scenarios | Forecasts/Scenarios | Actual future values |
| **Resolution** | Single resolution | Multiple resolutions | Single resolution |
| **Computation** | Medium | Fast (coarse far-term) | Medium |
| **Goal** | Online Planning | Long Horizon Planning | Benchmark/Upper Bound |

---

## 5. Developer Guide: Debugging & Modification

### How the Agent Uses These Algorithms
The `Agent` class (`src/decision.py`) act as a thinner wrapper:
1.  Environment triggers `agent.choose_action(obs)`.
2.  Agent fetches forecasts via `_get_forecasts()`.
3.  Agent calls `solver.solve(forecasts)`.
4.  Agent extracts the optimal action index from the `policy_table` for the current SoC.

### How to Debug
*   **Trace Costs**: In `sdp_algorithm.py`, print `stage_costs.min()` and `stage_costs.max()` inside the loop.
*   **Policy Check**: If `policy_table` contains `-1`, those states are infeasible (check grid limits).
*   **Oracle Compare**: Compare your agent's performance against Oracle to see the "performance gap" due to uncertainty.

### How to Modify
*   **Change Horizon**: Pass `horizon=96` to the Agent for 48-hour planning.
*   **Change Resolution**: Increase `soc_resolution` (e.g., 20 $\to$ 50) for smoother control.
*   **Custom Sub-horizons (MRDP)**: Modify `subhorizon_specs` in `MRDPSolver` for custom time/resolution trade-offs.

---

## 6. References
- **Original Implementation**: Based on [khalida/optimal-energy-storage](https://github.com/khalida/optimal-energy-storage).
- **Battery Model**: Muenzel et al. (2015), "Multi-Factor Battery Cycle Life Prediction".
- **Dynamic Programming**: Bertsekas, "Dynamic Programming and Optimal Control".
