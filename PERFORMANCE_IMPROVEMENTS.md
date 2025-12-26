# Performance Improvements Analysis

This document identifies slow or inefficient code patterns in the energydecision repository and documents the improvements that have been implemented.

## Overview

After comprehensive analysis of the codebase, the following performance bottlenecks and inefficiencies were identified and optimized:

---

## 1. EnergySimEnv.py - Environment Initialization ✅ IMPLEMENTED

### Issue: Inefficient min/max computation in `__init__`

**Location:** Lines 113-114

**Previous Code:**
```python
self.df_mins_for_obs = np.array([self.df.select(pl.min(col)).item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
self.df_maxs_for_obs = np.array([self.df.select(pl.max(col)).item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
```

**Problem:** Made N separate queries to the DataFrame (one per column), which was inefficient.

**Implemented Fix:** Single Polars aggregation query that computes all min/max values at once:
```python
# Compute min/max in a single aggregation query for efficiency
agg_exprs = []
for col in self.ordered_df_cols_for_obs:
    agg_exprs.extend([pl.min(col).alias(f"{col}_min"), pl.max(col).alias(f"{col}_max")])
stats = self.df.select(agg_exprs)
self.df_mins_for_obs = np.array([stats[f"{col}_min"].item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
self.df_maxs_for_obs = np.array([stats[f"{col}_max"].item() for col in self.ordered_df_cols_for_obs], dtype=np.float32)
```

**Impact:** Reduces DataFrame scans from 2N to 1.

---

## 2. EnergySimEnv.py - Observation Normalization ✅ IMPLEMENTED

### Issue: Loop-based normalization in `_get_observation_components`

**Location:** Lines 193-200

**Previous Code:**
```python
norm_by_capacity_cols = {"SolarGen", "HouseLoad", "FutureSolar", "FutureLoad"}
normalized_df_values = []
for i, col in enumerate(self.ordered_df_cols_for_obs):
    if col in norm_by_capacity_cols:
        normalized_df_values.append(raw_df_values[i] / (self.battery_capacity + 1e-9))
    else:
        normalized_df_values.append((raw_df_values[i] - self.df_mins_for_obs[i]) / self.df_ranges_for_obs[i])
normalized_df_values = np.array(normalized_df_values, dtype=np.float32)
```

**Problem:** Python loop was slower than vectorized numpy operations for per-step normalization.

**Implemented Fix:** Pre-computed normalization mask during `__init__` and vectorized numpy operations:
```python
# In __init__:
self._norm_by_capacity_mask = np.array([col in norm_by_capacity_cols for col in self.ordered_df_cols_for_obs])

# In _get_observation_components:
normalized_df_values = np.where(
    self._norm_by_capacity_mask,
    raw_df_values / (self.battery_capacity + 1e-9),
    (raw_df_values - self.df_mins_for_obs) / self.df_ranges_for_obs
).astype(np.float32)
```

**Impact:** Eliminates Python loop overhead in the hot path (called every step).

---

## 3. quantile_scenarios.py - Global Quantile Computation ✅ IMPLEMENTED

### Issue: Repeated single-column queries in `generate_time_step_scenarios`

**Location:** Lines 395-397

**Previous Code:**
```python
for vname, col in variables.items():
    qs = var_quantiles[vname]
    values = np.array([df.select(pl.col(col).quantile(q)).to_series().item() for q in qs], dtype=float)
```

**Problem:** Made Q separate queries per variable (4 * Q total queries).

**Implemented Fix:** Batched all quantile computations in a single query:
```python
# Build all quantile expressions at once
quantile_exprs = []
for vname, col in variables.items():
    for i, q in enumerate(var_quantiles[vname]):
        quantile_exprs.append(pl.col(col).quantile(q).alias(f"{vname}_q{i}"))

# Single query
quantiles_result = df.select(quantile_exprs)
```

**Impact:** Reduces from 4*Q queries to 1 query.

---

## 4. decision.py - SDP Stage Cost Computation ✅ IMPLEMENTED

### Issue: Redundant scenario cache initialization checks

**Location:** Lines 665-670

**Previous Code:**
```python
# Inside _calculate_sdp_stage_cost:
if self._scenario_cache is None:
    try:
        self._scenario_cache = self.scenario_generator.generate_time_step_scenarios(self.env.df)
    except Exception:
        self._scenario_cache = None
```

**Problem:** This check was performed inside the method, adding overhead on every call.

**Implemented Fix:** Removed the redundant check since `_solve_sdp` already initializes the cache:
```python
def _calculate_sdp_stage_cost(self, row_idx, soc_kwh, battery_flow_rate, battery_flow_energy, forecast_step):
    """
    Note: Assumes self._scenario_cache is already initialized by _solve_sdp() before calling this method.
    """
    # Directly use self._scenario_cache without re-initialization check
```

**Impact:** Eliminates redundant checks inside time-critical loops.

---

## 5. helper.py - DataFrame Transformation ✅ IMPLEMENTED

### Issue: Multiple separate column operations in `transform_polars_df`

**Location:** Lines 107-118

**Previous Code:**
```python
pivot = pivot.with_columns(pl.col("GG").fill_null(0.0).alias("SolarGen"))
if "CL" not in pivot.columns:
    pivot = pivot.with_columns(pl.lit(0.0).alias("CL"))
pivot = pivot.with_columns([(pl.col("GC").fill_null(0.0) + pl.col("CL").fill_null(0.0)).alias("HouseLoad")])
```

**Problem:** Multiple calls to `with_columns` created intermediate DataFrames.

**Implemented Fix:** Batched column operations:
```python
new_cols = [pl.col("GG").fill_null(0.0).alias("SolarGen")]
if "CL" not in pivot.columns:
    new_cols.append(pl.lit(0.0).alias("CL"))
pivot = pivot.with_columns(new_cols)
pivot = pivot.with_columns([(pl.col("GC").fill_null(0.0) + pl.col("CL").fill_null(0.0)).alias("HouseLoad")])
```

**Impact:** Reduces intermediate DataFrame allocations.

---

## 6. batterydeg.py - Numerical Computation ✅ IMPLEMENTED

### Issue: Lambda recreation and repeated constant calculations

**Location:** Lines 14-33

**Previous Code:**
```python
def nCL_Id(Id):
    e, f, g, h = 4464.0, -0.1382, -1519, -0.4305
    return (e * math.exp(f * Id) + g * math.exp(h * Id)) / (e * math.exp(f * Id_nom) + g * math.exp(h * Id_nom))

def nCL_SoC_DoD(SoC, DoD):
    CL4 = lambda DoD, SoC: ...  # Lambda created on every call
    num = _ensure_positive(CL4(DoD, SoC))
    den = _ensure_positive(CL4(DoD_nom, SoC_nom))  # Constant denominator computed on every call
    return num / den
```

**Problem:** Lambda function was recreated on every call and constant denominators were recomputed.

**Implemented Fix:** Pre-compute all constant denominators at module load time:
```python
# Pre-compute nominal denominators (these are constants)
_nCL_Id_nom_denom = 4464.0 * math.exp(-0.1382 * Id_nom) + (-1519) * math.exp(-0.4305 * Id_nom)
_nCL_Ich_nom_denom = 5963.0 * math.exp(-0.6531 * Ich_nom) + 321.4 * math.exp(0.03168 * Ich_nom)

# Helper function defined once (not as lambda)
def _CL4(DoD, SoC):
    q, s, t, u, v = 1471.0, 214.3, 0.6111, 0.3369, -2.295
    return q + (20.0 * (s + 100.0 * u) - 200.0 * t) * DoD + s * SoC + t * DoD**2 + u * DoD * SoC + v * SoC**2

# Pre-compute nominal denominator
_nCL_SoC_DoD_nom_denom = _ensure_positive(_CL4(DoD_nom, SoC_nom))
```

**Impact:** Avoids lambda creation overhead and eliminates repeated constant computations.

---

## Summary of Implemented Changes

| File | Issue | Status | Impact |
|------|-------|--------|--------|
| EnergySimEnv.py | Multiple min/max queries | ✅ Done | Medium |
| EnergySimEnv.py | Loop-based normalization | ✅ Done | High (hot path) |
| quantile_scenarios.py | Repeated quantile queries | ✅ Done | Medium |
| decision.py | Redundant cache checks | ✅ Done | Low |
| helper.py | Multiple with_columns calls | ✅ Done | Low |
| batterydeg.py | Lambda recreation/constants | ✅ Done | Low |

## Benchmarking

Performance benchmarks are included in the test suite. Run with:

```bash
# Run performance tests with output
pytest tests/test_performance.py -v -s

# Run all tests
pytest tests/ -v
```

### Standalone Benchmark Script

For detailed MRDP performance analysis:

```bash
python src/sdp_performance_benchmark.py
```

Example output:
```
Environment observation benchmark (1000 calls):
  Total time: 0.0380s
  Per-call time: 0.0380ms

Degradation calculation benchmark (10000 calls):
  Total time: 0.0316s
  Per-call time: 0.0032ms

Quantile scenario generation benchmark (1000 rows, 5 scenarios):
  Total time: 0.0006s
```

## Test Suite Organization

The test suite has been reorganized for better maintainability:

### Directory Structure
```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Shared pytest fixtures
├── test_environment.py      # SolarBatteryEnv tests
├── test_decision_agent.py   # Agent/SDP/Oracle tests
├── test_performance.py      # Performance benchmarks
└── test_quantile_scenarios.py  # QuantileScenarioGenerator tests
```

### Test Categories

| Test File | Purpose | Test Count |
|-----------|---------|------------|
| test_environment.py | Environment functionality, observation handling | 9 |
| test_decision_agent.py | SDP solver, Oracle agent, policy computation | 8 |
| test_performance.py | Performance benchmarks and optimization validation | 8 |
| test_quantile_scenarios.py | Quantile scenario generation | 21 |

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run performance benchmarks with output
pytest tests/test_performance.py -v -s

# Run with timing info
pytest tests/ -v --durations=10
```

### Example Scripts (in src/)

The following scripts are standalone examples/benchmarks (not pytest tests):
- `src/sdp_performance_benchmark.py` - MRDP performance comparison
- `src/mrdp_validation_example.py` - MRDP vs single-horizon SDP comparison

## Test Results

All 46 tests pass after optimizations:
- `tests/test_environment.py`: 9/9 passed
- `tests/test_decision_agent.py`: 8/8 passed  
- `tests/test_performance.py`: 8/8 passed
- `tests/test_quantile_scenarios.py`: 21/21 passed

## Related Documentation

- [README.md](README.md) - Main project documentation
- [MRDP_README.md](MRDP_README.md) - Multi-Resolution Dynamic Programming documentation
- [README.scenario-support.md](README.scenario-support.md) - Scenario generation for uncertainty modeling

