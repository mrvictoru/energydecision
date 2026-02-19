# AEMO Battery Degradation Integration Plan

## Summary

This document describes the integration of physics-based battery degradation modelling into the `AEMOBatteryTradingEnv` environment, and evaluates the suitability of the existing rainflow counting approach for grid-scale battery operation.

---

## 1. Background

The `AEMOBatteryTradingEnv` previously used a simplified linear degradation approximation:

```python
dod = abs(new_soc - old_soc) / self.battery_capacity
degradation_cost = dod * self.battery_life_cost * 0.0001
```

This approach does not capture the non-linear relationship between cycle depth, SoC range, C-rate, and battery wear. The `SolarBatteryEnv` (household environment) already integrates the Muenzel et al. (2015) multi-factor degradation model with rainflow cycle counting. The goal is to bring equivalent fidelity to the AEMO grid-scale environment.

---

## 2. Suitability of Rainflow Counting for Grid-Scale Operation

### 2.1 Why Rainflow Counting is Suitable

Rainflow cycle counting (ASTM E1049-85) is well-suited for grid-scale BESS for the following reasons:

1. **Irregular cycling patterns**: Grid-scale batteries participate in energy arbitrage and FCAS markets, resulting in highly irregular charge/discharge profiles (partial cycles, micro-cycles from frequency regulation, deep cycles from arbitrage). Rainflow counting correctly decomposes these irregular profiles into equivalent full and half cycles.

2. **Industry adoption**: Rainflow counting is the standard method used by battery manufacturers and grid operators (e.g., Tesla, Fluence, Wartsila) for warranty cycle counting and degradation estimation in grid-scale applications.

3. **Captures depth-of-discharge non-linearity**: Shallow FCAS-driven cycles cause disproportionately less wear than deep arbitrage cycles. The rainflow approach, combined with the Muenzel et al. model, correctly captures this via the DoD-dependent cycle life polynomial.

4. **C-rate sensitivity**: Grid-scale batteries (typically 1C–2C rated) may operate at varying power levels. The model accounts for charge and discharge current effects on cycle life.

### 2.2 Considerations and Limitations

1. **Calendar aging**: The current model only captures cycle aging. Grid-scale batteries also experience calendar aging (degradation from time and temperature regardless of cycling). For long-horizon simulations, a calendar aging term should be added in a future iteration.

2. **Temperature model calibration**: The Muenzel et al. temperature coefficients were derived from small-format cells. Grid-scale systems have active thermal management that keeps cells within a narrow band (typically 20–30°C). The default `degradation_temperature=25.0°C` is appropriate for most NEM installations.

3. **C-rate scaling**: Grid-scale BESS are typically rated at 0.5C to 2C. The model's C-rate curves (valid up to ~3C) cover this range adequately.

---

## 3. Implementation Details

### 3.1 Changes to `AEMOBatteryEnv.py`

The following changes were made to integrate the degradation model:

- **Imports**: Added `DegradationModel` and `RainflowCounter` from `batterydeg.py`.
- **Constructor**: Added `degradation_mode` parameter (`'rainflow'` or `'simple'`) and `degradation_temperature` parameter. The `'rainflow'` mode (default) initializes the full Muenzel et al. model and a `RainflowCounter`. The `'simple'` mode preserves the original linear approximation for backward compatibility.
- **State tracking**: Added `initial_battery_capacity`, `total_degradation`, `_rainflow_deg_cumulative`, `_rainflow_num_cycles`, and `soc_history` tracking.
- **Step logic**: SOC is recorded as a percentage of initial capacity after each step. In rainflow mode, the `RainflowCounter` processes each SOC point and returns any newly closed cycles. Each cycle's degradation is computed via `DegradationModel.degradation_per_cycle()`.
- **Capacity fade**: After each step, effective `battery_capacity` is reduced proportionally to accumulated degradation: `capacity = initial_capacity * (1 - total_degradation)`.
- **Reward**: Degradation cost (`step_degradation * battery_life_cost`) is subtracted from the reward, giving the agent an incentive to minimize battery wear.
- **Info dict**: Extended with `step_degradation`, `total_degradation`, `capacity_mwh`, `rainflow_cumulative_deg`, and `rainflow_num_cycles` for monitoring and debugging.
- **Reset**: All degradation state is reset on episode reset, including reinitializing the `RainflowCounter`.

### 3.2 Consistency with SolarBatteryEnv

The implementation follows the same patterns as `SolarBatteryEnv.step()`:
- Same `RainflowCounter` and `DegradationModel` classes
- Same SOC-percentage-based tracking
- Same capacity fade mechanism
- Same `_safe_degradation_per_cycle` error handling pattern

### 3.3 Configurable Degradation Mode

Users can select the degradation mode:

```python
# Physics-based rainflow counting (recommended, default)
env = AEMOBatteryTradingEnv(
    aemo_data=data,
    degradation_mode='rainflow',
    degradation_temperature=25.0,
)

# Simplified linear model (backward compatible)
env = AEMOBatteryTradingEnv(
    aemo_data=data,
    degradation_mode='simple',
)
```

---

## 4. Future Work

1. **Calendar aging**: Add time-based degradation component (e.g., SEI growth model) that accumulates independently of cycling.
2. **Temperature dynamics**: Model battery temperature as a function of power throughput and ambient conditions, rather than using a fixed temperature.
3. **Cell-level heterogeneity**: For large BESS, model variation across cell modules.
4. **Warranty-aligned cycle counting**: Align degradation thresholds with manufacturer warranty terms (e.g., 80% capacity at 10 years or N equivalent cycles).

---

## 5. References

- Muenzel, V., et al. (2015). "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management."
- ASTM E1049-85, "Standard Practices for Cycle Counting in Fatigue Analysis."
- AEMO (2024). "Battery Energy Storage System Registration and Compliance Guide."
