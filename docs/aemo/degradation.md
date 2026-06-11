# AEMO battery degradation design note

This is the **design note** for how battery degradation modeling is handled in the AEMO environment.

Use this document when you need:

- the rationale behind the degradation models
- the implementation summary for `AEMOBatteryTradingEnv`
- model limitations and future extension ideas

This is primarily a background/design document rather than an operational workflow guide. For the full AEMO docs map, start with [README.md](README.md).

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

## 4. Real-World BESS Degradation Model (Implemented)

The `RealWorldBESSDegradationModel` addresses the calendar aging gap and other limitations listed above. It is adapted from the framework presented in Kampker et al. (2025, doi:10.3390/batteries11110392).

### 4.1 What's New

- **Calendar aging**: Time- and temperature-dependent capacity loss using Arrhenius kinetics and SOC stress, applied every simulation timestep regardless of cycling.
- **Arrhenius temperature model**: Physically grounded `exp(-Ea/(R·T))` for both calendar and cycle aging, replacing the Muenzel cubic polynomial.
- **Chemistry presets**: NMC and LFP parameter sets, with LFP recommended for modern utility-scale BESS (e.g., Tesla Megapack, BYD).
- **Separate tracking**: `calendar_degradation` and `cycle_degradation` are independently accumulated and reported in the `info` dict.

### 4.2 Usage

```python
# Real-world BESS degradation (recommended for AEMO grid-scale)
env = AEMOBatteryTradingEnv(
    aemo_data=data,
    degradation_mode='real_world',
    degradation_chemistry='LFP',       # or 'NMC'
    degradation_temperature=30.0,      # Australian summer
)

# Physics-based rainflow only (Muenzel et al., no calendar aging)
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

See [docs/household/degradation.md](../household/degradation.md) for the full mathematical formulation and parameter tables.

---

## 5. Future Work

1. **Temperature dynamics**: Model battery temperature as a function of power throughput and ambient conditions (as done in the paper's thermal module), rather than using a fixed temperature.
2. **Throughput-based cycling model**: Optionally implement the continuous Ah-throughput form (paper Eq. 5) as an alternative to the per-cycle formulation, for users who prefer rate-equation integration.
3. **Cell-level heterogeneity**: For large BESS, model variation across cell modules (as explored in the paper's pack-level simulations).
4. **Warranty-aligned cycle counting**: Align degradation thresholds with manufacturer warranty terms (e.g., 80% capacity at 10 years or N equivalent cycles).
5. **Resistance growth**: Add internal resistance increase tracking (R₀ growth) alongside capacity fade, as modeled in the paper.

---

## 6. References

- Muenzel, V., et al. (2015). "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management."
- Kampker, A.; Späth, B.; Song, X.; Wang, D. (2025). "Modelling of Battery Energy Storage Systems Under Real-World Applications and Conditions." *Batteries* 11(11):392. doi:10.3390/batteries11110392.
- Wang, J.; Liu, P.; Hicks-Garner, J.; et al. (2011). "Cycle-life model for graphite-LiFePO4 cells." *Journal of Power Sources*, 196(8):3942–3948.
- ASTM E1049-85, "Standard Practices for Cycle Counting in Fatigue Analysis."
- AEMO (2024). "Battery Energy Storage System Registration and Compliance Guide."
