# Battery Degradation Modeling Guide

This document provides a deep dive into the battery degradation model implemented in `src/batterydeg.py`, based on the multi-factor cycle life prediction methodology by Muenzel et al. (2015).

---

## 1. Overview

Batteries are the most expensive component of a distributed energy system. Every charge and discharge cycle causes microscopic physical damage, leading to capacity fade and eventual replacement. To accurately benchmark control algorithms, we must quantify this cost.

The project uses a **semi-empirical model** that accounts for multiple stress factors simultaneously.

---

## 2. The Mathematical Model

The model predicts the **Cycle Life (CL)**—the number of equivalent full cycles the battery can survive—under specific conditions. The total fractional degradation for one cycle is then $1/CL$.

### Multi-Factor Multiplier
The core equation follows a multiplicative approach:
$$CL_{actual} = CL_{nom} \cdot nCL_T(T) \cdot nCL_{Id}(I_d) \cdot nCL_{Ich}(I_{ch}) \cdot nCL_{SOC,DOD}(SOC_{av}, DOD)$$

Where:
*   $CL_{nom}$ is the nominal cycle life (e.g., 3650 cycles).
*   $nCL_x$ are **normalized coefficients** ($\frac{CL(condition)}{CL(nominal)}$).

### Stress Factors

#### A. Temperature ($T$)
Modeled as a cubic polynomial. Battery health is generally optimal around $25^{\circ}C$ and degrades at extremes.
$$CL(T) = aT^3 - bT^2 + cT + d$$

#### B. Charge/Discharge Rates ($C$-rate)
Higher currents (fast charging/discharging) cause more internal heating and mechanical stress. These are modeled as sums of exponentials:
$$CL(I) = e \cdot \exp(f \cdot I) + g \cdot \exp(h \cdot I)$$

#### C. SoC Average ($SOC_{av}$) and Depth of Discharge ($DOD$)
This is the most complex factor, representing how "deep" a cycle is and where it sits in the capacity range (e.g., a 20% cycle from 80% to 100% vs 0% to 20%). It is modeled as a constrained 2D second-order polynomial:
$$CL_4(DOD, SOC_{av}) = q + r \cdot DOD + s \cdot SOC_{av} + t \cdot DOD^2 + u \cdot DOD \cdot SOC_{av} + v \cdot SOC_{av}^2$$

---

## 3. Software Implementation

### `DegradationModel` Class
This class encapsulates the coefficients and normalization logic.

```python
from src.batterydeg import DegradationModel

# 1. Initialize with nominal parameters
model = DegradationModel(
    CL_nom=3650, 
    T_nom=25, 
    Id_nom=0.25, 
    Ich_nom=0.125
)

# 2. Compute degradation for a specific cycle
# T: Celsius, Id/Ich: C-rate, SOCav: %, DOD: %
deg_fraction = model.degradation_per_cycle(
    T=25.0, Id=0.3, Ich=0.1, SOCav=50.0, DOD=80.0
)

# Cost in dollars
step_cost = deg_fraction * battery_replacement_cost
```

#### Public Methods (DegradationModel)

- `__init__(CL_nom, T_nom, Id_nom, Ich_nom, SOCav_nom=50.0, DOD_nom=90.0, enforce_feasible_region=True)`
    - **Args**: nominal parameters for the multi-factor model. `CL_nom` is the baseline cycles-to-EOL, while the other arguments define the “nominal condition” around which normalized multipliers are computed. `enforce_feasible_region` controls whether SOC/DOD pairs are validated when computing the SOC–DOD multiplier.
    - **Returns**: instance of `DegradationModel` with pre-computed denominator factors for numerical stability.

- `cycle_life(T, Id, Ich, SOCav, DOD)`
    - **Args**: stresses for a candidate cycle: temperature in °C, charge/discharge C-rates, average SoC (%), and depth-of-discharge (%).
    - **Returns**: `CL = CL_nom * nCL_T * nCL_Id * nCL_Ich * nCL_SOCav_DOD`, i.e., the expected number of equivalent full cycles under those conditions. Values below or equal to zero are treated as invalid (raises in downstream callers).

- `degradation_per_cycle(T, Id, Ich, SOCav, DOD)`
    - **Args**: same stress inputs as `cycle_life`. Internally clamps SOC/DOD to the polynomial’s feasible region, caps C-rates to [0, 3], and enforces the inclined safety guard `DOD > 3` to avoid noise at tiny swings.
    - **Returns**: fractional degradation `1 / CL` computed by the multi-factor equation. The method raises `ValueError` if the computed cycle life is non-finite or non-positive.

- `debug_degradation_per_cycle(T, Id, Ich, SOCav, DOD)`
    - **Args**: identical to the other methods; intended for diagnostics.
    - **Returns**: dictionary capturing clipped/normalized inputs, each normalized multiplier (`nCL_T`, `nCL_Id`, `nCL_Ich`, `nCL_SOCav_DOD`), the combined `mult`, and the resulting `CL`/`degradation` value for tracing failure cases.

- `nCL_T`, `nCL_Id`, `nCL_Ich`, `nCL_SOCav_DOD`
    - **Args**: a single stress variable or SOC/DOD pair.
    - **Returns**: dimensionless multiplier relative to the nominal condition. Each helper guards against invalid inputs by returning `1.0` when the raw factor is non-finite or negative.

- `static_degradation(Id, Ich, SoC_avg, DoD)`
    - **Args**: convenience wrapper that instantiates a default `DegradationModel` and calls `degradation_per_cycle` (temperature fixed at 25 °C here).
    - **Returns**: the degradation fraction and prints the equivalent cycle life to stdout for quick sanity checks.

### `RainflowCounter` Class
In real-world operation, batteries don't follow clean cycles. They have partial charges and micro-discharges. The `RainflowCounter` implements the **ASTM E1049-85 standard** to extract cycles from a varying State of Charge (SoC) profile.

```python
from src.batterydeg import RainflowCounter

counter = RainflowCounter(step_duration=0.5)

# Feed SoC values one by one
soc_profile = [20, 25, 40, 35, 60]
for soc in soc_profile:
    closed_cycles = counter.update(soc) # Returns list of (SoC_avg, DoD, Id, Ich)
```

#### Public Methods (RainflowCounter)

- `__init__(step_duration=1.0, eps=0.1, max_c_rate=1.0)`
    - **Args**: `step_duration` is the time between SOC samples (hours), `eps` is the plateau tolerance (percent SoC) used in turning-point detection, and `max_c_rate` clamps the inferred C-rate per cycle.
    - **Returns**: counter that maintains a stack of turning points; `step_duration` is reused when converting SoC deltas to currents.

- `update(soc)`
    - **Args**: incoming SoC percentage ([0, 100]) at the next timestep.
    - **Returns**: list of closed cycles detected since the previous update. Each tuple carries `(SoC_avg, DoD, Id_cycle, Ich_cycle)` describing the average SoC, depth-of-discharge, and implied charge/discharge C-rate for that cycle. The method discards tiny cycles below `eps` and ensures `DoD` remains positive before reporting.

- `rainflow_counting(soc_profile, step_duration=1.0, eps=1e-6)`
    - **Args**: helper function that accepts a full SoC profile (sequence) and optional granularity parameters.
    - **Returns**: same sequence of closed cycles as repeatedly calling `RainflowCounter.update`; useful when you have a batch SoC trace instead of streaming data.

---

## 4. Usage in the Environment

The `SolarBatteryEnv` uses this model automatically:
1.  On every `step(action)`, the result SoC change is recorded.
2.  The `RainflowCounter` tracks the internal state and identifies when a cycle is "closed".
3.  The agent receives a `deg_cost` in the `info` dictionary, which is subtracted from the reward.

---

## 5. Feasibility Constraints
The model enforces the physical reality of the $SOC_{av}$ and $DOD$:
*   $DOD \leq 2 \cdot SOC_{av}$
*   $DOD \leq 2 \cdot (100 - SOC_{av})$

If an agent attempts a cycle outside these bounds, the `degradation_per_cycle` method will clamp the inputs to the nearest valid point to ensure the polynomial fit remains physically grounded.

---

## 6. References
*   **Primary Paper**: Muenzel, V., et al. (2015). "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management."
*   **Standard**: ASTM E1049-85, "Standard Practices for Cycle Counting in Fatigue Analysis."
