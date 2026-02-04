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
