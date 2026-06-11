# Battery Degradation Modeling Guide

This is the **household degradation deep dive** for `src/batterydeg.py`.

Use this document when you need:

- the degradation math and terminology
- the Muenzel-style model details
- rainflow cycle counting behavior
- the implementation notes used by `SolarBatteryEnv`

For the broader household docs map, start with [README.md](README.md).

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

---

## 7. Real-World BESS Degradation Model (`RealWorldBESSDegradationModel`)

### Overview

For utility-scale grid simulations (e.g., AEMO environment), the Muenzel et al. (2015) model has significant limitations:
- **No calendar aging**: Grid-scale BESS sits idle for long periods (overnight, weekends). Calendar aging from SEI growth is absent.
- **Empirical temperature model**: Uses a cubic polynomial fitted to small lab cells rather than the physically-grounded Arrhenius equation.

The `RealWorldBESSDegradationModel` class addresses these gaps. It is adapted from the framework described in:

> Kampker, A.; Späth, B.; Song, X.; Wang, D. (2025). "Modelling of Battery Energy Storage Systems Under Real-World Applications and Conditions." *Batteries* 11(11):392. doi:[10.3390/batteries11110392](https://doi.org/10.3390/batteries11110392)

### Relationship to the Paper

The paper presents a modular simulation framework integrating electrical, thermal, and aging models for LFP cells. Its aging module (Section 3.4, adapted from Wang et al. 2011) uses a continuous throughput-based differential equation (Eq. 5):

$$\frac{dQ_{loss}}{dt} = \left(\frac{15}{C_{rate}}\right)^{1/3} \cdot 10000 \cdot e^{(-31700 + 370.3 \cdot C_{rate})/(R \cdot T)} \cdot 0.55 \cdot Ah^{-0.45} \cdot C_{rate}^{2/3}$$

Our `RealWorldBESSDegradationModel` is a **practical adaptation** of the paper's principles for use in an RL environment with discrete timesteps and rainflow cycle counting. Rather than directly porting Eq. 5 (which requires continuous Ah throughput tracking), we decompose aging into two independently evaluated components:

| Aspect | Paper (Eq. 5) | Our Adaptation |
|--------|--------------|----------------|
| **Form** | Continuous differential (dQloss/dt) | Discrete per-timestep (calendar) + per-cycle (cycle) |
| **Temperature** | Arrhenius with C-rate-coupled Ea | Normalized Arrhenius with fixed Ea per aging mode |
| **C-rate** | Coupled in exponent + power-law prefactor | Linear sensitivity factor in cycle aging |
| **DoD/throughput** | Cumulative Ah^(-0.45) (sublinear) | Per-cycle (DoD/100)^α power-law |
| **Calendar aging** | Described qualitatively (Arrhenius) | Explicit: k_cal · arr(T) · soc_stress · Δt |
| **Chemistry** | LFP only | NMC and LFP presets |

**Key principles preserved from the paper:**
1. ✅ Combined calendar + cycle aging (paper Section 3.4)
2. ✅ Arrhenius temperature dependency for both aging modes
3. ✅ C-rate sensitivity in cycle aging
4. ✅ DoD/cycling depth dependency
5. ✅ SOC-dependent calendar aging (high SOC accelerates degradation)
6. ✅ Capacity fade: $C(t) = C_{nom} \cdot (1 - Q_{loss})$

### Mathematical Formulation

**Total capacity loss** (fraction of nominal capacity):
$$Q_{loss} = Q_{cal} + Q_{cyc} \quad (\text{capped at 1.0})$$

**Calendar aging per timestep** $\Delta t$ (hours):
$$\Delta Q_{cal} = k_{cal} \cdot \frac{\exp(-E_{a,cal}/(R \cdot T_K))}{\exp(-E_{a,cal}/(R \cdot T_{ref}))} \cdot [1 + k_{soc} \cdot (SOC_{frac} - 0.5)] \cdot \Delta t$$

where:
- $k_{cal}$ — calendar aging rate [capacity fraction / hour] at $T_{ref}$, 50% SOC
- $E_{a,cal}$ — activation energy for calendar aging [J/mol]
- $R = 8.314$ J/(mol·K) — universal gas constant
- $T_K$ — cell temperature [K] = T_celsius + 273.15
- $T_{ref}$ = 298.15 K (25°C)
- $k_{soc}$ — SOC stress coefficient; higher SOC increases calendar degradation
- $SOC_{frac} \in [0, 1]$

**Cycle aging per detected rainflow cycle:**
$$\Delta Q_{cyc} = k_{cyc} \cdot \frac{\exp(-E_{a,cyc}/(R \cdot T_K))}{\exp(-E_{a,cyc}/(R \cdot T_{ref}))} \cdot \left(\frac{DOD}{100}\right)^{\alpha} \cdot (1 + \beta \cdot C_{rate})$$

where:
- $k_{cyc}$ — cycle aging coefficient [capacity fraction / full-DoD cycle] at $T_{ref}$, 1C
- $E_{a,cyc}$ — activation energy for cycle aging [J/mol]
- $\alpha$ — DoD power-law exponent (≥ 0)
- $\beta$ — C-rate linear sensitivity factor (≥ 0)
- $DOD$ — depth of discharge [0–100%]
- $C_{rate}$ — equivalent C-rate of the cycle

### Chemistry Presets

| Parameter | NMC | LFP |
|-----------|-----|-----|
| `k_cal_rate` | 2.85 × 10⁻⁶ /h | 1.20 × 10⁻⁶ /h |
| `Ea_cal` | 28,500 J/mol | 17,500 J/mol |
| `k_soc` | 0.5 | 0.2 |
| `k_cyc` | 3.5 × 10⁻⁴ /cycle | 1.95 × 10⁻⁴ /cycle |
| `Ea_cyc` | 17,100 J/mol | 10,000 J/mol |
| `alpha_dod` | 1.2 | 0.5 |
| `beta_crate` | 0.5 | 0.3 |
| Approx. cycle life (100% DoD, 1C, 25°C) | ~2,000 | ~5,000 |
| Approx. calendar EOL (25°C, 50% SOC) | ~12–15 years | ~20+ years |

### Usage

```python
from src.batterydeg import RealWorldBESSDegradationModel

# Initialize with LFP preset (recommended for utility-scale BESS)
model = RealWorldBESSDegradationModel(chemistry='LFP')

# Calendar aging: 30-minute step at 35°C, 80% SOC
cal_loss = model.calendar_aging_per_step(T_celsius=35.0, soc_frac=0.8, dt_hours=0.5)

# Cycle aging: one cycle at 80% DoD, 0.5C, 25°C
cyc_loss = model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=80.0, c_rate=0.5)

# Custom parameters (override preset)
custom = RealWorldBESSDegradationModel(chemistry='NMC', k_cal_rate=3e-6, alpha_dod=1.0)

# Inspect parameters
print(model.describe())
```

### Integration in `AEMOBatteryTradingEnv`

Select the `'real_world'` degradation mode:

```python
env = AEMOBatteryTradingEnv(
    aemo_data=data,
    degradation_mode='real_world',
    degradation_chemistry='LFP',
    degradation_temperature=35.0,  # hot Australian climate
)
```

The environment computes:
- **Calendar aging every step** (even when idle) based on current SOC and temperature
- **Cycle aging per rainflow-detected cycle** using DoD, C-rate, and temperature
- Separate tracking: `info['calendar_degradation']` and `info['cycle_degradation']`
- Total: `info['total_degradation']` = calendar + cycle (capped at 1.0)

### References
- Kampker, A.; Späth, B.; Song, X.; Wang, D. (2025). "Modelling of Battery Energy Storage Systems Under Real-World Applications and Conditions." *Batteries* 11(11):392.
- Wang, J.; Liu, P.; Hicks-Garner, J.; et al. (2011). "Cycle-life model for graphite-LiFePO4 cells." *Journal of Power Sources*, 196(8):3942–3948.
