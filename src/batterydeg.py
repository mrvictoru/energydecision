import math
import numpy as np

# Parameters (example values, adjust as needed)
CL_nom = 3650.0  # Nominal cycle life
Id_nom = 0.25  # Nominal discharge current (C-rate)
Ich_nom = 0.125  # Nominal charge current (C-rate)
SoC_nom = 50.0  # Nominal state of charge (%)
DoD_nom = 90.0  # Nominal depth of discharge (%)

# The following parameters are derived from "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management (2015)" 
# by V. Muenzel, J. D. Hoog, M. Brazil, A. Vishwanath, and S. Kalya-naraman

# Pre-compute nominal denominators for efficiency (these are constants)
_nCL_Id_nom_denom = 4464.0 * math.exp(-0.1382 * Id_nom) + (-1519) * math.exp(-0.4305 * Id_nom)
_nCL_Ich_nom_denom = 5963.0 * math.exp(-0.6531 * Ich_nom) + 321.4 * math.exp(0.03168 * Ich_nom)

# Normalized cycle life functions
def nCL_Id(Id):
    e, f, g, h = 4464.0, -0.1382, -1519, -0.4305
    return (e * math.exp(f * Id) + g * math.exp(h * Id)) / _nCL_Id_nom_denom

def nCL_Ich(Ich):
    m, n, o, p = 5963.0, -0.6531, 321.4, 0.03168
    return (m * math.exp(n * Ich) + o * math.exp(p * Ich)) / _nCL_Ich_nom_denom

def _ensure_positive(value: float, eps: float = 1e-9) -> float:
    if not np.isfinite(value) or value <= eps:
        return eps
    return value


# Helper function for cycle life formula (defined once, not as lambda)
def _CL4(DoD, SoC):
    """Cycle life formula with precomputed coefficients."""
    q, s, t, u, v = 1471.0, 214.3, 0.6111, 0.3369, -2.295
    return q + (20.0 * (s + 100.0 * u) - 200.0 * t) * DoD + s * SoC + t * DoD**2 + u * DoD * SoC + v * SoC**2

# Pre-compute nominal denominator for nCL_SoC_DoD (constant)
_nCL_SoC_DoD_nom_denom = _ensure_positive(_CL4(DoD_nom, SoC_nom))

def nCL_SoC_DoD(SoC, DoD):
    num = _ensure_positive(_CL4(DoD, SoC))
    return num / _nCL_SoC_DoD_nom_denom

# Static multi-factor degradation model, provides the fractional life utilization of a battery for a given charge or discharge decision
def static_degradation(Id, Ich, SoC, DoD):
    nCL = (nCL_Id(Id) * nCL_Ich(Ich) * nCL_SoC_DoD(SoC, DoD))
    denom = _ensure_positive(CL_nom * nCL)
    return 0.5 / denom

"""
# Example usage
Id = 0.3  # Discharge current (C-rate)
Ich = 0.1  # Charge current (C-rate)
SoC = 60  # Average state of charge (%)
DoD = 80  # Depth of discharge (%)

degradation = static_degradation(Id, Ich, SoC, DoD)
print(f"Degradation for this cycle: {degradation:.6f}")
"""

def _turning_points_with_plateau_handling(soc_profile, eps: float = 1e-6):
    """
    Return turning points as (index, soc), robust to plateaus and quantization.

    - Removes consecutive duplicates within eps.
    - Uses strict peak/trough test with eps margin.
    - Always includes first and last points.
    """
    arr = np.asarray(soc_profile, dtype=float)
    n = arr.size
    if n == 0:
        return []
    if n == 1:
        return [(0, float(arr[0]))]

    # 1) compress plateaus: keep only points that change by > eps
    keep_idx = [0]
    for i in range(1, n):
        if abs(arr[i] - arr[keep_idx[-1]]) > eps:
            keep_idx.append(i)

    arr2 = arr[keep_idx]
    m = arr2.size
    if m == 1:
        return [(keep_idx[0], float(arr2[0]))]
    if m == 2:
        return [(keep_idx[0], float(arr2[0])), (keep_idx[1], float(arr2[1]))]

    # 2) turning points
    tps = [(keep_idx[0], float(arr2[0]))]
    for i in range(1, m - 1):
        prev_, curr_, next_ = arr2[i - 1], arr2[i], arr2[i + 1]
        if (curr_ - prev_ > eps) and (curr_ - next_ > eps):   # peak
            tps.append((keep_idx[i], float(curr_)))
        elif (curr_ - prev_ < -eps) and (curr_ - next_ < -eps):  # trough
            tps.append((keep_idx[i], float(curr_)))
    tps.append((keep_idx[-1], float(arr2[-1])))
    return tps


def rainflow_counting(soc_profile, step_duration=1.0):
    """
    Identifies charge-discharge cycles using a simplified four-point rainflow algorithm.
    Returns list of cycles: (SoC_avg, DoD, Id_cycle, Ich_cycle)
    """
    turning_points = _turning_points_with_plateau_handling(soc_profile, eps=1e-6)

    cycles = []
    stack = []
    for tp in turning_points:
        stack.append(tp)
        while len(stack) >= 4:
            r1 = abs(stack[-1][1] - stack[-2][1])
            r2 = abs(stack[-2][1] - stack[-3][1])
            r3 = abs(stack[-3][1] - stack[-4][1])

            if r2 <= r1 and r2 <= r3:
                idx1, soc1 = stack[-3]
                idx2, soc2 = stack[-2]

                SoC_max = max(soc1, soc2)
                SoC_min = min(soc1, soc2)
                DoD = SoC_max - SoC_min
                if DoD <= 1e-9:
                    # degenerate cycle (flat), discard and remove middle points
                    del stack[-3:-1]
                    continue

                SoC_avg = (SoC_max + SoC_min) / 2.0
                delta_time = abs(idx2 - idx1) * step_duration

                if delta_time <= 1e-12:
                    Id_cycle = Ich_cycle = 0.0
                else:
                    if soc2 > soc1:
                        Ich_cycle = (soc2 - soc1) / (100.0 * delta_time)
                        Id_cycle = 0.0
                    elif soc2 < soc1:
                        Id_cycle = (soc1 - soc2) / (100.0 * delta_time)
                        Ich_cycle = 0.0
                    else:
                        Id_cycle = Ich_cycle = 0.0

                cycles.append((SoC_avg, DoD, Id_cycle, Ich_cycle))
                del stack[-3:-1]
            else:
                break

    return cycles

def degradation_per_cycle(Id, Ich, SoC_avg, DoD):
    """
    Calculates the fractional degradation caused by a single cycle using
    effective discharge and charge C-rates (Id and Ich) for that cycle.
    """
    nCL = (nCL_Id(Id) * nCL_Ich(Ich) * nCL_SoC_DoD(SoC_avg, DoD))
    denom = _ensure_positive(nCL * CL_nom)
    return 1 / denom

# Dynamic degradation model, provides the fractional life utilization of a battery for a given charge or discharge decision
# Total degradation calculation
def dynamic_degradation(soc_profile, step_duration=0.5):
    """
    Calculates the total degradation over a given SoC profile.
    It now uses the enhanced rainflow counting to return cycles with their
    average SoC, DoD, and effective discharge and charging C-rates.
    """
    cycles = rainflow_counting(soc_profile, step_duration)
    total_degradation = 0.0
    for SoC_avg, DoD, Id_cycle, Ich_cycle in cycles:
        degradation = degradation_per_cycle(Id_cycle, Ich_cycle, SoC_avg, DoD)
        total_degradation += degradation
    return total_degradation, len(cycles)
  

# simulate RL with Hybrid degradation approach
def hybrid_rl_simulation(steps, soc_profile, correction_interval):
    static_cumulative_degradation = 0
    dynamic_cumulative_degradation = 0
    correction_factor = 1.0
    degradation_history = []

    for step in range(steps):
        # Example operational parameters
        Id, Ich, SoC, DoD = 0.3, 0.1, soc_profile[step], 20  # Example values

        # Static degradation estimation
        degradation = static_degradation(Id, Ich, SoC, DoD, correction_factor)
        static_cumulative_degradation += degradation

        # Save degradation history (for dynamic correction)
        degradation_history.append(SoC)

        # Periodic correction using the dynamic model
        if step > 0 and step % correction_interval == 0:
            dynamic_cumulative_degradation = dynamic_degradation(degradation_history)
            correction_factor = dynamic_cumulative_degradation / static_cumulative_degradation
            print(f"Correction factor updated to: {correction_factor:.3f}")
            # Reset history for next correction interval
            degradation_history = []

    print(f"Total Static Degradation: {static_cumulative_degradation:.6f}")
    print(f"Total Dynamic Degradation: {dynamic_cumulative_degradation:.6f}")

"""
# Example usage
soc_profile = np.linspace(20, 80, 100)  # Simplified SoC profile over 100 steps
hybrid_rl_simulation(steps=100, soc_profile=soc_profile, correction_interval=20)
"""
