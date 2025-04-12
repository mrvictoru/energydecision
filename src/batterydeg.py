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
# Normalized cycle life functions
def nCL_Id(Id):
    e, f, g, h = 4464.0, -0.1382, -1519, -0.4305
    return (e * math.exp(f * Id) + g * math.exp(h * Id)) / (e * math.exp(f * Id_nom) + g * math.exp(h * Id_nom))

def nCL_Ich(Ich):
    m, n, o, p = 5963.0, -0.6531, 321.4, 0.03168
    return (m * math.exp(n * Ich) + o * math.exp(p * Ich)) / (m * math.exp(n * Ich_nom) + o * math.exp(p * Ich_nom))

def nCL_SoC_DoD(SoC, DoD):
    q, s, t, u, v = 1471.0, 214.3, 0.6111, 0.3369, -2.295
    CL4 = lambda DoD, SoC: q + (20.0 * (s + 100.0 * u) - 200.0 * t) * DoD + s * SoC + t * DoD**2 + u * DoD * SoC + v * SoC**2
    return CL4(DoD, SoC) / CL4(DoD_nom, SoC_nom)

# Static multi-factor degradation model, provides the fractional life utilization of a battery for a given charge or discharge decision
def static_degradation(Id, Ich, SoC, DoD):
    nCL = (nCL_Id(Id) * nCL_Ich(Ich) * nCL_SoC_DoD(SoC, DoD))
    denom = CL_nom * nCL
    if abs(denom) < 1e-12:
        return 0
    with np.errstate(divide='ignore', invalid='ignore'):
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

# Four-point rainflow counting algorithm
def rainflow_counting(soc_profile, step_duration=1.0):
    """
    Identifies charge-discharge cycles using the rainflow counting algorithm.
    Now returns a list of cycles where each cycle is a tuple:
        (SoC_avg, DoD, Id_cycle, Ich_cycle)
    where the c–rates are computed as:
        rate = (SoC_max - SoC_min) / (100 * (time difference in hours))
    """
    # Record turning points as (index, soc)
    turning_points = []
    for i in range(1, len(soc_profile) - 1):
        if (soc_profile[i] > soc_profile[i - 1] and soc_profile[i] > soc_profile[i + 1]) or \
           (soc_profile[i] < soc_profile[i - 1] and soc_profile[i] < soc_profile[i + 1]):
            turning_points.append((i, soc_profile[i]))

    cycles = []
    stack = []
    for tp in turning_points:
        stack.append(tp)
        while len(stack) >= 4:
            r1 = abs(stack[-1][1] - stack[-2][1])
            r2 = abs(stack[-2][1] - stack[-3][1])
            r3 = abs(stack[-3][1] - stack[-4][1])
            if r2 <= r1 and r2 <= r3:
                # Identify cycle using the two central turning points
                idx1, soc1 = stack[-3]
                idx2, soc2 = stack[-2]
                SoC_max = max(soc1, soc2)
                SoC_min = min(soc1, soc2)
                DoD = SoC_max - SoC_min
                SoC_avg = (SoC_max + SoC_min) / 2.0
                # Estimate cycle duration in hours. Assume turning point indices are 1 step apart in time (scaled by step_duration)
                delta_time = abs(idx2 - idx1) * step_duration
                # Avoid division by zero
                if delta_time == 0:
                    Id_cycle = Ich_cycle = 0.0
                else:
                    # Determine the direction of the half-cycle
                    if soc2 > soc1:
                        # Charging half-cycle: rising SoC -> charge current nonzero
                        Ich_cycle = (soc2 - soc1) / (100.0 * delta_time)
                        Id_cycle = 0.0
                    elif soc2 < soc1:
                        # Discharging half-cycle: falling SoC -> discharge current nonzero
                        Id_cycle = (soc1 - soc2) / (100.0 * delta_time)
                        Ich_cycle = 0.0
                    else:
                        # No change
                        Id_cycle = Ich_cycle = 0.0

                cycles.append((SoC_avg, DoD, Id_cycle, Ich_cycle))
                # Remove the middle two turning points from the stack
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
    denom = nCL * CL_nom
    if abs(denom) < 1e-12:
        return 0
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
    total_degradation = 0
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
