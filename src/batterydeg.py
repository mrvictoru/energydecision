import math
import numpy as np
from scipy.signal import find_peaks

# Parameters (example values, adjust as needed)
CL_nom = 3650  # Nominal cycle life
Id_nom = 0.25  # Nominal discharge current (C-rate)
Ich_nom = 0.125  # Nominal charge current (C-rate)
SoC_nom = 50  # Nominal state of charge (%)
DoD_nom = 100  # Nominal depth of discharge (%)

# The following parameters are derived from "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management (2015)" 
# by V. Muenzel, J. D. Hoog, M. Brazil, A. Vishwanath, and S. Kalya-naraman
# Normalized cycle life functions
def nCL_Id(Id):
    e, f, g, h = 4464, -0.1382, -1519, -0.4305
    return (e * math.exp(f * Id) + g * math.exp(h * Id)) / (e * math.exp(f * Id_nom) + g * math.exp(h * Id_nom))

def nCL_Ich(Ich):
    m, n, o, p = 5963, -0.6531, 321.4, 0.03168
    return (m * math.exp(n * Ich) + o * math.exp(p * Ich)) / (m * math.exp(n * Ich_nom) + o * math.exp(p * Ich_nom))

def nCL_SoC_DoD(SoC, DoD):
    q, s, t, u, v = 1471, 214.3, 0.6111, 0.3369, -2.295
    CL4 = lambda DoD, SoC: q + (20 * (s + 100 * u) - 200 * t) * DoD + s * SoC + t * DoD**2 + u * DoD * SoC + v * SoC**2
    return CL4(DoD, SoC) / CL4(DoD_nom, SoC_nom)

# Static multi-factor degradation model
def static_degradation(Id, Ich, SoC, DoD, correction_factor = 1.0):
    nCL = correction_factor * (nCL_Id(Id) * nCL_Ich(Ich) * nCL_SoC_DoD(SoC, DoD))
    return 0.5 / (CL_nom * nCL)

"""
# Example usage
Id = 0.3  # Discharge current (C-rate)
Ich = 0.1  # Charge current (C-rate)
SoC = 60  # Average state of charge (%)
DoD = 80  # Depth of discharge (%)

degradation = static_degradation(Id, Ich, SoC, DoD)
print(f"Degradation for this cycle: {degradation:.6f}")
"""

# Example parameters for degradation
CL_nom = 3650  # Nominal cycle life
B = 5.0        # Battery capacity (kWh)
DoD_nom = 100  # Nominal depth of discharge (%)

# Rain-flow cycle counting (simplified)
def rainflow_counting(soc_history):
    peaks, _ = find_peaks(soc_history)
    troughs, _ = find_peaks(-soc_history)
    extrema = sorted(np.concatenate((peaks, troughs)))
    cycles = []
    
    # Calculate cycles based on peaks and troughs
    for i in range(0, len(extrema) - 1, 2):
        start = extrema[i]
        end = extrema[i + 1]
        cycle_DoD = abs(soc_history[end] - soc_history[start])
        avg_SoC = (soc_history[end] + soc_history[start]) / 2
        cycles.append((cycle_DoD, avg_SoC))
    return cycles

# Normalized cycle life function for a cycle
def normalized_cycle_life(DoD, avg_SoC):
    # Assume degradation increases with depth of discharge and deviates with SoC
    degradation_factor = (DoD / DoD_nom) ** 1.5 * (1 + 0.1 * abs(avg_SoC - 50) / 50)
    return 1 / degradation_factor

# Dynamic degradation model
def dynamic_degradation(soc_history):
    cycles = rainflow_counting(soc_history)
    total_degradation = 0
    for DoD, avg_SoC in cycles:
        nCL = normalized_cycle_life(DoD, avg_SoC)
        degradation_per_cycle = 1 / (CL_nom * nCL)
        total_degradation += degradation_per_cycle
    return total_degradation
  
"""
# Example usage
soc_history = [20, 80, 40, 90, 30, 70, 50, 100, 10]  # Example SoC (%) over time
degradation = dynamic_degradation(soc_history)
print(f"Total battery degradation: {degradation:.6f}")
"""

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
