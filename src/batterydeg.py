import math
import numpy as np

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

# Example parameters for degradation
CL_nom = 3650  # Nominal cycle life
B = 5.0        # Battery capacity (kWh)
DoD_nom = 100  # Nominal depth of discharge (%)

def custom_find_peaks(data: np.array) -> np.array:
    """
    Finds indices of local peaks in a 1D NumPy array.
    A peak is defined as a point where the derivative changes from positive to negative.
    """
    # Compute the difference between consecutive elements.
    diff = np.diff(data)
    # Identify where the diff changes sign (from positive to negative).
    peaks = np.where((np.hstack([diff, 0]) > 0) & (np.hstack([0, diff]) < 0))[0]
    return peaks

def custom_find_troughs(data: np.array) -> np.array:
    """
    Finds indices of local troughs in a 1D NumPy array.
    A trough is defined as a point where the derivative changes from negative to positive.
    """
    diff = np.diff(data)
    troughs = np.where((np.hstack([diff, 0]) < 0) & (np.hstack([0, diff]) > 0))[0]
    return troughs

# Rain-flow cycle counting (optimized for np.array soc_history)
def rainflow_counting(soc_history: np.array):
    """
    Compute rainflow cycle metrics for a state-of-charge (SoC) history array.
    This function implements the rainflow counting algorithm to identify charge-discharge 
    cycles from a given SoC time series. It first identifies local peaks and troughs in the 
    data, pairs consecutive extrema to define cycles, and then computes the depth of discharge 
    (DoD) and the average SoC for each cycle.
    Detailed Steps:
        1. Peak and Trough Detection:
           - The function calls custom_find_peaks to obtain indices of local maxima.
           - It calls custom_find_troughs to obtain indices of local minima.
           - These indices represent potential turning points (extrema) in the SoC data.
        2. Extrema Consolidation:
           - The indices from peaks and troughs are concatenated and then sorted. This ensures
             that the extrema are in chronological order, reflecting the actual progression 
             of the SoC history.
        3. Pairing Extrema:
           - To ensure an even number of extrema (a necessity for pairing), if there's an odd 
             number of extrema, the last one is discarded.
           - The resulting list of indices is reshaped into pairs. Each pair of consecutive 
             extrema represents one complete cycle of charge and discharge.
        4. Cycle Metrics Calculation:
           - The depth of discharge (DoD) for each cycle is computed as the absolute difference 
             between the SoC values at the two extrema of the pair.
           - The average SoC for each cycle is calculated as the arithmetic mean of the two 
             corresponding SoC values.
           - These metrics convey the cycle's severity and its midpoint charge level.
        5. Result Assembly:
           - The cycle DoDs and average SoCs are combined into a 2-column numpy array, where 
             each row corresponds to one identified cycle.
             - Column 1: Cycle DoD (absolute difference between the two associated SoC values).
             - Column 2: Average SoC for that cycle.
    Parameters:
        soc_history (np.array): A numpy array representing the state-of-charge history over time. 
                                It is expected to contain the SoC values that will be analyzed.
    Returns:
        np.ndarray: A 2-column array where each row corresponds to a cycle:
                    - The first column contains the depth of discharge (DoD) for each cycle.
                    - The second column contains the average state-of-charge (SoC) for that cycle.
    """
    # soc_history is expected to be a numpy array
    peaks = custom_find_peaks(soc_history)
    troughs = custom_find_troughs(soc_history)
    extrema = np.sort(np.concatenate((peaks, troughs)))
    
    # Ensure an even number of extrema by discarding the last one if necessary
    n = len(extrema) - (len(extrema) % 2)
    paired = extrema[:n].reshape(-1, 2)
    
    # Vectorized computation of cycle depth (DoD) and average SoC
    cycle_DoDs = np.abs(soc_history[paired[:, 1]] - soc_history[paired[:, 0]])
    avg_SoCs = (soc_history[paired[:, 1]] + soc_history[paired[:, 0]]) / 2.0
    cycles = np.column_stack((cycle_DoDs, avg_SoCs))
    
    return cycles

# Normalized cycle life function for a cycle
def normalized_cycle_life(DoD, avg_SoC):
    # Assume degradation increases with depth of discharge and deviates with SoC
    degradation_factor = (DoD / DoD_nom) ** 1.5 * (1 + 0.1 * abs(avg_SoC - 50) / 50)
    if degradation_factor == 0:
        return float('inf')
    return 1 / degradation_factor

# Dynamic degradation model, provides the fractional life utilization of a battery for a given charge or discharge decision
def dynamic_degradation(soc_history: list):
    # convert to numpy array
    soc_history = np.array(soc_history)
    cycles = rainflow_counting(soc_history)
    total_degradation = 0
    for DoD, avg_SoC in cycles:
        nCL = normalized_cycle_life(DoD, avg_SoC)
        if math.isinf(nCL):
            degradation_per_cycle = 0
        else:
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
