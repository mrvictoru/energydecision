import math
import numpy as np

# Parameters (example values, adjust as needed)
CL_nom = 3650.0  # Nominal cycle life
T_nom = 25.0  # Nominal temperature (°C)
Id_nom = 0.25  # Nominal discharge current (C-rate)
Ich_nom = 0.125  # Nominal charge current (C-rate)
SoC_nom = 50.0  # Nominal state of charge (%)
DoD_nom = 90.0  # Nominal depth of discharge (%)

# The following parameters are derived from "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management (2015)" 
# by V. Muenzel, J. D. Hoog, M. Brazil, A. Vishwanath, and S. Kalya-naraman

# =============================================================================
# Muenzel et al. (2015) multi-factor cycle life model
#
# Implements:
#   - Eq. (4), (7): temperature model + normalization
#   - Eq. (5), (8): discharge current model + normalization
#   - Eq. (6), (9): charge current model + normalization
#   - Eq. (13): constrained SOCav–DOD 2D polynomial CL4(DOD, SOCav)
#   - Normalization consistent with Eq. (1)/(3): nCL = CL(condition) / CL(nominal)
#   - Eq. (3): CL = CL_nom * Π nCL(...)
#   - Per-cycle degradation as 1 / CL (as used for Dk in Eq. (24))
#
# Notes:
#   - SOCav and DOD are in percent [0, 100].
#   - Currents are in C-rate units.
#   - Temperature in °C.
#   - The SOC–DOD polynomial is only meaningful in the feasible region:
#       DOD <= 2*SOCav and DOD <= 2*(100 - SOCav).
# =============================================================================


def _ensure_positive(x: float, eps: float = 1e-12) -> float:
    if not np.isfinite(x) or x <= eps:
        return eps
    return x


def _in_feasible_soc_dod_region(soc_av: float, dod: float) -> bool:
    return (
        0.0 <= soc_av <= 100.0
        and 0.0 <= dod <= 100.0
        and dod <= 2.0 * soc_av
        and dod <= 2.0 * (100.0 - soc_av)
    )


class DegradationModel:
    """
    Implementation of Multi-Factor Battery Cycle Life Prediction Methodology Muenzel et al. (2015).

    You must provide:
      - CL_nom: nominal cycle life (cycles to EOL at nominal conditions)
      - nominal conditions: T_nom, Id_nom, Ich_nom, SOCav_nom, DOD_nom

    Then call:
      - cycle_life(...)
      - degradation_per_cycle(...)
    """

    # ---- Table 2 coefficients (as shown in your extracted text) ----
    # Temperature model: CL(T) = a*T^3 - b*T^2 + c*T + d
    _a, _b, _c, _d = 0.0039, 1.95, 67.51, 2070.0

    # Discharge current model: CL(Id) = e*exp(f*Id) + g*exp(h*Id)
    _e, _f, _g, _h = 4464.0, -0.1382, -1519.0, -0.4305

    # Charge current model: CL(Ich) = m*exp(n*Ich) + o*exp(p*Ich)
    _m, _n, _o, _p = 5963.0, -0.6531, 321.4, 0.03168

    # SOCav–DOD constrained polynomial coefficients (Eq. 13 fit): q, s, t, u, v
    _q, _s, _t, _u, _v = 1471.0, 214.3, 0.6111, 0.3369, -2.295

    def __init__(
        self,
        *,
        CL_nom: float = CL_nom,
        T_nom: float = T_nom,
        Id_nom: float = Id_nom,
        Ich_nom: float = Ich_nom,
        SOCav_nom: float = SoC_nom,
        DOD_nom: float = DoD_nom,
        enforce_feasible_region: bool = True,
    ):
        self.CL_nom = float(CL_nom)
        self.T_nom = float(T_nom)
        self.Id_nom = float(Id_nom)
        self.Ich_nom = float(Ich_nom)
        self.SOCav_nom = float(SOCav_nom)
        self.DOD_nom = float(DOD_nom)
        self.enforce_feasible_region = bool(enforce_feasible_region)

        # Precompute denominators for normalization
        self._den_T = _ensure_positive(self._CL_T(self.T_nom))
        self._den_Id = _ensure_positive(self._CL_Id(self.Id_nom))
        self._den_Ich = _ensure_positive(self._CL_Ich(self.Ich_nom))
        self._den_soc_dod = _ensure_positive(self._CL4(self.DOD_nom, self.SOCav_nom))

    # -------------------------------------------------------------------------
    # Individual CL factor models (unnormalized), from the paper
    # -------------------------------------------------------------------------
    def _CL_T(self, T: float) -> float:
        a, b, c, d = self._a, self._b, self._c, self._d
        return a * T**3 - b * T**2 + c * T + d

    def _CL_Id(self, Id: float) -> float:
        e, f, g, h = self._e, self._f, self._g, self._h
        return e * math.exp(f * Id) + g * math.exp(h * Id)

    def _CL_Ich(self, Ich: float) -> float:
        m, n, o, p = self._m, self._n, self._o, self._p
        return m * math.exp(n * Ich) + o * math.exp(p * Ich)

    def _CL4(self, DOD: float, SOCav: float) -> float:
        """
        Constrained SOCav–DOD polynomial CL4(DOD, SOCav) as per Eq. (13).

        Eq. (13) is the same quadratic form as Eq. (10), but with r constrained.
        r is defined by Eq. (12):
            r = (u/(2v))*(s + 100u) - 200t
        """
        q, s, t, u, v = self._q, self._s, self._t, self._u, self._v
        r = (u / (2.0 * v)) * (s + 100.0 * u) - 200.0 * t
        return q + r * DOD + s * SOCav + t * DOD**2 + u * DOD * SOCav + v * SOCav**2

    # -------------------------------------------------------------------------
    # Normalized cycle life multipliers nCL (paper framework Eq. 3)
    # -------------------------------------------------------------------------
    def nCL_T(self, T: float) -> float:
        return _ensure_positive(self._CL_T(float(T))) / self._den_T

    def nCL_Id(self, Id: float) -> float:
        return _ensure_positive(self._CL_Id(float(Id))) / self._den_Id

    def nCL_Ich(self, Ich: float) -> float:
        return _ensure_positive(self._CL_Ich(float(Ich))) / self._den_Ich

    def nCL_SOCav_DOD(self, SOCav: float, DOD: float) -> float:
        SOCav = float(SOCav)
        DOD = float(DOD)

        if self.enforce_feasible_region and (not _in_feasible_soc_dod_region(SOCav, DOD)):
            raise ValueError(
                f"Infeasible SOCav/DOD combination: SOCav={SOCav}, DOD={DOD}. "
                "Feasible region requires DOD <= 2*SOCav and DOD <= 2*(100 - SOCav)."
            )

        return _ensure_positive(self._CL4(DOD, SOCav)) / self._den_soc_dod

    # -------------------------------------------------------------------------
    # Combined cycle life + degradation
    # -------------------------------------------------------------------------
    def cycle_life(self, *, T: float, Id: float, Ich: float, SOCav: float, DOD: float) -> float:
        """
        Combined cycle life per Eq. (3):
            CL = CL_nom * nCL(T) * nCL(Id) * nCL(Ich) * nCL(SOCav, DOD)
        """
        mult = (
            self.nCL_T(T)
            * self.nCL_Id(Id)
            * self.nCL_Ich(Ich)
            * self.nCL_SOCav_DOD(SOCav, DOD)
        )
        return _ensure_positive(self.CL_nom * mult)

    def degradation_per_cycle(self, *, T: float, Id: float, Ich: float, SOCav: float, DOD: float) -> float:
        """
        Fractional degradation per cycle, consistent with Eq. (24) style usage:
            D = 1 / CL
        """
        CL = self.cycle_life(T=T, Id=Id, Ich=Ich, SOCav=SOCav, DOD=DOD)
        return 1.0 / _ensure_positive(CL)


def static_degradation(Id, Ich, SoC_avg, DoD):
    """Alias for per-cycle degradation used in static estimations."""
    model = DegradationModel()

    d = model.degradation_per_cycle(T=25.0, Id=Id, Ich=Ich, SOCav=SoC_avg, DOD=DoD)
    print("Degradation per cycle:", d)
    print("Equivalent cycle life:", 1.0 / d)
    return d

"""
# Example usage
Id = 0.3  # Discharge current (C-rate)
Ich = 0.1  # Charge current (C-rate)
SoC = 60  # Average state of charge (%)
DoD = 80  # Depth of discharge (%)

degradation = degradation_per_cycle(Id, Ich, SoC, DoD)
print(f"Degradation for this cycle: {degradation:.6f}")
"""

class RainflowCounter:
    def __init__(self, step_duration=1.0, eps=1e-6):
        self.step_duration = step_duration
        self.eps = eps

        self.stack = []          # persistent turning point stack for rainflow
        self.tp_buffer = []      # last few SOC points for turning point detection
        self.last_soc = None     # for plateau handling
        self.index = 0           # global index counter

    def _maybe_add_turning_point(self, idx, soc):
        """
        Incremental turning point detection using your logic.
        Returns a list of newly detected turning points (0 or 1).
        """

        # 1. Plateau handling: ignore if change < eps
        if self.last_soc is not None and abs(soc - self.last_soc) <= self.eps:
            return []  # no turning point

        self.last_soc = soc
        self.tp_buffer.append((idx, soc))

        # Need at least 3 points to detect a turning point
        if len(self.tp_buffer) < 3:
            return []

        # Extract last three points
        (i1, s1), (i2, s2), (i3, s3) = self.tp_buffer[-3:]

        # Peak or trough detection
        is_peak = (s2 - s1 > self.eps) and (s2 - s3 > self.eps)
        is_trough = (s2 - s1 < -self.eps) and (s2 - s3 < -self.eps)

        if is_peak or is_trough:
            # Turning point detected at (i2, s2)
            return [(i2, s2)]

        return []

    def update(self, soc):
        """
        Feed one new SOC value.
        Returns list of newly closed cycles.
        """
        closed_cycles = []

        # 1. Detect turning points incrementally
        new_tps = self._maybe_add_turning_point(self.index, soc)
        self.index += 1

        # 2. Add turning points to rainflow stack
        for tp in new_tps:
            self.stack.append(tp)

            # Try to close cycles
            while len(self.stack) >= 4:
                r1 = abs(self.stack[-1][1] - self.stack[-2][1])
                r2 = abs(self.stack[-2][1] - self.stack[-3][1])
                r3 = abs(self.stack[-3][1] - self.stack[-4][1])

                if r2 <= r1 and r2 <= r3:
                    idx1, soc1 = self.stack[-3]
                    idx2, soc2 = self.stack[-2]

                    SoC_max = max(soc1, soc2)
                    SoC_min = min(soc1, soc2)
                    DoD = SoC_max - SoC_min

                    if DoD > self.eps:
                        SoC_avg = (SoC_max + SoC_min) / 2.0
                        delta_time = abs(idx2 - idx1) * self.step_duration

                        if delta_time <= 1e-12:
                            Id_cycle = Ich_cycle = 0.0
                        else:
                            if soc2 > soc1:
                                Ich_cycle = (soc2 - soc1) / delta_time
                                Id_cycle = 0.0
                            else:
                                Id_cycle = (soc1 - soc2) / delta_time
                                Ich_cycle = 0.0

                        closed_cycles.append((SoC_avg, DoD, Id_cycle, Ich_cycle))
                        

                    # Remove middle points
                    del self.stack[-3:-1]
                else:
                    break

        return closed_cycles


def rainflow_counting(soc_profile, step_duration=1.0, eps=1e-6):
    """Return all closed cycles for the provided SoC profile."""

    counter = RainflowCounter(step_duration=step_duration, eps=eps)
    closed_cycles = []
    for soc in soc_profile:
        closed_cycles.extend(counter.update(soc))
    return closed_cycles


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
