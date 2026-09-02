import math
from typing import Optional

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


def _safe_den(value: float, name: str) -> float:
    """
    Safe denominator guard for RL environments.
    If the nominal denominator is invalid, fall back to 1.0 (neutral multiplier).
    This avoids exceptions and avoids catastrophic degradation.
    """
    if not np.isfinite(value) or value <= 0.0:
        # Log once if you want, but do NOT raise
        print(f"[WARN] Invalid denominator {name}={value}, using 1.0 fallback")
        return 1.0
    return value



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
        self._den_T = _safe_den(self._CL_T(self.T_nom), "CL_T(T_nom)")
        self._den_Id = _safe_den(self._CL_Id(self.Id_nom), "CL_Id(Id_nom)")
        self._den_Ich = _safe_den(self._CL_Ich(self.Ich_nom), "CL_Ich(Ich_nom)")
        self._den_soc_dod = _safe_den(self._CL4(self.DOD_nom, self.SOCav_nom), "CL4(DOD_nom, SOCav_nom)")


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
        raw = self._CL_T(float(T))
        if raw <= 0.0 or not np.isfinite(raw):
            # temperature outside fit → treat as no degradation change
            return 1.0
        return raw / self._den_T

    def nCL_Id(self, Id: float) -> float:
        raw = self._CL_Id(float(Id))
        if raw <= 0.0 or not np.isfinite(raw):
            return 1.0
        return raw / self._den_Id

    def nCL_Ich(self, Ich: float) -> float:
        raw = self._CL_Ich(float(Ich))
        if raw <= 0.0 or not np.isfinite(raw):
            return 1.0
        return raw / self._den_Ich

    def nCL_SOCav_DOD(self, SOCav: float, DOD: float) -> float:
        SOCav = float(SOCav)
        DOD = float(DOD)

        if self.enforce_feasible_region and (not _in_feasible_soc_dod_region(SOCav, DOD)):
            # outside feasible region → clamp in degradation_per_cycle, not here
            raise ValueError(
                f"Infeasible SOCav/DOD combination: SOCav={SOCav}, DOD={DOD}. "
                "Feasible region requires DOD <= 2*SOCav and DOD <= 2*(100 - SOCav)."
            )

        CL4_raw = self._CL4(DOD, SOCav)

        # Key change: negative or tiny CL4 means "model not reliable here" → no extra wear
        if (not np.isfinite(CL4_raw)) or CL4_raw <= 0.0:
            return 1.0

        return CL4_raw / self._den_soc_dod

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
        return self.CL_nom * mult

    def degradation_per_cycle(
        self,
        *,
        T: float,
        Id: float,
        Ich: float,
        SOCav: float,
        DOD: float,
    ) -> float:
        """
        Robust, paper-faithful degradation per cycle.

        - Enforces SOC–DOD feasibility
        - Caps C-rates to physically meaningful bounds
        - Rejects invalid cycle life instead of masking it
        """

        # ---- Clamp SOC/DOD to feasible region (Eqs. 14–17) ----
        SOCav = float(np.clip(SOCav, 0.0, 100.0))
        DOD = float(max(DOD, 0.0))
        DOD = min(DOD, 2.0 * SOCav, 2.0 * (100.0 - SOCav))

        if DOD <= 3.0 or (DOD < 5.0 and SOCav > 96.0):
            return 0.0  # no cycle / known instability at high SOC → no degradation 

        # ---- Cap C-rates (paper data valid up to ~3C) ----
        Id = float(min(max(Id, 0.0), 3.0))
        Ich = float(min(max(Ich, 0.0), 3.0))

        # ---- Compute cycle life ----
        CL = self.cycle_life(
            T=T,
            Id=Id,
            Ich=Ich,
            SOCav=SOCav,
            DOD=DOD,
        )

        # ---- Explicit validation instead of masking ----
        if not np.isfinite(CL) or CL <= 0.0:
            raise ValueError(
                f"Invalid cycle life detected: "
                f"CL={CL}, SOCav={SOCav}, DOD={DOD}, Id={Id}, Ich={Ich}"
            )

        return 1.0 / CL
    
    def debug_degradation_per_cycle(
        self,
        *,
        T: float,
        Id: float,
        Ich: float,
        SOCav: float,
        DOD: float,
    ) -> dict:
        SOCav_c = float(np.clip(SOCav, 0.0, 100.0))
        DOD_c = float(max(DOD, 0.0))
        DOD_c = min(DOD_c, 2.0 * SOCav_c, 2.0 * (100.0 - SOCav_c))

        Id_c = float(min(max(Id, 0.0), 5.0))
        Ich_c = float(min(max(Ich, 0.0), 5.0))

        nT = self.nCL_T(T)
        nId = self.nCL_Id(Id_c)
        nIch = self.nCL_Ich(Ich_c)
        CL4 = self._CL4(DOD_c, SOCav_c)
        nSOC = self.nCL_SOCav_DOD(SOCav_c, DOD_c)

        mult = nT * nId * nIch * nSOC
        CL = self.CL_nom * mult

        return {
            "SOCav_in": SOCav,
            "DOD_in": DOD,
            "SOCav": SOCav_c,
            "DOD": DOD_c,
            "Id": Id_c,
            "Ich": Ich_c,
            "nCL_T": nT,
            "nCL_Id": nId,
            "nCL_Ich": nIch,
            "CL4_raw": CL4,
            "nCL_SOCav_DOD": nSOC,
            "mult": mult,
            "CL": CL,
            "degradation": None if CL <= 0 else 1.0 / CL,
        }


class RainflowCounter:
    def __init__(self, step_duration=1.0, eps=0.1, max_c_rate=1.0): # eps in percent SoC, smaller than 0.1% is ignored
        self.step_duration = step_duration
        self.eps = eps          # tolerance for turning point detection

        self.stack = []          # persistent turning point stack for rainflow
        self.tp_buffer = []      # last few SOC points for turning point detection
        self.last_soc = None     # for plateau handling
        self.index = 0           # global index counter
        self.max_c_rate = max_c_rate

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
                                # conduct clamping based on max C-rate
                                Ich_cycle = min(Ich_cycle, self.max_c_rate)
                                Id_cycle = 0.0
                            else:
                                Id_cycle = (soc1 - soc2) / delta_time
                                # conduct clamping based on max C-rate
                                Id_cycle = min(Id_cycle, self.max_c_rate)
                                Ich_cycle = 0.0

                        closed_cycles.append((SoC_avg, DoD, Id_cycle, Ich_cycle))
                        

                    # Remove middle points
                    del self.stack[-3:-1]
                else:
                    break

        return closed_cycles



# =============================================================================
# Real-World BESS Degradation Model
# =============================================================================
#
# Based on the modeling framework presented in:
#   "Modelling of Battery Energy Storage Systems Under Real-World Applications
#    and Conditions", MDPI Batteries 11(11):392, 2025
#   doi: 10.3390/batteries11110392
#
# Key differences from Muenzel et al. (2015) / current DegradationModel:
#
#   1. CALENDAR AGING — not present in Muenzel (2015).
#      Grid-scale BESS sits idle for many hours (overnight, weekends).
#      Calendar aging from SEI growth is a major degradation pathway that is
#      completely absent in the existing model, making it unsuitable for
#      utility-scale long-run simulations.
#
#   2. ARRHENIUS TEMPERATURE DEPENDENCY — physically grounded.
#      Muenzel (2015) uses an empirical cubic polynomial CL(T) fitted to
#      small lab cells.  The paper uses the Arrhenius equation:
#          rate(T) ∝ exp(−Ea / (R·T))
#      which is valid over a wide temperature range and is standard for
#      electrochemical aging models.
#
#   3. POWER-LAW DOD / C-RATE FOR CYCLE AGING — more compact and
#      interpretable than the two-dimensional constrained polynomial in
#      Muenzel (2015) and better calibrated to grid-scale NMC/LFP cells.
#
#   4. CELL-CHEMISTRY PRESETS — NMC (default) and LFP, the two dominant
#      chemistries in utility-scale BESS deployments (e.g., Tesla Megapack,
#      BYD Battery Box).
#
# Model equations
# ---------------
# Total capacity loss (fraction of nominal capacity):
#   Q_loss = Q_cal + Q_cyc  (capped at 1.0)
#
# Calendar aging per timestep Δt (hours):
#   ΔQ_cal = k_cal · exp(−Ea_cal/(R·T_K)) / exp(−Ea_cal/(R·T_ref_K))
#            · [1 + k_soc · (soc_frac − 0.5)] · Δt
#   where soc_frac ∈ [0, 1] is the current state of charge / capacity.
#
# Cycle aging per detected cycle (from rainflow counting):
#   ΔQ_cyc = k_cyc · (DOD/100)^α · (1 + β_c · C_rate)
#            · exp(−Ea_cyc/(R·T_K)) / exp(−Ea_cyc/(R·T_ref_K))
#
# Battery capacity:
#   C(t) = C_nom · (1 − Q_loss)
#
# =============================================================================

#: Universal gas constant (J mol⁻¹ K⁻¹)
_R_GAS = 8.314
#: Reference temperature (K) — 25 °C
_T_REF_K = 298.15


def _arrhenius(Ea_J_per_mol: float, T_K: float) -> float:
    """Normalised Arrhenius factor (= 1.0 at T_ref = 25 °C)."""
    return math.exp(-Ea_J_per_mol / (_R_GAS * T_K)) / math.exp(
        -Ea_J_per_mol / (_R_GAS * _T_REF_K)
    )


#: Chemistry preset parameter dictionaries.
#: Each entry defines the five parameters used by RealWorldBESSDegradationModel.
BESS_CHEMISTRY_PRESETS: dict = {
    # NMC (Nickel-Manganese-Cobalt) — common in early utility deployments.
    # Calendar life (calendar-only, 25 °C, 50 % SOC): EOL ~12–15 years
    # Cycle life (100 % DoD, 1C, 25 °C): ~2 000 cycles
    # Sources: Wang et al. (2011), Petit et al. (2016), Hesse et al. (2017).
    "NMC": dict(
        k_cal_rate=2.85e-6,   # capacity loss fraction per hour at T_ref, SOC=50 %
        Ea_cal=28_500.0,      # J/mol  — calendar aging activation energy
        k_soc=0.5,            # SOC stress coefficient (linear)
        k_cyc=3.5e-4,         # capacity loss fraction per full-DoD cycle at T_ref
        Ea_cyc=17_100.0,      # J/mol  — cycle aging activation energy
        alpha_dod=1.2,        # DoD power-law exponent
        beta_crate=0.5,       # C-rate linear sensitivity factor
    ),
    # LFP (Lithium-Iron-Phosphate) — preferred for current utility BESS
    # (e.g., Tesla Megapack Gen 3, BYD).  Very cycle-stable, moderate calendar.
    # Calendar life (25 °C, 50 % SOC): EOL ~20+ years
    # Cycle life (100 % DoD, 1C, 25 °C): ~4 000–6 000 cycles
    # Sources: Naumann et al. (2017), Xu et al. (2016).
    "LFP": dict(
        k_cal_rate=1.20e-6,   # slower calendar aging
        Ea_cal=17_500.0,      # J/mol  — LFP less temperature-sensitive
        k_soc=0.2,            # LFP nearly SOC-insensitive for calendar aging
        k_cyc=1.95e-4,        # capacity loss per full-DoD cycle at T_ref
        Ea_cyc=10_000.0,      # J/mol  — very low temp sensitivity for cycling
        alpha_dod=0.5,        # shallower power-law: LFP tolerates partial cycling
        beta_crate=0.3,       # lower C-rate sensitivity
    ),
}


class RealWorldBESSDegradationModel:
    """
    Combined calendar + cycle aging model for utility-scale BESS.

    Designed for the AEMO environment where:
    - Episodes may span many idle hours (calendar aging matters).
    - Australian outdoor temperatures can deviate significantly from 25 °C.
    - Dominant chemistries are NMC and LFP.

    Parameters
    ----------
    chemistry : str
        One of ``'NMC'`` or ``'LFP'`` (case-insensitive).  Selects the
        corresponding parameter preset from ``BESS_CHEMISTRY_PRESETS``.
        Ignored when any explicit parameter is supplied.
    k_cal_rate : float
        Calendar aging rate [capacity loss fraction per hour] at the
        reference temperature (25 °C) and 50 % SOC.
    Ea_cal : float
        Activation energy for calendar aging [J/mol].
    k_soc : float
        SOC stress coefficient for calendar aging.  The SOC multiplier is
        ``1 + k_soc × (soc_frac − 0.5)`` so values > 0 increase degradation
        at high SOC and decrease it at low SOC.
    k_cyc : float
        Cycle aging coefficient [capacity loss fraction per full-DoD cycle]
        at the reference temperature (25 °C) and 1C charge/discharge rate.
    Ea_cyc : float
        Activation energy for cycle aging [J/mol].
    alpha_dod : float
        Power-law exponent for DoD in cycle aging (``≥ 0``).
    beta_crate : float
        Linear C-rate sensitivity factor for cycle aging (``≥ 0``).

    Examples
    --------
    >>> model = RealWorldBESSDegradationModel(chemistry='LFP')
    >>> # Calendar aging: 30-minute step at 35 °C, 80 % SOC
    >>> model.calendar_aging_per_step(T_celsius=35.0, soc_frac=0.8, dt_hours=0.5)
    >>> # Cycle aging: one cycle at 80 % DoD, 0.5C, 25 °C
    >>> model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=80.0, c_rate=0.5)
    """

    def __init__(
        self,
        chemistry: str = "NMC",
        *,
        k_cal_rate: Optional[float] = None,
        Ea_cal: Optional[float] = None,
        k_soc: Optional[float] = None,
        k_cyc: Optional[float] = None,
        Ea_cyc: Optional[float] = None,
        alpha_dod: Optional[float] = None,
        beta_crate: Optional[float] = None,
    ):
        key = chemistry.upper()
        if key not in BESS_CHEMISTRY_PRESETS:
            raise ValueError(
                f"Unknown chemistry '{chemistry}'. "
                f"Choose from: {list(BESS_CHEMISTRY_PRESETS.keys())}"
            )
        preset = BESS_CHEMISTRY_PRESETS[key]

        self.chemistry = key
        self.k_cal_rate: float = float(k_cal_rate if k_cal_rate is not None else preset["k_cal_rate"])
        self.Ea_cal: float = float(Ea_cal if Ea_cal is not None else preset["Ea_cal"])
        self.k_soc: float = float(k_soc if k_soc is not None else preset["k_soc"])
        self.k_cyc: float = float(k_cyc if k_cyc is not None else preset["k_cyc"])
        self.Ea_cyc: float = float(Ea_cyc if Ea_cyc is not None else preset["Ea_cyc"])
        self.alpha_dod: float = float(alpha_dod if alpha_dod is not None else preset["alpha_dod"])
        self.beta_crate: float = float(beta_crate if beta_crate is not None else preset["beta_crate"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calendar_aging_per_step(
        self, T_celsius: float, soc_frac: float, dt_hours: float
    ) -> float:
        """
        Incremental calendar capacity loss for one simulation timestep.

        Parameters
        ----------
        T_celsius : float
            Ambient / cell temperature in degrees Celsius.
        soc_frac : float
            State of charge as a fraction of nominal capacity in [0, 1].
        dt_hours : float
            Timestep duration in hours.

        Returns
        -------
        float
            Fractional capacity loss (≥ 0).  Add to running ``Q_cal``.
        """
        T_K = T_celsius + 273.15
        arr = _arrhenius(self.Ea_cal, T_K)

        soc_frac_clipped = float(np.clip(soc_frac, 0.0, 1.0))
        soc_stress = max(0.0, 1.0 + self.k_soc * (soc_frac_clipped - 0.5))

        return self.k_cal_rate * arr * soc_stress * max(0.0, dt_hours)

    def cycle_aging_per_cycle(
        self, T_celsius: float, dod_pct: float, c_rate: float
    ) -> float:
        """
        Capacity loss for one detected charge/discharge cycle.

        Parameters
        ----------
        T_celsius : float
            Temperature in degrees Celsius during the cycle.
        dod_pct : float
            Depth of discharge of the cycle in percent [0, 100].
        c_rate : float
            Equivalent C-rate of the cycle (≥ 0).

        Returns
        -------
        float
            Fractional capacity loss per cycle (≥ 0).
        """
        if dod_pct <= 0.0:
            return 0.0

        T_K = T_celsius + 273.15
        arr = _arrhenius(self.Ea_cyc, T_K)

        dod_frac = float(np.clip(dod_pct, 0.0, 100.0)) / 100.0
        dod_factor = dod_frac ** self.alpha_dod

        c_rate_clamped = float(max(0.0, c_rate))
        c_rate_factor = 1.0 + self.beta_crate * c_rate_clamped

        return self.k_cyc * arr * dod_factor * c_rate_factor



