"""
AEMO Oracle Solver — perfect-foresight LP co-optimizer for energy + 8 FCAS services.

Provides the theoretical maximum profit (upper bound) for a battery participating
in the NEM energy spot market and all 8 FCAS services, given perfect knowledge
of future prices.

Two variants (same LP, different price sources):
  - Oracle_PT (price-taking): uses historical RRP/FCAS prices as-is.
  - Oracle_MI (market-impact): uses realized prices from the impact model (Phase 2+).

Formulation:
  Maximise Σ_t [ energy_revenue_t + fcas_revenue_t ]
  subject to SOC dynamics, charge/discharge rate limits, and FCAS + energy
  competing for the same battery headroom (matching _compute_fcas_enablement constraints).
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import numpy as np
import polars as pl
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, eye, vstack


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OracleResult:
    """Output of the Oracle LP solver for a single episode."""
    total_profit: float          # $/episode
    energy_revenue: float        # $/ep
    fcas_revenue: float          # $/ep
    total_dispatch_mwh: float    # total MWh discharged over episode
    total_charge_mwh: float      # total MWh charged over episode
    n_intervals: int
    optimal_dispatch: np.ndarray       # (T,) net dispatch MW (+ discharge, - charge)
    optimal_raise_bids: np.ndarray     # (T, 8) FCAS raise bid MW
    optimal_lower_bids: np.ndarray     # (T, 8) FCAS lower bid MW
    optimal_soc: np.ndarray            # (T+1,) SOC at start of each interval
    per_step_fcas_revenue: np.ndarray  # (T,) FCAS revenue per step
    per_step_energy_revenue: np.ndarray  # (T,) energy revenue per step
    solver_status: int
    solver_message: str


# ---------------------------------------------------------------------------
# FCAS service index constants (order matches the env's full_fcas action layout)
# ---------------------------------------------------------------------------

FCAS_SERVICES_RAISE = ['RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN', 'RAISEREG']
FCAS_SERVICES_LOWER = ['LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN', 'LOWERREG']
FCAS_SERVICES = FCAS_SERVICES_RAISE + FCAS_SERVICES_LOWER
N_FCAS = len(FCAS_SERVICES)          # 8
N_FCAS_RAISE = len(FCAS_SERVICES_RAISE)  # 4
N_FCAS_LOWER = len(FCAS_SERVICES_LOWER)  # 4


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class AEMOOracleSolver:
    """
    Perfect-foresight LP co-optimizer for battery energy + 8 FCAS markets.

    Builds a time-coupled LP over T 5-min intervals (default 1728 = 144 h)
    and solves with scipy.optimize.linprog (HiGHS).
    """

    def __init__(self,
                 battery_capacity: float = 10.0,    # MWh
                 max_battery_flow: float = 5.0,     # MW
                 step_duration: float = 0.5,        # hours
                 init_soc: float = 5.0,             # MWh
                 min_soc: float = 0.0,
                 max_soc: float = 10.0):
        self.capacity = battery_capacity
        self.max_flow = max_battery_flow
        self.step_h = step_duration
        self.init_soc = init_soc
        self.min_soc = min_soc
        self.max_soc = max_soc

    def solve(self, prices: pl.DataFrame, region: str = "",
              verbose: bool = False) -> OracleResult:
        """
        Solve the Oracle LP for a single episode.

        Args:
            prices: DataFrame with columns
                ['SETTLEMENTDATE', 'RRP', 'FCAS_RAISE6SEC', 'FCAS_RAISE60SEC',
                 'FCAS_RAISE5MIN', 'FCAS_RAISEREG',
                 'FCAS_LOWER6SEC', 'FCAS_LOWER60SEC', 'FCAS_LOWER5MIN', 'FCAS_LOWERREG']
            region: Optional region name for logging.
            verbose: Print solver progress.

        Returns:
            OracleResult with optimal trajectory and total profit.
        """
        T = prices.height
        if T == 0:
            raise ValueError("Empty prices DataFrame")

        # ---- extract price vectors ----
        rrp = prices['RRP'].to_numpy().astype(float)              # (T,)
        fcas_prices = np.zeros((T, N_FCAS), dtype=float)
        for i, svc in enumerate(FCAS_SERVICES):
            col = f'FCAS_{svc}'
            if col in prices.columns:
                fcas_prices[:, i] = prices[col].to_numpy().astype(float)

        # ---- variable indexing ----
        # Per interval t ∈ [0, T-1]:
        #   charge[t]        idx = t*VARS_PER_T + 0
        #   discharge[t]     idx = t*VARS_PER_T + 1
        #   soc[t]           idx = t*VARS_PER_T + 2
        #   raise_bid[t*4 + r]  for r=0..3  idx = t*VARS_PER_T + 3 + r
        #   lower_bid[t*4 + l]  for l=0..3  idx = t*VARS_PER_T + 7 + l
        #   raise_hdrm[t]    idx = t*VARS_PER_T + 11
        #   lower_hdrm[t]    idx = t*VARS_PER_T + 12
        # Total vars per interval:
        N = 2 + 1 + N_FCAS + 2  # = 13
        # Additional variable: soc[T] (terminal SOC after last interval)
        total_vars = T * N + 1  # +1 for terminal soc[T]
        soc_T_idx = T * N

        # Helper: extract slice for variable v at timestep t
        def idx(t, v): return t * N + v

        I_C = 0   # charge
        I_D = 1   # discharge
        I_S = 2   # soc (at START of interval)
        IR = 3    # raise bids  (offset, 4 vars)
        IL = 7    # lower bids  (offset, 4 vars)
        I_RH = 11 # raise headroom
        I_LH = 12 # lower headroom

        # ---- bounds ----
        ub = np.full(total_vars, np.inf)
        lb = np.zeros(total_vars)
        for t in range(T):
            ub[idx(t, I_C)] = self.max_flow
            ub[idx(t, I_D)] = self.max_flow
            ub[idx(t, I_S)] = self.max_soc
            lb[idx(t, I_S)] = self.min_soc
            for r in range(N_FCAS_RAISE):
                ub[idx(t, IR + r)] = self.max_flow
            for l in range(N_FCAS_LOWER):
                ub[idx(t, IL + l)] = self.max_flow
            ub[idx(t, I_RH)] = self.max_flow
            ub[idx(t, I_LH)] = self.max_flow
        ub[soc_T_idx] = self.max_soc
        lb[soc_T_idx] = self.min_soc

        # ---- objective (maximize profit → minimize -profit) ----
        c = np.zeros(total_vars)
        for t in range(T):
            step_mwh = self.step_h
            # Energy revenue: discharge * RRP * step_h - charge * RRP * step_h
            # (sell at RRP when discharging, pay RRP when charging)
            c[idx(t, I_D)] = -rrp[t] * step_mwh   # negative → minimize -profit
            c[idx(t, I_C)] = rrp[t] * step_mwh

            # FCAS revenue
            for r in range(N_FCAS_RAISE):
                price = fcas_prices[t, r]
                c[idx(t, IR + r)] = -price * step_mwh
            for l in range(N_FCAS_LOWER):
                price = fcas_prices[t, N_FCAS_RAISE + l]
                c[idx(t, IL + l)] = -price * step_mwh
        # c is the gradient of -profit, so linprog minimizes c^T x = -profit

        # ---- equality constraints ----
        # SOC(t+1) = SOC(t) - (discharge - charge) * step_h
        # => SOC(t+1) - SOC(t) + discharge*step_h - charge*step_h = 0
        A_eq_rows = []
        b_eq = []
        row = 0
        init_soc = self.init_soc

        for t in range(T):
            # At t=0: soc[0] is fixed. But soc[0] is the START SOC of interval 0.
            # We add a constraint: soc[0] = init_soc (via equality constraint)
            if t == 0:
                r = np.zeros(total_vars)
                r[idx(0, I_S)] = 1.0
                A_eq_rows.append(csr_matrix(r))
                b_eq.append(init_soc)
                row += 1

            # SOC update: soc[t+1] = soc[t] - (discharge[t] - charge[t]) * step_h
            r = np.zeros(total_vars)
            next_soc_idx = idx(t+1, I_S) if t+1 < T else soc_T_idx
            r[next_soc_idx] = 1.0              # soc[t+1]
            r[idx(t, I_S)] = -1.0              # -soc[t]
            r[idx(t, I_D)] = self.step_h       # + discharge * step_h
            r[idx(t, I_C)] = -self.step_h      # - charge * step_h
            A_eq_rows.append(csr_matrix(r))
            b_eq.append(0.0)
            row += 1

        A_eq = vstack(A_eq_rows) if A_eq_rows else csr_matrix((0, total_vars))
        b_eq = np.array(b_eq)

        # ---- inequality constraints (A_ub @ x <= b_ub) ----
        A_ub_rows = []
        b_ub = []

        for t in range(T):
            prev_soc_idx = idx(t, I_S)
            soc_val = None  # not directly used in LP (it's a variable)

            # 1) sum(raise_bid) + discharge_t <= raise_headroom_t
            #    (env: discharging consumes raise headroom; raise_used = max(0, -actual_power))
            r = np.zeros(total_vars)
            r[idx(t, I_RH)] = -1.0
            r[idx(t, I_D)] = 1.0
            for rv in range(N_FCAS_RAISE):
                r[idx(t, IR + rv)] = 1.0
            A_ub_rows.append(csr_matrix(r))
            b_ub.append(0.0)

            # 2) sum(lower_bid) + charge_t <= lower_headroom_t
            #    (env: charging consumes lower headroom; lower_used = max(0, actual_power))
            r = np.zeros(total_vars)
            r[idx(t, I_LH)] = -1.0
            r[idx(t, I_C)] = 1.0
            for lv in range(N_FCAS_LOWER):
                r[idx(t, IL + lv)] = 1.0
            A_ub_rows.append(csr_matrix(r))
            b_ub.append(0.0)

            # 3) raise_headroom <= soc[t] / step_duration
            #    => raise_headroom * step_h - soc[t] <= 0
            if self.step_h > 0:
                r = np.zeros(total_vars)
                r[idx(t, I_RH)] = self.step_h
                r[prev_soc_idx] = -1.0
                A_ub_rows.append(csr_matrix(r))
                b_ub.append(0.0)

            # 4) lower_headroom <= (capacity - soc[t]) / step_duration
            #    => lower_headroom * step_h + soc[t] <= capacity
            if self.step_h > 0:
                r = np.zeros(total_vars)
                r[idx(t, I_LH)] = self.step_h
                r[prev_soc_idx] = 1.0
                A_ub_rows.append(csr_matrix(r))
                b_ub.append(self.capacity)

        A_ub = vstack(A_ub_rows) if A_ub_rows else csr_matrix((0, total_vars))
        b_ub = np.array(b_ub)

        # ---- solve ----
        if verbose:
            print(f"[Oracle] Solving LP: {total_vars} vars, "
                  f"{len(b_eq)} eq + {len(b_ub)} ineq constraints "
                  f"({region}, {T} intervals)")

        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=list(zip(lb, ub)),
            method='highs',
            options={'disp': verbose, 'time_limit': 300}
        )

        if verbose:
            print(f"[Oracle] Status {res.status}: {res.message}")

        # ---- unpack solution ----
        if not (res.success or res.status == 4):
            print(f"[Oracle] Solver failed: {res.message}")
            return OracleResult(
                total_profit=-1e12, energy_revenue=0.0, fcas_revenue=0.0,
                total_dispatch_mwh=0.0, total_charge_mwh=0.0,
                n_intervals=T,
                optimal_dispatch=np.zeros(T),
                optimal_raise_bids=np.zeros((T, N_FCAS_RAISE)),
                optimal_lower_bids=np.zeros((T, N_FCAS_LOWER)),
                optimal_soc=np.full(T+1, self.init_soc),
                per_step_fcas_revenue=np.zeros(T),
                per_step_energy_revenue=np.zeros(T),
                solver_status=res.status, solver_message=res.message,
            )

        x = res.x
        profit = -res.fun  # negate because we minimized -profit

        dispatch = np.array([x[idx(t, I_D)] - x[idx(t, I_C)] for t in range(T)])
        raise_bid = np.array([[x[idx(t, IR + r)] for r in range(N_FCAS_RAISE)]
                               for t in range(T)])
        lower_bid = np.array([[x[idx(t, IL + l)] for l in range(N_FCAS_LOWER)]
                               for t in range(T)])

        # Recompute SOC from variables
        soc = np.zeros(T + 1)
        soc[0] = x[idx(0, I_S)]
        for t in range(1, T + 1):
            soc[t] = (x[idx(t-1, I_S)]
                      - (x[idx(t-1, I_D)] - x[idx(t-1, I_C)]) * self.step_h)

        # Per-step revenues
        energy_rev = np.array([
            (x[idx(t, I_D)] - x[idx(t, I_C)]) * rrp[t] * self.step_h
            for t in range(T)
        ])
        fcas_rev = np.array([
            sum(x[idx(t, IR + r)] * fcas_prices[t, r]
                for r in range(N_FCAS_RAISE)) * self.step_h
            + sum(x[idx(t, IL + l)] * fcas_prices[t, N_FCAS_RAISE + l]
                  for l in range(N_FCAS_LOWER)) * self.step_h
            for t in range(T)
        ])

        # Sanity: check SOC bounds
        if np.any(soc < -0.01) or np.any(soc > self.max_soc + 0.01):
            print(f"  [Oracle] WARNING: SOC out of bounds [{soc.min():.2f}, {soc.max():.2f}]")

        total_dispatch_mwh = float(dispatch[dispatch > 0].sum() * self.step_h)
        total_charge_mwh = float((-dispatch[dispatch < 0]).sum() * self.step_h)

        result = OracleResult(
            total_profit=float(profit),
            energy_revenue=float(energy_rev.sum()),
            fcas_revenue=float(fcas_rev.sum()),
            total_dispatch_mwh=total_dispatch_mwh,
            total_charge_mwh=total_charge_mwh,
            n_intervals=T,
            optimal_dispatch=dispatch,
            optimal_raise_bids=raise_bid,
            optimal_lower_bids=lower_bid,
            optimal_soc=soc,
            per_step_fcas_revenue=fcas_rev,
            per_step_energy_revenue=energy_rev,
            solver_status=res.status,
            solver_message=res.message,
        )

        if verbose:
            print(f"  Profit: ${result.total_profit:,.0f}/ep (energy ${result.energy_revenue:,.0f} + FCAS ${result.fcas_revenue:,.0f})")
            print(f"  Dispatch: {total_dispatch_mwh:.0f} MWh, charge: {total_charge_mwh:.0f} MWh")

        return result
