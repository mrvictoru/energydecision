"""
AEMO SDP/MPC executor — the honest (non-clairvoyant) planner for Stage A.

Stage A replaces the perfect-foresight Oracle LP inside the hierarchical
``dt_soc_oracle`` policy with a stochastic dynamic program that plans energy
dispatch against a *seasonal forecast* of RRP (built only from training-era
data), plus a greedy current-price FCAS bidder. No future prices are used.

Two-step split:
  1. **Energy / SOC timing** — SDP backward induction over a seasonal
     time-of-day RRP profile, pinned to reach the waypoint target SOC.
  2. **FCAS bidding** — greedy per-step allocation of the residual headroom
     (after energy dispatch) to the highest-priced FCAS service, using only
     *current* FCAS prices (spikes are unpredictable, so a myopic bidder is
     the honest choice).

The seasonal RRP profile is the "predictable component" of the price process
(diurnal + monthly structure); the unpredictable spike component is
deliberately averaged away. This is the honest analogue of the perfect-
foresight LP and directly lifts the foresight caveat documented in
``docs/aemo_dt_preferred_policy_plan.md``.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import polars as pl

from aemo_sdp_solver import AEMOSDPSolver

# Env FCAS service order (matches AEMOBatteryTradingEnv._fcas_services).
FCAS_ORDER = ["RAISEREG", "LOWERREG", "RAISE6SEC", "LOWER6SEC",
              "RAISE60SEC", "LOWER60SEC", "RAISE5MIN", "LOWER5MIN"]
RAISE_SERVICES = ["RAISEREG", "RAISE6SEC", "RAISE60SEC", "RAISE5MIN"]
LOWER_SERVICES = ["LOWERREG", "LOWER6SEC", "LOWER60SEC", "LOWER5MIN"]


# ---------------------------------------------------------------------------
# Seasonal RRP profile (honest forecast source)
# ---------------------------------------------------------------------------

def _parse_dates(filename: str) -> tuple[str, str] | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", filename)
    if not m:
        return None
    return m.group(1), m.group(2)


def find_training_parquet(cache_dir: Path, region: str) -> Path | None:
    """Pick the longest training-era (pre-2024) processed parquet for a region."""
    pattern = str(cache_dir / f"processed_{region}_*_0.0833h.parquet")
    best: tuple[int, Path] | None = None
    for p in glob.glob(pattern):
        path = Path(p)
        dates = _parse_dates(path.name)
        if dates is None:
            continue
        start, end = dates
        if start >= "2024":
            continue  # must not use eval-era data for the forecast
        # score by date span (rough: string length is fixed, use lexicographic span)
        span = (end >= start)
        if not span:
            continue
        # crude span length: days between
        import datetime
        d0 = datetime.datetime.strptime(start, "%Y-%m-%d")
        d1 = datetime.datetime.strptime(end, "%Y-%m-%d")
        days = (d1 - d0).days
        if best is None or days > best[0]:
            best = (days, path)
    return best[1] if best is not None else None


def build_seasonal_rrp_profile(
    cache_dir: Path, region: str, step_duration: float = 5.0 / 60.0
) -> Callable[[int, int], float]:
    """Return a callable profile(month, hour) -> mean RRP for a region.

    Computed from the longest training-era (pre-2024) processed parquet. The
    profile is cached to ``data/aemo_sdp/seasonal_rrp_{region}.json`` so it is
    only computed once.
    """
    out_dir = cache_dir.parent / "aemo_sdp"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"seasonal_rrp_{region}.json"

    if cache_path.is_file():
        data = json.loads(cache_path.read_text())
        table = {int(k): float(v) for k, v in data.items()}
        return lambda month, hour: table.get(month * 100 + hour, table.get(0, 50.0))

    src = find_training_parquet(cache_dir, region)
    if src is None:
        raise FileNotFoundError(f"No pre-2024 training parquet found for {region}")

    df = pl.read_parquet(src, columns=["SETTLEMENTDATE", "RRP"])
    df = df.with_columns(
        pl.col("SETTLEMENTDATE").dt.month().alias("month"),
        pl.col("SETTLEMENTDATE").dt.hour().alias("hour"),
    )
    prof = df.group_by(["month", "hour"]).agg(pl.col("RRP").mean().alias("rrp_mean"))
    table: dict[int, float] = {}
    for row in prof.iter_rows(named=True):
        table[int(row["month"]) * 100 + int(row["hour"])] = float(row["rrp_mean"])

    cache_path.write_text(json.dumps({str(k): v for k, v in table.items()}))
    return lambda month, hour: table.get(month * 100 + hour, 50.0)


def build_rrp_forecast(
    aemo_data: pl.DataFrame,
    profile: Callable[[int, int], float],
) -> list[dict]:
    """Build a per-step honest RRP forecast (seasonal mean) from episode times."""
    months = aemo_data["SETTLEMENTDATE"].dt.month().to_numpy()
    hours = aemo_data["SETTLEMENTDATE"].dt.hour().to_numpy()
    forecast = []
    for m, h in zip(months, hours):
        forecast.append({"RRP": float(profile(int(m), int(h)))})
    return forecast


# ---------------------------------------------------------------------------
# SDP energy dispatch (backward induction pinned to a target SOC)
# ---------------------------------------------------------------------------

def sdp_energy_dispatch(
    env,
    forecast: Sequence[dict],
    start_soc: float,
    target_soc: float,
    soc_resolution: int = 40,
    action_resolution: int = 41,
    terminal_penalty: float = 1e6,
    deg_cost_per_mwh: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan energy dispatch from start_soc to target_soc over ``forecast`` steps.

    Runs SDP backward induction with a soft terminal cost (quadratic in the
    SOC distance from ``target_soc``) so the plan is pinned to reach the target
    waypoint while remaining robust to SOC discretisation. Returns
    (energy_per_step, soc_trajectory), both in MWh; energy > 0 = charging,
    < 0 = discharging (env convention).

    This is the honest energy planner: it uses only the *seasonal forecast*
    of RRP (no realized future prices). Degradation is handled two ways: the
    repo's rainflow DegradationCalculator (which returns ~0 for sub-3% DoD
    transitions, so it under-counts cycling) PLUS a linear throughput
    surrogate ``deg_cost_per_mwh`` (|energy| * \$/MWh) so multi-step daily
    cycling is priced — matching the RealWorldBESS cycle aging the env
    actually charges.
    """
    capacity = float(env.battery_capacity)
    horizon = len(forecast)
    if horizon == 0:
        return np.array([]), np.array([start_soc])

    solver = AEMOSDPSolver(
        env=env,
        horizon=horizon,
        soc_resolution=soc_resolution,
        action_resolution=action_resolution,
        use_monte_carlo=False,
    )
    soc_levels = solver.soc_levels_kwh
    action_energies = solver.battery_flow_energies
    n_soc = len(soc_levels)

    # Linear throughput degradation surrogate (per action, broadcast over SOC).
    linear_deg = np.abs(action_energies)[None, :] * deg_cost_per_mwh

    # Terminal cost: soft quadratic pin to target_soc.
    cost_to_go = np.full((horizon + 1, n_soc), np.inf)
    cost_to_go[horizon, :] = terminal_penalty * ((soc_levels - target_soc) / max(capacity, 1e-9)) ** 2
    policy = np.full((horizon, n_soc), -1, dtype=int)

    for t in range(horizon - 1, -1, -1):
        stage = solver._compute_stage_costs(forecast[t], 0, None)
        stage = stage + linear_deg  # add linear throughput degradation
        future = solver._compute_future_costs(cost_to_go[t + 1, :])
        total = np.where(np.isfinite(stage) & np.isfinite(future), stage + future, np.inf)
        best = np.argmin(total, axis=1)
        row_min = np.min(total, axis=1)
        finite = np.isfinite(row_min)
        policy[t, :] = -1
        policy[t, finite] = best[finite]
        cost_to_go[t, :] = row_min

    # Roll out the policy from start_soc.
    energy = np.zeros(horizon, dtype=float)
    soc = np.zeros(horizon + 1, dtype=float)
    soc[0] = float(np.clip(start_soc, 0.0, capacity))
    for t in range(horizon):
        s_idx = int(np.argmin(np.abs(soc_levels - soc[t])))
        a_idx = policy[t, s_idx]
        if a_idx < 0:
            a_idx = 0
        e = float(action_energies[a_idx])
        # clip to capacity
        e = float(np.clip(e, -soc[t], capacity - soc[t]))
        energy[t] = e
        soc[t + 1] = soc[t] + e

    return energy, soc


# ---------------------------------------------------------------------------
# Greedy current-price FCAS bidder
# ---------------------------------------------------------------------------

def greedy_fcas_bids(
    env,
    dispatch_mw: float,
    soc_before: float,
    fcas_prices: dict[str, float],
) -> list[float]:
    """Greedy FCAS bids (8 fractions) given energy dispatch + current prices.

    dispatch_mw > 0 = charging, < 0 = discharging (env convention). Residual
    raise/lower headroom (after the energy action) is allocated to the single
    highest-priced raise / lower service respectively. Returns a dict mapping
    FCAS service -> bid fraction in [0,1]. Honest: current prices only.
    """
    max_flow = float(env.max_battery_flow)
    step_h = float(env.step_duration)
    capacity = float(env.battery_capacity)

    max_raise = min(max_flow, soc_before / step_h) if step_h > 0 else max_flow
    max_lower = min(max_flow, (capacity - soc_before) / step_h) if step_h > 0 else max_flow
    raise_used = max(0.0, -dispatch_mw)
    lower_used = max(0.0, dispatch_mw)
    raise_headroom = max(0.0, max_raise - raise_used)
    lower_headroom = max(0.0, max_lower - lower_used)

    raise_prices = {s: fcas_prices.get(f"FCAS_{s}", 0.0) for s in RAISE_SERVICES}
    lower_prices = {s: fcas_prices.get(f"FCAS_{s}", 0.0) for s in LOWER_SERVICES}
    best_raise = max(raise_prices, key=raise_prices.get)
    best_lower = max(lower_prices, key=lower_prices.get)

    bids = {s: 0.0 for s in FCAS_ORDER}
    if raise_headroom > 0 and raise_prices[best_raise] > 0:
        bids[best_raise] = 1.0
    if lower_headroom > 0 and lower_prices[best_lower] > 0:
        bids[best_lower] = 1.0
    return bids


# ---------------------------------------------------------------------------
# SDP cost-to-go value table (Stage C): J_t(soc) as the RTG token
# ---------------------------------------------------------------------------

def compute_cost_to_go_table(
    env,
    forecast: Sequence[dict],
    soc_resolution: int = 40,
    action_resolution: int = 41,
    deg_cost_per_mwh: float = 200.0,
    terminal_soc: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unconstrained (free-terminal) SDP value table for the RTG token.

    Returns ``(cost_to_go, soc_levels_kwh)`` where::

        cost_to_go[t, s] = optimal expected future *cost* from time `t`
                            at SOC level `s`, under the seasonal-RRP forecast.

    The terminal is **free** (``cost_to_go[horizon, :] = 0``) unless
    ``terminal_soc`` is provided, so the value encodes the market's intrinsic
    value from each SOC level — NOT the value conditional on being pinned to a
    waypoint. This is the whole point of using ``J_t(soc)`` as the DT's RTG:
    it tells the model "how much future value is left from here", calibrating
    energy arbitrage per SOC level (Stage C).

    RTG at inference = ``-cost_to_go[t, s]``  (cost -> return).

    Same honest forecast source as :func:`sdp_energy_dispatch` (seasonal
    RRP, no realized future prices) and the same ``deg_cost_per_mwh`` linear
    throughput degradation surrogate.
    """
    capacity = float(env.battery_capacity)
    horizon = len(forecast)
    if horizon == 0:
        return np.zeros((1, soc_resolution)), np.linspace(0, capacity, soc_resolution)

    solver = AEMOSDPSolver(
        env=env,
        horizon=horizon,
        soc_resolution=soc_resolution,
        action_resolution=action_resolution,
        use_monte_carlo=False,
    )
    soc_levels = solver.soc_levels_kwh
    action_energies = solver.battery_flow_energies
    n_soc = len(soc_levels)

    # Linear throughput degradation surrogate (per action, broadcast over SOC).
    linear_deg = np.abs(action_energies)[None, :] * deg_cost_per_mwh

    # Terminal cost: FREE by default (Stage C); optional soft pin for A/B.
    cost_to_go = np.full((horizon + 1, n_soc), np.inf)
    if terminal_soc is None:
        cost_to_go[horizon, :] = 0.0
    else:
        cost_to_go[horizon, :] = 1e6 * ((soc_levels - terminal_soc) / max(capacity, 1e-9)) ** 2
    policy = np.full((horizon, n_soc), -1, dtype=int)

    for t in range(horizon - 1, -1, -1):
        stage = solver._compute_stage_costs(forecast[t], 0, None)
        stage = stage + linear_deg  # add linear throughput degradation
        future = solver._compute_future_costs(cost_to_go[t + 1, :])
        total = np.where(np.isfinite(stage) & np.isfinite(future), stage + future, np.inf)
        best = np.argmin(total, axis=1)
        row_min = np.min(total, axis=1)
        finite = np.isfinite(row_min)
        policy[t, :] = -1
        policy[t, finite] = best[finite]
        cost_to_go[t, :] = row_min

    return cost_to_go, soc_levels


_COST_TO_GO_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def get_cost_to_go_table(
    aemo_data: pl.DataFrame,
    env,
    deg_cost_per_mwh: float = 200.0,
    soc_resolution: int = 40,
    forecast: Sequence[dict] | None = None,
    profile: Callable[[int, int], float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Memoized ``J_t(soc)`` table keyed by (region, step_count, capacity, deg).

    Args:
        aemo_data: The episode's processed frame (used for REGIONID + step count).
        env: The AEMOBatteryTradingEnv instance.
        deg_cost_per_mwh: Throughput degradation surrogate ($/MWh).
        soc_resolution: Number of discrete SOC levels in the value table.
        forecast: Optional pre-built RRP forecast. If None, built from
            ``aemo_data`` times + (optional ``profile``; else seasonal cache).
        profile: Optional seasonal RRP profile callable. If None, built from
            the training-era cache for the region.

    Returns ``(cost_to_go, soc_levels_kwh)`` matching :func:`compute_cost_to_go_table`.
    """
    region = str(aemo_data["REGIONID"][0]) if "REGIONID" in aemo_data.columns else "?"
    cap = float(env.battery_capacity)
    step_h = float(env.step_duration)

    if forecast is None:
        if profile is None:
            cache_dir = getattr(env, '_aemo_data_dir', None)
            if cache_dir is None:
                from pathlib import Path as _Path
                cache_dir = _Path(__file__).resolve().parent.parent / "data" / "aemo"
            key_p = (region, step_h)
            if key_p not in _COST_TO_GO_CACHE:
                profile = build_seasonal_rrp_profile(cache_dir, region, step_h)
            else:
                profile = _COST_TO_GO_CACHE[key_p]
        forecast = build_rrp_forecast(aemo_data, profile)

    key = (region, len(forecast), cap, deg_cost_per_mwh, soc_resolution)
    if key not in _COST_TO_GO_CACHE:
        _COST_TO_GO_CACHE[key] = compute_cost_to_go_table(
            env, forecast, soc_resolution=soc_resolution,
            deg_cost_per_mwh=deg_cost_per_mwh,
        )
    return _COST_TO_GO_CACHE[key]


def lookup_j_t_soc(
    cost_to_go: np.ndarray,
    soc_levels: np.ndarray,
    t: int,
    soc_kwh: float,
) -> float:
    """Return ``-J_t(soc)`` (RTG) from a value table at step ``t`` and SOC (MWh).

    Converts the SDP *cost*-to-go into a *return*-to-go (higher = better).
    """
    s_idx = int(np.argmin(np.abs(soc_levels - soc_kwh)))
    t_idx = min(max(int(t), 0), cost_to_go.shape[0] - 1)
    return -float(cost_to_go[t_idx, s_idx])
