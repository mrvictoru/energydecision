"""Deterministic tariff optimization and uncertainty reporting for H1/H3.

Unlike :mod:`household_replay`, this module optimizes a new feasible battery
dispatch.  All routines operate on normalized portal data in kW at five-minute
resolution and are deliberately segment/day scoped: callers must not bridge
telemetry gaps when constructing evaluation units.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import polars as pl

from household_replay import Tariff


STEP_HOURS = 5.0 / 60.0


@dataclass(frozen=True)
class OptimizationResult:
    """Cost-minimizing dispatch and bill for one contiguous evaluation frame."""

    bill_aud: float
    actions_kw: np.ndarray
    soc_kwh: np.ndarray
    import_kwh: float
    export_kwh: float
    free_import_kwh: float


def apply_tariff(frame: pl.DataFrame, tariff: Tariff) -> pl.DataFrame:
    """Replace stored default tariff columns using the specified tariff."""
    timestamps = frame["Timestamp"].to_list()
    return frame.with_columns([
        pl.Series("ImportEnergyPrice", [tariff.import_price(ts) for ts in timestamps]),
        pl.lit(tariff.feed_in_price()).alias("ExportEnergyPrice"),
    ])


def _prices(frame: pl.DataFrame, tariff: Tariff) -> tuple[np.ndarray, np.ndarray]:
    timestamps = frame["Timestamp"].to_list()
    return (
        np.asarray([tariff.import_price(ts) for ts in timestamps], dtype=float),
        np.full(len(frame), tariff.feed_in_price(), dtype=float),
    )


def optimize_dispatch(
    frame: pl.DataFrame,
    *,
    tariff: Tariff,
    capacity_kwh: float = 5.0,
    max_flow_kw: float = 3.3,
    roundtrip_eff: float = 0.80,
    initial_soc: float = 0.5,
    soc_resolution: int = 31,
    action_resolution: int = 21,
) -> OptimizationResult:
    """Solve a deterministic finite-horizon DP for a contiguous frame.

    Actions represent grid-side battery energy (+ charge, - discharge).
    Charge/discharge efficiency is split symmetrically, and the stage cost
    values imports at the tariff and exports at its feed-in price.  The
    terminal value is zero, deliberately making this an honest cost-minimizing
    lower-bound comparator rather than a replay of observed dispatch.
    """
    if len(frame) == 0:
        raise ValueError("Cannot optimize an empty frame")
    if capacity_kwh <= 0 or max_flow_kw <= 0:
        raise ValueError("capacity_kwh and max_flow_kw must be positive")
    if not 0 < roundtrip_eff <= 1:
        raise ValueError("roundtrip_eff must be in (0, 1]")
    if soc_resolution < 2 or action_resolution < 2:
        raise ValueError("State and action resolutions must be at least two")

    load = np.asarray(frame["HouseLoad"], dtype=float)
    solar = np.asarray(frame["SolarGen"], dtype=float)
    if not np.isfinite(load).all() or not np.isfinite(solar).all() or (load < 0).any() or (solar < 0).any():
        raise ValueError("Frame requires finite non-negative HouseLoad and SolarGen")
    import_price, export_price = _prices(frame, tariff)
    states = np.linspace(0.0, capacity_kwh, soc_resolution)
    actions = np.linspace(-max_flow_kw * STEP_HOURS, max_flow_kw * STEP_HOURS, action_resolution)
    eff = roundtrip_eff ** 0.5
    horizon = len(frame)
    value = np.zeros(soc_resolution, dtype=float)
    policy = np.zeros((horizon, soc_resolution), dtype=np.int16)

    state_grid = states[:, None]
    action_grid = actions[None, :]
    next_state = state_grid + np.where(action_grid >= 0, action_grid * eff, action_grid / eff)
    feasible = (next_state >= -1e-9) & (next_state <= capacity_kwh + 1e-9)
    clipped_next = np.clip(next_state, 0.0, capacity_kwh)

    for index in range(horizon - 1, -1, -1):
        grid = load[index] * STEP_HOURS - solar[index] * STEP_HOURS + action_grid
        stage = np.where(grid >= 0, grid * import_price[index], grid * export_price[index])
        stage = np.broadcast_to(stage, (soc_resolution, action_resolution))
        future = np.interp(clipped_next.ravel(), states, value).reshape(clipped_next.shape)
        total = np.where(feasible, stage + future, np.inf)
        policy[index] = np.argmin(total, axis=1)
        value = np.min(total, axis=1)

    soc = float(np.clip(initial_soc, 0.0, 1.0)) * capacity_kwh
    soc_path = np.empty(horizon + 1, dtype=float)
    actions_kwh = np.empty(horizon, dtype=float)
    soc_path[0] = soc
    for index in range(horizon):
        state_index = int(np.argmin(np.abs(states - soc)))
        action = actions[policy[index, state_index]]
        next_soc = soc + (action * eff if action >= 0 else action / eff)
        # The DP state grid is approximate; enforce physical feasibility in rollout.
        if next_soc > capacity_kwh:
            action = (capacity_kwh - soc) / eff
        elif next_soc < 0:
            action = -soc * eff
        soc += action * eff if action >= 0 else action / eff
        actions_kwh[index] = action
        soc_path[index + 1] = soc

    grid = load * STEP_HOURS - solar * STEP_HOURS + actions_kwh
    imports = np.maximum(grid, 0.0)
    exports = np.maximum(-grid, 0.0)
    return OptimizationResult(
        bill_aud=float(np.sum(imports * import_price - exports * export_price)),
        actions_kw=actions_kwh / STEP_HOURS,
        soc_kwh=soc_path,
        import_kwh=float(imports.sum()),
        export_kwh=float(exports.sum()),
        free_import_kwh=float(imports[import_price == 0.0].sum()),
    )


def greedy_self_consumption_actions(
    frame: pl.DataFrame,
    *,
    capacity_kwh: float = 5.0,
    max_flow_kw: float = 3.3,
    initial_soc: float = 0.5,
) -> np.ndarray:
    """A causal rule baseline: absorb surplus and cover deficits from storage."""
    soc = float(np.clip(initial_soc, 0.0, 1.0)) * capacity_kwh
    actions = np.empty(len(frame), dtype=float)
    load = np.asarray(frame["HouseLoad"], dtype=float)
    solar = np.asarray(frame["SolarGen"], dtype=float)
    for index, net_kw in enumerate(load - solar):
        requested = np.clip(-net_kw, -max_flow_kw, max_flow_kw)
        energy = np.clip(requested * STEP_HOURS, -soc, capacity_kwh - soc)
        actions[index] = energy / STEP_HOURS
        soc += energy
    return actions


def bill_for_actions(
    frame: pl.DataFrame,
    actions_kw: Sequence[float],
    tariff: Tariff,
) -> float:
    """Price any feasible action series; caller owns physical feasibility."""
    actions = np.asarray(actions_kw, dtype=float)
    if actions.shape != (len(frame),):
        raise ValueError("actions_kw length must equal frame length")
    import_price, export_price = _prices(frame, tariff)
    grid = (
        np.asarray(frame["HouseLoad"], dtype=float) * STEP_HOURS
        - np.asarray(frame["SolarGen"], dtype=float) * STEP_HOURS
        + actions * STEP_HOURS
    )
    return float(np.sum(np.maximum(grid, 0.0) * import_price - np.maximum(-grid, 0.0) * export_price))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap a 95% CI over independent segment/day evaluation units."""
    sample = np.asarray(values, dtype=float)
    if len(sample) < 1 or not np.isfinite(sample).all():
        raise ValueError("Bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        means[index] = rng.choice(sample, size=len(sample), replace=True).mean()
    return {
        "mean": float(sample.mean()),
        "ci_lower": float(np.quantile(means, 0.025)),
        "ci_upper": float(np.quantile(means, 0.975)),
        "n": int(len(sample)),
    }
