"""
Endogenous market-impact models for AEMOBatteryTradingEnv.

Provides pluggable impact functions that replace the price-taking assumption
(historical RRP/FCAS_* prices read directly as revenue multipliers) with a
realized price that depends on the battery's own dispatch and FCAS bids.

Impact model classes:
  - IdentityImpact:     price-taking (default, backward-compat).
  - PiecewiseMeritOrder: realized energy price from a supply-curve shift;
                          realized FCAS price from depth-proportional attenuation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class MarketImpactModel(ABC):
    """Base class for impact models.  Subclasses must implement both methods."""

    @abstractmethod
    def realized_energy_price(
        self,
        base_price: float,
        battery_dispatch_mw: float,   # +discharge, -charge  (from env actual_power)
        energy_price: float,          # RRP ()/MWh
        market_state: dict,
    ) -> float: ...

    @abstractmethod
    def realized_fcas_price(
        self,
        service: str,
        base_price: float,
        battery_enabled_mw: float,    # MW enabled for this FCAS service
        market_state: dict,
    ) -> float: ...


# ---------------------------------------------------------------------------
# Identity (no impact — reproduces existing env)
# ---------------------------------------------------------------------------

class IdentityImpact(MarketImpactModel):
    """Price-taking: realized = base.  Backward-compatible default."""

    def __init__(self, **kwargs):
        super().__init__()

    def realized_energy_price(self, base_price, battery_dispatch_mw, energy_price, market_state):
        return energy_price  # same

    def realized_fcas_price(self, service, base_price, battery_enabled_mw, market_state):
        return base_price


# ---------------------------------------------------------------------------
# Piecewise-linear merit-order impact
# ---------------------------------------------------------------------------

class PiecewiseMeritOrderImpact(MarketImpactModel):
    """
    Realised energy price is read from a supply curve shifted by the battery's
    net dispatch.  Realised FCAS price is attenuated proportionally to the
    battery's share of market depth.

    Requires pre-built supply curves (from ``build_supply_curve``) and
    FCAS depth series (from ``aggregate_fcas_market_depth``).
    """

    def __init__(
        self,
        supply_curves: Optional[pl.DataFrame] = None,  # from build_supply_curve()
        fcas_depth: Optional[pl.DataFrame] = None,     # from aggregate_fcas_market_depth()
        impact_intensity: float = 1.0,
    ):
        super().__init__()
        self.intensity = impact_intensity

        # ---- Pre-index supply curves: {SETTLEMENTDATE: (costs, cum_mw)} ----
        self._supply_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if supply_curves is not None and 'MARGINAL_COST' in supply_curves.columns:
            for row in supply_curves.group_by('SETTLEMENTDATE', maintain_order=True).agg([
                pl.col('MARGINAL_COST'),
                pl.col('CUMULATIVE_MW'),
            ]).iter_rows(named=True):
                dt = row['SETTLEMENTDATE']
                ts = int(dt.timestamp())
                self._supply_map[ts] = (
                    np.asarray(row['MARGINAL_COST'], dtype=float),
                    np.asarray(row['CUMULATIVE_MW'], dtype=float),
                )

        # ---- Pre-index FCAS depth: {SETTLEMENTDATE: {service: depth_mw}} ----
        self._fcas_depth_map: dict[int, dict[str, float]] = {}
        if fcas_depth is not None:
            depth_cols = [c for c in fcas_depth.columns
                          if c.startswith('FCAS_DEPTH_') and not c.endswith('_normalized')]
            if depth_cols:
                for row in fcas_depth.select(['SETTLEMENTDATE'] + depth_cols).iter_rows(named=True):
                    dt = row['SETTLEMENTDATE']
                    ts = int(dt.timestamp())
                    svc_map = {}
                    for c in depth_cols:
                        svc_name = c.replace('FCAS_DEPTH_', '').replace('_MW', '')
                        svc_map[svc_name] = float(row[c] or 0)
                    self._fcas_depth_map[ts] = svc_map

    def _ts(self, market_state: dict) -> int:
        """Extract timestamp of the current interval from market_state."""
        dt = market_state.get('SETTLEMENTDATE')
        if dt is None:
            return 0
        return int(dt.timestamp()) if hasattr(dt, 'timestamp') else int(dt)

    def realized_energy_price(
        self,
        base_price: float,
        battery_dispatch_mw: float,
        energy_price: float,
        market_state: dict,
    ) -> float:
        ts = self._ts(market_state)
        supply = self._supply_map.get(ts)
        if supply is None:
            return energy_price  # fallback: no supply curve for this interval

        costs, cum_mw = supply
        total_demand = market_state.get('TOTALDEMAND', 0.0) or 0.0

        # Battery net effect on demand:
        #   battery_dispatch_mw > 0 (discharging) → adds supply → reduces effective demand
        #   battery_dispatch_mw < 0 (charging)    → adds demand → increases effective demand
        effective_demand = total_demand + battery_dispatch_mw

        # Clamp to the supply curve range.
        if effective_demand <= cum_mw[0]:
            price = costs[0]
        elif effective_demand >= cum_mw[-1]:
            price = costs[-1]
        else:
            price = float(np.interp(effective_demand, cum_mw, costs))

        # Blend: realized = base + intensity * (realized - base)
        return energy_price + self.intensity * (price - energy_price)

    def realized_fcas_price(
        self,
        service: str,
        base_price: float,
        battery_enabled_mw: float,
        market_state: dict,
    ) -> float:
        if battery_enabled_mw <= 0 or base_price <= 0:
            return base_price

        ts = self._ts(market_state)
        svc_map = self._fcas_depth_map.get(ts, {})
        depth = svc_map.get(service.upper(), 0.0)

        if depth <= 0:
            return base_price  # no market depth → no impact

        # The battery's bid increases reserve supply, pushing the clearing price down.
        # Effective depth = requirement + battery contribution.
        # Price attenuation proportional to the battery's share of total depth.
        # When intensity=1 and battery provides 100% of depth, price goes to 0.
        share = battery_enabled_mw / (depth + battery_enabled_mw)
        realized = base_price * (1.0 - self.intensity * share)
        return max(realized, 0.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_IMPACT_REGISTRY: dict[str, type[MarketImpactModel]] = {
    'identity': IdentityImpact,
    'piecewise_merit_order': PiecewiseMeritOrderImpact,
}


def create_impact_model(
    kind: str,
    **kwargs,
) -> MarketImpactModel:
    cls = _IMPACT_REGISTRY.get(kind.lower().strip())
    if cls is None:
        raise ValueError(
            f"Unknown impact model {kind!r}. "
            f"Available: {list(_IMPACT_REGISTRY)}"
        )
    return cls(**kwargs)
