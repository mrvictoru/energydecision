"""
AEMO-specific SDP solver for energy-only arbitrage.

Reuses the backward induction engine from SDPSolver but replaces the household
cost function (solar/load/import/export) with an AEMO energy-market revenue
function (RRP price only).

The solver generates provably optimal energy-dispatch trajectories that can be
added to the FCAS-rich offline dataset for DT training.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from sdp_algorithm import SDPSolver


class AEMOSDPSolver(SDPSolver):
    """SDP solver adapted for AEMO energy-only arbitrage.

    The cost function is ``cost = net_charge × RRP`` (positive when buying,
    negative when selling), which means the backward-induction minimiser
    learns to charge on cheap RRP and discharge on expensive RRP.
    Degradation cost (rainflow-based) is added on top, identical to the
    household solver.
    """

    def __init__(self, env: Any, **kwargs: Any) -> None:
        grid_limit = getattr(env, "max_grid_energy", None)
        if grid_limit is None or grid_limit <= 0:
            object.__setattr__(env, "max_grid_energy", float("inf"))
        # Ensure a harmless df attribute so the scenario-cache code does not
        # crash (it is never accessed when Monte Carlo is disabled).
        if not hasattr(env, "df"):
            object.__setattr__(env, "df", None)
        super().__init__(env, **kwargs)

    # ------------------------------------------------------------------
    # AEMO cost:  cost = net_charge_energy * RRP
    # SDP minimises cost →  charge on cheap RRP, discharge on expensive RRP
    # ------------------------------------------------------------------

    def _compute_grid_cost_deterministic(
        self, energy: float, forecast_step: dict[str, Any]
    ) -> float:
        rrp = float(forecast_step.get("RRP", 0.0))
        # energy > 0 → charging (buying, positive cost)
        # energy < 0 → discharging (selling, negative cost = revenue)
        return energy * rrp
