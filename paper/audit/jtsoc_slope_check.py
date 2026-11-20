#!/usr/bin/env python3
"""Reproduce the marginal-value figure quoted in the Figure 3 caption.

Builds the same seasonal-forecast cost-to-go table used for Figure 3 (NSW1,
October 2024, 144 h at five-minute resolution) and measures the near-linear
slope of the value prompt with respect to state of charge. That slope is the
marginal value of stored energy, net of the degradation surrogate.

Writes ``paper/audit/artifacts/jtsoc_slope.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aemo_sdp_executor import (  # noqa: E402
    build_rrp_forecast,
    build_seasonal_rrp_profile,
    compute_cost_to_go_table,
)

OUT = ROOT / "paper/audit/artifacts/jtsoc_slope.json"
PROCESSED = ROOT / "data/aemo/processed_NSW1_2024-10-01_2024-10-14_0.0833h.parquet"
HORIZON = 1728  # 144 h at five-minute steps (standard-surface horizon)
STEP_HOURS = 5.0 / 60.0
REGION = "NSW1"


def main() -> None:
    frame = pl.read_parquet(PROCESSED).head(HORIZON)
    profile = build_seasonal_rrp_profile(ROOT / "data/aemo", REGION, STEP_HOURS)
    forecast = build_rrp_forecast(frame, profile)
    env = SimpleNamespace(
        battery_capacity=10.0,
        max_battery_flow=10.0,
        step_duration=STEP_HOURS,
        max_grid_energy=float("inf"),
        df=None,
        battery_life_cost=1_312_500.0,
        degradation_temperature=25.0,
    )
    cost_to_go, soc_levels = compute_cost_to_go_table(
        env, forecast, deg_cost_per_mwh=50.0, deg_calibration=0.12
    )
    rtg = -cost_to_go
    soc = np.asarray(soc_levels)

    slopes = []
    for t in range(rtg.shape[0]):
        values = rtg[t]
        finite = np.isfinite(values)
        if finite.sum() > 3:
            slopes.append(float(np.polyfit(soc[finite], values[finite], 1)[0]))
    slopes_arr = np.asarray(slopes)

    payload = {
        "source": str(PROCESSED.relative_to(ROOT)),
        "region": REGION,
        "horizon_steps": HORIZON,
        "step_hours": STEP_HOURS,
        "deg_cost_per_mwh": 50.0,
        "deg_calibration": 0.12,
        "slope_units": "A$/MWh",
        "slope_median": float(np.median(slopes_arr)),
        "slope_mean": float(np.mean(slopes_arr)),
        "slope_p10": float(np.percentile(slopes_arr, 10)),
        "slope_p90": float(np.percentile(slopes_arr, 90)),
        "n_timesteps": int(slopes_arr.size),
        "method": "linear fit of -J_t(soc) against SOC at each timestep; "
                  "median across timesteps",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
