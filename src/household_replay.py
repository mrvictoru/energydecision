"""Replay recorded household battery behaviour under a configurable tariff.

The SMA portal gives us, per 5-min step:
  HouseLoad [kW], SolarGen [kW], BatteryPower [kW] (VPP net: + = charging),
  BatterySOC [%].

This module replays the *recorded* battery actions on the *recorded* load and
solar, enforcing the physical battery limits (capacity, max flow), and prices
the resulting grid exchange under an arbitrary tariff. It also supports scaling
the battery size (capacity and max-flow by the same factor) to estimate how a
larger battery would have changed the bill under the *same* observed behaviour.

No FRP / re-optimisation is performed -- this is a what-actually-happened
replay, exactly as observed in the data.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


@dataclass
class Tariff:
    import_cents_per_kwh: float = 31.042
    feed_in_cents_per_kwh: float = 1.0
    free_window_start_hour: int = 11  # super off-peak: import is free
    free_window_end_hour: int = 14    # [start, end)

    def import_price(self, ts: _dt.datetime) -> float:
        if self.free_window_start_hour <= ts.hour < self.free_window_end_hour:
            return 0.0
        return self.import_cents_per_kwh / 100.0

    def feed_in_price(self) -> float:
        return self.feed_in_cents_per_kwh / 100.0


def load_normalized_year(normalized_dir: str | Path) -> pl.DataFrame:
    files = sorted(Path(normalized_dir).glob("*_normalized.parquet"))
    if not files:
        raise FileNotFoundError(f"No *_normalized.parquet under {normalized_dir}")
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = df.unique(subset=["Timestamp"], keep="first").sort("Timestamp")
    # edge-fill any residual nulls so replay never sees NaN
    df = df.with_columns([
        pl.col(c).fill_null(strategy="forward").fill_null(strategy="backward")
        for c in ["HouseLoad", "SolarGen", "BatteryPower", "BatterySOC"]
    ])
    return df


def _net_import_kwh(
    load_kw: float,
    solar_kw: float,
    action_kw: float,
    dt_h: float,
) -> tuple[float, float]:
    """Grid exchange given an action in env convention (+ = charging)."""
    grid_net_kw = load_kw - solar_kw + action_kw
    if grid_net_kw >= 0:
        return grid_net_kw * dt_h, 0.0
    return 0.0, -grid_net_kw * dt_h


def replay(
    df: pl.DataFrame,
    capacity_kwh: float,
    max_flow_kw: float,
    tariff: Tariff,
    size_factor: float = 1.0,
    action_sign: float = 1.0,
    roundtrip_eff: float = 1.0,
):
    """Replay recorded actions for a battery scaled by ``size_factor``.

    ``roundtrip_eff`` splits into charge/discharge efficiencies
    (sqrt each) so the predicted SOC matches the measured trajectory.

    Returns a dict with total bill (AUD), import/export kWh, free-window import,
    and the predicted SOC trajectory for validation.
    """
    cap = capacity_kwh * size_factor
    flow = max_flow_kw * size_factor
    eff_in = roundtrip_eff ** 0.5
    eff_out = roundtrip_eff ** 0.5
    dt_h = 5.0 / 60.0

    ts = df["Timestamp"].to_list()
    load = np.asarray(df["HouseLoad"], dtype=float)
    solar = np.asarray(df["SolarGen"], dtype=float)
    batt = action_sign * np.asarray(df["BatteryPower"], dtype=float)
    rec_soc = np.asarray(df["BatterySOC"], dtype=float)

    n = len(ts)
    soc = float(rec_soc[0]) if rec_soc[0] == rec_soc[0] else 0.5
    pred_soc = np.empty(n, dtype=float)
    pred_soc[0] = soc

    bill = 0.0
    import_kwh = 0.0
    export_kwh = 0.0
    free_import_kwh = 0.0
    soc_err = 0.0

    for i in range(n):
        a = batt[i]
        a = min(max(a, -flow), flow)
        e = a * dt_h
        if e > 0:
            room = cap * (1.0 - soc)
            e = min(e, room / eff_in)
        elif e < 0:
            avail = cap * soc * eff_out
            e = max(e, -avail)
        a = e / dt_h
        # energy actually stored (charge) or delivered (discharge)
        soc = soc + (e * eff_in if e > 0 else e / eff_out) / cap

        imp, exp = _net_import_kwh(load[i], solar[i], a, dt_h)
        price = tariff.import_price(ts[i])
        bill += imp * price - exp * tariff.feed_in_price()
        import_kwh += imp
        export_kwh += exp
        if price == 0.0 and imp > 0:
            free_import_kwh += imp

        if i + 1 < n:
            pred_soc[i + 1] = soc
        soc_err += abs(soc - rec_soc[i]) if rec_soc[i] == rec_soc[i] else 0.0

    return {
        "size_factor": size_factor,
        "capacity_kwh": cap,
        "max_flow_kw": flow,
        "bill_aud": bill,
        "import_kwh": import_kwh,
        "export_kwh": export_kwh,
        "free_import_kwh": free_import_kwh,
        "mean_soc_abs_err": soc_err / n,
        "pred_soc": pred_soc,
    }


def detect_action_sign(df: pl.DataFrame, tariff: Tariff,
                       capacity_kwh: float, max_flow_kw: float) -> float:
    """Pick the action sign that best reproduces the recorded SOC."""
    best, best_err = 1.0, float("inf")
    for sign in (1.0, -1.0):
        r = replay(df, capacity_kwh, max_flow_kw, tariff, 1.0, sign)
        if r["mean_soc_abs_err"] < best_err:
            best_err, best = r["mean_soc_abs_err"], sign
    return best


def detect_capacity_and_eff(df: pl.DataFrame, max_flow_kw: float, sign: float,
                            tariff: Tariff, lo: float = 2.0, hi: float = 15.0):
    """Joint fit of effective capacity and round-trip efficiency to SOC."""
    best = (7.0, 1.0)
    best_err = float("inf")
    c = lo
    while c <= hi:
        for eff in (0.80, 0.85, 0.90, 0.95, 1.0):
            r = replay(df, c, max_flow_kw, tariff, 1.0, sign, eff)
            if r["mean_soc_abs_err"] < best_err:
                best_err, best = r["mean_soc_abs_err"], (c, eff)
        c += 0.5
    return best[0], best[1], best_err


def no_battery_bill(df: pl.DataFrame, tariff: Tariff) -> float:
    dt_h = 5.0 / 60.0
    ts = df["Timestamp"].to_list()
    load = np.asarray(df["HouseLoad"], dtype=float)
    solar = np.asarray(df["SolarGen"], dtype=float)
    bill = 0.0
    for i in range(len(ts)):
        net = load[i] - solar[i]
        if net >= 0:
            bill += net * dt_h * tariff.import_price(ts[i])
        else:
            bill -= (-net) * dt_h * tariff.feed_in_price()
    return bill
