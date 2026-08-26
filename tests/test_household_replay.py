import sys
sys.path.insert(0, "src")

from household_replay import (
    Tariff, detect_action_sign, load_normalized_year, no_battery_bill, replay,
)

NOMINAL_CAP = 7.0
NOMINAL_FLOW = 3.3


def _mini_df():
    import datetime as dt
    import polars as pl
    rows = []
    base = dt.datetime(2026, 5, 20)
    for i in range(288):
        ts = base + dt.timedelta(minutes=5 * i)
        # simple day: solar 0 at night, load 1 kW, battery charges 1 kW day
        solar = 3.0 if 8 <= ts.hour < 16 else 0.0
        load = 1.0
        batt = -1.0 if ts.hour >= 18 else (1.0 if ts.hour >= 10 else 0.0)
        soc = 0.5 + 0.001 * i
        rows.append({
            "Timestamp": ts, "HouseLoad": load, "SolarGen": solar,
            "BatteryPower": batt, "BatterySOC": soc,
        })
    return pl.DataFrame(rows)


def test_detect_action_sign_picks_reproducible():
    df = _mini_df()
    tariff = Tariff()
    sign = detect_action_sign(df, tariff, NOMINAL_CAP, NOMINAL_FLOW)
    assert sign in (1.0, -1.0)


def test_replay_runs_and_bill_finite():
    df = _mini_df()
    tariff = Tariff()
    sign = detect_action_sign(df, tariff, NOMINAL_CAP, NOMINAL_FLOW)
    r = replay(df, NOMINAL_CAP, NOMINAL_FLOW, tariff, 1.0, sign)
    assert r["bill_aud"] == r["bill_aud"]  # not NaN
    assert r["import_kwh"] >= 0
    assert r["export_kwh"] >= 0


def test_no_battery_bill_positive_and_larger_than_replay():
    df = _mini_df()
    tariff = Tariff()
    base = no_battery_bill(df, tariff)
    assert base > 0
    sign = detect_action_sign(df, tariff, NOMINAL_CAP, NOMINAL_FLOW)
    r = replay(df, NOMINAL_CAP, NOMINAL_FLOW, tariff, 1.0, sign)
    assert r["bill_aud"] <= base + 1e-6


def test_scaling_capacity_does_not_crash():
    df = _mini_df()
    tariff = Tariff()
    sign = detect_action_sign(df, tariff, NOMINAL_CAP, NOMINAL_FLOW)
    r2 = replay(df, NOMINAL_CAP, NOMINAL_FLOW, tariff, 2.0, sign)
    assert r2["capacity_kwh"] == 14.0
    assert r2["bill_aud"] == r2["bill_aud"]
