"""CLI: replay the recorded household battery year under a configurable tariff.

Computes the bill for:
  - no battery (baseline)
  - the recorded 7 kWh / 3.3 kW battery
  - scaled battery sizes (default 1.5x and 2.0x) replaying the SAME behaviour
and reports savings vs the no-battery baseline and the incremental gain.
"""
import argparse
import sys

sys.path.insert(0, "src")

from household_replay import (
    Tariff, detect_action_sign, detect_capacity_and_eff, load_normalized_year,
    no_battery_bill, replay,
)

NOMINAL_CAPACITY = 7.0
NOMINAL_MAX_FLOW = 3.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normalized-dir",
                    default="data/household/real/normalized")
    ap.add_argument("--import-cents", type=float, default=31.042)
    ap.add_argument("--feed-in-cents", type=float, default=1.0)
    ap.add_argument("--free-start", type=int, default=11)
    ap.add_argument("--free-end", type=int, default=14)
    ap.add_argument("--capacity", type=float, default=NOMINAL_CAPACITY)
    ap.add_argument("--max-flow", type=float, default=NOMINAL_MAX_FLOW)
    ap.add_argument("--factors", type=float, nargs="*",
                    default=[1.0, 1.5, 2.0])
    ap.add_argument("--out", default="data/household/real/replay_results.csv")
    args = ap.parse_args()

    tariff = Tariff(args.import_cents, args.feed_in_cents,
                    args.free_start, args.free_end)
    df = load_normalized_year(args.normalized_dir)

    sign = detect_action_sign(df, tariff, args.capacity, args.max_flow)
    print(f"Detected action sign = {sign:+.0f} "
          f"(validates BatteryPower convention)")

    if args.capacity == NOMINAL_CAPACITY:
        cap, eff, soc_err = detect_capacity_and_eff(df, args.max_flow, sign, tariff)
        print(f"Inferred effective battery: capacity = {cap:.1f} kWh, "
              f"round-trip eff = {eff:.2f} (SOC fit MAE = {soc_err:.4f})")
    else:
        cap, eff = args.capacity, 1.0
    base = no_battery_bill(df, tariff)
    print(f"\nNo-battery baseline bill: ${base:,.2f}\n")

    rows = []
    prev_bill = None
    print(f"{'factor':>6} {'cap_kWh':>8} {'flow_kW':>8} "
          f"{'bill_AUD':>11} {'save_vs_0':>10} {'incr_vs_1x':>11} "
          f"{'imp_kWh':>10} {'exp_kWh':>10} {'free_imp':>9} {'soc_err':>8}")
    print("-" * 95)
    for f in args.factors:
        r = replay(df, cap, args.max_flow, tariff, f, sign, eff)
        save0 = base - r["bill_aud"]
        incr = (prev_bill - r["bill_aud"]) if (prev_bill is not None and f != 1.0) else float("nan")
        prev_bill = r["bill_aud"]
        rows.append({
            "factor": f, "capacity_kwh": r["capacity_kwh"],
            "max_flow_kw": r["max_flow_kw"], "bill_aud": r["bill_aud"],
            "save_vs_no_battery": save0, "incremental_vs_prev": incr,
            "import_kwh": r["import_kwh"], "export_kwh": r["export_kwh"],
            "free_import_kwh": r["free_import_kwh"],
            "mean_soc_abs_err": r["mean_soc_abs_err"],
        })
        print(f"{f:>6.1f} {r['capacity_kwh']:>8.1f} {r['max_flow_kw']:>8.1f} "
              f"{r['bill_aud']:>11.2f} {save0:>10.2f} "
              f"{('' if incr != incr else incr):>11} "
              f"{r['import_kwh']:>10.0f} {r['export_kwh']:>10.0f} "
              f"{r['free_import_kwh']:>9.0f} {r['mean_soc_abs_err']:>8.4f}")

    import polars as pl
    pl.DataFrame(rows).write_csv(args.out)
    print(f"\nwrote {args.out}")

    r1 = rows[0]
    print("\n=== interpretation (replay of OBSERVED behaviour only) ===")
    print(f"Baseline (no battery):      ${base:,.2f}/yr")
    print(f"Actual {r1['capacity_kwh']:.1f} kWh battery: "
          f"${r1['bill_aud']:,.2f}/yr  -> saves ${r1['save_vs_no_battery']:,.2f}/yr")
    if len(rows) > 1:
        last = rows[-1]
        extra = last["save_vs_no_battery"] - r1["save_vs_no_battery"]
        print(f"Doubled to {last['capacity_kwh']:.1f} kWh: "
              f"${last['bill_aud']:,.2f}/yr  -> saves "
              f"${last['save_vs_no_battery']:,.2f}/yr "
              f"(only ${extra:,.2f} MORE than actual)")
        print("NOTE: this replays the recorded self-consumption actions. The")
        print("free 11:00-14:00 window is barely used by that behaviour")
        print(f"({r1['free_import_kwh']:.0f} kWh/yr charged free). A RE-OPTIMISED")
        print("controller that shifts charging into the free window would show")
        print("the true upper-bound value of a larger battery -- a different")
        print("experiment (requires an optimiser, not a replay).")


if __name__ == "__main__":
    main()
