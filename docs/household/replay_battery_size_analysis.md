# Why a bigger battery doesn't save more (replay analysis)

**Date:** 2026-08 · **Data:** SMA portal exports, VPP household, 5-min resolution
**Method:** `scripts/replay_household_year.py` replays the recorded battery
actions (`BatteryPower`) on the recorded load/solar under a configurable
tariff, scaling battery size (capacity AND power by the same factor).

## Tariff

| Component | Price |
|---|---|
| Import | 31.042 c/kWh |
| Super off-peak import (11:00–14:00) | 0 c/kWh |
| Solar feed-in | 1 c/kWh |

## Results

| Config | Bill/yr | Saves vs no-battery | Marginal vs previous |
|---|---:|---:|---:|
| No battery | $1,253 | — | — |
| **5 kWh (actual, inferred)** | $907 | **$346** | — |
| 7.5 kWh (1.5×) | $889 | $364 | +$18 |
| 10 kWh (2×) | $889 | $364 | ≈$0 |

Inferred battery: ~5 kWh effective, round-trip efficiency 0.80
(joint capacity/efficiency fit to the recorded SOC trajectory, MAE 0.10).

## Why more capacity doesn't help

1. **Storage saturation.** On all 153 days with meaningful solar surplus the
   battery reaches ≥98% SOC — typically by two-thirds through the sunlight
   window. Extra capacity has nothing left to store.
2. **Marginal capture collapses.** Of 3,779 kWh/yr exported, doubling
   recovers only ~86 kWh (97.7% unreachable): +83 kWh (5→7.5 kWh),
   +3 kWh (7.5→10 kWh), +5 kWh (10→15 kWh).
3. **Round-trip efficiency tax.** At 80% efficiency each recovered kWh
   displaces 31¢ imports at 25¢ value minus 1¢ forgone feed-in → net
   ~24¢/kWh. 86 kWh ≈ $20/yr gross — matches the measured +$18.
4. **Dispatch is fixed, not hardware-limited.** The replay preserves the
   observed self-consumption pattern; power scales with capacity so the
   time-to-full is unchanged. The free 11:00–14:00 window captures only
   527 kWh/yr — the largest untapped arbitrage requires smarter control,
   not bigger cells.
5. **Seasonality.** Winter surplus (357 kWh) vs summer (1,431 kWh): extra
   capacity idles roughly half the year even with perfect dispatch.

## Conclusion

The system is **storage-saturated, dispatch-constrained**: solar surplus
arrives faster than the battery can absorb it, everything else leaves as
near-worthless export (1 c/kWh vs 31 c/kWh import asymmetry makes storage
valuable in principle — but only if dispatch exploits it). The path to
further savings is **re-optimised dispatch** (shift charging into the free
window, pre-charge before evening peaks), i.e. the H2/H3 optimiser work,
not capacity.

## Re-optimized dispatch follow-up

The replay result above is deliberately a lower bound: it holds the recorded
VPP actions fixed. `scripts/evaluate_household_tariffs.py` now solves a
deterministic, 5-minute dynamic program over each complete contiguous real
day, using the inferred 5 kWh / 3.3 kW hardware and 0.80 round-trip
efficiency. It never crosses a renovation-gap seam. Bootstrap CIs use the
1,084 complete days as resampling units (500 resamples, seed 42); annual
figures below multiply daily estimates and CI bounds by 365, so they are
annualized rather than a claim about one calendar year.

| Tariff | Observed replay bill/yr | Re-optimized bill/yr | Dispatch gap/yr (95% bootstrap CI) | No-battery bill/yr |
|---|---:|---:|---:|---:|
| Flat 31.042c import, 1c FiT | $1,186 | $973 | **$214 ($209–$219)** | $1,568 |
| 31.042c import, free 11:00–14:00, 1c FiT | $1,077 | $722 | **$355 ($342–$368)** | $1,424 |

The free window increases the value of re-optimization by about $141/year:
the optimizer pre-charges before the evening deficit rather than following
the observed self-consumption-only behavior. This confirms the capacity-sweep
interpretation: dispatch control, not additional capacity, is the near-term
savings lever. A spot-pass-through comparison remains blocked on a
time-aligned household retail spot-price series; the harness must consume
observed prices rather than inventing them.

## Caveats

- Replay scales observed behaviour; it does not re-optimise. Savings figures
  are lower bounds on what a cost-minimising controller could achieve.
- Inferred parameters (5 kWh / 0.80 eff) come from SOC-trajectory fitting;
  residual MAE 0.10 suggests measurement noise/rounding in portal SOC.
- Flat-tariff defaults were replaced with the user-supplied tariff above;
  actual retailer plan may include demand charges or TOU blocks not modelled.
