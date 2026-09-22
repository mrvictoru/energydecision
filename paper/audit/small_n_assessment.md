# Small-n assessment — how far each identity surface can be expanded

**Date:** 2026-09-22 · **Frozen release:** `preprint-v1`
**Question:** the headline identity surfaces have n=5 / 6 / 27 / 6 scenario cells.
How many more can each support from cached data, and at what cost?

## Cost model (measured)

The evaluator runs one deterministic episode per (scenario, battery) cell.
PPO and `fcas_rule` rollouts are served from `eval_output/reference_cache/`; only
the DT is re-inferred. Measured on this machine (RTX 2080 Ti):

| Horizon | steps | DT inference / cell |
|---|---:|---:|
| 144 h (standard, dispatch) | 1,728 | ~25–30 s |
| 288 h (expanded, 2025 OOD) | 3,456 | ~55–60 s |

The 2024 AEMO data is already preprocessed into long-span parquets
(`data/aemo/processed_{REGION}_2024-*`), so **no NEMOSIS fetch is needed for any
2024 expansion**. 2025 is only cached for NSW1/QLD1/SA1 × Jan–Feb.

## Data coverage (5-min processed cache)

| Region | 2024 coverage | 2025 coverage |
|---|---|---|
| NSW1 | full year | Jan, Feb |
| QLD1 | full year | Jan, Feb |
| SA1 | full year | Jan, Feb |
| VIC1 | full year | — |
| TAS1 | full year | — |

## Expansion options

### standard — current n=5 (Oct 2024 × 5 regions, `medium_1c`)

| Option | n | Method | Added runtime |
|---|---:|---|---:|
| A. Temporal | 30 | 5 regions × {Jan, Mar, May, Jul, Sep, Nov} 2024 | ~12 min |
| B. Multi-asset | 20 | {medium_1c, fast_375c, large_07c, small_05c} × 5 regions (Oct) | ~8 min |
| C. Both | 120 | 4 assets × 5 regions × 6 periods | ~50 min |

### dispatch — current n=6 (SA1 Jul–Dec 2024, `dispatch_asset_template`)

| Option | n | Method | Added runtime |
|---|---:|---|---:|
| D. Full year | 12 | SA1 Jan–Dec 2024 | ~5 min |

### expanded — current n=27 (5 regions × 6 periods, `medium`)

| Option | n | Method | Added runtime |
|---|---:|---|---:|
| E. Complete TAS1 | 30 | add TAS1 Jul/Sep/Nov (data exists; config currently omits) | ~3 min |
| F. Monthly | 60 | 5 regions × 12 months 2024 | ~55 min |

### 2025 OOD — current n=6 (NSW1/SA1/QLD1 × Jan/Feb, `medium`)

| Option | n | Method | Added runtime |
|---|---:|---|---:|
| G. Cached max | 6 | no change (data-bound) | — |
| H. Fetch VIC1/TAS1 | 10 | +2 regions × Jan/Feb 2025 — **needs NEMOSIS fetch** | ~10 min + fetch |
| I. Fetch more months | 18–30 | +more 2025 months — fetch-bound | fetch-dominated |

## Recommendation

1. **Do E (expanded 27→30)** — trivial, removes a config omission, ~3 min.
2. **Do D (dispatch 6→12)** — the cheapest way to lift a low-n surface without
   changing its meaning (SA1 dispatch-asset, now full-year), ~5 min.
3. **For standard, prefer A (temporal, n=30)** over B: a single-month surface is
   the most cherry-pickable and the n=5 is the weakest headline. A full-year
   standard surface is both more representative and ~6× the sample. **Keep the
   existing Oct n=5 surface as a named sub-surface** so the traced A1 number
   stays valid; add the year surface as the headline. ~12 min.
4. **Leave 2025 OOD at n=6** and frame it explicitly as an OOD spot-check; do not
   spend a fetch cycle on it before the preprint. State the n.
5. Do **not** do C/F (120/60 cells) — marginal power gain, disproportionate time,
   and cell correlation (same region across months) means effective n grows
   sub-linearly anyway.

**Total recommended added runtime ≈ 20 min** to reach n = 30 / 12 / 30 / 6.

## Statistical caveat to state in the paper

Adding months within a region increases the number of cells but not the number of
independent regions; the between-region variation is what limits generalisation.
The bootstrap already resamples scenario cells, which is the correct unit, but a
reviewer should be told that temporal cells within a region are correlated. The
paired-difference CI remains the primary evidence; Wilcoxon is indicative at
n<10 and becomes usable once standard/dispatch exceed 10.
