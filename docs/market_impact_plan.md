# Market-Impact Modeling for Large BESS (AEMO/NEM Track)

## Overview

Add endogenous market-impact modeling to `AEMOBatteryTradingEnv` so that a
large BESS injection/withdrawal moves the clearing price in energy and FCAS
markets. The env is currently fully price-taking — the action never feeds back
to the price.  This extension makes price a function of `(base_rrp,
battery_mw, residual_supply, fcas_depth)` via a piecewise-linear merit-order
reconstruction.  Paired with a **perfect-foresight AEMO Oracle** (LP
co-optimizer), we can answer: "how much does self-impact cost?" and "does the
DT's advantage persist when the battery moves the market?"

## Design decisions (locked)

| Decision | Choice |
|----------|--------|
| Impact model form | Piecewise-linear merit-order reconstruction |
| Markets | Energy + all 8 FCAS services |
| FCAS depth data source | v1: sum of per-unit cleared enablement from DISPATCHLOAD (free, already on disk). Upgrade to aggregate MMSDM table only if Phase 3 results hinge on depth accuracy. |
| Oracle obs compatibility | Both modes behind a config flag: `identity` (obs_dim=18, old checkpoints load) and `expose_impact_state` (expanded obs for Phase 4 retrain). |

## Goal

1 publishable rigor item (AEMO Oracle) + 1 novel method (market-impact env
extension), plus statistical rigor on the AEMO headline tables that have been
point-estimate-only.

## Complementary research thread: synthetic FCAS generation

A separate motivation for impact modeling surfaced during planning: instead
of perturbing historical prices with an impact function, generate *synthetic*
price trajectories that already embed market impact. This avoids the domain
shift caused by ESS-trained-on-historical-prices vs. ESS-evaluated-on-impacted
prices (Phase 4 retrain approach). It would also unlock unlimited training
data.

**Blocker:** TTM (and any pure price-history conditional model) cannot
generate realistic FCAS prices — measured TTM-FCAS correlations are ~0.01–0.07
(report.md §8.2.8). The reason is documented below: FCAS prices are
spike-driven, regime-switching, and not predictable from price history alone.
A generator that conditions only on past prices cannot reproduce their
distribution.

Nonetheless, the *impact model produces exactly the training signal that a
future FCAS generator would need*: the impact function computes `(base_price,
battery_mw, depth) -> realized_price`. A learned generator can be trained to
reproduce this mapping across many unseen market states. So the impact model
is a prerequisite and enables a future generative FCAS pipeline (see Phase 7).

### FCAS data characteristics (measured on cached 5-min 2024 H1, 3 regions)

**Distribution shape (pool SA1+NSW1+QLD1, 5-min, H1 2024):**

| Service | Mean | Std | Median | p99 | Max | %Zero | Skew | lag1 ACF |
|---------|---:|---:|---:|---:|---:|---:|---:|---:|
| LOWER5MIN | 0.39 | 3.18 | 0.20 | 4.0 | 999 | 8.7% | 208 | 0.25 |
| LOWER60SEC | 3.27 | 128.7 | 0.39 | 27.7 | 16600 | 0.6% | 114 | 0.90 |
| LOWER6SEC | 9.87 | 347.2 | 0.30 | 40.6 | 16600 | 5.9% | 42 | 0.92 |
| LOWERREG | 2.99 | 4.82 | 1.84 | 19.5 | 999 | 2.7% | 62 | 0.33 |
| RAISE5MIN | 0.37 | 0.47 | 0.38 | 0.7 | 43.8 | 1.4% | 61 | 0.72 |
| RAISE60SEC | 0.57 | 3.44 | 0.39 | 3.0 | 788.9 | 2.3% | 158 | 0.73 |
| RAISE6SEC | 1.12 | 5.18 | 0.39 | 19.9 | 269.4 | 3.5% | 18 | 0.71 |
| RAISEREG | 5.47 | 5.03 | 4.22 | 20.7 | 158.5 | 0.1% | 5 | 0.58 |

Key patterns:

- **Two regimes within FCAS.** Contingency services (6SEC, 60SEC, 5MIN) have
  extreme skew (17–208), heavy tails, and price-cap spikes (capped at
  \$16,600). Regulation services (RAISEREG, LOWERREG) are far smoother
  (skew 5–62) and rarely hit caps.
- **Linear autocorrelation collapses fast for some services.** LOWER5MIN
  lag12 ACF ≈ 0.02; LOWER6SEC lag12 ≈ 0.001. RAISEREG retains lag288 ACF =
  0.21. So pure autoregressive/TTM-style generators fail for the contingency
  services.
- **Cross-service correlation matches physical structure.** Within-direction
  spike co-occurrence is 43–71% (RAISE→RAISE, LOWER→LOWER). Across-direction
  spike co-occurrence is only 1–6%. RAISE and LOWER services are essentially
  independent processes with separate drivers.
- **Linear correlation with RRP/demand/wind/solar is near zero** (all < 0.10)
  — confirming §8.2.8's TTM-near-zero-correlation finding. **But** spike
  co-occurrence with RRP spikes is strongly asymmetric: when RRP spikes,
  RAISE6SEC averages 9× its normal value; LOWER services often go *down*
  when RRP spikes. This means the signal lives in joint spike events, not
  linear levels — generative models must capture tail dependence, not
  correlation.
- **Weak diurnal pattern.** RAISE6SEC averages 0.25 at midday vs 6.07 at
  18:00 (the evening ramp). Useful as a baseline, not a generator.

### Candidate generative models for FCAS

Ranked by fit to the observed structure (spike regime, weak linear autocorr,
tail dependence, direction-asymmetric drivers):

1. **HMM regime-switching + copula (recommended v1).** Hidden Markov model
   with 2–3 states (normal, RAISE-stressed, LOWER-stressed) per direction;
   per-state emissions from heavy-tailed distributions (generalized Pareto
   for the tail, lognormal for the body); copula binding within-direction
   services to reproduce 43–71% spike co-occurrence; transition probabilities
   conditioned on exogenous features (demand ramp, wind/solar change, RRP
   spike indicator, hour). Captures the dominant structure with low complexity
   and high interpretability.
2. **Conditional diffusion model.** Generate (RRP, 8×FCAS) jointly, conditioned
   on (demand, wind, solar, hour, day-of-week, RRP-spike indicator).
   Diffusion handles heavy tails and joint spike dependence natively. Stronger
   research contribution but more compute and harder to validate.
3. **Conditional VAE with a "market stress" latent.** Shared latent encodes
   the directional stress state; per-service decoder emits heavy-tailed
   prices. Natural for the "shared RAISE trigger" structure, but VAEs tend to
   underrepresent tail magnitude.
4. **Heavy-tailed sequence model (Student-t output head).** Reuse the DT/TTM
   transformer stack with a Student-t or generalized-Pareto output
   distribution and exogenous conditioning. Lowest implementation cost (since
   the transformer infra exists), but likely remains weakest on rare spikes.

**Recommendation:** ship (1) as v1 — interpretable, captures regime-switching
and within-direction tail dependence, addresses the documented TTM failure
mode directly. Try (2) only if v1 is insufficiently expressive.

## Trackable checklist

### Phase 0 — Data foundation (~1 wk)

- [x] Extend `fetch_aemo_unit_dispatch` `columns_to_keep` (`src/aemo_data.py:1307-1309`) to retain `AVAILABILITY`, `INITIALMW`, `RAISEREGENABLEMENTMAX/MIN`, `LOWERREGENABLEMENTMAX/MIN`, `SEMIDISPATCHCAP`, `RAMPUP/DOWNRATE`.  No re-download needed — already in cached `.feather` files under `data/aemo/`.
- [x] Implement `aggregate_fcas_market_depth(region, start, end)` — per-service FCAS market depth MW per interval. Uses DISPATCHLOAD enablement sum when nonzero; falls back to TOTALDEMAND-ratio heuristic.
- [x] Implement `aggregate_residual_supply(region, start, end)` — `sum(AVAILABILITY)` per interval.
- [x] Implement `build_supply_curve(region, interval)` — sort generators by fuel-tier inferred marginal cost, accumulate MW → price-MW ladder.
- [x] Pipe new columns through `AEMODataPreprocessor` (`_merge_datasets`, `_normalize_features`).
- [x] Tests pass: 320 pass, 3 pre-existing failures (Distrobox path issue).

### Phase 1 — AEMO Oracle (rigor) (~1.5 wk)

- [ ] Create `src/aemo_oracle_algo.py` — perfect-foresight LP co-optimizer over energy + 8 FCAS services with SOC dynamics, ramp, and enablement constraints mirroring `_compute_fcas_enablement` (`src/AEMOBatteryEnv.py:942-998`). Use `pulp` (zero-friction, no solver install).
  - [ ] `Oracle_PT` (price-taking): consumes exogenous `RRP` + `FCAS_*` → ceiling for the existing price-taking env.
  - [ ] `Oracle_MI` (market-impact-aware): same constraints but uses the impact function (Phase 2) in its objective → ceiling under self-impact.
- [ ] Register both Oracle variants as evaluator baselines in `scripts/autoresearch_evaluator.py` and `AEMOAgent(algorithm='aemo_oracle', ...)`.
- [ ] Invariant tests: `Oracle_PT ≥ any replayed policy`, `Oracle_PT ≥ Oracle_MI` (self-impact only hurts).
- [ ] Run `Oracle_PT` against the dispatch-matched benchmark — must beat DT ($10,138/ep); sanity check.

### Phase 2 — Market-impact env extension (novel method) (~2.5 wk total)

- [ ] Create `src/market_impact.py` with:
  - [ ] `MarketImpactModel` base class — interface `realized_price(base_price, battery_mw, market_state) -> realized_price`.
  - [ ] `IdentityImpact` — default; must reproduce existing env byte-for-byte (golden-value test).
  - [ ] `PiecewiseMeritOrderEnergyImpact` — shift residual demand curve by battery net injection; read realized RRP off the supply ladder.
  - [ ] `PiecewiseMeritOrderFCASImpact` — per-service enablement MW shifts per-service depth curve; read realized FCAS price off the reserve-cost ladder.
  - [ ] Combined `PiecewiseMeritOrderImpact` that wraps both.
- [ ] Hook into `AEMOBatteryTradingEnv`:
  - [ ] At `src/AEMOBatteryEnv.py:879` — replace `market_data.get('RRP', 0)` with `self._impact.realized_energy(...)`.
  - [ ] At `src/AEMOBatteryEnv.py:895` (full_fcas loop) and `:918-919` (multi_market) — replace `market_data.get(f'FCAS_{service}')` with `self._impact.realized_fcas(...)`.
  - [ ] Degradation, SOC, FCAS enablement clipping unchanged.
- [ ] Config surfacing:
  - [ ] env kwargs: `impact_model`, `impact_intensity` (sweepable).
  - [ ] Evaluator surfaces gain per-policy `impact_config` field.
- [ ] Observation config flag `expose_impact_state`: default `False` → obs_dim stays 18 (all current checkpoints load). When `True` → obs gains `RESIDUAL_SUPPLY_MW` + per-service `FCAS_DEPTH_*_MW` (only used for Phase 4 retrain).
- [ ] Golden-value test: `identity` impact must reproduce existing trajectory logs within numerical tolerance.
- [ ] Pytest gate green.

### Phase 3 — Re-evaluation under impact (no retraining) (~1.5 wk)

- [ ] Expand `configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json` to multi-season/multi-station (minimum: SA1 Q4 2024 + 3 additional seasons/regions on the dispatch asset + 2 large-station sizes).
- [ ] Re-run existing policies (modern v2 DT, PPO ref, all dispatch replays, Hornsdale, Torrens Island) under 4-condition matrix:
  - [ ] {identity, market-impact} × {dispatch asset 8MWh/3.75C, large stations 150MW/250MW}
  - [ ] Report all in a single leaderboard with Oracle_PT + Oracle_MI ceilings as new rows.
- [ ] Apply bootstrap CIs + paired Wilcoxon (`bootstrap_confidence_intervals`, `paired_comparison` in `src/helper.py`) on every headline table.
- [ ] Check off README roadmap item: "Statistical confidence on AEMO headlines".

### Phase 4 — Impact-aware DT retraining (optional, MoLab; deferred until Phase 3 justifies)

- [ ] Hook impact-enabled env into `src/generate_fcas_dataset.py`.
- [ ] Regenerate FCAS dataset at true grid-scale asset sizes with `expose_impact_state=True`.
- [ ] Retrain DT on MoLab; compare impact-aware vs impact-naive DT under MI-enabled evaluation.
- [ ] Check if DT learns to moderate dispatch to avoid self-impact (a robustness dimension untouched by §8 results).

### Phase 5 — Documentation & wrap-up

- [ ] Finalize this plan file with experimental learnings.
- [ ] Add new report.md §8.2.9 "Market-Impact-Aware Evaluation".
- [ ] Check off relevant README roadmap items: "AEMO Oracle upper bound", "Statistical confidence on AEMO headlines", "Market-impact BESS evaluation".
- [ ] Update `docs/research/README.md` to link this plan.

### Phase 6 — Synthetic FCAS data generation (parallel/research thread)

Standalone research direction that both complements the impact model and
addresses the open §8.2.8 negative result on forecast DT. Decisions on whether
to enter Phase 6 are deferred until Phase 2–3 produce base impact results.

- [ ] Build evaluation harness for FCAS generators (per-service MAE/RMSE, tail
      KS-test, spike-event recall, joint spike co-occurrence, discriminative
      score vs. real held-out episodes).
- [ ] Implement v1: HMM regime-switching + copula generator conditioned on
      exogenous features (demand ramp, wind/solar delta, RRP-spike indicator,
      hour). Direction-asymmetric (two HMMs: RAISE family, LOWER family).
      Generalized Pareto tail per service.
- [ ] Compare v1 generator vs. real held-out FCAS distribution (KS test,
      autocorrelation match, spike-rate match).
- [ ] Conditional diffusion model v2 (only if v1 is insufficient).
- [ ] Validate downstream utility: train DT on synthetic-only episodes,
      evaluate against real-data-trained DT on the standard surface.

### Phase 7 — Combine impact model + synthetic FCAS (speculative)

- [ ] Use the impact model to generate interactions between battery and FCAS
      market depth.
- [ ] Train the FCAS generator on `(base_price, battery_mw, depth, state) →
      realized_price` samples from the impact model.
- [ ] Generate unlimited synthetic episodes with impact baked in.
- [ ] Retrain DT on synthetic under-impact data; compare with
      impact-aware-trained DT (Phase 4).

## Key code hooks (verified by explore)

| Hook point | File:line | What changes |
|------------|-----------|--------------|
| Price read (energy) | `src/AEMOBatteryEnv.py:879` | `market_data.get('RRP',0)` → `self._impact.realized_energy(...)` |
| Price read (FCAS, full_fcas) | `src/AEMOBatteryEnv.py:895` | `market_data.get(f'FCAS_{service}')` → `self._impact.realized_fcas(...)` |
| Price read (FCAS, multi_market) | `src/AEMOBatteryEnv.py:918-919` | same pattern for RAISEREG/LOWERREG |
| Market data row fetch | `src/AEMOBatteryEnv.py:596` | unchanged — new columns flow through `aemo_data` row |
| Observation builder | `src/AEMOBatteryEnv.py:800-848` | flag-gated addition of residual_supply + fcas_depth dims |
| obs_dim | `src/AEMOBatteryEnv.py:464,759` | conditional on `expose_impact_state` flag |
| DISPATCHLOAD columns_to_keep | `src/aemo_data.py:1307-1309` | extend list to retain availability/ramp/init columns |
| Data bundle fetcher | `src/aemo_data.py:2393-2457` | wire new aggregate helpers |
| Evaluator baselines | `scripts/autoresearch_evaluator.py` | register Oracle_PT + Oracle_MI as new baselines |
| Agent dispatch | `src/decision.py::AEMOAgent` | add `algorithm='aemo_oracle'` branch |

## Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Small asset (30 MW) sees no energy impact vs SA1 ~1.5 GW demand | Expected. FCAS impact bites (small depth: ~50-150 MW per service). Large-station condition (150/250 MW) provides the impact-signal for energy. |
| Piecewise supply curve drifts from generator static table mismatch | Cross-check `sum(AVAILABILITY)` vs `TOTALDEMAND` on cached 2020–2025 months. Flag if mismatch > 10%. |
| Oracle must be MILP (not LP) if impact is piecewise in objective | Use `pulp` with auxiliary binary decomposition, or switch to a smooth surrogate (e.g. `p * exp(-mw/depth)`) for the MIP-Oracle only. |
| FCAS depth approximation (sum of enablement) wrong vs true cleared depth | v1 risk is low (cached DISPATCHLOAD enablement is the actual cleared amount); after Phase 3 results, decide if MMSDM aggregate table is needed. |

## Session diary

<!-- Agents: add dated entries at the bottom as work progresses. Include: date, what was done, findings, blocks, and next steps. -->

### 2026-07-29 — Plan created
- Branch `feature/market-impact-modeling` created off `main` (7ea332d).
- Plan finalized: piecewise-linear merit-order impact, energy + FCAS, free aggregate depth v1, dual obs-mode config flag.
- Companion rigor item: AEMO Oracle (perfect-foresight LP co-optimizer).

### 2026-07-29 — Synthetic FCAS direction added after FCAS data investigation
- Analysed cached 5-min 2024-H1 data across SA1/NSW1/QLD1 (157K rows).
- FCAS data are regime-switching with heavy tails (skew 17–208 for
  contingency services, 5–62 for regulation) and price-cap spikes
  (\$16,600) — not a smooth time series.
- Within-direction spike co-occurrence 43–71%; cross-direction <6%. RAISE
  and LOWER services are essentially independent processes.
- Linear correlation with RRP/demand/wind/solar is near zero (all <0.10),
  explaining the §8.2.8 TTM-FCAS negative result; the signal lives in joint
  spike *events*, not levels.
- Added Phase 6 (synthetic FCAS generation) and Phase 7 (combine with impact
  model) as parallel/research threads. Recommended v1: HMM + copula with
  heavy-tailed emissions and exogenous conditioning. Deferred until Phase 2–3
  base results land.

### 2026-07-29 — Phase 0.1: Extended columns_to_keep in fetch_aemo_unit_dispatch
- `columns_to_keep` at `src/aemo_data.py:1307-1311` extended to retain
  `AVAILABILITY`, `INITIALMW`, `RAISEREGENABLEMENTMAX/MIN`,
  `LOWERREGENABLEMENTMAX/MIN`, `SEMIDISPATCHCAP`, `RAMPUPRATE`,
  `RAMPDOWNRATE`, `AGCSTATUS`.
- Empty-return schema and docstring updated to match.
- 2 dispatch tests pass; 4 pre-existing failures unrelated.
- Added cross-reference from `docs/dt_improvement_roadmap.md`.

### 2026-07-29 — Phase 0.2–0.5: aggregate helpers + preprocessor wiring
- **`aggregate_fcas_market_depth`**: sums per-unit DISPATCHLOAD enablement
  per 5-min interval. Falls back to TOTALDEMAND-ratio heuristic when
  enablement is zero (SA1 imports FCAS via interconnectors → zero local
  enablement).  Confirmed DISPATCHLOAD path works for NSW1 (6/8 services
  nonzero).
- **`aggregate_residual_supply`**: sums `AVAILABILITY` from DISPATCHLOAD
  per interval.
- **`build_supply_curve`**: merit-order ladder sorted by fuel-tier
  marginal cost (0 → 999 \$/MWh, 4+ tiers).  Verified monotonic cumulative
  MW and correct total match with aggregate_residual_supply.
- **New maps**: `FUEL_MARGINAL_COST_TIERS`, `_FUEL_SOURCE_TO_KEY`,
  `_infer_marginal_cost()` for fuel-type→cost-tier inference from AEMO
  static generator table.
- **Preprocessor**: `prepare_aemo_data` / `_merge_datasets` accept
  optional `fcas_depth` and `availability_sum` args; `_normalize_features`
  normalises FCAS_DEPTH_* and AVAILABILITY_SUM_MW to [0,1] with
  `_normalized` suffix columns.
- Tests: 320 pass, 3 pre-existing Distrobox path failures.

_Next: Phase 1 — AEMO Oracle (price-taking LP co-optimizer)._
