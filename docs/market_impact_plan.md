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

- [x] Create `src/aemo_oracle_algo.py` — perfect-foresight LP co-optimizer over energy + 8 FCAS services with SOC dynamics, ramp, and enablement constraints matching `_compute_fcas_enablement`. Uses scipy.linprog (HiGHS). ~0.02s for 288 intervals.
  - [x] `Oracle_PT` (price-taking): consumes exogenous `RRP` + `FCAS_*` → ceiling for the existing price-taking env. Validated: 0.1% gap vs env execution.
  - [ ] `Oracle_MI` (market-impact-aware): same constraints but uses the impact function (Phase 2) in its objective → ceiling under self-impact. Deferred to Phase 3.
- [x] Register Oracle_PT as evaluator baseline in `scripts/autoresearch_evaluator.py` and `AEMOAgent(algorithm='aemo_oracle', ...)`.
- [ ] Invariant tests: `Oracle_PT ≥ any replayed policy` (tbd, requires running Oracle + DT on same episodes).
- [ ] Run `Oracle_PT` against the dispatch-matched benchmark — must beat DT ($10,138/ep); sanity check.

### Phase 2 — Market-impact env extension (novel method) (~2.5 wk total)

- [x] Create `src/market_impact.py` with:
  - [x] `MarketImpactModel` base class — interface `realized_price(base_price, battery_mw, market_state) -> realized_price`.
  - [x] `IdentityImpact` — default; must reproduce existing env byte-for-byte (golden-value test).
  - [x] `PiecewiseMeritOrderImpact` — energy + FCAS combined. Energy: supply-curve shift. FCAS: depth-proportional price attenuation.
- [x] Hook into `AEMOBatteryTradingEnv`:
  - [x] At `_calculate_reward` — replace `market_data.get('RRP', 0)` with `self._impact.realized_energy_price(...)`.
  - [x] At FCAS loop and multi_market — replace `market_data.get(f'FCAS_{service}')` with `self._impact.realized_fcas_price(...)`.
  - [x] Degradation, SOC, FCAS enablement clipping unchanged.
- [x] Config surfacing:
  - [x] env kwargs: `impact_model`, `impact_intensity`, `supply_curves`, `fcas_depth`.
  - [ ] Evaluator surfaces gain per-policy `impact_config` field. (Deferred to Phase 3.)
- [ ] Observation config flag `expose_impact_state`. (Deferred to Phase 4.)
- [x] Golden-value test: `identity` impact matches default env across 100 random steps (byte-for-byte identical).
- [x] Pytest gate green (320 pass).

### Phase 3 — Re-evaluation under impact (no retraining) (~1.5 wk)

- [x] Expand to multi-season/multi-station: 3 scenarios (SA1 Oct/Nov, VIC1 Oct) × 3 battery sizes (8 MWh / 150 MW / 250 MW).
- [x] Re-run existing policies (modern v2 DT, PPO, Oracle_PT, Oracle_MI, FCAS rule) under {identity, market-impact} × {small, hornsdale, torrens}.
- [x] Report all in a single leaderboard with Oracle_PT + Oracle_MI ceilings as rows.
- [x] Bootstrap CIs (`phase3_bootstrap_over_scenarios.py`, over scenarios) + paired Wilcoxon (`phase3_paired_wilcoxon.py`, over scenario×battery cells) on headline tables.
- [x] Check off README roadmap item: "Statistical confidence on AEMO headlines".

### Phase 4 — Impact-aware DT retraining (optional, MoLab; deferred until Phase 3 justifies)

- [x] Regenerate dataset at true grid-scale asset sizes under market impact (via `scripts/generate_impact_dataset.py`; not `generate_fcas_dataset.py`).
- [x] Assemble parquet + upload to HF `mrvictoru/AEMO_simulated_trade_impact` (1,169 eps, 29.3M rows).
- [x] Create Marimo MoLab retrain notebook `notebooks/molab_notebook_dt_impact.py`; training launched on MoLab (assumed complete 2026-08-04, uploads to `mrvictoru/energydecision-dt-v2-impact`).
- [ ] Validate impact-aware DT checkpoint (architecture + load), then compare impact-aware vs impact-naive DT under MI-enabled evaluation on the Phase 3 surface.
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

### 2026-07-29 — Phase 1: AEMO Oracle (Oracle_PT) complete
- Created `src/aemo_oracle_algo.py` — LP co-optimizer for energy + 8 FCAS
  services. Uses scipy.linprog (HiGHS). Solves 288 intervals in ~0.02s.
- **Validated with 0.1% gap**: Oracle profit $10,117 vs env execution
  $10,111 on a 1-day SA1 test (zero degradation). Energy rev matches within
  $2, FCAS rev within $4.
- **Bugs found and fixed during validation**:
  1. `fetch_aemo_data_bundle_with_dispatch` argument-ordering bug
     (pre-existing, region passed as duids positionally).
  2. LP headroom constraints swapped (raise↔lower vs charge↔discharge).
  3. Oracle dispatch sign convention opposite to env's convention
     (positive=discharge vs positive=charge).
  4. FCAS bid slot order mismatched env's `_fcas_services` ordering
     (RAISEREG first, not logical-order first).
- **Integrated** into AEMOAgent (`src/decision.py`) via `_init_oracle()`
  and `_oracle_action()`. Registered as evaluator baseline
  (`scripts/autoresearch_evaluator.py` via `policy_kind="oracle"`).
- Added `oracle_pt` policy entry to
  `configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json`.

### 2026-07-29 — Phase 3: first market-impact re-evaluation results

**Hardware:** 2080 Ti (0.44/21 GB VRAM), 24-core CPU (62 GB RAM).
DT inference: 21.6s per 14-day episode (3744 steps). Not memory-bound.

**Key findings** (SA1 Oct+Nov 2024, 8MWh/30MW, full_fcas, zero degradation):

| Impact | Oracle_PT | DT (RTG=10) | DT as % of Oracle |
|--------|----------:|------------:|:-----------------:|
| identity | $204,165 → $195,368 | $13,358 → $10,325 | **6.5%** |
| merit-order | $28,712 → $46,936 | $11,846 → $9,142 | **41%** |
| **Change** | **-81%** | **-12%** | — |

- **DT is the most impact-resilient policy.** Loses only 12% under
  market impact while Oracle collapses 81% and FCAS rule drops 93%.
- **Conservative dispatch is a natural hedge.** DT earns $374 energy
  rev vs Oracle's $129K. Under impact, the Oracle's aggressive
  arbitrage moves the price heavily against itself.
- **FCAS-rule is the worst under impact** ($33K → $2.6K in Oct).

**Caveats:**
- DT evaluated at RTG=10 (calibrated for price-taking). Impact-aware
  RTG calibration might improve DT under impact.
- Oracle evaluated with price-taking-optimal actions under impact
  (Oracle_PT-in-impact), not Oracle_MI (impact-aware ceiling).
- Only 3 scenarios; expand to 5 regions for full CIs later.

### 2026-07-30 — Phase 3: PPO + VIC1 added, RTG calibration complete

Added PPO reference policy via single-obs predict (no VecEnv) and VIC1
Oct 2024 scenario. Confirmed RTG calibration with `run_episode()`.
DT responds to RTG prompts now; optimal RTG depends on both scenario and
impact mode.

**Impact resilience (% profit retained under impact, best RTG per cell):**

| Policy | SA1 Oct | SA1 Nov | VIC1 Oct | Avg |
|---|:---:|:---:|:---:|:---:|
| DT | +3% | -9% | -33% | -13% |
| PPO | -45% | -37% | -33% | -38% |
| Oracle | -86% | -76% | -71% | -78% |
| FCAS Rule | -93% | -94% | -65% | -84% |

DT reaches 48% of Oracle under impact (vs 6.7% without).

_Next: decide on Phase 4 retraining._
- Created `src/market_impact.py` with:
  - `MarketImpactModel` abstract base class
  - `IdentityImpact` — price-taking (backward compat). Golden-value test
    passes: byte-for-byte identical to default env across 100 random steps.
  - `PiecewiseMeritOrderImpact` — realized energy price from supply-curve
    shift; realized FCAS price from depth-proportional attenuation.
  - `create_impact_model()` factory function.
- Wired into `AEMOBatteryTradingEnv`:
  - New kwargs: `impact_model` (str or instance), `impact_intensity`,
    `supply_curves`, `fcas_depth`.
  - Energy price read at `_calculate_reward` (line 937) → routed through
    `self._impact.realized_energy_price()`.
  - FCAS price read (line 953) and multi_market reads (976-977) → routed
    through `self._impact.realized_fcas_price()`.
- Tests: 320 pass, 3 pre-existing.

_Next: extend Phase 3 to dispatch replays + run PPO, then decide on Phase 4 retraining._
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

_Next: Phase 2 — Market-impact env extension._

### 2026-07-31 — CRITICAL CORRECTION: Phase 3 used the LEGACY model, not v2

All Phase 3 impact-resilience numbers up to this point were generated with
`models/aemo/dt/hf_eval/aemo_dt_fcas_model.pt`, which is the **legacy 8×384
MoLab-style checkpoint** (confirmed: `embed_return`/`blocks`/`predict_` keys,
loads as `LegacyDecisionTransformer`). The SOTA **modern v2** (8×768 GQA,
12 heads, ctx=210, return_scale=1.0) lives at `mrvictoru/energydecision-dt-v2`
and was NOT being used.

**Fix:** copied the v2 checkpoint to `models/aemo/dt/hf_v2_modern/` and pointed
`phase3_impact_eval.py` at it. Verified: loads as modern `DecisionTransformer`,
forward returns `act_preds [B,T,9]`, return_scale=1.0.

**v2 validation (SA1 Oct, small battery):**
| impact | rtg | v2 profit | legacy profit (WRONG) |
|---|---|---|---|
| identity | 0.0 | $19,310 | $13,420 |
| identity | 10.0 | $33,985 | $11,772 |
| impact | 0.0 | $14,920 | $13,855 |
| impact | 10.0 | $22,518 | $7,375 |

**Action:** the full 3-scenario × 3-battery × 2-impact sweep is being
re-run with v2 (launched 2026-07-31, ~2 hr). Bootstrap CI + Wilcoxon script
(`scripts/phase3_bootstrap.py`) ready to run after. All prior Phase 3 tables
in the diary/report are superseded by the v2 run.

_Next: v2 full sweep → bootstrap CIs → update report §8.2.9 with v2 numbers._

### 2026-07-31 — v2 Phase 3 sweep complete (supersedes all legacy numbers)

Full 3-scenario × 3-battery × 2-impact sweep with the modern v2 (8×768) model.
Legacy 8×384 numbers in earlier diary/report entries are WRONG and superseded.

**Corrected v2 impact resilience (avg % profit retained):**
| battery | DT | PPO | Oracle |
|---|---:|---:|---:|
| small (8 MWh) | 62% | 62% | 22% |
| hornsdale (150 MW) | 83% | 40% | 4% |
| torrens (250 MW) | 49% | 32% | 2% |

Key: at small scale v2 DT == PPO resilience (62%); the DT edge is
scale-dependent (83% vs 40% at 150 MW). v2 DT earns 1.4–1.8× PPO absolute
identity profit. Oracle collapses at scale (model-independent).

Per-episode ~59s (v2). Results: eval_output/phase3_v2/ (gitignored).

_Next: bootstrap CI + Wilcoxon, then update report §8.2.9 with v2 numbers._

### 2026-07-31 — Phase 3 complete: v2-based results + statistical confidence

- **Checkpoint correction:** retired the legacy 8×384 model; all Phase 3
  measurement now uses the modern v2 (8×768) from `mrvictoru/energydecision-dt-v2`
  (copied to `models/aemo/dt/hf_v2_modern/`). Legacy numbers in earlier diary
  entries are superseded.
- **Full v2 sweep** (3 scenarios × 3 batteries × 2 impacts × DT-sweep/PPO/
  Oracle/Oracle_MI/rule): `eval_output/phase3_v2/sweep_full.txt` (gitignored).
- **Corrected v2 impact resilience (DT best RTG):** small 62% [53–66%],
  hornsdale 78% [31–133%], torrens 49% [44–56%]. DT == PPO at small scale;
  DT edge is scale-dependent. Oracle collapses 22%→2%.
- **Bootstrap CIs** over scenarios: `scripts/phase3_bootstrap_over_scenarios.py`.
- **Paired Wilcoxon** (DT vs PPO, n=9 matched scenario×battery cells):
  impact p=0.098, DT wins 8/9 cells, mean diff +$18,421.
  `scripts/phase3_paired_wilcoxon.py`. n=9 < 10 → flagged for 5-region expansion.
- **Report:** report.md §8.2.9 + takeaway #9 updated with v2 numbers.
- **Concurrency finding:** neither multiprocessing nor batched inference helps
  this legacy-manual-attention-era DT (measured); serial ~59s/ep is the floor.

Phase 3 complete. Remaining: Phase 4 (impact-aware retrain, optional, MoLab)
and Phase 6 (synthetic FCAS) are the open research threads.

### 2026-07-31 — Phase 4 finalized: impact-aware dataset design + generation started

**Dataset spec (finalized with user):**
- **Batteries:** diverse 8 / 50 / 150 / 250 MWh (FCAS impact visible at 8–50;
  energy impact only at 150–250).
- **Sources (mix):** Oracle_MI ~600 (impact-optimal), PPO ~400, modern-v2 DT
  self-generated ~300, A2C ~200, Oracle_PT ~150 (failure-mode contrast),
  fcas_rule ~150. Total ~1,800 eps.
- **Oracle diversity:** Oracle_MI/Oracle_PT are deterministic LP solves —
  diversity comes from solving over sampled 14-day sub-windows per region
  (distinct trajectory per window), not re-running.
- **Horizons:** short 12-day (40%, matches eval), medium 8-week (30%),
  long 26-week (20%), ~9-month (10%). Long horizons teach capacity-fade-aware
  operation (degradation). Oracle LP sources restricted to short/medium
  (LP size grows with horizon).
- **Dates:** reuse 2021-2023 training windows (processed data cached); supply
  curves + FCAS depth precomputed for these windows (~7 hr DISPATCHLOAD
  download, one-time, launched in background).
- **Degradation:** real_world (LFP, 30C) in training AND eval. Phase 3 impact
  eval switched none -> real_world to match.
- **Obs:** state_dim stays 18 (no horizon feature added — model learns horizon
  from day/sin-cos time features + timestep embedding + RTG scale; matches the
  real-world setting where remaining-fraction isn't a fixed input).
- **Files:** `scripts/generate_impact_dataset.py` (generator),
  `scripts/precompute_supply_curves.py` (7 hr supply/depth precompute).
- **Next:** after precompute -> generate ~1,800 eps -> assemble -> upload to
  HF (`mrvictoru/AEMO_simulated_trade_impact`) -> MoLab pilot retrain (~500 eps)
  -> validate vs pretrained v2 on Phase 3 surface -> full retrain.

### 2026-08-01 — Phase 4 dataset generation LAUNCHED (running)

- **Precompute** (`precompute_supply_curves.py`) running: 2021-2023 supply
  curves + FCAS depth per region. DISPATCHLOAD backfill (~2020-12→2022-12)
  now cached; remaining regions reuse it. FCAS depth switched to fast
  demand-heuristic (the multi-year DISPATCHLOAD aggregation was 30+ min/
  region; heuristic is instant and consistent across regions).
- **Generation** (`generate_impact_dataset.py`, `run_impact_gen_all.sh`)
  launched with a 10-worker process pool (uses the 12-core CPU). Orchestrator
  auto-launches each region as its supply cache appears. Validated with a
  12-episode pilot (all sources write valid parquet).
- **Sources:** oracle_mi (LP, short-biased 75%), oracle_pt, ppo, dt_v2
  (self-gen), a2c, fcas_rule. Batteries 8/50/150/250. Horizons short/medium/
  long/xlong. Degradation real_world (LFP, 30C). Impact piecewise_merit_order.
- **Known cost:** medium-horizon oracle_mi is slow (~5-9 min/LP); oracle LP
  iterations cut to 3. Full ~1800-ep run is a multi-hour background job
  (expected ~3-5 hr given long-horizon rollouts).
- **Next (after completion):** assemble parquet → upload to HF
  (`mrvictoru/AEMO_simulated_trade_impact`) → MoLab pilot retrain.

### 2026-08-04 — Phase 4 dataset + retrain DONE (training assumed complete)

- **Dataset finalized:** `data/aemo_dt_impact/aemo_impact_dataset.parquet`
  (1,169 eps, 29,270,943 rows, 1.5 GB) assembled with exact 18-dim
  normalized obs reconstructed from aemo_data + recorded SOC (validated 0.0
  diff vs env replay). Uploaded to HF
  `mrvictoru/AEMO_simulated_trade_impact` (live, last modified 2026-08-03).
- **Composition:** regions NSW1/QLD1/SA1/TAS1/VIC1 (2021-23); batteries
  b08/b50/b150/b250; horizons short/medium/long/xlong; real_world deg (LFP,
  30C); impact piecewise_merit_order. Sources: oracle_mi 342, ppo 245,
  dt_v2 213, a2c 170, oracle_pt 100, fcas_rule 99.
- **MoLab retrain:** launched via
  `notebooks/molab_notebook_dt_impact.py` (data repo
  `AEMO_simulated_trade_impact`). TRAINING ASSUMED COMPLETE; model uploads
  to **`mrvictoru/energydecision-dt-v2-impact`**.
- **Config caveat:** notebook default `epochs_per_session=2`. Halved dataset
  (~4,300 gradient steps) vs original 2,401-ep run (~11,400). If the run
  used 2 epochs, treat any under-performance vs pretrained v2 as partly
  under-training; a 3-4 epoch rerun is the cheap mitigation.
- **Next agent session (validate → evaluate → report):**
  1. Download `mrvictoru/energydecision-dt-v2-impact`; verify it loads with
     the modern-v2 config (`src/aemo_dt_hf.py` helpers). Verify architecture
     from embedded weights, not docs (AGENTS.md rule).
  2. Evaluate on the Phase 3 surface with `scripts/phase3_impact_eval.py`
     (3 scenarios × 3 batteries × identity/impact), comparing impact-aware
     DT vs canonical v2 (`models/aemo/dt/hf_v2_modern/aemo_dt_fcas_model.pt`).
  3. Re-run bootstrap CIs (`phase3_bootstrap_over_scenarios.py`) + paired
     Wilcoxon (`phase3_paired_wilcoxon.py`) with the new model added.
  4. Answer the research question: does the DT moderate dispatch to avoid
     self-impact? Compare per-interval dispatch/FCAS action magnitudes under
     MI-enabled eval vs the naive v2.
  5. Update report §8.2.9 + Phase 5 docs with results.
