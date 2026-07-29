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

## Trackable checklist

### Phase 0 — Data foundation (~1 wk)

- [ ] Extend `fetch_aemo_unit_dispatch` `columns_to_keep` (`src/aemo_data.py:1307-1309`) to retain `AVAILABILITY`, `INITIALMW`, `RAISEREGENABLEMENTMAX/MIN`, `LOWERREGENABLEMENTMAX/MIN`, `SEMIDISPATCHCAP`, `RAMPUP/DOWNRATE`.  No re-download needed — already in cached `.feather` files under `data/aemo/`.
- [ ] Implement `aggregate_fcas_market_depth(region, services, start, end)` — sum per-unit enablement across all DUIDs per service per interval per region → `FCAS_DEPTH_<SERVICE>_MW`.
- [ ] Implement `aggregate_residual_supply(region, start, end)` — `sum(AVAILABILITY) - TOTALDEMAND` per interval per region → `RESIDUAL_SUPPLY_MW`.
- [ ] Implement `build_supply_curve(region, interval)` — sort generators by fuel-tier inferred marginal cost, accumulate MW → price-MW ladder.
- [ ] Pipe new columns through `AEMODataPreprocessor` (`_resample_*` family, `src/AEMOBatteryEnv.py:155-225`) and `_normalize_features` (`:281-327`).
- [ ] Tests pass: `python -m pytest tests/ -v`.

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

- [ ] Write `docs/market_impact_plan.md` (this file) — finalize, add learnings, and publish results.
- [ ] Add new report.md §8.2.9 "Market-Impact-Aware Evaluation".
- [ ] Check off relevant README roadmap items: "AEMO Oracle upper bound", "Statistical confidence on AEMO headlines", "Market-impact BESS evaluation".
- [ ] Update `docs/research/README.md` to link this plan.

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

_Next: Phase 0 — extend DISPATCHLOAD columns_to_keep and implement aggregate helpers._
