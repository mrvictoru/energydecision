# Known Issues and Modelling Caveats

This file tracks code-level inconsistencies, modelling approximations, and
documentation defects that are **not** covered by the other tracking surfaces:

- `docs/FUTURE_PLAN.md` — forward research plan / open experiments
- `results.tsv` — experiment ledger
- `report.md` Appendix C — implementation notes embedded in the report

It exists because several of these items were only discoverable by reading the
source, and a PhD-level write-up must not silently build on an undocumented
approximation. Each entry gives the location, the impact, and a suggested
resolution. When an item is fixed, move it to the **Resolved** section (or
delete it) rather than leaving a stale entry.

Status key: `OPEN` = not addressed; `WORKAROUND` = mitigated but not fixed;
`INTENTIONAL` = deliberate, needs documenting, not fixing; `STALE-DOC` =
documentation-only correction.

---

## A. Household environment

### A1. `degradation_mode="full"` is identical to `"cycle_only"` — no calendar aging

- **Location:** `src/EnergySimEnv.py:136-141` (mode selection), `SolarBatteryEnv.step`
  (no `degradation_per_timestep` call anywhere in the file).
- **Detail:** `"full"` selects the Muenzel `DegradationModel` and `"cycle_only"`
  selects `CycleOnlyDegradationModel`, but `step()` only ever calls
  `degradation_per_cycle` (via the rainflow counter). The `RealWorldBESSDegradationModel`
  calendar+cycle model is wired only into `AEMOBatteryEnv`.
- **Impact:** the H4.5 `full_realistic` arm does not test calendar aging; CLI help
  in `scripts/h4_degradation_study.py` / `scripts/generate_household_sdp_trajectories.py`
  that implies "calendar+cycle" is misleading.
- **Resolved (2026-09-13):** `SolarBatteryEnv(mode="full")` now adds
  `RealWorldBESSDegradationModel.calendar_aging_per_step` (Arrhenius + SOC
  stress; NMC/LFP preset via new `degradation_chemistry` arg, default LFP) on
  top of the rainflow/Muenzel cycle aging; `info["calendar_degradation"]` is
  exposed. `cycle_only`/`disabled` are unchanged. Idle 24 h now accrues ~2.9e-5
  fractional loss (LFP, 25 °C, 50% SOC). Tests:
  `tests/test_environment_calendar.py`. Note: cycle aging remains Muenzel, so
  `full` is a calendar(RealWorld)+cycle(Muenzel) composite — deliberate.
  Status: `RESOLVED` (re-baseline pending).

### A2. No round-trip efficiency in the env, but the SDP teacher assumes `sqrt(0.80)`

- **Location:** `src/EnergySimEnv.py:399` (`new_level = battery_level + battery_flow_energy`);
  `src/household_optimization.py:60,89,98` (`roundtrip_eff=0.80`, `eff = roundtrip_eff ** 0.5`).
- **Detail:** the env is lossless; the teacher plans with symmetric charge/discharge
  losses. Teacher SoC paths and realized env SoC therefore diverge.
- **Impact:** distillation labels/RTG are computed under dynamics the env does not
  have; any "honest teacher" claim must state which efficiency convention is used.
- **Resolved (2026-09-13):** `SolarBatteryEnv` now has a `roundtrip_eff` arg
  (default **0.80**); the stored-energy change applies symmetric one-way
  efficiency (`eff = sqrt(roundtrip_eff)`; charge stores `ge*eff`, discharge
  draws `ge/eff`). The safety `DailyThroughputProjector` is now
  efficiency-aware, and the teacher generator passes its `roundtrip_eff` to the
  env too. Tests: `tests/test_environment_efficiency.py` (SOC-limit/projector
  tests set `roundtrip_eff=1.0` to stay focused). This changes household
  dynamics, so all household results shift. Status: `RESOLVED` (re-baseline
  pending).

### A3. Rainflow inferred C-rate is ~100× too large and saturates the clamp

- **Location:** `src/batterydeg.py:359-373` (current computed as `ΔSoC% / hours`,
  with no `/100` conversion to C-rate), `batterydeg.py:7-12`
  (`Id_nom=0.25`, `Ich_nom=0.125` are true C-rates).
- **Impact:** the C-rate multipliers (`nCL_Id`, `nCL_Ich`) are effectively pinned
  near the `max_c_rate` clamp for realistic cycles, weakening the rate dependence.
- **Suggested fix:** divide by 100 (or carry SOC as a fraction), then re-validate
  the nominal multipliers and any reported degradation figures.
- **Resolved (2026-09-13):** `RainflowCounter.update` now computes the inferred
  current as `ΔSoC% / 100 / Δt` (a true C-rate). Unit test:
  `tests/test_batterydeg_units.py`. Controlled delta on a fixed square-wave
  profile: household 7-day cycle-only cumulative degradation 0.02058 → 0.01487
  (**−27.7%**); AEMO SA1 Oct 8 MWh/30 MW `real_world` LFP 288-step degradation
  cost $4,456.77 → $2,739.34 (**−38.5%**). Because this is shared code, both
  tracks shift; see §D.0. Status: `RESOLVED` (re-baseline pending).

### A4. `EnergySimEnv.reset()` drops the rainflow C-rate cap

- **Location:** `src/EnergySimEnv.py:143` constructs `RainflowCounter(...,
  max_c_rate=max_battery_flow/initial_capacity)`; `reset()` at
  `src/EnergySimEnv.py:303` reconstructs it **without** `max_c_rate`, reverting to
  the default `1.0`.
- **Note:** `AEMOBatteryEnv` does **not** have this bug — its reset passes
  `max_c_rate` (`src/AEMOBatteryEnv.py:501/508/634`).
- **Impact:** household degradation after `reset()` uses a different clamp than the
  constructor implies; results depend on init/reset path.
- **Suggested fix:** pass `max_c_rate` in `EnergySimEnv.reset()`.
- **Resolved (2026-09-13):** `EnergySimEnv.reset()` now passes
  `max_c_rate = max_battery_flow / initial_battery_capacity`, matching the
  constructor. Test: `tests/test_batterydeg_units.py`. Status: `RESOLVED`.

### A5. `SDPSolver` prices degradation at a fixed representative SoC

- **Location:** `src/sdp_algorithm.py:316,325-329` (`rep_soc = capacity/2`).
- **Impact:** the DP's degradation cost is state-independent, so it cannot trade
  off wear against SOC position. `OracleSolver` uses the true state
  (`src/oracle_algorithm.py`), so SDP and Oracle are not directly comparable on
  degradation.
- **Resolved (2026-09-13):** the fixed-midpoint approximation was not just
  state-independent — the underlying `DegradationCalculator.compute_rainflow_degradation`
  fed only 3 points to `RainflowCounter`, which can never close a cycle, so the
  SDP/MRDP/Oracle stage-cost degradation term was **always zero**. Replaced with
  `compute_step_degradation` (a per-step half-cycle estimate at the true SoC
  and C-rate) and `SDPSolver._deg_cost_grid` now prices wear state-dependently,
  precomputed once per solver. Tests: `tests/test_physics_v2_planner.py`. Status:
  `RESOLVED` (planner policies are no longer degradation-blind; re-baseline
  pending).

### A6. Household SDP teacher is degradation-blind; `J_t(soc)` excludes wear

- **Location:** `src/household_optimization.py:54-141` (`optimize_dispatch` has no
  degradation term); `scripts/generate_household_sdp_trajectories.py:105-107`
  (`rtg_value = -J_t(soc)`, grid-cost only).
- **Impact:** household distillation labels and RTG do not encode degradation; wear
  reaches the DT only through realized reward and capacity fade. `report.md` §4.3
  describes the AEMO teacher's `λ_deg` and can mislead readers about the household
  teacher.
- **Resolved (2026-09-13):** `optimize_dispatch` and
  `build_j_t_soc_prompt_provider` now accept `deg_cost_per_mwh` (linear
  throughput wear term; default 0.0 preserves the degradation-blind oracle).
  `scripts/generate_household_sdp_trajectories.py` exposes
  `--deg-cost-per-mwh` (default **50.0**) so teacher labels are wear-aware.
  Delta on a representative synth day: throughput 24.7 kWh (λ=0) → 15.1 (λ=20)
  → 14.8 (λ=50), with a ~$0.024/day bill increase. Tests:
  `tests/test_physics_v2_planner.py`. Status: `RESOLVED` (teacher λ calibration
  still open; re-baseline pending).

### A7. Household observation degradation-cost normalization is tied to `battery_life_cost`

- **Location:** `src/EnergySimEnv.py:15,185` — normalizer is `0.001 * battery_life_cost`;
  obs dim 11 is `deg_cost / normalizer`.
- **Impact:** the H4.5 `$1,000`/`$10,000` life-cost arms change the observation
  scale and perturb the policy; this confounds those comparisons.
- **Suggested fix:** use a fixed reference or capacity-relative normalizer.
  Status: `OPEN`.

### A8. Capacity-fade normalization stays pinned to initial capacity

- **Location:** `src/EnergySimEnv.py:522-525` — `capacity = initial*(1 - total_degradation)`
  while observation normalizers remain anchored to `initial_battery_capacity`.
- **Impact:** observation ranges drift as the battery fades; minor for short
  windows, relevant for H4.9 long-horizon runs. Status: `OPEN` / document.

---

## B. AEMO environment, planners, and data

### B1. Impact-aware `J_t(soc)` uses the opposite dispatch sign to the env

- **Location:** `src/aemo_sdp_executor.py:359`
  (`dispatch_mw = -energy/step_duration`, where `energy > 0` = charging);
  `src/market_impact.py:142` (`effective_demand = total_demand + battery_dispatch_mw`);
  `src/AEMOBatteryEnv.py:689,960,1045` (env passes `actual_power`, positive = charging).
- **Detail:** with the env convention (positive = charging) the impact model
  correctly raises demand on charge. The cost-to-go table passes `-energy/Δt`,
  so charging yields a **negative** dispatch and *lowers* demand — the opposite of
  the environment. The `market_impact.py` docstring/comment (`:33`, `:140`) also
  labels positive as discharge, which is wrong relative to the env.
- **Impact:** the H1 "impact-aware `J_t(soc)`" table does not match env-realized
  prices; the verified impact-gate pass is currently carried by `rtg_mode="auto"`
  falling back to constant RTG, not by H1 being correct.
- **Suggested fix:** align the sign in `compute_cost_to_go_table` with the env,
  fix the comments, and re-run the impact gate to test whether explicit
  `j_t_soc` becomes viable.
- **Resolved (2026-09-13):** sign corrected in `src/aemo_sdp_executor.py:359`
  (and the misleading `src/market_impact.py` comments fixed) so the cost-to-go
  table uses the env's positive=charging convention; regression coverage added
  in `tests/test_market_impact.py`, and end-to-end by
  `scripts/verify_jtsoc_impact_sign.py` (SA1 Oct, piecewise merit-order): with
  the corrected sign and checkpoint `return_scale`, explicit j_t_soc beats
  constant on all three batteries. Full suite green (378 passed, 2026-09-13).

### B2. FCAS service ordering differs between the env and the Oracle

- **Location:** env `_fcas_services` (`src/AEMOBatteryEnv.py:489`) interleaves
  raise/lower; the Oracle groups raise then lower (`src/aemo_oracle_algo.py:55`).
- **Detail:** `src/decision.py` remaps the Oracle output, but any direct consumer
  of the oracle bid arrays must apply the same mapping.
- **Resolved (2026-09-13):** mapping centralised in
  `oracle_fcas_bids_to_env_order` (`src/aemo_oracle_algo.py`) and used by both
  `decision.py` call sites; the misleading "order matches the env" comment was
  corrected. Regression tests in `tests/test_aemo_fcas_order.py`.

### B3. `aggregate_fcas_market_depth` is referenced but undefined

- **Location:** imported in `scripts/phase3_v2_validate.py:18` and
  `scripts/phase3_impact_eval.py:21`; referenced in comments in
  `src/aemo_oracle_algo.py:411`, `src/AEMOBatteryEnv.py:93`,
  `src/market_impact.py:76,82`. No `def aggregate_fcas_market_depth` exists in the
  repository.
- **Live substitute:** the demand-proportional heuristic `fast_fcas_depth` in
  `scripts/precompute_supply_curves.py:18-36`, written to the precomputed cache.
- **Impact:** the importing scripts raise `ImportError` if run directly; impact-gate
  supply/depth data must come from the cache.
- **Suggested fix:** implement `aggregate_fcas_market_depth` or update the
  importers/comments to the live function.
- **Resolved (2026-09-13):** added `aggregate_fcas_market_depth(region, start,
  end, demand_series=None)` to `src/aemo_data.py` as the demand-proportional
  heuristic, so the Phase 3 scripts' imports resolve;
  `scripts/precompute_supply_curves.py` keeps its local fast path. Tested in
  `tests/test_market_impact.py`.

### B4. Env and SDP planner use different degradation models

- **Location:** env `real_world` uses `RealWorldBESSDegradationModel`; the SDP
  executor's internal estimator uses the rainflow `DegradationCalculator` plus a
  linear throughput surrogate (`src/aemo_sdp_executor.py:142,264`).
- **Impact:** the teacher's planned wear cost differs from the env's realized wear.
  Partly documented in `report.md` §8.2.10 (the sub-3% DoD note) but not framed as
  a model mismatch.
- **Documented (2026-09-13):** the mismatch is now stated in the
  `sdp_energy_dispatch` docstring (`src/aemo_sdp_executor.py`) and this file.
  Using one model in both remains a follow-up. Status: `DOCUMENTED`.

### B5. Muenzel model returns zero degradation for DoD ≤ 3%

- **Location:** `src/batterydeg.py:219-220`.
- **Impact:** 5-minute partial cycles routinely fall under this threshold, so the
  rainflow path under-counts; the SDP teacher compensates with a linear
  `λ_deg` throughput surrogate.
- **Status:** documented in `report.md` §8.2.10; keep visible for new users.
  `WORKAROUND`.

### B6. The shipped Stage C checkpoint does not embed its architecture

- **Location:** `src/transformer_training.py:1453` saves a pure `state_dict`.
  Architecture and calibration live in the training surface manifest
  (`models/aemo/dt/aemo_dt_sdp_jtsoc_fullcorpus_loss_surface_manifest.json`) and
  the `.meta.json` sidecar (`return_scale=25988.190625`, `max_timestep=2016`,
  `rope_enabled=true`, `action_head_mode="mixed"`).
- **Impact:** the published `configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json`
  is the **wrong** config for Stage C. Any loader that assumes an embedded config
  will fail or load the wrong architecture.
- **Documented (2026-09-13):** `report.md` §4.2 corrected and a note added at the
  `torch.save` site in `src/transformer_training.py`; loaders must use the
  sidecars. Writing the architecture into future checkpoints remains a follow-up.
  Status: `DOCUMENTED`.

### B7. The env step reward is scaled by 1/1000

- **Location:** `src/AEMOBatteryEnv.py:1033` (`normalized_reward = reward/1000`).
- **Impact:** raw dollars live only in `info`; a common source of confusion when
  comparing logged rewards to profits. Documented but easy to miss.

### B8. `phase3_impact_eval.py` does not apply the checkpoint's `return_scale`

- **Location:** `scripts/phase3_impact_eval.py:182-186` loads the checkpoint with
  `torch.load(...)` and passes the resulting **dict** to
  `DecisionTransformer.load_from_checkpoint`, which only reads the
  `<ckpt>.meta.json` sidecar when given a **path**
  (`src/decision_transformer.py:539-551`). The modern model therefore keeps its
  constructor default `return_scale=1.0` (`src/decision_transformer.py:435`),
  while the requested `return_scale` is not consulted because it lives outside
  the filtered constructor kwargs.
- **Impact:** for the shipped Stage C model the correct value is
  **25988.190625**; any impact-eval run of Stage C through this script (notably
  the explicit `j_t_soc`/`auto` investigation in
  `docs/aemo_dt_preferred_policy_plan.md`) used `1.0`, i.e. an RTG prompt
  ~26,000× too large. This may partly explain the reported hornsdale/torrens
  collapse and must be re-checked before those numbers are used.
- **Suggested fix:** after `load_from_checkpoint`, read the sidecar
  (`<ckpt>.meta.json`) and set `dt_model.return_scale` (or pass the checkpoint
  path to `load_from_checkpoint`). A focused verification harness
  (`scripts/verify_jtsoc_impact_sign.py`) loads the sidecar correctly and keeps
  an explicit `--return-scale` override for comparison.
- **Resolved (2026-09-13):** `phase3_impact_eval.py` now applies the sidecar
  `return_scale`. Verified on SA1 Oct (piecewise impact): with
  `return_scale=1.0` the j_t_soc collapse reproduces (hornsdale −$46k, torrens
  −$326k); with the correct `25988.19` explicit j_t_soc is positive and beats
  constant on all three batteries (+$33.7k / +$238.8k / +$134.6k).
  **Implication:** the reported "price-taking J_t(soc) fails under impact"
  conclusion was dominated by this bug, so the shipped `rtg_mode="auto"` gating
  should be re-derived on the canonical impact benchmark before it is final.
- **All-scenario follow-up (2026-09-13, `jtsoc_sign_verification_all.json`):**
  across 3 scenarios × 3 batteries, explicit j_t_soc no longer collapses
  anywhere. It wins the three SA1 Oct cells (incl. hornsdale +$238.8k vs
  +$115.6k), ties SA1 Nov small / VIC1 small+hornsdale, and loses SA1 Nov
  hornsdale/torrens and VIC1 torrens where it over-trades energy (SA1 Nov
  torrens energy −$68k). Constant is marginally ahead in total (901.6k vs
  879.5k over 9 cells, ~2.5%). So `auto` (constant fallback under impact)
  remains defensible, but the "catastrophic collapse" justification is invalid;
  a surface/battery/season-aware mode selection may beat both. Status:
  `RESOLVED` (report narrative still to be revised).

---

## C. Stale documentation claims (corrected in this pass)

- `report.md` §2 had the "Reinforcement learning for battery control" paragraph
  duplicated verbatim.
- `report.md` §8.2.4 claimed GRPO "transfers", contradicting §8.2.7/abstract
  (GRPO does not improve the modern 8×768 model).
- `report.md` §8.1 and the Conclusion generalised a legacy Ausgrid
  "DT beats Oracle" result that is reversed on the modern household rebuild
  (§8.1.1).
- `report.md` §4.2 conflated the historical modern-v2 config (RoPE off,
  `return_scale=1.0`) with the shipped Stage C config (RoPE on,
  `max_timestep=2016`, auto `return_scale`), and claimed architecture was
  recoverable from an "embedded config".
- `report.md` referenced `src/pretrain_decision_transformer.py`; the entrypoint is
  `scripts/pretrain_decision_transformer.py`.
- `AGENTS.md` stated the household DT "beats Oracle" without scoping it to the
  legacy benchmark.

---

## D. Remediation plan for the result-invalidating physics issues (A1–A8)

These changes alter household degradation physics and/or observation scaling, so
they invalidate every existing household number (H1–H4.x and the legacy Ausgrid
results in `report.md` §8.1) and require regenerating teacher corpora,
retraining, and re-running evaluation. Do them as one coherent
"degradation-physics v2" change, not piecemeal.

### D.0 Scope decision (required before any code change)

**Cross-track caveat:** `RainflowCounter`/`DegradationModel`
(`src/batterydeg.py`) are shared by `SolarBatteryEnv` **and**
`AEMOBatteryTradingEnv`. A3 (C-rate units) and A4 (reset cap) are therefore
**not** household-only. Impact by track:

- **Household (H1–H4.x):** the degradation term enters the reward directly, so
  every result shifts — plausibly the most, given H4.5's conclusions hinge on
  net-of-wear sign.
- **AEMO (Stage C identity surfaces + impact gate):** degradation cost changes,
  so the just-corrected canonical results (B1/B8) would need re-running again.

Options:

1. **Global fix + re-baseline both tracks** — scientifically cleanest; highest
   cost (regenerate corpora, retrain, re-run AEMO identity/gate). Recommended
   end state.
2. **Household-scoped correct-units path** — add a `batterydeg` option (e.g.
   `c_rate_units="fraction"`) used only by `SolarBatteryEnv`, leaving the AEMO
   default unchanged to preserve its results. Fast, but leaves AEMO carrying the
   approximation and creates a two-convention divergence needing explicit
   labelling everywhere.
3. **Document only** — no code change; cite A1–A8 as limitations. Cheapest,
   weakest.

**Recommendation:** option 2 first as a clearly-labelled experiment branch to
quantify the household delta, then option 1 once the delta is understood and the
household track is frozen. Fix A3 and A4 together so the clamp is consistent.

### Coupling and order

1. **A3 (rainflow C-rate units) + A4 (reset cap)** — fix together. A3 changes the
   inferred current by ~100×, so the `max_c_rate` clamp (A4) stops being the
   dominant factor. Recompute the nominal denominators exactly once.
2. **A2 (round-trip efficiency)** — add symmetric efficiency to
   `SolarBatteryEnv` (default RTE = 1.0) and set the teacher `roundtrip_eff`
   consistently. Decide the canonical RTE (0.80 vs 1.0) before regenerating data.
3. **A1 (calendar aging)** — wire `RealWorldBESSDegradationModel` (calendar+cycle)
   into `SolarBatteryEnv` for `degradation_mode="full"`, or explicitly retire the
   `"full"` label so `"full"`/`"cycle_only"` cannot be confused. Depends on A2/A3
   for a clean per-step wear number.
4. **A5/A6 (planner degradation)** — give `optimize_dispatch` a `λ_deg`
   throughput term (so the teacher is degradation-aware) and make `SDPSolver` use
   the true SOC (or document the C/2 approximation). This changes teacher
   trajectories, so it must precede corpus regeneration.
5. **A7/A8 (obs normalization)** — use fixed/reference normalizers rather than
   `battery_life_cost`- or initial-capacity-anchored ones, so H4.5 cost sweeps and
   long-horizon runs are not confounded.

### Validation gates

- Unit tests with hand-computed expected wear for representative cycles (A3), a
  reset-path equality test (A4), an efficiency round-trip test (A2), and a
  calendar-aging monotonicity test (A1).
- A "delta report" on the fixed 10-window real-OOD surface and one synthetic
  surface: old vs new physics, same policies, before any retraining.
- Only after the delta is understood: regenerate the SDP-teacher corpora, retrain
  the household DT, and re-issue H2/H4 numbers under the new protocol.
- Update `report.md` §8.1.x and this file; supersede the prior household
  headlines explicitly.

### Recommendation

Treat this as a scoped project (branch + plan) rather than a hotfix, because the
blast radius is the entire household track. If the current results are needed
as-is for a deadline, keep the present physics and cite A1–A8 as documented
limitations instead.

## Resolved

- **B1** (impact-aware `J_t(soc)` dispatch sign) — fixed 2026-09-13; full suite
  green (378 passed).
- **B2** (Oracle vs env FCAS ordering) — fixed 2026-09-13.
- **B3** (`aggregate_fcas_market_depth` undefined) — fixed 2026-09-13.
- **B4** (env vs planner degradation model) — documented 2026-09-13.
- **B6** (checkpoint architecture not embedded) — documented 2026-09-13.
- **B8** (`phase3_impact_eval.py` ignored checkpoint `return_scale`) — fixed and
  verified 2026-09-13; this reverses the reported j_t_soc impact collapse.
- **A3** (rainflow C-rate units) — fixed 2026-09-13 globally; household/AEMO
  degradation drops ~28%/~38% on controlled cycles. Re-baseline pending.
- **A4** (`EnergySimEnv.reset` dropped `max_c_rate`) — fixed 2026-09-13.
- **A5** (planner degradation was state-independent and always zero) — fixed
  2026-09-13; state-dependent step estimator + precomputed grid. Re-baseline
  pending.
- **A6** (household teacher degradation-blind) — fixed 2026-09-13; `λ_deg`
  parameter, generator default 50 $/MWh. Re-baseline pending.
- **A1** (household `full` had no calendar aging) — fixed 2026-09-13; calendar
  term added in `full` mode. Re-baseline pending.
- **A2** (env had no round-trip efficiency) — fixed 2026-09-13; `roundtrip_eff`
  default 0.80 + efficiency-aware projector. Re-baseline pending.

### Fixes applied on 2026-09-13 (verified: 378 tests pass)

- `src/aemo_sdp_executor.py` — `compute_cost_to_go_table` now passes
  `+energy/step_duration` to the impact model, matching the env's
  positive=charging convention (was negated).
- `src/market_impact.py` — corrected the dispatch-sign comments/docstring.
- `src/aemo_data.py` — added `aggregate_fcas_market_depth` (demand-heuristic
  FCAS depth proxy).
- `tests/test_market_impact.py` — new: impact-sign monotonicity, identity
  price-taking, depth schema/values, and a cost-to-go dispatch-sign regression.
- `src/aemo_oracle_algo.py` / `src/decision.py` — centralised the Oracle→env
  FCAS bid remap in `oracle_fcas_bids_to_env_order` (B2).
- `src/aemo_sdp_executor.py` — documented the planner-vs-env degradation-model
  mismatch in the `sdp_energy_dispatch` docstring (B4).
- `src/transformer_training.py` — noted the `state_dict`-only checkpoint
  contract at the save site (B6).
- `tests/test_aemo_fcas_order.py` — new: env FCAS order + Oracle remap tests.
- `scripts/phase3_impact_eval.py` — apply the checkpoint `.meta.json`
  `return_scale` (fixes B8).
- `scripts/verify_jtsoc_impact_sign.py` + `configs/aemo_decision_transformer_model_kwargs_sdp_jtsoc_fullcorpus.json`
  — new focused harness proving explicit j_t_soc is impact-viable when
  `return_scale` is correct.
- `src/batterydeg.py` / `src/EnergySimEnv.py` — A3/A4 C-rate units + reset cap.
- `src/algorithm_helpers.py` — `compute_step_degradation` (the old 3-point
  rainflow path always returned 0); `src/sdp_algorithm.py` state-dependent
  `_deg_cost_grid`; `src/oracle_algorithm.py` updated to match (A5).
- `src/household_optimization.py` + `scripts/generate_household_sdp_trajectories.py`
  — `deg_cost_per_mwh` (A6); `tests/test_physics_v2_planner.py` — new.
- `src/EnergySimEnv.py` — `roundtrip_eff` (A2) + calendar aging in `full` (A1);
  `scripts/evaluate_household_ood_baselines.py` projector efficiency-aware;
  `tests/test_environment_efficiency.py` + `tests/test_environment_calendar.py` — new.

**Still to do:** re-run the impact gate with explicit `j_t_soc` (B1 follow-up)
to confirm H1 is now consistent with the env.
