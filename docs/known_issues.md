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
- **Suggested fix:** either call `degradation_per_timestep` in `SolarBatteryEnv.step`
  for `"full"`, or relabel the mode/arms and correct the CLI help. Status: `OPEN`.

### A2. No round-trip efficiency in the env, but the SDP teacher assumes `sqrt(0.80)`

- **Location:** `src/EnergySimEnv.py:399` (`new_level = battery_level + battery_flow_energy`);
  `src/household_optimization.py:60,89,98` (`roundtrip_eff=0.80`, `eff = roundtrip_eff ** 0.5`).
- **Detail:** the env is lossless; the teacher plans with symmetric charge/discharge
  losses. Teacher SoC paths and realized env SoC therefore diverge.
- **Impact:** distillation labels/RTG are computed under dynamics the env does not
  have; any "honest teacher" claim must state which efficiency convention is used.
- **Suggested fix:** add configurable symmetric efficiency to `SolarBatteryEnv`
  (default 1.0) or set teacher `roundtrip_eff=1.0` to match the env, and document
  the choice. Status: `OPEN`.

### A3. Rainflow inferred C-rate is ~100× too large and saturates the clamp

- **Location:** `src/batterydeg.py:359-373` (current computed as `ΔSoC% / hours`,
  with no `/100` conversion to C-rate), `batterydeg.py:7-12`
  (`Id_nom=0.25`, `Ich_nom=0.125` are true C-rates).
- **Impact:** the C-rate multipliers (`nCL_Id`, `nCL_Ich`) are effectively pinned
  near the `max_c_rate` clamp for realistic cycles, weakening the rate dependence.
- **Suggested fix:** divide by 100 (or carry SOC as a fraction), then re-validate
  the nominal multipliers and any reported degradation figures. Status: `OPEN`.

### A4. `EnergySimEnv.reset()` drops the rainflow C-rate cap

- **Location:** `src/EnergySimEnv.py:143` constructs `RainflowCounter(...,
  max_c_rate=max_battery_flow/initial_capacity)`; `reset()` at
  `src/EnergySimEnv.py:303` reconstructs it **without** `max_c_rate`, reverting to
  the default `1.0`.
- **Note:** `AEMOBatteryEnv` does **not** have this bug — its reset passes
  `max_c_rate` (`src/AEMOBatteryEnv.py:501/508/634`).
- **Impact:** household degradation after `reset()` uses a different clamp than the
  constructor implies; results depend on init/reset path.
- **Suggested fix:** pass `max_c_rate` in `EnergySimEnv.reset()`. Status: `OPEN`.

### A5. `SDPSolver` prices degradation at a fixed representative SoC

- **Location:** `src/sdp_algorithm.py:316,325-329` (`rep_soc = capacity/2`).
- **Impact:** the DP's degradation cost is state-independent, so it cannot trade
  off wear against SOC position. `OracleSolver` uses the true state
  (`src/oracle_algorithm.py`), so SDP and Oracle are not directly comparable on
  degradation.
- **Suggested fix:** use the true SOC in the stage cost or document the
  approximation explicitly. Status: `INTENTIONAL` / needs documenting.

### A6. Household SDP teacher is degradation-blind; `J_t(soc)` excludes wear

- **Location:** `src/household_optimization.py:54-141` (`optimize_dispatch` has no
  degradation term); `scripts/generate_household_sdp_trajectories.py:105-107`
  (`rtg_value = -J_t(soc)`, grid-cost only).
- **Impact:** household distillation labels and RTG do not encode degradation; wear
  reaches the DT only through realized reward and capacity fade. `report.md` §4.3
  describes the AEMO teacher's `λ_deg` and can mislead readers about the household
  teacher.
- **Suggested fix:** add a `λ_deg` term to `optimize_dispatch`, or explicitly
  document the household teacher as degradation-blind. Status: `OPEN`.

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
  in `tests/test_market_impact.py`. Re-running the impact gate with explicit
  `j_t_soc` remains recommended before relying on H1. **Full test-suite
  verification pending.**

### B2. FCAS service ordering differs between the env and the Oracle

- **Location:** env `_fcas_services` (`src/AEMOBatteryEnv.py:489`) interleaves
  raise/lower; the Oracle groups raise then lower (`src/aemo_oracle_algo.py:55`).
- **Detail:** `src/decision.py` remaps the Oracle output, but any direct consumer
  of the oracle bid arrays must apply the same mapping.
- **Suggested fix:** centralise the ordering in one constant and add a test.
  Status: `WORKAROUND`.

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
- **Suggested fix:** use one model in both, or document the mismatch and its
  direction. Status: `INTENTIONAL` / needs documenting.

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
- **Suggested fix:** docs are corrected in `report.md` §4.2; consider writing the
  architecture into future checkpoints. Status: `OPEN` (docs) / `WORKAROUND`.

### B7. The env step reward is scaled by 1/1000

- **Location:** `src/AEMOBatteryEnv.py:1033` (`normalized_reward = reward/1000`).
- **Impact:** raw dollars live only in `info`; a common source of confusion when
  comparing logged rewards to profits. Documented but easy to miss.

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

## Resolved

- **B1** (impact-aware `J_t(soc)` dispatch sign) — fixed 2026-09-13; full-suite
  verification pending.
- **B3** (`aggregate_fcas_market_depth` undefined) — fixed 2026-09-13.

### Fixes applied on 2026-09-13 (pending full-suite verification)

- `src/aemo_sdp_executor.py` — `compute_cost_to_go_table` now passes
  `+energy/step_duration` to the impact model, matching the env's
  positive=charging convention (was negated).
- `src/market_impact.py` — corrected the dispatch-sign comments/docstring.
- `src/aemo_data.py` — added `aggregate_fcas_market_depth` (demand-heuristic
  FCAS depth proxy).
- `tests/test_market_impact.py` — new: impact-sign monotonicity, identity
  price-taking, depth schema/values, and a cost-to-go dispatch-sign regression.

**Still to do:** run `python3 -m pytest tests/ -v` inside `energydecision-gpu`
and re-run the impact gate with explicit `j_t_soc`.
