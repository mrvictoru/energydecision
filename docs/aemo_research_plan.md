# AEMO Research Plan (Consolidated)

One plan covering the AEMO/NEM grid-scale track: market-impact modeling + the
perfect-foresight Oracle, and Decision-Transformer model improvement (GRPO,
retraining, forecast conditioning). Consolidates the former
`docs/market_impact_plan.md` and `docs/dt_improvement_roadmap.md`, and includes
the forecast/future-step integration experiment (PR #32). Status is synced with
the root `README.md` roadmap checklist.

## Overview

Two complementary research threads on the AEMO track:

- **Thread 1 — Market-impact modeling + AEMO Oracle.** Add endogenous
  market-impact to `AEMOBatteryTradingEnv` (a large BESS moves the clearing
  price) and a perfect-foresight LP Oracle as the revenue ceiling. Answers:
  "how much does self-impact cost?" and "does the DT's advantage persist when
  the battery moves the market?"
- **Thread 2 — Decision Transformer improvement.** Whether offline DT training
  can match/exceed online RL on energy-plus-FCAS bidding, and whether GRPO or
  explicit forecast conditioning improve the pretrained policy. Includes the
  modern v2 (8×768) retrain, the impact-aware DT retrain, and the
  forecast-conditioned DT experiment.

## Status summary (2026-08-07)

| Area | Status |
|---|---|
| Thread 1 Phases 0–5 (impact env + Oracle + impact-DT) | ✅ Complete (merged in PR #33) |
| Thread 1 Phase 6 (synthetic FCAS generation) | 🔴 **ON HOLD** (see below) |
| Thread 1 Phase 7 (impact + synthetic combine) | 🔴 **ON HOLD** (contingent on Phase 6) |
| Modern v2 DT (8×768) full retrain | ✅ Done — SOTA ($4,630/ep standard, $10,138/ep dispatch-matched) |
| GRPO post-training pipeline + sweeps | ✅ Done (does not improve the modern v2 pretrained) |
| Impact-aware DT retrain (Phase 4) | ✅ Done (beats PPO significantly; edges v2 6/9 cells) |
| Forecast-conditioned DT (PR #32) | ✅ Done — **negative result** ($4,564/ep vs modern v2 $4,991/ep) |
| Multi-round GRPO self-improvement | ⬜ Open (deferred — GRPO doesn't beat pretrained) |
| Offline dataset studies / longer context / FCAS-weighted loss / multi-agent / sim-to-real / artifact provenance | ⬜ Open (see README checklist below) |

---

# Thread 1 — Market-Impact Modeling & AEMO Oracle

## Design decisions (locked)

| Decision | Choice |
|----------|--------|
| Impact model form | Piecewise-linear merit-order reconstruction |
| Markets | Energy + all 8 FCAS services |
| FCAS depth data source | v1: sum of per-unit cleared enablement from DISPATCHLOAD (free, on disk) |
| Oracle obs compatibility | Config flag: `identity` (obs_dim=18, old checkpoints load) and `expose_impact_state` (superseded — Phase 4 kept obs_dim=18) |

## Phase checklist

### Phase 0 — Data foundation ✅
`fetch_aemo_unit_dispatch` `columns_to_keep` extended (AVAILABILITY, INITIALMW,
enablement max/min, SEMIDISPATCHCAP, ramp rates, AGCSTATUS); aggregate
`fcas_market_depth`, `residual_supply`, `build_supply_curve`; piped through the
preprocessor.

### Phase 1 — AEMO Oracle (rigor) ✅
- `src/aemo_oracle_algo.py`: perfect-foresight LP co-optimizer over energy +
  8 FCAS with SOC/ramp/enablement constraints (`Oracle_PT` price-taking,
  `Oracle_MI` impact-aware). ~0.02s per 288 intervals.
- Registered as evaluator baseline + `AEMOAgent(algorithm='aemo_oracle')`.
- **Invariant (PASS):** Oracle_PT ≥ every replayed policy (DT v2, impact-DT,
  PPO, FCAS rule, dispatch) on 9/9 Phase-3 cells; revenue dominates 6/6
  dispatch-matched episodes. Net-profit ceiling only at **zero degradation**
  (LP is degradation-blind; upgrade path: linear $/MWh-throughput surrogate).
- Dispatch-matched: Oracle nets $238.8K/ep vs DT $23.4K (6-mo mean); per-episode
  Oracle wins 4/6 net (small-asset degradation gap), 6/6 revenue.

### Phase 2 — Market-impact env extension (novel method) ✅
`src/market_impact.py`: `MarketImpactModel` base, `IdentityImpact` (byte-for-byte
default, golden-value tested), `PiecewiseMeritOrderImpact` (energy supply-curve
shift + FCAS depth-proportional price attenuation). Hooked into
`AEMOBatteryTradingEnv._calculate_reward` for energy + all FCAS services.

### Phase 3 — Re-evaluation under impact (no retraining) ✅
3 scenarios (SA1 Oct/Nov, VIC1 Oct) × 3 battery sizes (8 MWh / 150 MW / 250 MW),
identity vs merit-order. Modern v2 DT retains **62/83/49%** of identity profit
at 8/150/250 MW vs PPO's 62/40/32%. Oracle_PT + Oracle_MI ceiling rows added.
Bootstrap CIs + paired Wilcoxon applied to the headline tables.

### Phase 4 — Impact-aware DT retrain ✅
- Dataset: 1,169 episodes / 29.3M rows at true grid-scale asset sizes under
  impact (`data/aemo_dt_impact`, HF `mrvictoru/AEMO_simulated_trade_impact`).
- Retrained via MoLab notebook (`notebooks/molab_notebook_dt_impact.py`) →
  `mrvictoru/energydecision-dt-v2-impact`.
- **Results (243 cells, best RTG, under impact):** impact-DT beats PPO
  significantly (+$115K/cell, p=0.004) and edges the naive v2 6/9 cells
  (+$96K/cell, not significant; CI crosses zero). Biggest wins at grid scale.
- **Dispatch moderation:** the impact-DT learned to moderate dispatch — trades
  0.19–0.33× v2's energy and 0.39–0.53× grid-scale FCAS, confirming the
  conservative posture is a *learned hedge* against self-impact.

### Phase 5 — Documentation & wrap-up ✅
Plan finalized, report.md §8.2.9/§8.2.9.1 updated, README roadmap items
checked, research index updated.

### Phase 6 — Synthetic FCAS data generation 🔴 ON HOLD
- **Done:** eval harness (`src/fcas_generator_eval.py`), v1 HMM/copula generator
  (`src/synthetic_fcas.py`). v1 transfers co-occurrence + spike rates but fails
  tail-value KS (fit-window price caps dominate the empirical tail).
- **v2 (conditional diffusion) concluded as a dead end for the original goal
  (2026-08-07):** the strict tail-KS acceptance gate is provably unreachable by
  any fit-window-trained generator (fit→holdout tail magnitudes drift beyond
  what the conditioning features capture). Fully-synthetic RRP+FCAS episodes do
  not transfer to real-market DT evaluation, but the synthetic **FCAS** channel
  from a broad-trained generator (full-year × SA1/NSW1/QLD1) recovers ~80% of
  real-data FCAS revenue. The blocker is the synthetic **RRP** channel (the
  joint cross-region model dilutes regional energy prices).
- **Circle-back condition:** a separate **per-region RRP generator** (FCAS is
  national, RRP is regional). The FCAS machinery — template-stamped burst
  schedule, schedule-gated tail, broad-generator training — is the reusable
  core. Full write-up: `docs/fcas_diffusion_v2_plan.md` on branch
  `feature/fcas-diffusion-v2` (closed PR #34).

### Phase 7 — Combine impact model + synthetic FCAS 🔴 ON HOLD
Contingent on Phase 6 validating; parked with it.

## Key code hooks

| Hook | File |
|------|------|
| Energy price read | `src/AEMOBatteryEnv.py` `_calculate_reward` |
| FCAS price reads | `src/AEMOBatteryEnv.py` (full_fcas + multi_market loops) |
| Impact model | `src/market_impact.py` |
| Oracle | `src/aemo_oracle_algo.py`, `AEMOAgent(algorithm='aemo_oracle')` |
| Impact eval surface | `scripts/phase3_impact_eval.py` |

---

# Thread 2 — Decision Transformer Improvement

## Current state

- **SOTA: modern v2 pretrained (8×768 GQA)** — $4,630/ep standard, $10,138/ep
  dispatch-matched, no GRPO needed. See report.md §8.
- Training data v2: 2,401 episodes / 77M rows, 4 realistic battery configs
  (1C, 0.7C, 0.5C, 3.75C); source policies PPO, A2C, DDPG, SAC, TD3, FCAS rule.
- **Retrains done:** modern v2 (8×768, `mrvictoru/energydecision-dt-v2`);
  impact-aware DT (Phase 4, `mrvictoru/energydecision-dt-v2-impact`);
  forecast DT (negative, `mrvictoru/energydecision-dt-v2-forecast`).

## GRPO post-training

Phase 1 features (all implemented): degradation-penalty reward shaping,
periodic reference-model sync, adaptive RTG sampling, larger rollout groups.

Key results (see `docs/grpo_experiments.md` for the full log):
- **GRPO does not improve the modern v2 pretrained model.** Phase C (144h
  multi-region) comes within 5–11% of pretrained but never exceeds it; the
  architecture improvements (GQA, RMSNorm, weight tying) captured GRPO's
  benefits.
- Legacy Phase 1 GRPO overfit to dispatch-matched SA1 Q4 2024 ($8,242/ep DM
  collapses to $1,533/ep standard).
- Hyperparameter sweep (21 runs): iterations=5, KL coeff 0.02, entropy 0.0,
  lr=1e-5 (144h), RTG count 4, multi-region best. **24h proxy metrics do not
  predict 144h performance.**
- RTG calibration is architecture-dependent (modern peaks at rtg=0; legacy at
  0.5). Always calibrate per model and account for return_scale.

## Retraining summary

| Model | Dataset | Result |
|-------|---------|--------|
| Modern v2 (8×768) | 2,401-ep realistic-battery | SOTA — $4,630 std / $10,138 DM |
| Impact-aware DT | 1,169-ep impact dataset (MoLab) | Beats PPO (+$115K/cell, p=0.004), edges v2 6/9 |
| Forecast DT | FCAS-rich + SDP + GRPO + TTM tokens | Negative — $4,564/ep vs modern v2 $4,991/ep |

## Standard-tier leaderboard (Oct 2024, 5 regions, 144h, medium_1c)

| Model | Profit/ep | FCAS/ep | Deg/ep | Best RTG |
|---|:---:|:---:|:---:|:---:|
| Modern v2 pretrained | **$4,991** | $4,836 | $229 | 10.0 |
| Dispatch Dalrymple North | $4,660 | $2,287 | $1,020 | — |
| Forecast DT (normalized) | $4,564 | $3,663 | $270 | 50.0 |
| Phase C GRPO (mod v2) | $4,322 | $2,508 | $1,058 | 10.0 |
| PPO reference | $2,353 | $2,192 | $236 | — |

---

# Forecast / Future-Step Integration Experiment (PR #32)

**The experiment the DT roadmap previously tracked in `docs/aemo_hybrid_dt_plan.md`
(now removed — this section supersedes it).**

## Motivation

The SDP paper (Abdulla et al. 2016) achieves optimal battery control via
explicit forecasts, backward induction, and Monte Carlo scenarios. The modern
v2 DT conditions only on a 210-step history window and has **no explicit
forward-looking signal** (report.md §8.2.8). The plan was to inject
planning/forecast awareness directly into the transformer, building on the
implemented SDP/MRDP solvers.

## Design

- **ForecastDecisionTransformer** (`src/forecast_decision_transformer.py`):
  modern v2 backbone with a **48-step TTM forecast token prefix** and RoPE.
- **TTM forecasts:** Granite TTM-R3 fine-tuned on 6 price channels (RRP,
  TOTALDEMAND, 4 FCAS services), 512-context → 48-step forecasts; full-year
  lookup stored in `data/aemo_dt_forecast/ttm_forecasts.npz`.
- **Data:** trajectory datasets enriched with `episode_start` for forecast
  alignment; trained on MoLab → `mrvictoru/energydecision-dt-v2-forecast`.

## Result — NEGATIVE

- **$4,564/ep vs modern v2's $4,991/ep** on the standard surface. Explicit
  point forecasts do **not** beat the implicit context the transformer already
  has.
- **Why:** FCAS prices are spike-driven and essentially unpredictable from price
  history — measured TTM-FCAS correlation is ~0.01–0.07 (this is also what
  killed the synthetic-FCAS-via-TTM idea in Phase 6). The forecast tokens added
  no decision-relevant signal.
- **Distributional-conditioning follow-up (documented, not built):** a
  *probabilistic* forecast (spike-risk features from a calibrated generator)
  could revisit §8.2.8 where point forecasts failed. Contingent on Phase 6 v2
  passing, which it did not — so this remains speculative.

---

# Open Items & README Roadmap Sync

Status synced with the root `README.md` "Roadmap" checklist (2026-08-07):

| README item | Status |
|---|:---:|
| Offline dataset studies (DT sensitivity to policy mixtures / curation) | ⬜ Open |
| Long-context DT experiments (larger `context_len`, RoPE, seasonal/weekly) | ⬜ Open (ctx=2016 now feasible on the 22 GB GPU) |
| Multi-agent extension (microgrid, multiple households) | ⬜ Open |
| Sim-to-real readiness (safety wrappers, hardware-in-the-loop) | ⬜ Open |
| Artifact provenance (checksums/config logging) | ⬜ Open |
| Multi-round GRPO self-improvement (GRPO rollouts → retrain → repeat) | ⬜ Open (deferred — GRPO doesn't beat pretrained) |
| Phase 6/7 synthetic FCAS + impact combine | 🔴 On hold (per-region RRP generator is the circle-back lever) |

Everything else in the README roadmap is checked off (core env, SDP/MRDP, RL,
DT, evaluation, degradation, FCAS-rich data, FCAS-rich DT, GRPO, autoresearch,
forecast DT, market-impact, Oracle, statistical confidence).

## Next-priority shortlist

1. **Offline dataset studies** — ablate source policies (rule vs SDP vs SB3) to
   see which contribute most value.
2. **Longer-context sweep** — ctx=288/576/1008/2016 on the modern v2 8×768 with
   FCAS data (the old sweep used the 8×384 model on FCAS-poor data).
3. **FCAS-weighted loss** — per-dim action loss weighting in
   `transformer_training.py` (all 9 dims currently equal).
4. **Per-region RRP generator** — unlocks Phase 6/7 (the synthetic-FCAS circle
   back).
5. **Multi-agent extension** — `PettingZoo` parallel `SolarBatteryEnv`.

---

# Session diary (condensed)

Detailed per-day records live in the git history (commit messages on
`feature/market-impact-modeling`, `main`, and the research branches). Key
milestones:

- 2026-07-29 — plan created; FCAS data investigation; Phase 6 added; Phase 0
  data foundation.
- 2026-07-29→08-01 — Oracle built + validated (0.1% gap); impact env extension;
  golden-value test.
- 2026-08-02→04 — Phase 3 re-evaluation under impact; Phase 4 impact dataset +
  MoLab retrain launched.
- 2026-08-05 — Phase 4 impact-DT eval (243 cells): beats PPO significantly,
  edges v2; dispatch-moderation confirmed. Phase 4 closed.
- 2026-08-06 — Phase 1 oracle invariant PASS + dispatch-matched sanity check;
  headline CIs; Phase 6 v1 generator evaluated (insufficient → v2 triggered);
  PR split (PR #33 = Phases 0–5 + v1).
- 2026-08-07 — Phase 6 v2 conditional-diffusion experiments (PR #34): tail-KS
  gate proven unreachable, two-stage hybrid built, downstream tests showed
  synthetic FCAS transfers ~80% but fully-synthetic data doesn't → Phase 6/7
  parked. Plan consolidated into this file.
