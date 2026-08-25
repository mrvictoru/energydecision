# AEMO Research Plan (Consolidated) — **STALE**

> **STATUS (2026-08-25):** This file is **STALE** — it was the living plan during the 2026-07/08 research sessions and its endpoint was the Option C decision (2026-08-11). The durable findings live in `report.md §8.2.1a`, `report.md §8.2.10`, `docs/aemo_dt_preferred_policy_plan.md`, and the new forward plan at **`docs/FUTURE_PLAN.md`** (the single source of truth for forward work).

This file is retained for historical context only. **Do not update this file**; update `docs/FUTURE_PLAN.md` instead.

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

# Re-examining assumptions + progress-measurement (2026-08-07)

This workstream is a **critical re-examination of what we assumed true**, not
just a benchmark revision. The earlier narrative ("the offline DT is making
progress and is the best") turned out to be **multi-dimensional**, not
unidirectional. What we re-examined and found:

| Assumption we held | Re-examined finding |
|---|---|
| "The modern v2 DT is SOTA / beats PPO" | **Surface-specific.** It wins on narrow/mild surfaces (Oct 2024, dispatch-matched, example) but loses broadly to PPO on the full-2024 5-min surface (DT $4.6k vs PPO $15k/ep), driven by **FCAS under-bidding** in spike months (PPO $10.2k FCAS vs DT $4.8k). Under market impact the *impact-trained* DT wins. "Best" is multi-dimensional. |
| "Eval configs are protocol-consistent" | **False.** Several configs used 30-min steps vs the 5-min training data, nearly halving the DT. Fixed all AEMO eval configs + the env default to 5-min. |
| "RTG prompting is calibrated" | **Out-of-distribution + constant.** Training RTGs are tiny (p50≈0, p90≈1.5, max ~9–13) and *decaying* over episodes; eval uses fixed rtg 0–50 (a strategy selector far above training) fed as a *constant* each step. |
| "Train the DT on PPO data → it bids FCAS like PPO" | **Not validated.** A PPO-only DT (8×384) beats PPO on total profit ($17.6k vs $15k) but is energy-driven ($17k energy, $2.1k FCAS) — it does not learn PPO's FCAS bidding. |

Two benchmark surfaces (below) make this measurable going forward.

## Next stage: actually building the best DT in the AEMO sim

> **Profit-comparability correction (2026-08-10):** revenue-decomposition
> tables (profit / FCAS / energy) are not directly "who wins" — a model with
> lower FCAS can still win on total profit. On the 5-min expanded surface the
> **PPO-only DTs beat SB3 PPO on total profit** ($17.6–17.8k vs $15.0k),
> compensating for lower FCAS ($2.1k vs $10.2k) with much higher energy
> arbitrage ($17k vs $8.8k). So "the DT under-bids FCAS" describes one revenue
> dimension, not overall underperformance — total profit is the headline.

The session's experiments (RTG sweep, FCAS-weighted loss, PPO-only data,
GRPO fine-tune, architecture change) all landed on the same conclusion: the
DT's behaviour is **bounded by its offline data**, and none of the
data-side/objective/online levers tried moved the broad-surface profile. Three
options were documented; **Option C is the decision (2026-08-11)**:

- **Option A — FCAS-focused data (tested, limited headroom).** The
  FCAS-heavy-policy subset (real A2C/TD3/SAC/DDPG eps) raised FCAS capture
  +23% ($4.8k → $5.9k) but still trails PPO 1.7×; synthetic-FCAS generation
  cancelled (PR #34).
- **Option B — full-PPO (value-critic) fine-tune (tested, no improvement).**
  Implemented `--use-critic`; a full-2024 fine-tune evaluated on
  out-of-distribution 2025 earns $6.8k FCAS but **−$0.7k total profit** (far
  below PPO's $14.3k). Consistent with GRPO being flat.
- **✅ Option C — accept PPO as the broad-year/out-of-distribution leader.**
  The DT is positioned for the surfaces where it genuinely wins.

### The DT's home surfaces (under Option C)

The DT should be presented/developed for these surfaces, not as a broad-year
leader:

1. **Market-impact (grid-scale)** — impact-DT beats PPO 9/9 cells (+$115K/cell, p=0.004).
2. **Dispatch-matched** — modern v2 $10,138/ep vs PPO $7,757.
3. **Standard (Oct 2024)** — modern v2 $4,630 vs PPO $2,353.
4. **Mild-market months / FCAS capture** — the DT wins mild months (e.g. Jan)
   and earns strong FCAS revenue with low degradation.

Cross-cutting: **always-on correct protocol** — evaluate every candidate on
the 5-min broad surface (regime-shift benchmark) and the 2025
out-of-distribution surface before any "best" claim, so "best" is measured,
not assumed.

## Progress-measurement benchmarks

Two repeatable benchmark surfaces measure progress and relevance. Both
answer "which model is best, and does the answer hold up?" — the key gap the
earlier "is this proof?" review identified. **Runtime note:** the evaluator now
batches the DT candidate rollout (one transformer forward per step across
episodes) and defaults to thread-parallel (`parallelize_candidate_dt`), so a
full expanded eval (135 DT episodes, 5-min) runs in ~30–40 min on the 22 GB
GPU instead of many hours.

## Impact-standard benchmark (market-impact env)

Config-driven (`--impact-config configs/impact_benchmark.json` on
`scripts/phase3_impact_eval.py`), grid-scale batteries, {identity,
piecewise_merit_order}. Canonical leaderboard:
`eval_output/final/impact_benchmark/impact_benchmark_leaderboard.csv`
(aggregated from the validated 255-cell Phase 3/4 results):

| Comparison (piecewise merit order, 9 cells, best RTG) | Result |
|---|:---:|
| impact-DT vs naive v2 | 6/9 wins, +$96K/cell (p=0.164, not significant) |
| impact-DT vs PPO | 9/9 wins, **+$115K/cell (p=0.004, significant)** |
| naive v2 vs PPO | 8/9 wins, +$19K/cell |

Under the realistic market-impact assumption, the impact-trained DT is the
leading model at grid scale. Oracle_MI is the impact-aware ceiling.

## Regime-shift / expanded-surface benchmark

The modern v2 DT (trained on 2021–2023) evaluated on the **full 2024 expanded
surface** (5 regions × Jan/Mar/May/Jul/Sep/Nov, 288h episodes, medium
10MWh/5MW battery): `eval_output/final/regime_shift/summary.json`.

| Policy | Profit/ep | Note |
|---|:---:|---|
| PPO (online RL) | **$15,017** | Wins 5/6 periods; 2.9–10× the DT in FCAS-spike months |
| Modern v2 DT (rtg=10) | $4,596 | Wins Jan, close in Jul; loses in Mar/May/Sep/Nov |
| FCAS rule / rule | −$35.4k / −$20.1k | Baselines |

> **Protocol correction (2026-08-07):** the earlier expanded run used **30-min
> steps** (off-protocol for the 5-min-trained DT), which nearly halved the DT
> ($2,421). All AEMO eval configs + the env default are now **5-min**
> (`step_duration=0.083333`). At the correct protocol the DT more than doubles
> ($4,596) but **PPO improves too** ($15,017), so the broad-surface gap is real:
> **PPO wins ~3.3×**, driven by the DT under-capturing FCAS in spike months
> (PPO FCAS $10.2k vs DT $4.8k). RTG sweep (0–50) is flat, so this is not a
> prompting effect.

**Critical finding:** the DT's SOTA status is **surface-specific**. It wins on
the narrow example / dispatch-matched / Oct-standard surfaces and mild months
(Jan), but on the broad full-year surface PPO dominates — the DT under-bids
FCAS during large spike events. Even on the DT's own training battery
(medium_1c 10/10, 5-min, Sep–Nov) PPO edges it ($4,520 vs $3,453). The choice
of evaluation surface materially changes who "wins".

## Root cause of the DT's underperformance: FCAS under-capture

The broad-surface gap is **not a prompting artifact** (RTG sweep 0–50 is flat)
and **not a seasonal data bias** (training spans all seasons 2021–2023). The
driving cause is **the DT under-bids FCAS during large spike events**: at 5-min
on the expanded surface PPO earns $10.2k FCAS/ep vs the DT's $4.8k (2.1×).
PPO wins every FCAS-spike month decisively (Nov $35.5k vs $3.5k; May $25k vs
$8.6k; Sep $16k vs $2.8k) while the DT wins only mild months (Jan $3.8k vs
$2.0k). This is the same learned **conservative FCAS moderation** documented in
the market-impact work (the impact-DT trades 0.39–0.53× grid-scale FCAS) — the
offline behaviour-cloning objective averages over the dataset's
conservative+aggressive mixture and lands on under-bidding FCAS.

**Direction (2026-08-07): PPO-informed DT training, not GRPO.** GRPO does not
improve the modern v2. The hypothesis is that the DT should be trained to bid
FCAS like PPO by:
1. **Reward/return-weighted training** (upsample or weight high-return and
   high-FCAS-revenue trajectories, e.g. PPO's best rollouts) — a
   reward-weighted-regression treatment of the behaviour-cloning loss, and
2. **Adding more aggressive FCAS behaviour to the offline mixture** (generate
   additional PPO FCAS-heavy rollouts and retrain).
Then re-measure on the 5-min expanded surface to confirm the FCAS gap closes.
A dynamic evaluation dashboard (power/energy/revenue visualisation) is a
deferred side-goal for making these surfaces more intuitive.

### PPO-only DT experiment (2026-08-07) — energy-heavy, does NOT close FCAS

Retrained the 8×384 FCAS-rich DT on the **PPO-only subset** of the v2 dataset
(900 eps, 28.3M rows) and evaluated on the 5-min expanded surface:

| Model | Profit/ep | FCAS/ep | Energy/ep |
|---|:---:|:---:|:---:|
| PPO-only DT (8×384) | **$17,606** | $2,140 | **$17,099** |
| PPO (reference) | $15,017 | $10,204 | $8,766 |
| Modern v2 (mixed data, 8×768) | $4,596 | $4,774 | $281 |

**Reading:** the PPO-only DT *beats PPO on total profit* ($17.6k vs $15k) and
triples the mixed-DT profit — but it is **energy-arbitrage-driven** (17k
energy, only $2.1k FCAS). The "train on PPO to bid FCAS like PPO" hypothesis
is **not validated**: PPO-only data with this recipe teaches aggressive energy
trading, and FCAS capture is *lower* than the mixed model. Caveats: 8×384 vs
8×768 architecture and return_scale/RTG semantics differ, so not
apples-to-apples. The FCAS under-bidding likely needs a targeted treatment
(e.g. FCAS-weighted action loss, or upsampling the *mixed* set rather than
PPO-only, or the return-weighted loss — next on the list) rather than
PPO-only data alone.

### FCAS-weighted loss result (2026-08-07) — no effect; data is the ceiling

Retrained the 8×384 DT on the PPO-only v2 subset **with the FCAS-weighted
action loss** (`--action-dim-weights 1,3,3,3,3,3,3,3,3`):

| Model | Profit/ep | FCAS/ep | Energy/ep |
|---|:---:|:---:|:---:|
| PPO-only DT (baseline) | $17,606 | $2,140 | $17,099 |
| PPO-only DT + FCAS-weighted | $17,671 | $2,145 | $17,170 |

**FCAS capture did not move.** Root cause found by measuring the v2 dataset's
episode composition: **the v2 PPO episodes are energy-dominant, not
FCAS-dominant** (mean FCAS/energy ≈ 1.0, the *lowest* ratio of any source
policy; A2C is 49×, DDPG 16×, SAC 3.5×). So there is no higher-FCAS behavior
in the PPO-only data to amplify — behaviour cloning (with or without loss
weighting) cannot exceed what the data contains.

**General conclusion:** the DT-vs-PPO gap on the broad surface is a
**behaviour-cloning ceiling**, not a mixture or objective-tuning artifact:
- The v2 PPO episodes captured PPO's behaviour on *energy-dominant* training
  periods/batteries and do not reflect PPO's online-optimised FCAS skill at
  eval time (PPO adapts online).
- The mixed-data modern v2 under-earns absolute FCAS ($4.8k vs PPO $10.2k)
  *and* does essentially no energy arbitrage ($281 vs $8.8k) — it cloned the
  moderate-FCAS policies.
- To make the DT bid FCAS like PPO, the lever is **better offline data**
  (higher-FCAS trajectories, e.g. FCAS-focused reward shaping during
  generation, or capturing FCAS-spike periods), since neither data re-weighting
  nor the loss-weighting we tried can exceed the data, and GRPO (online
  fine-tuning) does not improve the modern v2.

### Online-RL (GRPO) fine-tune of the PPO-only DT (2026-08-07) — also flat

Fine-tuned the PPO-only DT (8×384, legacy-style arch, from-scratch) with the
repo's GRPO post-training (NSW1 Jan 2024, 144h, medium battery, 5 iters) and
evaluated on the 5-min expanded surface:

| Model | Profit/ep | FCAS/ep | Energy/ep |
|---|:---:|:---:|:---:|
| PPO-only DT (pre-GRPO) | $17,606 | $2,140 | $17,099 |
| PPO-only DT + GRPO | $17,212 | $1,845 | $16,860 |
| Modern v2 (mixed) | $4,596 | $4,774 | $281 |
| PPO reference | $15,017 | $10,204 | $8,766 |

**Online-RL fine-tuning did not help** — GRPO slightly *reduced* profit
(−$394), FCAS (−$295) and energy (−$239), matching the modern-v2 finding.
Note the PPO-only DT is the **legacy-style architecture** (8×384, no
GQA/qk_norm/tie_weights), so this does not test the modern architecture with
PPO-only data. Combined with the flat RTG sweep and the FCAS-weighted loss
no-op, the evidence is that neither data-side nor online-fine-tuning levers
move the DT's broad-surface performance; the remaining lever is better offline
data, or a full PPO (value-critic) fine-tune distinct from GRPO.

### Modern v2 (8×768) on PPO-only data (2026-08-10) — architecture does not matter

Retrained the **modern v2 architecture** (8×768 GQA, ctx=210) from scratch on
the same 900 PPO-only v2 episodes, evaluated on the 5-min expanded surface:

| Model | Profit/ep | FCAS/ep | Energy/ep |
|---|:---:|:---:|:---:|
| PPO-only DT (legacy 8×384) | $17,606 | $2,140 | $17,099 |
| **Modern v2 on PPO-only (8×768)** | $17,775 | $2,133 | $17,375 |
| Modern v2 (mixed data) | $4,596 | $4,774 | $281 |
| PPO reference | $15,017 | $10,204 | $8,766 |

**The architecture makes essentially no difference on PPO-only data.** Both
the legacy 8×384 and modern 8×768 clone the same energy-heavy profile
(FCAS ≈ $2.1k, energy ≈ $17k). **The data is the determinant, not the
architecture**: PPO's v2 episodes are energy-dominant, so no architecture can
produce FCAS bidding that the data does not contain. The modern arch only pays
off on the *mixed* data (where it extracts $4.8k FCAS vs the legacy's lower
capture). This closes the architecture-isolation question: to make the DT bid
FCAS like PPO, the offline data must contain higher-FCAS trajectories.

### FCAS-heavy-policy subset (2026-08-11) — partial validation, limited headroom

Retrained the modern v2 on the **real** FCAS-heavy policies already in the v2
corpus (A2C/TD3/SAC/DDPG, 1,200 eps — the policies with 3–49× the FCAS/energy
ratio of PPO):

| Model | Profit/ep | FCAS/ep | Energy/ep |
|---|:---:|:---:|:---:|
| Modern v2 on FCAS-heavy subset | $13,387 | **$5,860** | $12,578 |
| Modern v2 (mixed data) | $4,596 | $4,774 | $281 |
| Modern v2 on PPO-only | $17,775 | $2,133 | $17,375 |
| PPO reference | $15,017 | $10,204 | $8,766 |

**Reading:** the real-data composition lever works but has **limited headroom**.
Training on the FCAS-heavy policies raised the DT's FCAS capture **+23%**
($4.8k → $5.9k) over the mixed model — validating "more high-FCAS real
trajectories → higher DT FCAS" — but the DT still under-bids FCAS **1.7× vs
PPO** ($5.9k vs $10.2k). The FCAS-heavy subset also produced substantial energy
arbitrage ($12.6k), since those policies' episodes are not purely FCAS.
**Synthetic-FCAS data generation is cancelled** (PR #34 showed the synthetic
generator does not accurately reflect real FCAS), so the remaining gap is a
fundamental offline-data ceiling: the real corpus's FCAS bidding is below PPO's
online-optimised skill.

## RTG-distribution finding (prompting study, 2026-08-07)

The DT is prompted with a **fixed** RTG per eval, but RTG (returns-to-go) is
inherently dynamic — it decays over an episode as rewards accrue. Measured the
actual RTG distribution in the FCAS-rich training data (γ=0.95, sampled
episodes per policy):

- **Per-step RTG overall: p50 = −0.1, p90 = 0.3, max = 539** — overwhelmingly
  small (≈0); the DT mostly trains on near-zero RTG tokens.
- **Initial RTG (episode total discounted return): p50 ≈ 0.0–0.3, p90 ≈
  1.2–1.8, max ≈ 9–13** across all source policies (PPO similar to others).

Two consequences:
1. **Train/inference RTG mismatch.** At training the RTG token is the *decaying*
   realized return-to-go; at inference the evaluator feeds the *same constant*
   (config `rtg_value` = 0/10/20/50) at every step. The high eval RTGs
   (5–500× the training p50) act as a **strategy selector** (more aggressive
   bidding), which is why higher RTG sometimes beats lower — but they are far
   out of the training RTG distribution, so the DT is extrapolating.
2. **Dynamic-prompting hypothesis.** Prompting with a *decaying* RTG (start at
   the desired return and discount per step to match training semantics) may
   both stay in-distribution and preserve the strategy-selection effect.

Test to run: compare fixed vs decaying RTG prompts on the 5-min expanded
surface for the modern v2 DT (and check `return_scale` interaction).

# Open Items & README Roadmap Sync

Status synced with the root `README.md` "Roadmap" checklist (2026-08-07):

| README item | Status |
|---|:---:|
| Offline dataset studies (DT sensitivity to policy mixtures / curation) | ⬜ Open — highest priority, no hardware |
| **Regime-shift / full-scale robustness evaluation** (train 2022–23 → eval 2024; expanded surface + paired significance on the SOTA headline) | 🟡 **Measured (2026-08-07)** — expanded sweep ran: DT $2.4k/ep vs PPO $11.3k/ep (~4.7×); RTG-insensitive. Paired significance on the headline still open |
| Long-context DT experiments (larger `context_len`, RoPE, seasonal/weekly) | ⬜ Open (ctx=2016 now feasible on the 22 GB GPU) |
| Multi-agent extension (microgrid, multiple households) | ⬜ Open |
| Artifact provenance (checksums/config logging) | ⬜ Open |
| Sim-to-real readiness (safety wrappers, hardware-in-the-loop) | ⬜ Open — **lowest priority, requires hardware** |
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

> **Session 2026-08-07→08-10 (DT re-examination + progress benchmarks):**
> full diary + handoff in
> [`session_2026-08_aemo_dt_reexamination.md`](session_2026-08_aemo_dt_reexamination.md)
> (branch `feature/eval-progress-benchmarks`, PR #35).

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
