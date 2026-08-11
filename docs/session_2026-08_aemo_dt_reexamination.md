# Session diary: AEMO DT re-examination + progress benchmarks (2026-08-07 → 08-10)

Research diary and handoff for the work on `feature/eval-progress-benchmarks`
(PR #35). This is a companion to `docs/aemo_research_plan.md` (the consolidated
plan) — read that for the durable findings; this file records what was done,
what was learned, and the explicit handoff for the next agent session.

## Session goal

The stated goal shifted from "evaluate progress / revise benchmarks" to a
**critical re-examination of assumptions**: we believed "the offline DT is
making progress and is the best," but the evidence shows it is
**multi-dimensional**. The session built the measurement tooling to make that
visible and explored how to actually build the best DT in the AEMO sim before
moving to unchecked README items.

## What was done (chronological)

1. **Impact-standard benchmark** — `phase3_impact_eval.py` made config-driven
   (`--impact-config configs/impact_benchmark.json`); canonical leaderboard
   aggregated from the validated 255-cell Phase 3/4 results
   (`eval_output/final/impact_benchmark/`). Under market impact the
   impact-trained DT beats PPO 9/9 (+$115K/cell, p=0.004) and edges naive v2
   6/9 (+$96K, n.s.).
2. **Regime-shift / expanded-surface benchmark** — evaluated the modern v2 DT
   (trained 2021–23) on the full 2024 surface (5 regions × 6 periods).
3. **Protocol bug found + fixed** — several eval configs (example, expanded,
   dispatch_matched, ppo_compare) used **30-min steps** vs the 5-min training
   data, nearly halving the DT. Fixed all configs + the env default
   (`AEMOBatteryTradingEnv`, `AEMODataPreprocessor` → `step_duration=0.083333`).
4. **RTG-distribution study** — training returns-to-go are tiny (p50 ≈ 0,
   p90 ≈ 1.5, max ~9–13) and *decaying*; eval feeds constant rtg 0–50 (a
   strategy selector far out of distribution).
5. **PPO-only experiment** — 8×384 DT from scratch on the 900 PPO-only v2
   episodes: **energy-heavy** ($17.6k profit, $2.1k FCAS). Root cause: the v2
   **PPO episodes are the most energy-dominant** of all source policies
   (FCAS/energy ≈ 1.0; A2C is 49×, DDPG 16×).
6. **FCAS-weighted loss** — `--action-dim-weights` added; no effect (FCAS
   2,140 → 2,145). Data contains no higher-FCAS behaviour to amplify.
7. **Online-RL (GRPO) fine-tune** of the PPO-only DT — slightly negative
   (profit −$394, FCAS −$295). Consistent with GRPO not helping the modern v2.
8. **Architecture isolation** — modern v2 (8×768) on the SAME PPO-only data:
   nearly identical energy-heavy profile (FCAS $2.1k, energy $17k). **The data
   is the determinant, not the architecture.**
9. **Eval optimization** — batched the DT candidate rollout (one transformer
   forward per step across episodes, previously batch-1) + `parallelize_candidate_dt`
   now defaults True (8 workers). The 22 GB GPU now runs near-saturated.
10. **FCAS-heavy-policy subset (2026-08-11)** — modern v2 retrained on the real
    A2C/TD3/SAC/DDPG episodes (1,200 eps; the policies with 3–49× the
    FCAS/energy ratio of PPO). Result on the 5-min expanded surface: **FCAS
    capture +23%** ($4,774 → $5,860), profit $13,387, energy $12,578 — but
    still **1.7× below PPO's FCAS** ($10,204). The real-data composition lever
    works but has limited headroom. (Training hit a transient CUDA teardown
    error after the successful save — the checkpoint is valid.)
11. **FCAS-spike-period detector** — `scripts/find_fcas_spike_periods.py`
    ranks 12-day windows by FCAS intensity: training-era **SA1 Nov 2022**
    ($15,500 caps, no eval leakage), 2024 QLD1 May ($15,718) / SA1 Feb
    ($16,600 cap) / NSW1 Aug.
12. **Synthetic-FCAS generation cancelled** — per PR #34 the synthetic FCAS
    generator does not accurately reflect real FCAS, so synthetic-augmented
    training would be garbage-in.

## Key findings (durable)

- **The DT's "SOTA" is surface-specific.** Wins narrow/mild surfaces (Oct
  2024, dispatch-matched, example, impact) but loses broadly to PPO on the
  full-2024 5-min surface **on total profit**? No — see the correction below.
- **Profit-comparability correction (important):** revenue-decomposition
  tables (profit/FCAS/energy) are not "who wins." On the expanded surface the
  **PPO-only DTs beat SB3 PPO on total profit** ($17.6–17.8k vs $15.0k),
  despite much lower FCAS, via higher energy arbitrage. The "FCAS
  under-bidding" describes one dimension, not overall underperformance.
- **Behaviour-cloning ceiling.** None of RTG (flat sweep), FCAS-weighted loss
  (no-op), data re-weighting (energy-heavy), online fine-tuning (GRPO flat),
  or architecture (no change) moved the broad-surface profile. The DT is
  bounded by its offline data; PPO's v2 episodes do not contain PPO's
  online-optimised FCAS skill (PPO adapts online during eval).
- **Real-data composition has limited headroom.** The FCAS-heavy-policy subset
  raised the DT's FCAS capture +23% ($4.8k → $5.9k) but still trails PPO 1.7×.
  Combined with the above, the DT-vs-PPO FCAS gap is a **fundamental
  offline-data ceiling** (and synthetic generation is off the table).
- **Eval protocol matters enormously.** 30-min steps nearly halved the DT. All
  AEMO DT evaluation must use 5-min steps.

## Benchmarks now available (repeatable)

- `configs/impact_benchmark.json` + `phase3_impact_eval.py --impact-config`
  → impact-standard leaderboard.
- `configs/aemo_autoresearch_evaluator.expanded{,_rtg5/10/20/50}.json` (all
  now 5-min) → regime-shift / broad-surface benchmark.
- RTG sweep configs; `--action-dim-weights` in `pretrain_decision_transformer.py`;
  `--checkpoint` in `run_grpo_posttraining.py`; batched DT rollout in the evaluator.

## Three next-stage options (decision point for the next session)

- **Option A — FCAS-focused offline data generation.** Generate higher-FCAS
  trajectories (SB3 reward shaping weighted toward FCAS, or generate during
  known FCAS-spike periods such as 2024 May/Sep/Nov), append to the v2 corpus,
  retrain the modern v2. The only lever that can raise the data's FCAS ceiling.
- **Option B — full PPO (value-critic) online fine-tune.** Use the DT's
  return-prediction head as the critic with the clipped PPO objective, distinct
  from GRPO (group-relative, no critic). Uncertain (GRPO was flat) but untried.
  **Status (2026-08-11): implemented** (`--use-critic` in the GRPO trainer —
  advantage = discounted returns-to-go minus the DT's return-head value) and
  ran a full-2024 fine-tune of the modern v2 (NSW1, 144h, 5 iters). On the
  fine-tune surface the reward regressed (−14.5 → −17.5, like GRPO). **2025
  out-of-distribution eval pending** (fetching 2025 data).
- **Option C — accept PPO as the broad-surface leader.** Stop chasing the gap
  with current data; focus the DT on surfaces where it wins (impact,
  dispatch-matched, mild months).

## Handoff notes for the next agent

- Branch: `feature/eval-progress-benchmarks` (PR #35, open). Base: `main`.
  All work committed + pushed; `git log --oneline` on the branch for the full
  list (commits b66f14f → ce780ff).
- Artifacts live under `eval_output/final/{impact_benchmark,regime_shift,ppo_informed}/`.
- If pursuing Option A: the v2 dataset is `data/aemo_dt_fcas_v2/` (77M rows);
  the PPO-only subset is `data/aemo_dt_fcas_ppo_only/` (28M rows); the SB3
  models are `models/aemo_sb3/*_fcas_model.zip`; data generation entrypoint is
  `scripts/generate_fcas_dataset.py` (uses 5-min `step_duration=5/60`).
- **Option A status (2026-08-11):** the FCAS-heavy-policy subset test
  COMPLETED: modern v2 on real A2C/TD3/SAC/DDPG eps → FCAS capture +23%
  ($4.8k → $5.9k), profit $13.4k, but still 1.7× below PPO. **Limited
  headroom** — the real-data composition lever works but does not close the
  gap, and synthetic generation is cancelled (below).
- **Generation sub-step CANCELLED (2026-08-10, per research direction):** do
  NOT generate more synthetic FCAS episodes for training. The previous PR
  (#34, fcas-diffusion-v2) established the synthetic FCAS generator does not
  accurately reflect real FCAS data (tail/magnitude structure not
  transferable), so synthetic-FCAS-augmented training would be garbage-in.
  The FCAS-heavy-policy subset test above uses **real existing episodes**
  (A2C/TD3/SAC/DDPG already in the v2 corpus), which is valid.
- If pursuing Option B: GRPO infra is `src/grpo_posttraining.py` +
  `scripts/run_grpo_posttraining.py`; the DT's return head is the natural
  critic; `--checkpoint` accepts a local model.
- Evaluator runtime: batched DT rollout + thread-parallel are now default; a
  full expanded eval (135 DT episodes, 5-min) is ~30–40 min.
- **Protocol rule:** all AEMO DT evaluation must use `step_duration=0.083333`.
