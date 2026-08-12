# AEMO DT as Preferred Policy — Plan, Checklist & Session Diary

> **STATUS:** IN PROGRESS — plan created, PR #36 open.
> Branch: `feature/dt-preferred-aemo-policy`.
> This file is the living plan + checklist + session diary for the agent session
> that executes this PR. Keep every completed checkbox ticked and append dated
> diary entries at the bottom as work is done.

## 1. Goal

Make the **Decision Transformer the preferred control algorithm** for a battery
unit trading energy + all 8 FCAS services in AEMO, on the **actual objective:
total profit per episode, net of degradation**.

The repo currently ships "Option C" (PR #35): *PPO is the broad-year /
out-of-distribution leader; the DT is scoped to its winning surfaces
(impact, dispatch-matched, mild markets).* This workstream challenges that
scoping by exploiting a finding the repo already documented but never acted on:

> **PPO-only DTs beat SB3 PPO on total profit on the broad 2024 surface
> ($17.6–17.8k vs $15.0k/ep), despite much lower FCAS, via higher energy
> arbitrage.** (report.md §8.2.1a, `docs/aemo_research_plan.md`)

## 2. Environment decision (LOCKED)

**Primary development/evaluation surface: `identity` (price-taking, the env
default). Impact (`piecewise_merit_order`) is a REQUIRED validation gate for
any final "best" claim — not the primary training/eval surface.**

Rationale:

| Reason | Detail |
|---|---|
| Every baseline number lives on identity | PPO-only DT ($17.6–17.8k), PPO ($15.0k), the broad 2024 expanded surface, the 2025 OOD surface, the v2/PPO-only/FCAS-heavy training corpora, dispatch-matched, and the 5-min protocol all use `impact_model='identity'`. Switching surfaces invalidates every reference number. |
| The `autoresearch_evaluator` configs have no impact key | They are identity-only by construction. Adding impact would require env/eval changes (read-only during autoresearch). |
| Impact only matters at grid scale | At 8 MW, DT and PPO retain 62% of identity profit under impact identically; the DT edge only emerges at 150–250 MW (`report.md` §8.2.9). The broad-surface/FCAS-spike behavior is best studied on identity first. |
| Impact is already a solved DT win | The impact-trained DT (`energydecision-dt-v2-impact`) beats PPO 9/9 cells (+$115K/cell, p=0.004). We must not regress it, and we don't need to re-prove it. |

**Decision rule for the final claim:** any model promoted as "best" must (a)
win on the identity broad surface (2024 expanded + 2025 OOD) AND (b) not regress
on the impact benchmark (`configs/impact_benchmark.json` via
`scripts/phase3_impact_eval.py`). Both gates required.

## 3. Reference numbers (what we are trying to beat)

All identity, 5-min protocol (`step_duration=0.083333`).

### Broad 2024 expanded surface (5 regions × 6 periods, 288h, medium 10MWh/5MW)

| Model | Profit/ep | FCAS/ep | Energy/ep | Degradation note |
|---|---:|---:|---:|---|
| **PPO-only DT (8×384)** (`models/aemo/dt/ppo_only_dt_best.pt`) | **$17,606** | $2,140 | $17,099 | profit is already net of degradation |
| **PPO-only DT (modern 8×768)** (`models/aemo/dt/modernv2_ppo_only_dt_best.pt`) | **$17,775** | $2,133 | $17,375 | profit is already net of degradation |
| PPO reference (`models/aemo_sb3/ppo_aemo_model.zip`) | $15,017 | $10,204 | $8,766 | ~$310/ep deg in other surfaces |
| Modern v2 mixed (flagship) | $4,596 | $4,774 | $281 | — |
| FCAS-heavy-subset DT | $13,387 | $5,860 | $12,578 | — |

### 2025 OOD surface (NSW1/SA1/QLD1 × Jan/Feb) — THE UNKNOWN

| Model | Profit/ep | FCAS/ep | Energy/ep |
|---|---:|---:|---:|
| PPO reference | $14,320 | $10,637 | $9,539 |
| Full-PPO fine-tuned modern v2 (2025) | −$694 | $6,793 | −$1,220 |
| **PPO-only DT (pretrained) on 2025** | **?** | **?** | **?** |

> **CRITICAL GAP:** the offline PPO-only *pretrained* DT was **never evaluated
> on 2025**. The −$694 figure is the *full-PPO value-critic fine-tune*, a
> different model. This is Experiment 0 below. If the PPO-only pretrained DT
> generalizes to 2025, the "PPO is the OOD leader" narrative flips today at zero
> training cost.

### Profit is net of degradation — confirmed

Reward formula (src/AEMOBatteryEnv.py `_calculate_reward`):
`reward = energy_revenue + fcas_revenue − degradation_cost + soc_penalty`,
normalized `/1000`. Eval `avg_profit_per_episode` = denormalized sum, so **the
headline profit numbers already account for degradation.** No re-derivation
needed.

## 4. Diagnosis (why the DT structurally fails broad-surface)

1. **The DT is return-conditioned behaviour cloning** (the kzl/decision-transformer
   paradigm): causal GPT over (R,S,A) tokens, MSE on actions, bounded by the
   offline data. The modern backbone (GQA/QK-norm/RMSNorm/SwiGLU/weight-tying)
   does not change the paradigm.
2. **FCAS–energy headroom is hard-coupled** (`AEMOBatteryEnv.py:1032-1088`):
   discharging eats raise-headroom; over-bids are *proportionally scaled*, not
   clipped. Optimal joint bidding is non-trivial; the Oracle LP encodes it
   exactly (`aemo_oracle_algo.py:210-255`).
3. **FCAS spikes are rare + aggressive + unpredictable** (TTM-FCAS corr
   ~0.01–0.07): the correct spike action is far from the offline data mass →
   distributional shift → BC collapses exactly there. PPO's online value
   function adapts and can output the rare action.
4. **Consequence:** mixed-data DT is balanced-but-mediocre ($4.6k); PPO-only DT
   is energy-heavy-but-winning on total profit ($17.6k) but loses the FCAS
   dimension. Nobody has combined the two capabilities.

## 5. Prioritized experiment plan

Order matters: cheap/evidence-first, then structural. Everything is on
**identity** unless marked. All AEMO eval must use the **5-min protocol**.

### Experiment 0 — Validate the existing PPO-only DT on the missing surfaces (zero training cost)

The single highest-value/cheapest action. Evaluate the two existing PPO-only
checkpoints on: (a) **2025 OOD surface**, (b) **dispatch-matched**
(`configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json`), (c)
**standard Oct** (`eval_tier_standard`). Compare vs PPO.

- `models/aemo/dt/ppo_only_dt_best.pt` (8×384)
- `models/aemo/dt/modernv2_ppo_only_dt_best.pt` (8×768, preferred flagship)
- Surfaces: 2025 (use `configs/aemo_autoresearch_evaluator.2025.json` if present,
  else build from `aemo_autoresearch_evaluator_expanded.json` with 2025 dates),
  dispatch-matched, standard.
- Success: the 8×768 PPO-only DT stays **> PPO on 2025 total profit** and on
  broad-2024 (already known). If it does, the PPO-only DT becomes the broad
  flagship candidate. If it collapses on 2025 (like the full-PPO fine-tune),
  the data-composition route (Exp 1) is the fix.
- RTG: sweep 0/5/10/20/50 (RTG response is architecture-dependent; modern peaks
  at 0 on dispatch-matched but calibrate per surface).

### Experiment 1 — Profit-maximizing data mixture (in-surface, cheap)

The three endpoints are known (pure-PPO energy-heavy $17.8k; mixed balanced
$4.6k; FCAS-heavy $13.4k). The interior optimum was never searched. Probe a
small grid of PPO : FCAS-heavy : A2C-TD3-SAC-DDPG mixtures on the modern v2
8×768 architecture, ranked by broad-2024 total profit, with a 2025 check on the
winner.

- Data: `data/aemo_dt_fcas_v2/` (77M rows), PPO subset
  `data/aemo_dt_fcas_ppo_only/`, FCAS-heavy subset
  `data/aemo_dt_fcas_fcasheavy/`.
- Mixtures to try first: 60/20/20, 40/40/20, 50/50/0 (PPO/FCAS-heavy/other).
- Cost: each is a standard 8×768 retrain (~1–2h) + expanded eval (~30–40min).
- Success: profit > $17.8k AND FCAS > $3k (regaining FCAS without losing the
  energy edge). Log each in `results.tsv`.

### Experiment 2 — Fix the action head: FCAS Bernoulli/Sigmoid (in-surface)

Structural bug: pretraining regresses all 9 dims through one `Tanh`+MSE
(`decision_transformer.py:415,490`; `transformer_training.py:705-710`), so FCAS
targets in [0,1] can never be hit exactly, and there is **no inference-time
clip** (`decision.py:1097-1106`). GRPO already has the correct mixed
distribution (`grpo_posttraining.py:251-299`: Tanh-Normal for energy,
Sigmoid-Normal for the 8 FCAS dims). Port it into the **supervised** loss:

- Per-dim NLL: Gaussian-on-Tanh for energy dim, Gaussian-on-Sigmoid (or BCE)
  for the 8 FCAS dims.
- Inference clip of FCAS dims to [0,1] in `decision.py` DT rollout.
- Keeps `act_dim=9`, state/act dims unchanged → existing checkpoints/adapters
  compatible. Within the editable surface (`scripts/pretrain_decision_transformer.py`)
  + the already-sanctioned `transformer_training.py` loss path.
- Success: FCAS capture on the FCAS-heavy and mixed models improves with no
  profit loss. Re-run the prior `--action-dim-weights` result to confirm this
  fixes what weighting could not.

### Experiment 3 — Hierarchical DT (SOC trajectory) + Oracle-LP executor (structural build)

The strongest structural proposal. Decompose along the env's own coupling:

- **DT predicts the target SOC trajectory** (1-D, smooth, slow signal) —
  immune to the FCAS-spike contamination, plays to transformer sequence
  strengths, allows degradation shaping (cycle minimization) directly.
- **Oracle LP** (`aemo_oracle_algo.py`), given the target SOC path + current
  prices, solves per-step energy+FCAS co-optimization **optimally** →
  guaranteed FCAS capture *given* the SOC path.
- Net effect: DT learns "what SOC schedule maximizes profit under plausible
  prices" (what SDP does with MC scenarios, but with learned price dynamics);
  LP handles the hard 9-D joint bidding. This is the only proposal that
  structurally wins BOTH broad (energy arbitrage) AND narrow (FCAS capture).
- Needs: new adapter (DT output → SOC schedule → LP executor in `decision.py`),
  training data = Oracle-derived SOC schedules, eval on all surfaces.
- Reference: roadmap already names hierarchical SDP+DT inference
  (`report.md:839`) but never built it.

### Experiment 4 — Imitate the Oracle, but with limited lookahead (research PR)

PPO is a mediocre teacher; the Oracle LP is the provably-optimal one
(0.02s/288 steps). Generate Oracle episodes (perfect-foresight LP) across
regions/years and train the DT on them. **Trap:** cloning perfect-foresight
actions teaches anticipation the DT can't support (obs has no future prices,
idx 5–14 are current-step only). Fix: **MPC-Oracle with horizon H** — re-solve
the LP every H steps with a rolling point forecast (TTM forecasts are weak for
FCAS but fine for energy, which drives SOC timing). Then the DT clones
*realistically-informed* optimal behavior within its 210-step context.
- Needs: new data-generation script + adapter for the MPC-Oracle policy.
- Success: broad-2024 profit and FCAS both above the current mixed model.

### Experiment 5 — Offline value-based RL (IQL/CQL) for FCAS spikes (research PR)

Root cause is distributional shift on spike actions — a problem BC *cannot*
solve by design. IQL (Implicit Q-Learning) learns Q(s,a) with expectile
regression and can output actions outside the data support when the value says
so — the principled fix for FCAS-spike capture. The repo has never tried any
offline-Q method (only BC-DT and online-RL fine-tuning).
- Needs: new `src/` implementation (outside the autoresearch surface → a
  separate research PR, not folded into the constrained loop).
- Success: FCAS capture on spike months at or above PPO, total profit ≥ PPO.

### Experiment 6 — Distributional spike-risk conditioning (revisit forecast DT)

The forecast DT failed because TTM *point* forecasts have ~0 corr with FCAS
spikes. Don't use a point forecast — add a **spike-risk scalar** derivable from
history alone (recent price volatility, time-of-day, regime flag, headroom) as
an extra obs dim or conditioning token. Cheap to build, sidesteps the dead TTM
line. Defer unless Exps 0–3 plateau.

## 6. Evaluation protocol (non-negotiable)

- **5-min steps everywhere**: `step_duration=0.083333` (matches DT training
  data; 30-min steps nearly halve the DT).
- **Surfaces required for any "best" claim**: broad 2024 expanded (identity),
  2025 OOD (identity), dispatch-matched, standard Oct, AND the impact benchmark
  gate (`phase3_impact_eval.py --impact-config configs/impact_benchmark.json`).
- **RTG calibration per model+surface**: sweep 0/5/10/20/50; modern peaks at 0
  on dispatch-matched, legacy at 0.5 — never assume transfer.
- **Metric of record**: total profit/ep (already net of degradation). Report
  profit + FCAS + energy decomposition together (revenue decomposition is not
  "who wins").
- **Runtime**: full expanded eval ~30–40 min on the 22 GB GPU (batched DT
  rollout + thread-parallel are default).
- **Tests**: run `python -m pytest tests/ -v` before/after code changes.

## 7. Success criteria

1. A DT (recomposed data or hierarchical or Oracle-imitated) beats PPO on the
   **broad 2024 expanded surface on total profit** (≥ $17.8k baseline set by
   the PPO-only DT) AND **holds on 2025 OOD** (≥ PPO's $14.3k, or at least
   clearly positive).
2. FCAS capture is no longer the single point of failure: FCAS/ep on the broad
   surface ≥ $6k (up from the PPO-only DT's $2.1k; PPO is $10.2k) while keeping
   the energy edge.
3. No regression on the impact benchmark (impact-DT remains the grid-scale
   leader).
4. The claim "DT is the preferred algorithm for AEMO battery control" is backed
   by both required surfaces — measured, not assumed.

## 8. Checklist (agent session)

- [ ] **Exp 0** — Evaluate existing PPO-only DT checkpoints on 2025 OOD,
      dispatch-matched, standard Oct. Record in `results.tsv`.
- [ ] **Exp 1** — Data-mixture grid (60/20/20, 40/40/20, 50/50/0) on modern
      v2 8×768 → broad-2024 + 2025 check. Record in `results.tsv`.
- [ ] **Exp 2** — Port GRPO mixed distribution (Tanh-energy / Sigmoid-FCAS)
      into the supervised loss + inference FCAS clip; compare vs
      `--action-dim-weights` result.
- [ ] **Exp 3** — Hierarchical DT (SOC trajectory) + Oracle-LP executor
      prototype; eval on all surfaces.
- [ ] **Exp 4** — MPC-Oracle (limited lookahead) data generation + DT retrain.
- [ ] **Exp 5** — IQL/CQL baseline on the FCAS-rich corpus (separate research
      PR).
- [ ] **Exp 6** — Spike-risk conditioning token (only if 0–3 plateau).
- [ ] **Impact gate** — Run `phase3_impact_eval.py` on the final candidate;
      confirm no regression.
- [ ] **Docs** — Update `report.md` §8.2.1a/§8.3 and the README roadmap with
      final verdict; close or keep Option C accordingly.
- [ ] `python -m pytest tests/ -v` green before merge.
- [ ] Open/refresh PR description with the headline table + verdict.

## 9. Autoresearch constraints (reminder)

- Editable surface only: `scripts/pretrain_decision_transformer.py` (+ its
  already-sanctioned loss path in `transformer_training.py`). Env, evaluator,
  datasets, notebooks are read-only during the constrained loop.
- Exps 4–6 require new `src/` code → do them as clearly-separated commits/PRs
  or in the research-branch workflow, not inside the constrained autoresearch
  surface.
- Log every experiment in `results.tsv` (untracked): `commit\ttrack\tmetric\tstatus\tdescription`.
- Long GPU runs: `bash scripts/run_full_learning_baseline.sh <TAG>` with
  telemetry; only shut down after `SAFE_TO_SHUTDOWN.txt` exists.

## 10. Session diary

### 2026-08-12 — Session start
- Created `feature/dt-preferred-aemo-policy` branch (PR #36).
- Wrote this plan. Locked the environment decision: identity-first, impact gate.
- Key facts established: profit numbers are already net of degradation; the
  PPO-only DT was never evaluated on 2025; the mixed-distribution action head
  exists in GRPO but not in pretraining; no offline-Q (IQL/CQL) has ever been
  tried in this repo.
- Next action for the agent: **Exp 0** (zero-training-cost 2025 + dispatch-
  matched + standard eval of the existing PPO-only DTs).
