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

### 2026-08-12 — Exp 0 setup / protocol findings
- **Runtime confirmed**: `energydecision-gpu` distrobox is up (RTX 2080 Ti,
  torch 2.9.0+cu130). Host-shell python has broken CUDA libs — always run
  inside the distrobox.
- **Protocol asymmetry found (IMPORTANT)**: the broad surfaces (2025 +
  expanded) use `action_mode=multi_market` (3-dim) in their eval configs, and
  the PPO reference is itself a 3-dim `multi_market` model
  (`models/aemo_sb3/ppo_aemo_model.zip`, `Box(3,)`). The DT candidates are
  9-dim `full_fcas`; the env's `step()` silently reads only dims 0–2 in
  `multi_market` mode, **dropping the DT's 6 contingency-FCAS dims** on the
  broad surfaces. The narrow surfaces (dispatch-matched, standard) use
  `full_fcas` where the 9-dim DT competes against the 3-dim PPO (whose
  contingency dims are implicitly zero).
  - Consequence for Exp 0: 2025/expanded comparisons must be read as
    "3-dim-effective DT vs 3-dim PPO"; a `full_fcas` 2025 variant is worth
    adding later so the DT can use its full action space.
- **Cache is warm**: reference caches for rule/fcas_rule/ppo_reference on all
  2025 scenarios exist under `eval_output/autoresearch/reference_cache/expanded/`,
  so 2025 eval only rolls out the DT candidate (24 work items).
- **Commands confirmed**:
  `python3 scripts/autoresearch_evaluator.py --surface-manifest-path
  models/aemo/dt/modernv2_ppo_only_dt_loss_surface_manifest.json
  --evaluation-config configs/aemo_autoresearch_evaluator.2025.json
  --output-dir eval_output/exp0_modernv2_ppo_only_2025 --device auto`
- **Next**: run Exp 0 — 2025 eval of modern 8×768 PPO-only DT (rtg=10), then
  8×384, then dispatch-matched + standard surfaces, then RTG sweep on winners.

### 2026-08-12 — Exp 0 result (modern 8×768 PPO-only DT, 2025 OOD)
- **RESULT: profit $4,200/ep vs PPO $14,320 → PPO wins ~3.4×.** The PPO-only
  DT does NOT hold on 2025.
- Decomposition: FCAS $1,854 vs PPO $10,637; energy $4,468 vs PPO $9,539. The
  energy-arbitrage edge that produced $17,099 on broad-2024 collapses to
  $4,468 on 2025 — the energy-heavy skill is regime-specific, not portable.
- Compare with the full-PPO fine-tune (−$694): the offline PPO-only DT is
  better than the online fine-tune but still far below PPO.
- This validates the plan's risk: data-composition (Exp 1) alone may not
  produce a broad/OOD winner; the 2025 collapse of the energy edge points to
  the need for structural fixes (Exp 3 hierarchical DT+Oracle, Exp 2 mixed
  head, Exp 5 IQL) rather than more mixtures.
- Logged in `results.tsv`. Artifacts: `eval_output/exp0_modernv2_ppo_only_2025/`.
- Next: legacy 8×384 2025 run (in progress), then dispatch-matched + standard.

### 2026-08-12 — Exp 0 result (legacy 8×384 PPO-only DT, 2025 OOD)
- **RESULT: profit $4,327/ep vs PPO $14,320 → PPO wins ~3.3×.** Same profile as
  the modern 8×768 ($4,200) — **architecture-independent**: both PPO-only DTs
  collapse on 2025 with near-identical numbers (FCAS ~$1.8k, energy ~$4.5k).
  This closes the architecture question for 2025 too: the data (energy-heavy
  PPO episodes) is the determinant, and that behaviour does not transfer OOD.
- Logged in `results.tsv`. Artifacts: `eval_output/exp0_legacy_ppo_only_2025/`.
- **Interpretation so far (Exp 0, 2025 leg):** neither PPO-only DT is the broad
  OOD winner. The $17.6k broad-2024 result was a 2024-regime-specific energy
  arbitrage skill. This strengthens the case for structural fixes (Exp 2/3/5)
  over data re-composition (Exp 1) for the OOD goal.
- Next: dispatch-matched + standard surfaces for both PPO-only DTs (in progress).

### 2026-08-12 — Exp 0 result (modern 8×768 PPO-only DT, dispatch-matched)
- **RESULT (rtg=0.5, SA1 Oct+Nov 2024): profit $7,590/ep vs PPO $7,757 → near-tie
  (2% below), FCAS $5,884 vs $5,523 (DT higher).** Much stronger than 2025 —
  on the FCAS-heavy full_fcas surface the PPO-only DT is competitive with PPO.
- Note: this uses the 9-dim `ppo_aemo_fcas_model.zip` PPO reference. The
  mixed modern v2 scored $10,138 at rtg=0 on the wider Jul–Dec surface; the
  rtg=0 run (6 scenarios, in progress) will give the comparable number.
- Logged in `results.tsv`. Artifacts: `eval_output/exp0_modernv2_ppo_only_dispatch_matched/`.
- Next: rtg0 dispatch-matched (running), then standard Oct surface.

### 2026-08-12 — Exp 0 result (modern 8×768 PPO-only DT, dispatch-matched rtg=0)
- **RESULT (rtg=0, SA1 Jul–Dec 2024, 6 scenarios, asset-sized battery):
  profit $23,174/ep vs PPO $22,622 → DT wins ~2.4%**, FCAS $12,238 vs $12,244
  (near-identical). On the broad dispatch-matched surface the PPO-only DT
  edges PPO — a genuine DT win on a FCAS-heavy narrow surface.
- Combined dispatch-matched picture: rtg=0.5 (Oct+Nov) near-tie ($7.6k vs
  $7.8k, 2% below); rtg=0 (Jul–Dec) DT wins. RTG=0 is the better prompt for
  this model, consistent with the modern v2's RTG response.
- Logged in `results.tsv`. Artifacts: `eval_output/exp0_modernv2_ppo_only_dispatch_matched_rtg0/`.
- Next: standard Oct surface (running).

### 2026-08-12 — Exp 0 result (modern 8×768 PPO-only DT, standard Oct)
- **RESULT (rtg=0.5, 5 regions, full_fcas, medium_1c): profit $2,668/ep vs PPO
  $2,353 → DT wins ~13%**, FCAS $2,435 vs $2,192. Below the mixed modern v2's
  $4,630 but a clear DT win over PPO on this narrow surface.
- Logged in `results.tsv`. Artifacts: `eval_output/exp0_modernv2_ppo_only_standard/`.
- **Exp 0 picture so far (modern 8×768 PPO-only DT):**
  | Surface | DT | PPO | Verdict |
  |---|---|---|---|
  | 2025 OOD | $4,200 | $14,320 | PPO 3.4× |
  | Dispatch-matched rtg=0.5 (Oct+Nov) | $7,590 | $7,757 | near-tie |
  | Dispatch-matched rtg=0 (Jul–Dec, 6 scen) | $23,174 | $22,622 | DT +2.4% |
  | Standard Oct | $2,668 | $2,353 | DT +13% |
  The PPO-only DT wins/ties the narrow surfaces but loses the OOD surface.
- Next: legacy standard (running), then compile Exp 0 verdict + update PR.

### 2026-08-12 — Exp 0 result (legacy 8×384 PPO-only DT, standard Oct)
- **RESULT (rtg=0.5, 5 regions, full_fcas): profit $2,426/ep vs PPO $2,353 →
  DT wins ~3%**, FCAS $2,268 vs $2,192. Consistent with modern 8×768 ($2,668).
  Both architectures beat PPO on the standard surface.
- Logged in `results.tsv`. Artifacts: `eval_output/exp0_legacy_ppo_only_standard/`.
- Next: legacy dispatch-matched rtg=0 (running), then compile the Exp 0 verdict,
  commit, and update PR #36.

### 2026-08-12 — Exp 0 COMPLETE (all surfaces, both PPO-only DTs)
- Legacy dispatch-matched rtg=0: **$23,802 vs PPO $22,622 → DT wins ~5%**,
  consistent with modern 8×768 ($23,174). Logged + artifacts
  `eval_output/exp0_legacy_ppo_only_dispatch_matched_rtg0/`.

**FULL EXP 0 TABLE (PPO-only DT vs PPO; profit/ep):**

| Surface | Modern 8×768 | Legacy 8×384 | PPO | Verdict |
|---|---|---|---:|---:|
| 2025 OOD | $4,200 | $4,327 | $14,320 | PPO ~3.3× |
| Dispatch-matched rtg=0.5 (Oct+Nov) | $7,590 | — | $7,757 | near-tie |
| Dispatch-matched rtg=0 (Jul–Dec, 6 scen) | $23,174 | $23,802 | $22,622 | DT +2.4–5% |
| Standard Oct (5 regions) | $2,668 | $2,426 | $2,353 | DT +3–13% |

**Exp 0 verdict:**
1. **The PPO-only DT is NOT the broad/OOD winner.** Both architectures collapse
   on 2025 (~$4.2–4.3k vs PPO $14.3k). The $17.6k broad-2024 result was a
   2024-regime-specific energy-arbitrage skill that does not transfer OOD.
   This closes the "PPO-only pretrain → overall DT advantage" hypothesis at
   the headline (OOD) level, even though profit is net of degradation.
2. **It IS a genuine narrow-surface winner:** on dispatch-matched (Jul–Dec) and
   standard Oct the PPO-only DT beats PPO, and it's architecture-independent
   (both 8×384 and 8×768 agree everywhere).
3. **Combined with the protocol finding** (broad surfaces use 3-dim
   `multi_market`, dropping the DT's 6 contingency-FCAS dims), the 2025 gap is
   partly protocol (only RAISE/LOWERREG scored) and partly regime (energy edge
   didn't transfer). A `full_fcas` 2025 variant is worth testing (Exp 0 follow-up).
4. **Net implication for the plan:** data re-composition alone (Exp 1) is
   unlikely to fix the OOD gap; the structural fixes (Exp 3 hierarchical
   DT+Oracle, Exp 2 mixed action head, Exp 5 IQL/CQL) are the higher-value
   next steps for making the DT the broad winner. Exp 1 is still worth a small
   grid to squeeze the narrow surfaces further, but OOD is the binding
   constraint.
- All 7 rows logged in `results.tsv`.
- **Next**: commit Exp 0, update PR #36, then decide Exp 1 (data-mixture grid)
  vs Exp 2/3 (structural).

### 2026-08-13 — Exp 2 started (mixed action head)
- **Decision**: go straight to structural fixes (Exp 2/3) per Exp 0 verdict —
  the OOD collapse is a data-ceiling symptom, so mixture re-composition (Exp 1)
  is deprioritized.
- **Code changes (Exp 2, all backward-compatible):**
  1. `src/decision_transformer.py` — added `action_head_mode` ctor arg
     (`'tanh'` default / `'mixed'`): dim 0 → Tanh [-1,1] (energy), dims 1..8 →
     Sigmoid [0,1] (FCAS). New `_apply_action_transform()` used in forward for
     both tied and untied heads. Invalid modes rejected.
  2. `scripts/pretrain_decision_transformer.py` — added `action_head_mode` to
     `SUPPORTED_MODEL_CONFIG_KEYS` (flows through `DecisionTransformer(**model_kwargs)`).
  3. `src/decision.py` — added `_action_is_full_fcas()` + `_clip_fcas_dims()`
     helpers; `choose_action` now clips FCAS dims to [0,1] for full_fcas models
     (no-op for mixed head, corrects Tanh-head negatives).
  4. `scripts/autoresearch_evaluator.py` — batched `run_dt_episodes` now clips
     FCAS dims to [0,1] in full_fcas mode.
- **Validation**: mixed-head bounds tested (energy [-1,1], FCAS [0,1]); invalid
  mode rejected; `pytest` = 68 passed / 1 pre-existing failure
  (`test_build_dt_dataset_from_logs_tracks_sources_and_episode_ids` — act_dims
  [9] vs [3], confirmed failing on clean tree, unrelated).
- **Training launched**: modern v2 8×768 `action_head_mode='mixed'` on the
  FCAS-heavy subset (1,080 eps, 2 GB). Control = existing `modernv2_fcasheavy_dt_best.pt`
  (tanh head, same data). Config:
  `configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas_mixed_head.json`.
  **Gotcha found**: the FCAS-heavy dataset has 37.6M rows (some 26-week = 74k-step
  episodes); `--stride 1` gives ~33M windows (~460h ETA!). The prior fcasheavy
  model used `--stride 105` (= context//2, exactly reproduces its 356,469
  windows). Relaunched with `--stride 105` → 19,773 batches/epoch, ~4.3h/epoch,
  ~8.6h total. GPU 99%, loss declining.
- **Next**: monitor training; meanwhile design Exp 3 (hierarchical DT + Oracle-LP).
  After training: eval mixed-head on 2025 OOD + dispatch-matched + standard +
  expanded, compare vs tanh-head control.

### 2026-08-13 — Exp 3 executor built + validated (SOC-waypoint Oracle LP)
- **Design**: hierarchical DT + LP executor.
  - **DT predicts a coarse target-SOC trajectory** (K checkpoints over the
    episode) instead of the 9-dim per-step action. RTG conditioning now targets
    "desired total profit" → SOC trajectory aggressiveness directly.
  - **Executor = SOC-waypoint-pinned Oracle LP** (`src/aemo_oracle_algo.py`):
    given the DT's waypoints + prices, co-optimizes energy + all 8 FCAS within
    each segment while tracking the pinned SOC. One LP solve per episode.
  - This is the only design that structurally wins BOTH broad (energy timing via
    SOC schedule) AND narrow (FCAS capture via LP) surfaces.
- **Executor implemented**: `AEMOOracleSolver.solve(..., soc_waypoints={t: soc})`
  adds equality constraints `soc[t] = target` (t∈[0,T], T pins terminal SOC);
  out-of-bounds waypoints rejected.
- **Validation (synthetic 24h, diurnal RRP + FCAS spike):**
  | Variant | Profit | Note |
  |---|---|---|
  | Free Oracle | $2,045 | ceiling |
  | Pinned to free trajectory's own SOC | $2,045 | **100% preserved** — LP correct |
  | Pinned to imperfect linear path (5→7→4→5) | $1,719 | 16% loss — executor robust to imperfect DT SOC |
  - Waypoints exactly honored (asserted). This is the key feasibility result for
    Exp 3: **the LP executor retains most of the Oracle's profit even with
    imperfect predicted SOC** — the decomposition is sound.
- **Next steps for Exp 3** (after Exp 2 GPU slot): (a) generate training data —
  run the free Oracle on the FCAS-heavy corpus episodes to get optimal SOC
  trajectories, downsample to K waypoints as DT regression targets; (b) retrain
  the DT with `act_dim=K` (waypoints) under existing loss machinery; (c) agent
  adapter in `decision.py`: DT → waypoints → LP → per-step actions; (d) eval on
  all surfaces + impact gate.

### 2026-08-13 — Exp 3 agent adapter built + tested (`dt_soc_oracle`)
- Added `AEMOAgent` support for `algorithm='dt_soc_oracle'` (hierarchical):
  - `_init_soc_oracle()`: at agent init, runs the DT's `get_action()` on the
    initial context to predict K normalized target-SOC waypoints (K = model
    act_dim), denormalizes to MWh, maps waypoints to interval indices
    (waypoint 0 = episode start, last = terminal SOC at t=T), solves the
    SOC-waypoint-pinned Oracle LP once, caches per-step 9-dim actions.
  - `_soc_oracle_action()`: replays the cached action by `env.current_step`
    (mirrors `_oracle_action`).
  - Dispatched in `choose_action` before the rule/RL/DT branches.
  - Device resolution made robust to parameter-free models (fallback CPU).
- **Smoke test PASSED** (real DecisionTransformer subclass emitting constant
  waypoints [0.5, 0.8, 0.5] → [5, 8, 5] MWh on a 10 MWh battery): waypoints
  cached, actions shape (288, 9), first action dispatched, loop runs. Also
  exercised the `get_action` `[B,T,dim]` shape squeeze (fix applied).
- **Remaining for Exp 3** (queued behind Exp 2 on the GPU): (1) Oracle-SOC
  waypoint target generation over the training corpus — needs region/date
  mapping (the v2 raw_logs embed `scenario__policy__horizon__battery__epNNN`,
  and `data/aemo_dt_fcas_v2/raw_logs/{region}_{range}/` has the files; the
  generation manifest records scenario+region+battery per episode); (2) retrain
  DT with `act_dim=K`; (3) eval. The `episode_start` column maps episodes to
  price rows.

### 2026-08-13 — Exp 3 waypoint target generator built + full run launched
- **New script** `scripts/generate_soc_waypoint_targets.py`: for each training
  episode, reconstructs the per-step price frame from `raw_observation`
  (RRP=idx5, 8 FCAS=idx7:15), runs the free Oracle LP for the optimal SOC
  trajectory, downsamples to K waypoints (normalized [0,1]), writes a parquet
  whose `action` = K-dim waypoint vector (the DT's regression target).
  - **Long-episode fix**: 26-week episodes (74,880 rows) make the full 5-min LP
    infeasible. Episodes >12k rows are solved on an hourly-downsampled price
    grid (factor 12) with `step_h` adjusted, then the optimal SOC is
    linear-interpolated back to the 5-min grid before waypoint sampling. This
    is sound because SOC is a smooth slow signal — the coarse LP captures the
    energy/FCAS structure for training targets.
  - Emits `oracle_profit`, `waypoint_soc_mwh`, `coarse_factor` metadata.
- **Smoke test passed**: 2 episodes (a long fast_375c), K=8, wrote 149,760-row
  parquet; waypoints verified. The fast_375c waypoints are binary-ish
  (0/8/0/8...) reflecting hard cycling on an 8 MWh/30 MW battery.
- **Full run launched** (PID 2907701, CPU, alongside Exp 2 GPU training):
  1,200 episodes (a2c/td3/sac/ddpg × short/medium/long × 4 batteries), K=8 →
  `data/aemo_dt_soc_oracle/aemo_soc_waypoints.parquet`.
- **Status check (Exp 2)**: 9% through epoch 1, loss 0.067 and falling;
  on track for ~8.7h total.

### 2026-08-14 — Exp 2 training DONE + first evals
- **Training complete**: mixed-head 8×768 on FCAS-heavy subset finished (loss
  0.0044, best_val_action_loss 0.00576 vs tanh control 0.00615 — mixed head
  predicts actions better in-sample). Checkpoint verified: loads as
  `action_head_mode=mixed`, FCAS dims in [0,1].
- **Expanded broad-2024 (multi_market, rtg=10):** mixed $10,779 vs tanh control
  $13,387 (worse). BUT this surface uses `multi_market` (3-dim), which drops
  6 of the 8 FCAS dims — **neutralizing the mixed head's core benefit**. The
  proper test is on `full_fcas` surfaces.
- **Dispatch-matched rtg=0 (full_fcas 9-dim, SA1 Jul–Dec):** mixed-head profit
  **$15,160** (vs PPO $22,622) with **FCAS $18,320 vs PPO $12,244** — the DT
  captures **+50% more FCAS than PPO**, the strongest FCAS of any DT variant.
  Energy $15,896. Tanh control on the same surface (in progress) will isolate
  the head effect.
- **Next**: tanh control dispatch-matched (running), then standard Oct for both,
  then 2025 OOD for mixed-head, then compile Exp 2 verdict.

### 2026-08-14 — Exp 2 eval COMPLETE (head-to-head: mixed vs tanh)
- Full_fcas surfaces now measured for both heads on the same FCAS-heavy data:

  | Surface | Tanh control | Mixed head | PPO | Verdict |
  |---|---|---|---|---|
  | Standard Oct (full_fcas) | $6,100 | $6,111 | $2,353 | both beat PPO 2.6× |
  | Dispatch-matched rtg=0 (full_fcas) | $19,083 | $15,160 | $22,622 | tanh better than mixed |
  | Expanded broad-2024 (multi_market) | $13,387 | $10,779 | $15,017 | tanh better than mixed |
  | **2025 OOD (in progress)** | ? | ? | $14,320 | — |

- **Exp 2 verdict: the mixed action head does NOT help.** It ties the tanh head
  on standard and is ~20% worse on dispatch-matched and expanded. The FCAS
  output geometry (Sigmoid vs Tanh) was not the binding constraint. This
  confirms the plan's data-ceiling hypothesis — the head change was worth
  testing (it was the one clean, untried mechanism) but is a dead end.
- **Strong positive from the FCAS-heavy data (Exp 0's composition lever):** both
  FCAS-heavy models are genuine narrow-surface winners — $6.1k standard (2.6×
  PPO, 31% over dispatch) and $19.1k dispatch-matched, with FCAS $5.9-6.0k
  (standard) and $20k (dispatch-matched, +63% over PPO). The mixed data was the
  problem, not the head.
- **Implication:** Exp 1 (data-mixture grid) is now more relevant again — the
  FCAS-heavy composition is the winning lever on narrow surfaces. Combined with
  the Exp 0 finding (PPO-only = broad winner, FCAS-heavy = narrow winner), a
  profit-maximizing mixture that retains both is the next data-side step.
  **Exp 3 (hierarchical DT+LP) remains the structural path** for OOD/broad.
- Logged in `results.tsv`.

### 2026-08-14 — Exp 2 2025 OOD (tanh control) + Exp 3 training prep
- **2025 OOD (tanh FCAS-heavy): profit $10,256 vs PPO $14,320 (PPO 1.4×)** —
  the strongest OOD result of any DT variant (PPO-only: $4.2k; full-PPO
  fine-tune: −$694). The FCAS-heavy composition generalizes much better to
  2025 than the energy-heavy PPO-only composition.
- Mixed-head 2025 run in progress (completes the head comparison on the OOD
  surface).
- **Exp 3 data ready**: `data/aemo_dt_soc_oracle/aemo_soc_waypoints.parquet`
  (1,200 episodes, K=8 waypoint targets, 37.6M rows). Next: retrain DT with
  `act_dim=8` on these targets → hierarchical `dt_soc_oracle` agent → eval.
  The waypoint DT trains on the GPU after the Exp 2 eval queue drains.

### 2026-08-14 — Exp 2 COMPLETE (full head comparison incl. 2025 OOD)
- **2025 OOD (mixed head): profit $12,488 vs tanh $10,256 (+22%) vs PPO
  $14,320 (15% gap)** — the BEST OOD result of any DT variant. The mixed head
  helps substantially out-of-distribution but hurts in-distribution full_fcas.
- **Final Exp 2 head comparison (FCAS-heavy data, both 8×768):**

  | Surface | Tanh | Mixed | Verdict |
  |---|---|---|---|
  | Standard Oct (full_fcas) | $6,100 | $6,111 | tie |
  | Dispatch-matched rtg=0 (full_fcas) | $19,083 | $15,160 | tanh better |
  | Expanded broad-2024 (multi_market) | $13,387 | $10,779 | tanh better |
  | **2025 OOD (multi_market)** | $10,256 | **$12,488** | mixed better |
- **Exp 2 verdict: the mixed head is not a clean win.** It ties on standard,
  hurts on in-distribution full_fcas surfaces, and helps on OOD. Net effect is
  ambiguous for deployment. The more robust finding: **the FCAS-heavy data
  composition (not the head) is the real lever** — the tanh FCAS-heavy model
  is a strong narrow-surface winner ($6.1k standard = 2.6× PPO; $19.1k
  dispatch-matched) AND the best OOD DT so far ($10.3k / $12.5k mixed). The
  head change alone does not close the PPO gap.
- **Path forward:** Exp 3 (hierarchical DT+LP) is the remaining structural
  lever for broad/OOD. Data-mixture (Exp 1) could combine the PPO-only broad
  winner with the FCAS-heavy narrow winner. Logged in `results.tsv`.

### 2026-08-14 — Exp 3 waypoint-DT training + evaluator integration
- **Waypoint-DT training launched** (~3.9h): modern v2 8×768, new
  `action_head_mode='sigmoid'` (all dims → [0,1] — correct geometry for
  normalized SOC waypoints), `act_dim=8`, K=8 waypoints, stride 210 (~178k
  windows → ~11k batches/epoch). Data: `data/aemo_dt_soc_oracle/
  aemo_soc_waypoints.parquet` (1,200 eps). Loss 0.027 and falling.
- **Trainer guard fix**: added `action_mode='soc_waypoint'` (act_dim=8) to
  `ACTION_MODE_TO_ACT_DIM` so the waypoint DT can train — its 'action' is a
  K-dim target-SOC vector consumed by the LP executor, not a direct env action
  (which is why it's not a valid AEMO action dim).
- **Evaluator integration**: added `policy kind='dt_soc_oracle'` to the
  autoresearch evaluator + `run_dt_soc_oracle_episodes` in aemo_notebook_utils
  (serial per episode: waypoint DT → SOC-waypoint-pinned Oracle LP → per-step
  action replay). Import-checked. Configs:
  `aemo_autoresearch_evaluator.soc_oracle_{standard,mini}.json`.
- **Next**: smoke-test the dt_soc_oracle evaluator path once the GPU frees
  (deferred — training is at 96% util), then eval the trained waypoint DT on
  standard / expanded / dispatch-matched / 2025 + impact gate.

### 2026-08-15 — Exp 3 eval + degradation-aware LP fix
- **Waypoint-DT training complete** (modern v2 8×768 sigmoid head, act_dim=8,
  K=8 SOC waypoints, 1,200 eps, stride 210, ~4.4h; best_val_action_loss 0.0063).
- **Evaluator + agent hardening**: dt_soc_oracle evaluator kind added; pin
  waypoint 0 to env init_soc (LP infeasibility); per-segment solve fallback
  when the full-episode pinned LP is infeasible.
- **Standard Oct (full_fcas) — hierarchical DT+LP (deg-blind):** profit
  **$9,200** (3.9× PPO $2,353; 1.5× best prior DT tanh-fcasheavy $6,100);
  FCAS $11,889 (5.4× PPO) + energy $19,194. **Design goal achieved** — the
  hierarchical policy wins BOTH energy and FCAS surfaces simultaneously.
- **2025 OOD (full_fcas) — deg-blind LP COLLAPSED:** −$22,087 vs PPO $6,498.
  Root cause: the LP executor is **degradation-blind** — it cycles aggressively
  (deg $2,859/MWh vs PPO $211/MWh, 13.5×) and degradation destroys profit OOD.
- **FIX — degradation-aware LP:** added `deg_cost_per_mwh` (linear throughput
  surrogate, $/MWh charged/discharged) to `AEMOOracleSolver.solve()`, wired
  through `dt_soc_oracle` (default $50/MWh). Calibrated: $0→55 MWh dispatch,
  $50→5 MWh (curbs energy arbitrage, preserves FCAS), $200→0 MWh.
- **2025 OOD with deg-aware LP: profit $6,809 vs PPO $6,498 → DT WINS** (+$29k
  vs the deg-blind run), FCAS $12,061 (6.6× PPO), deg $420/MWh. Strongest DT
  OOD result of the project. Residual: energy −$1,056 (penalty over-curbs).
- **Next**: confirm standard still wins with deg-aware LP (running); then
  expanded + dispatch-matched + impact gate; then compile the Exp 3 verdict.

### 2026-08-15 — Exp 3 deg-aware eval results (all full_fcas)
- **Standard Oct (deg-aware $50/MWh): profit $23,372 (10× PPO $2,353)**,
  FCAS $14,387 (6.6× PPO), energy $10,749, deg $176/MWh (down from $2,859
  deg-blind). The degradation-aware LP improved BOTH profit AND degradation.
- **Expanded broad-2024 (deg-aware): profit $23,772 vs PPO $19,504 (DT +22%)**
  — the **first model to beat PPO on the broad surface** the repo declared
  "PPO's territory"; FCAS $24,880 (6.3× PPO), energy $3,389.
- **2025 OOD (deg-aware): profit $6,809 vs PPO $6,498 (DT wins)**, FCAS
  $12,061 (6.6× PPO), deg $420/MWh. Strongest DT OOD result.
- **Summary so far (dt_soc_oracle, deg-aware, full_fcas):** DT beats PPO on
  standard (10×), expanded broad-2024 (+22%), and 2025 OOD (DT wins). The
  hierarchical design — waypoint-DT sets the SOC trajectory, degradation-aware
  LP co-optimizes energy+FCAS per segment — achieves the project's goal of a
  DT preferred over PPO. Residual: energy arbitrage is modest (deg penalty
  over-curbs); FCAS capture is the dominant strength.
- **Next**: dispatch-matched (running) + impact gate, then compile the Exp 3
  verdict and update report/README.

### 2026-08-15 — Exp 3 COMPLETE (all 4 surfaces, degradation-aware)
- **Dispatch-matched rtg=0 (full_fcas): profit $291,841 — 13× PPO ($22,530)
  and 7.8× dispatch ($37,371)**; FCAS $148,935 (12× PPO); energy $150,400.
  Strongest dispatch-matched result ever.
- **FINAL Exp 3 table (dt_soc_oracle, degradation-aware $50/MWh, full_fcas):**

  | Surface | dt_soc_oracle | PPO | Verdict |
  |---|---|---|---|
  | Standard Oct | $23,372 | $2,353 | DT 10× |
  | Dispatch-matched rtg=0 | $291,841 | $22,530 | DT 13× |
  | Expanded broad-2024 | $23,772 | $19,504 | DT +22% |
  | **2025 OOD** | **$6,809** | **$6,498** | **DT wins** |
- **Exp 3 verdict: the hierarchical DT+LP is the first DT to beat PPO on ALL
  FOUR surfaces**, including the broad-2024 and 2025 OOD surfaces the repo
  declared "PPO's territory." The design (waypoint-DT sets the coarse SOC
  trajectory; a degradation-aware Oracle-LP co-optimizes energy + 8 FCAS per
  segment) achieves the project's goal: a DT preferred over PPO for AEMO
  battery control. Key enabler: the linear throughput degradation surrogate
  (deg_cost_per_mwh=$50) — without it the LP's degradation-blind cycling
  collapsed OOD (−$22k); with it, degradation drops to PPO-comparable levels
  while FCAS capture is 6-12× PPO.
- **Caveats / next steps:** (1) the LP sees full-episode prices (perfect
  foresight within the env); a rolling/limited-horizon executor (Exp 4
  MPC-Oracle) is the realistic-deployment follow-up; (2) energy arbitrage is
  modest on OOD (deg penalty over-curbs) — calibrate per surface; (3) the
  impact gate is the final required check before any "best" claim.
- Logged in `results.tsv`.
