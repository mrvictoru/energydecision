# Phase 2 — Personal Mastery Pass (study guide)

**Owner:** you. **I (the agent) do not do this for you.** My role from here is to
answer specific questions, look up code, and verify your understanding against the
artifacts — never to write the notes for you.

**Goal:** be able to explain, from memory and without notes, how the shipped
headline number is produced end to end — and survive a hostile technical
interview about it. This is what converts "an agent built it" into "I own it".

**Why it matters for the PhD push:** a supervisor will probe exactly the weak
spots (baseline tuning, circularity, simulator realism, small n, why distillation
beats a planner). If you can answer these cold, the repo becomes evidence of
research maturity instead of a black box.

---

## Ground rules

1. **Write before you move on.** For each module, write 5–15 lines in your own
   words *while looking at the code*, then close the code and rewrite from memory.
2. **If you can't explain a line, stop.** That is a gap (or a bug). Note it and
   either resolve it or ask me.
3. **Ask me only after you've formed a concrete question.** Good: "line 1643 in
   `decision.py` — why is `soc_kwh` rescaled by `capacity` before the table
   lookup?" Bad: "explain `_lookup_jtsoc_rtg`."
4. **Don't edit source during Phase 2.** Note issues in your log; we'll batch-fix
   later so the frozen release stays intact.
5. **Keep your notes out of the AI.** Type them yourself. (Optional: keep them in
   `paper/notes/` if you want them public as evidence of mastery — your call.)

**Runtime for every command below:** from the repo root inside the GPU box:

```bash
distrobox enter energydecision-gpu -- bash -lc 'cd /home/victoru/rescued/energydecision && <command>'
```

**Suggested budget:** ~6–8 focused hours, split into the six modules below, then
the synthesis + interview drill. Do not compress it into one sitting.

---

## Module 1 — The teacher solver (~60–75 min)

**Question to answer:** *What does the "honest SDP teacher" actually optimize, and
why is it "honest"?*

**Files / anchors**
- `src/aemo_sdp_solver.py` — `AEMOSDPSolver` (line 19). Note how thin it is; find
  the base class `SDPSolver` in `src/sdp_algorithm.py` and read that too.
- `src/aemo_sdp_executor.py`:
  - `build_seasonal_rrp_profile` (line 81) — which data does it read? (look at
    `find_training_parquet`, line 55)
  - `build_rrp_forecast` (line 117) — what does the planner actually "see" at each
    step?
  - `sdp_energy_dispatch` (line 134) — backward induction; what is the objective,
    what is the terminal penalty, what is `deg_cost_per_mwh` / `deg_calibration`?
  - `greedy_fcas_bids` (line 224) — how are FCAS bids decided?
  - `compute_cost_to_go_table` (line 265) — this produces `J_t(soc)`; what are the
    table axes and units?

**Verification**
```bash
python3 -m pytest tests/test_physics_v2_planner.py tests/test_aemo_degradation.py -q
```
Read the tests to see what invariants the team asserted.

**Write (from memory):** 10 lines — "The teacher plans energy + FCAS dispatch by
… It is honest because … It is *not* clairvoyant because …"

**Watch-outs to form explicit answers for:** Is the seasonal profile a forecast or
a peeking oracle? Does `deg_calibration` change the *policy* or just the reported
cost? What does "honest executor" mean vs the LP (perfect-foresight) stage?

---
**Summary:** The AEMO SDP solver represents the battery using a grid of SOC states and candidate charge/discharge actions. It works backward through time: for each SOC, it compares each feasible action’s forecast energy cost and wear cost plus the future cost of the resulting SOC. In waypoint dispatch, a terminal penalty encourages the plan to reach a specified SOC target; after building the policy, the solver rolls it forward from the starting SOC to produce actions. The price forecast is normally a month-and-hour average from pre-2024 data, not the episode’s actual future RRP. Separately, the free-terminal cost-to-go table estimates remaining cost for each time/SOC and supplies RTG = -cost_to_go as a DT prompt—it does not itself choose the dispatch path. FCAS bids are added by a separate greedy rule, not optimized by the energy SDP.
---

## Module 2 — Teacher → trajectory corpus (~60–75 min)

**Question to answer:** *How does one rollout become training rows, and what is
stored in each row?*

**Files / anchors**
- `scripts/generate_sdp_dt_trajectories.py`:
  - `REGION_FILES` (line 49), `HORIZON_STEPS` (line 46), `BATTERY_SPECS` (line 39)
  - `generate_slot` (line 124) — random window start, env construction, `run_episode`
  - `_build_slot_ctg` (line 92) and `_attach_jt_rtg` (line 69) — how the RTG column
    is attached per step
  - `main` (line 183) — the CLI that produced the shipped corpus
- Corpus + manifest (read the JSON, don't rerun generation):
  - `models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json` — dataset
    summary, split policy, `model_kwargs`, `training_kwargs`
  - `data/aemo_dt_sdp/dt_trajectories_jtsoc_v2cal_conservative.parquet` — inspect
    schema + 1 episode with polars (don't load all 400 MB into a notebook blindly).

**Verification**
```python
import polars as pl
lf = pl.scan_parquet("data/aemo_dt_sdp/dt_trajectories_jtsoc_v2cal_conservative.parquet")
print(lf.collect_schema())
print(lf.select("episode_id").unique().collect().head(10))
```
Confirm: how many episodes train/val, what the split unit is (`episode`), and that
the corpus stores **no timestamps** (the leakage caveat).

**Write:** 8 lines on the exact column contract
(`episode_id, step, norm_observation, action, reward, source_policy, rtg_value`)
and how `rtg_value` is computed.

---
***Summary:** generate_sdp_dt_trajectories.py samples historical episode windows by region, horizon, and battery. It requires a previously trained waypoint DT, whose SOC-waypoint labels came from offline Oracle LP trajectories. That model proposes coarse SOC targets; the SDP plans energy between them, and a greedy rule adds FCAS bids. The simulated episode produces observation, action, and reward rows for a new, standalone DT to learn from. With --rtg-mode j_t_soc, the script also adds a per-step cost-to-go-derived RTG value. Thus, this is a hybrid teacher pipeline—not a pure SDP generator—and its waypoint predictions are learned estimates, not guaranteed optima for each sampled episode. The waypoint-label and training history is documented in aemo_dt_preferred_policy_plan.md.
---

## Module 3 — Training the student (~75–90 min)

**Question to answer:** *What objective does the DT actually minimize, and what is
`return_scale` for?*

**Files / anchors**
- `scripts/pretrain_decision_transformer.py`:
  - `--rtg-source` (line 496) and `--auto-return-scale` (line 507) — read the help
    strings and then the code they trigger
  - the `return_scale` knob plumbing (lines 83, 114, 159)
- `src/transformer_training.py`:
  - `TrajectoryDataset` (line 198) — context windowing, how RTG is built per window
  - `train_decision_transformer` (line 577) — loss terms (action/state/return),
    the `action_loss_weight` defaults, checkpointing
  - `_checkpoint_return_scale` (line 564) and `episode_train_val_split` (line 1477)
- `src/decision_transformer.py`:
  - `DecisionTransformer.forward` (line 449) — inputs (state, rtg, timestep,
    actions, mask) and what it predicts
  - `return_scale` default (line 435) and `load_from_checkpoint` (line 531)
- `src/aemo_decision_transformer.py`? (if present, otherwise skip)

**Verification**
```bash
python3 -m pytest tests/test_pretrain_aemo_decision_transformer.py tests/test_decision_transformer.py -q
```
Then inspect the v2cal training outputs (already on disk):
`models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss.csv` (loss curves),
`..._v2cal_loss_surface_manifest.json` (`run_summary.best_val_*`).

**Write:** 10 lines — "The loss is `w_a·ActionLoss + w_s·StateLoss + w_r·ReturnLoss`;
RTG is scaled by `return_scale` so the 90th percentile of |J_t|/scale ≈ 10; the
`return_scale` value that ships is …; it matters because …"

**This is the B8 trap — make sure you can explain it.** Why did the impact
evaluation once "collapse"? (Loader passed a state dict, so the model kept
`return_scale=1.0` instead of the sidecar value; prompt ~26,000× too large.)
Read `docs/known_issues.md` B8 and `report.md` §8.2.10.

---
**Summary:** Training is a forward pass, loss calculation, backpropagation, and optimizer update. Same training loop as pretraining for LLM; each training batch, the dataset supplies short trajectory windows: observations, actions, RTGs, timesteps, and a mask for padding. the model reads the them as RTG, state and acction tokens and predicts actions with auxiliary state and RTG predictions. The loop compares those predictions with the recorded value using masked mean-square error.
total loss =
    action_loss_weight × action MSE
  + state_loss_weight  × state MSE
  + return_loss_weight × RTG MSE
The current latest model uses 0.999, 0,002 and 0.0001 respectively and can be override
return_scale normalise the RTG before it goes into the transformer. The checkpoint we created should store a checkpoint specific value that is pre-calculated from the training data.
---

## Module 4 — Inference: `rtg_mode="auto"` (~60 min)

**Question to answer:** *At eval time, what prompt does the agent feed the DT, and
how does it change under market impact?*

**Files / anchors**
- `src/decision.py` (`AEMOAgent`, class at line 583):
  - `_env_has_non_identity_impact` (line 692) and `_resolve_rtg_mode` (line 696) —
    the definition of `auto`
  - `_init_jtsoc_table` (line 1590) and `_lookup_jtsoc_rtg` (line 1643) — state →
    RTG lookup
  - `choose_action` (line 1411) — the DT forward path per step
  - `run_episode` (line 1653) — the rollout loop and RTG buffer updates
  - `stable_rtg_update` (imported from `src/grpo_posttraining.py`, line 115) — the
    constant-RTG recurrence and its clamp
- `src/market_impact.py`: `MarketImpactModel` (26), `IdentityImpact` (52),
  `PiecewiseMeritOrderImpact` (69), `create_impact_model` (191)

**Verification**
```bash
python3 -m pytest tests/test_market_impact.py tests/test_autoresearch_evaluator.py -q
```
Artifacts to read (don't rerun): `paper/audit/artifacts/stagec_v2cal_impact_analysis.json`
and `paper/audit/artifacts/cache_data_staleness_finding.md`.

**Write:** 12 lines — trace one identity step and one merit-order step: what
`self.rtg_mode` resolves to, where the RTG value comes from, and what changes under
impact. Then explain why `auto` under impact is a *robustness* choice, not a
collapse fix.

---
**Summary:** the auto rtg mode identify if the sim is in impact mode (which means if the demand will be impact by the action we take) it will use constant rtg rather than dynamic j_t(soc) to prompt the model, switching to j_t_soc will use impact-aware dynamic rtg from J_t(soc)
---

## Module 5 — Evaluation & statistics (~60–75 min)

**Question to answer:** *How does a checkpoint become the headline number, and what
exactly is being resampled in the CI?*

**Files / anchors**
- `scripts/autoresearch_evaluator.py`:
  - config resolution + policies; `load_dt_model` (line ~207)
  - the reference-rollout cache: `_reference_rollout_cache_path` (line ~397) — the
    `model_sha256` + `data_fingerprint` fields we just added
  - `_execute_rollout` (line ~1143) — cache hit vs fresh run
- `paper/audit/reproduce_significance.py` — the method (bootstrap 10k, seed 42,
  paired Wilcoxon over scenario cells)
- `configs/aemo_autoresearch_evaluator.sdp_teacher_{standard_year,dispatch_year,expanded_full}.json`
  and `..._2025_fresh.json` — the actual headline configs

**Verification**
```bash
python3 paper/audit/reproduce_significance.py --out /tmp/sig_check.json
```
Diff against `paper/audit/artifacts/stagec_v2cal_extended_significance.json` (should
match; if not, you've found drift — tell me).

**Write:** 10 lines — "A cell = (scenario, battery). Each policy gets one
deterministic episode per cell. The DT is deterministic; PPO/fcas-rule carry ~1%
noise. The bootstrap resamples **cells**, not episodes. The paired-difference CI is
primary at n<10; Wilcoxon is bounded (n=5 → 0.0625)."

**Make sure you can say why the small-n fix was legitimate** (more cached 2024
windows; no retraining) and why n=30 standard / 12 dispatch / 30 expanded / 6 OOD.

---
**Summary:** The evaluator tests the trained checkpoint on a set of predefined scenario-and-battery cells. In each cell, it runs the DT and reference policy, such as PPO, on the same conditions and records their profits. Subtracting PPO’s profit from the DT’s profit gives a paired difference for that cell. The mean of those differences summarizes the DT’s average advantage across the tested cells. To estimate uncertainty, the analysis repeatedly resamples the paired cell differences with replacement and recalculates their mean; the 2.5th and 97.5th percentiles of 10,000 bootstrap means form the 95% confidence interval. If the interval excludes zero, that supports an average advantage on the tested surface—not a guarantee of winning on every case or on untested markets. The reported extended surfaces contain 30 standard, 12 dispatch-matched, 30 expanded, and 6 OOD cells. Months within the same region may be correlated, so cell count is not the same as independent-region count. The small-n expansion added evaluation cases; it did not retrain the model.
---

## Module 6 — Provenance, leakage, and the ceiling story (~45 min)

**Question to answer:** *Why should anyone believe these numbers, and what is the
one result a reviewer will attack first?*

**Read:**
- `paper/audit/audit_findings.md` and `paper/audit/provenance.tsv` (every headline
  number → artifact)
- `paper/audit/artifacts/leakage_check.txt` (train pre-2024 vs eval 2024/25)
- `paper/audit/artifacts/cache_data_staleness_finding.md` (the stale-cache bug and
  its fix)
- `report.md` §8.2.10 (the within-behaviour-cloning ceiling: what failed and why)
  and §8.2.11 (shipped re-baseline)
- `docs/known_issues.md` — skim A1–A9, B1–B8; know which are fixed vs open

**Write:** the one-paragraph "why believe this" answer, and the one-paragraph
"biggest weakness" answer (hint: simulator-only; the next attack is baseline
strength/tuning).

---
**Summary:** The evaluation is traceable with defined config, fixed checkpoint and saved singificance analysis. Training was on pre 2024 data while evaluation is done on data from 2024. However the biggest weakness is that it is still simulation only.
---

## Synthesis task — the one-pager (do last, closed-book)

Write **one page** titled *"How the headline number is produced, end to end."*
No code open. Target 300–400 words covering: teacher → corpus → student → prompt →
eval → significance → provenance. Then reopen the code and mark every claim you got
wrong or couldn't state. Those marks are your revision list.

---

## Interview drill — 10 hostile questions

Answer each in ≤1 minute, aloud, then check against the artifact. Pointers in
parens.

1. **"How well-tuned is your PPO baseline?"** (eval sweep logs under
   `eval_output/opt_sweep*`, `baseline_*`; `report.md` §8.2.5)
   **answer:** It was tuned with a 10 trial optuna search over a bunch of hyper-parameter such as lr, dicount facotr, model size. then trained for up to 4 million steps on 2021-2023 data across 5 region and the shipped model as teacher all use 9-dim one.
2. **"Isn't distilling the same env's planner circular?"** (honest executor ≠
   foresight; §8.2.10 Stage A/B; leakage check)
   **answer:** the teacher sint clairvoyant - as it see seasonal mean for its own forecast rather than real future RRP; also no train/eval overlap; we also conduct 2025 OOD and impact check to see if student can generalise which it can.
3. **"It's all simulator — does any of this transfer?"** (sim-to-real is the stated
   open item; §9 Phase 4)
   **answer:** The sim result on 2024 and 2025 data shows the model can generalize across helout market period, included test with modeled market impact. while simulation is not the same as real hardware testing, it is resonable to believe it can transfer over (and it is the next stage of the evaluation)
4. **"Why does the DT beat a planner teacher that's supposedly better?"**
   (standalone vs solver-in-the-loop; §8.2.10)
   **answer:** The student doesnt outperform the teacher. it is about transferring useful planner behaviour efficently
5. **"n=30/12/30/6 — is that enough?"** (bootstrap over cells; paired CI primary;
   small-n assessment)
   **answer:** There are supportive evidence but not solid proof, more could improve.
6. **"Your Wilcoxon p is bounded at small n."** (n=6 → min 0.031; explain sign)
   **answer:** Yes, given there are only 6 testing (n=6 -> p ~ 0.031). there is a lower bound of p, and all paired diffs share a sign so what we got is the strongest attainable. the bootstrap pair-difference CI is the primary evidence.
7. **"You changed the physics and the numbers moved a lot."** (A1–A8; v1 vs v2cal;
   own it)
   **answer:** Error were found in the degradation model and planner's wear calculation. so it was corrected, and new data was regenerated and used to retrained the model. all evaluation was reran. evaluation surfaces were expanded and stale reference caches were fixed, so not every shift in the headline numbers is caused by physics alone.
8. **"Why not IQL/CQL — why decision transformer at all?"** (§8.2.10 positioning;
   FUTURE_PLAN 2.2)
   **answer:** DT is trained with offline data and gives sequence modeling plus RTG-conditioned controllability at inference; IQL/CQL are the natural offline-RL alternative and are planned comparison.
9. **"What are the market-impact model's assumptions?"**
   (`PiecewiseMeritOrderImpact`; price-taker vs endogenous)
   **answer:** Without impact, the battery is treated as a price-taker: it uses the historical RRP and FCAS prices, and its own dispatch doesn’t change them. With impact enabled, the model approximates an endogenous effect: charging adds demand and can raise the energy price; discharging adds supply and can lower it. FCAS prices are reduced according to the battery’s share of an estimated market depth. This is a simplified, per-interval market model—not a full AEMO market re-clear.
10. **"You said PPO isn't reproducible — is your comparison fair?"** (~1% noise;
    wide CIs; fixed cache keys)
    **answer:** The extended runs used fresh caches and consistent data/physics, and the paired confidence intervals excluded zero on all four reported surfaces. PPO did show small run-to-run variation on a long-episode probe (~0.7%), so the result isn’t bit-for-bit reproducible—but that’s different from the comparison being unfair. I’d describe it as strong comparative evidence with a reproducibility caveat, not definitive proof that PPO will always lose.
**Pass bar:** no note-lookups, and for each you cite a concrete artifact.

---

## Self-assessment gate (all must be YES before leaving Phase 2)

- [ ] I can explain what `J_t(soc)` is, its units, and how it's built
- [ ] I can state exactly what data the teacher sees (and doesn't)
- [ ] I can write the training loss and say what `return_scale` does
- [ ] I can explain `rtg_mode="auto"` identity vs impact, with the code path
- [ ] I can explain the B8 bug and why it reversed a published claim
- [ ] I can name the six deterministic-vs-nondeterministic facts
- [ ] I can describe the leakage check and its one caveat
- [ ] I can explain why the small-n fix needed no retraining
- [ ] I can list every headline number and its artifact path
- [ ] I can answer all 10 hostile questions from memory
- [ ] My one-pager matches the code on re-read
- [ ] I can say which parts were AI-written and what I personally verified

---

## When you're done

Tell me, and we'll move to **Phase 3 — scope lock + LaTeX onboarding** (choose the
paper's 3 claims, pick the template, and set up the Overleaf skeleton). If questions
come up mid-module, ask with the file + line number and I'll help — but the notes and
the answers stay yours.
