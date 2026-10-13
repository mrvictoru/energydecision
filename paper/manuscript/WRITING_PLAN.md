# Writing plan (Track A) — detailed, section by section

**Goal:** a ~9-page arXiv preprint built from `paper/manuscript/main.tex`, following
the 3-claim arc (C1 ceiling → C2 distillation → C3 impact gate).

**Rules (repeat):**
1. Never type a number from memory — copy from `paper/audit/provenance.tsv` /
   the artifact it names.
2. Every paragraph should serve C1, C2, or C3 (or be setup for one). Cut the rest.
3. Compile after every section. Write Limitations yourself, unprompted.
4. Verify every citation in `references.bib` before submission.

**Writing order (do NOT go top-to-bottom):**
Method → Experiments → Results → Market-Impact → Limitations → Introduction →
Related Work → Conclusion → Abstract → Appendix.

**Sync workflow:** write in Overleaf; at the end of each session:
Download `neurips_2026.tex` → overwrite `paper/manuscript/main.tex` → commit.
Or ask me to sync. Compile locally-readable backups monthly.

---

## §4 Method (~1.5 pages) — write FIRST

**Claim served:** C2 (the method). **Purpose:** the reader must be able to rebuild
the pipeline from this section alone.

### 4.1 Honest SDP teacher (3 paragraphs)
- **P1 — what the teacher sees.** Inputs are cached preprocessed 5-min AEMO data
  (RRP, total demand, 8 FCAS prices, solar/wind mix). Forecast = seasonal
  **month × hour mean** RRP built from pre-2024 data only. State explicitly: *no
  realized future prices* enter the loop (this is what "honest" means). Cite
  `src/aemo_sdp_executor.py` (`build_seasonal_rrp_profile`, `build_rrp_forecast`).
- **P2 — the three-stage hierarchy.** (1) SOC-waypoint DT predicts K=8 target-SOC
  waypoints; (2) SDP backward induction plans energy between waypoints under the
  forecast, with a degradation-aware per-step wear estimate scaled by
  `deg_calibration`; (3) a greedy current-price FCAS bidder allocates residual
  headroom. Say plainly that stage (1) is a learned helper, not an optimizer.
- **P3 — J_t(soc) cost-to-go.** The free-terminal recursion yields the remaining-cost
  table J_t(soc) over (time, SOC); this becomes the RTG prompt.
- **Pitfall:** don't call it "a pure SDP solver" — it's the hybrid pipeline (the HF
  card previously got this wrong; use the corrected wording).

### 4.2 Trajectory corpus (1 paragraph + schema sentence)
- Random windows: 5 regions × {short ≈ 14 d, medium ≈ 8 wk} × 4 battery sizes × 8 eps.
- Each row: self-consistent `(norm_obs 18-dim, action 9-dim, reward)`.
- Shipped corpus: 320 episodes / 3.13 M rows. Cite `generate_sdp_dt_trajectories.py`.
- State that no timestamps are stored (relevant to the leakage caveat).

### 4.3 Student DT (1 paragraph + equation)
- Architecture: 8 blocks × 768, GQA (12 Q / 6 KV), QK-Norm, RoPE, SwiGLU, ctx 210,
  mixed action head (tanh energy, sigmoid FCAS).
- Loss equation (display it):
  `L = w_a·MSE_a + w_s·MSE_s + w_r·MSE_r`, with `w_a=0.999, w_s=0.002, w_r=0.0001`.
- `return_scale` auto-calibrated so p90(|J_t|)/scale ≈ 10; shipped 23,284.83.
- **Pitfall:** quote the SHIPPED weights (0.999/0.002/0.0001), not the function defaults.

### 4.4 Return-to-go prompting and \texttt{auto} (1 paragraph)
- RTG = −J_t(soc), looked up per step from the table.
- `auto`: resolves to `j_t_soc` on price-taking surfaces and to a constant `rtg0.0`
  under market impact; explain it's a robustness choice (impact-aware prompts can
  over-trade energy), not collapse-avoidance.

---

## §5 Experiments (~0.75 page)

**Claim served:** setup for C2/C3.
- **Surfaces (with n):** standard 2024 (5 regions × 6 periods, n=30); dispatch-matched
  (SA1 full-year, n=12); expanded broad-2024 (n=30); 2025 OOD (n=6). One sentence each.
- **Baselines:** tuned PPO (Optuna, 10 trials, ≤4 M steps), `fcas_rule`, dispatch replay.
- **Protocol:** 5-min steps; one deterministic episode per (scenario, battery) cell;
  bootstrap 10k over cells + paired Wilcoxon; **paired-difference CI is primary**.
- **Reproducibility caveat:** DT deterministic; PPO/`fcas_rule` ~1% run-to-run on long
  episodes (cite the determinism probe). One sentence.

### 5.1 Headline Results (~0.5 page + Table 1 already present)
- Reference Table 1; state all four paired CIs exclude zero.
- Add the **retention** framing: the student keeps 51–91% of the teacher (fidelity),
  while beating PPO (competitiveness). Keep the two comparisons distinct.
- Note the surfaces are extended (more windows) vs earlier small-n single-month runs.

---

## §6 Market-Impact Gate (~0.75 page) — C3

- Describe the piecewise merit-order impact model and the 9 grid-scale cells
  (3 batteries × 3 scenarios), identity vs endogenous.
- Result: fixed `rtg0.0` fallback beats PPO on **9/9** cells; per-battery ratios
  small 4.36× / hornsdale 4.71× / torrens 4.67×; paired Δ +167,099, 95% CI
  [101,213, 227,362], p=0.0039 (cite `stagec_v2cal_impact_analysis.json`).
- Takeaway sentence: value-based prompts must be gated by market power → `auto`.

---

## §7 Limitations (~0.5 page) — write this yourself

Cover, unprompted:
1. Simulator-based (sim-to-real is the top open item).
2. Same-simulator teacher + evaluation (residual circularity; mitigated by 2025 OOD + impact).
3. Baseline strength/noise: PPO is tuned but retains ~1% noise; DT deterministic.
4. Small n on some surfaces; cells within a region are correlated.
5. Physics sensitivity (v1 → v2cal moved numbers); the stale-cache incident + fix.
6. Retention ≤ 100%: the student is not better than its teacher.

---

## §1 Introduction (~1 page) — write after the body

- P1: why multi-market BESS dispatch is hard (energy + 8 FCAS co-optimisation,
  degradation, online exploration is slow/risky) → offline learning is attractive.
- P2: the catch — behaviour cloning is bounded by logged behaviour (**C1**).
- P3: contribution bullets (3): C1 diagnose the ceiling + which levers fail;
  C2 planner distillation + J_t(soc); C3 market-impact gate.
- P4: results preview with [FACT] numbers (one sentence), plus the sim-to-real caveat.

## §2 Related Work (~0.75 page) — four paragraphs
1. Decision Transformer / offline RL sequence modeling.
2. Offline-Q alternatives (IQL/CQL) — name them as the natural comparison (future work).
3. Planner distillation / imitation learning (DAgger).
4. BESS scheduling, degradation-aware benchmarks, NEM market.

## §8 Conclusion (~2 paragraphs)
Restate C1/C2/C3; one-line takeaway ("most of the planner, none of the solver");
next step (sim-to-real / shadow deployment).

## Abstract (150–200 words) — write LAST
Problem → method → [FACT] numbers (n=30/12/30/6, all CIs exclude zero, 9/9 impact) →
one honest caveat (simulator-based).

## Appendix A — Provenance and Reproducibility
Point to `paper/audit/provenance.tsv`, `paper/environment.txt`, tag `preprint-v1`,
the HF artifacts (model/dataset/waypoint helper), the leakage check, the
determinism probe, and the 387-passing test suite.

---

## Definition of done (Phase 5)
- [ ] All `% TODO(author)` blocks replaced with prose
- [ ] Every number matches `paper/audit/provenance.tsv`
- [ ] Every citation verified against the original
- [ ] Limitations written unprompted
- [ ] Figures replaced from `\figph{}` (Phase 4) and referenced in text
- [ ] Compiles cleanly on Overleaf
- [ ] AI-assistance disclosure in the `ack` block
