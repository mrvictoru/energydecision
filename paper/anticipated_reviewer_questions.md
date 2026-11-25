# Anticipated reviewer questions — short answers

Prepared 2026-10-20. Answers are deliberately short; the bracketed pointers say
where the evidence lives. Use these to pre-empt the obvious objections.

## Method and claims

**Q1. Does the DT robustly beat PPO?**
Not uniformly. The corrected comparator is three seeded 4M-step PPO models tuned
on pre-2024 validation data. The DT has higher within-cell three-seed mean on all
four surfaces, but dispatch's CI includes zero and PPO seed 1 reverses three
surface means. The paper's claim is empirical competitiveness plus baseline
sensitivity, not universal superiority. [§5, Table 1, PPO retraining audit]

**Q2. Is "no solver in the control loop" honest if you compute a cost-to-go table?**
Yes. The $J_t(\mathrm{soc})$ table is a precomputed/cacheable function of the
seasonal forecast and the SOC grid; it is one backward-induction pass per
region/season/episode, not per decision. At deployment each step is a forward
pass plus a constant-time lookup. [§4.1, §4.4, Fig. 2]

**Q3. The teacher is called non-clairvoyant but uses a forecast. What is restricted?**
It uses month x hour seasonal means from pre-2024 data only; no realised future
prices. FCAS bidding uses current prices only. Example: NSW1 October seasonal
profile is about A$77/MWh at 07:00 and A$184/MWh at 18:00. [§4.1]

**Q4. Retention exceeds 100% on expanded/OOD — how can a student beat its teacher?**
Different surfaces and a teacher whose seasonal planning generalises worse there.
Per-cell ratios are noisy. The paper reports retention as surface-dependent and
does **not** claim uniform superiority. [§5.1, §7, `teacher_retention.json`]

## Baselines and statistics

**Q5. Is PPO strong and tuned fairly?**
The old 204,800-step default checkpoint was not. The corrected study uses ten
Optuna trials on a chronological pre-2024 validation split, then three final seeds
for 4,003,840 steps each on the same 5-minute, 9-action environment. Seed 0 is the
pre-specified median-validation checkpoint; the paper also reports all seeds and
the within-cell three-seed mean. [§5, §7, ppo_retrain_20261125.md]

**Q6. Small samples (n=6, n=12) — underpowered?**
Yes for a robust policy ranking. The corrected table has three CIs excluding zero
and dispatch including zero; only three PPO seeds are available, and the cell
bootstrap is conditional on those checkpoints. This limitation, cell correlation,
and the need for additional seeds are explicit. [§5, Table 1, §7]

**Q7. Why not compare with IQL/CQL?**
Scope: the diagnosis concerns the data source, not the learner. A direct
IQL/CQL comparison on the same corpus is listed as future work. [§2, FUTURE_PLAN 2.2]

**Q8. `auto` falls back to constant RTG under impact — is that post-hoc?**
No: it is a declared surface-aware rule (state-dependent prompt when roughly
price-taking; conservative constant when the battery can move prices), evaluated
as shipped. The paper explicitly does not claim one prompt is best everywhere.
[§4.4, §6, FUTURE_PLAN 3.1]

## Validity and threats

**Q9. Teacher and evaluation share the simulator — is this circular?**
Acknowledged in Limitations. Held-out 2025 data and the endogenous-price impact
gate mitigate it, but circularity is reduced, not removed. [§7]

**Q10. The market-impact model is an approximation — is the result decisive?**
No. It is a useful per-interval piecewise-merit-order stress test, not a full
market re-clear. Against the within-cell mean of three PPO seeds, the DT wins 8/9
cells (mean advantage $79,736), but seed-specific DT/PPO ratios range from 0.91x
to 2.96x. The paper therefore does not call it a universal gate pass. [§3.3, §6]

**Q11. Is the 8x768 architecture the reason for the result?**
Not established. Within-cloning architecture changes did not break the ceiling,
and the paper does not claim architecture optimality. Controlled architecture
comparisons are future work. [App. A, conclusion]

**Q12. Does it transfer off the simulator / to other markets or chemistries?**
Open. Results are simulator-based on the NEM with an LFP degradation preset;
sim-to-real and broader chemistry/market coverage are future work. [§7, §8]

## Presentation and reproducibility

**Q13. Why 5-minute steps?**
They match the DT training corpus and the AEMO dispatch interval. PPO is
step-agnostic, so this is conservative for the DT. [§3.1, §5]

**Q14. Can I reproduce it?**
Code, corpora, and weights are released; every headline number maps to
`paper/audit/provenance.tsv`; the environment is pinned in
`paper/environment.txt`; the suite has 387 tests at the release tag. [App. C]

**Q15. "Break the ceiling" sounds overbroad.**
The text scopes it to the tested data sources and levers (several places say
"tested"), and the abstract frames the ceiling as a diagnostic encountered en
route to the planner-distilled policy. [Abstract, §1, App. A]
