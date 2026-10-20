# Anticipated reviewer questions — short answers

Prepared 2026-10-20. Answers are deliberately short; the bracketed pointers say
where the evidence lives. Use these to pre-empt the obvious objections.

## Method and claims

**Q1. DTs usually only match offline RL. Why does yours beat PPO?**
The student is not a pure behaviour-cloner: it is trained on planner-generated
trajectories and conditioned on a planner-derived value prompt. PPO is tuned in
the same simulator; the DT has 210-step context plus an explicit value signal.
The claim is empirical (four surfaces, paired CIs), not a general theory.
[§4, §5, Table 1]

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
Optuna search (10 trials), up to 4M steps, same scenarios/caches, same 5-min
steps and 9-dim action. It varies ~1% run-to-run while the DT is deterministic;
the paper therefore makes the paired-difference CI primary. [§5, §7]

**Q6. Small samples (n=6, n=12) — underpowered?**
Paired bootstrap over matched cells (10k replicates); all four paired CIs
exclude zero. Wilcoxon is reported as indicative, and cell correlation is
flagged. [§5, Table 1]

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

**Q10. The market-impact model is an approximation — is the gate meaningful?**
It is a per-interval piecewise merit-order model, not a full market re-clear. It
tests robustness to endogenous prices; the DT wins 9/9 cells with a paired
advantage of $167,099 (CI [101,213, 227,362]). [§3.3, §6]

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
