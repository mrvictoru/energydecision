# FUTURE_PLAN.md — AEMO Track Forward Plan (2026-08-25+)

> This is the **living forward plan** for the AEMO/NEM track.
> It supersedes and consolidates the open items from:
> - `docs/aemo_research_plan.md` (historical — now **STALE**)
> - `docs/aemo_dt_preferred_policy_plan.md` (completed — now **ARCHIVED**)
> - `README.md` roadmap checklist
>
> Status key: ⬜ = open, 🟡 = in progress / partially measured, ✅ = done, 🔴 = on hold, 📝 = writing / meta.

---

## 0. Context & Positioning (for PhD narrative)

**What we have:** A standalone Decision Transformer (Stage C, SDP-distilled, `rtg_mode="auto"`) that **beats PPO on all 4 identity surfaces + the market-impact gate** ($11.6k vs $2.35k standard; $35.3k vs $22.5k dispatch-matched; $34.8k vs $19.5k expanded 2024; $25.9k vs $6.5k 2025 OOD; 2.5–3.1× impact resilience). Statistical rigor applied (bootstrap CIs, paired Wilcoxon, all six DT-vs-PPO CIs exclude zero).

**What we don't have (the open problems):**

| # | Problem | Why it matters for a PhD |
|---|---|---|
| 1 | **Sim-to-real gap** — all results simulator-based | The #1 credibility gap; a safety-wrapper + real settlement validation is the path to "deployable" |
| 2 | **Broad-surface FCAS under-bidding** — DT $4.8k vs PPO $10.2k on 5-min expanded 2024 | The behaviour-cloning ceiling is real; offline data quality is the binding constraint |
| 3 | **j_t_soc impact failure** — price-taking cost-to-go collapses at grid scale under merit-order | Open algorithmic problem: impact-aware cost-to-go or surface-aware mode gating |
| 4 | **Oracle_MI fixed-point artifact** — >100% PT at 150 MW+ | Numerical robustness for the impact-aware ceiling |
| 5 | **Full_fcas broad surface** — protocol asymmetry means current expanded eval uses 3-dim actions | Quick win to close a disclosed limitation |
| 6 | **Multi-agent NEM** — no learned multi-BESS interaction | Genuine research frontier with your impact model |
| 7 | **Offline-Q (IQL/CQL) vs planner-distillation** | Never tried; clean methods comparison on same ceiling |

---

## 1. Sim-to-Real Readiness (Priority 1 — PhD Flagship)

| ID | Task | Target | Notes |
|---|---|---|---|
| 1.1 | **Settlement-grade backtest** — compare simulator revenue against AEMO settlement data for Dalrymple North (same 2024 windows) | Close sim-vs-real revenue gap; produce calibration curve + uncertainty bounds | Data exists; need AEMO settlement API or manual CSV match |
| 1.2 | **Safety wrapper / constrained policy** — `ConstrainedAEMOAgent`: profit s.t. degradation budget ≤ B, SOC ∈ [guardrail], FCAS enablement ≤ cleared limits | Move from "reward shaping" to hard constraints; CMDP formulation; Certify on expanded surface | Reuses J_t(soc) as cost-to-go; add barrier/penalty functions |
| 1.3 | **Robustness benchmark suite** — adversarial price paths, held-out years (2022, 2023), FCAS-spike regime stress, market regime shift (pre/post-2024 rules) | "Robustness under regime shift" as first-class evaluation axis | Extend `scripts/regime_shift_eval.py` |
| 1.4 | **Hardware-in-the-loop feasibility** — partner with a BESS operator or use public FCAS enablement logs for closed-loop replay validation | Even a single pilot trace with real telemetry > "sim only" | Optional but high impact |

---

## 2. Broad-Surface FCAS Gap — Breaking the Behaviour-Cloning Ceiling

| ID | Task | Target | Rationale |
|---|---|---|---|
| 2.1 | **Full_fcas broad-surface evaluation** — run `configs/aemo_autoresearch_evaluator.expanded_rtg10.json` with `action_mode=full_fcas` (9-dim) for DT & PPO | Close disclosed protocol asymmetry; current expanded uses 3-dim `multi_market` | Quick win; closes disclosed limitation in report §6 |
| 2.2 | **Offline-Q baseline (IQL/CQL)** — implement IQL on the 2,425-ep FCAS-rich corpus; compare to planner-distillation on same data | Clean methods paper: planner-distillation vs offline-Q on same ceiling | Never tried in repo; clean comparison on same data |
| 2.3 | **Spike-risk conditioning (Exp 6 revisit)** — add distributional spike-risk token (volatility + time-of-day + headroom) to modern v2; test on expanded 2024 & 2025 OOD | Exp 6 deferred in plan; TTM point forecasts failed because FCAS spikes are unpredictable — risk scalar may work | Low effort if generator from Phase 6 usable |
| 2.4 | **Reward-weighted regression** — upsample high-FCAS-revenue trajectories in loss (PPO best rollouts, FCAS-heavy policies) | Data-side lever; "better offline data" without new generation | Low effort; `transformer_training.py` supports sample weighting |
| 2.5 | **FCAS-spike period dataset** — curate episodes from high-FCAS months (Nov 2024, May 2024, Sep 2024) for targeted retrain | Targeted data quality > brute-force volume | PPO FCAS $35.5k in Nov vs DT $3.5k — that's the gap |

---

## 3. Impact-Aware Policy & j_t_soc Failure Mode

| ID | Task | Target | Status |
|---|---|---|---|
| 3.1 | **Impact-aware J_t(soc) table (H1)** — recompute cost-to-go with `realized_energy_price` (post-impact) inside `compute_cost_to_go_table`; validate on `stagec_jtsoc_dispatch` + impact surface | Fix the price-taking over-prompt that collapses Hornsdale/Torrens under merit-order | Open — H1 in preferred policy plan |
| 3.2 | **Surface-aware mode gating (H3)** — `rtg_mode="auto"` already ships; validate fallback RTG=0.0 is conservative enough; sweep fallback RTG on impact surface | Already shipped; just needs validation sweep | 🟡 Partial (shipped, needs validation) |
| 3.3 | **Oracle_MI numerical fix** — the fixed-point solve exceeds 100% PT at 150 MW+ (unreliable ceiling). Diagnose: supply-curve step granularity? convergence tolerance? | Reliable impact-aware ceiling for eval | Open |
| 3.4 | **Dynamic RTG prompting** — match training-time decaying RTG at inference (start at target, discount per step). Test fixed vs decaying RTG on expanded 5-min surface. | RTG-distribution finding in research plan §RTG-distribution | Open |

---

## 4. Research Publications & Dissemination (PhD Portfolio)

| ID | Target | Venue | Status |
|---|---|---|---|
| 4.1 | **Benchmark paper**: "A Degradation-Aware Benchmark for Multi-Market BESS Dispatch in AEMO/NEM" — Stage C result + evaluation protocol + statistical rigor | NeurIPS Datasets & Benchmarks / ACM e-Energy / IEEE Trans Smart Grid | 📝 Writing |
| 4.2 | **Method paper**: "Planner-Distilled Decision Transformers Break the Behaviour-Cloning Ceiling in Multi-Market BESS Dispatch" — SDP distillation + J_t(soc) + impact gate | ICML / NeurIPS / ICLR | 📝 Writing |
| 4.3 | **Multi-agent NEM paper** — "Learned Market Power: Multi-BESS Coordination under Merit-Order Impact" — MARL with your impact model | ICML / NeurIPS / AAMAS | ⬜ Proposed PhD direction |
| 4.4 | **Offline-Q vs Planner-Distillation** — IQL/CQL vs SDP-distillation on same FCAS-rich data + impact surface | ICML / NeurIPS | ⬜ Proposed |
| 4.5 | **Website as academic project page** — add BibTeX, citation block, "applying for PhD at X" framing | Project page | 📝 Writing |

---

## 5. Engineering / Rigor / Reproducibility

| ID | Task | Priority |
|---|---|---|
| 5.1 | **Artifact provenance** — checksums + config logging for datasets, models, eval outputs (roadmap item) | High |
| 5.2 | **Statistical confidence on Stage C headlines** — bootstrap CIs + paired Wilcoxon already done via `scripts/stagec_statistical_significance.py`; ensure CI reported in report.md §8.2.10 | ✅ Done (2026-08-23) |
| 5.3 | **Long-context DT sweep** — ctx=288/576/1008/2016 on modern v2 8×768 with FCAS-rich data (ctx=2016 now feasible on 22 GB) | Medium |
| 5.3 | **Multi-agent extension** — PettingZoo parallel `SolarBatteryEnv` + AEMO equivalent; curriculum from single to multi | PhD direction |
| 5.4 | **Per-region RRP generator** — unblocks Phase 6/7 synthetic FCAS; needed for realistic long-horizon synthetic data | On hold |

---

## 6. Quick Wins (1–2 days each)

| ID | Task |
|---|---|
| Q1 | **Citation/BibTeX block on website** — add to `index.html` footer + `report.md` |
| Q2 | **Full_fcas broad-surface eval** (Task 2.1) |
| Q3 | **Artifact provenance script** — `scripts/log_artifact.py` (sha256 + config JSON + git SHA) |
| Q4 | **Website: add BibTeX block + "Applying for PhD at [X]" framing** |
| Q5 | **RTG decay vs fixed sweep** on expanded 5-min surface (Task 3.4) |

---

## 7. Deprioritized / On Hold (with rationale)

| ID | Task | Rationale |
|---|---|---|
| D1 | Synthetic FCAS + impact combine (Phase 6/7) | v2 diffusion dead end; per-region RRP generator needed first |
| D2 | Full-PPO value-critic fine-tune | Tested — flat/negative on modern v2; GRPO doesn't beat pretrained |
| D3 | Multi-round GRPO self-improvement | Deferred — GRPO doesn't beat pretrained |
| D3 | Long-context DT experiments (de-prioritized) | ctx=2016 feasible but not the binding constraint; data quality is |
| D4 | Offline dataset sensitivity studies (de-prioritized) | 2026 session showed re-composition only +23% FCAS; data ceiling, not mixture |
| D5 | Forecast DT follow-up | Negative result; point forecasts ~0 FCAS corr; contingent on better generator |

---

## 8. Traceability to Existing Artifacts

| Artifact | Status | Notes |
|---|---|---|
| `docs/aemo_research_plan.md` | **STALE** (historical — Option C decision was its endpoint) | Marked stale in this plan |
| `docs/aemo_dt_preferred_policy_plan.md` | **ARCHIVED** (completed — Stage C shipped) | Marked archived; final session 2026-08-23 |
| `README.md` roadmap | **SYNCED** (below) | Updated in this PR |
| `report.md` §8.2.10 / §9 | **CURRENT** | Source of truth for Stage C result |
| `docs/website_requirements.md` | **CURRENT** | Drives `index.html` |
| `docs/FUTURE_PLAN.md` | **THIS FILE** | Living forward plan |

---

## 9. PhD Application Alignment

**Research statement framing:**

> "I built a planner-distilled offline Decision Transformer that beats online RL and real operators on a degradation-aware, market-impact-aware NEM battery dispatch benchmark — with full statistical rigor. The three open problems I propose for PhD research are: (1) closing the sim-to-real gap via settlement-grade backtesting and safety-constrained policies, (2) breaking the behaviour-cloning ceiling for FCAS spike bidding via offline-Q / risk-conditioning / targeted data, and (3) multi-agent coordination of BESS fleets under endogenous market impact."

**Publication pipeline:** Benchmark paper (4.1) → Method paper (4.2) → PhD proposal multi-agent (4.3).

---

## 10. Dependencies & Blockers

| Blocker | Depends on | Mitigation |
|---|---|---|
| Sim-to-real data | AEMO settlement access / operator partner | Public dispatch replay already 80% there; calibration study without settlement still valuable |
| Multi-agent | PettingZoo integration + impact model scaling | Start with 2-agent deterministic game; scale later |
| IQL/CQL impl | New `src/` module (outside autoresearch surface) | Implement in research branch, not constrained loop |
| FCAS-spike dataset | Episode tagging from existing logs | Already have scenario labels; just filter + curate |

---

## 11. Timeline (Indicative)

| Horizon | Focus |
|---|---|
| **0–1 month** | Q1–Q4 quick wins; website + citation block; full_fcas expanded eval; paper drafts |
| **1–3 months** | Settlement backtest (1.1); safety wrapper design (1.2); IQL/CQL impl (2.2); benchmark paper submission |
| **3–6 months** | Safety wrapper implementation + evaluation; spike-risk conditioning (2.3); multi-agent prototype |
| **6–12 months** | Multi-agent NEM paper; sim-to-real pilot if partner secured; PhD thesis chapters |

---

*This plan is the single source of truth for forward work. Update it, don't create new plan files. All prior plan files in `docs/` are now either STALE or ARCHIVED.*