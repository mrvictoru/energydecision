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

## 6b. Household Track — Modern Real-Data Rebuild (Priority 2, parallel with AEMO)

> **Motivation:** the Ausgrid benchmark (2010–2013, 300 homes, half-hourly) predates
> the collapse of Australian feed-in tariffs (~20¢ → ~5¢/kWh) and the rise of small
> home batteries. The economics are arguably obsolete. We have access to a real
> household (solar + small home battery) telemetry from **2019 onward**, downloadable
> weekly as high-resolution CSVs from an online portal.

### Why this is easier than AEMO

| AEMO element | Household equivalent |
|---|---|
| FCAS co-optimization (9-D coupled action) | **Gone** — 1D dispatch action, no headroom coupling |
| Price forecasting / SDP MC scenarios | **Trivial** — ToU tariffs are deterministic schedules; uncertainty shifts to solar/load |
| Dalrymple dispatch replay | **The household's actual battery operation** — real ground truth for the replay-gap study |
| Planner distillation + J_t(soc) + CI discipline | Ports directly — `src/sdp_algorithm.py` / MRDP already support the household env |

### Research questions

1. **Does the Ausgrid-era result survive a decade?** Retrain/evaluate old baselines on 2019+ data under modern tariff economics. Expectation: the old DT's learned behaviour does not transfer — a motivating distribution-shift finding.
2. **How suboptimal is real home-battery operation?** Replay the household's actual battery actions vs optimized policies on identical windows → "real households leave $X/year on the table" (household analog of the Dalrymple-North comparison, with ground truth).
3. **Does planner distillation transfer across scales?** Port the full Stage C recipe: SDP-teacher trajectories → standalone DT → J_t(soc)-style cost-to-go prompting (exact here, since ToU is deterministic). Compare against cloning-era DT on the same data.
4. **Degradation realism at home scale.** Small LFP battery cycled daily; high-resolution data enables sharper rainflow counting and calendar+cycle aging (`RealWorldBESSDegradationModel`) — relatively more important than in grid-scale arbitrage.

### Phased checklist

| Phase | Task | Notes |
|---|---|---|
| ✅ **H0** | **Data pipeline**: SMA 'Energy balance - Day' parser (`src/household_ingest.py`) — 12-h dotted clock, [W]/[kW] variants auto-scaled, battery SOC/power channels preserved; year dataset builder (`build_year_dataset`): merge/dedupe/convert kW→kWh + day-ahead persistence `FutureSolar`/`FutureLoad`; privacy-anonymized manifest. **First real year ingested: 365/365 days, 105,108 rows, zero gaps** (2025-08-25 → 2026-08-24, VPP household). Env smoke-tested end-to-end on the full year (12-D obs, 5-min steps inferred correctly). | Done |
| ✅ **H0** | **Privacy guardrails**: raw + normalized telemetry gitignored under `data/household/real/`; only the anonymized manifest (`sma_<date>` identities, sha256 checksums) is committed; privacy leak test enforced. Protocol: `docs/household/real_data_protocol.md`. Data source is a **VPP-coordinated** battery (grid-charging schedule) — noted for H3 interpretation. | Done |
| ✅ **H0** | **Gap & disconnection guards**: raw exports contain month-scale holes (renovation disconnection) and offline periods log as hard zeros, not NaN. Defenses: `find_gap_boundaries`/`split_segments` (>90-min timestamp jumps → contiguous episodes with `SegmentID`, DST-passing threshold); seam-row kW→kWh conversion capped at nominal step; `drop_dead_runs` removes sustained all-zero stretches (HouseLoad AND SolarGen == 0 for ≥2h); per-file `exact_zero_rows` (sustained) and `suspect_zero_rows` (isolated ≤10-min all-zero runs flanked by normal data, interpolated as dropped samples) manifest stats. Ingest CLI supports `--start-date`/`--end-date` range filtering. Env must always run per-segment. Corpus: 1,153 days (2023-02-24 → 2026-08-24), 319,170 rows, 5 segments; 105 isolated zero-rows across 101 files interpolated. | Done |
| 🟡 **H1** | **Re-establish benchmark**: rerun rule / oracle / SB3 PPO / current DT on the real-year dataset (env verified working); define surfaces — train vs OOD splits, seasonal splits | In progress — env ready, baselines next |
| ✅ **H1.5** | **Synthetic diverse-household generator** — see detailed plan below | Core generator, G1–G6 gates, env-view export, OOD holdout, and reproducible corpus CLI implemented; optional Granite TTM adapter remains separately provisioned |
| ⬜ **H2** | **Port the AEMO playbook**: trajectory collection (rule/SDP/PPO) → SDP-teacher distillation → standalone DT → cost-to-go prompting → eval tiers with bootstrap CIs | Reuse `TrajectoryDataset`, trainer, evaluator patterns verbatim |
| 🟡 **H3** | **Replay-gap analysis**: deterministic 5-minute optimizer now compares recorded VPP actions with re-optimized dispatch per contiguous real day. At 5 kWh / 3.3 kW / 0.80 RTE: the observed-to-optimal gap is annualized **$214** under flat pricing and **$355** with free 11:00–14:00 pricing (95% bootstrap CIs documented in `docs/household/replay_battery_size_analysis.md`). | Spot pass-through remains pending a time-aligned retail spot-price series |
| 🟡 **H3** | **Tariff-policy experiments**: flat and realistic free-window ToU sweep implemented in `scripts/evaluate_household_tariffs.py`; prices are re-derived rather than using stored 0.30/0.05 defaults. | Add observed retail spot-price input before claiming a spot pass-through result |

### H1.5 — Synthetic diverse-household generator (detailed plan)

> **Problem:** one real household cannot provide behavioral diversity
> (occupancy patterns, appliance stocks, solar/battery sizing). Policies
> trained only on it will not generalize. Goal: a controllable corpus of
> synthetic households recomposed from REAL components, validated against
> real statistics.

#### Architecture: statistical recomposition (primary), TTM as auxiliary

Generative LLM/diffusion approaches (TimeGAN, Diffusion-TS) are rejected:
finicky to train, weak control over semantics, no guarantee samples stay
physically consistent. Bootstrap-from-real keeps every sample traceable.

```
synthetic household = archetype(load profile)
                    × season/day-type resampling weights
                    × injected appliance blocks (EV / AC / pool)
                  + residual noise (TTM-imputed or bootstrap)
with roof = f(solar scaling factor), battery = g(capacity, flow)
```

1. **Archetypes** (occupancy/behavior classes, each parameterized from the
   real data's daily-profile clusters):
   - `retiree-low`: flat low load, early peak, minimal evening spike
   - `family-ev`: double peak + 7–14 kWh overnight/evening EV charge,
     random weekdays-only or daily
   - `ac-heavy`: summer afternoon duty-cycled AC blocks (2–5 kW),
     temperature-driven frequency
   - `wfh-daytime`: elevated daytime baseline, midday peaks
   - `shift-worker`: inverted schedule (overnight activity)
2. **Day resampling**: for target (season × weekday/weekend × archetype),
   bootstrap whole days from the real corpus's matching cluster, then scale
   by household-size factor λ ∈ [0.4, 3.0]. Whole-day resampling preserves
   realistic intra-day autocorrelation (row-wise i.i.d. noise destroys it).
3. **Appliance injection**: add stochastic blocks ON TOP of resampled days
   (EV start time ~ N(18h, 45m) truncated, duration from battery-size draw;
   AC duty cycle Markov chain conditioned on hour & season). Injection is
   additive on HouseLoad; never negative-clipped silently.
4. **Solar synthesis**: reuse the real solar shape scaled by installed kW
   (3–15 kW) × orientation derate (0.75–1.0); optionally swap in TTM
   weather-residual variation so different years feel like different weather.
5. **Battery assignment**: capacity ∈ {5, 7, 10, 13.5, 20} kWh, flow ∈
   {3.3, 5, 7} kW — env already parameterizes this.
6. **TTM role (auxiliary, optional)**: `ibm-granite/granite-timeseries-ttm-r2`
   (few-M params, CPU-fine) for (a) plausible imputation of the Feb–Jun 2024
   renovation gap (labeled SYNTHETIC if used in training), (b) weather-driven
   residual generation on top of bootstrapped profiles. TTM is NOT the
   primary generator — it reproduces its context window; it does not invent
   new households.

#### Validation gate (all gates must pass before a synthetic day is accepted)

| Gate | Statistic | Tolerance vs real corpus |
|---|---|---|
| G1 | Daily energy distribution (per archetype×season) | KS test p > 0.05 against matching real cluster |
| G2 | Peak timing histogram (morning/evening modes) | ±1 h mode shift |
| G3 | Ramp-rate distribution (95th pct 5-min ΔkW) | within ±20% |
| G4 | Autocorrelation (lag 1–12 steps) | within ±0.1 |
| G5 | Zero-energy rows | 0 (no fake idle days) |
| G6 | Physical sanity | HouseLoad ≥ 0, SolarGen ≥ 0, no NaN |

Gate failures loop back to resampling/injection parameters, never hand-fixed.

#### Corpus target & surfaces

- 5 archetypes × 4 seasons × 3 battery sizes × 20 seeds ≈ **1,200 episodes**
  (episode = one segment-week; matches AEMO-track corpus scale)
- Splits: train 70% (archetype-balanced), val 15%, test 15% + held-out
  OOD surface = the REAL household segments (never trained on)
- Output: `data/household/synth/<archetype>/<seed>_ep<id>.parquet` +
  `manifest.json` (params per episode for reproducibility)
- CLI: `python3 scripts/build_household_synth_corpus.py` (defaults to 1,200
  seven-day episodes, seed 42, and a 15% real-source OOD holdout)
- The manifest records every generation knob, source date/cluster, gate
  metric, split, battery capacity/flow, and whether TTM was used. Generated
  parquets are env-view-compatible (`SolarBatteryEnv` observes 12 features).

#### Implementation phases

| Step | Deliverable | Test |
|---|---|---|
| ✅ 1 | `src/household_synthetic.py`: day clustering (season×daytype k-means on normalized profiles) | cluster purity > 0.8 on held-out labels |
| ✅ 2 | Resampler + λ-scaling + injection blocks | G1–G6 harness green |
| ✅ 3 | Battery/solar assignment + env-view export | env instantiates per episode, 12-D obs |
| ✅ 4 | Corpus build CLI (`scripts/build_household_synth_corpus.py`) | manifest complete, counts exact |
| 🟡 5 | Optional TTM imputation module (isolated flag) | `--use-ttm` and explicit modes are documented; the adapter fails explicitly until Granite runtime provisioning and imputation validation are complete |

#### Risks

- Archetype parameters are guesses until more real households exist — keep
  every knob explicit and versioned in the manifest.
- Over-trusting synthetic diversity: the OOD surface of record stays the
  real household; synthetic is for training breadth only.
- EV/AC injection may dominate small-λ archetypes (appliance block bigger
  than base load) — cap injection at ≤60% of daily energy.

### Quick wins

- ✅ QH1: Ingestion script (`scripts/ingest_household_portal_csv.py` + `src/household_ingest.py`) — normalization to env schema, gap/DST/duplicate/negative-value validation, sha256 manifest (share-safe, privacy-tested). PR `feature/household-modern-data`.
- ⬜ QH2: One-week pilot — blocked on first real portal download; will finalize column hints + units (kW vs kWh) against a real sample (see `docs/household/real_data_protocol.md` "Known format unknowns").

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