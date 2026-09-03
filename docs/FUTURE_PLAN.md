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
| ✅ **H1** | **Re-established legacy benchmark**: existing rule, perfect-foresight oracle, SB3 PPO, and cloning-era DT evaluated on five gap-separated real OOD segments at 5 kWh / 3.3 kW under the legacy flat 30c/5c tariff. Annualized bootstrap results: no battery $1,254 (95% CI $700–$1,894), rule $1,185 (saves $70), PPO $1,252 (saves $2), DT $1,260 (loses $6), oracle $525 (saves $729). | Confirms the Ausgrid-era PPO/DT policies do not generalize to modern telemetry; H2 retraining is required |
| ✅ **H1.5** | **Synthetic diverse-household generator** — see detailed plan below | Core generator, G1–G6 gates, env-view export, OOD holdout, and reproducible corpus CLI implemented; optional Granite TTM adapter remains separately provisioned |
| ✅ **H2** | **SDP-teacher distillation transfer test completed — POSITIVE TRANSFER**: checkpoint-specific configs now persist alongside weights (`h2_*_model_kwargs.json`); 840/180 synthetic teacher episodes regenerated under realistic tariff (31.042c import, free 11:00–14:00, 1c FiT) with correct RTE=0.80; inference precomputes exact segment-local `-J_t(soc)` prompt per calendar day. **Standard-RTG baseline on 2×128 ctx60 achieves +$254/yr** (beats rule +$82, 28% of oracle gap). **J_t(soc) on 8×512 ctx576 with corrected RTE=0.80 achieves +$300/yr** (3.6× rule, 24% of oracle gap). Both models beat rule and demonstrate successful planner-distillation transfer at household scale. | Distillation transfers across scales when: (1) config persists correctly, (2) RTE matches env (0.80), (3) model capacity ≥ AEMO-scale (8×512/ctx576), (4) teacher data uses realistic tariff. J_t(soc) inference works but requires sufficient model capacity; standard-RTG remains a strong baseline.
| ✅ **H3** | **Replay-gap analysis completed**: deterministic 5-minute optimizer compares recorded VPP actions with re-optimized dispatch per contiguous real day. At 5 kWh / 3.3 kW / 0.80 RTE: observed-to-optimal gap is **$355/yr** with free 11:00–14:00 pricing (95% bootstrap CI $342–$368 over 1,084 complete days); **$214/yr** under flat pricing. Spot pass-through remains pending a time-aligned retail spot-price series. |
| ✅ **H3** | **Tariff-policy experiments**: flat and realistic free-window ToU sweep implemented in `scripts/evaluate_household_tariffs.py`; prices are re-derived rather than using stored 0.30/0.05 defaults. | Spot pass-through remains pending a time-aligned retail spot-price series. |

### Household follow-on work (H4 — bring household evidence to AEMO standard)

H0–H3 establish a working household pipeline and a positive distillation
signal, but the evidence is not yet comparable to the AEMO track's
multi-surface result. The current synthetic corpus is fixed at seven-day
episodes, the forecast channels use a simple persistence baseline, and the
PPO comparison is the legacy checkpoint rather than a policy freshly trained
on the modern synthetic corpus. These are research gaps, not implementation
failures.

| ID | Next experiment | Acceptance criteria |
|---|---|---|
| ✅ H4.1 | **Horizon- and scenario-diverse synthetic corpus** — generate episodes spanning one week, several weeks, roughly six months, and multi-year horizons across the existing five archetypes, seasons/day-types, appliance recipes, solar sizing/orientation, and battery configurations. Preserve contiguous calendar order within each episode and record horizon, source dates, tariff, battery size, degradation model, and seed in the manifest. The first H4.1 build also supplies the held-out synthetic surfaces for H4.4 evaluation. | Degradation and calendar aging affect the objective and learned policy; no horizon/scenario leakage across train/validation/test; short- and long-horizon results reported separately. **Done (2026-09-01):** full build at `data/household/synth_h4_1` — 240 episodes (5 archetypes × 4 seasons × 3 capacities × horizons 1w/2w/6m/2y; 55,860 episode-days), schema-v2 manifest with per-episode horizon, degradation (`calendar_cycle_rainflow`, life cost $5,000), solar, appliance and provenance params; splits 165/35/40 with the same 158 real source dates held out for OOD. Note: 23/40 train 2y episodes terminate early (`total_degradation >= 1.0`) — degradation exhaustion is a genuine long-horizon signal, identically present in all forecast variants. |
| ✅ **H4.2** | **Forecast ablation and improved forecast generation** — inference-only substitution showed no value from the old forecast fields, so three matched 8×512/context-576 policies were retrained. Granite TTM-R3 runs offline in an isolated CUDA Distrobox and improved real-OOD solar/load MAE by **39.2%/17.5%** over current-value persistence. Under causal standard RTG at the shared training-median prompt (RTG=−2), annualized savings were **TTM +$258.50**, **24-hour persistence +$216.74**, and **no forecast +$155.12**. TTM beat persistence by **+$41.75/yr** (95% CI **+$16.56–$69.43**, 9/10 wins, p=0.0068) and no forecast by **+$103.37/yr** (95% CI **+$78.23–$127.67**, 10/10 wins, p=0.0010). The oracle-assisted J_t(soc) comparison independently favored TTM by +$117.36/year. | Matched standard-RTG controls establish that forecast features are useful and TTM adds value beyond 24-hour persistence. Keep the offline TTM channels in the household observation pipeline. RTG prompt sensitivity and the current single-household OOD surface remain explicit H4.4 limitations. |
| ✅ **H4.3** | **Fresh modern-data SB3 baseline** — fresh PPO was trained for 250k steps on the H4.1 train split with 12 parallel CPU environments and evaluated at the matched 5 kWh/3.3 kW configuration. On the five real OOD segments it saves **+$27/yr**, below the rule **+$81/yr** and H2 DT **+$300/yr**; the legacy PPO remains only a historical transfer baseline. | Initial fresh-PPO comparison complete; retraining SB3 alone does not solve household transfer. Longer/multi-seed PPO and optional SAC/TD3 remain follow-ups after forecast and degradation studies. |
| ✅ **H4.4** | **AEMO-equivalent household evaluation surfaces** — full-corpus (H4.1) matched three-way forecast comparison plus fresh full-corpus PPO, evaluated on the fixed 10-window real-OOD surface and a 20-window synthetic-test surface with per-episode battery configs. | **Done (2026-09-02).** Matched standard-RTG DTs (8×512/ctx576, shared RTG=−2 from training median −1.78) on real OOD: **TTM +$357.29/yr, 24h-persistence +$309.35/yr, no-forecast +$310.90/yr, fresh full-corpus PPO +$23.66/yr, rule +$58.03/yr, oracle +$738.96/yr**. TTM beats persistence by **+$47.94/yr** (95% CI +$27.99–$67.59, 9/10, Wilcoxon p=0.0020) and no-forecast by **+$46.39/yr** (95% CI +$23.78–$68.91, 9/10, p=0.0020). Persistence≈no-forecast now collapses (−$1.55, p=0.46): with diverse data only the genuinely better TTM channel helps. All savings are higher than the H4.2 7-day-corpus run (TTM +$98.79), confirming the offline-forecast benefit generalizes beyond the controlled corpus. **Limitation:** on the synthetic multi-battery test surface the three DTs are statistically indistinguishable (TTM−no_forecast −$15.49/yr, p=0.86; per-horizon mixed) — the forecast advantage is proven on the held-out real household, not yet on the broad synthetic surface. Commands in `workflow.md` §6c; artifacts `eval_output/household/h4_4_*`, `models/household/dt/h4_4_*`. |
| 🟡 **H4.5** | **Degradation-aware policy study** — compare degradation disabled, cycle-only, calendar-plus-cycle, and realistic battery-life-cost settings across the horizon-diverse corpus. | Report whether the DT learns economically meaningful cycling restraint and whether degradation changes the policy ranking. **In progress** (scripts/h4_degradation_study.py implemented). |
| 🔵 **H4.6** | **Optional extensions** — add a time-aligned retail spot-price pass-through study and provision the isolated TTM gap/weather-residual adapter only after the statistical baselines are complete. | Spot and TTM claims remain separately gated and do not replace the bootstrap/recomposition baselines. |

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
| ✅ 5 | TTM integration superseded | The in-generator `--use-ttm`/`--ttm-mode` stub was removed (Sep 2026 cleanup); TTM forecasts are now handled by the offline causal sidecar pipeline (H4.2/H4.4, `src/household_forecast.py` + `scripts/run_household_ttm_forecasts.sh`), never inside the generator or simulator. |

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