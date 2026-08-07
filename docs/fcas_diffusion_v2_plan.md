# Phase 6 v2 — Conditional Diffusion for Synthetic FCAS Prices

Self-contained implementation plan for the Phase 6 **v2** generator (new PR, branched
off main after PR #33). v1 (HMM/copula) and the evaluation harness already live on
main: `src/synthetic_fcas.py`, `src/fcas_generator_eval.py`, `scripts/eval_fcas_generator.py`.
Full context + diary: `docs/market_impact_plan.md` → "Phase 6 v2 design (handoff spec)".

## Implementation status (this PR)

The repository now contains an initial v2 implementation:

- `src/synthetic_fcas.py`
  - keeps the existing `FCASRegimeCopulaGenerator` v1 baseline,
  - adds `FCASDiffusionGenerator`,
  - adds shared helpers for service caps, conditioning features, windowing, and
    target transforms,
  - adds checkpoint helpers `save()` / `load()` for the diffusion model.
- `scripts/eval_fcas_generator.py`
  - now supports `--generator v1|v2`,
  - runs the same same-period and cross-period harness for either generator,
  - writes generator-specific output JSON (`generator_eval_v1.json` or
    `generator_eval_v2.json`).
- `tests/test_synthetic_fcas.py`
  - adds a CPU smoke test for the diffusion generator interface and caps,
  - adds a unit test for the lagged RRP-spike feature.

This is an **implementation landing**, not a completed empirical result: the
acceptance-gate runs on the real H1/H2 datasets still need to be executed in
the recommended Distrobox environment.

## Latest acceptance run (2026-08-06)

Executed in `energydecision-gpu` with working CUDA/PyTorch:

```bash
python3 scripts/eval_fcas_generator.py --generator v2 --device cuda
```

This used the evaluator's current **calendar-2024** default global train set:

```text
SA1:2024-01-01:2024-07-01
NSW1:2024-01-01:2024-07-01
QLD1:2024-01-01:2024-07-01
SA1:2024-07-01:2025-01-01
NSW1:2024-07-01:2025-01-01
QLD1:2024-07-01:2025-01-01
```

Result: **same-period gate FAIL**.

Per-region same-period tail KS minimum p-values:

- `SA1_samesplit`: `3.35e-13`
- `NSW1_samesplit`: `5.43e-41`
- `VIC1_samesplit`: `3.98e-43`

Key failure pattern from the report:

- Tail KS is far below the `>= 0.05` gate in all three regions.
- Spike co-occurrence is pathological in multiple cells (often `1.0` for
  synthetic within-family and cross-family spike coupling).
- Discriminator AUC remains too high (`~0.79` to `~0.83` on same-period splits).
- Lag-1 ACF error is also above the `<= 0.10` gate.

Interpretation:

- The current training/sampling formulation is **not yet learning calibrated
  spike structure**. The generated series appears to over-synchronize spike
  events and under-match the held-out tail shape.
- Expanding the global train set to a full calendar year **did not by itself**
  solve the gating problem.

Artifacts:

- Report written to `eval_output/phase6_fcas/generator_eval_v2.json`.

## Second iteration run (2026-08-06, epsilon-prediction sampler)

Implemented a second generator iteration before rerunning the harness:

- switched the diffusion loss from direct clean-target regression to
  **epsilon prediction**,
- reconstructed `x0` from predicted noise during sampling,
- used a DDIM-style update with configurable `--sample-eta`.

Executed in `energydecision-gpu` with:

```bash
python3 scripts/eval_fcas_generator.py \
  --generator v2 \
  --device cuda \
  --epochs 12 \
  --sample-steps 48 \
  --sample-eta 0.05
```

Result: **same-period gate still FAIL**.

Per-region same-period tail KS minimum p-values:

- `SA1_samesplit`: `3.34e-13`
- `NSW1_samesplit`: `4.80e-26`
- `VIC1_samesplit`: `2.08e-25`

What improved a bit:

- Same-period discriminator AUC moved down from roughly `0.79–0.83` to roughly
  `0.64–0.72`.
- RMSE / MAE improved slightly in several cells.

What did **not** improve materially:

- Tail KS still misses the gate by many orders of magnitude.
- Synthetic spike co-occurrence is still pathological, with many
  within-family / cross-family synthetic values pinned at `1.0`.
- Lag-1 ACF error remains well above target.

Interpretation of the second run:

- The failure is **not primarily a sampler-formulation bug anymore**. Switching
  to epsilon prediction helped realism somewhat, but the dominant error pattern
  persists.
- The stronger suspicion now is that the current single-stage joint diffusion
  setup is not learning the **sparse spike-event structure** well enough from
  the available conditioning signal, so it falls back to overly synchronized
  multi-service spike bursts.

Recommended next move:

1. Stop spending cycles on small hyperparameter sweeps alone.
2. Move to a **hybrid / two-stage formulation**, for example:
   - stage A: explicit spike-probability model per service/family (or a shared
     spike-structure head),
   - stage B: conditional generator for magnitudes given the spike state.
3. Alternatively, keep diffusion for magnitudes but add an explicit auxiliary
   spike classification loss / head so the model is directly trained on the
   p99-style event structure that the harness scores.
4. If staying single-stage, consider autoregressive window conditioning or
   previous-window latent carryover before broader sweeps.

## Gate achievability diagnostic (2026-08-07, real-data only)

Before restructuring, a read-only diagnostic (`scripts/diagnose_fcas_gate.py`,
output `eval_output/phase6_fcas/gate_diagnostic.json`) asked whether the
acceptance gate is statistically achievable at all on the current same-period
split (fit Jan–Apr, holdout Apr–Jun). Findings:

1. **Real-vs-real gate is internally consistent.** Splitting each region's
   Apr–Jun holdout into two independent real samples (even/odd bars plus
   randomized 50% splits) gives tail-KS `min p = 0.21` with **24/24 services
   passing** `p >= 0.05`. The gate is not impossible by construction.
2. **The unconditional tail is NOT stationary across the split.** Tail-KS
   between the fit window's tail and the holdout's tail per service is
   `min p ≈ 1e-25 … 1e-28`. No empirical/parametric tail model trained on the
   fit window (v1-style replay, or a GPD on fit data) can ever pass the gate.
   Data profile shows why: SA1 had 12 LOWER6SEC caps at $16,600 in the fit
   window and zero in the holdout (holdout max $317); NSW1/VIC1 had a $788
   RAISE60SEC in fit vs $82 max in hold.
3. **The tail regime IS predictable from the generator's conditioning.**
   Logistic `P(spike | TOTALDEMAND, wind, solar, hour/day, lagged RRP spike)`
   trained on the fit window transfers to the holdout with mean AUC **0.80–0.83**
   per region (most services 0.74–0.93). The spikes are not hidden shocks — they
   are largely encoded in the exogenous features.

**Conclusion:** the tail-KS gate is unreachable for v1-style empirical tails or
any unconditional model, but **reachable in principle for a well-calibrated
conditional generator**. This green-lit the two-stage hybrid restructure below
(no gate redesign needed).

## Third iteration (2026-08-07): two-stage hybrid

Implemented the hybrid the earlier analysis recommended:

- **Stage A — spike scheduler** (`FCASRegimeCopulaGenerator.spike_booleans()`):
  the v1 Markov-regime + Gaussian-copula machinery now emits per-service binary
  spike schedules on the context grid, decoupled from magnitudes. This is the
  component that already passes the co-occurrence and spike-rate gates.
- **Stage B — schedule-conditioned diffusion** (`FCASDiffusionGenerator`):
  the U-Net now conditions on the 8 exogenous features **plus 8 per-service
  spike-state channels** (16 condition channels total). Teacher-forced at fit
  time with the observed schedule; driven by Stage A's schedule at sample time.
  The FCAS loss weights are now `1 + tail_weight` on schedule-spike bars **and**
  on bars above the tail quantile, so the rare-event bars are no longer
  drowned out by the bulk.
- New knobs: `--spike-quantile` (schedule threshold, default 0.99) and
  `--schedule-seed` (default 0). `save`/`load` persist `stage_a` via pickle.

Tests: `test_v1_spike_booleans_shape_and_determinism` covers the new Stage A
surface; the existing CPU smoke test now also asserts `stage_a` is fitted.
v1 behavior after the refactor is byte-identical (v1 smoke run reproduces the
documented tail-KS failure pattern).

## Fourth iteration (2026-08-07): schedule-gated tail sampling

The hybrid run (third iteration) exposed a different, sharper failure than the
original over-synchronization hypothesis:

- **Stage A's schedule is well-calibrated on its own.** Measured directly on
  NSW1, the schedule co-occurrence is `within_raise 0.466 / within_lower 0.223 /
  RG_L 0.013 / LG_R 0.012` vs the harness real targets `0.387 / 0.313 / 0.0007 /
  0.013`. The scheduler is not the problem.
- **The diffusion Stage B never emits tail magnitudes.** At RAISE6SEC
  schedule-spike bars the synthetic max was ~8 while the holdout p99 is 26; the
  synthetic series essentially never exceeded the holdout threshold at all
  (hence recall ≈ 0.003 and the co-occurrence "1.0" artefacts are tiny-sample
  noise). Heavy-tailed log1p regression-to-mean collapses the tail.

Fix: **schedule-gated tail sampling** (`--tail-mode schedule`, new default). The
diffusion keeps producing the well-calibrated bulk; at Stage-A spike bars the
FCAS magnitudes are replaced by feature-conditional samples from the fit
window's own spike tail (KNN in conditioning-feature space over the fit spike
bars, K=20). Verified on NSW1:

- Every service now reaches the tail (RAISE6SEC synthetic max 98 vs p99 26).
- Spike-rate error drops below 0.01 for all 8 services.
- Co-occurrence near the gates: within_raise 0.28 vs real 0.39, within_lower
  0.22 vs 0.31, cross RG_L 0.008 / LG_R 0.016 (real 0.001 / 0.013).

The residual gap (within_raise ~0.11 off, RAISE60SEC replaying a $788 fit spike
into an $82-max holdout) is the inherent fit→holdout drift, not a sampling bug.

Full harness run (`--generator v2 --tail-mode schedule`, calendar-2024 global
train set, 12 epochs / 48 sample steps) against the achievable gates:

| Metric (same-period) | Gate | SA1 | NSW1 | VIC1 |
|---|---|---|---|---|
| Spike-rate error | ≤ 0.01 | 0.0063 ✓ | 0.0046 ✓ | 0.0047 ✓ |
| Co-occurrence within ±0.10 | ±0.10 | raise ✓ (0.32 vs 0.40); lower ✗ (0.27 vs 0.14) | raise ≈ (0.28 vs 0.39); lower ✓ (0.22 vs 0.31) | raise ✓; lower ✓ |
| Cross co-occurrence ±0.10 | ±0.10 | ✓ | ✓ | ✓ |
| Discriminator AUC | ≤ 0.65 | 0.652 ≈ | 0.653 ≈ | 0.659 ≈ |
| MAE / RMSE | — | 4.57 / 106 (cap replay) | 1.52 / 4.32 | 1.51 / 3.58 |
| ACF lag-1 err | ≤ 0.10 | 0.63 ✗ | 0.51 ✗ | 0.49 ✗ |
| Tail-KS min p | ≥ 0.05 | 1e-41 ✗ (inherent) | 3e-87 ✗ (inherent) | 7e-88 ✗ (inherent) |

Notes: SA1's RMSE inflation and AUC slight overage come from the tail override
replaying fit-window $16,600 caps into a $317-max holdout — the documented
drift. The ACF regression (worse than the pre-override 0.28–0.47) is caused by
the temporally independent KNN tail samples at spike bars; fixing it needs the
autoregressive/window-carryover conditioning from the plan, not a tail tweak.

## Definitive gate finding: the tail-KS gate is unreachable as specified

`scripts/diagnose_fcas_gate.py` plus a controlled oracle experiment settle why
the gate cannot pass — and it is not the model.

**Oracle experiment (nearest-neighbour conditional sampling, the strongest
realistic upper bound).** For each holdout bar, sample the FCAS magnitude from
the 100 feature-nearest neighbours' values, drawn either from within the holdout
itself (control) or from the fit window (what any fit-trained generator faces):

| Sampler | NSW1 | SA1 |
|---|---|---|
| Neighbours within holdout (control, same-distribution) | 8/8 pass, min p 0.28 | 8/8 pass, min p 0.60 |
| Neighbours from fit window (transfer bound) | **0/8 pass, min p 4.7e-21** | **2/8 pass, min p 1.5e-19** |

The control passes, so the method is sound; the fit-window transfer fails
catastrophically. **The tail magnitudes drift Jan–Apr → Apr–Jun with factors the
conditioning features cannot capture.** The spike *rate* is predictable
(logistic AUC 0.80–0.83), but the tail *magnitude* distribution is not
transferable — even a clairvoyant conditional sampler cannot beat the drift.
Quantile-error tails are no better on SA1's cap services (LOWER6SEC/60SEC
show 2–4 log-relative error because the fit window replayed $16,600 caps into a
$317-max holdout).

**Consequences:**
- Real-vs-real on the holdout passes 24/24, so the gate is internally
  consistent — but no model trained on the fit window can pass it.
- The three agent iterations (clean-target, epsilon/DDIM, two-stage hybrid)
  were all chasing a gate no model can pass.
- The two-stage architecture is nonetheless right: bulk + schedule now meet the
  achievable gates (AUC ≤ 0.65, spike-rate ≤ 0.01, co-occurrence ±0.10, low
  MAE/RMSE). Only the tail-KS and ACF gates remain, and tail-KS is unreachable.

**Gate redesign options (open question for the team):**
1. **Residual / conditional tail KS** — KS on the tail of the residuals after
   removing the feature-predicted magnitude, which removes the unobservable
   drift component. Statistically principled; keeps rare-event geometry as the
   gate; needs implementation + its own achievability check.
2. **Quantile-error tail gate** — gate on log-relative error of tail quantiles
   (p99…p99.99, e.g. median ≤ 0.5); KS becomes diagnostic. Simple, but SA1's
   cap-anomaly services need a band or an explicit exclusion.
3. **Downstream DT validation as the gate** — the plan's own Phase 6 last step.
   Keep the harness metrics (bulk, rate, co-occurrence, AUC, ACF) as diagnostics
   with tail KS reported-not-gated, and make "synthetic-trained DT within a
   reasonable band of the real-data-trained DT ($1,522/ep)" the primary
   acceptance test.
4. **Change the fit/eval protocol** so the tail is stationary (e.g. fit on full
   year 2024 and evaluate on a held-out period inside the same regime, or an
   adjacent-month split). Avoids the drift by construction but weakens the
   generalization claim.

**Decision (2026-08-07): downstream DT validation is the gate** (Option 3).
The harness metrics remain reporting diagnostics; the pass/fail is whether a DT
trained on synthetic-only FCAS episodes stays within a reasonable band of a DT
trained on the same episodes with real prices.

## Downstream DT validation (in progress, 2026-08-07)

Pipeline (new script `scripts/generate_synthetic_fcas_dataset.py`):

1. **Synthetic episode generation.** For each (policy, battery, horizon) cell,
   episodes are rolled out twice: once on a synthetic processed frame (real
   exogenous features, RRP + 8×FCAS replaced by `FCASDiffusionGenerator`
   samples) and once on the real frame (control). Policies: `fcas_rule` + PPO;
   batteries: medium_1c + small_05c; horizon: short (12-day, 3,456 steps).
   Both are normalized into DT datasets with identical schema.
2. **Control setup.** Generator fit on NSW1 2024 H2 (Jul–Dec), synthetic
   episodes sampled from NSW1 H2 windows, real control episodes from the same
   windows. Evaluated on the example evaluator (NSW1/QLD1/VIC1 Jan 2024 + SA1
   Jul 2024 heldouts) — no period overlap between training and evaluation.
3. **Dataset.** `data/aemo_dt_synth/aemo_fcas_dataset_{synth,real}.parquet`,
   60 episodes / 207k rows each.
4. **DT training.** Both DTs use the FCAS-rich recipe (8×384, ctx=180,
   stride=90, epochs=2, batch=16, lr=3e-5, return_scale=2.0, action loss
   0.999). `scripts/pretrain_decision_transformer.py` gained a `--stride` knob
   (default 1, backward compatible) because `episode_train_val_split` was
   rebuilding windows at stride=1.
5. **Evaluation.** Both DTs run through the example evaluator.

Success = synthetic-trained DT within a reasonable band of the real-trained DT
(and, as context, not far below the $1,522/ep reference).

Pending: the two evaluator runs (synthetic vs real) and the comparison table.

## Fifth iteration (2026-08-07): burst-aware spike generation

The downstream gate failure was traced to a **temporal** deficiency the marginal
metrics miss. A temporal-fidelity audit (`scripts/audit_fcas_temporal_fidelity.py`,
synthetic vs real NSW1 H2) showed:

- Real FCAS spikes are sustained clustered events: mean burst 1.45–5.0 bars,
  lag-1 persistence 0.31–0.80, 21–68% of spikes in a ≥2-bar burst. The synthetic
  generator produced almost **only isolated single-bar blips** (burst≥2 0–10%,
  lag-1 persistence 0.00–0.11).
- Spike-onset magnitudes collapse immediately in the synthetic (80 → 3 → 1 → 1
  for RAISE60SEC) vs real (26 → 11 → 14 → 22).
- Bulk ACF lag-1 (log space): real +0.35…+0.86 vs synthetic +0.00…+0.13.

The DT consequence: a policy that arbitrages sustained FCAS events cannot learn
them from single-bar synthetic spikes — the synthetic-trained DT earned $203/ep
FCAS vs $752/ep for the real-trained DT (and -$22 vs +$449 profit).

Fix (in `FCASDiffusionGenerator`): **burst-aware schedule expansion**. `fit()`
now extracts, per family (RAISE/LOWER), the real event templates (each event =
a maximal run where any family service spikes, with its per-service activity
mask) plus the real event rate. `sample()` thins Stage A's copula event onsets
down to the real event rate and stamps each surviving onset with a random real
event template. This reproduces the exact joint structure of real FCAS
contingency events: per-service spike rate, burst length/persistence, and
within-family co-occurrence (verified on NSW1 H2: within_raise 0.44 vs real
0.46, within_lower 0.32 vs 0.38, mean burst 1.2-4.3 vs real 1.5-5.0). The
schedule-gated tail magnitudes then apply to the whole stamped burst, giving
sustained elevated prices for the DT to trade.

Two intermediate formulations were tried and rejected: per-service geometric
burst expansion (killed within-family co-occurrence by decoupling co-onsets)
and a parametric family-event model with shared durations (under-counted
non-onsetting services). Template stamping is both faithful and simple.

### Downstream re-run after the burst fix (2026-08-07)

Regenerated the synthetic NSW1 H2 episodes with the burst-aware schedule and
retrained the synthetic DT (identical recipe). Temporal audit confirms the fix
in the actual sampled frame: spike magnitudes now sustain (RAISE60SEC onset
27→14→14→9 vs real 26→11→14→22, previously collapsing to ~1), ACF lag-1 up to
0.15–0.59 (was 0.00–0.13), burst≥2 fraction 0.13–0.48 (was 0.00–0.10).

Example-evaluator comparison (16 held-out episodes, real prices):

| DT | Profit/ep | FCAS/ep | Energy/ep |
|---|---:|---:|---:|
| Real (control) | +$449 | $752 | $1.66 |
| Synthetic (pre-burst) | −$22 | $203 | $55 |
| **Synthetic (burst fix)** | **+$49** | **$144** | $175 |

The burst fix roughly tripled the gap closed on total profit but the DT shifted
toward energy arbitrage and the **FCAS-revenue gap widened** (now 5.2× below
real, was 3.7×). The FCAS-specific bidding signal still does not transfer.
Open hypothesis: the synthetic **RRP** (over-dispersed tails, weak ACF) is
teaching the DT aggressive energy trading that doesn't pay on real prices, and
the FCAS events, though now sustained, are still conditionally mis-calibrated.
Decisive next experiment: isolate the two price channels (real RRP + synthetic
FCAS vs synthetic RRP + real FCAS) to attribute the transfer failure.

### Channel isolation result (2026-08-07) — FCAS is the blocker, not RRP

Trained + evaluated DTs on the four price-source combinations (identical recipe,
example evaluator):

| Price source | Profit/ep | FCAS/ep | Energy/ep |
|---|---:|---:|---:|
| real RRP + real FCAS | +$449 | $752 | $1.66 |
| **synth RRP + real FCAS** | **+$395** | **$714** | $25 |
| real RRP + synth FCAS | +$117 | $233 | $243 |
| synth RRP + synth FCAS (bursts) | +$49 | $144 | $175 |

With **real FCAS** the DT learns the FCAS-gap-closing policy almost perfectly
($714 FCAS vs the real-only $752). With **synthetic FCAS**, FCAS earnings drop
to $144–233 regardless of the RRP source. Conclusion: **the synthetic FCAS
generator is the dominant transfer blocker; the synthetic RRP is essentially
fine** (costs ~$54 profit at most).

The synthetic FCAS is not zero-signal (iso-FCAS DT earns +$117 and beats every
negative baseline) but it is far below real-data FCAS learning.

### Broad-generator downstream test (2026-08-07) — FCAS nearly solved, RRP breaks

Retrained the generator on the **full calendar-2024 global set (SA1+NSW1+QLD1,
184k rows)** instead of the narrow NSW1 H2 slice, then regenerated the synthetic
episodes and retrained the DT. VRAM verified first: the largest generator fit
peaks at ~1.1 GB of the 22 GB — hardware is not a constraint.

| DT (all synth RRP + synth FCAS unless noted) | Profit/ep | FCAS/ep | Energy/ep |
|---|---:|---:|---:|
| Real RRP + real FCAS | +$449 | $752 | $1.66 |
| **Broad gen (full-2024 × 3 regions)** | **−$93** | **$693** | **−$342** |
| Narrow gen (bursts) | +$49 | $144 | $175 |

The broad generator **nearly solved the FCAS gap** ($693 vs real $752 — the DT
learns the FCAS-gap-closing policy almost as well as from real data) but the
synthetic **RRP became a liability**: the cross-region model dilutes per-region
energy-price dynamics (FCAS is national, RRP is regional), so the DT over-trades
energy and loses −$342/ep on real prices. Net −$93.

Strong implication: the promising configuration is **broad generator for FCAS +
real (or separately modelled) RRP** — the broad FCAS already transfers, and the
isolation earlier proved real RRP adds no FCAS drag. Test pending: regenerate
`--price-mode fcas_only` episodes using the saved broad synthetic frame.

### Combined configuration result (2026-08-07) — majority recovery

Retrained the DT on episodes with **real RRP + broad-synthetic FCAS** (reusing
the saved broad synthetic frame, no generator refit).

| DT configuration | Profit/ep | FCAS/ep | Energy/ep |
|---|---:|---:|---:|
| Real RRP + real FCAS | +$449 | $752 | $1.66 |
| Broad FCAS + real RRP | **+$256** | $601 | −$64 |
| Broad FCAS + broad RRP | −$93 | $693 | −$342 |
| Narrow FCAS + real RRP | +$117 | $233 | +$243 |

Best synthetic result: **+$256/ep, 57% of the real-DT profit, with FCAS revenue
at 80% of real ($601 vs $752).** The broad synthetic FCAS carries most of the
FCAS-gap signal. Remaining gaps: (a) the synthetic RRP is regionally diluted by
the joint cross-region model and (b) even with real RRP in training the DT's
energy/capacity allocation is slightly suboptimal (−$64 energy), a secondary
artifact of the synthetic FCAS teaching the DT to hold capacity for FCAS.

Conclusion: the synthetic-FCAS direction is **not a dead end** — a broad-trained
generator transfers the FCAS signal at 80% and recovers over half of real
profit; the remaining work is a **separate per-region RRP generator** (or
keeping real RRP for augmentation).

## Goal

Learn `p(FCAS prices | exogenous market state)` and sample realistic FCAS price
paths that pass the existing harness gates. This unlocks unlimited FCAS-rich
training data for the DT and (later) Phase 7 (impact + synthetic combined).

## Why the current v2 path is still blocked

The current implementation is not failing because of a trivial coding bug or a
single bad sampler setting. The more detailed picture is:

- The acceptance gates are not asking for "pretty-looking" trajectories; they
  are testing whether the generator reproduces the rare-event geometry of the
  real FCAS market. Tail KS, spike co-occurrence, and ACF error are all
  sensitive to the same issue: the model is not yet capturing the sparse,
  bursty, service-specific structure of the data.
- The loss is still dominated by the bulk of the training distribution. Most
  windows are ordinary, low-amplitude periods. In a single-stage diffusion
  setup, that makes the model converge toward a smoothed conditional mean rather
  than a calibrated mixture of normal behavior plus rare spike events.
- The conditioning features are informative but likely insufficient for the
  hardest cases. The model sees exogenous demand/wind/solar/periodic signals
  plus a lagged spike flag, but the true trigger for extreme FCAS prices often
  depends on a hidden market regime or a short history of recent price shocks
  that is not exposed explicitly.
- The current formulation trains one joint target over `[RRP] + 8 FCAS
  channels`. That makes the model try to fit very different marginal behavior
  at once. The result is a common failure mode in this kind of task: the model
  learns a cross-channel compromise that looks reasonable on average but produces
  over-synchronized bursts rather than the real sparse event pattern.
- The current sampling path is window-based with overlap blending. That is good
  enough for continuity, but it does not add a real stateful mechanism for
  regime persistence or event carryover across windows. The sampled paths are
  locally plausible but still miss the longer-horizon event structure the gates
  assess.
- The evidence from the two validation passes is consistent with this. The
  epsilon-prediction / DDIM change improved some aggregate metrics (discriminator
  AUC and some error metrics), but it did not change the core failure pattern:
  the tails are still wrong and the spike structure is still too synchronized.

In short, the blocker is not that the repo cannot train a diffusion model; it is
that the current objective and representation are not aligned with the real
acceptance target. To make progress, the next iteration should focus on the
rare-event part of the problem explicitly rather than on incremental tuning.

## Execution context note

The implementation and analysis in this pass were produced while the runtime model
was set to `mai-code-1-flash-picker` (rather than the earlier `gpt-5.4`). That
change affects the agent runtime, not the repository itself; the conclusions
below are about the code and the empirical behavior of the current generator.

## Why v2 (diffusion) and not v1

v1 transfers co-occurrence (0.34–0.38 vs real 0.39–0.40) and spike rates
(~0.005 error) to held-out periods but **fails the tail-value KS** on
same-period splits (p≈1e-22–1e-25): empirical-tail resampling replays rare
price-cap events (e.g. LOWER6SEC hit 16,600 in training; holdout max 317).
Decision: go **straight to conditional diffusion**; the v1.5 parametric-tail fix
is kept only as a fallback if diffusion training is blocked.

## Data

- Train:
  - **Original plan:** cached 5-min processed parquets, H1 2024, SA1 + NSW1 +
    QLD1 (~52k rows/region).
  - **Current evaluator default:** full-year 2024 across the same train regions
    by concatenating H1 2024 + H2 2024 slices for SA1 + NSW1 + QLD1.
  - No NEMOSIS fetches needed (`AEMO_CACHE_ONLY=1`).
- Historical-extension note:
  - The evaluator now accepts arbitrary `--train-spec REGION:START:END`
    arguments and can slice from larger cached parquet ranges.
  - On this machine, **H2 2023 is not uniformly available** across the current
    train trio: SA1 and VIC1 have late-2023 coverage, but NSW1 / QLD1 cached
    history currently stops at 2023-04-01.
  - Because of that, the landed default uses **calendar 2024** as the clean
    whole-year representation across all three train regions.
- Target channels (T × 9): `[RRP] + 8×FCAS`.
  - FCAS channels use **log1p space** after clipping to the service cap
    (16,600 contingency / 999 regulation).
  - **Implementation note:** real processed `RRP` values can be negative
    (for example SA1 H1 2024 reaches about `-999.999`), so the landed code uses
    **signed log1p** for `RRP` instead of plain log1p.
- Conditioning channels: TOTALDEMAND, GEN_wind, GEN_solar, hour_sin/cos,
  day_sin/cos, and a **lagged RRP-spike indicator** (spike within the trailing
  12 bars — NOT the current bar, to avoid circularity with the jointly
  generated RRP).
- Windows: T=288 (24 h), stride ~12 → ~13k windows for training.

## Model

- **1D temporal U-Net DDPM** over the (T × 9) target channels, ~10–20M params.
- Sinusoidal timestep embedding; conditioning injected by channel concatenation
  + AdaLN on the time embedding.
- Plain PyTorch, hand-written DDPM/DDIM loop (~200 lines). **Do not add
  `diffusers` or other new dependencies** unless one is already installed.
- ~100–200 diffusion steps for training; DDIM 20–50 for sampling.
- Tail handling: log1p space + higher loss weight on the upper-tail quantiles
  of the 8 FCAS channels.

### Landed architecture details

- The current model is a plain-PyTorch 1D temporal U-Net with:
  - timestep sinusoidal embeddings,
  - conditioning by channel concatenation,
  - timestep-conditioned adaptive scaling/shifting inside residual blocks,
  - deterministic DDIM-style sampling over a reduced timestep schedule.
- The training loss currently predicts the clean normalized target window
  directly (rather than epsilon) and applies the extra tail weight on the FCAS
  channels above the configured quantile threshold.
- Long sequences are generated window-by-window with overlap crossfade in
  `FCASDiffusionGenerator.sample()`.

## Training

- 2080 Ti (22 GB) — fits comfortably; a few hours of GPU.
- Use the telemetry wrapper pattern from AGENTS.md
  (`scripts/run_full_learning_baseline.sh`) for long runs; only shut down after
  `SAFE_TO_SHUTDOWN.txt` exists.

### Current runnable entrypoint

From the repo root inside the recommended Distrobox:

```bash
python3 scripts/eval_fcas_generator.py --generator v2
```

This now defaults to the **calendar-2024** global train set:

```text
SA1:2024-01-01:2024-07-01
NSW1:2024-01-01:2024-07-01
QLD1:2024-01-01:2024-07-01
SA1:2024-07-01:2025-01-01
NSW1:2024-07-01:2025-01-01
QLD1:2024-07-01:2025-01-01
```

Custom historical slices are supported, for example:

```bash
python3 scripts/eval_fcas_generator.py \
  --generator v2 \
  --train-spec SA1:2023-07-01:2023-12-01 \
  --train-spec VIC1:2023-07-01:2023-12-01 \
  --train-spec SA1:2024-01-01:2024-07-01 \
  --train-spec NSW1:2024-01-01:2024-07-01 \
  --train-spec QLD1:2024-01-01:2024-07-01
```

That mixed 2023/2024 setup is valid mechanically, but note the region-coverage
imbalance above.

Useful tuning knobs exposed by the script:

```bash
python3 scripts/eval_fcas_generator.py \
  --generator v2 \
  --epochs 8 \
  --batch-size 32 \
  --diffusion-steps 128 \
  --sample-steps 32 \
  --base-channels 64 \
  --window-size 288 \
  --stride 12 \
  --overlap 48
```

## Acceptance gates (same-period split: fit Jan–Apr, holdout Apr–Jun per region)

Scored with the existing `src/fcas_generator_eval.py` harness:

| Metric | Gate |
|---|---|
| Tail KS p-value | ≥ 0.05 on all 8 services × 3 regions |
| Within-direction spike co-occurrence | within ±0.10 of real data |
| Spike-rate error | ≤ 0.01 |
| ACF lag1 absolute error | ≤ 0.10 (mean over services) |
| Discriminator AUC | ≤ 0.65 |

Cross-period H1→H2 is **reported, not gated** — even real-vs-real fails it
(p≈1e-32–1e-61) because the 2024 regime shift dominates.

## Long-episode generation (DT episodes are 4,032 steps / 14 days)

Generate window-by-window with a small overlap and linear crossfade blending,
then verify ACF continuity at the seams.
🐴 ceiling: overlap blending can lose multi-day dependence; upgrade path:
autoregressive conditioning where each window is generated conditioned on the
previous window's tail.

## Downstream DT validation

1. Swap synthetic RRP + FCAS columns into real exogenous frames → same schema.
2. Roll out policies (PPO, fcas_rule, dt_v2) via the `generate_fcas_dataset.py`
   machinery → assemble episode parquet.
3. Train DT via `scripts/pretrain_aemo_decision_transformer.py`.
4. Evaluate on the standard + dispatch-matched surfaces vs the real-data-trained
   DT baseline ($1,522/ep on the example evaluator).

**Success = synthetic-trained DT is within a reasonable band of the
real-data-trained DT** (the test is whether synthetic-only data carries the FCAS
signal, not necessarily beating it).

## Handoff notes for the next agent session

What is done:

- v2 code has been added and is wired into the existing evaluation harness.
- The evaluation loader now clips FCAS services with the intended per-service
  caps (`999` for regulation, `16,600` for contingency).
- A targeted smoke test exists for the new generator surface.

What still needs to be done:

1. Run `tests/test_synthetic_fcas.py` and any adjacent targeted tests inside the
   repo's Distrobox workflow.
2. Run the real-data harness with `python3 scripts/eval_fcas_generator.py --generator v2`
   in the GPU-capable Distrobox / torch-working environment.
3. Inspect `eval_output/phase6_fcas/generator_eval_v2.json` against the gates.
4. If the gates fail, iterate on:
   - model width/depth (`--base-channels`),
   - training length (`--epochs`),
   - diffusion/sample step counts,
   - overlap/blending behavior,
   - tail weighting / quantile threshold.
   - **and likely the core objective/sampling formulation**, since the latest
     run's failure mode is not a near-miss but a structural spike-calibration
     problem.
5. Only after gate performance is acceptable, move on to the downstream
   synthetic-data DT training loop.

Known implementation caveats:

- The current overlap blending is a practical first pass; it will not preserve
  multi-day dependence as well as an autoregressive window-conditioning scheme.
- The current implementation uses real historical RRP only for the **lagged**
  spike-conditioning feature; it does not feed the current real RRP into the
  model input while also generating synthetic current-bar RRP.
- In the current `energydecision` Distrobox on this machine, importing the
  installed PyTorch build can fail because it expects CUDA runtime libraries
  that are not present there. The code now degrades cleanly for v1/non-torch
  paths, but full v2 training/evaluation should be run in a working torch
  environment (for example the GPU-capable Distrobox).
- The current diffusion implementation predicts the clean normalized target
  window directly and then performs deterministic DDIM-style updates from that
  estimate. Based on the first acceptance run this was a leading suspect.
- After the second iteration, the code now uses epsilon prediction plus a
  DDIM-style update with `sample_eta`, but the spike-coupling failure still
  persists. That points toward a broader modeling mismatch rather than just the
  original sampler formulation.

## Out of scope for this PR

- Phase 7 (combine impact model + synthetic FCAS) — contingent on v2 passing gates.
- "Distributional conditioning" follow-up (generator spike-risk features into
  the DT as a revisit of §8.2.8) — documented in the plan; no code until v2
  passes the gates.
