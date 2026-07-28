# AEMO Hybrid DT: SDP-Forecast Informed Decision Transformer

This document is a forward-looking research plan.

Use it when you need:

- the rationale for forecast-conditioned DT work
- a record of the next-stage research direction
- implementation ideas that may not yet be productized or standardized in the repo

For stable usage and development guidance, use [aemo/README.md](aemo/README.md), [aemo/workflow.md](aemo/workflow.md), and [development.md](development.md).

## Status: In Progress (PR #32)

This plan supersedes an earlier now-removed planning note and
consolidates the next research phase: **bridging the SDP paper's forecast-aware planning into the modern
Decision Transformer** for the AEMO utility-scale battery trading environment.

### Current Progress (Aug 2026)

| Milestone | Status | Notes |
|---|---|---|
| AEMO-adapted SDP solver + trajectory generator | ✅ Done | 1200 episodes, 5 regions × 2 batteries, 2022–2023 |
| Forecast token architecture + tests | ✅ Done | `ForecastDecisionTransformer` with RoPE, 48-step prefix, 15/15 tests |
| TTM fine-tuning + forecast data generation | ✅ Done | Granite TTM-R3 fine-tuned on 6 price channels, 420K × 48 × 6 forecast lookup |
| Forecast DT trained on MoLab | ✅ Done | Model uploaded to `mrvictoru/energydecision-dt-v2-forecast` |
| HF datasets enriched + uploaded | ✅ Done | All trajectory datasets have `episode_start` for TTM alignment |
| **Evaluator integration for forecast DT** | ⏳ **Next** | `AEMOAgent` needs forecast buffer; evaluator needs `ForecastDecisionTransformer` support |
| Evaluate forecast DT | ❌ Not started | Requires evaluator integration first |

---

## Summary of Prior Work (PR#31 — Completed)

All experiments from the previous session are finished:

| Milestone | Result |
|---|---|
| Modern v2 DT (8×768 GQA) pretrained | SOTA — $10,138/ep dispatch-matched, $4,630/ep standard |
| GRPO RTG overflow fixed (`stable_rtg_update`) | gamma<1 now safe on long horizons |
| GRPO Phase C (144h, 3 regions) | Closes to within 5-11% of pretrained but does not surpass |
| Overfitting hypothesis confirmed | Legacy Phase 1 GRPO drops to $1,533/ep on standard surface |
| Gradient accumulation + skip-ref-forward | Memory-safe training on 22 GB GPU |
| RTG calibration difference found | Modern peaks at rtg=0.0 (vs legacy at 0.5) |

**Conclusion**: The modern pretrained model is SOTA. GRPO does not improve it. The next frontier is
bringing **explicit forecast awareness** into the model.

---

## Motivation

The SDP paper (Abdulla et al. 2016, `docs/references/Optimal_Operation_of_Energy_Storage_Systems_Consid.pdf`)
achieves optimal battery control via:

1. **Explicit forecasts** of future solar, load, and prices — re-planning at each step
2. **Backward induction** — the Bellman equation guarantees optimality under the discretization
3. **Monte Carlo scenarios** — uncertainty modeled via quantile distributions

The modern DT beats all baselines on both evaluation surfaces, but it does **not** receive forecasts — it
relies entirely on a 210-step context window to infer future market conditions. This is a fundamental
information bottleneck.

The AEMO environment is a particularly good fit for this approach because:
- The market has well-defined **price signals** (RRP + 8 FCAS services) that can be forecast
- The `QuantileScenarioGenerator` (`src/quantile_scenarios.py`) already handles Polars DataFrames
- The SDP solver (`src/sdp_algorithm.py`) is fully implemented and tested
- No household solar/load uncertainty — just market prices

---

## Primary Objective: SDP Trajectories + Forecast-Conditioned Architecture (Path B' + Path A1)

The modern DT at rtg=0.0 earns $10,138/ep on dispatch-matched, but $10,068 of that is FCAS revenue.
Energy arbitrage accounts for only ~$70/ep. The DT has no explicit forecast mechanism — it relies
entirely on a 210-step context window to infer future market conditions.

The primary objective combines two complementary thrusts:

### Thrust 1: Energy-SDP Trajectories in Training Data (Path B')

Run the existing SDP/MRDP solver on the AEMO environment in `action_mode='simple'` (1D energy
charge/discharge) to generate **provably optimal energy arbitrage trajectories**. Add these
trajectories to the FCAS-rich training dataset.

**Why**: SDP computes the *theoretically optimal* energy arbitrage schedule against historical
spot prices — potentially adding $500-2,000/ep to the mix. The DT can learn SDP's energy timing
while keeping its FCAS bidding from PPO demonstrations.

**How**:
1. **Adapt `sdp_algorithm.py` to AEMO**: Replace household variables (SolarGen, HouseLoad) with
   AEMO variables (RRP, TOTALDEMAND). The reward function becomes `max(energy_dispatch × RRP) -
   degradation`. The `QuantileScenarioGenerator` generates scenarios from historical RRP and FCAS
   price columns (already works on Polars DataFrames).
2. **Generate SDP trajectory logs**: Run SDP for each region × time period × battery config, convert
   optimal energy actions to Parquet format, combine with existing FCAS-rich dataset.
3. **Retrain DT** on the augmented dataset — learns optimal energy timing from SDP, retains FCAS
   bidding from PPO demonstrations.
4. **Evaluate** on dispatch-matched + standard surfaces.

**Effort**: ~4h generate + retrain. **Expected gain**: $500-2,000/ep.

### Thrust 2: Forecast-Conditioned DT Architecture (Path A1)

Add a forecast token stream to the DT that interleaves predicted future market states alongside
historical observations. The model attends to both past prices (in context window) and future
forecasts (explicit predictions).

**Why**: This removes the information bottleneck of pure history conditioning, giving the DT
explicit forward-looking information. The SDP paper's core advantage is forecast-informed
planning — this brings that capability directly into the transformer.

**How**:
1. **Extend the token sequence**: The DT currently processes interleaved `(rtg_t, state_t, action_t)`
   for historical timesteps. Add a second segment `(forecast_rtg, forecast_state, None)` covering
   the next N timesteps of predicted market conditions (RRP, FCAS prices, demand).
2. **Token type embeddings**: Add learnable embeddings to distinguish observation vs forecast tokens.
3. **Attention mask**: Extend the causal mask so historical tokens attend to forecast tokens.
4. **Forecast source**: The `QuantileScenarioGenerator` produces price scenarios from historical data.
   Even a simple persistence forecast (next 24h ≈ last 24h) would be a strong baseline.
5. **Train the augmented DT** on the existing FCAS-rich dataset (no SDP trajectories needed for this
   thrust, though they can be combined).

**Effort**: ~1 week (architecture + retraining). **Expected gain**: $1,000-4,000/ep.

---

## Forecast Generation Strategy

Thrust 2 (forecast tokens) requires realistic forecasts that the model learns to trust without
cheating. This section defines how forecasts are generated, how many timesteps are used, and how
they interact with the DT's context window.

### 1. Three-Phase Forecast Maturity

| Phase | Forecast source | Realism | Purpose |
|---|---|---|---|
| **1: Perfect foresight (validate)** | Actual future values from `aemo_data` | ❌ Unrealistic but fast | Prove the architecture works — if the model can't exploit perfect forecasts, the design is wrong |
| **2: Statistical forecast (baseline)** | `QuantileScenarioGenerator` (already in repo) | ⚠️ Honest, imperfect | Minimum viable forecast. The model learns to work with uncertainty because forecasts are wrong sometimes |
| **3: Learned forecast (current focus)** | Fine-tuned time-series model (e.g. IBM Granite TTM, 1.41M params, Apache 2.0) | ✅ Realistic | Closer to what a real operator would use; forecast quality is bounded by the model |

The current focus is Phase 3 (skipping ahead to a learned forecaster). The `ForecastDecisionTransformer`
architecture is already built and validated (15/15 tests pass). The immediate next step is to generate a
**forecast-augmented dataset**: for every step in the existing trajectory parquets, append the next
48 timesteps of predicted market conditions (RRP, FCAS prices, demand) as forecast tokens generated
by IBM Granite TTM.

#### TTM Fine-Tuning: Diverse Few-Shot Sampling

The current TTM fine-tuning uses `fewshot_location="first"` on 5% of training data
(only ~14K rows from Jan-Feb 2021). This undersamples market diversity. A simple
improvement is to curate a diverse few-shot set spanning multiple seasons and years:

```python
# Instead of: get_datasets(tsp, pdf, split_config, fewshot_fraction=0.05, fewshot_location="first")
# Sample from specific date ranges across different seasons:
fewshot_indices = (
    list(range(start_idx_apr_2021, end_idx_apr_2021)) +  # 1 week spring
    list(range(start_idx_jun_2022, end_idx_jun_2022)) +  # 2 weeks summer
    list(range(start_idx_dec_2023, end_idx_dec_2023))    # 1 week winter
)
# Pass as fewshot_indices or build custom dataset
```

This exposes the TTM to multiple market regimes within the same 5% few-shot
budget, producing better generalisation across the full 2021-2025 range. See
`scripts/ttm_finetune_and_forecast.py` for the current implementation.

The TTM model runs on CPU/GPU and produces forecasts from historical price context. The output is a
new parquet with the same schema as the original trajectory data, plus forecast columns appended as
extra lists. The `ForecastTrajectoryDataset` reads these and yields both history and forecast windows.

### 2. Forecast Horizon

| Horizon | Steps (5-min) | What it captures | Added tokens (3×) |
|---|---:|---|:---:|
| 1 hour | 12 | One charge cycle for 1C battery | 36 |
| **2 hours** | **24** | Full charge/discharge cycle | 72 |
| **4 hours** | **48** | Morning/evening ramp + 2–4 cycles | 144 |
| 8 hours | 96 | Day-ahead pattern (SDP default) | 288 |

**Recommendation: 48 forecast steps (4 hours).** Because:

- A medium_1c battery charges in ~60 minutes (12 steps). A fast_375c cycles in ~16 minutes (3–4 steps).
  48 steps covers multiple full cycles regardless of battery config.
- Beyond 4 hours, forecast quality degrades and the battery is physically constrained by its capacity
  anyway — a full battery at hour 2 means the hour-8 forecast cannot change the hour-2 action.
- With GQA (12 heads, 6 KV heads), the added attention memory for 48 forecast steps is ~110 MB
  on top of the existing model — negligible on 22 GB.

### 3. Interaction with Context Window

**Design**: Forecast tokens are **prepended as a prefix** before the existing history tokens.
Causal attention means history tokens attend to all forecast tokens + prior history.

```
Position:  [forecast_rtg_0, forecast_s_0, pad_0, ... forecast_rtg_47, forecast_s_47, pad_47,
            rtg_0, s_0, a_0, ... rtg_209, s_209, a_209]
           |<----- 3 × 48 = 144 forecast tokens ------>|<------- 3 × 210 = 630 history tokens ------>|
```

Total sequence: 144 + 630 = **774 tokens** (vs 630 without forecasts).

| Option | History | Forecast | Total | Trade-off |
|---|---:|---:|---:|:---|
| **A: Fixed total** | 162 | 48 | 630 | Sacrifices 48 history steps for forecasts. Loses 4h of market context. |
| **B: Extended (recommended)** | 210 | 48 | 774 | Full 17.5h history + 4h forecast. 774 tokens fits in 22 GB. |

Option B is preferred. The 144 extra tokens increase attention complexity from O(630²) to O(774²)
but with GQA and 768-dim this is ~110 MB additional memory — well within the 22 GB budget.

The `context_len` parameter increases from 210 to 258 (210 history + 48 forecast). A separate
`forecast_len` parameter tracks the split. The action-prediction head reads from the last history
action position — unchanged. Forecast action positions are zero-padded and masked from the loss.

### 4. No Environment Changes

The forecast tokens are constructed **at DT input time** by reading future values from the
`aemo_data` DataFrame (which already covers the full episode at both training and inference time).
The environment's 18D observation space stays unchanged. Other agents (PPO, dispatch, rule) are
unaffected — only the DT receives forecast tokens.

At training time, the forecast is extracted from the logged trajectory's future steps (trivially
available). At inference time, the `aemo_data` for the loaded episode provides the same look-ahead.

---

## Secondary Objectives

### 1. SDP-Computed RTG (Quick Win)

**What**: Replace the fixed RTG prompt (typically 0.0 or 0.5) with the SDP's cost-to-go value computed
at the battery's current state at inference time.

**Why**: The SDP outputs the exact expected optimal return-to-go for any SoC at any timestep. Using
this as the DT's RTG gives the model a *forecast-informed*, *time-varying*, *feasible* target instead of
a hand-tuned scalar. No retraining needed.

### 2. Hierarchical Inference: SDP Energy + DT FCAS (Path C')

**What**: At inference time, split the 9D action: SDP produces the energy dispatch (dim 0), DT
produces the FCAS bids (dims 1–8).

**Why**: Energy arbitrage is a convex optimization problem that SDP solves optimally. FCAS bidding
is a learned pattern that DT excels at. Together they cover both revenue streams.

### 3. Remaining Items from `docs/dt_improvement_roadmap.md`

These were deferred from the previous roadmap and remain relevant:

- **FCAS-weighted loss**: Higher training weight on FCAS action dimensions (currently all 9 dims
  treated equally).
- **Longer context sweep**: Re-test context lengths (288, 576, 1008, 2016) on the modern v2
  model with FCAS-rich data.
- **Multi-round self-improvement**: Generate new rollouts from improved model → append to dataset
  → retrain → repeat.

---

## Evaluation Plan

All experiments use the standardised evaluation surfaces:

| Surface | Config | Regions | Hours | What it tests |
|---------|--------|:-------:|:-----:|--------------|
| Dispatch-matched | `q4_dispatch_matched` | SA1 | 144h | Same-asset head-to-head |
| Standard | `eval_tier_standard` | 5 regions | 144h | Cross-region generalisation |

Success criteria:
- Primary: DT retrained with SDP trajectories **exceeds** pretrained baseline on at least one surface
- Secondary: SDP-guided RTG improves inference-time profit without retraining
- Hard: Combined SDP+DT system exceeds $12,000/ep on dispatch-matched or $6,000/ep on standard

---

## Milestones

| # | Milestone | Thrust | Dependencies | Est. effort | Status |
|---|-----------|--------|-------------|:-----------:|:------:|
| 1 | AEMO-adapted SDP solver and trajectory generator | Path B' | None | 2h | ✅ Done |
| 2 | Forecast token architecture (ForecastDecisionTransformer) | Path A1 | None | 1 week | ✅ Done |
| 3 | 15 verification tests for forecast token mechanics | Path A1 | Milestone 2 | 2h | ✅ Done |
| 4 | TTM fine-tuning + forecast data generation | Path A1 | None | 1–2 weeks | ✅ Done |
| 5 | Train ForecastDecisionTransformer on MoLab | Path A1 | Milestone 4 | 6h | ✅ Done |
| 6 | **Integrate evaluator with ForecastDecisionTransformer** | Both | Milestone 5 | 4h | ⏳ |
| 7 | Evaluate forecast DT on dispatch-matched + standard | Both | Milestone 6 | 2h | ❌ |
| 8 | SDP-guided RTG inference (quick win) | Secondary | Milestone 1 | 2h | ❌ |
| 9 | Combined SDP trajectories + forecast DT training | Both | Milestones 4, 6 | 6h | ❌ |

### Milestone 6 Details: Evaluator Integration

The current evaluator (`scripts/autoresearch_evaluator.py`) imports `DecisionTransformer` from
`decision_transformer` and has no support for forecast token inference. To evaluate the
forecast-trained model properly, these changes are needed:

**`src/decision.py`** (~40 lines):
- Create `ForecastAEMOAgent` subclass (or add forecast mode to `AEMOAgent`) that:
  - Maintains a rolling forecast buffer (48 timesteps of future market states)
  - At each step, reads the next 48 timesteps from `env.aemo_data` (perfect foresight for eval)
  - Passes `forecast_states`, `forecast_rtgs`, `forecast_timesteps` to `model.get_action()`
  - Updates the forecast buffer after each step (slide window forward)

**`scripts/autoresearch_evaluator.py`** (~30 lines):
- Detect forecast model from `model_kwargs` (check for `forecast_len > 0`)
- Import `ForecastDecisionTransformer` instead of `DecisionTransformer` when forecast model
- Load checkpoint from state dict (MoLab saves as `model_state_dict`)
- Use `ForecastAEMOAgent` instead of `AEMOAgent` when `forecast_len > 0`
- Surface manifest should include `forecast_len` or model type field

**Later: TTM at inference** — replace perfect foresight with actual TTM forecasts from
`ttm_forecasts.npz` lookup (loaded alongside the evaluator). The model was trained on TTM
forecasts, so perfect-foresight eval creates a train-eval gap.

---

## Files to Read for This Phase

| File | Why |
|------|-----|
| `src/aemo_sdp_solver.py` | ✅ AEMO-adapted SDP solver (energy-only) |
| `scripts/generate_sdp_aemo_trajectories.py` | ✅ Parallel trajectory generator |
| `data/aemo_dt_sdp/aemo_sdp_trajectories.parquet` | ✅ Generated SDP trajectories (1200 episodes) |
| `src/sdp_algorithm.py` | Base SDP solver class |
| `src/mrdp_algorithm.py` | Multi-resolution variant (better for long horizons) |
| `src/decision.py` (`AEMOAgent`) | Agent wrappers for DT + SDP integration |
| `src/quantile_scenarios.py` | Scenario generator — already works on Polars DataFrames |
| `src/decision_transformer.py` | Modern v2 DT (8×768 GQA) — target architecture |
| `src/pretrain_decision_transformer.py` | DT training entrypoint — needs multi-dataset loading |
| `scripts/pretrain_aemo_decision_transformer.py` | AEMO DT training wrapper — needs multi-dataset loading |
| `docs/DP_ALGORITHM_README.md` | SDP/MRDP algorithm deep dive |
| `docs/modern_transformer_improvements.md` | Architecture reference |
| [`https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3`](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3) | IBM Granite TTM (Phase 3 forecast model, 1.41M params, Apache 2.0) |
| `configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json` | Primary eval config |
| `configs/eval_tier_standard.json` | Secondary eval config |
