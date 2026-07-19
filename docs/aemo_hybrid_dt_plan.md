# AEMO Hybrid DT: SDP-Forecast Informed Decision Transformer

## Status: In Progress (New PR)

This plan supersedes `docs/next_stage_instructions.md` (completed — see Summary of Prior Work below) and
consolidates the next research phase: **bridging the SDP paper's forecast-aware planning into the modern
Decision Transformer** for the AEMO utility-scale battery trading environment.

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

## Primary Objective: Energy-SDP Trajectories in DT Training Data (Path B')

### What

Run the existing SDP/MRDP solver on the AEMO environment in `action_mode='simple'` (1D energy
charge/discharge) to generate **provably optimal energy arbitrage trajectories**. Add these
trajectories to the FCAS-rich training dataset and retrain the DT.

### Why

The modern DT at rtg=0.0 earns $10,138/ep on dispatch-matched, but $10,068 of that is FCAS revenue.
Energy arbitrage accounts for only ~$70/ep. The SDP can compute the *theoretically optimal* energy
arbitrage schedule against historical spot prices — potentially adding $500-2,000/ep to the mix.
The DT can learn SDP's energy timing while keeping its FCAS bidding from PPO demonstrations.

### How

1. **Adapt `sdp_algorithm.py` to AEMO**:
   - Replace household variables (SolarGen, HouseLoad) with AEMO variables (RRP, TOTALDEMAND)
   - The reward function becomes: maximize `energy_revenue = energy_dispatch × RRP` minus degradation
   - The `QuantileScenarioGenerator` generates scenarios from historical RRP and FCAS price columns

2. **Generate SDP trajectory logs**:
   - Run SDP solver on AEMO data for each region × time period × battery config
   - Convert resulting optimal energy actions to trajectory format (parquet)
   - Combine with existing FCAS-rich dataset

3. **Retrain modern v2 DT** (8×768, full_fcas) on the augmented dataset
   - The DT learns optimal energy timing from SDP trajectories
   - FCAS capability retained from PPO demonstrations

4. **Evaluate** on dispatch-matched + standard surfaces

### Effort: ~4h generate + retrain
### Expected gain: $500-2,000/ep

---

## Secondary Objectives

### 1. SDP-Computed RTG (Quick Win)

**What**: Replace the fixed RTG prompt (typically 0.0 or 0.5) with the SDP's cost-to-go value computed
at the battery's current state at inference time.

**Why**: The SDP outputs the exact expected optimal return-to-go for any SoC at any timestep. Using
this as the DT's RTG gives the model a *forecast-informed*, *time-varying*, *feasible* target instead of
a hand-tuned scalar.

**How**: At each DT inference step, run SDP forward (fast — cached value function), extract the
cost-to-go at current SoC, convert to RTG, and feed it to `get_action()`. No retraining needed.

### 2. Hierarchical Inference: SDP Energy + DT FCAS (Path C')

**What**: At inference time, split the 9D action: SDP produces the energy dispatch (dim 0), DT
produces the FCAS bids (dims 1–8).

**Why**: Energy arbitrage is a convex optimization problem that SDP solves optimally. FCAS bidding
is a learned pattern that DT excels at. Together they cover both revenue streams without the DT
having to learn energy timing from scratch.

**How**: SDP runs alongside DT at each step. The DT receives the SDP's energy recommendation as an
extra input channel.

### 3. Forecast-Conditioned DT Architecture (Path A1)

**What**: Add a forecast token stream to the DT that interleaves predicted future market states
alongside historical observations. The model attends to both past prices (in context window) and
future forecasts (explicit predictions).

**Why**: This is the most fundamental improvement — it removes the information bottleneck of pure
history conditioning, giving the DT explicit forward-looking information.

**How**: Extend the DT's interleaved token sequence `(rtg_t, state_t, action_t)` with a second
segment `(forecast_rtg, forecast_state, None)` covering the next N timesteps of predicted prices.
Augment token type embeddings to distinguish observation vs forecast. Retrain on the augmented
dataset.

### 4. Remaining Items from `docs/dt_improvement_roadmap.md`

These were deferred from the previous roadmap and remain relevant:

- **FCAS-weighted loss**: Higher training weight on FCAS action dimensions (currently all 9 dims
  treated equally). Expected to improve FCAS bid quality further.
- **Longer context sweep**: Re-test context lengths (288, 576, 1008, 2016) on the modern v2
  model with FCAS-rich data. The earlier sweep used the legacy 8×384 model.
- **Multi-round self-improvement**: Generate new rollouts from improved model → append to dataset
  → retrain → repeat. Requires successful primary objective first.

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

| # | Milestone | Dependencies | Est. effort |
|---|-----------|-------------|:-----------:|
| 1 | AEMO-adapted SDP (`action_mode='simple'`) verified on 1 region | None | 2h |
| 2 | SDP trajectory logs generated (3 regions × 2 batteries) | Milestone 1 | 4h |
| 3 | DT retrained on augmented dataset (SDP + FCAS-rich) | Milestone 2 | 6h |
| 4 | Evaluate on dispatch-matched + standard | Milestone 3 | 2h |
| 5 | SDP-computed RTG inference (quick win) | Milestone 1 | 2h |
| 6 | Hierarchical SDP+DT inference (if primary underperforms) | Milestones 1, 3 | 4h |

---

## Files to Read for This Phase

| File | Why |
|------|-----|
| `src/sdp_algorithm.py` | SDP solver to adapt for AEMO energy-only optimization |
| `src/mrdp_algorithm.py` | Multi-resolution variant (better for long horizons) |
| `src/decision.py` (`AEMOAgent`) | Agent wrappers for DT + SDP integration |
| `src/quantile_scenarios.py` | Scenario generator — already works on Polars DataFrames |
| `src/decision_transformer.py` | Modern v2 DT (8×768 GQA) — target architecture |
| `docs/DP_ALGORITHM_README.md` | SDP/MRDP algorithm deep dive |
| `docs/modern_transformer_improvements.md` | Architecture reference |
| `configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json` | Primary eval config |
| `configs/eval_tier_standard.json` | Secondary eval config |
