# AEMO Decision Transformer Roadmap

This document is a research roadmap, not a stable onboarding or operations guide.

Use it when you need:

- historical and current AEMO DT findings
- open research questions
- candidate next experiments

For current workflow and setup guidance, start with [README.md](README.md), [architecture.md](architecture.md), [development.md](development.md), and [aemo/README.md](aemo/README.md).

## Executive summary

This roadmap consolidates the DT and GRPO work for AEMO trading. The central question is whether offline Decision Transformer training can match or exceed online RL on energy-plus-FCAS bidding, and whether GRPO fine-tuning can improve the pretrained policy further without sacrificing safety or profitability.

## Current state (July 2026)

### Dataset and training context

- Training data v1: 2,425 episodes across 5 regions, 78.4M rows, 0.5C battery.
- Training data v2: 2,401 episodes, 77M rows, 4 realistic battery configurations (1C, 0.7C, 0.5C, 3.75C).
- Source policies: PPO, A2C, DDPG, SAC, TD3, and FCAS rule.
- Best DT baseline: v2 HF model with roughly $1,714/ep profit and $2,743/ep FCAS revenue.
- Best GRPO result: multi-region setup with $4,357/ep profit and strong FCAS revenue.

### Evaluation results

#### Expanded evaluation (135 episodes, 5 regions, 6 periods, 12-day episodes)

| Policy | Mean Reward | Profit/Ep | FCAS Rev | Deg Cost | Sharpe |
|--------|:-----------:|:---------:|:--------:|:--------:|:-----:|
| PPO (online RL) | +12.82 | +$12,839 | $10,628 | $1,458 | +1.26 |
| DT full-pretrained (8×512, ctx=180) | -3.11 | -$1,396 | $77 | $2,503 | -0.40 |
| Rule heuristic | -4.82 | -$3,562 | $0 | $15,400 | -0.48 |
| DT old pretrain (4×128, ctx=1152) | -13.55 | -$10,620 | $2,328 | $12,975 | -2.20 |

#### FCAS-rich dataset evaluation (16 episodes, 4 regions, 144h episodes)

| Policy | Mean Reward | Profit/Ep | FCAS Rev | Deg Cost | Sharpe |
|--------|:-----------:|:---------:|:--------:|:--------:|:-----:|
| DT (8×384, FCAS-rich) | -1.31 | +$1,522 | $1,383 | $212 | -1.07 |
| PPO (online RL) | -1.35 | +$1,444 | $1,616 | $609 | -1.01 |
| Rule heuristic | -3.03 | -$2,477 | $0 | $3,998 | -1.26 |
| FCAS rule | -4.24 | -$3,569 | $146 | $4,764 | -0.80 |

## What we learned

### 1. FCAS gap is largely closed

The historical FCAS gap was severe: PPO earned about $10,628/ep in FCAS revenue while the old DT earned only $77/ep. After retraining on FCAS-rich data, the DT reached about $1,383/ep in FCAS revenue, close to PPO at $1,616/ep. The gap is no longer a fundamental blocker.

### 2. Offline DT can learn FCAS behavior from demonstrations

The DT learned to bid in FCAS markets from offline trajectories, including PPO-generated rollouts. That means FCAS bidding is not necessarily limited to online interaction.

### 3. Remaining gap is likely in conditioning and reward shaping

The best remaining opportunities are RTG calibration, FCAS-aware loss weighting, longer context, and degradation-aware GRPO reward shaping.

### 4. GRPO helps, but degradation must be controlled

GRPO-side updates can increase FCAS revenue substantially, but excessive degradation can erase the gain. The objective should remain profit-oriented while preserving safe battery operation.

## Prioritized work

### Phase 1: GRPO-side improvements (no retraining required)

These changes target the current training pipeline and can run on the local RTX 2080 Ti.

1. Periodic reference model sync
   - Sync the reference model to the current policy every few iterations to avoid KL-driven collapse after early gains.

2. Larger rollout groups
   - Increase group size from the current small default to 8 or 16 to reduce variance in advantage estimation.

3. Adaptive RTG sampling
   - Sample RTG prompts using the realized return distribution rather than a fixed Gaussian around an overly generic scale.

4. Degradation-weighted reward shaping
   - Penalize degradation more strongly during GRPO training so the policy keeps FCAS gains without excessive battery wear.

5. Combined training run
   - Validate all Phase 1 changes together with a longer multi-iteration run before moving to retraining.

### Phase 2: Full retraining (requires MoLab or equivalent GPU)

1. Multi-round self-improvement
   - Use GRPO-improved rollouts as new training data, retrain the DT, and repeat until performance stabilizes.

2. Reward restructuring
   - If degradation remains too high, update the pretraining reward at the environment level and regenerate the dataset before retraining.

## Implementation notes

- The current best context length for the earlier setup was 180, but longer context lengths such as 288, 576, 1008, or 2016 should be retested on the FCAS-rich corpus.
- The current best action focus is on energy and FCAS dimensions; FCAS-aware loss weighting may improve the remaining gap.
- Evaluation should continue to use the mini, example, expanded, dispatch-matched, and held-out configs so the results are comparable across settings.
- Existing evaluator baselines include rule, FCAS rule, dispatch replay, and PPO reference policies.

## Success criteria

- Mean reward greater than 0.0 on target evaluation.
- FCAS revenue above $2,000/ep on the example evaluator and above $4,000/ep on stronger setups.
- Profit per episode above $2,000/ep.
- Beat the FCAS rule baseline in paired comparison.
- Keep degradation controlled so the gains are not offset by battery wear.

## Open questions

1. ~~Does RTG conditioning work as a reliable strategy selector for AEMO environments?~~ **Resolved July 2026.**
   RTG is a strong strategy selector. Every DT variant (pretrained, GRPO, forecast) responds to higher RTG values, with gains of 5-75% over the default 0.0-0.5 range. The original calibration range was too narrow — the peak is often at RTG=10-50 depending on the model's return_scale. See leaderboard below.

2. Does the FCAS-rich result hold on the expanded evaluator at full scale?

3. Can FCAS-aware loss weighting or longer context close the remaining margin to PPO?

4. How many GRPO iterations are sustainable before the policy drifts from the pretrained reference?

5. **TTM forecast quality**: can diverse few-shot fine-tuning improve TTM forecast accuracy and thereby boost forecast DT profit? (Currently $4,564/ep vs Mv2's $4,991/ep.)

6. **Forecast quality measurement**: no metric currently exists to quantify TTM forecast accuracy. Need per-channel MAE/RMSE against actual normalized observations.

## Next Priority Items

| # | Item | Why | Approach |
|---|------|-----|----------|
| 1 | **RTG calibration for all models** | Gain 5-75% profit immediately; the 0.0-2.0 range was too narrow | Add RTG sweep script; document optimal values per model; integrate into evaluator |
| 2 | **Re-fine-tune TTM with diverse time sampling** | Current TTM trained on 14K rows of early 2021 only; misses market diversity | Replace `fewshot_location="first"` with curated three-season sampling in `ttm_finetune_and_forecast.py` |
| 3 | **Forecast quality metrics** | Can't tell if TTM forecasts are good enough; need error baseline | Compare npz `forecast_map` vs actual `processed_*_0.0833h.parquet` values; report MAE/each channel |
| 4 | **Longer context sweep** | Re-test ctx=288, 576, 1008 on modern v2 8x768 with FCAS data | Use existing context-sweep infrastructure |
| 5 | **FCAS-weighted loss** | Add training weight on FCAS action dims to encourage bidding | Modify pretrain loss with per-dim weights |

## Standard Tier Leaderboard (Oct 2024, 5 regions, 144h, medium_1c)

| Model | Profit/ep | FCAS/ep | Deg/ep | Best RTG |
|---|:---:|:---:|:---:|:---:|
| Modern v2 pretrained | **$4,991** | $4,836 | $229 | 10.0 |
| Dispatch Dalrymple North | $4,660 | $2,287 | $1,020 | — |
| Forecast DT (normalized) | $4,564 | $3,663 | $270 | 50.0 |
| Phase C GRPO (mod v2) | $4,322 | $2,508 | $1,058 | 10.0 |
| Modern v2 (default rtg=0.5) | $4,726 | $3,063 | $232 | 0.5 |
| PPO reference | $2,353 | $2,192 | $236 | — |
| Phase 1 GRPO (legacy) | $2,678 | $2,914 | $384 | 50.0 |

All models evaluated on the same 5 regions × 144h Oct 2024 episodes with medium_1c battery.
The forecast DT was trained on FCAS-rich + SDP + GRPO data with TTM forecast tokens
and the npz normalized to [0,1] matching the observation space.

## Status summary

- FCAS-rich dataset assembled: done.
- FCAS rule baseline implemented: done.
- GRPO post-training pipeline implemented: done.
- Forecast DT evaluator integration: done.
- Forecast DT evaluated: done (3rd place at $4,564/ep with rtg=50).
- TTM npz normalized to [0,1] (was raw — fixed July 2026): done.
- RTG calibration sweep (0.0-50.0) for all models: done.
- Full retraining and dataset regeneration: deferred until Phase 1 is exhausted.
