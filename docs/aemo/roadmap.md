# AEMO Decision Transformer Roadmap

## Current State

- **Training data (v1)**: 2,425 episodes across 5 regions (2021-2023), 78.4M rows — all at 0.5C (2h) battery
- **Training data (v2)**: 2,401 episodes, 77M rows — **4 realistic battery configs** (1C, 0.7C, 0.5C, 3.75C)
- **Source policies**: PPO, A2C, DDPG, SAC, TD3, FCAS rule
- **Best DT baseline**: v2 HF model ($1,714/ep profit, $2,743 FCAS/ep) — 2× better than v1 ($874/ep)
- **Best GRPO result**: Multi-region (NSW1+SA1+QLD1), 5 iter, +1.60 reward, $4,357/ep profit
- **GRPO on v2 model**: +$1,885/ep (+$171 vs baseline), +$4,033 FCAS (+47% improvement)
- **Evaluation**: 4-tier configs (mini/example/expanded/dispatch_matched) + Q4 2024 held-out multi-station
- **GPU**: 22GB VRAM (RTX 2080 Ti) — power capped at 205W to prevent Xid79 crashes
- **Baselines**: rule, fcas_rule, dispatch replay (Dalrymple North, Torrens Island, Hornsdale), PPO reference

## Priorities

### 1. Multi-Round Iterative Self-Improvement (Highest Impact)

Inspired by DeepSeek-R1's self-improvement pipeline, the key limitation is that GRPO fine-tunes the same model on the same fixed dataset. Each GRPO iteration improves the policy slightly, but without new data the improvements plateau.

Proposed pipeline:
```
Round 1: GRPO on v2 model → grpo_v1 model (DONE, +$1,885/ep)
Round 2: Run grpo_v1 on env → generate NEW rollouts
         Append to training dataset
         Retrain DT from scratch (MoLab or locally)
Round 3: GRPO on retrained DT → grpo_v2
Repeat Rounds 2-3 until convergence
```

This addresses the fundamental data limitation — the model generates its own improved training data, bootstrapping to higher performance.

### 2. Larger Group Sizes for Better Advantage Estimation

- Current: `group_size=4`, `rtg_count=4` → only 4 samples per advantage group
- Proposed: increase to 8-16 (DeepSeek-R1 uses 64)
- Trade-off: training time scales linearly (8× = 40 min, 16× = 80 min)
- Benefit: much lower variance in advantage estimates → stable updates

### 3. Adaptive RTG Based on Realized Returns

- Currently: RTG is sampled from a fixed distribution around `return_scale`
- Proposed: after each iteration, use the actual mean return from rollouts as the next iteration's optimal RTG
- Ensures RTG targets stay realistic and achievable
- Low effort: ~1h code change

### 4. Degradation-Weighted Reward Shaping

- Current reward: `energy_rev + fcas_rev - deg_cost` (all equal weight)
- GRPO teaches FCAS bidding but also increases degradation 1.7×
- Proposed: `modified_reward = energy_rev + fcas_rev - lambda * deg_cost` with lambda > 1
- Explicitly penalizes battery wear, teaching the model to balance FCAS vs battery life
- Low effort: ~30min environment change

### 5. Dynamic KL / Reference Model Sync

- Current: fixed KL coefficient (0.02), frozen reference model
- Problem: after 5 iterations, policy drifts → KL penalty fights further improvement
- Proposed options:
  a) Adaptive KL: increase coefficient when divergence exceeds threshold
  b) Periodic reference sync: copy policy → reference every 5 iterations
- Enables 20+ stable iterations instead of plateauing at 5

## What's Done

| Item | Status |
|------|--------|
| 2,425-episode FCAS dataset (v1) | ✅ |
| 2,401-episode realistic battery dataset (v2) | ✅ |
| 5-region data fetch | ✅ |
| FCAS rule algorithm | ✅ |
| FCAS-aware evaluator configs | ✅ |
| GRPO post-training pipeline (`src/grpo_posttraining.py`) | ✅ |
| Multi-region GRPO training | ✅ |
| GRPO hyperparameter sweep (21 experiments) | ✅ |
| Legacy DT checkpoint compatibility | ✅ |
| PPO comparison eval config | ✅ |
| Dispatch-matched multi-station eval | ✅ |
| Documentation refresh | ✅ |

## Success Criteria

- Mean reward > 0.0 (positive profit)
- FCAS revenue > $4,000/ep (indicates strong FCAS bidding)
- Beat dispatch replay baseline in paired comparison (mean_diff > 0)
- Degradation < $1,500/ep (balanced operation)
