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

## Current status (July 2026)

- Phase 1 GRPO-side work is now implemented and validated: degradation-weighted reward shaping, periodic reference-model syncing, adaptive RTG sampling, and larger rollout groups are available in the current CLI workflow.
- The evaluation framing has been corrected to use dispatch-matched, same-asset comparisons with per-MWh reporting so the results are easier to compare across batteries and policies.
- The current best RTG prompt for the Phase 1 model is 0.5 on the dispatch-matched setup, and the remaining frontier is reducing degradation while keeping the FCAS gains.

## Principles

All priorities are classified by **training dependency**:
- **Phase 1 (GRPO-side)** — modifies only `grpo_posttraining.py` / training scripts. No SB3 retraining, no DT pretraining, no dataset regeneration. Can run entirely on the local RTX 2080 Ti.
- **Phase 2 (Full retrain)** — requires DT pretraining on MoLab RTX 6000 Pro. These are higher-effort and should only be attempted after Phase 1 improvements are exhausted.

## Phase 1: GRPO-Side Improvements (No Retraining Required)

### 1. Periodic Reference Model Sync

**Problem:** After ~5 GRPO iterations, the policy has diverged from the reference model. The KL penalty then fights further improvement, causing the plateau we observed.

**Solution:** Periodically sync the reference model to the current policy. This lets the model continue improving without being pulled back to the original.

```python
# Every N iterations:
self.reference_model = copy.deepcopy(self.model)
self.reference_model.eval()
```

- **Impact**: Enables 20+ stable iterations instead of plateauing at 5
- **Effort**: ~1h code change in `grpo_posttraining.py`
- **Risk**: Too-frequent sync defeats the purpose of KL regularization. Start with sync every 5 iterations.

### 2. Larger Group Sizes for Better Advantage Estimation

**Problem:** The current `group_size=4` with `rtg_count=4` produces only 4 samples per group-relative advantage computation. This is noisy — DeepSeek-R1 uses 64.

**Solution:** Increase `--group-size` to 8 or 16.

- **Impact**: Much lower variance in advantage estimates → more stable gradient updates
- **Effort**: Just a CLI flag change (`--group-size 8`)
- **Trade-off**: Training time scales linearly with group size (8× = ~40 min, 16× = ~80 min for 5 iterations)

### 3. Adaptive RTG Based on Realized Returns

**Problem:** RTG values are sampled from a fixed Gaussian distribution around `return_scale=1.0`. Values like `[1.0, -1.38, 1.72, -4.69]` include negative targets that may confuse the model (the pretrained DT was trained on positive returns).

**Solution:** After each GRPO iteration, use the actual mean return from that iteration's rollouts as the next iteration's optimal RTG:

```python
mean_return = float(batch.returns.mean())
rtg_values = sample_rtg_values(optimum=mean_return, spread=2.0, count=4)
```

- **Impact**: RTG targets stay realistic and achievable. The model always sees feasible targets.
- **Effort**: ~1h code change in `grpo_posttraining.py`
- **Risk**: Low — if returns are noisy, the RTG may jump around. Smooth with EWMA: `rtg = 0.9 * rtg + 0.1 * mean_return`.

### 4. GRPO-Side Degradation-Weighted Reward

**Problem:** GRPO increases FCAS revenue (+47%) but also increases degradation (+67%). The net profit gain is modest (+$171/ep) because FCAS gains are eaten by battery wear.

**Solution:** Inside the GRPO rollout loop, modify the reward signal to penalize degradation more:

```python
# In GRPOTrainer.collect_rollouts(), after env.step():
deg_cost = info.get("current_step_deg_cost", 0.0)
reward = float(reward) - 0.5 * float(deg_cost)  # extra 50% degradation penalty
```

This does NOT change the environment's reward — it only changes the signal GRPO sees during training. The model learns to avoid actions that cause degradation, while still pursuing FCAS revenue.

- **Impact**: Explicitly shapes the model toward conservative operation
- **Effort**: ~30min code change in `grpo_posttraining.py`. No SB3/DT retraining needed.
- **Risk**: Too high a penalty kills FCAS learning. Start with +50% penalty (weight=1.5).

### 5. Combined Run: All Phase 1 Improvements Together

Once the individual changes are implemented, run a full GRPO training with:
```bash
python3 src/run_grpo_multi_region.py \
  --iterations 20 \
  --group-size 8 \
  --kl-coeff 0.02 \
  --dt-gamma 0.95 \
  --sync-reference-every 5 \
  --adaptive-rtg \
  --deg-penalty-weight 1.5
```

The Phase 1 options are now wired through the multi-region runner:
- `--sync-reference-every N` periodically copies the policy weights into the reference model.
- `--adaptive-rtg` resamples RTG prompts after each iteration using an EWMA-smoothed mean return.
- `--adaptive-rtg-ewma-alpha` controls the smoothing strength for adaptive RTG updates.
- `--deg-penalty-weight` applies a degradation-shaped reward penalty to the GRPO training signal without changing the environment reward.

Expected outcome: stable training for 20+ iterations, higher FCAS revenue, lower degradation.

## Phase 2: Full Retrain (Requires MoLab RTX 6000 Pro)

### 6. Multi-Round Iterative Self-Improvement

Inspired by DeepSeek-R1's self-improvement pipeline. Once Phase 1 GRPO is exhausted:

```
Round 1: GRPO on v2 model → grpo_v1 (Phase 1)
Round 2: Run grpo_v1 on env → generate NEW rollouts (2021-2023 data)
         Append to existing v2 dataset
         Retrain DT from scratch on combined dataset  ← MoLab
         GRPO Phase 1 on retrained DT → grpo_v2
Repeat until convergence
```

This addresses the fundamental data limitation: the DT can only be as good as its training data. By using GRPO-improved rollouts as training data for the next DT generation, each round bootstraps to higher performance. This is the mechanism behind DeepSeek-R1's breakthrough.

### 7. Full Environment Reward Restructuring

If Phase 1's degradation-weighted GRPO reward is insufficient, a more thorough approach is to change the environment's reward function at the SB3/DT pretraining level:
- Tighten `battery_life_cost` parameter
- Add explicit cycle-count penalty
- Retrain SB3 models on the new reward  ← MoLab
- Regenerate the v3 pretraining dataset  ← MoLab
- Retrain DT from scratch  ← MoLab

This is a last resort because it requires the full retraining pipeline.

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
