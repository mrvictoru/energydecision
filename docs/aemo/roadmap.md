# AEMO Decision Transformer Roadmap

## Current State

- **Training data**: 2,425 episodes across 5 regions (2021-2023), 78.4M rows
- **Source policies**: PPO, A2C, DDPG, SAC, TD3, FCAS rule, old rule
- **FCAS signal**: Strong (PPO $8.98/step FCAS revenue, FCAS rule $2,941/ep)
- **Evaluation**: 3-tier configs (mini/example/expanded) with fcas_rule as reference
- **GPU**: 22GB VRAM (RTX 2080 Ti) — context up to 2016 feasible
- **Baselines**: rule, fcas_rule, dispatch replay, PPO reference

## Priorities

### 1. Establish FCAS baseline

- Train DT on FCAS dataset with best known hyperparams (8×512, ctx=180, batch=16, lr=3e-5)
- Evaluate against FCAS rule ($2,941/ep) and PPO ($10,628/ep)
- Target: beat FCAS rule in mean reward and FCAS revenue

### 2. RTG prompt calibration

- Sweep RTG values (0, 5, 10, 15, 20) on the best checkpoint
- RTG sensitivity was observed in household DT — test on AEMO

### 3. Loss weight calibration

- Grid sweep (June 2026) found: `action_loss_weight=0.999, state_loss_weight=0.002, return_loss_weight=0.0001` gives best evaluator results (mean_reward > 0 on mini evaluator).
- Near-zero state/return weights mean the model focuses almost entirely on action prediction accuracy.
- See `results.tsv` `20260702-lw-sweep` entries for the full sweep.
- The MoLab-trained model uses these weights — see HF model card.

### 4. Context sweep

- With 22GB VRAM, test ctx=288, 576, 1008, 2016
- Longer context may help capture weekly FCAS market patterns

### 5. Hybrid approaches

- If DT still lags PPO, explore DT fine-tuned online with PPO
- Prompt bank for different market regimes

## What's Done

| Item | Status |
|------|--------|
| 2,425-episode FCAS dataset | ✅ |
| 5-region data fetch | ✅ |
| FCAS rule algorithm | ✅ |
| Stratified pilot dataset | ✅ |
| FCAS-aware evaluator configs | ✅ |
| Documentation refresh | ✅ |

## Success Criteria

- Mean reward > 0.0 (positive profit)
- FCAS revenue > $2,000/ep (indicates learned FCAS bidding)
- Beat FCAS rule baseline in paired comparison
