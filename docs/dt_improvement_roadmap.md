# DT AEMO Performance: Diagnosis and Improvement Roadmap

## Current State

On the expanded 135-episode AEMO evaluation (5 regions, 6 periods, 12-day episodes):

| Policy | Mean Reward | Profit/Ep | FCAS Rev | Deg Cost | Sharpe |
|--------|:-----------:|:---------:|:--------:|:--------:|:-----:|
| **PPO (online RL)** | **+12.82** | **+$12,839** | **$10,628** | $1,458 | +1.26 |
| DT full-pretrained (8×512, ctx=180) | -3.11 | -$1,396 | $77 | $2,503 | -0.40 |
| Rule heuristic | -4.82 | -$3,562 | $0 | $15,400 | -0.48 |
| DT old pretrain (4×128, ctx=1152) | -13.55 | -$10,620 | $2,328 | $12,975 | -2.20 |

---

## 1. Root Cause Analysis: Why PPO Outperforms DT

### 1.1 The FCAS Gap (138× difference)

PPO earns **$10,628/ep** in FCAS ancillary market revenue. DT earns **$77/ep**. This single factor accounts for nearly the entire performance gap.
PPO learned FCAS bidding strategies through millions of online interactions with the AEMO environment. DT was trained on ~24 offline episodes — none of which came from policies that aggressively participated in FCAS markets.

**The DT training data sources (from `source_policy` field):**
- `nsw1_2021_2023__rule__small` — rule-based, $0 FCAS
- `vic1_2021_2023__td3__medium` — TD3, unknown FCAS behavior
- `sa1_2022_2023__rule__small` — rule-based, $0 FCAS
- `qld1_2021_2023__a2c__small` — A2C, unknown FCAS behavior
- `tas1_2021_2023__ddpg__small` — DDPG, unknown FCAS behavior
- `sa1_2022_2023__dispatch__hornsdale_replay` — historical dispatch, possibly FCAS-active

None of these source policies were explicitly optimized for FCAS revenue. The DT can only learn what's in its training data, and there are essentially no examples of profitable FCAS bidding in the dataset.

### 1.2 Online vs Offline Learning

PPO interacts with the environment online, receiving immediate reward feedback: if an FCAS bid earns revenue, the gradient pushes the policy toward bidding more. DT learns from static historical trajectories — it can't "discover" profitable strategies that aren't already demonstrated in the data.

### 1.3 Training Scale

- **DT full-pretrained**: 24 training episodes × ~2000 steps each = ~48,000 transitions
- **PPO**: Millions of environment steps during training

The DT is dramatically data-starved. Even with perfect architecture, 24 episodes is insufficient to learn complex multi-market bidding strategies.

### 1.4 Short Context Window

The current ctx=180 (15 hours) was optimal in the pilot autorresearch sweep. But the pilot data only came from rule-based policies that rarely exploited longer-term patterns. With PPO-generated data that captures 12-day market cycles, longer context might be beneficial.

### 1.5 Return Conditioning Mismatch

The DT was evaluated with `rtg_value=0.0` — essentially saying "break even." But PPO achieves +12.82 mean reward. The DT may be capable of better performance with a higher RTG prompt (e.g., +10.0 or +20.0), but was never tested with aggressive RTG values on AEMO.

---

## 2. Prioritized Improvement Ideas

### Tier 1: Immediate (low effort, high expected impact)

#### 2.1 Train DT on PPO-Generated Trajectories
**Effort**: Medium | **Expected Impact**: High

The single most impactful change. Generate 100–500 AEMO episodes using PPO and use them as DT training data. This directly addresses the FCAS gap by providing the DT with examples of profitable multi-market strategies.

- Generate PPO rollouts across all 5 regions and multiple seasons (2021–2023 training data, 2024 held-out)
- Include `source_policy = "ppo"` in the dataset
- Key question: does the DT learn to replicate PPO's FCAS bidding, or does it learn something even better?

#### 2.2 RTG Prompt Calibration
**Effort**: Low | **Expected Impact**: Medium

The evaluator already computes `recommended_rtg` from the evaluation logs. Run a sweep of RTG values (e.g., 0, 5, 10, 15, 20) on the full-pretrained DT. A higher RTG might elicit stronger FCAS participation from the existing model.

- The household DT showed RTG sensitivity: different prompts changed degradation 22×
- Run `mini_evaluator` with different RTG values on the expanded scenarios
- If the DT responds to RTG, this is a zero-effort improvement

#### 2.3 Scale Up Training Data (Mix Policies)
**Effort**: Medium | **Expected Impact**: High

Expand the training dataset beyond the current 6–24 episodes:
- Run rule, PPO, and historical dispatch replay across more scenarios
- Target: 100+ training episodes with diverse source policies
- Include explicit FCAS classification in metadata (FCAS-active vs FCAS-inactive)
- Use the full 2021–2023 corpus (162 episodes in `aemo_dt_dataset.parquet`)

### Tier 2: Short-term (medium effort, medium-to-high impact)

#### 2.4 FCAS-Aware Architecture / Loss
**Effort**: Medium | **Expected Impact**: Medium

Modify the DT to explicitly model FCAS as a separate prediction target:
- Split action head into: `energy_dispatch_head`, `fcas_raise_head`, `fcas_lower_head`
- Add FCAS revenue prediction as an auxiliary loss
- Weight the FCAS action loss higher than energy action loss
- Could use the existing 3D action space with per-dimension weights

#### 2.5 Longer Context Window
**Effort**: Low | **Expected Impact**: Low-Medium

The autorresearch context sweep found ctx=180 optimal for the pilot model. But with FCAS-rich training data, longer contexts (288–576) might be necessary to capture weekly market patterns. With PPO data in 288h (12-day) episodes:
- Try ctx=288 and ctx=576 with the same 8×512 architecture
- Monitor VRAM: ctx=576 at 8×512 should fit within 8GB at inference time
- Only re-run the context sweep if training data changes (PPO-generated)

#### 2.6 Ensemble / Prompt Bank
**Effort**: Low | **Expected Impact**: Low-Medium

The DT's RTG-conditioning allows zero-shot trade-off control without retraining:
- Evaluate the same model with multiple RTG prompts
- Use different prompts for different market conditions (high/low volatility, peak/off-peak)
- Ensemble the outputs (weighted by market regime)

### Tier 3: Medium-term (higher effort, research-oriented)

#### 2.7 Online Fine-Tuning (RLPD-style)
**Effort**: High | **Expected Impact**: Medium

Use the pretrained DT as a policy prior and fine-tune online with RL:
- Initialize the RL policy from DT weights
- Use DT as a "behavior prior" with KL regularization
- Fine-tune in the AEMO environment for a small number of online episodes
- This combines offline pre-training (data efficiency) with online adaptation (reward optimization)
- Reference: RLPD (Ball et al., 2023), Decision Transformer as a Prior (Zheng et al., 2022)

#### 2.8 Larger Model Architecture
**Effort**: Medium | **Expected Impact**: Low-Medium

Scale up the DT architecture beyond the current frontier (8×512):
- Try 12×512 or 12×768 with ctx=288 on the expanded PPO dataset
- Memory budget: 8GB VRAM. Training with batch_size=1 might be feasible
- Risk: the RTX 3060 Ti constraint may make inference slow for large models
- If a larger model helps at ctx=180, inference stays fast

#### 2.9 Transformer Architecture Variants
**Effort**: Medium-High | **Expected Impact**: Unknown

Experiment with different transformer backbones:
- **Flash Attention**: Lower memory, could enable longer contexts or larger models
- **Mamba/SSM**: State-space models are efficient for long sequences; AEMO data has strong temporal structure
- **RoPE improvements**: Test different base frequencies (currently 10000) and position interpolation
- **SwiGLU → GEGLU / ReGLU**: Activation function ablation
- **Pre-norm vs Post-norm**: Already using pre-norm (good). Test LayerNorm placement

#### 2.10 Data Augmentation
**Effort**: Low | **Expected Impact**: Medium

Augment existing trajectories to increase FCAS exposure:
- Add synthetic FCAS-bid examples to rule/dispatch trajectories
- Use Gaussian noise on FCAS bids around PPO-generated actions
- Mirror strategies: flip charge/discharge while preserving FCAS bid patterns
- Time-shift episodes to expose the model to different market starting points

### Tier 4: Long-term (research-scale)

#### 2.11 Multi-Objective / Preference-Conditioned DT
**Effort**: High | **Expected Impact**: Medium

Instead of a single RTG (return-to-go), condition on multiple objectives:
- `(rtg_profit, rtg_deg, rtg_fcas)` — profit, degradation, FCAS
- This gives explicit control over the profit-vs-degradation tradeoff
- The household DT already showed RTG-conditioning works for degradation control
- Preference-conditioned DT (P-DT): condition on a preference vector, not a scalar

#### 2.12 Hierarchical / Two-Stage DT
**Effort**: High | **Expected Impact**: Medium

Separate decision-making into two stages:
1. **High-level model** (e.g., 1 decision/hour): Plan energy allocation and FCAS posture
2. **Low-level model** (5-min): Execute with more granular dispatch

This mirrors how real BESS operators work (day-ahead planning + real-time dispatch).

#### 2.13 Cross-Attention between State and Return Streams
**Effort**: Medium | **Expected Impact**: Unknown

The current DT processes (rtg, state, action) as a flat interleaved sequence. Alternative:
- Process state history and return-to-go in separate transformer streams
- Cross-attend between them at each layer
- This allows the model to condition on return more explicitly

#### 2.14 Train with PPO as a Data Oracle + Online RL Fine-Tuning
**Effort**: Medium | **Expected Impact**: High

A 2-phase approach:
1. **Offline phase**: Train DT on extensive PPO-generated data (100+ episodes)
2. **Online phase**: Fine-tune the DT in the AEMO environment using a small number of online RL updates
3. The offline data provides broad coverage; online fine-tuning sharpens performance on specific market conditions

---

## 3. Lessons from the Autoresearch Program

### What worked well
1. **Frontier hyperparameters** (8×512, ctx=180, drop_p=0.15): Transformed DT from -13.55 → -3.11 mean_reward
2. **Context sweep methodology**: Systematically found ctx=180 optimal for pilot data
3. **Parallel evaluator** (8 workers, `parallelize_candidate_dt=true`): Completed 135-episode eval in ~10 minutes
4. **Full-corpus training** (24+16 episodes): Significantly better than 6-episode pilot
5. **Explicit validation split**: Prevents data leakage across episodes

### What didn't work or was suboptimal
1. **Pilot data was too small and biased**: 6 episodes, all from different policies/regions, none FCAS-optimized
2. **Training on mixed source policies diluted the signal**: Including rule-based and dispatch replay data gave the DT examples of "bad" strategies
3. **No RTG calibration on AEMO**: Always evaluated at rtg=0.0, never explored the prompt space
4. **Context sweep used pilot data, not PPO data**: The optimal ctx=180 may not generalize to FCAS-rich trajectories
5. **No FCAS-specific metrics in training**: The loss weights treat all action dimensions equally
6. **Training time was compressed**: 2-3 epochs on 3.6M windows might not be enough for complex multi-market learning

### Key design constraints
- **8GB VRAM (RTX 3060 Ti)**: Limits feasible model size and context length
- **Single GPU**: No distributed training
- **NEMOSIS data fetching**: AEMO data downloads are slow (Jul 2024 was missing, ~30min to fetch)
- **Python GIL**: ThreadPoolExecutor is limited for CPU-bound env stepping
- **No online training loop**: DT inference and training are separate from the RL training harness

---

## 4. Next-Session Action Plan (prioritized)

1. **Generate PPO trajectory data**: Run PPO for 100+ episodes across all regions/seasons (2021-2023 data for training, save 2024 for held-out eval).
2. **Train DT on PPO-only data** (no rule/dispatch mixing initially): Evaluate whether the DT can learn FCAS bidding from PPO demonstrations.
3. **RTG calibration sweep**: Test rtg_value = [-5, 0, 5, 10, 15, 20] on the full-pretrained DT to see if a higher prompt improves performance.
4. **Train with scaled-up data**: Use the full 162-episode corpus + PPO-generated data, train for more epochs.
5. **Implement and test FCAS-weighted loss**: Give the FCAS action dimensions higher loss weight.
6. **Re-run context sweep with PPO data**: Determine if optimal context length changes when training data includes FCAS strategies.

### Success Metrics for Next Session
| Experiment | Target | Current baseline |
|-----------|--------|------------------|
| DT trained on PPO-only data | mean_reward > +5.0 | -3.11 (mixed data) |
| RTG-calibrated DT | mean_reward > -1.0 | -3.11 (rtg=0.0) |
| FCAS-weighted loss | FCAS rev > $2,000/ep | $77/ep |
| Scaled-up training data | mean_reward > 0.0 | -3.11 |
| Hybrid PPO+DT ensemble | mean_reward > +12.82 | PPO alone is +12.82 |

---

## 5. Open Questions for Investigation

1. **Can the DT learn FCAS bidding from PPO demonstrations, or does FCAS require online interaction?**
   - If DT can replicate PPO's FCAS strategy, online RL may not be strictly necessary.
   - If DT can't, we need online fine-tuning or hybrid approaches.

2. **Is the DT architecture sufficient for multi-market strategies, or is it fundamentally limited by offline learning?**
   - The transformer can model complex sequences, but it can't "explore" — it can only replicate what's in the data.
   - This is an inherent limitation of offline RL that architecture changes alone won't fix.

3. **How much training data is "enough"?**
   - 6 episodes → -13.55 (pilot)
   - 24 episodes → -3.11 (full pretrained)
   - 100 episodes → ?
   - 1000 episodes → ?
   - Where does the learning curve plateau?

4. **Does RTG-conditioning work as a "strategy selector" for AEMO as it does for household?**
   - Household: RTG changed degradation 22× while keeping similar returns
   - AEMO: Unknown — never tested RTG sweep

5. **Is the gap mainly data quantity, data quality (FCAS examples), or architecture?**
   - Need ablation: same data → better architecture; better data → same architecture; both → ideal

---

## 6. Required Implementation Work (tracking items)

- [ ] Script to generate PPO rollout data in DT-compatible parquet format
- [ ] Config for training DT on PPO-only dataset (hardcoded in `src/pretrain_decision_transformer.py`)
- [ ] RTG sweep script for the mini evaluator (already exists, needs sweep loop)
- [ ] FCAS-weighted loss implementation in `src/transformer_training.py`
- [ ] Training manifest for scaled-up dataset (162 episodes + PPO data)
- [ ] Context sweep script adapted for PPO-generated data
- [ ] Evaluation sweep comparing DT variants (data, architecture, RTG)
- [ ] Entry in `results.tsv` for each experiment

---

*Last updated: 2026-06-11. This file is intended as a research brainstorming document and is not yet committed to execution.*
