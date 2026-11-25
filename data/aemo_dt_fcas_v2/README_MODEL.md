---
tags:
- energy-trading
- battery-storage
- FCAS
- AEMO
- NEM
- decision-transformer
- grpo
- reinforcement-learning
base_model: mrvictoru/energydecision-dt
pretty_name: AEMO Battery Trading Decision Transformer — GRPO Fine-Tuned
---

> **⚠️ ARCHIVED / HISTORICAL — Legacy Phase 1 GRPO.** All figures in this card
> are historical. They were measured on a single narrow dispatch-matched
> surface (SA1 Q4 2024, Dalrymple North) and were later shown to be overfit to
> that surface — the "beats the pretrained baseline and PPO" claim below did
> not generalise. This GRPO-tuned model is superseded by the shipped
> physics-v2 `v2cal` planner-distilled Decision Transformer (no GRPO). For
> current claims and released artifacts see
> [`docs/reproduction.md`](../../docs/reproduction.md) and
> [`docs/huggingface/energydecision-dt-v2-sdp_README.md`](../../docs/huggingface/energydecision-dt-v2-sdp_README.md).
>
> **Corrected publication state** (release tag
> `preprint-v2-baseline-corrected`): current DT-vs-PPO claims use three tuned
> 4M-step PPO seeds, with within-cell DT/PPO seed-mean ratios
> **1.70× / 1.20× / 1.69× / 1.60×** (dispatch's paired CI includes zero; one
> PPO seed reverses three surface means); under market impact the seed-mean
> ratio is **1.60× with 8/9 wins**, but seed-specific ratios span
> **0.91×–2.96×**. The older 200k / 204,800-step default-PPO ratios are
> historical. Do not cite the numbers below as current headline results.

# AEMO Decision Transformer — GRPO Post-Trained

This is a **GRPO online-RL fine-tuned** version of the
[v2 pretrained DT](https://huggingface.co/mrvictoru/energydecision-dt) for
battery energy storage trading in Australia's NEM market.

## Performance Summary

Evaluated on Q4 2024 held-out SA1 data with dispatch-matched battery
(Dalrymple North: 8 MWh / 30 MW, 3.75 C, `full_fcas` action mode):

| Metric | Value |
|--------|-------|
| **Profit/episode** | **+$8,242** |
| **Profit/MWh** (normalised) | **+$1,030** |
| **FCAS revenue/ep** | $7,686 |
| **Degradation cost/ep** | $760 |
| **vs v2 pretrained baseline** | +285% profit |
| **vs PPO** (SB3) | +6% profit |

On that single dispatch-matched surface it appeared (historically) to beat
both the v2 pretrained baseline and a trained PPO reference — a result later
shown to be overfit to that narrow surface and not to generalise (see the
archived banner above).

## GRPO Configuration (Phase 1)

| Parameter | Value |
|-----------|-------|
| Iterations | 5 |
| Learning rate | 1e‑5 |
| KL coefficient | 0.02 |
| RTG count | 4 |
| RTG spread | 2.0 |
| RTG distribution | Gaussian |
| Group size | 8 |
| Degradation penalty weight | 1.5 |
| Reference sync every | 5 iterations |
| Adaptive RTG | Yes (EWMA α=0.1) |
| Training regions | NSW1, SA1, QLD1, VIC1, TAS1 |
| Training date range | 2024-01-01 → 2024-09-30 |
| Training step duration | 5 minutes |
| Optimal eval RTG | 0.5 |

## Architecture

| Parameter | Value |
|-----------|-------|
| state_dim | 18 |
| act_dim | 9 |
| n_block | 8 |
| h_dim | 384 |
| n_heads | 8 |
| context_len | 180 |
| drop_p | 0.15 |
| Action mode | `full_fcas` (energy + 8 FCAS services) |

## Training Data

The base model was pretrained on the v2 FCAS-rich dataset
(2,401 episodes, 77 M rows, 4 realistic battery configurations).
GRPO post-training used online rollouts from the AEMO simulation
environment on 2024 data (Jan–Sep), sampled across all 5 NEM regions.

## Recommended RTG

For evaluation, use `rtg_value=0.5` (not 0.0). The RTG prompt
significantly affects performance. See the calibration sweep:

| RTG | Profit/ep | FCAS/ep | Deg/ep |
|:---:|:---------:|:-------:|:------:|
| 0.0 | $5,451 | $7,962 | $2,769 |
| **0.5** | **$8,242** | $7,637 | $1,323 |
| 1.0 | $7,901 | $7,781 | $1,207 |
| 2.0 | $6,477 | $7,219 | $963 |

## Standalone Inference

```python
from decision_transformer import DecisionTransformer
from huggingface_hub import hf_hub_download

model_kwargs = {
    "state_dim": 18, "act_dim": 9, "n_block": 8,
    "h_dim": 384, "context_len": 180, "n_heads": 8,
    "drop_p": 0.15, "max_timestep": 100000,
}
checkpoint = hf_hub_download("mrvictoru/energydecision-dt", "aemo_dt_grpo_model.pt")
model = DecisionTransformer(**model_kwargs)
model.load_from_checkpoint(checkpoint)
model.eval()
```
