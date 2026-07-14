# Next Stage: Evaluate & Fine-Tune the New Modern DT

This document contains everything an AI agent needs to:
1. Load the new modern Decision Transformer from HuggingFace
2. Evaluate it against baselines
3. Run GRPO Phase 1 fine-tuning
4. Re-evaluate the fine-tuned model

## New Model

- **HF repo**: `mrvictoru/energydecision-dt-v2`
- **Checkpoint**: `aemo_dt_fcas_model.pt` (modern architecture with GQA, RMSNorm, QK-norm, and tied heads)
- **Architecture**: Modern `DecisionTransformer` (NOT `LegacyDecisionTransformer`)

## Model kwargs for the modern architecture

```python
model_kwargs = {
    "state_dim": 18,
    "act_dim": 9,
    "n_block": 8,
    "h_dim": 768,
    "context_len": 210,
    "n_heads": 12,
    "n_kv_heads": 6,
    "drop_p": 0.15,
    "max_timestep": 100000,
    "qk_norm": True,
    "tie_weights": True,
    "rope_enabled": False,
    "rope_max_position": 4096,
    "rope_base": 10000.0,
}
```

This matches the current HF v2 artifact layout: 8 blocks, width 768, 12 query heads, 6 KV heads, QK-norm enabled, tied prediction heads, and RoPE disabled.

## Step 1: Create a surface manifest for the evaluator

The evaluator needs a JSON manifest pointing to the model. Create one with the helper:

```bash
python3 scripts/create_hf_surface_manifest.py
```

## Step 2: Evaluate the baseline model

Use the `q4_dispatch_matched.json` config for fair comparison:

```bash
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/hf_v2_modern/hf_modern_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json \
  --output-dir eval_output/hf_modern_baseline \
  --device auto
```

Also run an RTG calibration sweep (the new model may have a different optimal RTG):

```bash
python3 scripts/run_aemo_dt_rtg_sweep.py \
  --surface-manifest-path models/aemo/dt/hf_v2_modern/hf_modern_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json \
  --output-dir eval_output/hf_modern_rtg \
  --device auto
```

## Step 3: Run GRPO Phase 1 Fine-Tuning

The modern model uses a DIFFERENT forward pass signature than the legacy model.
The `_is_legacy_dt()` check in `GRPOTrainer._forward_dt()` will return `False`,
so the code will use the modern path: `model(states, rtgs, timesteps, actions)`.

The modern model is no longer compatible with the old hard-coded GRPO width/head defaults.
Use the updated runners, which now default to the v2 repo and the matching modern-v2 model config.

### Multi-region + multi-battery training with Phase 1 features:

The GRPO training should expose the model to both diverse regions AND battery
configs (matching the v2 pretraining distribution). The `--battery-configs`
flag randomly samples from the 4 battery types each episode:

```bash
python3 src/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --battery-configs medium_1c,large_07c,small_05c,fast_375c \
  --start-date 2024-01-01 --end-date 2024-09-30 \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --kl-coeff 0.02 \
  --rtg-count 4 --rtg-spread 2.0 --dt-gamma 0.95 \
  --group-size 8 \
  --sync-reference-every 5 \
  --adaptive-rtg --adaptive-rtg-ewma-alpha 0.1 \
  --deg-penalty-weight 1.5 \
  --output-dir models/aemo/dt/grpo_modern
```

The updated GRPO runners already default to:
- `--hf-repo mrvictoru/energydecision-dt-v2`
- `--hf-filename aemo_dt_fcas_model.pt`
- `--model-config configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json`

### Alternative: Single-region for faster iteration:

```bash
python3 src/run_grpo_posttraining.py \
  --region NSW1 --start-date 2024-01-01 --end-date 2024-01-14 \
  --iterations 5 --episode-hours 144 --step-duration 0.083333 \
  --output-dir models/aemo/dt/grpo_modern_single
```

## Step 4: Evaluate the Fine-Tuned Model

After GRPO training, evaluate with the optimal RTG found in Step 2:

```bash
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_modern/grpo_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json \
  --output-dir eval_output/grpo_modern_final \
  --device auto
```

## Expected Results for Comparison

The Phase 1 GRPO on the legacy model achieved:

| Metric | Value |
|--------|-------|
| Profit/ep | $8,242 |
| FCAS/ep | $7,686 |
| Deg/ep | $760 |
| Profit/MWh | $1,030 |
| Sharpe | -14.08 |

The modern model should match or exceed these due to GQA and improved architecture.

## Troubleshooting

### Issue: `_is_legacy_dt()` detection
The GRPO code has `_is_legacy_dt()` which checks for `embed_return` attribute.
The modern model uses `embed_rtg` instead, so `_is_legacy_dt()` returns `False`.
The `_forward_dt()` method handles both paths automatically.

### Issue: Loading the modern checkpoint
`load_from_checkpoint()` in `DecisionTransformer` will NOT trigger legacy
detection since the modern model has `transformer.*` keys, not `blocks.*`.
It should load normally.

### Issue: HF download fails
The scripts may try to download from the old repo. Use the symlink workaround
above if needed.
