# Next Stage: Evaluate & Fine-Tune the New Modern DT

This document contains everything an AI agent needs to:
1. Load the new modern Decision Transformer from HuggingFace
2. Evaluate it against baselines
3. Run GRPO Phase 1 fine-tuning
4. Re-evaluate the fine-tuned model

## New Model

- **HF repo**: `mrvictoru/energydecision-dt-v2`
- **Checkpoint**: `aemo_dt_model.pt` (modern architecture with GQA, RMSNorm, weight tying)
- **Architecture**: Modern `DecisionTransformer` (NOT `LegacyDecisionTransformer`)

## Model kwargs for the modern architecture

```python
model_kwargs = {
    "state_dim": 18,
    "act_dim": 9,
    "n_block": 8,
    "h_dim": 384,
    "context_len": 180,
    "n_heads": 8,
    "drop_p": 0.15,
    "max_timestep": 100000,
    "rope_enabled": True,
    "rope_max_position": 540,
    "rope_base": 10000.0,
}
```

Note: `rope_enabled=True` — this is different from the legacy model.

## Step 1: Create a surface manifest for the evaluator

The evaluator needs a JSON manifest pointing to the model. Create one:

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

REPO_ROOT = Path("/path/to/energydecision")
checkpoint = hf_hub_download("mrvictoru/energydecision-dt-v2", "aemo_dt_model.pt")
out_dir = REPO_ROOT / "models" / "aemo" / "dt" / "hf_v2_modern"
out_dir.mkdir(parents=True, exist_ok=True)

manifest = {
    "schema": "energydecision.dt_training_surface.v1",
    "surface_preset": "hf_modern_v2",
    "model_variant": "full_fcas",
    "action_mode": "full_fcas",
    "model_kwargs": {
        "state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384,
        "context_len": 180, "n_heads": 8, "drop_p": 0.15, "max_timestep": 100000,
        "rope_enabled": True, "rope_max_position": 540, "rope_base": 10000.0,
    },
    "paths": {
        "save_path": checkpoint,
        "loss_csv_path": str(out_dir / "dummy_loss.csv"),
    },
}
(out_dir / "hf_modern_surface_manifest.json").write_text(json.dumps(manifest, indent=2))
(out_dir / "dummy_loss.csv").write_text("epoch,train_total,train_action,val_total,val_action\n1,0.0,0.0,,\n")
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
for RTG in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
  python3 -c "
import json
with open('configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json') as f:
    cfg = json.load(f)
for p in cfg['policies']:
    if p['kind'] == 'dt':
        p['rtg_value'] = $RTG
with open('/tmp/eval_cfg_$RTG.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
  python3 src/autoresearch_evaluator.py \
    --surface-manifest-path models/aemo/dt/hf_v2_modern/hf_modern_surface_manifest.json \
    --evaluation-config /tmp/eval_cfg_$RTG.json \
    --output-dir eval_output/hf_modern_rtg/$RTG --device auto
done

# Find best RTG
for RTG in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
  python3 -c "
import json
with open('eval_output/hf_modern_rtg/$RTG/evaluation_summary.json') as f:
    d = json.load(f)
for m in d['heldout_evaluation']['aggregate_metrics']:
    if m['experiment'] == 'candidate_dt':
        print(f'RTG=$RTG: profit=\${m[\"avg_profit_per_episode\"]:,.0f}')
"
done
```

## Step 3: Run GRPO Phase 1 Fine-Tuning

The modern model uses a DIFFERENT forward pass signature than the legacy model.
The `_is_legacy_dt()` check in `GRPOTrainer._forward_dt()` will return `False`,
so the code will use the modern path: `model(states, rtgs, timesteps, actions)`.

Also note: the modern model may have `rope_enabled=True`. The GRPO code doesn't
need any special handling for RoPE — it's handled inside the model's forward.

### Multi-region training with Phase 1 features:

```bash
python3 src/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --start-date 2024-01-01 --end-date 2024-09-30 \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --kl-coeff 0.02 \
  --battery-capacity 10 --max-power 10 \
  --rtg-count 4 --rtg-spread 2.0 --dt-gamma 0.95 \
  --group-size 8 \
  --sync-reference-every 5 \
  --adaptive-rtg --adaptive-rtg-ewma-alpha 0.1 \
  --deg-penalty-weight 1.5 \
  --output-dir models/aemo/dt/grpo_modern
```

**Important**: The script downloads from `mrvictoru/energydecision-dt` by default.
The new model is on `mrvictoru/energydecision-dt-v2`. The fallback is to place
the model file at `models/aemo/dt/grpo_phase1/dt_model_grpo_multi.pt`.

To fix: either modify the script's fallback path, or create a symlink:

```bash
ln -sf /path/to/hf/cache/aemo_dt_model.pt \
  models/aemo/dt/grpo_phase1/dt_model_grpo_multi.pt
```

Or modify `run_grpo_multi_region.py` to change the default `--hf-repo` and
`--hf-filename` parameters.

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
