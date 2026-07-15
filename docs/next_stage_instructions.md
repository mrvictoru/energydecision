# Next Stage: GRPO Fine-Tuning Instructions (Updated)

This document consolidates findings from the evaluation report and provides
step-by-step instructions for running GRPO fine-tuning on BOTH the legacy and
modern models with multi-region + multi-battery training.

## Key Findings from Eval Report

### 1. Two different eval surfaces give different answers

| Eval Config | Battery | Regions | Best for |
|-------------|---------|---------|----------|
| `q4_dispatch_matched` | 8 MWh / 30 MW (dispatch-matched) | SA1 only | Head-to-head vs dispatch replay |
| `q4_multi_station` | 10 MWh / 10 MW (fixed 1C) | 5 regions | Cross-region generalisation |

**The same model can win on one and lose on the other.** The report's Phase 1
GRPO model gets $8,242/ep on dispatch-matched but only $1,855/ep on
multi-station. The pretrained v2 gets $7,392/ep on dispatch-matched but
$3,098/ep on multi-station. There is no single "best" model — it depends on
the eval surface.

### 2. RTG calibration matters more than GRPO

| Model | RTG 0.0 | RTG 0.5 | Gain |
|-------|:-------:|:-------:|:----:|
| Phase 1 GRPO on dispatch-matched | $5,451 | $8,242 | **+51%** |

Half the headline gain comes from prompt tuning, not GRPO itself.

### 3. Modern v2 baseline is stronger than expected

On `q4_dispatch_matched`, the modern v2 model scores ~$7,392/ep without any
GRPO. This is close to the legacy Phase 1 GRPO's $8,242/ep. The modern
architecture (GQA, RoPE) may already capture much of the benefit that GRPO
provides for the legacy model.

### 4. GRPO needs group-size >= 4

The earlier GRPO run used `group-size=2` which is too small for stable
advantage estimation. Use at least `group-size=4`.

---

## Step 1: Evaluate the Modern Baseline (v2 from HF)

Create a surface manifest and evaluate on both dispatch_matched and
multi_station to establish the baseline:

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

checkpoint = hf_hub_download("mrvictoru/energydecision-dt-v2", "aemo_dt_model.pt")

model_kwargs = {
    "state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384,
    "context_len": 180, "n_heads": 8, "drop_p": 0.15,
    "max_timestep": 100000,
    "rope_enabled": True,
    "rope_max_position": 540,
    "rope_base": 10000.0,
}

manifest = {
    "schema": "energydecision.dt_training_surface.v1",
    "model_kwargs": model_kwargs,
    "paths": {
        "save_path": checkpoint,
        "loss_csv_path": "/tmp/dummy_loss.csv",
    },
}
Path("/tmp/modern_manifest.json").write_text(json.dumps(manifest, indent=2))
Path("/tmp/dummy_loss.csv").write_text("epoch,train_total,train_action,val_total,val_action\n1,0.0,0.0,,\n")
```

Evaluate on both surfaces:

```bash
# 1. RTG calibration (use smoke tier to find optimal RTG)
for RTG in 0.0 0.5 1.0 1.5 2.0; do
  python3 -c "
import json
cfg = json.load(open('configs/eval_tier_smoke.json'))
for p in cfg['policies']:
    if p['kind'] == 'dt':
        p['rtg_value'] = $RTG
json.dump(cfg, open(f'/tmp/eval_smoke_rtg$RTG.json', 'w'), indent=2)
"
  python3 src/autoresearch_evaluator.py \
    --surface-manifest-path /tmp/modern_manifest.json \
    --evaluation-config /tmp/eval_smoke_rtg$RTG.json \
    --output-dir eval_output/modern_rtg/$RTG --device auto

  python3 -c "
import json
d = json.load(open(f'eval_output/modern_rtg/$RTG/evaluation_summary.json'))
for m in d['heldout_evaluation']['aggregate_metrics']:
    if m['experiment'] == 'candidate_dt':
        print(f'RTG=$RTG: profit=\${m[\"avg_profit_per_episode\"]:,.0f}')
"
done

# 2. Full eval on both surfaces using best RTG
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/modern_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/modern_standard --device auto

python3 src/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/modern_manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/modern_comprehensive --device auto
```

---

## Step 2: GRPO Fine-Tuning (Both Models)

### Legacy Model (mrvictoru/energydecision-dt)

The legacy model uses `run_grpo_multi_region.py` which downloads from
`mrvictoru/energydecision-dt` by default:

```bash
python3 src/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --battery-configs medium_1c,large_07c,small_05c,fast_375c \
  --start-date 2024-01-01 --end-date 2024-09-30 \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --kl-coeff 0.02 \
  --rtg-count 4 --rtg-spread 2.0 --dt-gamma 0.95 \
  --group-size 4 \
  --sync-reference-every 5 \
  --adaptive-rtg --adaptive-rtg-ewma-alpha 0.1 \
  --deg-penalty-weight 1.5 \
  --output-dir models/aemo/dt/grpo_legacy_multibat
```

### Modern Model (mrvictoru/energydecision-dt-v2)

**Important**: The script downloads from `mrvictoru/energydecision-dt` by
default. To use the modern v2 model, create a symlink:

```bash
# Find the cached checkpoint path
python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('mrvictoru/energydecision-dt-v2', 'aemo_dt_model.pt'))
"

# Symlink it to where the script expects it
ln -sf /path/to/cached/aemo_dt_model.pt \
  models/aemo/dt/grpo_phase1/dt_model_grpo_multi.pt
```

Then run the same GRPO command. The `_is_legacy_dt()` check in the GRPO code
will return `False` for the modern model, so it uses the correct forward path.

Alternatively, modify the script's HF repo defaults in the source code.

---

## Step 3: Evaluate Both GRPO-Tuned Models

After GRPO, evaluate on both surfaces:

```bash
# Legacy GRPO model
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_legacy_multibat/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/legacy_grpo_standard --device auto

python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_legacy_multibat/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/legacy_grpo_comprehensive --device auto

# Modern GRPO model
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_modern_multibat/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/modern_grpo_standard --device auto

python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_modern_multibat/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/modern_grpo_comprehensive --device auto
```

---

## Expected Results (Reference)

From the eval report (legacy models on dispatch-matched):

| Model | Eval config | RTG | Profit/ep | Notes |
|-------|-------------|:---:|:---------:|-------|
| v2 HF DT (pretrained) | q4_dispatch_matched | 0.5 | ~$7,392 | Modern baseline |
| Phase 1 GRPO (legacy) | q4_dispatch_matched | 0.0 | $5,451 | Without RTG tuning |
| Phase 1 GRPO (legacy) | q4_dispatch_matched | 0.5 | $8,242 | With RTG calibration |
| Phase 1 GRPO (legacy) | q4_multi_station | 0.5 | $1,855 | Different eval surface |
| v2 HF DT (pretrained) | q4_multi_station | default | $3,098 | Better on multi-station |

The multi-battery GRPO may improve both surfaces simultaneously by reinforcing
the full pretraining distribution.

---

## GPU Memory Notes

| Group size | VRAM required | Feasible on 22 GB? |
|:----------:|:-------------:|:------------------:|
| 8 | ~14 GB | ✅ |
| 4 | ~8 GB | ✅ (recommended) |
| 2 | ~5 GB | ✅ (but too noisy) |

Use `--group-size 4` for stable training on a 22 GB RTX 2080 Ti. If you
still encounter OOM, check for zombie GPU processes first:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
kill -9 <orphaned_pid>
```
