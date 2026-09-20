# GRPO Fine-Tuning Guide

This guide documents the GRPO post-training workflow for advanced AEMO experiments.

Use it when you need:

- the GRPO CLI entrypoints
- the currently recommended GRPO flags
- post-training evaluation commands
- RTG prompt calibration for GRPO-tuned models

If you are new to the repo, start with [README.md](README.md), [architecture.md](architecture.md), [development.md](development.md), and [aemo/README.md](aemo/README.md) first.

Group Relative Policy Optimization (GRPO) improves a pretrained Decision
Transformer through online interaction with the simulation environment.
No dataset regeneration or SB3 retraining is needed — the model learns by
trying actions and observing rewards.

## Quick Start: Single-Region

```bash
python3 scripts/run_grpo_posttraining.py \
  --region NSW1 --start-date 2024-01-01 --end-date 2024-01-14 \
  --iterations 5 --episode-hours 144 --step-duration 0.083333
```

This downloads the HF model from `mrvictoru/energydecision-dt` automatically
and runs 5 GRPO iterations on NSW1 January 2024 data at 5-minute resolution.

## Quick Start: Multi-Region (Recommended)

```bash
python3 scripts/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --start-date 2024-01-01 --end-date 2024-09-30 \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --kl-coeff 0.02 \
  --battery-capacity 10 --max-power 10 \
  --rtg-count 4 --dt-gamma 1.0
```

## Phase 1 GRPO Features (Recommended Config)

The best results come from enabling all Phase 1 improvements together.
## ⚠️ Pre-flight: Clean GPU Memory

Before any GRPO training, kill orphaned GPU processes that cause false OOM:

```bash
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # should show < 500 MiB
```

Use `--group-size 4` instead of 8 on a 22 GB GPU to avoid OOM:

```bash
python3 scripts/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --battery-configs medium_1c,large_07c,small_05c,fast_375c \
  --start-date 2024-01-01 --end-date 2024-09-30 \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --kl-coeff 0.02 \
  --rtg-count 4 --rtg-spread 2.0 --dt-gamma 1.0 \
  --group-size 4 \
  --sync-reference-every 5 \
  --deg-penalty-weight 1.5 \
  --output-dir models/aemo/dt/grpo_modern
```

> `--dt-gamma 1.0` is the recommended default (matches the legacy PR#30 recipe).
> Adaptive RTG resampling was removed in the September 2026 cleanup (it measured
> +8% worse in the experiments below). Discounted RTG (`0.99`–`0.995`) is now safe
> to try via the `stable_rtg_update` clamp, but validate against a `gamma=1.0`
> baseline first.

The example figures in this guide are historical reference points, not guaranteed outcomes for a fresh run. Treat them as experiment context rather than as the stable contract of the workflow.

**GPU memory**: `--group-size 8` requires ~14 GB VRAM; `--group-size 4` requires
~8 GB. Both work on a 22 GB RTX 2080 Ti. Use 4 if you encounter OOM.

## Feature Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 5 | Number of GRPO rounds. >5 regresses without `--sync-reference-every` |
| `--lr` | 1e-5 | Learning rate. 5e-5 works on 24h eps but 1e-5 best for 144h |
| `--kl-coeff` | 0.02 | Higher values prevent policy drift but slow learning |
| `--group-size` | 4 | Episodes per advantage group. 8 gives lower variance |
| `--sync-reference-every` | 0 (off) | Sync policy → reference every N iterations (enables >5 iters) |
| `--deg-penalty-weight` | 1.0 | Extra degradation penalty (1.5 reduces deg by 67%) |
| `--dt-gamma` | 1.0 | RTG discount factor. 1.0 (undiscounted) recommended; `0.99`–`0.995` now safe via `stable_rtg_update` clamp. `0.95` caused RTG overflow on long horizons before the clamp fix |
| `--rtg-count` | 4 | Number of RTG values to sample for group diversity |
| `--battery-capacity` | 10.0 | Battery capacity in MWh |
| `--max-power` | 5.0 | Max charge/discharge rate in MW |
| `--episode-hours` | 24.0 | Episode length. 48h balances speed and quality |
| `--step-duration` | 0.08333 | Step duration in hours (5 min, matching DT training data) |

## Choosing the Right Battery

The v2 pretrained model was trained on 4 battery configurations. Match the
GRPO training battery to your target use case:

| Use Case | `--battery-capacity` | `--max-power` | C-rate |
|----------|:--------------------:|:-------------:|:------:|
| General purpose (1C) | 10.0 | 10.0 | 1.0C |
| Fast FCAS response | 8.0 | 30.0 | 3.75C |
| Legacy (slow) | 10.0 | 5.0 | 0.5C |
| Large grid support | 50.0 | 35.0 | 0.7C |

## Loading a Custom Model from HuggingFace

Both training scripts download from `mrvictoru/energydecision-dt` by default.
To use a different checkpoint:

```bash
# Option 1: Symlink override
# The scripts fall back to models/aemo/dt/grpo_phase1/dt_model_grpo_multi.pt
# if the HF download fails. Point this symlink to your custom checkpoint:
ln -sf /path/to/your/model.pt models/aemo/dt/grpo_phase1/dt_model_grpo_multi.pt

# Option 2: Modify the source
# In run_grpo_multi_region.py, change the HF repo/filename defaults.
```

## Outputs

After training completes, the `--output-dir` contains:

| File | Description |
|------|-------------|
| `dt_model_grpo_multi.pt` | Fine-tuned model weights (230 MB) |
| `grpo_loss_history.csv` | Training loss per iteration |
| `grpo_surface_manifest.json` | Model manifest for the evaluator |
| `training.log` | Full training log |

## Post-Training Evaluation

After GRPO, evaluate against baselines using the standardised tiers described in [evaluation_guide.md](evaluation_guide.md):

```bash
# Smoke test (fast)
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_modern/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_smoke.json \
  --output-dir eval_output/grpo_modern_smoke

# Standard benchmark (core comparison)
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_modern/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/grpo_modern_standard

# Comprehensive profile
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_modern/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/grpo_modern_comprehensive
```

## RTG Prompt Calibration

Find the optimal RTG prompt using the smoke tier:

```bash
for RTG in 0.0 0.5 1.0 1.5 2.0; do
  python3 -c "
import json
cfg = json.load(open('configs/eval_tier_smoke.json'))
for p in cfg['policies']:
    if p['kind'] == 'dt':
        p['rtg_value'] = $RTG
json.dump(cfg, open(f'/tmp/eval_smoke_rtg$RTG.json', 'w'), indent=2)
"
  python3 scripts/autoresearch_evaluator.py \
    --surface-manifest-path models/aemo/dt/grpo_modern/grpo_surface_manifest.json \
    --evaluation-config /tmp/eval_smoke_rtg$RTG.json \
    --output-dir eval_output/rtg_sweep/$RTG --device auto

  python3 -c "
import json
d = json.load(open(f'eval_output/rtg_sweep/$RTG/evaluation_summary.json'))
for m in d['heldout_evaluation']['aggregate_metrics']:
    if m['experiment'] == 'candidate_dt':
        print(f'RTG=$RTG: profit=\${m[\"avg_profit_per_episode\"]:,.0f}')
"
done
```

Use the best RTG for the Standard and Comprehensive tiers.

For broader experiment context and historical comparisons, see [research/README.md](research/README.md) and [grpo_experiments.md](grpo_experiments.md).
