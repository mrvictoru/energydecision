# GRPO Fine-Tuning Guide

Group Relative Policy Optimization (GRPO) improves a pretrained Decision
Transformer through online interaction with the simulation environment.
No dataset regeneration or SB3 retraining is needed — the model learns by
trying actions and observing rewards.

## Quick Start: Single-Region

```bash
python3 src/run_grpo_posttraining.py \
  --region NSW1 --start-date 2024-01-01 --end-date 2024-01-14 \
  --iterations 5 --episode-hours 144 --step-duration 0.083333
```

This downloads the HF model from `mrvictoru/energydecision-dt` automatically
and runs 5 GRPO iterations on NSW1 January 2024 data at 5-minute resolution.

## Quick Start: Multi-Region (Recommended)

```bash
python3 src/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --start-date 2024-01-01 --end-date 2024-09-30 \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --kl-coeff 0.02 \
  --battery-capacity 10 --max-power 10 \
  --rtg-count 4 --dt-gamma 0.95
```

## Phase 1 GRPO Features (Recommended Config)

The best results come from enabling all Phase 1 improvements together:

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
  --deg-penalty-weight 1.5
```

Expected improvement: +$8,242/ep profit, $1,030/MWh normalized, beats PPO
and dispatch Dalrymple North on a matched battery.

## Feature Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 5 | Number of GRPO rounds. >5 regresses without `--sync-reference-every` |
| `--lr` | 1e-5 | Learning rate. 5e-5 works on 24h eps but 1e-5 best for 144h |
| `--kl-coeff` | 0.02 | Higher values prevent policy drift but slow learning |
| `--group-size` | 4 | Episodes per advantage group. 8 gives lower variance |
| `--sync-reference-every` | 0 (off) | Sync policy → reference every N iterations (enables >5 iters) |
| `--adaptive-rtg` | off | Resample RTG prompts from EWMA of realised returns |
| `--deg-penalty-weight` | 1.0 | Extra degradation penalty (1.5 reduces deg by 67%) |
| `--dt-gamma` | 1.0 | RTG discount factor. Match training discount; 0.95 is typical |
| `--rtg-count` | 4 | Number of RTG values to sample for group diversity |
| `--battery-capacity` | 10.0 | Battery capacity in MWh |
| `--max-power` | 5.0 | Max charge/discharge rate in MW |
| `--episode-hours` | 24.0 | Episode length. 48h balances speed and quality |
| `--step-duration` | 0.5 | Step duration in hours. Use 0.08333 (5 min) to match pretrained data |

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

After GRPO, evaluate against baselines:

```bash
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_phase1/grpo_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json \
  --output-dir eval_output/my_eval --device auto
```

## RTG Prompt Calibration

Find the optimal RTG prompt for your fine-tuned model:

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
    --surface-manifest-path models/aemo/dt/grpo_phase1/grpo_surface_manifest.json \
    --evaluation-config /tmp/eval_cfg_$RTG.json \
    --output-dir eval_output/rtg_sweep/$RTG --device auto
done

# Then compare results:
for RTG in 0.0 0.5 1.0 1.5 2.0 2.5 3.0; do
  python3 -c "
import json
with open('eval_output/rtg_sweep/$RTG/evaluation_summary.json') as f:
    d = json.load(f)
for m in d['heldout_evaluation']['aggregate_metrics']:
    if m['experiment'] == 'candidate_dt':
        print(f'RTG=$RTG: profit=\${m[\"avg_profit_per_episode\"]:,.0f} deg=\${m[\"avg_total_degradation_cost_per_episode\"]:,.0f}')
"
done
```
