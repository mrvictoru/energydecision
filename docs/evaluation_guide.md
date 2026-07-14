# Evaluation Guide — Loading, Fine-Tuning & Running Models

This document covers how to load a trained Decision Transformer model (from
disk or HuggingFace), run **GRPO online RL fine-tuning**, evaluate its
performance against baselines, and interpret results.

## Quick Start: Evaluate a Model Against Baselines

The fastest way to evaluate any DT model is the `autoresearch_evaluator.py`
script. It runs the model on held-out scenarios and compares against dispatch
replay, PPO, FCAS rule, and other baselines.

### 1. Prepare a surface manifest

The evaluator needs a JSON manifest describing the model architecture and
checkpoint path:

```json
{
  "schema": "energydecision.dt_training_surface.v1",
  "model_kwargs": {
    "state_dim": 18,
    "act_dim": 9,
    "n_block": 8,
    "h_dim": 384,
    "context_len": 180,
    "n_heads": 8,
    "drop_p": 0.15,
    "max_timestep": 100000
  },
  "paths": {
    "save_path": "/path/to/model.pt",
    "loss_csv_path": "/path/to/dummy_loss.csv"
  }
}
```

For HuggingFace models, use `hf_hub_download` to get the local path:

```python
from huggingface_hub import hf_hub_download
checkpoint = hf_hub_download("mrvictoru/energydecision-dt", "aemo_dt_grpo_model.pt")
```

### 2. Run the evaluator

```bash
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path /path/to/manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json \
  --output-dir eval_output/my_eval \
  --device auto
```

### 3. Read the results

The output directory contains:

| File | Contents |
|------|----------|
| `evaluation_summary.json` | All metrics in JSON |
| `heldout_metrics.csv` | Tabular metrics per policy |
| `plots/mean_reward.svg` | Mean reward comparison bar chart |
| `plots/risk_return.svg` | Risk-return scatter plot |
| `plots/episode_distribution.svg` | Return distribution box plot |
| `plots/grid_energy.svg` | Grid energy & degradation |

## Evaluating a Single Episode

For interactive debugging, run a single episode with `AEMOAgent`:

```python
from pathlib import Path
import torch
from decision_transformer import DecisionTransformer
from grpo_posttraining import load_pretrained_dt_for_grpo
from aemo_notebook_utils import create_aemo_env, fetch_and_preprocess_aemo_scenarios, resolve_battery_variants
from decision import AEMOAgent
from datetime import datetime

# 1. Load model
model_kwargs = {"state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384,
                "context_len": 180, "n_heads": 8, "drop_p": 0.15, "max_timestep": 100000}
model, _ = load_pretrained_dt_for_grpo(model_kwargs, "path/to/model.pt", device="cuda")
model.eval()

# 2. Load market data
processed, _ = fetch_and_preprocess_aemo_scenarios(
    scenarios=[{"label": "test", "region": "NSW1",
                "start_date": datetime(2024, 10, 1), "end_date": datetime(2024, 10, 14)}],
    cache_dir=Path("data/aemo"), step_duration=5/60, refresh=False,
)
battery = resolve_battery_variants([{"name": "medium_1c", "capacity_mwh": 10.0,
                                      "max_power_mw": 10.0, "init_soc_ratio": 0.5}])[0]
env = create_aemo_env(processed_data=processed["test"], battery_variant=battery,
                      max_step=288, step_duration=5/60, action_mode="full_fcas",
                      random_episode_start=True)

# 3. Run episode (use rtg_value=0.5 for best results)
agent = AEMOAgent(env, algorithm="dt", model=model, rtg_value=0.5, dt_gamma=0.95)
episode_df, _ = agent.run_episode()

# 4. Inspect results
info = episode_df["info"].struct.unnest()
print(f"Total reward: {episode_df['reward'].sum():.2f}")
print(f"Energy revenue: ${info['energy_revenue'].sum():,.0f}")
print(f"FCAS revenue: ${info['fcas_revenue'].sum():,.0f}")
print(f"Degradation cost: ${info['degradation_cost'].sum():,.0f}")
```

## Available Evaluation Configs

| Config | Regions | Battery | Baselines | Use Case |
|--------|---------|---------|-----------|----------|
| `q4_dispatch_matched.json` | SA1 | Dispatch-matched (Dalrymple North) | dispatch, PPO, FCAS rule | Head-to-head vs real operators |
| `q4_multi_station.json` | 5 regions | Fixed 1C (10MWh/10MW) | dispatch, PPO, FCAS rule | Cross-region generalisation |
| `q4_2024_heldout.json` | 5 regions | Fixed 1C | dispatch, PPO, FCAS rule | Held-out Q4 2024 evaluation |
| `dispatch_matched.json` | SA1 (4 seasons) | Dispatch-matched | dispatch, FCAS rule | Multi-season dispatch comparison |
| `example.json` | 4 regions | Medium + small | rule, FCAS rule, 2×dispatch, PPO | Full promotion check |
| `mini.json` | 2 regions | Medium | DT, FCAS rule | Quick smoke test (~2 min) |

## RTG Prompt Calibration

The RTG value passed to the DT affects performance significantly. For the best
Phase 1 GRPO model, `rtg_value=0.5` is optimal. To find the best RTG for a
different model, run the evaluator at multiple values:

```bash
# Test RTG values 0.0, 0.5, 1.0, 1.5, 2.0
for RTG in 0.0 0.5 1.0 1.5 2.0; do
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
    --surface-manifest-path models/aemo/dt/hf_pretrained_v2/hf_v2_surface_manifest.json \
    --evaluation-config /tmp/eval_cfg_$RTG.json \
    --output-dir eval_output/rtg_test/$RTG --device auto
done
```

## Loading from HuggingFace

```python
from huggingface_hub import hf_hub_download
from decision_transformer import DecisionTransformer
from grpo_posttraining import load_pretrained_dt_for_grpo

# For the GRPO-tuned model
checkpoint = hf_hub_download("mrvictoru/energydecision-dt", "aemo_dt_grpo_model.pt")

# For the pretrained baseline
checkpoint = hf_hub_download("mrvictoru/energydecision-dt", "aemo_dt_fcas_model.pt")

model_kwargs = {"state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384,
                "context_len": 180, "n_heads": 8, "drop_p": 0.15, "max_timestep": 100000}
model, _ = load_pretrained_dt_for_grpo(model_kwargs, checkpoint, device="cuda")
```

Both checkpoints use the same architecture. The `load_pretrained_dt_for_grpo`
function handles legacy MoLab-style checkpoints automatically.

## Comparing Two Models

Run the evaluator separately for each model, then compare the
`evaluation_summary.json` files:

```python
import json
for label, path in [("Model A", "eval_output/model_a/evaluation_summary.json"),
                    ("Model B", "eval_output/model_b/evaluation_summary.json")]:
    with open(path) as f:
        d = json.load(f)
    for m in d['heldout_evaluation']['aggregate_metrics']:
        if m['experiment'] == 'candidate_dt':
            print(f"{label}: profit=${m['avg_profit_per_episode']:,.0f} "
                  f"fcas=${m['avg_fcas_revenue_per_episode']:,.0f} "
                  f"deg=${m['avg_total_degradation_cost_per_episode']:,.0f}")
```

For paired statistical comparison, use the `paired_comparisons_vs_reference`
field in the evaluation summary.

---

## GRPO Online RL Fine-Tuning

Group Relative Policy Optimization (GRPO) improves a pretrained DT through
online interaction with the simulation environment. No dataset regeneration or
SB3 retraining is needed — the model learns by trying actions and observing
rewards.

### Quick Start: Single-Region

```bash
python3 src/run_grpo_posttraining.py \
  --region NSW1 --start-date 2024-01-01 --end-date 2024-01-14 \
  --iterations 5 --episode-hours 144 --step-duration 0.083333
```

This downloads the HF model automatically and runs 5 GRPO iterations on
NSW1 January 2024 data at 5-minute resolution.

### Quick Start: Multi-Region (Recommended)

```bash
python3 src/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --start-date 2024-01-01 --end-date 2024-09-30 \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --kl-coeff 0.02 \
  --battery-capacity 10 --max-power 10 \
  --rtg-count 4 --dt-gamma 0.95
```

### Phase 1 GRPO Features (Recommended Config)

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

### Feature Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 5 | Number of GRPO rounds. >5 regresses without `--sync-reference-every` |
| `--lr` | 1e-5 | Learning rate. 5e-5 works on 24h eps but 1e-5 best for 144h |
| `--kl-coeff` | 0.02 | Higher values prevent policy drift but slow learning |
| `--group-size` | 4 | Episodes per advantage group. 8 gives lower variance |
| `--sync-reference-every` | 0 (off) | Sync policy → reference every N iterations (enables >5 iters) |
| `--adaptive-rtg` | off | Resample RTG prompts from EWMA of realised returns |
| `--deg-penalty-weight` | 1.0 | Extra degradation penalty (1.5 reduces deg by 67%) |
| `--dt-gamma` | 1.0 | RTG discount factor. Match training; 0.95 is typical |
| `--rtg-count` | 4 | Number of RTG values to sample for group diversity |
| `--battery-capacity` | 10.0 | Battery capacity in MWh |
| `--max-power` | 5.0 | Max charge/discharge rate in MW |

### Choosing the Right Battery

The v2 pretrained model was trained on 4 battery configurations. Match the
GRPO training battery to your target use case:

| Use Case | `--battery-capacity` | `--max-power` | C-rate |
|----------|:--------------------:|:-------------:|:------:|
| General purpose (1C) | 10.0 | 10.0 | 1.0C |
| Fast FCAS response | 8.0 | 30.0 | 3.75C |
| Legacy (slow) | 10.0 | 5.0 | 0.5C |
| Large grid support | 50.0 | 35.0 | 0.7C |

### RTG Prompt Calibration (Post-Training)

After GRPO, find the optimal RTG for evaluation by sweeping values:

```bash
for RTG in 0.0 0.5 1.0 1.5 2.0; do
  python3 src/autoresearch_evaluator.py \
    --surface-manifest-path models/aemo/dt/grpo_phase1/grpo_surface_manifest.json \
    --evaluation-config configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json \
    --output-dir eval_output/rtg_sweep/$RTG --device auto
done
```

The Phase 1 model performs best at `rtg_value=0.5`.

### Loading a Custom HF Model (not the default)

Both `run_grpo_posttraining.py` and `run_grpo_multi_region.py` download
from `mrvictoru/energydecision-dt` by default. To use a different HF repo:

```bash
# The scripts download via hf_hub_download("mrvictoru/energydecision-dt", "aemo_dt_fcas_model.pt")
# To override, either:
# 1. Upload your model to the same repo under a different filename and modify the source,
# 2. Or load locally:
python3 src/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  ... [other args] ...
  # The script falls back to models/aemo/dt/grpo_phase1/dt_model_grpo_multi.pt
  # if the HF download fails. Point this symlink to your custom checkpoint:
  # ln -sf /path/to/your/model.pt models/aemo/dt/grpo_phase1/dt_model_grpo_multi.pt
```

### Outputs

| File | Description |
|------|-------------|
| `dt_model_grpo_multi.pt` | Fine-tuned model weights |
| `grpo_loss_history.csv` | Training loss per iteration |
| `grpo_surface_manifest.json` | Model manifest (for evaluator) |
