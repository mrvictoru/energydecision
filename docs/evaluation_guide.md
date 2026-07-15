# Evaluation Guide — Standardised Evaluation for AEMO Models

This document defines a **standardised 3-tier evaluation system** so results are
comparable apple-to-apple across any Decision Transformer model (legacy,
modern, GRPO-tuned).

## Quick Start

```bash
# 1. Create a surface manifest (see below)
# 2. Run the evaluator
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path /path/to/manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/my_eval \
  --device auto

# 3. Read results from eval_output/my_eval/evaluation_summary.json
```

---

## The 3 Standard Tiers

Every model should be evaluated on all three tiers, from fastest to most thorough.
This table tells you which config to use and what it tests:

| Tier | Config | Time | Episodes | Regions | Battery | Baselines | What it tests |
|------|--------|:----:|:--------:|:-------:|:--------|-----------|---------------|
| **Smoke** | `eval_tier_smoke.json` | ~2 min | 24h (2 reg.) | NSW1, SA1 | 1C (10/10) | dispatch, PPO, FCAS rule | Sanity check — does the model run without errors? |
| **Standard** | `eval_tier_standard.json` | ~15 min | 144h (5 reg.) | All 5 NEM | 1C (10/10) | dispatch, PPO, FCAS rule | Core benchmark — profit, FCAS, deg vs baselines |
| **Comprehensive** | `eval_tier_comprehensive.json` | ~30 min | 144h × 2 (7 sc.) | All 5 NEM | 1C + small | dispatch, PPO, FCAS rule | Full profile — cross-region + 2 battery sizes |

### Common properties across all tiers

| Property | Value | Rationale |
|----------|-------|-----------|
| **Action mode** | `full_fcas` (9-dim) | Matches all v2+ models |
| **Step duration** | 0.08333h (5 min) | Matches pretrained data resolution |
| **Reference policy** | `dispatch_dalrymple_north` | Real operator baseline |
| **PPO model** | `ppo_aemo_fcas_model.zip` | FCAS-capable PPO |
| **DT RTG** | `0.5` (default, calibrate per model) | See RTG section below |
| **Degradation** | `real_world` (LFP, 30°C) | Standard chemistry |

---

## Required: Create a Surface Manifest

The evaluator needs a JSON file describing your model's architecture and
checkpoint path:

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

checkpoint = hf_hub_download("mrvictoru/energydecision-dt-v2", "aemo_dt_model.pt")

# For legacy model (w/embed_return):
model_kwargs_legacy = {
    "state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384,
    "context_len": 180, "n_heads": 8, "drop_p": 0.15,
    "max_timestep": 100000,
}

# For modern model (w/GQA, RoPE, embed_rtg):
model_kwargs_modern = {
    "state_dim": 18, "act_dim": 9, "n_block": 8, "h_dim": 384,
    "context_len": 180, "n_heads": 8, "drop_p": 0.15,
    "max_timestep": 100000,
    "rope_enabled": True,
    "rope_max_position": 540,
    "rope_base": 10000.0,
}

manifest = {
    "schema": "energydecision.dt_training_surface.v1",
    "model_kwargs": model_kwargs_modern,
    "paths": {
        "save_path": checkpoint,
        "loss_csv_path": "/tmp/dummy_loss.csv",
    },
}
Path("/tmp/manifest.json").write_text(json.dumps(manifest, indent=2))
Path("/tmp/dummy_loss.csv").write_text("epoch,train_total,train_action,val_total,val_action\n1,0.0,0.0,,\n")
```

---

## Running the 3 Tiers

```bash
# Tier 1: Smoke
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/manifest.json \
  --evaluation-config configs/eval_tier_smoke.json \
  --output-dir eval_output/smoke

# Tier 2: Standard (core benchmark)
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/standard

# Tier 3: Comprehensive (full profile)
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/comprehensive
```

---

## Interpreting Results

The `evaluation_summary.json` contains `aggregate_metrics` — one object per
policy. The key fields:

| Field | What it means | Target |
|-------|---------------|:------:|
| `mean_reward` | Average per-step reward (scaled) | Higher is better |
| `avg_profit_per_episode` | Total revenue minus degradation cost ($) | Higher is better |
| `avg_energy_revenue_per_episode` | Energy arbitrage revenue ($) | Higher is better |
| `avg_fcas_revenue_per_episode` | FCAS market revenue ($) | Higher is better |
| `avg_total_degradation_cost_per_episode` | Battery wear cost ($) | **Lower** is better |
| `avg_profit_per_mwh` | Normalised profit ($/MWh capacity) | Higher is better |
| `sharpe_ratio` | Risk-adjusted return (mean/std) | Higher is better |
| `paired_comparisons_vs_reference` | Statistical comparison vs dispatch | Positive = beats dispatch |

### Reading the output

```python
import json
with open("eval_output/standard/evaluation_summary.json") as f:
    d = json.load(f)
for m in d["heldout_evaluation"]["aggregate_metrics"]:
    print(f'{m["experiment"]:35s} profit=${m["avg_profit_per_episode"]:>8,.0f} '
          f'fcas=${m["avg_fcas_revenue_per_episode"]:>8,.0f} '
          f'deg=${m["avg_total_degradation_cost_per_episode"]:>8,.0f} '
          f'profit/MWh=${m["avg_profit_per_mwh"]:>7,.1f}')
```

---

## RTG Calibration

The `rtg_value` passed to the DT affects profit by up to 50%. Always calibrate
for each new model:

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
  python3 src/autoresearch_evaluator.py \
    --surface-manifest-path /tmp/manifest.json \
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

**Known optimal values** (starting point):
- Legacy HF model: `rtg_value=0.0`
- Phase 1 GRPO on legacy: `rtg_value=0.5`
- Modern model: calibrate (start with 0.5)

---

## GPU Memory Requirements

| Operation | Memory | Config |
|-----------|:------:|--------|
| Evaluating (inference only) | ~1 GB | Single model forward pass |
| GRPO training (group=4) | ~8 GB | 1 policy + 1 reference model |
| GRPO training (group=8) | ~14 GB | Larger rollout batch |

### GRPO OOM troubleshooting

If GRPO training OOMs with a modern model:

1. **Reduce `--group-size`** from 8 to 4 (saves ~6 GB, halves rollout memory)
2. **Reduce `--episode-hours`** from 48 to 24 (fewer steps, smaller batch)
3. **Check `max_timestep`** — the modern model's `embed_timestep` has 100,000
   entries × 384 dim = 153 MB. This is fine on 22 GB.
4. **Check for zombie processes** — `nvidia-smi` may show orphaned GPU processes
   from previous runs. Kill them with:
   ```bash
   ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
   ```

### Recommended GRPO config for 22 GB GPU

```bash
python3 src/run_grpo_multi_region.py \
  --regions NSW1,SA1,QLD1,VIC1,TAS1 \
  --battery-configs medium_1c,large_07c,small_05c,fast_375c \
  --step-duration 0.083333 --episode-hours 48 \
  --iterations 5 --lr 1e-5 --group-size 4 \
  --sync-reference-every 5 --deg-penalty-weight 1.5 \
  --output-dir models/aemo/dt/grpo_modern
```

`--group-size 4` instead of 8 saves ~6 GB of VRAM.

---

## Critical Insights from Eval Report

### Eval surface matters: different configs, different winners

The same model can rank differently depending on the eval config. Always
evaluate on at least two surfaces:

| Surface | Battery | Regions | What it tests |
|---------|---------|---------|---------------|
| **Standard** (recommended) | 1C fixed (10/10) | 5 regions | Cross-region generalisation |
| **Dispatch-matched** (supplement) | Station-specific | SA1 only | Head-to-head vs real operators |

From actual results:
- Phase 1 GRPO on Standard: **$1,855/ep** (worse than baseline)
- Phase 1 GRPO on Dispatch-matched: **$8,242/ep** (beats baseline)
- v2 pretrained on Standard: **$3,098/ep** (better than GRPO)
- v2 pretrained on Dispatch-matched: **$7,392/ep** (close to GRPO)

**Conclusion**: Report the Standard tier as the primary benchmark. Use
Dispatch-matched as a secondary comparison.

### RTG calibration is half the improvement

| Model | RTG 0.0 | RTG 0.5 | Gain |
|-------|:-------:|:-------:|:----:|
| Phase 1 GRPO on dispatch-matched | $5,451 | $8,242 | **+51%** |

Always calibrate RTG before reporting final results.

### The modern v2 model is already strong

The modern v2 pretrained model ($7,392/ep on dispatch-matched) nearly matches
the legacy Phase 1 GRPO ($8,242/ep). The modern architecture (GQA, RoPE) may
capture much of the benefit that GRPO provides for legacy models.

---

## Summary: Before & After Comparison

| Before (inconsistent) | After (standardised) |
|-----------------------|----------------------|
| 10 config files, mixed `multi_market`/`full_fcas` | 3 tier configs, all `full_fcas` |
| Mixed 30-min / 5-min steps | All 5-min (matches data) |
| Mixed PPO models (`_model.zip` vs `_fcas_model.zip`) | All use `ppo_aemo_fcas_model.zip` |
| No clear tier guidance | Smoke → Standard → Comprehensive |
| RTG not mentioned | RTG calibration required, default 0.5 |
| OOM guidance missing | Memory + group-size advice included |
