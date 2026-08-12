# Evaluation Guide — Standardised Evaluation for AEMO Models

This document defines a **standardised 3-tier evaluation system** so results are
comparable apple-to-apple across any Decision Transformer model (legacy,
modern, GRPO-tuned).

## Quick Start

```bash
# 1. Create a surface manifest (see below)
# 2. Run the evaluator
python3 scripts/autoresearch_evaluator.py \
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

> **Required for any "best"/SOTA claim:** in addition to the tiers above, run
> the **expanded broad surface** (`configs/aemo_autoresearch_evaluator.expanded_rtg10.json`,
> 5-min, 5 regions × 6 periods of 2024) **and the 2025 out-of-distribution
> surface** (`configs/aemo_autoresearch_evaluator.2025.json`, 5-min,
> NSW1/SA1/QLD1 × Jan/Feb 2025). A model that wins the narrow tiers but loses
> on the broad year or on a genuinely unseen year is not the leader (see
> "Critical Insights").

### Common properties across all tiers

| Property | Value | Rationale |
|----------|-------|-----------|
| **Action mode** | `full_fcas` (9-dim) | Matches all v2+ models |
| **Step duration** | 0.08333h (5 min) | Matches pretrained data resolution |
| **Reference policy** | `dispatch_dalrymple_north` | Real operator baseline |
| **PPO model** | `ppo_aemo_fcas_model.zip` | FCAS-capable PPO |
| **DT RTG** | Calibrate per model **and per surface** | RTG is a strategy selector; see RTG section below |
| **Degradation** | `real_world` (LFP, 30°C) | Standard chemistry |

---

## Required: Create a Surface Manifest

The evaluator needs a JSON file describing your model's architecture and
checkpoint path:

> **Do not hand-write `model_kwargs`.** Load them from the canonical config so
> they always match the weights, and verify any checkpoint's true architecture
> from its embedded `config`. For the modern v2 checkpoint, the canonical repo
> config is `configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json`.
> The v2 model is **8×768, 12 heads, 6 KV heads (GQA), qk_norm, tie_weights,
> ctx=210, learned timestep embeddings (rope_enabled=false)**, and its HF
> filename is **`aemo_dt_fcas_model.pt`**.

```python
from pathlib import Path
import json, sys
from huggingface_hub import hf_hub_download

sys.path.insert(0, "src")
from aemo_dt_hf import (
    MODERN_V2_HF_REPO, MODERN_V2_HF_FILENAME,
    modern_v2_model_config_path, load_model_kwargs, build_surface_manifest,
    write_placeholder_loss_csv,
)

checkpoint = hf_hub_download(MODERN_V2_HF_REPO, MODERN_V2_HF_FILENAME)  # aemo_dt_fcas_model.pt
model_kwargs = load_model_kwargs(modern_v2_model_config_path())        # verified 8x768 config
loss_csv = write_placeholder_loss_csv("/tmp/dummy_loss.csv")

manifest = build_surface_manifest(
    model_kwargs=model_kwargs, save_path=checkpoint, loss_csv_path=loss_csv,
    hf_repo=MODERN_V2_HF_REPO, hf_filename=MODERN_V2_HF_FILENAME,
)
Path("/tmp/manifest.json").write_text(json.dumps(manifest, indent=2))
```

---

## Running the 3 Tiers

```bash
# Tier 1: Smoke
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/manifest.json \
  --evaluation-config configs/eval_tier_smoke.json \
  --output-dir eval_output/smoke

# Tier 2: Standard (core benchmark)
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/standard

# Tier 3: Comprehensive (full profile)
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/comprehensive
```

---

## Forecast Decision Transformer

The `ForecastDecisionTransformer` extends the modern v2 architecture with
explicit forecast tokens (48-step prefix). Evaluating one requires extra steps:

### Surface manifest

The manifest must include `model_kwargs.model_class: "ForecastDecisionTransformer"`
and `model_meta: {"return_scale": 2.0}`. Use the helper script:

```bash
# Creates models/aemo/dt/hf_forecast/surface_manifest.json
# Uses the shared create_hf_surface_manifest.py with forecast-specific flags
python3 scripts/create_hf_surface_manifest.py \
  --hf-repo mrvictoru/energydecision-dt-v2-forecast \
  --model-config configs/aemo_decision_transformer_model_kwargs_forecast.json \
  --output-dir models/aemo/dt/hf_forecast \
  --surface-preset hf_modern_forecast
```

This downloads the checkpoint from `mrvictoru/energydecision-dt-v2-forecast`,
loads `configs/aemo_decision_transformer_model_kwargs_forecast.json`, and
injects the required extra fields.

### Running the evaluator

Pass the TTM forecast lookup file via `--forecast-npz-path`:

```bash
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/hf_forecast/surface_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/forecast_dt_standard \
  --device cuda \
  --forecast-npz-path data/aemo_dt_forecast/ttm_forecasts.npz
```

If omitted, the evaluator defaults to `data/aemo_dt_forecast/ttm_forecasts.npz`
relative to the repo root.

### Forecast index alignment

The TTM forecast `.npz` is indexed by global position in the full merged AEMO
dataset. The evaluator aligns it to the per-scenario data using the
`SETTLEMENTDATE` timestamps — the `AEMOAgent` computes an offset that maps
row 0 of the scenario slice to the correct npz position, then looks up
forecasts at position `offset + episode_start_idx + context_len + fi`.

### Current evaluation results

The forecast DT underperforms the modern v2 baseline on the Standard tier:

| Model | Profit/ep | FCAS/ep | Energy/ep | Deg/ep |
|-------|----------:|--------:|----------:|-------:|
| Modern v2 DT (rtg=0.5) | **$4,726** | $3,063 | $1,896 | $232 |
| Forecast DT (rtg=0.5) | -$302 | $25 | $283 | $610 |
| Forecast DT (rtg=1.0) | -$291 | $25 | $282 | $598 |
| Forecast DT (rtg=0.0) | -$324 | $25 | $283 | $633 |
| dispatch_dalrymple_north | $4,660 | $2,287 | $3,394 | $1,020 |
| ppo_reference | $2,353 | $2,192 | $396 | $236 |

The forecast DT shows negligible FCAS revenue ($25/ep vs $3,063 for modern v2)
and elevated degradation ($600+/ep vs $232). This is consistent with the plan
document's finding that the modern v2 pretrained is SOTA — GRPO could not
improve it, and the forecast architecture does not help on the Standard surface.

**Known limitations of the current evaluation:**
- The TTM npz channels map to obs[5:11] (RRP, DEMAND, 4 FCAS); the remaining
  12 obs dims are zero. The model was trained with the same channel layout.
- The forecast is a 48-step look-ahead starting at episode position
  `context_len` (210), matching the training sliding-window convention.
- `return_scale=2.0` is set via `model_meta` in the surface manifest. Calibrate
  RTG for your model — start with `rtg=0.5` (which maps to 0.25 model-space).

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
  python3 scripts/autoresearch_evaluator.py \
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

**Known optimal values** (starting points — calibrate per surface):
- Legacy HF model: `rtg_value=0.0`
- Phase 1 GRPO on legacy: `rtg_value=0.5`
- Modern v2: **surface-dependent** — peaks at `10` on the Oct-standard surface,
  `0` on dispatch-matched, and is **flat** (0–50) on the broad expanded 2024
  surface. Always sweep on the exact surface you will report.

> **RTG-distribution note (2026-08-07):** training returns-to-go are tiny
> (p50 ≈ 0, p90 ≈ 1.5, max ~9–13) and *decay over an episode*, but the
> evaluator feeds a *constant* rtg (0–50) at every step — far outside the
> training distribution. High RTG acts as an aggressive strategy selector, not
> a faithful return estimate. A dynamic/decaying RTG prompt is an open
> experiment.

---

## ⚠️ Pre-flight: Clean GPU Memory (Required Before Each Run)

**The #1 cause of OOM is orphaned GPU processes, not model size.**
Python processes that crash or are killed leave GPU memory allocated.
A single zombie can hold 14+ GB on a 22 GB card.

Always run this **before any training or evaluation**:

```bash
# 1. Check what's using GPU memory
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# 2. Kill ALL Python processes (training scripts restart fresh)
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

# 3. Verify GPU is free
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # expect < 500 MiB
```

## GPU Memory (8×768 modern model, ctx=210)

| Operation | group_size | Rollout batch | Total VRAM | On 22 GB? |
|-----------|:----------:|:-------------:|:----------:|:---------:|
| Evaluation only | — | — | ~1 GB | ✅ |
| GRPO training | 4 | ~6 GB | ~13 GB | ✅ Safe (recommended) |
| GRPO training | 8 | ~12 GB | ~19 GB | ✅ After zombie cleanup |
| GRPO training | 2 | ~3 GB | ~10 GB | ⛔ Too noisy — skip |

**Do not use `--group-size 2`.** Advantage estimates become unstable.
`--group-size 4` is the minimum for reliable training.

**No code changes needed.** Gradient checkpointing and mixed precision are
not required — the 8×768 model fits comfortably at `group_size=4` on a
22 GB card after cleaning zombies.

---

## Critical Insights from Eval Report

### Eval surface matters: different configs, different winners

The same model can rank differently depending on the eval config. Always
evaluate on at least two surfaces:

| Surface | Battery | Regions | What it tests |
|---------|---------|---------|---------------|
| **Standard** (recommended) | 1C fixed (10/10) | 5 regions | Cross-region generalisation |
| **Expanded / regime-shift** | 10/5 medium | 5 regions × 6 periods of 2024 | Broad-year stability (see below) |
| **Dispatch-matched** (supplement) | Station-specific | SA1 only | Head-to-head vs real operators |

From actual results:
- Phase 1 GRPO on Standard: **$1,855/ep** (worse than baseline)
- Phase 1 GRPO on Dispatch-matched: **$8,242/ep** (beats baseline)
- v2 pretrained on Standard: **$3,098/ep** (better than GRPO)
- v2 pretrained on Dispatch-matched: **$7,392/ep** (close to GRPO)

**Regime-shift / broad-surface finding (2026-08-07):** on the **expanded 2024
surface** (5 regions × 6 periods, 5-min, `expanded_rtg10.json`) the modern v2
DT earns **$4.6k/ep vs PPO's $15.0k** — the DT's "SOTA" is **surface-specific**
(it wins Oct-standard, dispatch-matched, and mild months like Jan, but loses
broadly to PPO in FCAS-spike months due to FCAS under-bidding). PPO's FCAS
capture is ~2× the DT's on the broad year.

**Conclusion**: **any claim that a model is "best" / SOTA now requires BOTH the
Standard tier, the expanded broad surface (`expanded_rtg10.json`), AND the
2025 out-of-distribution surface (`aemo_autoresearch_evaluator.2025.json`)**.
A model that wins Standard or even the expanded year but loses on a genuinely
unseen year (2025) is not the leader. Dispatch-matched remains a secondary
comparison for head-to-head vs real operators. The canonical launch plan
(`launch_aemo_training.py`) lists the broad surface in
`recommended_evaluation_configs`.

### RTG calibration is half the improvement

| Model | RTG 0.0 | RTG 0.5 | Gain |
|-------|:-------:|:-------:|:----:|
| Phase 1 GRPO on dispatch-matched | $5,451 | $8,242 | **+51%** |

Always calibrate RTG before reporting final results.

### The modern v2 model is already strong

The modern v2 pretrained model ($7,392/ep on dispatch-matched) nearly matches
the legacy Phase 1 GRPO ($8,242/ep). The modern architecture (GQA, RoPE) may
capture much of the benefit that GRPO provides for legacy models. **Caveat
(2026-08-07):** this strength is surface-specific — on the broad expanded 2024
surface PPO dominates (see "Eval surface matters" above), because the DT's
offline data under-represents FCAS-spike behaviour.

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
