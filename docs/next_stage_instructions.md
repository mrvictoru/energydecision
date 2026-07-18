# Next Stage: GRPO Fine-Tuning Instructions (Updated)

## Session Summary (2026-07-16)

- Confirmed that the current GRPO run used the modern v2 Hugging Face checkpoint and modern v2 model config (`mrvictoru/energydecision-dt-v2` + `configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json`).
- Cached the checkpoint locally before training, then ran a multi-region / multi-battery GRPO fine-tuning pass with the repo-native script `scripts/run_grpo_multi_region.py`.
- The new checkpoint was written to `models/aemo/dt/grpo_modern_v2_multibat_144h/dt_model_grpo_multi.pt`.
- The training run completed successfully; the simple rollout eval improved from baseline `-18.29` to post-GRPO `-0.25` on the built-in GRPO eval (`+18.04` improvement).
- Evaluated the new checkpoint on two surfaces:
  - Example evaluator (`configs/aemo_autoresearch_evaluator.example.json`): the new checkpoint beat the FCAS-rule reference in paired comparison (`mean_diff +4.08`, `p=3.05e-05`) but still posted negative profit on this surface (about `-$156/ep`).
  - Dispatch-matched evaluator (`configs/aemo_autoresearch_evaluator.dispatch_matched.json`): the new checkpoint scored about `-$206/ep` profit vs dispatch replay `+$964/ep` and FCAS-rule `-$67k/ep` on the same asset. This means the new GRPO checkpoint did not yet match the historical legacy Phase 1 GRPO dispatch-matched result on this surface.
- Comparison vs prior baselines:
  - Modern v2 pretrained standard surface: about `+$4,726/ep` profit, `+$3,063/ep` FCAS revenue.
  - Legacy pretrained standard surface: about `+$645/ep` profit, `+$2,518/ep` FCAS revenue.
  - Legacy Phase 1 GRPO (historical report): about `+$8,242/ep` profit and `+$7,686/ep` FCAS revenue on the dispatch-matched surface with RTG calibration.
  - Previous modern-v2 GRPO checkpoint (earlier run): about `+$1,697/ep` profit on the standard surface.
- The current run therefore looks promising as a modern-v2 GRPO experiment, but it has not yet reproduced the strongest legacy GRPO gain on the dispatch-matched surface. The next worthwhile step is a standard-tier evaluation sweep for this new checkpoint (and optionally an RTG sweep) to see whether the present training recipe is competitive with the pretrained modern-v2 baseline on the standard surface.

## Follow-up Finding: why the current modern-v2 GRPO run collapsed (now fixed in code)

- The current 144h modern-v2 checkpoint was re-evaluated on the dispatch-matched surface with `rtg_value=0.5`, and it produced the same collapsed result as the `rtg_value=0.0` evaluation: about `-$206/ep` profit with `0` FCAS revenue. This rules out "wrong RTG at eval time" as the primary explanation.
- The root cause was in `src/grpo_posttraining.py`: during rollout collection, RTG was updated as `next_rtg = (rtg_t - reward_t) / dt_gamma` whenever `dt_gamma != 1.0`.
- That formula is the *correct mathematical inverse* of the training convention (`helper._compute_rtgs_from_rewards`: `rtg[t] = r[t] + gamma * rtg[t+1]`), but it is numerically unstable: with `dt_gamma=0.95` it amplifies RTG by `1 / 0.95` every step, compounding as `(1/gamma)^t` on long horizons:
  - 288 steps: about `2.6e6`
  - 576 steps: about `6.8e12`
  - 1728 steps: about `3.1e38`
- This matched the observed failure mode of the 144h run:
  - runtime warning: `overflow encountered in cast`
  - epoch-1 `train_total` on the 144h run jumped to about `3.2e15`
  - the trained policy collapsed to near-zero actions on dispatch-matched eval
- The earlier modern-v2 GRPO run did not collapse as hard because it used 48h episodes, so the same RTG amplification was present but less catastrophic. It still underperformed the pretrained modern-v2 baseline, which is consistent with a degraded but not fully overflowed training signal.
- This also explains why the legacy PR#30 setup could improve while the new modern-v2 runs did not: the older `src/run_grpo_posttraining.py` path defaulted `dt_gamma=1.0`, so it avoided the RTG blow-up entirely.

### Fix landed (2026-07-17)

- `src/grpo_posttraining.py` now exposes `stable_rtg_update(current_rtg, reward, dt_gamma=..., initial_rtg=...)`. It keeps the exact undiscounted recurrence for `gamma==1.0` and, for `gamma<1.0`, applies the discounted update **clamped to the trained RTG envelope** (`max(4*|initial_rtg|, 20)`), so RTG can no longer compound to overflow. The `_validate_grpo_rtg` guard remains as a safety net.
- `src/decision.py` inference RTG updates (both `AEMOAgent` codepaths) now call the same `stable_rtg_update`, keeping eval-time behaviour consistent with rollout.
- Regression tests in `tests/test_grpo_posttraining.py` assert: `gamma=1.0` is exact, short-horizon `gamma<1` matches the exact inverse before the clamp binds, and a 1728-step `gamma=0.95` rollout stays finite and bounded.
- **With this fix, `dt_gamma<1.0` is safe again.** We still recommend starting the next comparison at `dt_gamma=1.0` (see the recipe below) and only introducing discounting once the `gamma=1.0` baseline is understood.
- Secondary contributors that likely hurt the modern-v2 runs further:
  - `adaptive_rtg=True` is already documented as counterproductive when realized returns are negative.
  - the newer multi-region / multi-battery training recipe is not apples-to-apples with the older single-region legacy GRPO recipe.
  - the built-in `run_grpo_multi_region.py` baseline/post-GRPO eval is a lightweight sanity check only; it is not a substitute for the standard or dispatch-matched benchmark surfaces.

## Handoff Notes for the Next Agent Session

### What `dt_gamma` does

- `dt_gamma` is the discount factor used when the GRPO rollout code updates the RTG buffer after each environment step.
- In this repo, the update goes through `stable_rtg_update`: `rtg_{t+1} = rtg_t - reward_t` for `dt_gamma==1.0`, and the clamped discounted form `clip((rtg_t - reward_t) / dt_gamma, ±envelope)` for `dt_gamma<1.0`.
- The clamp (`envelope = max(4*|initial_rtg|, 20)`) keeps RTG inside the trained distribution, so `dt_gamma<1.0` no longer explodes on long horizons (this was the 144h collapse cause, now fixed).

### Acceptable range for `dt_gamma`

- `dt_gamma=1.0` remains the recommended default for the first comparison runs — it is the exact undiscounted recurrence and matches the legacy PR#30 recipe, so it is the cleanest apples-to-apples baseline.
- `dt_gamma` in `[0.99, 0.995]` is now safe to try thanks to the clamp; use it only after the `gamma=1.0` baseline is understood.
- `dt_gamma=0.95` no longer overflows (the clamp bounds it), but it is aggressive and pushes RTG to the envelope edge quickly on long horizons — prefer values closer to 1.0.
- `dt_gamma > 1.0` is not standard for this setting and should be avoided unless you deliberately want a different semantics.

### Why the legacy GRPO model looked better on dispatch-matched but worse on standard

- The legacy GRPO result is consistent with overfitting to the narrower dispatch-matched setup rather than learning a broadly robust policy.
- The dispatch-matched eval is a focused benchmark: one region, one battery asset, one time window. A policy that is tuned to that surface can look very strong there.
- The standard eval is broader and more diverse, so a specialist policy often loses.
- This is the strongest explanation we have from the evidence, but it is still an interpretation rather than a proven cause. The next agent should treat it as a working hypothesis to test.

### Recommended starting recipe for the next comparison run

1. Use `dt_gamma=1.0`.
2. Keep `adaptive_rtg` disabled for the first comparison.
3. Start with the simpler single-region setup (for example NSW1) before reintroducing multi-region and multi-battery randomness.
4. Run explicit RTG sweeps on both the dispatch-matched and standard surfaces.
5. Compare against: modern-v2 pretrained, earlier modern-v2 GRPO, and the historical legacy GRPO result.

### Session handoff location

- The session plan file is at `/home/victoru/.copilot/session-state/d8301de1-1bfe-413b-b401-0b3a5c97df26/plan.md`.
- The corrected modern-v2 checkpoint from this session is at `models/aemo/dt/grpo_modern_v2_single_nsw1_gamma1/dt_model_grpo_multi.pt`.
- The corrected evaluation outputs are under `eval_output/autoresearch/grpo_gamma1_dispatch_rtg_*` and `eval_output/autoresearch/grpo_gamma1_standard_rtg_*`.

## Files the Next Agent Must Read

| # | File | Why |
|---|------|-----|
| 1 | **This file** (`docs/next_stage_instructions.md`) | Step-by-step instructions for GRPO + eval |
| 2 | `docs/grpo_finetuning_guide.md` | GRPO feature reference, battery selection, RTG calibration |
| 3 | `docs/evaluation_guide.md` | Standardised 3-tier evaluation system |
| 4 | `docs/grpo_experiments.md` | Previous experiment results for comparison reference |
| 5 | `scripts/run_grpo_multi_region.py` | The actual GRPO training script (must read its args) |
| 6 | `src/aemo_dt_hf.py` | Modern v2 model defaults (repo, filename, config path) |
| 7 | `configs/eval_tier_standard.json` | Standard eval config (primary benchmark) |
| 8 | `README.md` | Repo overview |

## Summary of What to Do

1. **Kill zombie GPU processes** (pre-flight step, see below)
2. **Evaluate the pretrained v2 baseline** on both standard and dispatch-matched surfaces
3. **Calibrate RTG** on the smoke tier (find optimal RTG for this model)
4. **Run GRPO Phase 1 fine-tuning** with multi-region + multi-battery config
5. **Re-evaluate the GRPO-tuned model** on both surfaces
6. **Compare results** against the baseline and the legacy Phase 1 reference

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

> **Architecture — verified from the checkpoint weights (2026-07-17).**
> The v2 model is **8 blocks × h_dim 768, n_heads 12, n_kv_heads 6 (GQA),
> qk_norm=true, tie_weights=true, ctx=210, return_scale=2.0**, and it uses
> **learned timestep embeddings, NOT RoPE** (`rope_enabled=false`).
> The HF filename is **`aemo_dt_fcas_model.pt`** (not `aemo_dt_model.pt`).
> Do not hand-write `model_kwargs`; load them from the canonical config so they
> always match the weights. You can verify any checkpoint's true architecture
> from its embedded `config` / `model_state_dict` (see "Verifying a checkpoint's
> architecture" below).

```python
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

# Repo helpers keep the HF repo id, filename, and model-config path in one place.
import sys; sys.path.insert(0, "src")
from aemo_dt_hf import (
    MODERN_V2_HF_REPO, MODERN_V2_HF_FILENAME,
    modern_v2_model_config_path, load_model_kwargs, build_surface_manifest,
    write_placeholder_loss_csv,
)

checkpoint = hf_hub_download(MODERN_V2_HF_REPO, MODERN_V2_HF_FILENAME)  # aemo_dt_fcas_model.pt

# Verified 8x768 / 12 heads / 6 kv-heads / qk_norm / tie_weights / ctx=210 / rope=false
model_kwargs = load_model_kwargs(modern_v2_model_config_path())

loss_csv = write_placeholder_loss_csv("/tmp/dummy_loss.csv")
manifest = build_surface_manifest(
    model_kwargs=model_kwargs, save_path=checkpoint, loss_csv_path=loss_csv,
    hf_repo=MODERN_V2_HF_REPO, hf_filename=MODERN_V2_HF_FILENAME,
)
Path("/tmp/modern_manifest.json").write_text(json.dumps(manifest, indent=2))
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
  python3 scripts/autoresearch_evaluator.py \
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
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/modern_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/modern_standard --device auto

python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path /tmp/modern_manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/modern_comprehensive --device auto
```

---

## ⚠️ Pre-flight: Clean GPU memory

GRPO OOMs are almost always caused by orphaned GPU processes from previous
runs, not by the model itself. **Always** run this cleanup **before** any
GRPO training or evaluation:

```bash
# 1. Check what's using GPU memory
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# 2. Kill all Python processes on GPU (safe: grpo scripts will restart fresh)
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null

# 3. Verify GPU is clean
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # expect < 500 MiB
```

A typical zombie holds 14+ GB. Cleaning before each run prevents spurious OOM.

---

## Step 2: GRPO Fine-Tuning (Both Models)

The modern model is **8×768 (12 heads, 6 KV heads), ctx=210** — fine-tune with
the **same config** as pretraining. No architecture changes needed.

### Memory budget (8×768, ctx=210, 22 GB GPU)

| group_size | Rollout batch | Total VRAM | Feasible? |
|:----------:|:-------------:|:----------:|:---------:|
| 4 | ~6 GB | ~13 GB | ✅ Safe |
| 8 | ~12 GB | ~19 GB | ✅ After zombie cleanup |

### Both models: recommended command (group_size=4, safe)

```bash
# Legacy model: downloads from mrvictoru/energydecision-dt automatically
# Modern model: see symlink note below

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
  --mixed-precision \
  --output-dir models/aemo/dt/grpo_modern_multibat
```

> **Recipe note.** `--dt-gamma 1.0` (undiscounted RTG) is the recommended
> starting point — it matches the proven legacy PR#30 recipe and avoids the RTG
> instability that collapsed earlier runs. `--adaptive-rtg` is intentionally
> **omitted** for the first comparison (it is counterproductive when realized
> returns are negative). Reintroduce discounting (`0.99`–`0.995`, now safe via
> the clamp) and multi-region/multi-battery randomness only one variable at a
> time, after the `gamma=1.0` single-region baseline is established.

### If GPU is clean and you want better advantage estimates (group_size=8)

Add `--group-size 8` to the command above. This doubles the rollout batch
(~19 GB total) but still fits on a clean 22 GB card. Skip if you see OOM.

### Modern Model: Loading from `mrvictoru/energydecision-dt-v2`

`scripts/run_grpo_multi_region.py` already defaults to the modern v2 model —
`--hf-repo mrvictoru/energydecision-dt-v2`, `--hf-filename aemo_dt_fcas_model.pt`,
and the correct model config (`aemo_dt_hf.modern_v2_model_config_path()`). No
symlink or manual download is needed; just run the command above.

To pre-cache the checkpoint before training (optional):

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from aemo_dt_hf import MODERN_V2_HF_REPO, MODERN_V2_HF_FILENAME
from huggingface_hub import hf_hub_download
print(hf_hub_download(MODERN_V2_HF_REPO, MODERN_V2_HF_FILENAME))  # aemo_dt_fcas_model.pt
"
```

The GRPO code auto-detects the modern architecture (`_is_legacy_dt()` returns
``False``) and uses the correct forward path.

---

## Step 3: Evaluate Both GRPO-Tuned Models

After GRPO, evaluate on both surfaces:

```bash
# Legacy GRPO model
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_legacy_multibat/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/legacy_grpo_standard --device auto

python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_legacy_multibat/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_comprehensive.json \
  --output-dir eval_output/legacy_grpo_comprehensive --device auto

# Modern GRPO model
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/grpo_modern_multibat/grpo_surface_manifest.json \
  --evaluation-config configs/eval_tier_standard.json \
  --output-dir eval_output/modern_grpo_standard --device auto

python3 scripts/autoresearch_evaluator.py \
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

## GPU Memory Notes (8×768 modern model)

**The #1 cause of OOM is orphaned GPU processes, not model size.** Always
clean zombies before starting (see Pre-flight section above).

| group_size | Rollout batch | Total VRAM | On 22 GB? |
|:----------:|:-------------:|:----------:|:---------:|
| 4 | ~6 GB | ~13 GB | ✅ Safe |
| 8 | ~12 GB | ~19 GB | ✅ After zombie cleanup |
| 2 | ~3 GB | ~10 GB | ✅ (but too noisy — skip) |

`--group-size 4` is the recommended default. Upgrade to 8 only after verifying
GPU memory is clean. `--group-size 2` gives unstable advantage estimates and
should not be used.

**No code changes needed** — gradient checkpointing, mixed precision, and other
memory optimisations are not required. The 8×768 model fits comfortably with
`group_size=4` on a standard 22 GB card once zombie processes are killed.

---

## Verifying a checkpoint's architecture

Never trust docs (or a `model_kwargs` snippet) over the weights. Every checkpoint
in this repo carries an embedded `config` and its `model_state_dict` shapes reveal
the true architecture. Use this to confirm any `.pt` before loading it:

```python
import torch, re

ckpt = torch.load("path/to/model.pt", map_location="cpu", weights_only=False)
print("embedded config:", ckpt.get("config"))            # authoritative training config
sd = ckpt["model_state_dict"]

blocks = {int(m.group(1)) for k in sd if (m := re.search(r"\.(\d+)\.", k))}
print("n_block:", max(blocks) + 1)                        # number of transformer blocks
print("h_dim:", sd["transformer.0.attn.q_proj.weight"].shape[1])
kv = sd["transformer.0.attn.k_proj.weight"].shape[0]
print("kv width:", kv, "-> GQA active if kv < h_dim")
print("qk_norm:", any("q_norm" in k for k in sd))
print("rope_enabled:", not any("embed_timestep" in k for k in sd))  # timestep emb => RoPE off
print("modern (embed_rtg):", any("embed_rtg" in k for k in sd),
      "| legacy (embed_return):", any("embed_return" in k for k in sd))
```

For `mrvictoru/energydecision-dt-v2` / `aemo_dt_fcas_model.pt` this prints:
`n_block=8, h_dim=768, kv width=384 (GQA, 6 kv-heads), qk_norm=True,
rope_enabled=False, modern=True`, matching
`configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json`.
