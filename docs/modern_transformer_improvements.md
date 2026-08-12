# Modern DecisionTransformer Improvements

> **OBSOLETE (2026-08-11):** the modern architecture improvements described
> here are **complete and landed** — the modern v2 (8×768 GQA) is the deployed
> SOTA. Architecture is no longer the lever: the DT-vs-PPO gap is an
> offline-data ceiling (report.md §8.2.1a). Historical design note — superseded
> by `aemo_research_plan.md`.

This document is a research/design note for transformer architecture changes.

It is useful when reviewing model-evolution ideas and implementation tradeoffs, but it is not the source of truth for the currently recommended training workflow. For that, use [aemo/README.md](aemo/README.md), [aemo/workflow.md](aemo/workflow.md), and [development.md](development.md).

This document outlines architectural upgrades to the modern `DecisionTransformer`
class in `src/decision_transformer.py`. The goal is to reduce memory usage and
parameter count while enabling larger models and longer contexts.

## Current Architecture (Modern, lines 292–416)

| Component | Current implementation | Notes |
|-----------|----------------------|-------|
| Attention | `CausalSelfAttention` — MHA, fused QKV, FlashAttention via SDPA | Solid foundation |
| Norm | `RMSNorm` in `ModernBlock`, `nn.LayerNorm` at `embed_ln` | **Inconsistent** |
| RoPE | Custom rotary, decoupled from QKV weight init | Works correctly |
| FFN | `SwiGLU(h_dim → 4*h_dim)` | Standard |
| Biases | `qkv` is bias-free; `proj`, FFN, prediction linears have bias | Mixed |
| Embedding | Separate `embed_rtg`, `embed_state`, `embed_act` + `embed_timestep` | No weight sharing |
| Prediction | Separate `pred_rtg`, `pred_state`, `pred_act` | Decoupled from embeddings |

---

## Proposed Improvements

### 1. Grouped Query Attention (GQA)

**File**: `decision_transformer.py` — `CausalSelfAttention`

**Current**: `n_heads` query heads, `n_heads` key/value heads — full multi-head attention.
```python
self.qkv = nn.Linear(h_dim, 3 * h_dim, bias=False)
```

**Proposed**: Add `n_kv_heads` parameter. Reduce K/V heads (e.g., `n_kv_heads = n_heads // 4`).
```python
self.q_proj = nn.Linear(h_dim, n_heads * head_dim, bias=False)
self.k_proj = nn.Linear(h_dim, n_kv_heads * head_dim, bias=False)
self.v_proj = nn.Linear(h_dim, n_kv_heads * head_dim, bias=False)
```

During forward, repeat K/V to match `n_heads`:
```python
k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
v = v.repeat_interleave(n_heads // n_kv_heads, dim=1)
```

**Benefit**: ~40% fewer K/V parameters per attention layer. Directly enables larger
`h_dim` or more `n_block` within the same GPU memory budget.

**Effort**: ~2h (modify `__init__`, `forward`, update tests).

---

### 2. Bias-Free Linears (Llama-style)

**File**: `decision_transformer.py` — `CausalSelfAttention`, `SwiGLU`, prediction heads

**Current**: `qkv` is bias-free; `proj`, FFN `w1`/`w2`/`w3`, and prediction `Linear` layers
include bias.

**Proposed**: Remove bias from all `nn.Linear` layers in the transformer blocks.
```python
self.proj = nn.Linear(h_dim, h_dim, bias=False)
self.w1 = nn.Linear(dim, hidden_dim, bias=False)
self.w2 = nn.Linear(dim, hidden_dim, bias=False)
self.w3 = nn.Linear(hidden_dim, dim, bias=False)
self.pred_act = nn.Sequential(
    nn.Linear(h_dim, h_dim, bias=False),
    nn.GELU(),
    nn.Linear(h_dim, act_dim, bias=False),
    nn.Tanh(),
)
```

**Rationale**: Llama-style architectures remove biases from all linears — they provide
negligible benefit for deep transformers and waste parameters.

**Effort**: ~30min.

---

### 3. Consistent RMSNorm

**File**: `decision_transformer.py` — `DecisionTransformer.__init__`

**Current**: `self.embed_ln = nn.LayerNorm(h_dim)` (line 335) while all block norms use
`RMSNorm`.

**Proposed**:
```python
self.embed_ln = RMSNorm(h_dim)
```

**Rationale**: Maintains a single normalisation strategy throughout the model.
RMSNorm is computationally cheaper and empirically equivalent or better for
transformer architectures.

**Effort**: ~5min.

---

### 4. QK-Norm (Optional, for Training Stability)

**File**: `decision_transformer.py` — `CausalSelfAttention.__init__` + `forward`

**Proposed**: Add optional RMSNorm after Q and K projections (before RoPE), as used
by PaLM and ViT-22B for stable training at scale.
```python
self.q_norm = RMSNorm(head_dim)
self.k_norm = RMSNorm(head_dim)

# In forward:
q = self.q_norm(q)   # shape [B, n_heads, T, head_dim]
k = self.k_norm(k)   # shape [B, n_kv_heads, T, head_dim]
```

**Benefit**: Prevents attention logit growth at larger scales. Only needed for
very deep or wide models (e.g., `n_block > 12` or `h_dim > 768`). Can be
disabled by default.

**Effort**: ~30min.

---

### 5. Shared Input/Output Embedding (Weight Tying)

**File**: `decision_transformer.py` — `DecisionTransformer.__init__` + `forward`

**Current**: Input embedding layers (`embed_state`, `embed_rtg`, `embed_act`) and
output prediction layers (`pred_state`, `pred_rtg`, `pred_act`) are independent
parameter groups.

**Proposed**: Tie the output `pred_*` weights to the input `embed_*` weights after
initialisation so they share parameters.
```python
self.embed_state = nn.Linear(state_dim, h_dim)
self.pred_state = nn.Linear(h_dim, state_dim, bias=False)
self.pred_state.weight = self.embed_state.weight  # share
```

For `act_dim` and `rtg` (1-dim), the same pattern applies. Note the output
layers must be `bias=False` for weight tying to be valid (the input projection
already has bias).

**Parameter savings**:

| Shared pair | Formula | Parameters saved (8×384) |
|-------------|---------|:------------------------:|
| `embed_state` ↔ `pred_state` | `h_dim * state_dim` | 384 × 18 = 6,912 |
| `embed_act` ↔ `pred_act[0]` | `h_dim * act_dim` | 384 × 9 = 3,456 |
| `embed_rtg` ↔ `pred_rtg` | `h_dim * 1` | 384 × 1 = 384 |
| **Total** | | **~10,752 (~2% of total)** |

The actual saving is modest for the current architecture but becomes significant
for larger models (e.g., 8×768 saves ~18K params).

**Effort**: ~1h.

---

## Combined Effect on Model Capacity

| Improvement | Memory / param freed | What you can add instead |
|-------------|:--------------------:|--------------------------|
| GQA (n_kv=2 for n_heads=8) | ~15% of attention | +1–2 more `n_block` |
| Bias removal | ~2% | Slightly larger `h_dim` |
| Weight tying | ~2% | +1 more block or longer context |
| QK-Norm (optional) | Negligible overhead | Enables deeper models |
| **Total** | **~19% free** | **8×384 → 10×384 or 8×512** |

---

## Implementation Order

| Step | Change | Files | Time |
|------|--------|-------|:----:|
| 1 | GQA implementation | `CausalSelfAttention` | 2h |
| 2 | Bias-free linears | `CausalSelfAttention.proj`, `SwiGLU`, prediction heads | 30min |
| 3 | Consistent RMSNorm | `DecisionTransformer.embed_ln` | 5min |
| 4 | QK-Norm (optional) | `CausalSelfAttention` | 30min |
| 5 | Weight tying | `DecisionTransformer.init/forward` | 1h |
| 6 | Unit tests + integration test | `tests/` | 1h |
| | **Total** | | **~5h** |

---

## To Use the Updated Architecture

1. Apply the changes above to `src/decision_transformer.py`.
2. Verify tests pass: `python3 -m pytest tests/ -v`.
3. Train the modern DT from scratch on MoLab (RTX 6000 Pro):
   ```bash
   python3 scripts/pretrain_aemo_decision_transformer.py \
     --dataset-path data/aemo_dt_fcas_v2/aemo_fcas_dataset.parquet \
     --model-config configs/aemo_decision_transformer_model_kwargs_full_fcas.json \
     --epochs 2 --batch-size 64 --lr 3e-5
   ```
4. GRPO post-train the new model (local RTX 2080 Ti).
5. Evaluate with `q4_dispatch_matched.json` config.
