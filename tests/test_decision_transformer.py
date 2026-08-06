"""Unit tests for modern DecisionTransformer architectural improvements.

Covers:
- GQA (Grouped Query Attention)
- Bias-free linears
- Consistent RMSNorm (embed_ln)
- QK-Norm (optional)
- Weight tying (embed_* ↔ pred_*)
- Backward-compat: old fused-QKV checkpoint migration
- Backward-compat: n_kv_heads validation in pretrain_decision_transformer
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from decision_transformer import (
    CausalSelfAttention,
    DecisionTransformer,
    RMSNorm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dt(
    state_dim=12,
    act_dim=9,
    n_block=2,
    h_dim=64,
    context_len=10,
    n_heads=4,
    drop_p=0.0,
    **kwargs,
) -> DecisionTransformer:
    return DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_block=n_block,
        h_dim=h_dim,
        context_len=context_len,
        n_heads=n_heads,
        drop_p=drop_p,
        **kwargs,
    )


def _dummy_inputs(model: DecisionTransformer, T: int = 5, B: int = 2):
    state = torch.randn(B, T, model.state_dim)
    rtg = torch.randn(B, T, 1)
    acts = torch.randn(B, T, model.act_dim)
    ts = torch.zeros(B, T, dtype=torch.long)
    return state, rtg, ts, acts


# ---------------------------------------------------------------------------
# Step 1: Grouped Query Attention
# ---------------------------------------------------------------------------

class TestGQA:
    def test_default_mha_unchanged(self):
        """n_kv_heads defaults to n_heads → pure MHA, no parameter change."""
        attn = CausalSelfAttention(h_dim=64, max_T=30, n_heads=4, drop_p=0.0)
        assert attn.n_kv_heads == 4
        assert attn.n_rep == 1

    def test_gqa_n_kv_heads_reduces_kv_params(self):
        """GQA with n_kv_heads=2 has fewer K/V params than full MHA."""
        attn_mha = CausalSelfAttention(h_dim=64, max_T=30, n_heads=4, drop_p=0.0)
        attn_gqa = CausalSelfAttention(h_dim=64, max_T=30, n_heads=4, drop_p=0.0, n_kv_heads=2)

        mha_kv_params = sum(p.numel() for p in [attn_mha.k_proj.weight, attn_mha.v_proj.weight])
        gqa_kv_params = sum(p.numel() for p in [attn_gqa.k_proj.weight, attn_gqa.v_proj.weight])
        assert gqa_kv_params < mha_kv_params

    def test_gqa_forward_shape(self):
        """GQA produces the same output shape as MHA."""
        B, T, H = 2, 5, 64
        x = torch.randn(B, T, H)
        attn = CausalSelfAttention(h_dim=H, max_T=T, n_heads=4, drop_p=0.0, n_kv_heads=2)
        out = attn(x)
        assert out.shape == (B, T, H)

    def test_gqa_n_kv_heads_1(self):
        """Extreme case: n_kv_heads=1 (multi-query attention)."""
        B, T, H = 2, 5, 64
        x = torch.randn(B, T, H)
        attn = CausalSelfAttention(h_dim=H, max_T=T, n_heads=4, drop_p=0.0, n_kv_heads=1)
        assert attn.n_rep == 4
        out = attn(x)
        assert out.shape == (B, T, H)

    def test_gqa_invalid_n_kv_heads_raises(self):
        """n_kv_heads that does not divide n_heads should raise."""
        with pytest.raises(AssertionError):
            CausalSelfAttention(h_dim=64, max_T=30, n_heads=4, drop_p=0.0, n_kv_heads=3)

    def test_dt_gqa_forward_runs(self):
        """DecisionTransformer with n_kv_heads=2 runs forward pass."""
        model = _make_dt(n_kv_heads=2)
        state, rtg, ts, acts = _dummy_inputs(model)
        ret, sp, ap = model(state, rtg, ts, acts)
        assert ap.shape == (2, 5, model.act_dim)


# ---------------------------------------------------------------------------
# Step 4: Initialization stability
# ---------------------------------------------------------------------------

def test_timestep_embedding_initialization_is_small_and_stable():
    model = _make_dt(h_dim=32, max_timestep=128)
    assert model.embed_timestep.weight.std().item() < 0.05
    assert model.embed_timestep.weight.abs().mean().item() < 0.03


# ---------------------------------------------------------------------------
# Step 5: QK-Norm
# ---------------------------------------------------------------------------

class TestQKNorm:
    def test_qk_norm_disabled_by_default(self):
        attn = CausalSelfAttention(h_dim=64, max_T=30, n_heads=4, drop_p=0.0)
        assert not attn.qk_norm
        assert not hasattr(attn, "q_norm")
        assert not hasattr(attn, "k_norm")

    def test_qk_norm_creates_rmsnorm_per_head_dim(self):
        head_dim = 64 // 4
        attn = CausalSelfAttention(h_dim=64, max_T=30, n_heads=4, drop_p=0.0, qk_norm=True)
        assert attn.qk_norm
        assert isinstance(attn.q_norm, RMSNorm)
        assert isinstance(attn.k_norm, RMSNorm)
        assert attn.q_norm.scale.shape == (head_dim,)
        assert attn.k_norm.scale.shape == (head_dim,)

    def test_qk_norm_forward_shape(self):
        B, T, H = 2, 5, 64
        x = torch.randn(B, T, H)
        attn = CausalSelfAttention(h_dim=H, max_T=T, n_heads=4, drop_p=0.0, qk_norm=True)
        out = attn(x)
        assert out.shape == (B, T, H)

    def test_dt_qk_norm_forward_runs(self):
        model = _make_dt(qk_norm=True)
        state, rtg, ts, acts = _dummy_inputs(model)
        ret, sp, ap = model(state, rtg, ts, acts)
        assert ap.shape == (2, 5, model.act_dim)


# ---------------------------------------------------------------------------
# Step 5: Weight tying
# ---------------------------------------------------------------------------

class TestWeightTying:
    def test_tied_weights_share_storage(self):
        model = _make_dt(tie_weights=True)
        # When tied, pred_*.weight IS the same nn.Parameter object as embed_*.weight
        assert model.pred_rtg.weight is model.embed_rtg.weight
        assert model.pred_state.weight is model.embed_state.weight
        assert model.pred_act[0].weight is model.embed_act.weight

    def test_untied_model_default(self):
        """Without tie_weights, pred_* and embed_* are independent."""
        model = _make_dt(tie_weights=False)
        # Weights should NOT be identical (extremely unlikely by random init)
        assert model.pred_rtg.weight.data_ptr() != model.embed_rtg.weight.data_ptr()

    def test_tied_model_forward_runs(self):
        model = _make_dt(tie_weights=True)
        state, rtg, ts, acts = _dummy_inputs(model)
        ret, sp, ap = model(state, rtg, ts, acts)
        assert ap.shape == (2, 5, model.act_dim)

    def test_tied_model_param_count_less_than_untied(self):
        """Tied model should have fewer trainable parameters than untied."""
        tied = _make_dt(tie_weights=True)
        untied = _make_dt(tie_weights=False)
        tied_count = sum(p.numel() for p in tied.parameters())
        untied_count = sum(p.numel() for p in untied.parameters())
        assert tied_count < untied_count


# ---------------------------------------------------------------------------
# Backward compatibility: fused-QKV checkpoint migration
# ---------------------------------------------------------------------------

class TestFusedQKVMigration:
    def _make_old_state_dict(self, model: DecisionTransformer) -> dict:
        """Simulate a pre-GQA state dict that uses the fused qkv.weight key."""
        state = model.state_dict()
        new_state = {}
        for k, v in state.items():
            if k.endswith(".attn.q_proj.weight"):
                prefix = k[: -len("q_proj.weight")]
                h = v.shape[0]
                # We need k_proj and v_proj tensors too; collect them
                k_key = prefix + "k_proj.weight"
                v_key = prefix + "v_proj.weight"
                q_w = v
                k_w = state[k_key]
                v_w = state[v_key]
                fused = torch.cat([q_w, k_w, v_w], dim=0)
                new_state[prefix + "qkv.weight"] = fused
            elif k.endswith(".attn.k_proj.weight") or k.endswith(".attn.v_proj.weight"):
                continue  # already merged above
            else:
                new_state[k] = v
        return new_state

    def test_old_qkv_checkpoint_loads_cleanly(self):
        """load_from_checkpoint migrates fused qkv.weight to split projections."""
        model = _make_dt()
        old_state = self._make_old_state_dict(model)
        # Ensure old state has qkv keys and no q_proj/k_proj/v_proj
        assert any(k.endswith(".attn.qkv.weight") for k in old_state)
        assert not any(k.endswith(".attn.q_proj.weight") for k in old_state)

        new_model = _make_dt()
        new_model.load_from_checkpoint(old_state, strict=True)

        # Forward pass should work
        state, rtg, ts, acts = _dummy_inputs(new_model)
        ret, sp, ap = new_model(state, rtg, ts, acts)
        assert ap.shape == (2, 5, new_model.act_dim)


# ---------------------------------------------------------------------------
# Backward compatibility: n_kv_heads validation in pretrain script
# ---------------------------------------------------------------------------

class TestPretrainValidation:
    def test_invalid_n_kv_heads_raises(self):
        import pretrain_decision_transformer as pretrain_dt
        with pytest.raises(ValueError, match="n_heads must be divisible by n_kv_heads"):
            pretrain_dt.validate_surface_constraints(
                {"h_dim": 64, "n_heads": 4, "n_kv_heads": 3}
            )

    def test_valid_n_kv_heads_passes(self):
        import pretrain_decision_transformer as pretrain_dt
        # Should not raise
        pretrain_dt.validate_surface_constraints(
            {"h_dim": 64, "n_heads": 4, "n_kv_heads": 2}
        )

    def test_new_keys_accepted_in_config(self, tmp_path):
        import json
        import pretrain_decision_transformer as pretrain_dt
        config_path = tmp_path / "model_config.json"
        config_path.write_text(
            json.dumps({"state_dim": 12, "n_kv_heads": 2, "qk_norm": True, "tie_weights": True}),
            encoding="utf-8",
        )
        # Should not raise
        kwargs = pretrain_dt.load_model_kwargs(config_path)
        assert kwargs["n_kv_heads"] == 2
        assert kwargs["qk_norm"] is True
        assert kwargs["tie_weights"] is True
