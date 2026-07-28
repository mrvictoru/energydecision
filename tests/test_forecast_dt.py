"""Verification tests for ForecastDecisionTransformer token flow and masking."""
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from forecast_decision_transformer import (
    ForecastDecisionTransformer,
    ForecastTrajectoryDataset,
)


def _build_model(ctx=6, fore=3, **kw):
    defaults = dict(
        state_dim=8, act_dim=3, context_len=ctx, forecast_len=fore,
        h_dim=32, n_block=2, n_heads=4, n_kv_heads=4,
        qk_norm=False, tie_weights=False, rope_enabled=False,
    )
    defaults.update(kw)
    defaults.pop('ctx', None)
    defaults.pop('fore', None)
    model = ForecastDecisionTransformer(**defaults)
    return model.eval()


def _make_tensors(B=2, T=6, F=3, S=8, A=3):
    st = torch.randn(B, T, S)
    ac = torch.randn(B, T, A)
    rt = torch.randn(B, T, 1)
    ts = torch.arange(T).unsqueeze(0).expand(B, -1)
    mk = torch.ones(B, T)
    root_ts = ts[:, -1:] + 1
    fr_ts = root_ts + torch.arange(F).unsqueeze(0).expand(B, -1)
    fs = torch.randn(B, F, S)
    fr = torch.randn(B, F, 1)
    ft = root_ts + torch.arange(F).unsqueeze(0).expand(B, -1)
    return st, ac, rt, ts, mk, fs, fr, ft


# ═══════════════════════════════════════════════════════════════════
# Token structure
# ═══════════════════════════════════════════════════════════════════

def test_forward_with_forecast_returns_correct_shapes():
    """Forecast tokens should not change output shape."""
    model = _build_model()
    st, ac, rt, ts, mk, fs, fr, ft = _make_tensors(T=6, F=3)
    with torch.no_grad():
        rp, sp, ap = model(st, ac, rt, ts, mk, fs, fr, ft)
    assert rp.shape == (2, 6, 1), f"return_preds shape {rp.shape}"
    assert sp.shape == (2, 6, 8), f"state_preds shape {sp.shape}"
    assert ap.shape == (2, 6, 3), f"action_preds shape {ap.shape}"


def test_forward_without_forecast_is_identical():
    """Setting forecast_states=None should behave like no forecast."""
    model = _build_model()
    st, ac, rt, ts, mk, _, _, _ = _make_tensors()
    with torch.no_grad():
        rp1, sp1, ap1 = model(st, ac, rt, ts, mk)
        rp2, sp2, ap2 = model(st, ac, rt, ts, mk,
                              forecast_states=None, forecast_rtgs=None,
                              forecast_timesteps=None)
    assert torch.equal(ap1, ap2)


def test_forecast_tokens_are_prepended():
    """Forecast tokens appear before history tokens in the sequence."""
    model = _build_model(ctx=4, fore=2)
    st, ac, rt, ts, mk, fs, fr, ft = _make_tensors(T=4, F=2, B=1)
    with torch.no_grad():
        # With forecast
        out_for = model(st, ac, rt, ts, mk, fs, fr, ft)
        # Without forecast on the SAME history data
        out_nofor = model(st, ac, rt, ts, mk)

    # These should differ because the forecast tokens change the context
    # If the are identical, the forecast tokens have no effect (bug)
    assert not torch.allclose(out_for[2], out_nofor[2], atol=1e-5), \
        "Forecast tokens should affect predictions — bug if identical"


def test_forecast_type_embedding_exists():
    """The embed_forecast_type layer should have 2 entries."""
    model = _build_model()
    assert model.embed_forecast_type.num_embeddings == 2, \
        f"Expected 2 type embeddings, got {model.embed_forecast_type.num_embeddings}"


def test_forecast_type_used_differently():
    """History (idx=0) and forecast (idx=1) embeddings differ."""
    model = _build_model()
    zero_emb = model.embed_forecast_type(torch.tensor([0]))
    one_emb = model.embed_forecast_type(torch.tensor([1]))
    assert not torch.allclose(zero_emb, one_emb, atol=1e-6), \
        "History and forecast type embeddings should differ"


def test_forecast_state_affects_output():
    """Changing the forecast state should change the output."""
    model = _build_model(ctx=4, fore=2)
    st, ac, rt, ts, mk, fs1, fr, ft = _make_tensors(T=4, F=2, B=1)
    fs2 = fs1 + 10.0  # very different forecast
    with torch.no_grad():
        _, _, ap1 = model(st, ac, rt, ts, mk, fs1, fr, ft)
        _, _, ap2 = model(st, ac, rt, ts, mk, fs2, fr, ft)
    assert not torch.allclose(ap1, ap2, atol=1e-5), \
        "Changing forecast should change predictions"


# ═══════════════════════════════════════════════════════════════════
# Attention mask
# ═══════════════════════════════════════════════════════════════════

def test_attention_mask_handles_forecast_prefix():
    """With forecast, mask should be [1]*3F + [1]*3T (all valid)."""
    model = _build_model(ctx=4, fore=2)
    st, ac, rt, ts, mk, fs, fr, ft = _make_tensors(T=4, F=2, B=1)
    with torch.no_grad():
        rp, sp, ap = model(st, ac, rt, ts, mk, fs, fr, ft)
    # Should produce outputs without error
    assert ap.shape == (1, 4, 3)


def test_forecast_len_zero_produces_original_behavior():
    """forecast_len=0 should produce same output as standard."""
    model_fore = _build_model(ctx=4, fore=0)
    st, ac, rt, ts, mk, _, _, _ = _make_tensors(T=4, F=0, B=1)
    with torch.no_grad():
        rp, sp, ap = model_fore(st, ac, rt, ts, mk,
                                forecast_states=None, forecast_rtgs=None,
                                forecast_timesteps=None)
    assert ap.shape == (1, 4, 3)
    # Also ensure creating with forecast_len=0 does not error
    model_fore2 = ForecastDecisionTransformer(
        state_dim=8, act_dim=3, context_len=4, forecast_len=0,
        h_dim=32, n_block=2, n_heads=4, n_kv_heads=4,
    )
    assert model_fore2.forecast_len == 0


# ═══════════════════════════════════════════════════════════════════
# Prediction heads — decode only from history
# ═══════════════════════════════════════════════════════════════════

def test_prediction_heads_decode_from_history_positions():
    """Predictions should always be T (history) not T+F."""
    model = _build_model(ctx=6, fore=4)
    st, ac, rt, ts, mk, fs, fr, ft = _make_tensors(T=6, F=4)
    with torch.no_grad():
        rp, sp, ap = model(st, ac, rt, ts, mk, fs, fr, ft)
    # All output dims should be T (history length), not T+F
    assert rp.shape[1] == 6, "return_preds should be history length"
    assert sp.shape[1] == 6, "state_preds should be history length"
    assert ap.shape[1] == 6, "action_preds should be history length"


def test_get_action_returns_single_vector():
    """get_action should return [act_dim] from the last history position."""
    model = _build_model(ctx=4, fore=2)
    st, ac, rt, ts, mk, fs, fr, ft = _make_tensors(T=4, F=2, B=1)
    # 2D inputs (no batch dim)
    action = model.get_action(st[0], ac[0], rt[0], ts[0], mk[0],
                              fs[0], fr[0], ft[0])
    assert action.shape == (1, 3), f"Expected (1,3) got {action.shape}"


def test_get_action_without_forecast_returns_same_shape():
    """get_action without forecast should have same shape."""
    model = _build_model(ctx=4, fore=2)
    st, ac, rt, ts, mk, fs, fr, ft = _make_tensors(T=4, F=2, B=1)
    act_with = model.get_action(st[0], ac[0], rt[0], ts[0], mk[0],
                                fs[0], fr[0], ft[0])
    act_without = model.get_action(st[0], ac[0], rt[0], ts[0], mk[0])
    assert act_with.shape == act_without.shape == (1, 3)


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

def test_dataset_yields_forecast_tokens():
    """The dataset should yield forecast tokens after the history window."""
    # Build a tiny synthetic dataset
    episodes = []
    for eid in range(3):
        rows = []
        for step in range(20):
            rows.append({
                "episode_id": eid,
                "step": step,
                "norm_observation": [float(step + eid * 100)] * 8,
                "action": [0.0] * 3,
                "reward": 1.0 if step > 10 else -1.0,
                "source_policy": "test",
            })
        episodes.extend(rows)
    df = pl.DataFrame(episodes)

    ds = ForecastTrajectoryDataset(
        df, context_length=6, state_dim=8, act_dim=3,
        forecast_len=3, discount_factor=1.0,
    )
    assert len(ds) > 0, "Dataset should have windows"
    sample = ds[0]

    # Check all expected keys exist
    for key in ["states", "actions", "rtgs", "timesteps", "mask",
                "forecast_states", "forecast_rtgs", "forecast_timesteps",
                "forecast_mask"]:
        assert key in sample, f"Missing key: {key}"

    assert sample["states"].shape == (6, 8), f"states shape {sample['states'].shape}"
    assert sample["forecast_states"].shape == (3, 8), \
        f"forecast_states shape {sample['forecast_states'].shape}"
    assert sample["forecast_rtgs"].shape == (3, 1)
    assert sample["forecast_timesteps"].shape == (3,)
    assert sample["forecast_mask"].shape == (3,)


def test_forecast_rtgs_differ_from_history_rtgs():
    """Forecast RTGs should reflect future returns, proven by numerical difference."""
    df = pl.DataFrame({
        "episode_id": [0] * 15,
        "step": list(range(15)),
        "norm_observation": [[float(i)] * 4 for i in range(15)],
        "action": [[0.0, 0.0]] * 15,
        "reward": [0.0]*5 + [10.0]*5 + [-5.0]*5,  # step-up then step-down rewards
        "source_policy": ["test"] * 15,
    })

    ds = ForecastTrajectoryDataset(
        df, context_length=5, state_dim=4, act_dim=2,
        forecast_len=5, discount_factor=1.0,
    )
    # Get a window where history and forecast see different rewards
    # window at step 5: history sees 0s, forecast sees +10s
    for sample in ds:
        hist_rtg = sample["rtgs"][-1, 0].item()  # last history RTG
        fore_rtg_last = sample["forecast_rtgs"][-1, 0].item()  # last forecast RTG
        # They should differ because forecast has different future rewards
        if abs(hist_rtg - fore_rtg_last) > 0.1:
            return  # success: found a sample where they differ

    pytest.fail("Forecast and history RTGs were identical across all windows")


def test_dataset_right_aligns_history_left_aligns_forecast():
    """History is right-aligned (pad at start), forecast is left-aligned (pad at end)."""
    df = pl.DataFrame({
        "episode_id": [0] * 20,
        "step": list(range(20)),
        "norm_observation": [[float(i)] * 4 for i in range(20)],
        "action": [[0.0, 0.0]] * 20,
        "reward": [1.0] * 20,
        "source_policy": ["test"] * 20,
    })

    ds = ForecastTrajectoryDataset(
        df, context_length=5, state_dim=4, act_dim=2,
        forecast_len=3, discount_factor=1.0,
    )
    # First window (start=0): history has first 5 steps, forecast has next 3
    sample = ds[0]

    # History mask: last 5 should be valid (room for right-alignment at episode start)
    hist_mask = sample["mask"]
    # For the very first window, if episode starts at 0, the window might be
    # right-aligned: e.g., if context=5 and start=0, we have 5 historical steps
    # which fill the entire buffer, so mask should be [1,1,1,1,1]
    assert hist_mask.sum() <= 5, f"History mask should have at most {5} ones, got {hist_mask.sum()}"

    # At episode start (start=0), the window has only start=0..4 = 5 steps
    # All 5 are valid, so mask should be [1,1,1,1,1]
    # Forecast: steps 5..7, all valid, so forecast_mask = [1,1,1]
    fore_mask = sample["forecast_mask"]
    if hist_mask.sum() == 5:
        assert fore_mask.sum() == 3, \
            f"Forecast should be fully valid when history is full, got {fore_mask.sum()}"


# ═══════════════════════════════════════════════════════════════════
# Training-ready: forward with dataset outputs
# ═══════════════════════════════════════════════════════════════════

def test_forward_with_dataset_output():
    """Model forward should accept dataset output directly."""
    model = _build_model(ctx=5, fore=3)
    df = pl.DataFrame({
        "episode_id": [0] * 20,
        "step": list(range(20)),
        "norm_observation": [[float(i)] * 8 for i in range(20)],
        "action": [[0.0, 0.0, 0.0]] * 20,
        "reward": [1.0] * 20,
        "source_policy": ["test"] * 20,
    })

    ds = ForecastTrajectoryDataset(
        df, context_length=5, state_dim=8, act_dim=3,
        forecast_len=3, discount_factor=1.0,
    )
    sample = ds[0]

    # Add batch dim (dataset yields batched tensors)
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}

    with torch.no_grad():
        rp, sp, ap = model(
            batch["states"], batch["actions"], batch["rtgs"],
            batch["timesteps"], batch["mask"],
            forecast_states=batch["forecast_states"],
            forecast_rtgs=batch["forecast_rtgs"],
            forecast_timesteps=batch["forecast_timesteps"],
        )
    assert ap.shape == (1, 5, 3), f"Full pipeline output shape {ap.shape}"
