import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import json
    import os
    import time
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import HfApi, hf_hub_download

    print("✅ Imports ready")
    return (
        F, HfApi, Path, hf_hub_download, json, mo, nn, np, os, pl, plt, time, torch,
    )


@app.cell
def _(Path, torch):
    CHECKPOINT_DIR = Path("/workspace/dt_checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = CHECKPOINT_DIR / "latest_checkpoint.pt"
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"

    ckpt_info = "No local checkpoint found"
    if CHECKPOINT_PATH.exists():
        try:
            _checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
            _epoch = _checkpoint.get("epoch", "?")
            _best = _checkpoint.get("best_val_loss", float("inf"))
            ckpt_info = f"Resume ready: epoch={_epoch}, best_val={_best:.6f}"
        except Exception:
            ckpt_info = "Checkpoint exists but could not be read"
    return BEST_MODEL_PATH, CHECKPOINT_DIR, CHECKPOINT_PATH, ckpt_info


@app.cell
def _(json, mo, os):
    use_pilot = mo.ui.checkbox(label="Pilot mode (12 episodes, fast)", value=True)
    fresh_start = mo.ui.checkbox(label="Fresh start (delete checkpoint)", value=False)
    use_json_config = mo.ui.checkbox(label="Use JSON config", value=False)
    include_base_dataset = mo.ui.checkbox(label="Base dataset (aemo_fcas_dataset)", value=True)
    include_grpo_dataset = mo.ui.checkbox(label="GRPO dataset (aemo_fcas_grpo_dataset)", value=True)
    include_sdp_dataset = mo.ui.checkbox(label="SDP dataset (aemo_sdp_trajectories)", value=True)

    n_block = mo.ui.number(value=8, label="Blocks", full_width=True)
    h_dim = mo.ui.number(value=768, label="Hidden dim", full_width=True)
    n_heads = mo.ui.number(value=12, label="Heads", full_width=True)
    context_len = mo.ui.number(value=210, label="Context len", full_width=True)
    forecast_len = mo.ui.number(value=48, label="Forecast len", full_width=True)
    drop_p = mo.ui.number(value=0.15, label="Dropout", full_width=True)
    n_kv_heads = mo.ui.number(value=6, label="KV heads", full_width=True)
    qk_norm = mo.ui.checkbox(label="QK-Norm", value=True)
    tie_weights = mo.ui.checkbox(label="Tie weights", value=True)

    batch_size = mo.ui.number(value=64, label="Batch size", full_width=True)
    epochs_per_session = mo.ui.number(value=3, label="Epochs/session", full_width=True)
    lr = mo.ui.number(value=3e-5, label="Learning rate", full_width=True)

    action_loss_weight = mo.ui.number(value=0.999, label="Action loss weight", full_width=True)
    state_loss_weight = mo.ui.number(value=0.002, label="State loss weight", full_width=True)
    return_loss_weight = mo.ui.number(value=0.0001, label="Return loss weight", full_width=True)

    _DEFAULT_JSON = json.dumps({
        "state_dim": 18, "act_dim": 9,
        "n_block": 8, "h_dim": 768, "n_heads": 12,
        "context_len": 210, "forecast_len": 48, "drop_p": 0.15,
        "n_kv_heads": 6, "qk_norm": True, "tie_weights": True,
        "rope_enabled": True,
        "batch_size": 64, "epochs_per_session": 3, "lr": 3e-5,
        "action_loss_weight": 0.999, "state_loss_weight": 0.002,
        "return_loss_weight": 0.0001, "discount_factor": 0.95,
        "return_scale": 2.0, "weight_decay": 1e-4, "grad_clip_norm": 1.0,
    }, indent=2)
    json_config = mo.ui.text_area(value=_DEFAULT_JSON, label="JSON config", full_width=True)

    train_btn = mo.ui.run_button(label="Start Training", kind="success")
    upload_btn = mo.ui.run_button(label="Upload to HuggingFace", kind="info")
    hf_repo_id = mo.ui.text(value="mrvictoru/energydecision-dt-v2", label="HF repo", full_width=True)
    hf_token_input = mo.ui.text(value=os.environ.get("HF_TOKEN", ""), label="HF token", full_width=True)

    manual_controls = mo.vstack([
        mo.md("### Architecture"),
        mo.hstack([n_block, h_dim, n_heads], justify="start", gap=1),
        mo.hstack([context_len, forecast_len, drop_p], justify="start", gap=1),
        mo.hstack([n_kv_heads, qk_norm, tie_weights], justify="start", gap=2),
        mo.md("### Optimization"),
        mo.hstack([batch_size, epochs_per_session, lr], justify="start", gap=1),
        mo.hstack([action_loss_weight, state_loss_weight, return_loss_weight], justify="start", gap=1),
    ], gap=0.5)

    manual_controls
    return (
        action_loss_weight, batch_size, context_len, drop_p, epochs_per_session,
        forecast_len, fresh_start, h_dim, hf_repo_id, hf_token_input,
        include_base_dataset, include_grpo_dataset, include_sdp_dataset,
        json_config, lr, manual_controls, n_block, n_heads, n_kv_heads,
        qk_norm, resume_from_hf := mo.ui.checkbox(
            label="Resume from HF checkpoint", value=False
        ),
        hf_checkpoint_path := mo.ui.text(value="", label="HF checkpoint filename", full_width=True),
        return_loss_weight, state_loss_weight, tie_weights, train_btn,
        upload_btn, use_json_config, use_pilot,
    )


@app.cell
def _(
    BEST_MODEL_PATH, ckpt_info, forecast_len, fresh_start, hf_checkpoint_path,
    hf_repo_id, hf_token_input, include_base_dataset, include_grpo_dataset,
    include_sdp_dataset, json_config, manual_controls, mo, resume_from_hf,
    train_btn, upload_btn, use_json_config, use_pilot,
):
    _dataset_section = [
        mo.md("### Dataset"),
        mo.hstack([use_pilot, fresh_start, use_json_config], justify="start", gap=2),
    ]
    if not use_pilot.value:
        _dataset_section.extend([
            mo.hstack([include_base_dataset, include_grpo_dataset, include_sdp_dataset], justify="start", gap=2),
            mo.md("Enabled datasets are concatenated. SDP trajectories = optimal energy arbitrage."),
        ])

    _config_section = mo.vstack([
        mo.md("### Configuration"),
        json_config if use_json_config.value else manual_controls,
    ], gap=0.5)

    mo.vstack([
        mo.md(f"""
## Forecast Decision Transformer — AEMO Training
*Local checkpoint*: {ckpt_info}
*Forecast DT with RoPE, forecast_len={forecast_len.value}*
        """),
        mo.vstack(_dataset_section, gap=0.5),
        _config_section,
        mo.hstack([resume_from_hf, hf_checkpoint_path], justify="start", gap=2),
        mo.hstack([train_btn, upload_btn], justify="start", gap=2),
        hf_repo_id, hf_token_input,
    ])
    return


@app.cell
def _(
    Path, hf_hub_download, include_base_dataset, include_grpo_dataset,
    include_sdp_dataset, pl, use_pilot,
):
    REPO_ID = "mrvictoru/AEMO_simulated_trade"
    CACHE_DIR = Path("/workspace")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    selected_dfs = []

    if use_pilot.value:
        filename = "aemo_fcas_pilot.parquet"
        local_path = CACHE_DIR / filename
        if not local_path.exists():
            print(f"⬇️ Downloading {filename}...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(CACHE_DIR),
                           local_dir_use_symlinks=False, repo_type="dataset")
        df = pl.read_parquet(local_path)
        print(f"Loaded pilot: {len(df):,} rows")
    else:
        filenames = []
        if include_base_dataset.value:
            filenames.append("aemo_fcas_dataset.parquet")
        if include_grpo_dataset.value:
            filenames.append("aemo_fcas_grpo_dataset.parquet")
        if include_sdp_dataset.value:
            filenames.append("aemo_sdp_trajectories.parquet")
        if not filenames:
            raise ValueError("Select at least one dataset.")
        for fn in filenames:
            local_path = CACHE_DIR / fn
            if not local_path.exists():
                print(f"⬇️ Downloading {fn}...")
                hf_hub_download(repo_id=REPO_ID, filename=fn, local_dir=str(CACHE_DIR),
                               local_dir_use_symlinks=False, repo_type="dataset")
            else:
                print(f"📦 Cached: {fn}")
            selected_dfs.append(pl.read_parquet(local_path))

        if len(selected_dfs) > 1:
            _target = selected_dfs[0].schema
            _aligned = [selected_dfs[0]]
            for _other in selected_dfs[1:]:
                _cast = []
                for _cn, _ct in _other.schema.items():
                    if _cn in _target and _target[_cn] != _ct:
                        if isinstance(_ct, pl.List) and isinstance(_target[_cn], pl.List):
                            _cast.append(pl.col(_cn).cast(_target[_cn]))
                        elif pl.Float32 in (getattr(_ct, "inner", _ct), _ct) and pl.Float64 in (getattr(_target[_cn], "inner", _target[_cn]), _target[_cn]):
                            _cast.append(pl.col(_cn).cast(_target[_cn]))
                _aligned.append(_other.with_columns(*_cast) if _cast else _other)
            df = pl.concat(_aligned, how="vertical")
            print(f"Combined: {len(df):,} rows from {len(filenames)} files")
        else:
            df = selected_dfs[0]

    print(f"✅ Loaded {len(df):,} rows")
    return df,


# ── MODEL DEFINITION (self-contained for MoLab) ──────────────────────

@app.cell
def _(F, nn, torch):
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            norm_x = torch.mean(x * x, dim=-1, keepdim=True)
            x = x * torch.rsqrt(norm_x + self.eps)
            return x * self.scale

    class SwiGLU(nn.Module):
        def __init__(self, dim, hidden_dim, drop_p=0.0):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, dim, bias=False)
            self.dropout = nn.Dropout(drop_p)
        def forward(self, x):
            return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_position=4096, base=10000.0):
            super().__init__()
            if dim % 2 != 0:
                raise ValueError("RoPE requires even dimension")
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            positions = torch.arange(max_position, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", positions, inv_freq)
            self.register_buffer("cos", torch.cos(freqs))
            self.register_buffer("sin", torch.sin(freqs))
        def forward(self, seq_len):
            if seq_len > self.cos.shape[0]:
                raise ValueError(f"seq_len {seq_len} exceeds RoPE cache {self.cos.shape[0]}")
            return self.cos[:seq_len], self.sin[:seq_len]

    def apply_rotary_pos_emb(q, k, cos, sin):
        cos = cos.to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
        sin = sin.to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
        q_even, q_odd = q[..., ::2], q[..., 1::2]
        k_even, k_odd = k[..., ::2], k[..., 1::2]
        q = torch.cat([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1)
        k = torch.cat([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1)
        return q, k

    class CausalSelfAttention(nn.Module):
        def __init__(self, h_dim, max_T, n_heads, drop_p, rope_enabled=False,
                     rope_max_position=4096, rope_base=10000.0, n_kv_heads=None, qk_norm=False):
            super().__init__()
            assert h_dim % n_heads == 0
            self.n_heads = n_heads
            self.head_dim = h_dim // n_heads
            self.drop_p = drop_p
            self.rope_enabled = rope_enabled
            if n_kv_heads is None:
                n_kv_heads = n_heads
            assert n_heads % n_kv_heads == 0
            self.n_kv_heads = n_kv_heads
            self.n_rep = n_heads // n_kv_heads
            if rope_enabled:
                if self.head_dim % 2 != 0:
                    raise ValueError("RoPE requires even head_dim")
                self.rotary = RotaryEmbedding(self.head_dim, max_position=rope_max_position, base=rope_base)
            self.q_proj = nn.Linear(h_dim, n_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(h_dim, n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(h_dim, n_kv_heads * self.head_dim, bias=False)
            self.proj = nn.Linear(h_dim, h_dim, bias=False)
            self.proj_drop = nn.Dropout(drop_p)
            self.qk_norm = qk_norm
            if qk_norm:
                self.q_norm = RMSNorm(self.head_dim)
                self.k_norm = RMSNorm(self.head_dim)
            mask = torch.tril(torch.ones(max_T, max_T)).view(1, 1, max_T, max_T)
            self.register_buffer("mask", mask)
        def forward(self, x, key_padding_mask=None):
            B, T, C = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)
            if self.rope_enabled:
                cos, sin = self.rotary(T)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
            causal = self.mask[:, :, :T, :T].bool()
            if key_padding_mask is not None:
                kp = key_padding_mask.view(B, 1, 1, T).to(dtype=torch.bool)
                combined = causal & kp
                attn_mask = torch.zeros((B, 1, T, T), device=x.device, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(~combined, float("-inf"))
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                    dropout_p=self.drop_p if self.training else 0.0)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                    dropout_p=self.drop_p if self.training else 0.0)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.proj_drop(self.proj(y))
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            return y

    class ModernBlock(nn.Module):
        def __init__(self, h_dim, max_T, n_heads, drop_p, rope_enabled=False,
                     rope_max_position=4096, rope_base=10000.0, n_kv_heads=None, qk_norm=False):
            super().__init__()
            self.norm1 = RMSNorm(h_dim)
            self.attn = CausalSelfAttention(h_dim, max_T, n_heads, drop_p,
                rope_enabled=rope_enabled, rope_max_position=rope_max_position,
                rope_base=rope_base, n_kv_heads=n_kv_heads, qk_norm=qk_norm)
            self.norm2 = RMSNorm(h_dim)
            self.ffn = SwiGLU(h_dim, 4 * h_dim, drop_p)
        def forward(self, x, key_padding_mask=None):
            x = x + self.attn(self.norm1(x), key_padding_mask)
            x = x + self.ffn(self.norm2(x))
            return x

    print("✅ Model building blocks loaded")
    return (
        CausalSelfAttention, ModernBlock, RMSNorm, RotaryEmbedding,
        SwiGLU, apply_rotary_pos_emb,
    )


@app.cell
def _(
    F, ModernBlock, RMSNorm, nn, torch,
):
    class ForecastDecisionTransformer(nn.Module):
        """DT with RoPE enabled by default + forecast token prefix."""

        def __init__(
            self,
            state_dim=18, act_dim=9, n_block=8, h_dim=768,
            context_len=210, n_heads=12, drop_p=0.15,
            max_timestep=100000, forecast_len=48,
            rope_enabled=True, rope_max_position=4096, rope_base=10000.0,
            n_kv_heads=6, qk_norm=True, tie_weights=True,
        ):
            super().__init__()
            self.state_dim = state_dim
            self.act_dim = act_dim
            self.h_dim = h_dim
            self.context_len = context_len
            self.forecast_len = forecast_len
            self.n_heads = n_heads
            self.n_block = n_block
            self.drop_p = drop_p
            self.max_timestep = max_timestep
            self.rope_enabled = rope_enabled

            total_seq_len = 3 * (context_len + forecast_len)
            self.transformer = nn.ModuleList([
                ModernBlock(h_dim, total_seq_len, n_heads, drop_p,
                    rope_enabled=rope_enabled, rope_max_position=rope_max_position,
                    rope_base=rope_base, n_kv_heads=n_kv_heads, qk_norm=qk_norm)
                for _ in range(n_block)
            ])
            self.embed_ln = RMSNorm(h_dim)
            self.embed_timestep = nn.Embedding(max_timestep, h_dim)
            self.embed_rtg = nn.Linear(1, h_dim)
            self.embed_state = nn.Linear(state_dim, h_dim)
            self.embed_act = nn.Linear(act_dim, h_dim)
            self.embed_forecast_type = nn.Embedding(2, h_dim)
            self.ln_f = RMSNorm(h_dim)
            self._tie_weights = tie_weights
            if tie_weights:
                self.pred_rtg = nn.Linear(1, h_dim, bias=False)
                self.pred_state = nn.Linear(state_dim, h_dim, bias=False)
                pred_act_lin = nn.Linear(act_dim, h_dim, bias=False)
                self.pred_rtg.weight = self.embed_rtg.weight
                self.pred_state.weight = self.embed_state.weight
                pred_act_lin.weight = self.embed_act.weight
                self.pred_act = nn.Sequential(pred_act_lin, nn.Tanh())
            else:
                self.pred_rtg = nn.Linear(h_dim, 1, bias=False)
                self.pred_state = nn.Linear(h_dim, state_dim, bias=False)
                self.pred_act = nn.Sequential(nn.Linear(h_dim, act_dim, bias=False), nn.Tanh())
            self.return_scale = 1.0
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None,
                    forecast_states=None, forecast_rtgs=None, forecast_timesteps=None):
            B, T, _ = states.shape
            fore_len = self.forecast_len
            device = states.device

            states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
            returns_to_go = torch.nan_to_num(returns_to_go, nan=0.0, posinf=0.0, neginf=0.0)
            if returns_to_go.dim() == 1:
                returns_to_go = returns_to_go.unsqueeze(0).unsqueeze(-1)
            elif returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(-1)
            elif returns_to_go.dim() == 3 and returns_to_go.shape[-1] != 1:
                returns_to_go = returns_to_go.unsqueeze(-1)
            timesteps = timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)

            # History stream
            time_emb_h = self.embed_timestep(timesteps)
            hist_type = self.embed_forecast_type(torch.zeros(B, T, dtype=torch.long, device=device))
            state_emb = self.embed_state(states) + time_emb_h + hist_type
            rtg_emb = self.embed_rtg(returns_to_go) + time_emb_h + hist_type
            act_emb = self.embed_act(actions) + time_emb_h + hist_type
            h_hist = torch.stack([rtg_emb, state_emb, act_emb], dim=1)
            h_hist = h_hist.permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

            # Forecast stream (prefix)
            h_fore = None
            if forecast_states is not None and fore_len > 0:
                f_s = torch.nan_to_num(forecast_states, nan=0.0, posinf=0.0, neginf=0.0)
                f_r = torch.nan_to_num(forecast_rtgs, nan=0.0, posinf=0.0, neginf=0.0)
                if f_r.dim() == 2:
                    f_r = f_r.unsqueeze(-1)
                f_ts = forecast_timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)
                f_time = self.embed_timestep(f_ts)
                f_type = self.embed_forecast_type(torch.ones(B, fore_len, dtype=torch.long, device=device))
                f_se = self.embed_state(f_s) + f_time + f_type
                f_re = self.embed_rtg(f_r) + f_time + f_type
                f_ap = torch.zeros(B, fore_len, self.h_dim, device=device)
                h_fore = torch.stack([f_re, f_se, f_ap], dim=1)
                h_fore = h_fore.permute(0, 2, 1, 3).reshape(B, 3 * fore_len, self.h_dim)

            # Concat forecast prefix + history
            h = torch.cat([h_fore, h_hist], dim=1) if h_fore is not None else h_hist
            h = self.embed_ln(h)

            # Attention mask
            total_seq = h.shape[1]
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=device)
                attention_mask = attention_mask > 0
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                stacked = torch.stack([attention_mask, attention_mask, attention_mask], dim=1)
                stacked = stacked.permute(0, 2, 1).reshape(B, 3 * T)
                if h_fore is not None:
                    f_mask = torch.ones(B, 3 * fore_len, dtype=torch.bool, device=device)
                    full_mask = torch.cat([f_mask, stacked], dim=1)
                else:
                    full_mask = stacked
            else:
                full_mask = torch.ones(B, total_seq, dtype=torch.bool, device=device)

            for block in self.transformer:
                h = block(h, key_padding_mask=full_mask)
            h = self.ln_f(h)

            # Decode only from history positions (last T positions in the sequence)
            T_total = total_seq // 3
            h = h.reshape(B, T_total, 3, self.h_dim).permute(0, 2, 1, 3)
            h_hist_tok = h[:, :, -T:, :]

            if self._tie_weights:
                ret_preds = F.linear(h_hist_tok[:, 2], self.pred_rtg.weight.t())
                sta_preds = F.linear(h_hist_tok[:, 2], self.pred_state.weight.t())
                act_preds = torch.tanh(F.linear(h_hist_tok[:, 1], self.pred_act[0].weight.t()))
            else:
                ret_preds = self.pred_rtg(h_hist_tok[:, 2])
                sta_preds = self.pred_state(h_hist_tok[:, 2])
                act_preds = self.pred_act(h_hist_tok[:, 1])
            return (
                torch.nan_to_num(ret_preds, nan=0.0, posinf=0.0, neginf=0.0),
                torch.nan_to_num(sta_preds, nan=0.0, posinf=0.0, neginf=0.0),
                torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0),
            )

        def get_action(self, states, actions, returns_to_go, timesteps,
                       attention_mask=None, forecast_states=None,
                       forecast_rtgs=None, forecast_timesteps=None):
            ndim = states.dim()
            if ndim == 2:
                states = states.unsqueeze(0)
            if actions.dim() == ndim:
                actions = actions.unsqueeze(0)
            if returns_to_go.dim() == ndim:
                returns_to_go = returns_to_go.unsqueeze(0)
            if timesteps.dim() == ndim:
                timesteps = timesteps.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == ndim:
                attention_mask = attention_mask.unsqueeze(0)
            if forecast_states is not None and forecast_states.dim() == ndim:
                forecast_states = forecast_states.unsqueeze(0)
            if forecast_rtgs is not None and forecast_rtgs.dim() == ndim:
                forecast_rtgs = forecast_rtgs.unsqueeze(0)
            if forecast_timesteps is not None and forecast_timesteps.dim() == ndim:
                forecast_timesteps = forecast_timesteps.unsqueeze(0)
            if returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(-1)
            elif returns_to_go.dim() == 3 and returns_to_go.shape[-1] != 1:
                returns_to_go = returns_to_go.unsqueeze(-1)
            if forecast_rtgs is not None:
                if forecast_rtgs.dim() == 2:
                    forecast_rtgs = forecast_rtgs.unsqueeze(-1)
                elif forecast_rtgs.dim() == 3 and forecast_rtgs.shape[-1] != 1:
                    forecast_rtgs = forecast_rtgs.unsqueeze(-1)

            _, _, act_preds = self.forward(states, actions, returns_to_go, timesteps,
                attention_mask=attention_mask, forecast_states=forecast_states,
                forecast_rtgs=forecast_rtgs, forecast_timesteps=forecast_timesteps)
            act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)
            action = act_preds[0, -1]
            if action.ndim == 1:
                action = action.unsqueeze(0)
            if action.shape[-1] > 1:
                action = torch.cat([action[..., :1], torch.clamp(action[..., 1:], 0.0, 1.0)], dim=-1)
            return action

    print("✅ ForecastDecisionTransformer loaded")
    return (ForecastDecisionTransformer,)


@app.cell
def _(np, pl, torch):
    class ForecastTrajectoryDataset(torch.utils.data.Dataset):
        """Yields (history_window, forecast_window) pairs from a trajectory parquet."""

        def __init__(self, df, context_length=210, state_dim=18, act_dim=9,
                     forecast_len=48, discount_factor=0.95, min_episode_length=None):
            self.context_length = context_length
            self.forecast_len = forecast_len
            self.state_dim = state_dim
            self.act_dim = act_dim
            self.discount_factor = discount_factor

            df_clean = df.filter(
                (pl.col("action").list.len() == act_dim) &
                (pl.col("norm_observation").list.len() == state_dim)
            )
            n_removed = len(df) - len(df_clean)
            if n_removed > 0 < n_removed < len(df):
                print(f"⚠️ Filtered {n_removed:,} rows ({n_removed/len(df)*100:.1f}%) with bad dims")

            total_window = context_length + forecast_len
            min_len = min_episode_length or total_window

            self.episodes: list[dict] = []
            self.indices: list[tuple[int, int]] = []

            for eid in df_clean["episode_id"].unique().to_list():
                grp = df_clean.filter(pl.col("episode_id") == eid)
                states = np.stack(grp["norm_observation"].to_list()).astype(np.float32)
                actions = np.stack(grp["action"].to_list()).astype(np.float32)
                rewards = np.array(grp["reward"].to_list(), dtype=np.float32)
                timesteps = np.array(grp["step"].to_list(), dtype=np.int64)
                rtgs = np.zeros_like(rewards, dtype=np.float32)
                running = 0.0
                for i in reversed(range(len(rewards))):
                    running = rewards[i] + discount_factor * running
                    rtgs[i] = running
                ep_len = states.shape[0]
                if ep_len < min_len:
                    continue
                self.episodes.append({
                    "states": states, "actions": actions, "rtgs": rtgs,
                    "timesteps": timesteps, "length": ep_len,
                })
                stride = max(1, context_length // 2)
                for start in range(0, ep_len - total_window + 1, stride):
                    self.indices.append((len(self.episodes) - 1, start))

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            ep_idx, start = self.indices[idx]
            ep = self.episodes[ep_idx]
            T, F = self.context_length, self.forecast_len

            h_end = start + T
            h_actual = min(T, ep["length"] - start)
            f_start = h_end
            f_actual = max(0, min(F, ep["length"] - f_start))

            buf = lambda shape: np.zeros(shape, dtype=np.float32)
            states = buf((T, self.state_dim))
            actions = buf((T, self.act_dim))
            rtgs = buf((T, 1))
            timesteps = np.zeros(T, dtype=np.int64)
            mask = np.zeros(T, dtype=np.float32)

            states[-h_actual:] = ep["states"][start:start + h_actual]
            actions[-h_actual:] = ep["actions"][start:start + h_actual]
            rtgs[-h_actual:, 0] = ep["rtgs"][start:start + h_actual]
            timesteps[-h_actual:] = ep["timesteps"][start:start + h_actual]
            mask[-h_actual:] = 1.0

            f_states = buf((F, self.state_dim))
            f_rtgs = buf((F, 1))
            f_timesteps = np.zeros(F, dtype=np.int64)
            f_mask = np.zeros(F, dtype=np.float32)
            if f_actual > 0:
                f_states[:f_actual] = ep["states"][f_start:f_start + f_actual]
                f_rtgs[:f_actual, 0] = ep["rtgs"][f_start:f_start + f_actual]
                f_timesteps[:f_actual] = ep["timesteps"][f_start:f_start + f_actual]
                f_mask[:f_actual] = 1.0

            return {
                "states": torch.from_numpy(states),
                "actions": torch.from_numpy(actions),
                "rtgs": torch.from_numpy(rtgs),
                "timesteps": torch.from_numpy(timesteps),
                "mask": torch.from_numpy(mask),
                "forecast_states": torch.from_numpy(f_states),
                "forecast_rtgs": torch.from_numpy(f_rtgs),
                "forecast_timesteps": torch.from_numpy(f_timesteps),
                "forecast_mask": torch.from_numpy(f_mask),
            }

    print("✅ ForecastTrajectoryDataset loaded")
    return (ForecastTrajectoryDataset,)


# ── CONFIG ──────────────────────────────────────────────────────────

@app.cell
def _(
    action_loss_weight, batch_size, context_len, drop_p, epochs_per_session,
    forecast_len, h_dim, json, json_config, lr, n_block, n_heads, n_kv_heads,
    qk_norm, return_loss_weight, state_loss_weight, tie_weights, use_json_config,
):
    TRAIN_CFG = {
        "state_dim": 18, "act_dim": 9, "max_timestep": 100000,
        "forecast_len": 48, "rope_enabled": True,
        "discount_factor": 0.95, "val_split": 0.1, "return_scale": 2.0,
        "weight_decay": 1e-4, "grad_clip_norm": 1.0,
        "checkpoint_every_n_batches": 500, "max_training_seconds": 11 * 3600,
    }
    if use_json_config.value:
        try:
            TRAIN_CFG.update(json.loads(json_config.value))
        except Exception as exc:
            print(f"⚠️ Invalid JSON: {exc}")
    else:
        TRAIN_CFG.update({
            "n_block": n_block.value, "h_dim": h_dim.value, "n_heads": n_heads.value,
            "context_len": context_len.value, "forecast_len": forecast_len.value,
            "drop_p": drop_p.value, "batch_size": batch_size.value, "lr": lr.value,
            "n_kv_heads": int(n_kv_heads.value) if n_kv_heads.value else None,
            "qk_norm": qk_norm.value, "tie_weights": tie_weights.value,
            "epochs_per_session": epochs_per_session.value,
            "action_loss_weight": action_loss_weight.value,
            "state_loss_weight": state_loss_weight.value,
            "return_loss_weight": return_loss_weight.value,
        })
    print(f"📋 {TRAIN_CFG['n_block']} blk, {TRAIN_CFG['h_dim']} dim, "
          f"ctx={TRAIN_CFG['context_len']}, forecast={TRAIN_CFG.get('forecast_len')}")
    return (TRAIN_CFG,)


@app.cell
def _(ForecastTrajectoryDataset, TRAIN_CFG, df, torch):
    dataset = ForecastTrajectoryDataset(
        data_path=df,
        context_length=TRAIN_CFG["context_len"],
        state_dim=TRAIN_CFG["state_dim"],
        act_dim=TRAIN_CFG["act_dim"],
        forecast_len=TRAIN_CFG.get("forecast_len", 48),
        discount_factor=TRAIN_CFG["discount_factor"],
    )
    split = int(len(dataset) * (1 - TRAIN_CFG["val_split"]))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False, num_workers=0
    )
    print(f"📊 {len(dataset)} windows, {len(train_ds)} train + {len(val_ds)} val")
    return dataset, train_ds, train_loader, val_ds, val_loader


# ── TRAINING HELPERS ────────────────────────────────────────────────

@app.cell
def _(
    CHECKPOINT_DIR, CHECKPOINT_PATH, ForecastDecisionTransformer, TRAIN_CFG, time, torch,
):
    def load_or_create_model(cfg, device, fresh=False):
        model = ForecastDecisionTransformer(
            state_dim=cfg["state_dim"], act_dim=cfg["act_dim"],
            n_block=cfg["n_block"], h_dim=cfg["h_dim"],
            context_len=cfg["context_len"],
            forecast_len=cfg.get("forecast_len", 48),
            n_heads=cfg["n_heads"], drop_p=cfg["drop_p"],
            max_timestep=cfg["max_timestep"],
            rope_enabled=cfg.get("rope_enabled", True),
            n_kv_heads=cfg.get("n_kv_heads"),
            qk_norm=cfg.get("qk_norm", False),
            tie_weights=cfg.get("tie_weights", False),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
        se, gs, tl, vl, bv, ss = 0, 0, [], [], float("inf"), None
        if not fresh and CHECKPOINT_PATH.exists():
            try:
                ck = torch.load(CHECKPOINT_PATH, map_location=device)
                model.load_state_dict(ck["model_state_dict"])
                opt.load_state_dict(ck["optimizer_state_dict"])
                sch.load_state_dict(ck["scheduler_state_dict"])
                se = ck.get("epoch", 0) + 1
                gs = ck.get("global_step", 0)
                tl = ck.get("train_losses", [])
                vl = ck.get("val_losses", [])
                bv = ck.get("best_val_loss", float("inf"))
                ss = ck.get("scaler_state_dict")
                print(f"✅ Resumed epoch={se-1}, step={gs}, best_val={bv:.6f}")
            except Exception as e:
                print(f"⚠️ Load failed: {e}")
        return model, opt, sch, se, gs, tl, vl, bv, ss

    def save_checkpoint(model, opt, sch, epoch, step, tl, vl, bv, scaler=None):
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict(),
            "epoch": epoch, "global_step": step,
            "train_losses": tl, "val_losses": vl,
            "best_val_loss": bv, "return_scale": TRAIN_CFG["return_scale"],
            "forecast_len": TRAIN_CFG.get("forecast_len", 48), "timestamp": time.time(),
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(payload, CHECKPOINT_PATH)
        freq = TRAIN_CFG.get("checkpoint_every_n_batches", 500)
        if freq > 0 and step > 0 and step % freq == 0:
            torch.save(payload, CHECKPOINT_DIR / f"checkpoint_step_{step}.pt")
            print(f"💾 Step checkpoint: step_{step}.pt")

    def load_from_hf(repo_id, filename, device, cfg):
        from huggingface_hub import hf_hub_download
        ck = torch.load(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model"),
                       map_location=device)
        model, opt, sch, *_ = load_or_create_model(cfg, device, fresh=True)
        if "model_state_dict" in ck:
            model.load_state_dict(ck["model_state_dict"])
        else:
            model.load_state_dict(ck)
        torch.save({"model_state_dict": model.state_dict()}, CHECKPOINT_PATH)
        return model, opt, sch, 0, 0, [], [], float("inf"), None

    print("✅ Training helpers ready")
    return load_from_hf, load_or_create_model, save_checkpoint


# ── TRAINING LOOP ───────────────────────────────────────────────────

@app.cell
def _(
    F, TRAIN_CFG, load_or_create_model, mo, save_checkpoint, time, torch,
    train_btn, train_loader, val_loader,
):
    session_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fore_len = TRAIN_CFG.get("forecast_len", 48)
    tl, vl, bv, gs = [], [], float("inf"), 0

    if not train_btn.value:
        mo.stop(True, mo.md("> Click **Start Training**."))

    print("=" * 60)
    print("🎯 FORECAST DT TRAINING")
    print(f"💻 {device} | {TRAIN_CFG['n_block']} blk, {TRAIN_CFG['h_dim']} dim, forecast={fore_len}")
    print("=" * 60)

    model, opt, sch, start_epoch, gs, tl, vl, bv, ss = load_or_create_model(TRAIN_CFG, device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler and ss:
        try:
            scaler.load_state_dict(ss)
        except Exception:
            pass

    _total = len(train_loader)
    ckpt_freq = TRAIN_CFG.get("checkpoint_every_n_batches", 500)

    for epoch in range(start_epoch, start_epoch + TRAIN_CFG["epochs_per_session"]):
        model.train()
        tl_acc = al_acc = sl_acc = rl_acc = count = 0.0

        for bi, batch in enumerate(train_loader):
            st = batch["states"].to(device)
            ac = batch["actions"].to(device)
            rt = batch["rtgs"].to(device) / TRAIN_CFG["return_scale"]
            ts = batch["timesteps"].to(device)
            mk = batch["mask"].to(device)
            fs = batch.get("forecast_states")
            fr = batch.get("forecast_rtgs")
            ft = batch.get("forecast_timesteps")
            if fs is not None and fore_len > 0:
                fs = fs.to(device)
                fr = fr.to(device) / TRAIN_CFG["return_scale"]
                ft = ft.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                rp, sp, ap = model(st, ac, rt, ts, mk, forecast_states=fs,
                                   forecast_rtgs=fr, forecast_timesteps=ft)
                a_loss = F.mse_loss(ap, ac)
                s_loss = F.mse_loss(sp, st)
                r_loss = F.mse_loss(rp.squeeze(-1), rt)
                loss = (TRAIN_CFG["action_loss_weight"] * a_loss +
                        TRAIN_CFG["state_loss_weight"] * s_loss +
                        TRAIN_CFG["return_loss_weight"] * r_loss)

            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip_norm"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip_norm"])
                opt.step()

            tl_acc += loss.item()
            al_acc += a_loss.item()
            sl_acc += s_loss.item()
            rl_acc += r_loss.item()
            count += 1
            gs += 1

            if gs % ckpt_freq == 0:
                save_checkpoint(model, opt, sch, epoch, gs, tl, vl, bv, scaler)

            if bi % 100 == 0:
                print(f"  B{bi:5d}/{_total:5d} | gs={gs:6d} | loss={loss.item():.6f} "
                      f"| act={a_loss.item():.6f} | {time.time()-session_start:.0f}s")

        tl.append(tl_acc / max(1, count))

        # Validation
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vs = batch["states"].to(device)
                va = batch["actions"].to(device)
                vr = batch["rtgs"].to(device) / TRAIN_CFG["return_scale"]
                vt = batch["timesteps"].to(device)
                vm = batch["mask"].to(device)
                vfs = batch.get("forecast_states")
                vfr = batch.get("forecast_rtgs")
                vft = batch.get("forecast_timesteps")
                if vfs is not None and fore_len > 0:
                    vfs = vfs.to(device)
                    vfr = vfr.to(device) / TRAIN_CFG["return_scale"]
                    vft = vft.to(device)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    rp, sp, ap = model(vs, va, vr, vt, vm,
                                       forecast_states=vfs, forecast_rtgs=vfr,
                                       forecast_timesteps=vft)
                    v_loss += F.mse_loss(ap, va).item()
        v_loss /= max(1, len(val_loader))
        vl.append(v_loss)

        if v_loss < bv:
            bv = v_loss
            torch.save(model.state_dict(), Path("/workspace/dt_checkpoints/best_model.pt"))
            print(f"🏆 New best model! val_loss={bv:.6f}")

        save_checkpoint(model, opt, sch, epoch, gs, tl, vl, bv, scaler)
        print(f"📊 Epoch {epoch+1}: train={tl[-1]:.6f} val={vl[-1]:.6f} | "
              f"act={al_acc/count:.6f} | {time.time()-session_start:.0f}s")
        sch.step()

    print(f"✅ Done in {time.time()-session_start:.0f}s")
    return device, gs, session_start, tl, vl, bv


# ── PLOTS ───────────────────────────────────────────────────────────

@app.cell
def _(mo, np, plt, tl, vl):
    if tl and vl:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        a1.plot(tl, "b-o", label="Train", markersize=6)
        a1.plot(vl, "r-s", label="Val", markersize=6)
        a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend(); a1.grid(True, alpha=0.3)
        a2.bar(np.arange(len(tl)) - 0.175, tl, 0.35, label="Train", color="steelblue")
        a2.bar(np.arange(len(vl)) + 0.175, vl, 0.35, label="Val", color="coral")
        a2.legend(); a2.grid(True, alpha=0.3)
        plt.tight_layout()
        mo.mpl.interactive(fig)
    else:
        mo.md("> Loss curves after first epoch.")
    return


# ── UPLOAD ──────────────────────────────────────────────────────────

@app.cell
def _(
    CHECKPOINT_DIR, CHECKPOINT_PATH, ForecastDecisionTransformer, HfApi, TRAIN_CFG,
    hf_repo_id, hf_token_input, mo, os, torch, upload_btn,
):
    mo.stop(not upload_btn.value, mo.md("Press **Upload**."))
    repo_id = hf_repo_id.value.strip()
    if not repo_id:
        raise ValueError("Provide a HF repo ID.")
    token = (hf_token_input.value or os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise ValueError("Provide a HF token.")

    src = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None
    best = Path("/workspace/dt_checkpoints/best_model.pt")
    src = best if best.exists() else src
    if src is None:
        raise FileNotFoundError("No checkpoint found.")

    model = ForecastDecisionTransformer(
        state_dim=TRAIN_CFG["state_dim"], act_dim=TRAIN_CFG["act_dim"],
        n_block=TRAIN_CFG["n_block"], h_dim=TRAIN_CFG["h_dim"],
        context_len=TRAIN_CFG["context_len"],
        forecast_len=TRAIN_CFG.get("forecast_len", 48),
        n_heads=TRAIN_CFG["n_heads"], drop_p=TRAIN_CFG["drop_p"],
        max_timestep=TRAIN_CFG["max_timestep"],
        rope_enabled=TRAIN_CFG.get("rope_enabled", True),
        n_kv_heads=TRAIN_CFG.get("n_kv_heads"),
        qk_norm=TRAIN_CFG.get("qk_norm", False),
        tie_weights=TRAIN_CFG.get("tie_weights", False),
    )
    ck = torch.load(src, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck))
    meta = {k: v for k, v in ck.items() if k != "model_state_dict"}

    upload_path = Path("/workspace/dt_checkpoints/upload_model.pt")
    torch.save({"model_state_dict": model.state_dict(), **meta}, upload_path)
    HfApi().upload_file(path_or_fileobj=str(upload_path), path_in_repo="forecast_dt_model.pt",
                        repo_id=repo_id, repo_type="model", token=token)
    print(f"✅ Uploaded forecast_dt_model.pt → {repo_id}")
    return
