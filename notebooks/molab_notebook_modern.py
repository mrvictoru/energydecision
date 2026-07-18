import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import json
    import os
    import sys
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
        F,
        HfApi,
        Path,
        hf_hub_download,
        json,
        mo,
        nn,
        np,
        os,
        pl,
        plt,
        time,
        torch,
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
    use_json_config = mo.ui.checkbox(label="Use JSON config instead of individual controls", value=False)
    include_base_dataset = mo.ui.checkbox(label="Include base dataset (aemo_fcas_dataset.parquet)", value=True)
    include_grpo_dataset = mo.ui.checkbox(label="Include GRPO dataset (aemo_fcas_grpo_dataset.parquet)", value=True)

    n_block = mo.ui.number(value=8, label="Blocks", full_width=True)
    h_dim = mo.ui.number(value=384, label="Hidden dim", full_width=True)
    n_heads = mo.ui.number(value=8, label="Heads", full_width=True)
    context_len = mo.ui.number(value=180, label="Context len", full_width=True)
    drop_p = mo.ui.number(value=0.15, label="Dropout", full_width=True)
    n_kv_heads = mo.ui.number(value=8, label="KV heads (0=default)", full_width=True)
    qk_norm = mo.ui.checkbox(label="Enable QK-Norm", value=False)
    tie_weights = mo.ui.checkbox(label="Tie embeddings to predictions", value=False)

    batch_size = mo.ui.number(value=128, label="Batch size", full_width=True)
    epochs_per_session = mo.ui.number(value=3, label="Epochs/session", full_width=True)
    lr = mo.ui.number(value=3e-5, label="Learning rate", full_width=True)

    action_loss_weight = mo.ui.number(value=0.999, label="Action loss weight", full_width=True)
    state_loss_weight = mo.ui.number(value=0.002, label="State loss weight", full_width=True)
    return_loss_weight = mo.ui.number(value=0.0001, label="Return loss weight", full_width=True)

    _DEFAULT_JSON = json.dumps(
        {
            "state_dim": 18,
            "act_dim": 9,
            "n_block": 8,
            "h_dim": 384,
            "n_heads": 8,
            "context_len": 180,
            "drop_p": 0.15,
            "n_kv_heads": 8,
            "qk_norm": False,
            "tie_weights": False,
            "batch_size": 128,
            "epochs_per_session": 3,
            "lr": 3e-5,
            "action_loss_weight": 0.999,
            "state_loss_weight": 0.002,
            "return_loss_weight": 0.0001,
            "discount_factor": 0.95,
            "return_scale": 2.0,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
        },
        indent=2,
    )
    json_config = mo.ui.text_area(value=_DEFAULT_JSON, label="JSON config", full_width=True)

    train_btn = mo.ui.run_button(label="Start Training", kind="success")
    upload_btn = mo.ui.run_button(label="Upload to HuggingFace", kind="info")
    hf_repo_id = mo.ui.text(value="mrvictoru/energydecision-dt-v2", label="Hugging Face repo", full_width=True)
    hf_token_input = mo.ui.text(value=os.environ.get("HF_TOKEN", ""), label="Hugging Face token", full_width=True)

    resume_from_hf = mo.ui.checkbox(label="Resume from HuggingFace checkpoint", value=False)
    hf_checkpoint_path = mo.ui.text(value="", label="HF checkpoint filename (e.g. checkpoint_step_500.pt)", full_width=True)

    manual_controls = mo.vstack(
        [
            mo.md("### Architecture"),
            mo.hstack([n_block, h_dim, n_heads], justify="start", gap=1),
            mo.hstack([context_len, drop_p], justify="start", gap=1),
            mo.hstack([n_kv_heads, qk_norm, tie_weights], justify="start", gap=2),
            mo.md("### Optimization"),
            mo.hstack([batch_size, epochs_per_session, lr], justify="start", gap=1),
            mo.hstack([action_loss_weight, state_loss_weight, return_loss_weight], justify="start", gap=1),
        ],
        gap=0.5,
    )

    manual_controls
    return (
        action_loss_weight,
        batch_size,
        context_len,
        drop_p,
        epochs_per_session,
        fresh_start,
        h_dim,
        hf_checkpoint_path,
        hf_repo_id,
        hf_token_input,
        include_base_dataset,
        include_grpo_dataset,
        json_config,
        lr,
        manual_controls,
        n_block,
        n_heads,
        n_kv_heads,
        qk_norm,
        resume_from_hf,
        return_loss_weight,
        state_loss_weight,
        tie_weights,
        train_btn,
        upload_btn,
        use_json_config,
        use_pilot,
    )


@app.cell
def _(
    BEST_MODEL_PATH,
    ckpt_info,
    fresh_start,
    hf_checkpoint_path,
    hf_repo_id,
    hf_token_input,
    include_base_dataset,
    include_grpo_dataset,
    json_config,
    manual_controls,
    mo,
    resume_from_hf,
    train_btn,
    upload_btn,
    use_json_config,
    use_pilot,
):
    _dataset_section = [
        mo.md("### Dataset"),
        mo.hstack([use_pilot, fresh_start, use_json_config], justify="start", gap=2),
        mo.md(
            "Pilot mode uses the compact pilot parquet. Full mode lets you combine the base and GRPO datasets."
        ),
    ]
    if not use_pilot.value:
        _dataset_section.extend(
            [
                mo.hstack([include_base_dataset, include_grpo_dataset], justify="start", gap=2),
                mo.md("When both are enabled, the selected datasets are concatenated vertically before training."),
            ]
        )

    _config_section = mo.vstack(
        [
            mo.md("### Configuration"),
            mo.md("JSON mode overrides the individual controls below.") if use_json_config.value else mo.md("Using individual controls for the active training config."),
            json_config if use_json_config.value else manual_controls,
        ],
        gap=0.5,
    )

    mo.vstack(
        [
            mo.md(
                f"""
    ## AEMO Decision Transformer - Modern Training

    **Local checkpoint**: {ckpt_info}

    **Best model path**: {BEST_MODEL_PATH}
    """
            ),
            mo.vstack(_dataset_section, gap=0.5),
            _config_section,
            mo.md("### Checkpoint Resume"),
            mo.hstack([resume_from_hf, hf_checkpoint_path], justify="start", gap=2),
            mo.md("### Actions"),
            mo.hstack([train_btn, upload_btn], justify="start", gap=2),
            hf_repo_id,
            hf_token_input,
        ]
    )
    return


@app.cell
def _(
    Path,
    hf_hub_download,
    include_base_dataset,
    include_grpo_dataset,
    pl,
    use_pilot,
):
    REPO_ID = "mrvictoru/AEMO_simulated_trade"
    CACHE_DIR = Path("/workspace")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_df = None
    grpo_df = None

    if use_pilot.value:
        filename = "aemo_fcas_pilot.parquet"
        local_path = CACHE_DIR / filename

        if not local_path.exists():
            print(f"⬇️ Downloading {filename} from HuggingFace...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(CACHE_DIR), local_dir_use_symlinks=False, repo_type="dataset")
        else:
            print(f"📦 Using cached file: {local_path}")

        df = pl.read_parquet(local_path)
        print(f"Loaded pilot dataset with {len(df):,} rows from {local_path}")
    else:
        base_filename = "aemo_fcas_dataset.parquet"
        grpo_filename = "aemo_fcas_grpo_dataset.parquet"

        base_path = CACHE_DIR / base_filename
        grpo_path = CACHE_DIR / grpo_filename

        selected_dfs = []

        if include_base_dataset.value:
            if not base_path.exists():
                print(f"⬇️ Downloading {base_filename} from HuggingFace...")
                hf_hub_download(repo_id=REPO_ID, filename=base_filename, local_dir=str(CACHE_DIR), local_dir_use_symlinks=False, repo_type="dataset")
            else:
                print(f"📦 Using cached file: {base_path}")

            base_df = pl.read_parquet(base_path)
            selected_dfs.append(base_df)
            print(f"Loaded base dataset: {len(base_df):,} rows from {base_path}")

        if include_grpo_dataset.value:
            if not grpo_path.exists():
                print(f"⬇️ Downloading {grpo_filename} from HuggingFace...")
                hf_hub_download(repo_id=REPO_ID, filename=grpo_filename, local_dir=str(CACHE_DIR), local_dir_use_symlinks=False, repo_type="dataset")
            else:
                print(f"📦 Using cached file: {grpo_path}")

            grpo_df = pl.read_parquet(grpo_path)
            selected_dfs.append(grpo_df)
            print(f"Loaded GRPO dataset: {len(grpo_df):,} rows from {grpo_path}")

        if not selected_dfs:
            raise ValueError("Select at least one dataset checkbox when Pilot mode is disabled.")

        if len(selected_dfs) == 1:
            df = selected_dfs[0]
            print(f"Selected dataset rows: {len(df):,}")
        else:
            # Unify schemas: cast all float-like list columns to Float64 before concat
            _target_schema = selected_dfs[0].schema
            print(f"Target schema: {_target_schema}")
            _aligned = [selected_dfs[0]]
            for _other in selected_dfs[1:]:
                _cast_exprs = []
                for _col_name, _col_type in _other.schema.items():
                    if _col_name in _target_schema and _target_schema[_col_name] != _col_type:
                        if isinstance(_col_type, pl.List) and isinstance(_target_schema[_col_name], pl.List):
                            print(f"  Casting '{_col_name}': {_col_type} → {_target_schema[_col_name]}")
                            _cast_exprs.append(pl.col(_col_name).cast(_target_schema[_col_name]))
                        elif pl.Float32 in (getattr(_col_type, "inner", _col_type), _col_type) and pl.Float64 in (getattr(_target_schema[_col_name], "inner", _target_schema[_col_name]), _target_schema[_col_name]):
                            print(f"  Casting '{_col_name}': {_col_type} → {_target_schema[_col_name]}")
                            _cast_exprs.append(pl.col(_col_name).cast(_target_schema[_col_name]))
                if _cast_exprs:
                    _aligned.append(_other.with_columns(*_cast_exprs))
                else:
                    _aligned.append(_other)
            df = pl.concat(_aligned, how="vertical")
            print(f"Combined dataset rows: {len(df):,}")
            print(f"Combined schema: {df.schema}")
    return base_df, df, grpo_df


@app.cell
def _(df, pl):
    print("📊 Dataset profile")
    episode_stats = df.group_by("episode_id").agg(
        pl.col("step").len().alias("n_steps"),
        pl.col("reward").sum().alias("total_reward"),
    )
    print(episode_stats.head())
    return


@app.cell
def _(F, nn, torch):
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            norm_x = torch.mean(x * x, dim=-1, keepdim=True)
            x = x * torch.rsqrt(norm_x + self.eps)
            return x * self.scale

    class SwiGLU(nn.Module):
        def __init__(self, dim, hidden_dim, drop_p):
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
                raise ValueError("seq_len exceeds RoPE cache")
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
        def __init__(
            self,
            h_dim,
            max_T,
            n_heads,
            drop_p,
            rope_enabled=False,
            rope_max_position=4096,
            rope_base=10000.0,
            n_kv_heads=None,
            qk_norm=False,
        ):
            super().__init__()
            assert h_dim % n_heads == 0, "h_dim must be divisible by n_heads"
            self.n_heads = n_heads
            self.head_dim = h_dim // n_heads
            self.drop_p = drop_p
            self.rope_enabled = rope_enabled

            if n_kv_heads is None:
                n_kv_heads = n_heads
            assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
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
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.drop_p if self.training else 0.0,
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=True,
                    dropout_p=self.drop_p if self.training else 0.0,
                )

            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.proj_drop(self.proj(y))
            y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            return y

    class ModernBlock(nn.Module):
        def __init__(
            self,
            h_dim,
            max_T,
            n_heads,
            drop_p,
            rope_enabled=False,
            rope_max_position=4096,
            rope_base=10000.0,
            n_kv_heads=None,
            qk_norm=False,
        ):
            super().__init__()
            self.norm1 = RMSNorm(h_dim)
            self.attn = CausalSelfAttention(
                h_dim,
                max_T,
                n_heads,
                drop_p,
                rope_enabled=rope_enabled,
                rope_max_position=rope_max_position,
                rope_base=rope_base,
                n_kv_heads=n_kv_heads,
                qk_norm=qk_norm,
            )
            self.norm2 = RMSNorm(h_dim)
            self.ffn = SwiGLU(h_dim, 4 * h_dim, drop_p)

        def forward(self, x, key_padding_mask=None):
            x = x + self.attn(self.norm1(x), key_padding_mask)
            x = x + self.ffn(self.norm2(x))
            return x

    class DecisionTransformer(nn.Module):
        def __init__(
            self,
            state_dim,
            act_dim,
            n_block=8,
            h_dim=384,
            context_len=180,
            n_heads=8,
            drop_p=0.15,
            max_timestep=100000,
            use_rope=False,
            rope_base=10000.0,
            rope_max_position=4096,
            n_kv_heads=None,
            qk_norm=False,
            tie_weights=False,
        ):
            super().__init__()
            self.state_dim = state_dim
            self.act_dim = act_dim
            self.h_dim = h_dim
            self.context_len = context_len
            self.n_heads = n_heads
            self.n_block = n_block
            self.drop_p = drop_p
            self.max_timestep = max_timestep
            self.rope_enabled = use_rope

            input_seq_len = 3 * context_len
            self.transformer = nn.ModuleList(
                [
                    ModernBlock(
                        h_dim,
                        input_seq_len,
                        n_heads,
                        drop_p,
                        rope_enabled=use_rope,
                        rope_max_position=rope_max_position,
                        rope_base=rope_base,
                        n_kv_heads=n_kv_heads,
                        qk_norm=qk_norm,
                    )
                    for _ in range(n_block)
                ]
            )

            self.embed_ln = RMSNorm(h_dim)
            self.embed_timestep = nn.Embedding(max_timestep, h_dim)
            self.embed_rtg = nn.Linear(1, h_dim)
            self.embed_state = nn.Linear(state_dim, h_dim)
            self.embed_act = nn.Linear(act_dim, h_dim)
            self.ln_f = RMSNorm(h_dim)

            self._tie_weights = tie_weights
            if tie_weights:
                self.pred_rtg = nn.Linear(1, h_dim, bias=False)
                self.pred_state = nn.Linear(state_dim, h_dim, bias=False)
                pred_act_linear = nn.Linear(act_dim, h_dim, bias=False)
                self.pred_rtg.weight = self.embed_rtg.weight
                self.pred_state.weight = self.embed_state.weight
                pred_act_linear.weight = self.embed_act.weight
                self.pred_act = nn.Sequential(pred_act_linear, nn.Tanh())
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

        def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
            B, T, _ = states.shape

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

            time_emb = self.embed_timestep(timesteps)
            state_emb = self.embed_state(states) + time_emb
            rtg_emb = self.embed_rtg(returns_to_go) + time_emb
            act_emb = self.embed_act(actions) + time_emb

            h = torch.stack([rtg_emb, state_emb, act_emb], dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
            h = self.embed_ln(h)

            if attention_mask is not None:
                attention_mask = attention_mask.to(device=h.device)
                attention_mask = attention_mask > 0
                stacked_mask = torch.stack([attention_mask, attention_mask, attention_mask], dim=1)
                stacked_mask = stacked_mask.permute(0, 2, 1).reshape(B, 3 * T)
            else:
                stacked_mask = torch.ones(B, 3 * T, dtype=torch.bool, device=h.device)

            for block in self.transformer:
                h = block(h, key_padding_mask=stacked_mask)

            h = self.ln_f(h)
            h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

            if self._tie_weights:
                return_preds = F.linear(h[:, 2], self.pred_rtg.weight.t())
                state_preds = F.linear(h[:, 2], self.pred_state.weight.t())
                act_preds = torch.tanh(F.linear(h[:, 1], self.pred_act[0].weight.t()))
            else:
                return_preds = self.pred_rtg(h[:, 2])
                state_preds = self.pred_state(h[:, 2])
                act_preds = self.pred_act(h[:, 1])

            return_preds = torch.nan_to_num(return_preds, nan=0.0, posinf=0.0, neginf=0.0)
            state_preds = torch.nan_to_num(state_preds, nan=0.0, posinf=0.0, neginf=0.0)
            act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)
            return return_preds, state_preds, act_preds

        def get_action(self, states, actions, returns_to_go, timesteps, attention_mask=None):
            if states.dim() == 2:
                states = states.unsqueeze(0)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)
            if returns_to_go.dim() == 1:
                returns_to_go = returns_to_go.unsqueeze(0).unsqueeze(-1)
            elif returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(-1)
            if timesteps.dim() == 1:
                timesteps = timesteps.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            _, _, act_preds = self.forward(states, actions, returns_to_go, timesteps, attention_mask=attention_mask)
            act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)
            action = act_preds[0, -1]
            if action.ndim == 1:
                action = action.unsqueeze(0)
            if action.shape[-1] > 1:
                action = torch.cat([action[..., :1], torch.clamp(action[..., 1:], 0.0, 1.0)], dim=-1)
            return action

    print("✅ Modern DecisionTransformer model loaded")
    return (DecisionTransformer,)


@app.cell
def _(np, pl, torch):
    class TrajectoryDataset(torch.utils.data.Dataset):
        def __init__(self, df, context_length=180, state_dim=18, act_dim=9, discount_factor=0.95):
            self.context_length = context_length
            self.state_dim = state_dim
            self.act_dim = act_dim
            self.discount_factor = discount_factor

            # Filter out rows with inconsistent action/observation dimensions
            df_clean = df.filter(
                (pl.col("action").list.len() == act_dim) &
                (pl.col("norm_observation").list.len() == state_dim)
            )
            n_removed = len(df) - len(df_clean)
            if n_removed > 0:
                print(f"⚠️ Filtered out {n_removed:,} rows ({n_removed/len(df)*100:.1f}%) with mismatched dims")

            episodes = df_clean.group_by("episode_id").agg(
                pl.col("step").len().alias("n_steps"),
                pl.col("norm_observation").alias("obs"),
                pl.col("action").alias("act"),
                pl.col("reward").alias("rew"),
            )
            episodes = episodes.filter(pl.col("n_steps") >= context_length * 3)

            all_states, all_actions, all_rtgs, all_timesteps = [], [], [], []
            for row in episodes.iter_rows(named=True):
                obs_arr = np.array(row["obs"], dtype=np.float32)
                act_arr = np.array(row["act"], dtype=np.float32)
                rew_arr = np.array(row["rew"], dtype=np.float32)
                n = len(rew_arr)
                rtg = np.zeros(n, dtype=np.float32)
                running = 0.0
                for t in reversed(range(n)):
                    running = rew_arr[t] + discount_factor * running
                    rtg[t] = running

                stride = context_length // 2
                for i in range(0, n - context_length + 1, stride):
                    end = i + context_length
                    all_states.append(obs_arr[i:end])
                    all_actions.append(act_arr[i:end])
                    all_rtgs.append(rtg[i:end])
                    all_timesteps.append(np.arange(i, end, dtype=np.int64))

            self.states = np.stack(all_states).astype(np.float32)
            self.actions = np.stack(all_actions).astype(np.float32)
            self.rtgs = np.stack(all_rtgs).astype(np.float32)
            self.timesteps = np.stack(all_timesteps).astype(np.int64)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.states[idx]),
                torch.tensor(self.actions[idx]),
                torch.tensor(self.rtgs[idx]),
                torch.tensor(self.timesteps[idx]),
            )

    print("✅ TrajectoryDataset updated with dimension filtering")
    return (TrajectoryDataset,)


@app.cell
def _(base_df, grpo_df, mo, pl):
    mo.stop(base_df is None and grpo_df is None,
            mo.md("🔍 Skipping cross-dataset dimension audit in pilot mode."))

    def _describe_lengths(label, dataset):
        print(f"🔍 Checking action length distribution across {label} dataset...")
        act_lens = dataset.select(
            pl.col("action").list.len().alias("act_len")
        ).group_by("act_len").agg(pl.len().alias("count")).sort("act_len")
        print(act_lens)

        print(f"\n🔍 Checking observation length distribution across {label} dataset...")
        obs_lens = dataset.select(
            pl.col("norm_observation").list.len().alias("obs_len")
        ).group_by("obs_len").agg(pl.len().alias("count")).sort("obs_len")
        print(obs_lens)

    if base_df is not None:
        _describe_lengths("base", base_df)

    if grpo_df is not None:
        print()
        _describe_lengths("GRPO", grpo_df)
    return


@app.cell
def _(
    action_loss_weight,
    batch_size,
    context_len,
    drop_p,
    epochs_per_session,
    h_dim,
    json,
    json_config,
    lr,
    n_block,
    n_heads,
    n_kv_heads,
    qk_norm,
    return_loss_weight,
    state_loss_weight,
    tie_weights,
    use_json_config,
):
    TRAIN_CFG = {
        "state_dim": 18,
        "act_dim": 9,
        "max_timestep": 100000,
        "use_rope": False,
        "n_kv_heads": 8,
        "qk_norm": False,
        "tie_weights": False,
        "discount_factor": 0.95,
        "val_split": 0.1,
        "return_scale": 2.0,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "checkpoint_every_n_batches": 500,
        "max_training_seconds": 11 * 3600,
    }

    if use_json_config.value:
        try:
            TRAIN_CFG.update(json.loads(json_config.value))
            _source = "JSON config"
        except Exception as exc:
            print(f"⚠️ Invalid JSON: {exc}")
            _source = "Fallback controls"
    else:
        TRAIN_CFG.update(
            {
                "n_block": n_block.value,
                "h_dim": h_dim.value,
                "n_heads": n_heads.value,
                "context_len": context_len.value,
                "drop_p": drop_p.value,
                "n_kv_heads": int(n_kv_heads.value) if n_kv_heads.value else None,
                "qk_norm": qk_norm.value,
                "tie_weights": tie_weights.value,
                "batch_size": batch_size.value,
                "lr": lr.value,
                "epochs_per_session": epochs_per_session.value,
                "action_loss_weight": action_loss_weight.value,
                "state_loss_weight": state_loss_weight.value,
                "return_loss_weight": return_loss_weight.value,
            }
        )
        _source = "Individual controls"

    print(f"📋 Active config source: {_source}")
    print(f"   Model: {TRAIN_CFG['n_block']} blocks, {TRAIN_CFG['h_dim']} dim, {TRAIN_CFG['n_heads']} heads")
    print(f"   Modern knobs: n_kv_heads={TRAIN_CFG.get('n_kv_heads')}, qk_norm={TRAIN_CFG.get('qk_norm', False)}, tie_weights={TRAIN_CFG.get('tie_weights', False)}")
    return (TRAIN_CFG,)


@app.cell
def _(
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    DecisionTransformer,
    TRAIN_CFG,
    time,
    torch,
):
    def load_or_create_model(cfg, device, fresh=False):
        """Load from local checkpoint if it exists and fresh=False, otherwise create fresh."""
        model = DecisionTransformer(
            state_dim=cfg["state_dim"],
            act_dim=cfg["act_dim"],
            n_block=cfg["n_block"],
            h_dim=cfg["h_dim"],
            context_len=cfg["context_len"],
            n_heads=cfg["n_heads"],
            drop_p=cfg["drop_p"],
            max_timestep=cfg["max_timestep"],
            use_rope=cfg.get("use_rope", False),
            n_kv_heads=cfg.get("n_kv_heads"),
            qk_norm=cfg.get("qk_norm", False),
            tie_weights=cfg.get("tie_weights", False),
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        start_epoch = 0
        global_step = 0
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        scaler_state = None

        if not fresh and CHECKPOINT_PATH.exists():
            try:
                print(f"📂 Loading checkpoint from {CHECKPOINT_PATH}...")
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                global_step = checkpoint.get("global_step", 0)
                train_losses = checkpoint.get("train_losses", [])
                val_losses = checkpoint.get("val_losses", [])
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                scaler_state = checkpoint.get("scaler_state_dict")
                print(f"✅ Resumed: epoch={start_epoch - 1}, global_step={global_step}, best_val_loss={best_val_loss:.6f}")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}. Starting fresh.")

        return model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, scaler_state


    def load_checkpoint_from_hf(repo_id, filename, device, cfg):
        """Download a checkpoint from HuggingFace and load it, saving locally."""
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        print(f"📥 Downloaded HF checkpoint: {repo_id}/{filename} → {local_path}")

        checkpoint = torch.load(local_path, map_location=device)

        model = DecisionTransformer(
            state_dim=cfg["state_dim"],
            act_dim=cfg["act_dim"],
            n_block=cfg["n_block"],
            h_dim=cfg["h_dim"],
            context_len=cfg["context_len"],
            n_heads=cfg["n_heads"],
            drop_p=cfg["drop_p"],
            max_timestep=cfg["max_timestep"],
            use_rope=cfg.get("use_rope", False),
            n_kv_heads=cfg.get("n_kv_heads"),
            qk_norm=cfg.get("qk_norm", False),
            tie_weights=cfg.get("tie_weights", False),
        ).to(device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            global_step = checkpoint.get("global_step", 0)
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        else:
            # Plain state_dict — start fresh training state
            model.load_state_dict(checkpoint)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            start_epoch, global_step = 0, 0
            train_losses, val_losses = [], []
            best_val_loss = float("inf")

        scaler_state = checkpoint.get("scaler_state_dict") if "model_state_dict" in checkpoint else None

        # Persist locally so future resumes don't need HF
        save_checkpoint(model, optimizer, scheduler, start_epoch - 1, global_step, train_losses, val_losses, best_val_loss)

        return model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, scaler_state


    def save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_losses, val_losses, best_val_loss, scaler=None):
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "return_scale": TRAIN_CFG["return_scale"],
            "n_kv_heads": TRAIN_CFG.get("n_kv_heads"),
            "qk_norm": TRAIN_CFG.get("qk_norm", False),
            "tie_weights": TRAIN_CFG.get("tie_weights", False),
            "timestamp": time.time(),
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()

        # Always update the latest checkpoint
        torch.save(payload, CHECKPOINT_PATH)

        # Also save a step-specific checkpoint every checkpoint_every_n_batches
        ckpt_every = TRAIN_CFG.get("checkpoint_every_n_batches", 500)
        if ckpt_every > 0 and global_step > 0 and global_step % ckpt_every == 0:
            step_path = CHECKPOINT_DIR / f"checkpoint_step_{global_step}.pt"
            torch.save(payload, step_path)
            print(f"💾 Step checkpoint saved: {step_path.name}")


    print("✅ Training helpers ready")
    return load_checkpoint_from_hf, load_or_create_model, save_checkpoint


@app.cell
def _(
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    F,
    TRAIN_CFG,
    TrajectoryDataset,
    df,
    fresh_start,
    hf_checkpoint_path,
    hf_repo_id,
    load_checkpoint_from_hf,
    load_or_create_model,
    mo,
    resume_from_hf,
    save_checkpoint,
    time,
    torch,
    train_btn,
    use_pilot,
):
    session_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    global_step = 0
    USE_PILOT = use_pilot.value

    if not train_btn.value:
        mo.stop(True, mo.md("> Click **Start Training** to begin or resume training."))

    print("=" * 60)
    print("🎯 TRAINING SESSION STARTED")
    print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Device: {device}")
    print(f"📋 Pilot: {'ON' if use_pilot.value else 'OFF'}")
    print(f"📐 Model: {TRAIN_CFG['n_block']} blocks, {TRAIN_CFG['h_dim']} dim, {TRAIN_CFG['n_heads']} heads")
    print(f"📦 Batch: {TRAIN_CFG['batch_size']} | LR: {TRAIN_CFG['lr']:.2e} | Epochs: {TRAIN_CFG['epochs_per_session']}")
    print("=" * 60)

    # Determine the source of truth
    _using_remote = resume_from_hf.value and bool(hf_checkpoint_path.value.strip())
    _using_fresh = fresh_start.value and not _using_remote

    if _using_fresh:
        print("🧹 Fresh start: deleting all local checkpoints...")
        for ckpt in CHECKPOINT_DIR.glob("*.pt"):
            ckpt.unlink(missing_ok=True)

    print("Get Dataset")
    dataset = TrajectoryDataset(
        df,
        context_length=TRAIN_CFG["context_len"],
        state_dim=TRAIN_CFG["state_dim"],
        act_dim=TRAIN_CFG["act_dim"],
        discount_factor=TRAIN_CFG["discount_factor"],
    )
    split = int(len(dataset) * (1 - TRAIN_CFG["val_split"]))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    print("Split and put dataset to loader")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False, num_workers=0)

    print("Loading model")

    if _using_remote:
        _hf_repo = hf_repo_id.value.strip()
        _hf_file = hf_checkpoint_path.value.strip()
        print(f"🌐 Loading checkpoint from HuggingFace: {_hf_repo}/{_hf_file}")
        model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, scaler_state = load_checkpoint_from_hf(
            repo_id=_hf_repo,
            filename=_hf_file,
            device=device,
            cfg=TRAIN_CFG,
        )
    else:
        model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, scaler_state = load_or_create_model(
            TRAIN_CFG, device, fresh=fresh_start.value
        )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler is not None and scaler_state is not None:
        try:
            scaler.load_state_dict(scaler_state)
            print("✅ Restored GradScaler state")
        except Exception as e:
            print(f"⚠️ Could not restore GradScaler: {e}")

    print(f"📊 Starting from epoch={start_epoch}, global_step={global_step}")
    if train_losses:
        print(f"   Previous train losses: {[f'{l:.4f}' for l in train_losses[-5:]]}")
    if val_losses:
        print(f"   Previous val losses: {[f'{l:.4f}' for l in val_losses[-5:]]}")
    print(f"   Best val_loss so far: {best_val_loss:.6f}")
    if scaler is not None:
        print("   ⚡ AMP mixed precision enabled")

    print("\n" + "=" * 60)
    print(f"🏋️ TRAINING LOOP ({TRAIN_CFG['epochs_per_session']} epochs)")
    print("=" * 60)
    _total_batches = len(train_loader)
    ckpt_freq = TRAIN_CFG.get("checkpoint_every_n_batches", 500)

    for epoch in range(start_epoch, start_epoch + TRAIN_CFG["epochs_per_session"]):
        _epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_state_loss = 0.0
        total_return_loss = 0.0
        batches_seen = 0

        print(f"\n{'─' * 60}")
        print(f"📚 EPOCH {epoch}")
        print(f"{'─' * 60}")

        for batch_idx, (states, actions, rtgs, timesteps) in enumerate(train_loader):
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)
            rtgs = rtgs / TRAIN_CFG["return_scale"]

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                return_preds, state_preds, action_preds = model(states, actions, rtgs, timesteps)
                action_loss = F.mse_loss(action_preds, actions)
                state_loss = F.mse_loss(state_preds, states)
                return_loss = F.mse_loss(return_preds.squeeze(-1), rtgs)
                loss = (
                    TRAIN_CFG["action_loss_weight"] * action_loss
                    + TRAIN_CFG["state_loss_weight"] * state_loss
                    + TRAIN_CFG["return_loss_weight"] * return_loss
                )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip_norm"])
                optimizer.step()

            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_state_loss += state_loss.item()
            total_return_loss += return_loss.item()
            batches_seen += 1
            global_step += 1

            # Save checkpoint every ckpt_freq batches
            if global_step % ckpt_freq == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_losses, val_losses, best_val_loss, scaler)

            if batch_idx % 100 == 0:
                _elapsed = time.time() - session_start
                print(
                    f"  Batch {batch_idx:5d}/{_total_batches:5d} | gstep={global_step:6d} "
                    f"| loss={loss.item():.6f} | act={action_loss.item():.6f} "
                    f"| state={state_loss.item():.6f} | ret={return_loss.item():.6f} "
                    f"| {_elapsed:.0f}s"
                )

        # End of epoch
        train_losses.append(total_loss / max(1, batches_seen))

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            _val_start = time.time()
            for states, actions, rtgs, timesteps in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)
                rtgs = rtgs / TRAIN_CFG["return_scale"]
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    return_preds, state_preds, action_preds = model(states, actions, rtgs, timesteps)
                    val_loss += F.mse_loss(action_preds, actions).item()
            val_loss /= max(1, len(val_loader))
            _val_time = time.time() - _val_start
        val_losses.append(val_loss)

        _avg_action_loss = total_action_loss / max(1, batches_seen)
        _avg_state_loss = total_state_loss / max(1, batches_seen)
        _avg_return_loss = total_return_loss / max(1, batches_seen)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 New best model! val_loss={best_val_loss:.6f}")

        _epoch_elapsed = time.time() - _epoch_start
        _elapsed = time.time() - session_start

        # Save checkpoint at end of epoch
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_losses, val_losses, best_val_loss, scaler)

        print(
            f"📊 Epoch {epoch + 1} done: train_loss={train_losses[-1]:.6f} "
            f"val_loss={val_losses[-1]:.6f} "
            f"| action={_avg_action_loss:.6f} state={_avg_state_loss:.6f} ret={_avg_return_loss:.6f} "
            f"🕒 {_epoch_elapsed:.1f}s (epoch) / {_elapsed:.1f}s (total) "
            f"⚡ val in {_val_time:.1f}s"
        )

        scheduler.step()

    _total_time = time.time() - session_start
    _hours = int(_total_time // 3600)
    _minutes = int((_total_time % 3600) // 60)
    _seconds = _total_time % 60
    print(f"✅ Training complete! Total time: {_hours}h {_minutes}m {_seconds:.1f}s ({_total_time:.1f}s)")
    print(f"Latest checkpoint at {CHECKPOINT_PATH}")
    return (
        USE_PILOT,
        best_val_loss,
        device,
        global_step,
        session_start,
        train_losses,
        val_losses,
    )


@app.cell
def _(USE_PILOT, device, mo, session_start):
    mo.md(f"""
    # AEMO Decision Transformer Training Dashboard

    **Session**: `{session_start:.0f}` | **Device**: `{device}` | **Pilot**: `{USE_PILOT}`
    """)
    return


@app.cell
def _(mo, np, plt, train_losses, val_losses):
    if train_losses and val_losses:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(train_losses, "b-o", label="Train Loss", markersize=6)
        ax1.plot(val_losses, "r-s", label="Val Loss", markersize=6)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        epochs = list(range(len(train_losses)))
        x_pos = np.arange(len(epochs))
        width = 0.35
        ax2.bar(x_pos - width / 2, train_losses, width, label="Train", color="steelblue")
        ax2.bar(x_pos + width / 2, val_losses, width, label="Val", color="coral")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Loss per Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        _output = mo.mpl.interactive(fig)
    else:
        _output = mo.md(
            """
    > Training has not completed an epoch yet. Loss curves will appear here after the first epoch.
    """
        )

    _output
    return


@app.cell
def _(
    TRAIN_CFG,
    USE_PILOT,
    best_val_loss,
    device,
    global_step,
    mo,
    session_start,
    time,
    torch,
    train_losses,
    val_losses,
):
    _elapsed_hrs = (time.time() - session_start) / 3600
    _train_loss_str = f"{train_losses[-1]:.6f}" if train_losses else "N/A"
    _val_loss_str = f"{val_losses[-1]:.6f}" if val_losses else "N/A"

    _summary = mo.md(f"""
    ## Training Session Summary

    | Metric | Value |
    |--------|-------|
    | **Device** | `{device}` |
    | **Pilot mode** | `{USE_PILOT}` |
    | **Final train loss** | `{_train_loss_str}` |
    | **Final val loss** | `{_val_loss_str}` |
    | **Best val loss** | `{best_val_loss:.6f}` |
    | **Total epochs** | `{len(train_losses)}` |
    | **Total steps** | `{global_step}` |
    | **Training duration** | `{_elapsed_hrs:.2f} hours` |

    ### Configuration
    - **Model**: DecisionTransformer with {TRAIN_CFG["n_block"]} blocks, {TRAIN_CFG["h_dim"]} hidden dim, {TRAIN_CFG["n_heads"]} heads
    - **Context length**: {TRAIN_CFG["context_len"]}
    - **Batch size**: {TRAIN_CFG["batch_size"]}
    - **Learning rate**: {TRAIN_CFG["lr"]}
    - **Action loss weight**: {TRAIN_CFG["action_loss_weight"]}
    """)

    if torch.cuda.is_available():
        _allocated = torch.cuda.max_memory_allocated() / 1e9
        _reserved = torch.cuda.max_memory_reserved() / 1e9
        _gpu_stats = mo.md(f"""
    | GPU Stat | Value |
    |----------|-------|
    | **Max memory allocated** | `{_allocated:.2f} GB` |
    | **Max memory reserved** | `{_reserved:.2f} GB` |
    | **Total VRAM** | `{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB` |
    """)
        mo.vstack([_summary, _gpu_stats])
    else:
        _summary
    return


@app.cell
def _(
    BEST_MODEL_PATH,
    CHECKPOINT_PATH,
    DecisionTransformer,
    HfApi,
    Path,
    TRAIN_CFG,
    hf_repo_id,
    hf_token_input,
    mo,
    os,
    torch,
    upload_btn,
):
    mo.stop(not upload_btn.value, mo.md("Press **Upload to HuggingFace** to upload the model"))

    repo_id = hf_repo_id.value.strip()
    if not repo_id:
        raise ValueError("Please provide a Hugging Face repo ID.")

    token = (hf_token_input.value or os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise ValueError("Please provide a Hugging Face token or set the HF_TOKEN environment variable.")

    def _build_upload_model():
        return DecisionTransformer(
            state_dim=TRAIN_CFG["state_dim"],
            act_dim=TRAIN_CFG["act_dim"],
            n_block=TRAIN_CFG["n_block"],
            h_dim=TRAIN_CFG["h_dim"],
            context_len=TRAIN_CFG["context_len"],
            n_heads=TRAIN_CFG["n_heads"],
            drop_p=TRAIN_CFG["drop_p"],
            max_timestep=TRAIN_CFG["max_timestep"],
            use_rope=TRAIN_CFG.get("use_rope", False),
            n_kv_heads=TRAIN_CFG.get("n_kv_heads"),
            qk_norm=TRAIN_CFG.get("qk_norm", False),
            tie_weights=TRAIN_CFG.get("tie_weights", False),
        )

    tmp_path = Path("/workspace/best_model.pt")
    if BEST_MODEL_PATH.exists():
        print("📂 Loading best model weights...")
        state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")
        model_upload = _build_upload_model()
        model_upload.load_state_dict(state_dict)
        _best_val_loss = float("inf")
    elif CHECKPOINT_PATH.exists():
        print("📂 No best_model.pt found — loading from latest checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model_upload = _build_upload_model()
        model_upload.load_state_dict(checkpoint["model_state_dict"])
        _best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    else:
        raise FileNotFoundError(f"No trained model found. Expected {BEST_MODEL_PATH} or {CHECKPOINT_PATH}")

    torch.save(
        {
            "model_state_dict": model_upload.state_dict(),
            "val_loss": _best_val_loss,
            "config": {
                k: v
                for k, v in TRAIN_CFG.items()
                if k in ("state_dim", "act_dim", "n_block", "h_dim", "n_heads", "context_len", "drop_p", "return_scale", "discount_factor")
            },
        },
        tmp_path,
    )
    print(f"💾 Model saved to {tmp_path} ({tmp_path.stat().st_size / 1e6:.1f} MB)")

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        print(f"📂 Repo {repo_id} already exists")
    except Exception:
        print(f"🆕 Creating HF repo {repo_id}...")
        api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    print(f"📤 Uploading to {repo_id}/aemo_dt_fcas_model.pt...")
    api.upload_file(
        path_or_fileobj=str(tmp_path),
        path_in_repo="aemo_dt_fcas_model.pt",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"✅ Uploaded best model to HF: {repo_id}/aemo_dt_fcas_model.pt")

    if BEST_MODEL_PATH.exists() and str(BEST_MODEL_PATH) != str(tmp_path):
        print("📤 Also uploading aemo_dt_fcas_best_checkpoint.pt...")
        api.upload_file(
            path_or_fileobj=str(BEST_MODEL_PATH),
            path_in_repo="aemo_dt_fcas_best_checkpoint.pt",
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )

    print(f"✅ Uploaded model to https://huggingface.co/{repo_id}")
    return


@app.cell
def _(TRAIN_CFG, TrajectoryDataset, df):
    # Verify the fix works
    print("🔍 Verifying TrajectoryDataset can now be built without shape errors...")

    test_dataset = TrajectoryDataset(
        df,
        context_length=TRAIN_CFG["context_len"],
        state_dim=TRAIN_CFG["state_dim"],
        act_dim=TRAIN_CFG["act_dim"],
        discount_factor=TRAIN_CFG["discount_factor"],
    )

    print(f"✅ Success! Dataset built with {len(test_dataset)} samples")
    print(f"   States shape: {test_dataset.states.shape}")
    print(f"   Actions shape: {test_dataset.actions.shape}")
    print(f"   RTGs shape: {test_dataset.rtgs.shape}")
    print(f"   Timesteps shape: {test_dataset.timesteps.shape}")

    # Verify a sample
    s, a, r, t = test_dataset[0]
    print(f"\n📋 First sample shapes: states={s.shape}, actions={a.shape}, rtgs={r.shape}, timesteps={t.shape}")
    return


if __name__ == "__main__":
    app.run()
