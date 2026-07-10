import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    # Core imports (pre-installed on MoLab)
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import polars as pl
    import math
    import os
    import json
    import time
    import subprocess
    import shutil
    from pathlib import Path
    from dataclasses import dataclass

    # Data visualization
    import matplotlib.pyplot as plt

    # Additional packages that may need installation
    import micropip

    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        subprocess.check_call(["pip", "install", "huggingface_hub", "pyarrow"])
        from huggingface_hub import hf_hub_download, HfApi

    print("✅ All imports ready")
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
def _(Path, json, mo, torch):
    # ═══════════════════════════════════════════════════════════════
    # 🎛️ Control Panel — Configure, Train, Upload
    # ═══════════════════════════════════════════════════════════════

    CHECKPOINT_DIR = Path("/workspace/dt_checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pt"
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"

    # ── Checkpoint status ──
    _ckpt_exists = LATEST_CHECKPOINT.exists()
    _best_exists = BEST_MODEL_PATH.exists()
    if _ckpt_exists:
        _ckpt = torch.load(LATEST_CHECKPOINT, map_location="cpu")
        _ckpt_epoch = _ckpt.get("epoch", "?")
        _ckpt_val = _ckpt.get("best_val_loss", "?")
        _ckpt_info = f"✅ Checkpoint found: epoch={_ckpt_epoch}, best_val={_ckpt_val:.6f}" if isinstance(_ckpt_val, float) else f"✅ Checkpoint found: epoch={_ckpt_epoch}"
    else:
        _ckpt_info = "🆕 No checkpoint — will start fresh"

    # ── Config input mode ──
    use_json_config = mo.ui.checkbox(label="📋 Use JSON config instead of individual controls", value=False)

    # ── Individual controls (used when JSON mode is OFF) ──
    use_pilot = mo.ui.checkbox(label="Pilot mode (12 episodes, fast)", value=True)
    fresh_start = mo.ui.checkbox(label="🧹 Fresh start (delete checkpoint)", value=False)

    # Architecture
    n_block = mo.ui.number(value=8, label="Blocks", full_width=True)
    h_dim = mo.ui.number(value=384, label="Hidden dim", full_width=True)
    n_heads = mo.ui.number(value=8, label="Heads", full_width=True)
    context_len = mo.ui.number(value=180, label="Context len", full_width=True)
    drop_p = mo.ui.number(value=0.15, label="Dropout", full_width=True)

    # Training hyperparams
    batch_size = mo.ui.number(value=128, label="Batch size", full_width=True)
    epochs_per_session = mo.ui.number(value=3, label="Epochs/session", full_width=True)
    lr = mo.ui.number(value=3e-5, label="Learning rate", full_width=True)

    # Loss weights
    action_loss_weight = mo.ui.number(value=0.999, label="Action loss weight", full_width=True)
    state_loss_weight = mo.ui.number(value=0.002, label="State loss weight", full_width=True)
    return_loss_weight = mo.ui.number(value=0.0001, label="Return loss weight", full_width=True)

    # ── JSON config (used when JSON mode is ON) ──
    _DEFAULT_JSON = json.dumps({
        "state_dim": 18,
        "act_dim": 9,
        "n_block": 8,
        "h_dim": 384,
        "n_heads": 8,
        "context_len": 180,
        "drop_p": 0.15,
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
    }, indent=2)

    json_config = mo.ui.text_area(
        value=_DEFAULT_JSON,
        label="📋 Paste full JSON config here",
        full_width=True,
    )

    # ── Actions ──
    train_btn = mo.ui.run_button(label="🚀 Start Training", kind="success", tooltip="Apply config and start/resume training")
    upload_btn = mo.ui.run_button(label="🤗 Upload to HuggingFace", kind="info", tooltip="Upload best checkpoint to HF Hub")

    # ── Layout: Actions come FIRST so they're always visible ──
    _individual_section = mo.vstack([
        mo.md("### 🏗️ Architecture"),
        mo.hstack([n_block, h_dim, n_heads], justify="start", gap=1),
        mo.hstack([context_len, drop_p], justify="start", gap=1),
        mo.md("### 🎯 Training"),
        mo.hstack([batch_size, epochs_per_session], justify="start", gap=1),
        mo.hstack([lr], justify="start", gap=1),
        mo.md("### ⚖️ Loss Weights"),
        mo.hstack([action_loss_weight, state_loss_weight, return_loss_weight], justify="start", gap=1),
    ], gap=0.5)

    _json_section = mo.vstack([
        mo.md("### 📋 JSON Config"),
        mo.md("Paste a complete config JSON. All fields above are ignored when this is enabled."),
        json_config,
    ], gap=0.5)

    mo.vstack([
        mo.md(f"""
    ## 🎛️ AEMO Decision Transformer — Control Panel

    **Status**: {_ckpt_info}
    """),
        mo.hstack([use_pilot, fresh_start, use_json_config], justify="start", gap=2),
        _individual_section,
        # Actions are right here — always visible above JSON section
        mo.md("### 🎮 Actions"),
        mo.hstack([train_btn, upload_btn], justify="start", gap=2),
        _json_section,
        mo.md("---"),
    ])
    return (
        BEST_MODEL_PATH,
        CHECKPOINT_DIR,
        LATEST_CHECKPOINT,
        action_loss_weight,
        batch_size,
        context_len,
        drop_p,
        epochs_per_session,
        fresh_start,
        h_dim,
        json_config,
        lr,
        n_block,
        n_heads,
        return_loss_weight,
        state_loss_weight,
        train_btn,
        upload_btn,
        use_json_config,
        use_pilot,
    )


@app.cell
def _(mo, os):
    # ═══════════════════════════════════════════════════════════════
    # 🔑 HuggingFace Token (Required for Model Upload)
    # ═══════════════════════════════════════════════════════════════

    _hf_token_from_env = os.environ.get("HF_TOKEN", "")

    hf_token_input = mo.ui.text(
        value=_hf_token_from_env,
        label="🔑 HuggingFace Token",
        full_width=True,
    )

    mo.vstack([
        mo.md(f"""
    ## 🔑 HuggingFace Token

    Provide your token to enable model upload. Set `HF_TOKEN` env var for better security.
    > Token is **only** used when you click the upload button.
    """),
        hf_token_input,
    ])
    return (hf_token_input,)


@app.cell
def _(CHECKPOINT_DIR):
    # 🧹 Checkpoint directory setup

    # Ensure checkpoint directory exists
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Checkpoint directory ready at {CHECKPOINT_DIR}")

    # Signal variable — BYtC depends on this to enforce execution order
    ckpt_ready = True
    _ = f"Checkpoint directory: {CHECKPOINT_DIR}"
    return (ckpt_ready,)


@app.cell
def _(hf_hub_download, pl, use_pilot):
    # ═══════════════════════════════════════════════════════════════
    # Download AEMO FCAS Dataset from HuggingFace
    # ═══════════════════════════════════════════════════════════════

    REPO_ID = "mrvictoru/AEMO_simulated_trade"

    if use_pilot.value:
        # Try downloading pilot subset first; if missing, create from full dataset
        try:
            train_path = hf_hub_download(repo_id=REPO_ID, filename="pilot/train.parquet", repo_type="dataset", force_download=True)
            val_path = hf_hub_download(repo_id=REPO_ID, filename="pilot/val.parquet", repo_type="dataset", force_download=True)
            df = pl.read_parquet(train_path)
            val_df = pl.read_parquet(val_path)
            print(f"📦 Pilot dataset loaded: {df.height:,} train rows, {val_df.height:,} val rows")
        except Exception as e:
            print(f"⚠️  Pilot files not found on HF (expected). Creating pilot from full dataset...")
            # Download full dataset and create a small pilot split
            path = hf_hub_download(repo_id=REPO_ID, filename="aemo_fcas_dataset.parquet", repo_type="dataset", force_download=True)
            full_df = pl.read_parquet(path)
            print(f"📦 Full dataset loaded: {full_df.height:,} rows, {full_df['episode_id'].n_unique()} episodes")

            # Create pilot: use first 8 episodes for train, next 4 for val
            episode_ids = full_df.select(pl.col("episode_id").unique()).to_series().to_list()
            train_eps = episode_ids[:8]
            val_eps = episode_ids[8:12]
            df = full_df.filter(pl.col("episode_id").is_in(train_eps))
            val_df = full_df.filter(pl.col("episode_id").is_in(val_eps))
            print(f"📦 Pilot created: {df.height:,} train rows ({len(train_eps)} eps), {val_df.height:,} val rows ({len(val_eps)} eps)")
    else:
        # Download full FCAS dataset (force fresh download)
        path = hf_hub_download(repo_id=REPO_ID, filename="aemo_fcas_dataset.parquet", repo_type="dataset", force_download=True)
        df = pl.read_parquet(path)
        val_df = None
        print(f"📦 Full dataset loaded: {df.height:,} rows")

    print(f"  Columns: {df.columns}")
    print(f"  Episodes: {df['episode_id'].n_unique()}")
    return df, val_df


@app.cell
def _(df, pl):
    # 📊 Quick data profiling for hyperparameter validation

    print("=" * 60)
    print("📊 Data Profile for Hyperparameter Validation")
    print("=" * 60)

    # Episode stats
    episode_stats = df.group_by("episode_id").agg(
        pl.col("step").len().alias("n_steps"),
        pl.col("reward").mean().alias("avg_reward"),
        pl.col("reward").sum().alias("total_reward"),
        pl.col("reward").min().alias("min_reward"),
        pl.col("reward").max().alias("max_reward"),
    )

    print(f"\n📈 Episode Statistics ({len(episode_stats)} episodes):")
    print(f"  Episodes: {len(episode_stats)}")
    print(f"  Steps/episode - min: {episode_stats['n_steps'].min()}, max: {episode_stats['n_steps'].max()}, mean: {episode_stats['n_steps'].mean():.0f}")
    print(f"  Reward/episode - mean: {episode_stats['total_reward'].mean():.4f}, min: {episode_stats['total_reward'].min():.4f}, max: {episode_stats['total_reward'].max():.4f}")
    print(f"  Avg reward/step - mean: {episode_stats['avg_reward'].mean():.6f}, min: {episode_stats['avg_reward'].min():.6f}, max: {episode_stats['avg_reward'].max():.6f}")

    # Global reward stats
    print(f"\n💵 Global Reward Stats:")
    print(f"  Mean: {df['reward'].mean():.6f}")
    print(f"  Std:  {df['reward'].std():.6f}")
    print(f"  Min:  {df['reward'].min():.6f}")
    print(f"  Max:  {df['reward'].max():.6f}")
    print(f"  P1:   {df['reward'].quantile(0.01):.6f}")
    print(f"  P99:  {df['reward'].quantile(0.99):.6f}")

    # Source policy distribution
    print(f"\n🎯 Source Policy Distribution:")
    _policy_counts = df["source_policy"].value_counts()
    print(_policy_counts)

    # Episode ID is now integer - check range
    print(f"\n🔢 Episode ID range: {df['episode_id'].min()} to {df['episode_id'].max()}")

    # Check if episode_id overlaps with max_timestep (100000)
    print(f"⚠️  max_timestep=100000 and episode_id goes to {df['episode_id'].max()} — potential embedding collision!")
    print(f"   Embedding layers use timestep IDs, not episode IDs, so this should be fine.")
    return


@app.cell
def _(F, nn, torch):
    # ═══════════════════════════════════════════════════════════════
    # Decision Transformer Model Definition
    # ═══════════════════════════════════════════════════════════════

    class RMSNorm(nn.Module):
        """Root Mean Square Layer Normalization."""
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight


    class SwiGLU(nn.Module):
        """SwiGLU feed-forward network."""
        def __init__(self, dim, hidden_dim=None):
            super().__init__()
            hidden_dim = hidden_dim or dim * 4
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(hidden_dim, dim, bias=False)

        def forward(self, x):
            return self.w3(F.silu(self.w1(x)) * self.w2(x))


    class TransformerBlock(nn.Module):
        """Pre-norm transformer block with RoPE support."""
        def __init__(self, h_dim, n_heads, drop_p=0.1, use_rope=False, 
                     rope_base=10000, rope_max_position=None):
            super().__init__()
            self.ln1 = RMSNorm(h_dim)
            self.attn = nn.MultiheadAttention(
                h_dim, n_heads, dropout=drop_p, batch_first=True
            )
            self.ln2 = RMSNorm(h_dim)
            self.ffn = SwiGLU(h_dim)
            self.dropout = nn.Dropout(drop_p)
            self.use_rope = use_rope

        def forward(self, x, attn_mask=None):
            h = self.ln1(x)
            h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
            x = x + self.dropout(h)
            h = self.ln2(x)
            h = self.ffn(h)
            x = x + self.dropout(h)
            return x


    class DecisionTransformer(nn.Module):
        """Decision Transformer for AEMO battery trading.

        Predicts actions (energy dispatch + FCAS bids) conditioned on
        returns-to-go, states, and past actions.

        Action space (9-dim):
          - dim 0: energy dispatch in [-1, 1] (charge/discharge)
          - dims 1-8: FCAS contingency bids in [0, 1] (clamped at inference)
        """
        def __init__(self, state_dim, act_dim, n_block=8, h_dim=384, context_len=180,
                     n_heads=8, drop_p=0.15, max_timestep=100000,
                     use_rope=False, rope_base=10000, rope_max_position=None):
            super().__init__()
            self.state_dim = state_dim
            self.act_dim = act_dim
            self.context_len = context_len
            self.h_dim = h_dim

            self.embed_return = nn.Linear(1, h_dim)
            self.embed_state = nn.Linear(state_dim, h_dim)
            self.embed_action = nn.Linear(act_dim, h_dim)
            self.embed_timestep = nn.Embedding(max_timestep, h_dim)

            self.blocks = nn.ModuleList([
                TransformerBlock(h_dim, n_heads, drop_p, use_rope, rope_base, rope_max_position)
                for _ in range(n_block)
            ])
            self.ln_f = RMSNorm(h_dim)

            self.predict_return = nn.Linear(h_dim, 1)
            self.predict_state = nn.Linear(h_dim, state_dim)
            self.predict_action = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.GELU(),
                nn.Linear(h_dim, act_dim),
                nn.Tanh()  # Action output in [-1, 1]
            )

            self.return_scale = nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.drop_p = drop_p

            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
            B, T, _ = states.shape
            time_emb = self.embed_timestep(timesteps)
            state_emb = self.embed_state(states)
            action_emb = self.embed_action(actions)
            return_emb = self.embed_return(returns_to_go.unsqueeze(-1))

            stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
            x = stacked.permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
            x = x + time_emb.repeat_interleave(3, dim=1)

            if attention_mask is not None:
                attn_mask = attention_mask.repeat_interleave(3, dim=1)
                attn_mask = attn_mask.unsqueeze(1)
            else:
                attn_mask = None

            for block in self.blocks:
                x = block(x, attn_mask=attn_mask)

            x = self.ln_f(x)

            pred_mask = torch.zeros(3 * T, dtype=torch.bool)
            pred_mask[0::3] = True
            pred_mask[1::3] = True
            act_mask = torch.ones(3 * T, dtype=torch.bool)
            act_mask[2::3] = False

            x_pred = x[:, pred_mask]
            x_act = x[:, act_mask]

            return_preds = self.predict_return(x_pred[:, ::2])
            state_preds = self.predict_state(x_pred[:, 1::2])
            action_preds = self.predict_action(x_act[:, ::2])

            return action_preds, state_preds, return_preds

        def get_action(self, states, actions, returns_to_go, timesteps, attention_mask=None):
            """Inference: return predicted action for the last timestep.

            Clamps FCAS bids (dims 1-8) to [0, 1] while keeping energy dispatch
            (dim 0) in [-1, 1].
            """
            action_preds, _, _ = self.forward(
                states, actions, returns_to_go, timesteps, attention_mask
            )
            action = action_preds[:, -1]  # (B, act_dim)
            # Clamp FCAS bids (dims 1-8) to [0, 1], energy dispatch stays [-1, 1]
            action_fcas = torch.clamp(action[:, 1:], 0.0, 1.0)
            return torch.cat([action[:, :1], action_fcas], dim=-1)


    print("✅ DecisionTransformer model defined")
    return (DecisionTransformer,)


@app.cell
def _(np, pl, torch):
    # ═══════════════════════════════════════════════════════════════
    # TrajectoryDataset — Builds context windows for training
    # ═══════════════════════════════════════════════════════════════

    class TrajectoryDataset(torch.utils.data.Dataset):
        """Creates (state, action, RTG, timestep) context windows from episode data."""

        def __init__(self, df, context_length=180, state_dim=18, act_dim=9,
                     discount_factor=0.95, max_episodes=None, skip_short_episodes=True):
            self.context_length = context_length
            self.state_dim = state_dim
            self.act_dim = act_dim
            self.discount_factor = discount_factor

            # Group by episode_id and collect observations, actions, rewards
            episodes = df.group_by("episode_id").agg(
                pl.col("step").len().alias("n_steps"),
                pl.col("norm_observation").alias("obs"),
                pl.col("action").alias("act"),
                pl.col("reward").alias("rew"),
            )

            # Filter short episodes (need at least context_length * 3 for meaningful context)
            if skip_short_episodes:
                episodes = episodes.filter(pl.col("n_steps") >= context_length * 3)

            if max_episodes is not None:
                episodes = episodes.head(max_episodes)

            # Compute discounted returns-to-go for each episode
            all_states, all_actions, all_rtgs, all_timesteps = [], [], [], []
            n_skipped_dim = 0

            for row in episodes.iter_rows(named=True):
                obs_arr = np.array(row["obs"], dtype=np.float32)
                act_arr = np.array(row["act"], dtype=np.float32)
                rew_arr = np.array(row["rew"], dtype=np.float32)
                n = len(rew_arr)

                # Validate dimensions: skip episodes with mismatched dims
                if obs_arr.ndim != 2 or obs_arr.shape[1] != state_dim:
                    n_skipped_dim += 1
                    continue
                if act_arr.ndim != 2 or act_arr.shape[1] != act_dim:
                    n_skipped_dim += 1
                    continue

                # Calculate discounted returns-to-go (reverse cumulative sum)
                rtg = np.zeros(n, dtype=np.float32)
                running = 0.0
                for t in reversed(range(n)):
                    running = rew_arr[t] + discount_factor * running
                    rtg[t] = running

                # Create overlapping context windows
                stride = context_length // 2
                for i in range(0, n - context_length + 1, stride):
                    end = i + context_length
                    all_states.append(obs_arr[i:end])
                    all_actions.append(act_arr[i:end])
                    all_rtgs.append(rtg[i:end])
                    all_timesteps.append(np.arange(i, end, dtype=np.int64))

            if n_skipped_dim > 0:
                print(f"⚠️  Skipped {n_skipped_dim} episode(s) with mismatched dimensions")

            if len(all_states) == 0:
                raise RuntimeError(
                    f"No valid context windows created! Check that episodes have "
                    f"correct dimensions (state_dim={state_dim}, act_dim={act_dim})"
                )

            self.states = np.stack(all_states).astype(np.float32)
            self.actions = np.stack(all_actions).astype(np.float32)
            self.rtgs = np.stack(all_rtgs).astype(np.float32)
            self.timesteps = np.stack(all_timesteps).astype(np.int64)

            total_eps = len(episodes)
            used_eps = total_eps - n_skipped_dim
            print(f"📊 TrajectoryDataset: {len(self):,} contexts from {used_eps} episodes")

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.states[idx]),
                torch.tensor(self.actions[idx]),
                torch.tensor(self.rtgs[idx]),
                torch.tensor(self.timesteps[idx]),
            )


    print("✅ TrajectoryDataset defined")
    return (TrajectoryDataset,)


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
    return_loss_weight,
    state_loss_weight,
    use_json_config,
    use_pilot,
):
    # ═══════════════════════════════════════════════════════════════
    # 📋 Build TRAIN_CFG from Control Panel UI values or JSON
    # ═══════════════════════════════════════════════════════════════

    # ── Base config (always applied, acts as defaults) ──
    TRAIN_CFG = {
        "state_dim": 18,
        "act_dim": 9,
        "max_timestep": 100000,
        "use_rope": False,
        "discount_factor": 0.95,
        "val_split": 0.1,
        "return_scale": 2.0,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "checkpoint_every_n_batches": 500,
        "max_training_seconds": 11 * 3600,
    }

    # ── Choose input source ──
    if use_json_config.value:
        # ── JSON mode: parse from text area ──
        try:
            _parsed = json.loads(json_config.value)
            TRAIN_CFG.update(_parsed)
            _source = "📋 JSON config"
        except json.JSONDecodeError as _e:
            print(f"⚠️  Invalid JSON: {_e}")
            print("   Falling back to individual controls.\n")
            _source = "⚠️  Fallback (individual controls due to JSON error)"
            # Fall through to individual controls
            TRAIN_CFG.update({
                "n_block": n_block.value,
                "h_dim": h_dim.value,
                "n_heads": n_heads.value,
                "context_len": context_len.value,
                "drop_p": drop_p.value,
                "batch_size": batch_size.value,
                "lr": lr.value,
                "epochs_per_session": epochs_per_session.value,
                "action_loss_weight": action_loss_weight.value,
                "state_loss_weight": state_loss_weight.value,
                "return_loss_weight": return_loss_weight.value,
            })
    else:
        # ── Individual controls mode ──
        TRAIN_CFG.update({
            "n_block": n_block.value,
            "h_dim": h_dim.value,
            "n_heads": n_heads.value,
            "context_len": context_len.value,
            "drop_p": drop_p.value,
            "batch_size": batch_size.value,
            "lr": lr.value,
            "epochs_per_session": epochs_per_session.value,
            "action_loss_weight": action_loss_weight.value,
            "state_loss_weight": state_loss_weight.value,
            "return_loss_weight": return_loss_weight.value,
        })
        _source = "🎛️ Individual controls"

    # ── Validate critical fields ──
    _missing = [k for k in ("n_block", "h_dim", "n_heads", "context_len", "batch_size", "lr") if k not in TRAIN_CFG]
    if _missing:
        print(f"⚠️  Missing config fields: {_missing}. Using defaults.")
        TRAIN_CFG.update({
            "n_block": 8,
            "h_dim": 384,
            "n_heads": 8,
            "context_len": 180,
            "batch_size": 128,
            "lr": 3e-5,
            "epochs_per_session": 3,
            "action_loss_weight": 0.999,
            "state_loss_weight": 0.002,
            "return_loss_weight": 0.0001,
            "drop_p": 0.15,
        })

    # ── Display config summary ──
    print(f"📋 Active config source: {_source}")
    print(f"   Pilot mode: {'ON' if use_pilot.value else 'OFF'}")
    print(f"   Model: {TRAIN_CFG['n_block']} blocks, {TRAIN_CFG['h_dim']} dim, {TRAIN_CFG['n_heads']} heads")
    print(f"   Context: {TRAIN_CFG['context_len']} | Dropout: {TRAIN_CFG['drop_p']}")
    print(f"   Batch: {TRAIN_CFG['batch_size']} | LR: {TRAIN_CFG['lr']:.2e} | Epochs/session: {TRAIN_CFG['epochs_per_session']}")
    print(f"   Loss weights: action={TRAIN_CFG['action_loss_weight']} | state={TRAIN_CFG['state_loss_weight']} | return={TRAIN_CFG['return_loss_weight']}")
    return (TRAIN_CFG,)


@app.cell
def _(
    BEST_MODEL_PATH,
    DecisionTransformer,
    F,
    HfApi,
    LATEST_CHECKPOINT,
    Path,
    TRAIN_CFG,
    os,
    time,
    torch,
):
    # ═══════════════════════════════════════════════════════════════
    # Training Functions — Checkpoint, Resume, Training Loop
    # ═══════════════════════════════════════════════════════════════

    def load_or_create_model(cfg, device):
        """Load from checkpoint if exists, otherwise create new model."""
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
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        start_epoch = 0
        global_step = 0
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")

        # Resume from checkpoint if available
        if LATEST_CHECKPOINT.exists():
            print(f"🔄 Resuming from checkpoint: {LATEST_CHECKPOINT}")
            checkpoint = torch.load(LATEST_CHECKPOINT, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint.get("global_step", 0)
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            # Restore scaler state if saved
            scaler_state = checkpoint.get("scaler_state_dict", None)
            print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_val={best_val_loss:.6f}")
            return model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, scaler_state

        return model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, None


    def save_checkpoint(model, optimizer, scheduler, epoch, global_step, 
                        train_losses, val_losses, best_val_loss, scaler=None):
        """Save checkpoint to persistent storage."""
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "act_dim": TRAIN_CFG["act_dim"],
            "state_dim": TRAIN_CFG["state_dim"],
            "context_len": TRAIN_CFG["context_len"],
            "return_scale": TRAIN_CFG["return_scale"],
            "timestamp": time.time(),
        }
        if scaler is not None:
            ckpt["scaler_state_dict"] = scaler.state_dict()
        torch.save(ckpt, LATEST_CHECKPOINT)
        print(f"💾 Checkpoint saved at epoch {epoch}, step {global_step}")


    def upload_best_model_to_hf(model, best_val_loss, hf_token=None):
        """Upload the best model checkpoint to HuggingFace Model Hub.

        Args:
            model: The trained model
            best_val_loss: Best validation loss achieved
            hf_token: HuggingFace API token. If None, falls back to HF_TOKEN env var.
                      Token is only used when you explicitly call this function.
        """
        # Only use explicitly passed token or env var (NOT files on disk)
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")

        if hf_token is None:
            print("⚠️  No HF_TOKEN provided. Model saved locally at /workspace/best_model.pt")
            print("   To upload: paste your HuggingFace token in the 🔑 token cell, or set HF_TOKEN env var.")
            print("   Then run the 'Upload to HuggingFace' cell again.")
            return

        repo_id = "mrvictoru/energydecision-dt"

        # Save model temporarily to /workspace/ (persistent storage)
        tmp_path = Path("/workspace/best_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "val_loss": best_val_loss,
            "config": {k: v for k, v in TRAIN_CFG.items()
                       if k in ("state_dim", "act_dim", "n_block", "h_dim", "n_heads",
                                "context_len", "drop_p", "return_scale", "discount_factor")},
        }, tmp_path)
        print(f"💾 Model saved to {tmp_path} ({tmp_path.stat().st_size / 1e6:.1f} MB)")

        try:
            api = HfApi()

            # Check if repo exists, create if not
            try:
                api.repo_info(repo_id=repo_id, repo_type="model", token=hf_token)
                print(f"📂 Repo {repo_id} already exists")
            except Exception:
                print(f"🆕 Creating HF repo {repo_id}...")
                try:
                    api.create_repo(repo_id=repo_id, repo_type="model", token=hf_token, exist_ok=True)
                    print(f"✅ Repo created (or already exists)")
                except Exception as create_err:
                    print(f"⚠️  Could not create repo: {create_err}")
                    print(f"   Model saved locally at {tmp_path}")
                    print(f"   Please create the repo manually at https://huggingface.co/new")
                    return

            # Upload the file
            print(f"📤 Uploading to {repo_id}/aemo_dt_fcas_model.pt...")
            api.upload_file(
                path_or_fileobj=str(tmp_path),
                path_in_repo="aemo_dt_fcas_model.pt",
                repo_id=repo_id,
                repo_type="model",
                token=hf_token,
            )
            print(f"✅ Uploaded best model to HF: {repo_id}/aemo_dt_fcas_model.pt")
            print(f"   Best val_loss: {best_val_loss:.6f}")

            # Also upload best_model.pt from dt_checkpoints if it exists
            if BEST_MODEL_PATH.exists():
                print(f"📤 Also uploading aemo_dt_fcas_best_checkpoint.pt...")
                api.upload_file(
                    path_or_fileobj=str(BEST_MODEL_PATH),
                    path_in_repo="aemo_dt_fcas_best_checkpoint.pt",
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_token,
                )
                print(f"✅ Done!")
        except Exception as e:
            print(f"⚠️  Upload failed: {e}")
            print(f"   Model saved locally at {tmp_path} — you can upload manually.")
            import traceback
            traceback.print_exc()


    def train_epoch(model, optimizer, scheduler, dataloader, epoch, cfg, device, session_start,
                    train_losses, val_losses, best_val_loss, scaler=None):
        """Train one epoch. Uses AMP autocast if scaler is provided. Returns (avg_loss, num_batches, time_limit_hit)."""
        model.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_state_loss = 0.0
        total_return_loss = 0.0
        start_time = time.time()
        use_amp = scaler is not None

        for batch_idx, (states, actions, rtgs, timesteps) in enumerate(dataloader):
            # Check time limit
            if time.time() - session_start > cfg["max_training_seconds"]:
                print("⏰ Approaching 12h limit — finishing batch and saving checkpoint...")
                n = max(1, batch_idx)
                return total_loss / n, batch_idx, True

            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)

            # Normalize returns
            rtgs = rtgs / cfg["return_scale"]

            # Forward pass with AMP autocast
            with torch.cuda.amp.autocast(enabled=use_amp):
                action_preds, state_preds, return_preds = model(
                    states, actions, rtgs, timesteps
                )

                # Losses
                action_loss = F.mse_loss(action_preds, actions, reduction="mean")
                state_loss = F.mse_loss(state_preds, states, reduction="mean")
                return_loss = F.mse_loss(return_preds.squeeze(-1), rtgs, reduction="mean")

                loss = (cfg["action_loss_weight"] * action_loss +
                        cfg["state_loss_weight"] * state_loss +
                        cfg["return_loss_weight"] * return_loss)

            optimizer.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
                optimizer.step()

            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_state_loss += state_loss.item()
            total_return_loss += return_loss.item()

            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx:5d} | Loss: {loss.item():.6f} | "
                      f"Action: {action_loss.item():.6f} | State: {state_loss.item():.6f} | "
                      f"Return: {return_loss.item():.6f} | {elapsed:.0f}s")

            # Periodic checkpoint
            if batch_idx > 0 and batch_idx % cfg["checkpoint_every_n_batches"] == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, batch_idx,
                              train_losses, val_losses, best_val_loss, scaler)

        n = max(1, batch_idx)
        return total_loss / n, batch_idx, False


    def validate_epoch(model, dataloader, cfg, device, scaler=None):
        """Run validation and return average loss. Uses AMP autocast if scaler is provided."""
        model.eval()
        val_loss = 0.0
        use_amp = scaler is not None
        with torch.no_grad():
            for states, actions, rtgs, timesteps in dataloader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)
                rtgs = rtgs / cfg["return_scale"]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    action_preds, state_preds, return_preds = model(
                        states, actions, rtgs, timesteps
                    )
                    val_loss += F.mse_loss(action_preds, actions).item()
        val_loss /= max(1, len(dataloader))
        return val_loss


    print("✅ Training functions defined (AMP enabled)")
    return (
        load_or_create_model,
        save_checkpoint,
        train_epoch,
        upload_best_model_to_hf,
        validate_epoch,
    )


@app.cell
def _(
    BEST_MODEL_PATH,
    LATEST_CHECKPOINT,
    TRAIN_CFG,
    TrajectoryDataset,
    ckpt_ready,
    df,
    fresh_start,
    load_or_create_model,
    mo,
    save_checkpoint,
    time,
    torch,
    train_btn,
    train_epoch,
    use_pilot,
    val_df,
    validate_epoch,
):
    # ═══════════════════════════════════════════════════════════════
    # Main Training Loop (gated by Start Training button)
    # ═══════════════════════════════════════════════════════════════

    # ── Always define placeholder variables for downstream dashboard cells ──
    session_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    global_step = 0
    USE_PILOT = use_pilot.value

    # ── Gate: only proceed when the Start Training button was clicked ──
    if not train_btn.value:
        mo.stop(True, mo.md("""> 👆 Click the **🚀 Start Training** button above to begin or resume training. Adjust hyperparameters in the Control Panel first."""))

    # ═══════════════════════════════════════════════════════════════
    # 🎯 TRAINING HAS STARTED
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("🎯  TRAINING SESSION STARTED")
    print(f"⏰  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻  Device: {device}")
    print(f"📋  Pilot: {'ON' if use_pilot.value else 'OFF'}")
    print(f"📐  Model: {TRAIN_CFG['n_block']} blocks, {TRAIN_CFG['h_dim']} dim, {TRAIN_CFG['n_heads']} heads")
    print(f"📦  Batch: {TRAIN_CFG['batch_size']}  |  LR: {TRAIN_CFG['lr']:.2e}  |  Epochs: {TRAIN_CFG['epochs_per_session']}")
    print("=" * 60)

    # ── Fresh start: delete existing checkpoint if requested ──
    if fresh_start.value:
        for _p in [LATEST_CHECKPOINT, BEST_MODEL_PATH]:
            if _p.exists():
                _p.unlink()
        print("🧹 Fresh start: deleted existing checkpoint(s)")
        print("   Uncheck \"Fresh start\" and click \"Start Training\" again to continue from this new checkpoint.\n")

    # ── Dependency: enforce checkpoint directory setup runs first ──
    _ = ckpt_ready

    # ── Device setup ──
    print(f"\n🔧 Initializing...")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    session_start = time.time()
    _is_pilot = use_pilot.value

    # ── Create dataset ──
    print(f"\n📊 Building datasets (context_len={TRAIN_CFG['context_len']})...")
    if _is_pilot:
        train_ds = TrajectoryDataset(
            df, 
            context_length=TRAIN_CFG["context_len"],
            state_dim=TRAIN_CFG["state_dim"],
            act_dim=TRAIN_CFG["act_dim"],
            discount_factor=TRAIN_CFG["discount_factor"],
            max_episodes=None,
            skip_short_episodes=True,
        )
        val_ds = TrajectoryDataset(
            val_df,
            context_length=TRAIN_CFG["context_len"],
            state_dim=TRAIN_CFG["state_dim"],
            act_dim=TRAIN_CFG["act_dim"],
            discount_factor=TRAIN_CFG["discount_factor"],
            max_episodes=None,
            skip_short_episodes=True,
        )
    else:
        full_ds = TrajectoryDataset(
            df,
            context_length=TRAIN_CFG["context_len"],
            state_dim=TRAIN_CFG["state_dim"],
            act_dim=TRAIN_CFG["act_dim"],
            discount_factor=TRAIN_CFG["discount_factor"],
            max_episodes=None,
            skip_short_episodes=True,
        )
        _n_val = max(1, int(len(full_ds) * TRAIN_CFG["val_split"]))
        _n_train = len(full_ds) - _n_val
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [_n_train, _n_val])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False, num_workers=0
    )
    print(f"   ✅ Train: {len(train_loader)} batches  |  Val: {len(val_loader)} batches")

    # ── GradScaler for AMP (only on CUDA) ──
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Load or create model ──
    print(f"\n🧠 Loading/creating model...")
    model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, scaler_state = \
        load_or_create_model(TRAIN_CFG, device)

    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
        print(f"   🔄 AMP scaler resumed (growth_factor={scaler.get_scale():.0f})")

    print(f"   📍 Starting at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    if scaler:
        print(f"   ⚡ AMP mixed precision enabled")

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"🏋️  TRAINING LOOP  ({TRAIN_CFG['epochs_per_session']} epochs)")
    print(f"{'='*60}")
    time_limit_hit = False
    for epoch in range(start_epoch, TRAIN_CFG["epochs_per_session"] + start_epoch):
        print(f"\n{'─'*60}")
        print(f"📚 EPOCH {epoch}")
        print(f"{'─'*60}")

        avg_loss, steps, time_limit_hit = train_epoch(
            model, optimizer, scheduler, train_loader, epoch, TRAIN_CFG, device, session_start,
            train_losses, val_losses, best_val_loss, scaler
        )
        train_losses.append(avg_loss)

        val_loss = validate_epoch(model, val_loader, TRAIN_CFG, device, scaler)
        val_losses.append(val_loss)

        print(f"\n📈  Epoch {epoch} complete: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆  New best val_loss: {best_val_loss:.6f}")

        save_checkpoint(model, optimizer, scheduler, epoch, global_step + steps,
                        train_losses, val_losses, best_val_loss, scaler)
        global_step += steps

        if time_limit_hit:
            print("⏰  Time limit reached — session will auto-resume from checkpoint.")
            break

    elapsed = time.time() - session_start
    print(f"\n{'='*60}")
    print(f"✅  TRAINING SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"   Best val_loss: {best_val_loss:.6f}")
    print(f"   Duration:      {elapsed / 3600:.2f} hours ({elapsed / 60:.1f} minutes)")
    print(f"   Total steps:   {global_step}")
    print(f"{'='*60}")

    # Export mode flag for downstream dashboard cells
    USE_PILOT = _is_pilot
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
    # ═══════════════════════════════════════════════════════════════
    # Progress Dashboard
    # ═══════════════════════════════════════════════════════════════

    # Training summary dashboard
    mo.md(f"""
    # 🏋️ AEMO Decision Transformer Training Dashboard

    **Session**: `{session_start:.0f}` | **Device**: `{device}` | **Pilot**: `{USE_PILOT}`
    """)
    return


@app.cell
def _(mo, np, plt, train_losses, val_losses):
    # Loss curves
    if train_losses and val_losses:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves over epochs
        ax1.plot(train_losses, 'b-o', label='Train Loss', markersize=6)
        ax1.plot(val_losses, 'r-s', label='Val Loss', markersize=6)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Current loss bar chart
        epochs = list(range(len(train_losses)))
        x_pos = np.arange(len(epochs))
        width = 0.35
        ax2.bar(x_pos - width/2, train_losses, width, label='Train', color='steelblue')
        ax2.bar(x_pos + width/2, val_losses, width, label='Val', color='coral')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss per Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        _output = mo.mpl.interactive(fig)
    else:
        _output = mo.md("""
        > ⏳ Training hasn't completed an epoch yet. Loss curves will appear here after the first epoch.
        """)

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
    # ═══════════════════════════════════════════════════════════════
    # Training Summary
    # ═══════════════════════════════════════════════════════════════

    _elapsed_hrs = (time.time() - session_start) / 3600

    # Safe conversion for display (handle empty loss lists)
    _train_loss_str = f"{train_losses[-1]:.6f}" if train_losses else "N/A"
    _val_loss_str = f"{val_losses[-1]:.6f}" if val_losses else "N/A"

    mo.md(f"""
    ## 📊 Training Session Summary

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
    | **Model upload** | Use the upload cell below ⬇️ |

    ### Configuration
    - **Model**: DecisionTransformer with {TRAIN_CFG["n_block"]} blocks, {TRAIN_CFG["h_dim"]} hidden dim, {TRAIN_CFG["n_heads"]} heads
    - **Context length**: {TRAIN_CFG["context_len"]}
    - **Batch size**: {TRAIN_CFG["batch_size"]}
    - **Learning rate**: {TRAIN_CFG["lr"]}
    - **Action loss weight**: {TRAIN_CFG["action_loss_weight"]}

    ### GPU Memory
    """)

    if torch.cuda.is_available():
        _allocated = torch.cuda.max_memory_allocated() / 1e9
        _reserved = torch.cuda.max_memory_reserved() / 1e9
        mo.output.append(mo.md(f"""
    | GPU Stat | Value |
    |----------|-------|
    | **Max memory allocated** | `{_allocated:.2f} GB` |
    | **Max memory reserved** | `{_reserved:.2f} GB` |
    | **Total VRAM** | `{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB` |
    """))
    return


@app.cell
def _(mo):
    # ═══════════════════════════════════════════════════════════════
    # 🤗 Upload Best Model to HuggingFace (User-Triggered)
    # ═══════════════════════════════════════════════════════════════

    mo.vstack([
        mo.md("""
    ## 🤗 Upload Best Model to HuggingFace

    Click the **🤗 Upload to HuggingFace** button in the **Control Panel** above after training completes.

    The model will be published to **[mrvictoru/energydecision-dt](https://huggingface.co/mrvictoru/energydecision-dt)**.

    > ⚠️ **Prerequisites**:
    > 1. Provide your HF token in the **🔑 HuggingFace Token** cell above
    > 2. Training must have completed (best_model.pt must exist)
    """),
    ])
    return


@app.cell
def _(
    BEST_MODEL_PATH,
    DecisionTransformer,
    LATEST_CHECKPOINT,
    TRAIN_CFG,
    hf_token_input,
    mo,
    torch,
    upload_best_model_to_hf,
    upload_btn,
):
    # ═══════════════════════════════════════════════════════════════
    # Upload Execution (triggered by button click)
    # ═══════════════════════════════════════════════════════════════

    # Only proceed when button is clicked
    if not upload_btn.value:
        mo.stop(True, "👆 Click the upload button above to publish your trained model to HuggingFace.")

    # --- Button clicked: perform upload ---

    print("🚀 Starting HuggingFace upload...")
    print(f"   Token: {'***' + hf_token_input.value[-4:] if hf_token_input.value else 'EMPTY'}")
    print(f"   Model path: {BEST_MODEL_PATH}")
    print(f"   Model exists: {BEST_MODEL_PATH.exists()}")

    # ── Build the model architecture (must match training config) ──
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
        )

    if BEST_MODEL_PATH.exists():
        # BEST_MODEL_PATH is saved as raw torch.save(model.state_dict(), ...) — no wrapping
        print("📂 Loading best model weights...")
        _state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu")
        _model_upload = _build_upload_model()
        _model_upload.load_state_dict(_state_dict)
        _val_loss_upload = float("inf")  # raw state dict doesn't store val_loss
        upload_best_model_to_hf(_model_upload, _val_loss_upload, hf_token=hf_token_input.value or None)

    elif LATEST_CHECKPOINT.exists():
        # LATEST_CHECKPOINT is saved as a wrapped dict with "model_state_dict" key
        print("📂 No best_model.pt found — loading from latest checkpoint...")
        _ckpt = torch.load(LATEST_CHECKPOINT, map_location="cpu")
        _model_upload = _build_upload_model()
        _model_upload.load_state_dict(_ckpt["model_state_dict"])
        _val_loss_upload = _ckpt.get("best_val_loss", float("inf"))
        upload_best_model_to_hf(_model_upload, _val_loss_upload, hf_token=hf_token_input.value or None)

    else:
        print("❌ No trained model found. Run the training loop first!")
        print(f"   Expected at: {BEST_MODEL_PATH} or {LATEST_CHECKPOINT}")

    # Suppress expression display
    None
    return


@app.cell
def _(mo):
    # ═══════════════════════════════════════════════════════════════
    # 📚 User Operations Guide
    # ═══════════════════════════════════════════════════════════════

    mo.md(f"""
    ## 📚 User Operations Guide

    ### ▶️ How to Run Training

    Simply run all cells **in order**. The notebook is fully self-contained and will:

    1. Install any missing packages
    2. Download the FCAS dataset (pilot by default — 8 train + 4 val episodes)
    3. Create the Decision Transformer model
    4. Build trajectory datasets with context windows
    5. **Wait for you to click "Start Training"** — training won't start automatically

    > **Tip**: Make sure your HuggingFace token is saved at `/workspace/hf_token.txt`
    > **before** training starts, so it can be auto-detected for model upload later.

    ---

    ### 🎛️ Configuring Hyperparameters

    You have **two ways** to configure training:

    #### 1️⃣ Individual Controls (default)
    Use the number inputs in the Control Panel for precise values:
    - **Architecture**: Blocks, Hidden dim, Heads, Context length, Dropout
    - **Training**: Batch size, Epochs/session, Learning rate, Action loss weight
    - Toggle **Pilot mode** for fast testing with 12 episodes
    - Check **Fresh start** to delete checkpoints before training

    #### 2️⃣ JSON Config (advanced)
    Check **"Use JSON config instead of individual controls"** to paste a complete JSON config.
    This is useful for batch-updating settings or reusing configs from previous runs.
    All individual controls are ignored when JSON mode is active.

    ---

    ### 🔄 How to Resume Training After Session Timeout

    The notebook **auto-resumes** from the last checkpoint at `/workspace/dt_checkpoints/latest_checkpoint.pt`.
    Just re-run all cells — the training loop will detect the checkpoint and pick up where it left off.

    ---

    ### 🚀 How to Train on the Full Dataset (2,425 episodes)

    1. **Set** "Pilot mode" = OFF in the Control Panel
    2. **Run all cells** — the training loop will download the full 78.4M-row dataset
    3. Expect **~6-10 hours per epoch** (requires multiple 12h sessions)
    4. The notebook will auto-save checkpoints every 500 batches and cleanly stop before the 12h limit
    5. On the next session, it auto-resumes — just re-run all cells again

    > ⚠️ **Memory note**: The full dataset is large (~78.4M rows). Ensure your MoLab session has
    > sufficient RAM (32 GB should be enough for the full dataset).

    ---

    ### 🤗 How to Upload Model to HuggingFace

    1. **After training completes**, provide your HF Token in the token cell
    2. Click the **"🤗 Upload to HuggingFace"** button in the Control Panel
    3. The model will be published to **[mrvictoru/energydecision-dt](https://huggingface.co/mrvictoru/energydecision-dt)**

    > **If the HF model page doesn't exist yet**: That's okay! Go to [huggingface.co/new](https://huggingface.co/new)
    > and create a new model repository named `mrvictoru/energydecision-dt` (or edit the `repo_id` in the code).
    > Then click the upload button.

    ---

    ### 📁 Checkpoint File Structure

    | File | Location | Purpose |
    |------|----------|---------|
    | `latest_checkpoint.pt` | `/workspace/dt_checkpoints/` | Full training state (model + optimizer + scheduler + losses) |
    | `best_model.pt` | `/workspace/dt_checkpoints/` | Best model weights only (updated each time val_loss improves) |

    ---

    ### 🔧 Hyperparameter Tuning

    Key knobs:
    - **`lr`**: Learning rate (default: 3e-5) — try 1e-5 for fine-tuning
    - **`batch_size`**: Currently 128 — can increase to 256 on 96GB VRAM
    - **`action_loss_weight`**: Default 0.999 — tune for balancing loss components
    - **`drop_p`**: Dropout (default: 0.15) — increase for regularization on full data
    - **`context_len`**: Sequence length (default: 180) — shorter = faster but less context
    - **`n_block`/`h_dim`**: Model capacity — bigger = more expressive but slower
    """)
    return


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ---
    license: mit
    library_name: pytorch
    tags:
    - decision-transformer
    - energy-trading
    - battery-storage
    - fcas
    - aemo
    - reinforcement-learning
    - offline-rl
    ---

    # EnergyDecision-DT: Decision Transformer for AEMO FCAS Battery Trading

    ## Model Description

    **EnergyDecision-DT** is a **Decision Transformer** model trained on simulated battery dispatch data from the Australian Energy Market Operator (AEMO) Frequency Control Ancillary Services (FCAS) market. It models optimal battery dispatch as a sequence prediction problem, conditioning on returns-to-go, observed states, and past actions to predict the next action.

    The model learns to **dispatch battery energy storage** (charge/discharge) and **bid into 8 FCAS contingency markets** simultaneously, making it a 9-dimensional action space controller.

    ### Key Features

    - **Action Space (9-dim)**:
      - Dim 0: Energy dispatch in $$[-1, 1]$$ (charge/discharge)
      - Dims 1-8: FCAS contingency bids in $$[0, 1]$$
    - **State Space (18-dim)**: Normalized market observations including prices, demand, renewables penetration, and battery state-of-charge
    - **Context Length**: 180 timesteps (looks back ~15 hours of history)
    - **Training Approach**: Offline reinforcement learning via return-conditioned sequence modeling

    ## Intended Use

    This model is intended for:
    - **Research** into offline RL for energy markets
    - **Simulation** of battery trading strategies in the AEMO FCAS market
    - **Baseline** for comparing decision transformer approaches against traditional RL

    It is **not intended for live trading** without further validation, risk management, and regulatory compliance.

    ## Training Data

    - **Source**: [AEMO simulated trade dataset](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade)
    - **Size**: 75,945,600 rows across **2,405 episodes**
    - **Format**: Parquet file with 5 columns

    ### Dataset Schema

    | Column | Type | Description |
    |--------|------|-------------|
    | `episode_id` | `str` | Unique episode identifier (e.g. `nsw1_2021_2023_a2c_long_large_ep000` — encodes region, date range, RL algorithm, and episode number) |
    | `step` | `int64` | Timestep within the episode (0-indexed, 5-minute intervals) |
    | `norm_observation` | `list[f64]` (18-dim) | Normalized market observations including prices, demand, renewables penetration, and battery state-of-charge |
    | `action` | `list[f64]` (9-dim) | Battery dispatch action: energy charge/discharge (dim 0) + 8 FCAS contingency bids (dims 1-8) |
    | `reward` | `float64` | Scalar reward from the simulated market interaction |

    ### Observation Space (18-dim)

    The normalized observation vector includes:
    - Market prices and demand metrics
    - Renewables penetration indicators
    - Battery state-of-charge (SoC)
    - Time-of-day and seasonal features
    - Historical bid-ask spreads and clearance rates

    ### Episode Structure

    Each episode represents a simulated battery trading trajectory over a continuous period, with actions taken at 5-minute dispatch intervals. Episodes were generated using an A2C RL agent interacting with a market simulator calibrated to AEMO NEM data (NSW region, 2021-2023).

    ### Preprocessing

    - Observations normalized to zero mean, unit variance per feature
    - Returns-to-go computed with discount factor $$\\\\gamma = 0.95$$
    - Rewards used as-is from the simulator
    - No additional augmentation or filtering applied

    ## Model Architecture

    ```
    DecisionTransformer(
      (embed_return): Linear(1 -> 384)
      (embed_state): Linear(18 -> 384)
      (embed_action): Linear(9 -> 384)
      (embed_timestep): Embedding(100000 -> 384)
      (blocks): 8x TransformerBlock(
          (ln1): RMSNorm(384)
          (attn): MultiheadAttention(384, 8 heads)
          (ln2): RMSNorm(384)
          (ffn): SwiGLU(384 -> 1536 -> 384)
          (dropout): Dropout(p=0.15)
        )
      (ln_f): RMSNorm(384)
      (predict_action): Linear(384 -> 384) -> GELU -> Linear(384 -> 9) -> Tanh
      (predict_state): Linear(384 -> 18)
      (predict_return): Linear(384 -> 1)
    )
    ```

    ### Hyperparameters

    | Parameter | Value |
    |-----------|-------|
    | Blocks | 8 |
    | Hidden dim | 384 |
    | Attention heads | 8 |
    | Context length | 180 |
    | Dropout | 0.15 |
    | State dim | 18 |
    | Action dim | 9 |
    | Discount factor | 0.95 |
    | Return scale | 2.0 |
    | Action loss weight | 0.999 |
    | State loss weight | 0.002 |
    | Return loss weight | 0.0001 |

    ## Training Procedure

    - **Hardware**: NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM)
    - **Optimizer**: AdamW (lr=3e-5, weight_decay=1e-4)
    - **Batch size**: 128
    - **Epochs**: 2
    - **Mixed precision**: AMP (Automatic Mixed Precision) with GradScaler
    - **Gradient clipping**: 0.05
    - **Strategy**: Overlapping context windows with stride = context_len / 2

    > ⏳ *Training stats (duration, steps, loss progression) will be updated after retraining with the new hyperparameters.*

    ## Usage

    ### Loading the Model

    ```python
    import torch
    from huggingface_hub import hf_hub_download

    # Download model weights
    model_path = hf_hub_download(
        repo_id=\"mrvictoru/energydecision-dt\",
        filename=\"aemo_dt_fcas_model.pt\",
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=\"cpu\")

    # If it's a wrapped dict (from upload script):
    if \"model_state_dict\" in checkpoint:
        state_dict = checkpoint[\"model_state_dict\"]
        config = checkpoint.get(\"config\", dict())
        best_val_loss = checkpoint.get(\"val_loss\", float(\"inf\"))
    else:
        # If it's a raw state dict (from best_model.pt)
        state_dict = checkpoint
    ```

    ### Creating the Model

    ```python
    from model import DecisionTransformer

    model = DecisionTransformer(
        state_dim=18,
        act_dim=9,
        n_block=8,
        h_dim=384,
        context_len=180,
        n_heads=8,
        drop_p=0.15,
        max_timestep=100000,
    )
    model.load_state_dict(state_dict)
    model.eval()
    ```

    ### Inference

    ```python
    # Prepare inputs (batch_size, context_length, dim)
    states = torch.randn(1, 180, 18)
    actions = torch.zeros(1, 180, 9)  # Past actions (can be zero-padded)
    returns_to_go = torch.full((1, 180), 10.0)  # Target return
    timesteps = torch.arange(180).unsqueeze(0)

    # Get predicted action for the last timestep
    with torch.no_grad():
        predicted_action = model.get_action(
            states, actions, returns_to_go, timesteps
        )
        # predicted_action shape: (1, 9)
        # Dim 0: energy dispatch in [-1, 1]
        # Dims 1-8: FCAS bids in [0, 1]
    ```

    ## Files

    | File | Description |
    |------|-------------|
    | `aemo_dt_fcas_model.pt` | Full checkpoint with config (~230 MB) |
    | `aemo_dt_fcas_best_checkpoint.pt` | Best model weights only (~230 MB) |

    ## Citation

    If you use this model, please cite:

    ```bibtex
    @misc{energydecision-dt,
      author = {Victor Xu},
      title = {EnergyDecision-DT: Decision Transformer for AEMO FCAS Battery Trading},
      year = {2025},
      publisher = {HuggingFace},
      howpublished = {\\\\url{https://huggingface.co/mrvictoru/energydecision-dt}},
    }
    ```

    ## License

    MIT
    \"\"\")
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
