# MoLab AI Agent Instructions: Build AEMO Decision Transformer Training Notebook

## Overview

Build a **self-contained marimo notebook** (`train_aemo_dt_molab.py`) for training a Decision Transformer model on AEMO battery trading data. The notebook will run on MoLab with an RTX 6000 Pro GPU (96 GB VRAM, 125 TFLOPS) and must handle the 12-hour session limit through checkpoint/resume.

## Environment

- **Platform**: MoLab (cloud-hosted marimo notebook)
- **GPU**: RTX 6000 Pro Blackwell, 96 GB VRAM, 125 TFLOPS
- **CPU/RAM**: 4 CPUs, 32 GB RAM (limited — be memory-conscious)
- **Session limit**: 12 hours (90 min idle timeout). Auto-resumes on restart.
- **Storage**: Limited persistent storage at `/workspace/`. Use for checkpoints.
- **Python**: 3.10+ with pre-installed torch, numpy, polars. Additional packages auto-install on import.
- **Notebook file**: `notebooks/train_aemo_dt_molab.py` (committed to the repo, mirrored on MoLab)

## Notebook Structure

The notebook must be a single `.py` file containing multiple marimo cells. Each cell is a Python function or block separated by marimo's cell markers. It must be self-contained — all DT model code, dataset handling, and training logic must be inlined (the repo's `src/` modules are NOT importable on MoLab since only notebook files sync).

### Required Cells

#### Cell 1: Package Installation

MoLab pre-installs torch, numpy, polars. Install any additional dependencies:

```python
# MoLab's package manager will install these on first run
import micropip  # or use MoLab's built-in package manager
# If datasets/huggingface_hub not available:
# !pip install datasets huggingface_hub pyarrow
```

#### Cell 2: HF Dataset Download

Download the FCAS dataset from HuggingFace. Support both pilot and full configurations (the pilot split will need to be created separately):

```python
import polars as pl
from huggingface_hub import hf_hub_download

repo = "mrvictoru/AEMO_simulated_trade"
USE_PILOT = True  # Set False for full training (multi-session)

if USE_PILOT:
    # Download pilot subset (8 train + 4 val)
    train_path = hf_hub_download(repo_id=repo, filename="pilot/train.parquet", repo_type="dataset")
    val_path = hf_hub_download(repo_id=repo, filename="pilot/val.parquet", repo_type="dataset")
    df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
else:
    # Download full FCAS dataset
    path = hf_hub_download(repo_id=repo, filename="aemo_fcas_dataset.parquet", repo_type="dataset")
    df = pl.read_parquet(path)

# Convert to polars DataFrame
print(f"Loaded {df.height:,} rows")
```

**The HF dataset must contain these configurations:**
- `pilot`: 8 train + 4 val episodes (fast, fits in 12h)
- `full_fcas`: 2,425 episodes, 78.4M rows (requires checkpoint/resume across sessions)

**Dataset columns (matching TrajectoryDataset schema):**
- `episode_id` (string): unique identifier per episode
- `step` (int64): step index within episode
- `norm_observation` (list[float32]): 18-dim normalized observation vector
- `action` (list[float32]): 9-dim action vector [energy + 8 FCAS bids] (full_fcas mode)
- `reward` (float64): per-step reward (already normalized /1000)

#### Cell 3: Decision Transformer Model Definition

In-line the `DecisionTransformer` class from `src/decision_transformer.py`. Key design:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, h_dim, n_heads, drop_p=0.1, use_rope=False, rope_base=10000, rope_max_position=None):
        super().__init__()
        self.ln1 = RMSNorm(h_dim)
        self.attn = nn.MultiheadAttention(h_dim, n_heads, dropout=drop_p, batch_first=True)
        self.ln2 = RMSNorm(h_dim)
        self.ffn = SwiGLU(h_dim)
        self.dropout = nn.Dropout(drop_p)
        self.use_rope = use_rope
        # ... RoPE implementation if use_rope ...

    def forward(self, x, attn_mask=None):
        # Pre-norm + attention + residual
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(h)
        # Pre-norm + FFN + residual
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x

class DecisionTransformer(nn.Module):
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
            nn.Tanh()  # Action output in [-1, 1]; FCAS bids (dims 1-8) will be clamped to [0, 1] at inference
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

        # Interleave: (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        # Input length is 3*T
        stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)  # (B, 3, T, H)
        x = stacked.permute(0, 2, 1, 3).reshape(B, 3*T, self.h_dim)
        x = x + time_emb.repeat_interleave(3, dim=1)

        if attention_mask is not None:
            attn_mask = attention_mask.repeat_interleave(3, dim=1)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, 3T)
        else:
            attn_mask = None

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)

        # Predictions from the (r, s) stream, actions from the (r, s) stream
        # Position 0, 1 in each triplet = return and state
        pred_mask = torch.zeros(3*T, dtype=torch.bool)
        pred_mask[0::3] = True   # return tokens
        pred_mask[1::3] = True   # state tokens
        act_mask = torch.ones(3*T, dtype=torch.bool)
        act_mask[2::3] = False   # action is predicted from (r, s) tokens

        x_pred = x[:, pred_mask]   # (B, ~2T, H)
        x_act = x[:, act_mask]     # (B, ~2T, H)

        return_preds = self.predict_return(x_pred[:, ::2])   # return tokens
        state_preds = self.predict_state(x_pred[:, 1::2])    # state tokens
        action_preds = self.predict_action(x_act[:, ::2])    # action from (r, s) tokens

        return action_preds, state_preds, return_preds

    def get_action(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        """Inference: return predicted action for the last timestep."""
        action_preds, _, _ = self.forward(states, actions, returns_to_go, timesteps, attention_mask)
        return action_preds[:, -1]  # Last predicted action
```

#### Cell 4: TrajectoryDataset Definition

In-line the `TrajectoryDataset` class from `src/transformer_training.py`:

```python
class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, df, context_length=180, state_dim=18, act_dim=9,
                 discount_factor=0.95, max_episodes=None, skip_short_episodes=True):
        self.context_length = context_length
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.discount_factor = discount_factor

        # Group by episode_id
        episodes = df.group_by("episode_id").agg(
            pl.col("step").len().alias("n_steps"),
            pl.col("norm_observation").alias("obs"),
            pl.col("action").alias("act"),
            pl.col("reward").alias("rew"),
        )

        # Filter short episodes
        if skip_short_episodes:
            episodes = episodes.filter(pl.col("n_steps") >= context_length * 3)

        # Compute discounted returns-to-go
        all_states, all_actions, all_rtgs, all_timesteps = [], [], [], []
        for row in episodes.iter_rows(named=True):
            obs_arr = np.array(row["obs"].to_list(), dtype=np.float32)
            act_arr = np.array(row["act"].to_list(), dtype=np.float32)
            rew_arr = np.array(row["rew"].to_list(), dtype=np.float32)
            n = len(rew_arr)

            # Discounted returns-to-go
            rtg = np.zeros(n, dtype=np.float32)
            running = 0.0
            for t in reversed(range(n)):
                running = rew_arr[t] + discount_factor * running
                rtg[t] = running

            # Create context windows
            for i in range(0, n - context_length + 1, context_length // 2):
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
```

#### Cell 5: Training Loop with Checkpoint Resume

The training loop must support checkpoint/resume across MoLab sessions:

```python
import os
import json
import time
from pathlib import Path

# Config
CHECKPOINT_DIR = Path("/workspace/dt_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest_checkpoint.pt"

# Training hyperparameters
TRAIN_CFG = {
    "state_dim": 18,
    "act_dim": 9,          # full_fcas mode
    "n_block": 8,
    "h_dim": 384,
    "n_heads": 8,
    "context_len": 180,
    "drop_p": 0.15,
    "max_timestep": 100000,
    "use_rope": False,
    "batch_size": 128,     # Larger batch on 96 GB VRAM
    "lr": 3e-5,
    "epochs_per_session": 1,  # Per-session (will accumulate across checkpoints)
    "val_split": 0.1,
    "discount_factor": 0.95,
    "return_scale": 2.0,
    "action_loss_weight": 0.999,
    "state_loss_weight": 0.002,
    "return_loss_weight": 0.0001,
    "weight_decay": 1e-4,
    "grad_clip_norm": 0.05,
    "checkpoint_every_n_batches": 500,
    "max_training_seconds": 11 * 3600,  # 11 hours — leave 1h margin for saving
}

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    start_epoch = 0
    global_step = 0
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    #  Resume from checkpoint if available
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
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_val={best_val_loss:.6f}")

    return model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_losses, val_losses, best_val_loss):
    """Save checkpoint to persistent storage."""
    torch.save({
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
    }, LATEST_CHECKPOINT)
    print(f"💾 Checkpoint saved at epoch {epoch}, step {global_step}")

def upload_best_model_to_hf(model, best_val_loss):
    """Upload the best model checkpoint to HuggingFace Model Hub."""
    from huggingface_hub import HfApi, login
    # If HF_TOKEN env var is set, will auto-login
    api = HfApi()
    repo_id = "mrvictoru/energydecision-dt"

    # Save model temporarily
    tmp_path = Path("/workspace/best_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_loss": best_val_loss,
        "config": {k: v for k, v in TRAIN_CFG.items()
                   if k in ("state_dim", "act_dim", "n_block", "h_dim", "n_heads",
                            "context_len", "drop_p", "return_scale", "discount_factor")},
    }, tmp_path)

    try:
        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo="aemo_dt_fcas_model.pt",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"📤 Uploaded best model to HF: {repo_id}")
    except Exception as e:
        print(f"⚠️  Failed to upload to HF (may need token): {e}")

def train_epoch(model, optimizer, scheduler, dataloader, epoch, cfg, device):
    """Train one epoch, returning average loss and updated step count."""
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_state_loss = 0.0
    total_return_loss = 0.0
    start_time = time.time()

    for batch_idx, (states, actions, rtgs, timesteps) in enumerate(dataloader):
        # Check time limit
        if time.time() - session_start > cfg["max_training_seconds"]:
            print("⏰ Approaching 12h limit — finishing batch and saving checkpoint...")
            return total_loss / max(1, batch_idx), batch_idx, True

        states = states.to(device)
        actions = actions.to(device)
        rtgs = rtgs.to(device)
        timesteps = timesteps.to(device)

        # Normalize returns
        rtgs = rtgs / cfg["return_scale"]

        # Forward pass
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
                          train_losses, val_losses, best_val_loss)

    n = max(1, batch_idx)
    return total_loss / n, batch_idx, False

# ── Main Training ──

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

session_start = time.time()

# Load data
# ... (use the dataset from Cell 2, convert to TrajectoryDataset) ...

# Split train/val
n_val = max(1, int(len(train_ds) * TRAIN_CFG["val_split"]))
n_train = len(train_ds) - n_val
train_subset, val_subset = torch.utils.data.random_split(train_ds, [n_train, n_val])
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=TRAIN_CFG["batch_size"], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=TRAIN_CFG["batch_size"], shuffle=False)

# Create or load model
model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss = \
    load_or_create_model(TRAIN_CFG, device)

print(f"📊 Starting from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

# Training loop
time_limit_hit = False
for epoch in range(start_epoch, TRAIN_CFG["epochs_per_session"] + start_epoch):
    avg_loss, steps, time_limit_hit = train_epoch(
        model, optimizer, scheduler, train_loader, epoch, TRAIN_CFG, device
    )
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for states, actions, rtgs, timesteps in val_loader:
            states, actions = states.to(device), actions.to(device)
            rtgs, timesteps = rtgs.to(device), timesteps.to(device)
            rtgs = rtgs / TRAIN_CFG["return_scale"]
            action_preds, state_preds, return_preds = model(states, actions, rtgs, timesteps)
            val_loss += F.mse_loss(action_preds, actions).item()
    val_loss /= max(1, len(val_loader))
    val_losses.append(val_loss)

    print(f"Epoch {epoch}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")

    # Track best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
        print(f"🏆 New best val_loss: {best_val_loss:.6f}")
        upload_best_model_to_hf(model, best_val_loss)

    save_checkpoint(model, optimizer, scheduler, epoch, global_step + steps,
                    train_losses, val_losses, best_val_loss)

    if time_limit_hit:
        print("⏰ Time limit reached — session will auto-resume from checkpoint.")
        break

print(f"\n✅ Training session complete. Best val_loss: {best_val_loss:.6f}")
```

#### Cell 6: Progress Dashboard (marimo widget)

Use marimo's reactive widgets for a live training dashboard:

```python
import marimo as mo
import matplotlib.pyplot as plt

# Create reactive elements
progress = mo.ui.number(value=0, label="Epoch")
loss_plot = mo.ui.anywidget(...)  # or just matplotlib

mo.hstack([progress])

# Plot loss curves
if train_losses and val_losses:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train")
    ax1.plot(val_losses, label="val")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.bar(["train", "val"], [train_losses[-1], val_losses[-1]])
    ax2.set_title("Current Loss")
    mo.mpl.interactive(fig)
```

### Additional Notes for the AI Agent

1. **96 GB VRAM usage**: With such large memory, set `batch_size` high (128-256) for 8×384 models. Monitor with `torch.cuda.memory_summary()` if needed.

2. **32 GB RAM constraint**: The dataset loading code must be memory-efficient. For the full dataset (78.4M rows), consider using streaming or chunked loading. The pilot dataset fits easily.

3. **Checkpoint file location**: Save to `/workspace/dt_checkpoints/` — this is the persistent storage that survives session restarts.

4. **HF_TOKEN**: The notebook should check for a `HF_TOKEN` environment variable or marimo secret for uploading models to HuggingFace. Provide a fallback: save model locally and print a message if upload fails.

5. **Session resume detection**: On startup, check if `/workspace/dt_checkpoints/latest_checkpoint.pt` exists. If yes, load and resume. If no, start fresh.

6. **FCAS action clamping**: At inference, the DT's `get_action` returns `tanh` values in `[-1, 1]`. For `full_fcas` mode, the energy dispatch (dim 0) stays in `[-1, 1]`, but FCAS bids (dims 1-8) must be clamped to `[0, 1]`. Handle this in the `get_action` method.

7. **Training time estimation** (for 96GB RTX 6000 Pro):
   - Pilot (8+4 episodes): ~30-60 minutes per epoch
   - Full (2,425 episodes): ~6-10 hours per epoch
   The notebook should track elapsed time and stop cleanly before the 12-hour limit.

8. **Model architecture details to preserve from repo**:
   - Pre-norm architecture (RMSNorm before attention/FFN)
   - SwiGLU feed-forward (not standard MLP)
   - Rotary position embeddings (RoPE) as optional
   - The `return_scale` parameter for RTG normalization
   - Action loss weight: 0.75 (best from autoresearch sweeps)

9. **Output**: After training, produce a markdown summary cell showing:
   - Final train/val losses
   - Best checkpoint loss
   - GPU VRAM usage
   - Training duration
   - Whether model was uploaded to HF
