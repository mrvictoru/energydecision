import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import sys
    import time
    from pathlib import Path

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
        sys,
        time,
        torch,
    )


@app.cell
def _(mo):
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

    mo.vstack(
        [
            mo.md("## 🎛️ AEMO Decision Transformer — MoLab Training"),
            mo.hstack([use_pilot, fresh_start, use_json_config], justify="start", gap=2),
            mo.hstack([include_base_dataset, include_grpo_dataset], justify="start", gap=2),
            mo.hstack([n_block, h_dim, n_heads], justify="start", gap=1),
            mo.hstack([context_len, drop_p], justify="start", gap=1),
            mo.hstack([n_kv_heads, qk_norm, tie_weights], justify="start", gap=2),
            mo.hstack([batch_size, epochs_per_session, lr], justify="start", gap=1),
            mo.hstack([action_loss_weight, state_loss_weight, return_loss_weight], justify="start", gap=1),
            mo.md("### Actions"),
            mo.hstack([train_btn, upload_btn], justify="start", gap=2),
            json_config,
        ]
    )
    return (
        action_loss_weight,
        batch_size,
        context_len,
        drop_p,
        epochs_per_session,
        fresh_start,
        h_dim,
        include_base_dataset,
        include_grpo_dataset,
        json_config,
        lr,
        n_block,
        n_heads,
        n_kv_heads,
        qk_norm,
        return_loss_weight,
        state_loss_weight,
        tie_weights,
        train_btn,
        upload_btn,
        use_json_config,
        use_pilot,
    )


@app.cell
def _(hf_hub_download, Path, include_base_dataset, include_grpo_dataset, pl, use_pilot):
    REPO_ID = "mrvictoru/AEMO_simulated_trade"
    CACHE_DIR = Path("/workspace")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if use_pilot.value:
        filename = "aemo_fcas_pilot.parquet"
        local_path = CACHE_DIR / filename

        if not local_path.exists():
            print(f"⬇️ Downloading {filename} from HuggingFace...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(CACHE_DIR), local_dir_use_symlinks=False)
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
                hf_hub_download(repo_id=REPO_ID, filename=base_filename, local_dir=str(CACHE_DIR), local_dir_use_symlinks=False)
            else:
                print(f"📦 Using cached file: {base_path}")

            base_df = pl.read_parquet(base_path)
            selected_dfs.append(base_df)
            print(f"Loaded base dataset: {len(base_df):,} rows from {base_path}")

        if include_grpo_dataset.value:
            if not grpo_path.exists():
                print(f"⬇️ Downloading {grpo_filename} from HuggingFace...")
                hf_hub_download(repo_id=REPO_ID, filename=grpo_filename, local_dir=str(CACHE_DIR), local_dir_use_symlinks=False)
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
            df = pl.concat(selected_dfs, how="vertical")
            print(f"Combined dataset rows: {len(df):,}")

    return (df,)


@app.cell
def _(df, pl):
    print("📊 Dataset profile")
    episode_stats = df.group_by("episode_id").agg(
        pl.col("step").len().alias("n_steps"),
        pl.col("reward").sum().alias("total_reward"),
    )
    print(episode_stats.head())
    return (episode_stats,)


@app.cell
def _(F, nn, torch, Path, sys):
    repo_root = Path.cwd().resolve()
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from decision_transformer import DecisionTransformer as ModernDecisionTransformer

    class DecisionTransformer(ModernDecisionTransformer):
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
            super().__init__(
                state_dim=state_dim,
                act_dim=act_dim,
                n_block=n_block,
                h_dim=h_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=drop_p,
                max_timestep=max_timestep,
                rope_enabled=use_rope,
                rope_max_position=rope_max_position,
                rope_base=rope_base,
                n_kv_heads=n_kv_heads,
                qk_norm=qk_norm,
                tie_weights=tie_weights,
            )

        def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
            return super().forward(
                state=states,
                rtg=returns_to_go,
                timestep=timesteps,
                actions=actions,
                attention_mask=attention_mask,
            )

        def get_action(self, states, actions, returns_to_go, timesteps, attention_mask=None):
            action = super().get_action(
                states=states,
                actions=actions,
                rtg=returns_to_go,
                timesteps=timesteps,
                attention_mask=attention_mask,
            )
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

            episodes = df.group_by("episode_id").agg(
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
    n_kv_heads,
    qk_norm,
    return_loss_weight,
    state_loss_weight,
    tie_weights,
    use_json_config,
    use_pilot,
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
def _(DecisionTransformer, Path, TRAIN_CFG, torch, time):
    CHECKPOINT_PATH = Path("/workspace/latest_checkpoint.pt")
    BEST_MODEL_PATH = Path("/workspace/best_model.pt")

    def load_or_create_model(cfg, device):
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
        return model, optimizer, scheduler, 0, 0, [], [], float("inf"), None

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
        torch.save(payload, CHECKPOINT_PATH)

    print("✅ Training helpers ready")
    return (CHECKPOINT_PATH, BEST_MODEL_PATH, load_or_create_model, save_checkpoint)


@app.cell
def _(BEST_MODEL_PATH, CHECKPOINT_PATH, DecisionTransformer, Path, TRAIN_CFG, TrajectoryDataset, df, fresh_start, load_or_create_model, save_checkpoint, time, torch, train_btn, use_pilot):
    if not train_btn.value:
        print("Training button not clicked yet")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if fresh_start.value and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink(missing_ok=True)

    dataset = TrajectoryDataset(
        df,
        context_length=TRAIN_CFG["context_len"],
        state_dim=TRAIN_CFG["state_dim"],
        act_dim=TRAIN_CFG["act_dim"],
        discount_factor=TRAIN_CFG["discount_factor"],
    )
    split = int(len(dataset) * (1 - TRAIN_CFG["val_split"]))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=False, num_workers=0)

    model, optimizer, scheduler, start_epoch, global_step, train_losses, val_losses, best_val_loss, scaler_state = load_or_create_model(TRAIN_CFG, device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(TRAIN_CFG["epochs_per_session"]):
        model.train()
        total_loss = 0.0
        for states, actions, rtgs, timesteps in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            rtgs = rtgs.to(device)
            timesteps = timesteps.to(device)
            rtgs = rtgs / TRAIN_CFG["return_scale"]

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                action_preds, state_preds, return_preds = model(states, actions, rtgs, timesteps)
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

        train_losses.append(total_loss / max(1, len(train_loader)))
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for states, actions, rtgs, timesteps in val_loader:
                states = states.to(device)
                actions = actions.to(device)
                rtgs = rtgs.to(device)
                timesteps = timesteps.to(device)
                rtgs = rtgs / TRAIN_CFG["return_scale"]
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    action_preds, _, _ = model(states, actions, rtgs, timesteps)
                    val_loss += F.mse_loss(action_preds, actions).item()
            val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}: train_loss={train_losses[-1]:.6f} val_loss={val_losses[-1]:.6f}")
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_losses, val_losses, min(best_val_loss, val_loss), scaler)

    torch.save(model.state_dict(), BEST_MODEL_PATH)
    print(f"Saved best model to {BEST_MODEL_PATH}")
    return (train_losses, val_losses)


@app.cell
def _(train_losses, val_losses):
    if train_losses and val_losses:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(train_losses, label="train")
        ax.plot(val_losses, label="val")
        ax.legend()
        ax.set_title("Training loss")
        plt.show()
    return


if __name__ == "__main__":
    app.run()
