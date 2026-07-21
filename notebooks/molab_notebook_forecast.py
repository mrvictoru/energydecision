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

    # Import the forecast DT from src/
    sys.path.insert(0, str(Path.cwd()))
    from forecast_decision_transformer import (
        ForecastDecisionTransformer,
        ForecastTrajectoryDataset,
    )

    print("✅ Imports ready")
    return (
        F, ForecastDecisionTransformer, ForecastTrajectoryDataset,
        HfApi, Path, hf_hub_download, json, mo, nn, np, os, pl, plt, time, torch,
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
    use_pilot = mo.ui.checkbox(label="Pilot mode (fast, 12 episodes)", value=True)
    fresh_start = mo.ui.checkbox(label="Fresh start (delete checkpoint)", value=False)
    use_json_config = mo.ui.checkbox(label="Use JSON config instead of individual controls", value=False)
    include_base_dataset = mo.ui.checkbox(label="Include base dataset (aemo_fcas_dataset.parquet)", value=True)
    include_grpo_dataset = mo.ui.checkbox(label="Include GRPO dataset (aemo_fcas_grpo_dataset.parquet)", value=True)
    include_sdp_dataset = mo.ui.checkbox(label="Include SDP trajectory dataset (aemo_sdp_trajectories.parquet)", value=True)

    n_block = mo.ui.number(value=8, label="Blocks", full_width=True)
    h_dim = mo.ui.number(value=768, label="Hidden dim", full_width=True)
    n_heads = mo.ui.number(value=12, label="Heads", full_width=True)
    context_len = mo.ui.number(value=210, label="Context len (history)", full_width=True)
    forecast_len = mo.ui.number(value=48, label="Forecast len", full_width=True)
    drop_p = mo.ui.number(value=0.15, label="Dropout", full_width=True)
    n_kv_heads = mo.ui.number(value=6, label="KV heads", full_width=True)
    qk_norm = mo.ui.checkbox(label="Enable QK-Norm", value=True)
    tie_weights = mo.ui.checkbox(label="Tie embeddings to predictions", value=True)

    batch_size = mo.ui.number(value=64, label="Batch size", full_width=True)
    epochs_per_session = mo.ui.number(value=3, label="Epochs/session", full_width=True)
    lr = mo.ui.number(value=3e-5, label="Learning rate", full_width=True)

    action_loss_weight = mo.ui.number(value=0.999, label="Action loss weight", full_width=True)
    state_loss_weight = mo.ui.number(value=0.002, label="State loss weight", full_width=True)
    return_loss_weight = mo.ui.number(value=0.0001, label="Return loss weight", full_width=True)

    _DEFAULT_JSON = json.dumps(
        {
            "state_dim": 18, "act_dim": 9,
            "n_block": 8, "h_dim": 768, "n_heads": 12,
            "context_len": 210, "forecast_len": 48, "drop_p": 0.15,
            "n_kv_heads": 6, "qk_norm": True, "tie_weights": True,
            "rope_enabled": True,
            "batch_size": 64, "epochs_per_session": 3, "lr": 3e-5,
            "action_loss_weight": 0.999, "state_loss_weight": 0.002,
            "return_loss_weight": 0.0001, "discount_factor": 0.95,
            "return_scale": 2.0, "weight_decay": 1e-4, "grad_clip_norm": 1.0,
        },
        indent=2,
    )
    json_config = mo.ui.text_area(value=_DEFAULT_JSON, label="JSON config", full_width=True)

    train_btn = mo.ui.run_button(label="Start Training", kind="success")
    upload_btn = mo.ui.run_button(label="Upload to HuggingFace", kind="info")
    hf_repo_id = mo.ui.text(value="mrvictoru/energydecision-dt-v2", label="Hugging Face repo", full_width=True)
    hf_token_input = mo.ui.text(value=os.environ.get("HF_TOKEN", ""), label="Hugging Face token", full_width=True)

    resume_from_hf = mo.ui.checkbox(label="Resume from HuggingFace checkpoint", value=False)
    hf_checkpoint_path = mo.ui.text(value="", label="HF checkpoint filename", full_width=True)

    manual_controls = mo.vstack(
        [
            mo.md("### Architecture"),
            mo.hstack([n_block, h_dim, n_heads], justify="start", gap=1),
            mo.hstack([context_len, forecast_len, drop_p], justify="start", gap=1),
            mo.hstack([n_kv_heads, qk_norm, tie_weights], justify="start", gap=2),
            mo.md("### Optimization"),
            mo.hstack([batch_size, epochs_per_session, lr], justify="start", gap=1),
            mo.hstack([action_loss_weight, state_loss_weight, return_loss_weight], justify="start", gap=1),
        ],
        gap=0.5,
    )

    manual_controls
    return (
        action_loss_weight, batch_size, context_len, drop_p, epochs_per_session,
        forecast_len, fresh_start, h_dim, hf_checkpoint_path, hf_repo_id,
        hf_token_input, include_base_dataset, include_grpo_dataset, include_sdp_dataset,
        json_config, lr, manual_controls, n_block, n_heads, n_kv_heads,
        qk_norm, resume_from_hf, return_loss_weight, state_loss_weight,
        tie_weights, train_btn, upload_btn, use_json_config, use_pilot,
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
            mo.md("Enabled datasets are concatenated vertically before training. SDP trajectories provide optimal energy-arbitrage behavior."),
        ])

    _config_section = mo.vstack([
        mo.md("### Configuration"),
        mo.md("JSON mode overrides individual controls.") if use_json_config.value else mo.md("Using individual controls."),
        json_config if use_json_config.value else manual_controls,
    ], gap=0.5)

    mo.vstack([
        mo.md(f"""
## Forecast Decision Transformer — AEMO Training

**Local checkpoint**: {ckpt_info}
**Best model path**: {BEST_MODEL_PATH}
*Forecast DT with RoPE enabled, forecast_len={forecast_len.value}*
        """),
        mo.vstack(_dataset_section, gap=0.5),
        _config_section,
        mo.md("### Checkpoint Resume"),
        mo.hstack([resume_from_hf, hf_checkpoint_path], justify="start", gap=2),
        mo.md("### Actions"),
        mo.hstack([train_btn, upload_btn], justify="start", gap=2),
        hf_repo_id, hf_token_input,
    ])
    return


@app.cell
def _(
    Path, hf_hub_download, include_base_dataset, include_grpo_dataset,
    include_sdp_dataset, mo, pl, use_pilot,
):
    REPO_ID = "mrvictoru/AEMO_simulated_trade"
    CACHE_DIR = Path("/workspace")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    selected_dfs = []

    if use_pilot.value:
        filename = "aemo_fcas_pilot.parquet"
        local_path = CACHE_DIR / filename
        if not local_path.exists():
            print(f"⬇️ Downloading {filename} from HuggingFace...")
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
                print(f"⬇️ Downloading {fn} from HuggingFace...")
                hf_hub_download(repo_id=REPO_ID, filename=fn, local_dir=str(CACHE_DIR),
                               local_dir_use_symlinks=False, repo_type="dataset")
            else:
                print(f"📦 Using cached: {fn}")
            selected_dfs.append(pl.read_parquet(local_path))

        if len(selected_dfs) == 1:
            df = selected_dfs[0]
        else:
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
            print(f"Combined dataset: {len(df):,} rows, {len(filenames)} files")

    print(f"✅ Loaded: {len(df):,} rows")
    mo.stop(df is None, mo.md("❌ No data loaded."))
    return df,


@app.cell
def _(ForecastTrajectoryDataset, TRAIN_CFG, df, mo, torch):
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
    print(f"📊 Dataset: {len(dataset)} windows, {len(train_ds)} train + {len(val_ds)} val")
    print(f"   Sample keys: {list(dataset[0].keys())}")
    return dataset, train_ds, train_loader, val_ds, val_loader


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
            "n_block": n_block.value, "h_dim": h_dim.value,
            "n_heads": n_heads.value, "context_len": context_len.value,
            "forecast_len": forecast_len.value, "drop_p": drop_p.value,
            "n_kv_heads": int(n_kv_heads.value) if n_kv_heads.value else None,
            "qk_norm": qk_norm.value, "tie_weights": tie_weights.value,
            "batch_size": batch_size.value, "lr": lr.value,
            "epochs_per_session": epochs_per_session.value,
            "action_loss_weight": action_loss_weight.value,
            "state_loss_weight": state_loss_weight.value,
            "return_loss_weight": return_loss_weight.value,
        })

    print(f"📋 Config: {TRAIN_CFG['n_block']} blk, {TRAIN_CFG['h_dim']} dim, "
          f"ctx={TRAIN_CFG['context_len']}, forecast={TRAIN_CFG.get('forecast_len')}")
    return (TRAIN_CFG,)


@app.cell
def _(
    BEST_MODEL_PATH, CHECKPOINT_DIR, CHECKPOINT_PATH, F, TRAIN_CFG,
    ForecastDecisionTransformer, fresh_start, hf_checkpoint_path, hf_repo_id,
    load_checkpoint_from_hf, load_or_create_model, mo, save_checkpoint,
    time, torch, train_btn, train_loader, use_pilot, val_loader,
):
    session_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    global_step = 0
    USE_PILOT = use_pilot.value
    fore_len = TRAIN_CFG.get("forecast_len", 48)

    if not train_btn.value:
        mo.stop(True, mo.md("Click **Start Training** to begin."))

    print("=" * 60)
    print("🎯 FORECAST DT TRAINING STARTED")
    print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Device: {device}")
    print(f"📐 {TRAIN_CFG['n_block']} blocks, {TRAIN_CFG['h_dim']} dim, {TRAIN_CFG['n_heads']} heads")
    print(f"📦 Forecast len: {fore_len}, Batch: {TRAIN_CFG['batch_size']}, LR: {TRAIN_CFG['lr']:.2e}")
    print("=" * 60)

    _using_remote = hasattr(mo, 'ui') and mo.ui and hasattr(mo.ui, 'run_button') and False
    if not _using_remote:
        model, optimizer, scheduler, start_epoch, global_step, \
            train_losses, val_losses, best_val_loss, scaler_state = \
            load_or_create_model(TRAIN_CFG, device, fresh=fresh_start.value)

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler is not None and scaler_state is not None:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception:
            pass

    print(f"📊 Starting from epoch={start_epoch}, global_step={global_step}")
    print(f"   Best val_loss: {best_val_loss:.6f}")

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

        print(f"\n{'─'*60}")
        print(f"📚 EPOCH {epoch}")
        print(f"{'─'*60}")

        for batch_idx, batch in enumerate(train_loader):
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtgs = batch["rtgs"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["mask"].to(device)
            rtgs = rtgs / TRAIN_CFG["return_scale"]

            # Forecast tensors (may be None if forecast_len=0)
            f_states = batch.get("forecast_states")
            f_rtgs = batch.get("forecast_rtgs")
            f_ts = batch.get("forecast_timesteps")
            if f_states is not None and fore_len > 0:
                f_states = f_states.to(device)
                f_rtgs = f_rtgs.to(device) / TRAIN_CFG["return_scale"]
                f_ts = f_ts.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                return_preds, state_preds, action_preds = model(
                    states, actions, rtgs, timesteps, mask,
                    forecast_states=f_states,
                    forecast_rtgs=f_rtgs,
                    forecast_timesteps=f_ts,
                )
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

            if global_step % ckpt_freq == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                               train_losses, val_losses, best_val_loss, scaler)

            if batch_idx % 100 == 0:
                _elapsed = time.time() - session_start
                print(f"  Batch {batch_idx:5d}/{_total_batches:5d} | gstep={global_step:6d} "
                      f"| loss={loss.item():.6f} | act={action_loss.item():.6f} "
                      f"| state={state_loss.item():.6f} | ret={return_loss.item():.6f} "
                      f"| {_elapsed:.0f}s")

        train_losses.append(total_loss / max(1, batches_seen))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
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
                    val_loss += F.mse_loss(ap, va).item()
            val_loss /= max(1, len(val_loader))
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 New best model! val_loss={best_val_loss:.6f}")

        _epoch_elapsed = time.time() - _epoch_start
        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                       train_losses, val_losses, best_val_loss, scaler)
        print(f"📊 Epoch {epoch+1} done: train_loss={train_losses[-1]:.6f} "
              f"val_loss={val_losses[-1]:.6f} | 🕒 {_epoch_elapsed:.1f}s")
        scheduler.step()

    _total_time = time.time() - session_start
    print(f"✅ Training complete! {_total_time:.1f}s")
    return device, global_step, session_start, train_losses, val_losses, best_val_loss


@app.cell
def _(mo, np, plt, train_losses, val_losses):
    if train_losses and val_losses:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(train_losses, "b-o", label="Train", markersize=6)
        ax1.plot(val_losses, "r-s", label="Val", markersize=6)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.bar(np.arange(len(train_losses)) - 0.175, train_losses, 0.35, label="Train", color="steelblue")
        ax2.bar(np.arange(len(val_losses)) + 0.175, val_losses, 0.35, label="Val", color="coral")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        mo.mpl.interactive(fig)
    else:
        mo.md("> Loss curves will appear after the first epoch.")
    return


@app.cell
def _(
    TRAIN_CFG, best_val_loss, device, global_step, mo, session_start,
    time, train_losses, val_losses,
):
    _elapsed_hrs = (time.time() - session_start) / 3600
    _summary = mo.md(f"""
## Training Summary
| Metric | Value |
|---|---|
| Device | `{device}` |
| Final train loss | `{train_losses[-1]:.6f}` if train_losses else `N/A` |
| Final val loss | `{val_losses[-1]:.6f}` if val_losses else `N/A` |
| Best val loss | `{best_val_loss:.6f}` |
| Total epochs | `{len(train_losses)}` |
| Total steps | `{global_step}` |
| Duration | `{_elapsed_hrs:.2f}h` |
| Model | `{TRAIN_CFG['n_block']} blk, {TRAIN_CFG['h_dim']} dim, ctx={TRAIN_CFG['context_len']}, forecast={TRAIN_CFG.get('forecast_len')}` |
""")
    if torch.cuda.is_available():
        _alloc = torch.cuda.max_memory_allocated() / 1e9
        _summary + mo.md(f"| GPU | `{_alloc:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB` |")
    _summary
    return


@app.cell
def _(
    BEST_MODEL_PATH, CHECKPOINT_PATH, HfApi, TRAIN_CFG, forecast_len,
    hf_repo_id, hf_token_input, mo, os, torch, upload_btn,
):
    mo.stop(not upload_btn.value, mo.md("Press **Upload to HuggingFace** to upload."))

    repo_id = hf_repo_id.value.strip()
    if not repo_id:
        raise ValueError("Provide a Hugging Face repo ID.")
    token = (hf_token_input.value or os.environ.get("HF_TOKEN", "")).strip()
    if not token:
        raise ValueError("Provide a Hugging Face token or set HF_TOKEN.")

    best_path = BEST_MODEL_PATH if BEST_MODEL_PATH.exists() else CHECKPOINT_PATH
    if not best_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {best_path}")

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
    checkpoint = torch.load(best_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        meta = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        model.load_state_dict(checkpoint)
        meta = {}

    # Save + upload
    upload_path = CHECKPOINT_PATH.parent / "upload_model.pt"
    torch.save({"model_state_dict": model.state_dict(), **meta}, upload_path)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(upload_path),
        path_in_repo=f"forecast_dt_model.pt",
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"✅ Uploaded to {repo_id}")
    return


# ---------------------------------------------------------------------------
# Training helpers (copied from modern notebook)
# ---------------------------------------------------------------------------

@app.cell
def _(
    CHECKPOINT_DIR, CHECKPOINT_PATH, ForecastDecisionTransformer,
    TRAIN_CFG, time, torch,
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

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        start_epoch = 0
        global_step = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        scaler_state = None

        if not fresh and CHECKPOINT_PATH.exists():
            try:
                ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                global_step = ckpt.get("global_step", 0)
                train_losses = ckpt.get("train_losses", [])
                val_losses = ckpt.get("val_losses", [])
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                scaler_state = ckpt.get("scaler_state_dict")
                print(f"✅ Resumed: epoch={start_epoch - 1}, global_step={global_step}, best_val={best_val_loss:.6f}")
            except Exception as e:
                print(f"⚠️ Failed checkpoint load: {e}")

        return model, optimizer, scheduler, start_epoch, global_step, \
            train_losses, val_losses, best_val_loss, scaler_state

    def load_checkpoint_from_hf(repo_id, filename, device, cfg):
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
        ckpt = torch.load(local_path, map_location=device)

        model = load_or_create_model(cfg, device, fresh=True)[0]
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        start_epoch = ckpt.get("epoch", 0) + 1 if "model_state_dict" in ckpt else 0
        global_step = ckpt.get("global_step", 0) if "model_state_dict" in ckpt else 0
        train_losses = ckpt.get("train_losses", []) if "model_state_dict" in ckpt else []
        val_losses = ckpt.get("val_losses", []) if "model_state_dict" in ckpt else []
        best_val_loss = ckpt.get("best_val_loss", float("inf")) if "model_state_dict" in ckpt else float("inf")

        torch.save({k: ckpt[k] for k in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict"]
                    if k in ckpt}, CHECKPOINT_PATH)
        return model, optimizer, scheduler, start_epoch, global_step, \
            train_losses, val_losses, best_val_loss, None

    def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                        train_losses, val_losses, best_val_loss, scaler=None):
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch, "global_step": global_step,
            "train_losses": train_losses, "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "return_scale": TRAIN_CFG["return_scale"],
            "forecast_len": TRAIN_CFG.get("forecast_len", 48),
            "timestamp": time.time(),
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(payload, CHECKPOINT_PATH)

        ckpt_every = TRAIN_CFG.get("checkpoint_every_n_batches", 500)
        if ckpt_every > 0 and global_step > 0 and global_step % ckpt_every == 0:
            step_path = CHECKPOINT_DIR / f"checkpoint_step_{global_step}.pt"
            torch.save(payload, step_path)
            print(f"💾 Step checkpoint: {step_path.name}")

    print("✅ Training helpers ready")
    return load_checkpoint_from_hf, load_or_create_model, save_checkpoint
