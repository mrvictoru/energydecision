import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


# ── Cell 1: Clone repo and import ───────────────────────────────────

@app.cell
def _():
    import os, sys, subprocess, time
    from pathlib import Path

    REPO_DIR = Path("/workspace/energydecision")
    if not REPO_DIR.exists():
        print("⬇️ Cloning energydecision repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/mrvictoru/energydecision.git",
             str(REPO_DIR)],
            check=True, capture_output=True
        )
        print("✅ Cloned")

    sys.path.insert(0, str(REPO_DIR / "src"))

    import json
    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import HfApi, hf_hub_download
    from forecast_decision_transformer import (
        ForecastDecisionTransformer,
        ForecastTrajectoryDataset,
    )

    print("✅ Imports ready")
    return (
        F, ForecastDecisionTransformer, ForecastTrajectoryDataset,
        HfApi, Path, REPO_DIR, hf_hub_download, json, mo, nn, np, os,
        pl, plt, subprocess, time, torch,
    )


# ── Cell 2: Checkpoint paths ────────────────────────────────────────

@app.cell
def _(Path, torch):
    CHECKPOINT_DIR = Path("/workspace/dt_checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = CHECKPOINT_DIR / "latest_checkpoint.pt"
    BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
    ckpt_info = "No local checkpoint found"
    if CHECKPOINT_PATH.exists():
        try:
            _c = torch.load(CHECKPOINT_PATH, map_location="cpu")
            ckpt_info = f"Resume ready: epoch={_c.get('epoch','?')}, best_val={_c.get('best_val_loss',float('inf')):.6f}"
        except Exception:
            ckpt_info = "Checkpoint exists but unreadable"
    return BEST_MODEL_PATH, CHECKPOINT_DIR, CHECKPOINT_PATH, ckpt_info


# ── Cell 3: UI Controls ─────────────────────────────────────────────

@app.cell
def _(json, mo, os):
    use_pilot = mo.ui.checkbox(label="Pilot mode (12 eps, fast)", value=True)
    fresh_start = mo.ui.checkbox(label="Fresh start (delete checkpoint)", value=False)
    use_json_config = mo.ui.checkbox(label="Use JSON config", value=False)
    include_base_dataset = mo.ui.checkbox(label="FCAS dataset", value=True)
    include_grpo_dataset = mo.ui.checkbox(label="GRPO dataset", value=True)
    include_sdp_dataset = mo.ui.checkbox(label="SDP dataset", value=True)

    n_block = mo.ui.number(value=8, label="Blocks")
    h_dim = mo.ui.number(value=768, label="Hidden dim")
    n_heads = mo.ui.number(value=12, label="Heads")
    context_len = mo.ui.number(value=210, label="Context len")
    forecast_len = mo.ui.number(value=48, label="Forecast len")
    drop_p = mo.ui.number(value=0.15, label="Dropout")
    n_kv_heads = mo.ui.number(value=6, label="KV heads")
    qk_norm = mo.ui.checkbox(label="QK-Norm", value=True)
    tie_weights = mo.ui.checkbox(label="Tie weights", value=True)

    batch_size = mo.ui.number(value=64, label="Batch size")
    epochs_per_session = mo.ui.number(value=3, label="Epochs/session")
    lr = mo.ui.number(value=3e-5, label="Learning rate")

    action_loss_weight = mo.ui.number(value=0.999, label="Action loss weight")
    state_loss_weight = mo.ui.number(value=0.002, label="State loss weight")
    return_loss_weight = mo.ui.number(value=0.0001, label="Return loss weight")

    _DFLT = json.dumps({
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
    json_config = mo.ui.text_area(value=_DFLT, label="JSON config")

    train_btn = mo.ui.run_button(label="Start Training", kind="success")
    upload_btn = mo.ui.run_button(label="Upload to HuggingFace", kind="info")
    hf_repo_id = mo.ui.text(value="mrvictoru/energydecision-dt-v2", label="HF model repo")
    hf_token_input = mo.ui.text(value=os.environ.get("HF_TOKEN", ""), label="HF token")
    hf_data_repo = mo.ui.text(value="mrvictoru/AEMO_simulated_trade", label="Data repo")

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
        forecast_len, fresh_start, h_dim, hf_checkpoint_path := mo.ui.text(value="", label="HF checkpoint filename"),
        hf_data_repo, hf_repo_id, hf_token_input, include_base_dataset,
        include_grpo_dataset, include_sdp_dataset, json_config, lr, manual_controls,
        n_block, n_heads, n_kv_heads, qk_norm, resume_from_hf := mo.ui.checkbox(
            label="Resume from HF checkpoint", value=False
        ), return_loss_weight, state_loss_weight, tie_weights, train_btn,
        upload_btn, use_json_config, use_pilot,
    )


# ── Cell 4: Display ─────────────────────────────────────────────────

@app.cell
def _(
    BEST_MODEL_PATH, ckpt_info, forecast_len, fresh_start, hf_checkpoint_path,
    hf_data_repo, hf_repo_id, hf_token_input, include_base_dataset,
    include_grpo_dataset, include_sdp_dataset, json_config, manual_controls,
    mo, resume_from_hf, train_btn, upload_btn, use_json_config, use_pilot,
):
    ds = [
        mo.md("### Dataset"),
        mo.hstack([use_pilot, fresh_start, use_json_config], justify="start", gap=2),
    ]
    if not use_pilot.value:
        ds.append(mo.hstack([include_base_dataset, include_grpo_dataset, include_sdp_dataset], justify="start", gap=2))
    mo.vstack([
        mo.md(f"## Forecast DT — AEMO Training\n*forecast_len={forecast_len.value}, data repo: {hf_data_repo.value}*\n*ckpt: {ckpt_info}*"),
        mo.vstack(ds, gap=0.5),
        json_config if use_json_config.value else manual_controls,
        mo.hstack([resume_from_hf, hf_checkpoint_path], justify="start", gap=2),
        mo.hstack([train_btn, upload_btn], justify="start", gap=2),
        hf_repo_id, hf_token_input,
    ])
    return


# ── Cell 5: Data Loading ────────────────────────────────────────────

@app.cell
def _(
    Path, ForecastTrajectoryDataset, hf_data_repo, hf_hub_download,
    include_base_dataset, include_grpo_dataset, include_sdp_dataset,
    mo, np, pl, use_pilot,
):
    DATA_REPO = hf_data_repo.value.strip() or "mrvictoru/AEMO_simulated_trade"
    CACHE = Path("/workspace")
    CACHE.mkdir(exist_ok=True)
    selected = []
    forecast_npz = None

    # Download TTM forecasts
    npz_path = CACHE / "ttm_forecasts.npz"
    if not npz_path.exists():
        try:
            hf_hub_download(repo_id=DATA_REPO, filename="ttm_forecasts.npz",
                           local_dir=str(CACHE), local_dir_use_symlinks=False, repo_type="dataset")
        except Exception:
            pass
    if npz_path.exists():
        forecast_npz = str(npz_path)
        print(f"✅ TTM forecasts: {np.load(npz_path)['forecast_map'].shape}")

    if use_pilot.value:
        fn = "aemo_fcas_dataset.parquet"
        fp = CACHE / fn
        if not fp.exists():
            hf_hub_download(repo_id=DATA_REPO, filename=fn, local_dir=str(CACHE),
                           local_dir_use_symlinks=False, repo_type="dataset")
        df = pl.read_parquet(fp)
        print(f"Pilot: {len(df):,} rows")
    else:
        filenames = []
        if include_base_dataset.value: filenames.append("aemo_fcas_dataset.parquet")
        if include_grpo_dataset.value: filenames.append("aemo_fcas_grpo_dataset.parquet")
        if include_sdp_dataset.value: filenames.append("aemo_sdp_trajectories.parquet")
        if not filenames:
            raise ValueError("Select at least one dataset.")
        for fn in filenames:
            fp = CACHE / fn
            if not fp.exists():
                hf_hub_download(repo_id=DATA_REPO, filename=fn, local_dir=str(CACHE),
                               local_dir_use_symlinks=False, repo_type="dataset")
            selected.append(pl.read_parquet(fp))
        if len(selected) == 1:
            df = selected[0]
        elif len(selected) > 1:
            df = pl.concat(selected, how="vertical")
            print(f"Combined: {len(df):,} rows from {len(selected)} files")
        else:
            raise ValueError("Select at least one dataset.")

    mo.stop(df is None, mo.md("❌ No data"))
    print(f"✅ Loaded {len(df):,} rows")
    return df, forecast_npz,


# ── Cell 6: Config ──────────────────────────────────────────────────

@app.cell
def _(
    action_loss_weight, batch_size, context_len, drop_p, epochs_per_session,
    forecast_len, h_dim, json, json_config, lr, n_block, n_heads,
    n_kv_heads, qk_norm, return_loss_weight, state_loss_weight,
    tie_weights, use_json_config,
):
    CFG = {
        "state_dim": 18, "act_dim": 9, "max_timestep": 100000,
        "forecast_len": 48, "rope_enabled": True,
        "discount_factor": 0.95, "val_split": 0.1, "return_scale": 2.0,
        "weight_decay": 1e-4, "grad_clip_norm": 1.0,
        "checkpoint_every_n_batches": 500, "max_training_seconds": 11 * 3600,
    }
    if use_json_config.value:
        try:
            CFG.update(json.loads(json_config.value))
        except Exception:
            pass
    else:
        CFG.update(dict(
            n_block=n_block.value, h_dim=h_dim.value, n_heads=n_heads.value,
            context_len=context_len.value, forecast_len=forecast_len.value,
            drop_p=drop_p.value, batch_size=batch_size.value, lr=lr.value,
            n_kv_heads=int(n_kv_heads.value) if n_kv_heads.value else None,
            qk_norm=qk_norm.value, tie_weights=tie_weights.value,
            epochs_per_session=epochs_per_session.value,
            action_loss_weight=action_loss_weight.value,
            state_loss_weight=state_loss_weight.value,
            return_loss_weight=return_loss_weight.value,
        ))
    print(f"📋 {CFG['n_block']} blk, {CFG['h_dim']} dim, ctx={CFG['context_len']}, forecast={CFG.get('forecast_len')}")
    return (CFG,)


# ── Cell 7: Dataset creation ────────────────────────────────────────

@app.cell
def _(CFG, ForecastTrajectoryDataset, df, forecast_npz, torch):
    ds = ForecastTrajectoryDataset(
        df,
        context_length=CFG["context_len"],
        state_dim=CFG["state_dim"],
        act_dim=CFG["act_dim"],
        forecast_len=CFG.get("forecast_len", 48),
        discount_factor=CFG["discount_factor"],
        forecast_npz_path=forecast_npz,
    )
    split = int(len(ds) * (1 - CFG["val_split"]))
    train_ds, val_ds = torch.utils.data.random_split(ds, [split, len(ds) - split])
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=0
    )
    print(f"📊 {len(ds)} windows, {len(train_ds)} train + {len(val_ds)} val")
    return ds, train_ds, train_loader, val_ds, val_loader


# ── Cell 8: Training helpers ────────────────────────────────────────

@app.cell
def _(
    BEST_MODEL_PATH, CHECKPOINT_DIR, CHECKPOINT_PATH,
    CFG, ForecastDecisionTransformer, time, torch,
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
                se = ck.get("epoch", 0) + 1; gs = ck.get("global_step", 0)
                tl = ck.get("train_losses", []); vl = ck.get("val_losses", [])
                bv = ck.get("best_val_loss", float("inf"))
                ss = ck.get("scaler_state_dict")
                print(f"✅ Resumed epoch={se-1}, step={gs}, best_val={bv:.6f}")
            except Exception as e:
                print(f"⚠️ Load failed: {e}")
        return model, opt, sch, se, gs, tl, vl, bv, ss

    def save_checkpoint(model, opt, sch, epoch, step, tl, vl, bv, scaler=None):
        payload = {"model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict(),
            "epoch": epoch, "global_step": step,
            "train_losses": tl, "val_losses": vl, "best_val_loss": bv,
            "return_scale": CFG["return_scale"],
            "forecast_len": CFG.get("forecast_len", 48), "timestamp": time.time()}
        if scaler: payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(payload, CHECKPOINT_PATH)
        freq = CFG.get("checkpoint_every_n_batches", 500)
        if freq and step and step % freq == 0:
            torch.save(payload, CHECKPOINT_DIR / f"checkpoint_step_{step}.pt")
            print(f"💾 Step checkpoint: step_{step}.pt")

    print("✅ Training helpers ready")
    return load_or_create_model, save_checkpoint


# ── Cell 9: Training loop ───────────────────────────────────────────

@app.cell
def _(
    CFG, F, load_or_create_model, mo, save_checkpoint, time, torch,
    train_btn, train_loader, val_loader,
):
    session_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fl = CFG.get("forecast_len", 48)
    tl, vl, bv, gs = [], [], float("inf"), 0

    if not train_btn.value:
        mo.stop(True, mo.md("> Click **Start Training**."))

    print("=" * 60)
    print(f"🎯 FORECAST DT — {device} | {CFG['n_block']} blk {CFG['h_dim']} dim forecast={fl}")
    print("=" * 60)

    model, opt, sch, se, gs, tl, vl, bv, ss = load_or_create_model(CFG, device)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    if scaler and ss:
        try: scaler.load_state_dict(ss)
        except: pass

    total_b = len(train_loader)
    ckpt_freq = CFG.get("checkpoint_every_n_batches", 500)

    for epoch in range(se, se + CFG["epochs_per_session"]):
        model.train()
        loss_acc = al_acc = sl_acc = rl_acc = count = 0.0
        for bi, batch in enumerate(train_loader):
            st = batch["states"].to(device)
            ac = batch["actions"].to(device)
            rt = batch["rtgs"].to(device) / CFG["return_scale"]
            ts = batch["timesteps"].to(device)
            mk = batch["mask"].to(device)
            fs = batch.get("forecast_states"); fr = batch.get("forecast_rtgs"); ft = batch.get("forecast_timesteps")
            if fs is not None and fl > 0:
                fs, fr, ft = fs.to(device), fr.to(device) / CFG["return_scale"], ft.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                rp, sp, ap = model(st, ac, rt, ts, mk, forecast_states=fs, forecast_rtgs=fr, forecast_timesteps=ft)
                a_loss = F.mse_loss(ap, ac)
                s_loss = F.mse_loss(sp, st)
                r_loss = F.mse_loss(rp.squeeze(-1), rt)
                loss = CFG["action_loss_weight"] * a_loss + CFG["state_loss_weight"] * s_loss + CFG["return_loss_weight"] * r_loss

            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip_norm"])
                opt.step()

            loss_acc += loss.item(); al_acc += a_loss.item(); sl_acc += s_loss.item(); rl_acc += r_loss.item()
            count += 1; gs += 1
            if gs % ckpt_freq == 0:
                save_checkpoint(model, opt, sch, epoch, gs, tl, vl, bv, scaler)
            if bi % 100 == 0:
                print(f"  B{bi:5d}/{total_b:5d} | gs={gs:6d} | loss={loss.item():.6f} | act={a_loss.item():.6f} | {time.time()-session_start:.0f}s")

        tl.append(loss_acc / max(1, count))
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vs = batch["states"].to(device); va = batch["actions"].to(device)
                vr = batch["rtgs"].to(device) / CFG["return_scale"]
                vt = batch["timesteps"].to(device); vm = batch["mask"].to(device)
                vfs = batch.get("forecast_states"); vfr = batch.get("forecast_rtgs"); vft = batch.get("forecast_timesteps")
                if vfs is not None and fl > 0:
                    vfs, vfr, vft = vfs.to(device), vfr.to(device) / CFG["return_scale"], vft.to(device)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    rp, sp, ap = model(vs, va, vr, vt, vm, forecast_states=vfs, forecast_rtgs=vfr, forecast_timesteps=vft)
                    v_loss += F.mse_loss(ap, va).item()
        v_loss /= max(1, len(val_loader))
        vl.append(v_loss)

        if v_loss < bv:
            bv = v_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 New best model! val_loss={bv:.6f}")
        save_checkpoint(model, opt, sch, epoch, gs, tl, vl, bv, scaler)
        print(f"📊 Epoch {epoch+1}: train={tl[-1]:.6f} val={vl[-1]:.6f} | act={al_acc/count:.6f} | {time.time()-session_start:.0f}s")
        sch.step()

    print(f"✅ Done in {time.time()-session_start:.0f}s")
    return device, gs, session_start, tl, vl, bv


# ── Cell 10: Plots ──────────────────────────────────────────────────

@app.cell
def _(mo, np, plt, tl, vl):
    if tl and vl:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        a1.plot(tl, "b-o", label="Train", markersize=6)
        a1.plot(vl, "r-s", label="Val", markersize=6)
        a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend(); a1.grid(True, alpha=0.3)
        a2.bar(np.arange(len(tl))-0.175, tl, 0.35, label="Train", color="steelblue")
        a2.bar(np.arange(len(vl))+0.175, vl, 0.35, label="Val", color="coral")
        a2.legend(); a2.grid(True, alpha=0.3)
        plt.tight_layout()
        mo.mpl.interactive(fig)
    else:
        mo.md("> Loss curves after first epoch.")
    return


# ── Cell 11: Summary ────────────────────────────────────────────────

@app.cell
def _(CFG, bv, device, gs, mo, session_start, time, tl, vl):
    h = (time.time() - session_start) / 3600
    mo.md(f"""
## Summary
| Metric | Value |
|---|---|
| Device | `{device}` |
| Final train loss | `{tl[-1]:.6f}` |
| Final val loss | `{vl[-1]:.6f}` |
| Best val loss | `{bv:.6f}` |
| Epochs | `{len(tl)}` |
| Steps | `{gs}` |
| Duration | `{h:.2f}h` |
| Model | `{CFG['n_block']} blk {CFG['h_dim']} dim ctx={CFG['context_len']} forecast={CFG.get('forecast_len')}` |
""")


# ── Cell 12: Upload ─────────────────────────────────────────────────

@app.cell
def _(
    CHECKPOINT_DIR, CHECKPOINT_PATH, CFG, ForecastDecisionTransformer,
    HfApi, hf_repo_id, hf_token_input, mo, os, torch, upload_btn,
):
    mo.stop(not upload_btn.value, mo.md("Press **Upload**."))
    repo_id = hf_repo_id.value.strip()
    token = (hf_token_input.value or os.environ.get("HF_TOKEN", "")).strip()
    if not repo_id: raise ValueError("Provide a HF repo ID.")
    if not token: raise ValueError("Provide a HF token.")

    src = (Path("/workspace/dt_checkpoints/best_model.pt") if Path("/workspace/dt_checkpoints/best_model.pt").exists()
           else CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None)
    if src is None: raise FileNotFoundError("No checkpoint found.")

    model = ForecastDecisionTransformer(
        state_dim=CFG["state_dim"], act_dim=CFG["act_dim"],
        n_block=CFG["n_block"], h_dim=CFG["h_dim"],
        context_len=CFG["context_len"], forecast_len=CFG.get("forecast_len", 48),
        n_heads=CFG["n_heads"], drop_p=CFG["drop_p"],
        max_timestep=CFG["max_timestep"],
        rope_enabled=CFG.get("rope_enabled", True),
        n_kv_heads=CFG.get("n_kv_heads"), qk_norm=CFG.get("qk_norm", False),
        tie_weights=CFG.get("tie_weights", False),
    )
    ck = torch.load(src, map_location="cpu")
    model.load_state_dict(ck.get("model_state_dict", ck))
    meta = {k: v for k, v in ck.items() if k != "model_state_dict"}

    upload_path = Path("/workspace/dt_checkpoints/upload_model.pt")
    torch.save({"model_state_dict": model.state_dict(), **meta}, upload_path)
    HfApi().upload_file(path_or_fileobj=str(upload_path), path_in_repo="forecast_dt_model.pt",
                        repo_id=repo_id, repo_type="model", token=token)
    print(f"✅ Uploaded forecast_dt_model.pt → {repo_id}")
