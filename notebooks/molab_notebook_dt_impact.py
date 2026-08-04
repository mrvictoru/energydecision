import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os, sys, subprocess, time, shutil
    from pathlib import Path

    REPO_DIR = Path("/workspace/energydecision")
    BRANCH = "feature/market-impact-modeling"

    # Full clean removal
    if REPO_DIR.exists():
        print("🗑️ Removing existing clone...")
        shutil.rmtree(str(REPO_DIR), ignore_errors=True)
        time.sleep(0.5)

    print(f"⬇️ Cloning energydecision repo (branch: {BRANCH})...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1",
         "--branch", BRANCH,
         "https://github.com/mrvictoru/energydecision.git",
         str(REPO_DIR)],
        check=True, capture_output=True, timeout=120
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

    from decision_transformer import DecisionTransformer
    from transformer_training import TrajectoryDataset

    _NOTEBOOK_VERSION = "2025-08-04-stride-v2.1"
    print(f"✅ Imports ready  |  notebook version: {_NOTEBOOK_VERSION}")
    return (
        DecisionTransformer,
        F,
        HfApi,
        Path,
        TrajectoryDataset,
        hf_hub_download,
        json,
        mo,
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
            _c = torch.load(CHECKPOINT_PATH, map_location="cpu")
            ckpt_info = f"Resume ready: epoch={_c.get('epoch','?')}, best_val={_c.get('best_val_loss',float('inf')):.6f}"
        except Exception:
            ckpt_info = "Checkpoint exists but unreadable"
    return BEST_MODEL_PATH, CHECKPOINT_DIR, CHECKPOINT_PATH, ckpt_info


@app.cell
def _(mo):
    use_pilot = mo.ui.checkbox(label="Pilot mode (12 eps, fast)", value=True)
    fresh_start = mo.ui.checkbox(label="Fresh start (delete checkpoint)", value=False)
    hf_data_repo = mo.ui.text(value="mrvictoru/AEMO_simulated_trade_impact", label="Data repo (impact-aware dataset)", full_width=True)
    return fresh_start, hf_data_repo, use_pilot


@app.cell
def _(mo):
    n_block = mo.ui.number(value=8, label="Blocks", full_width=True)
    h_dim = mo.ui.number(value=768, label="Hidden dim", full_width=True)
    n_heads = mo.ui.number(value=12, label="Heads", full_width=True)
    context_len = mo.ui.number(value=210, label="Context len", full_width=True)
    drop_p = mo.ui.number(value=0.15, label="Dropout", full_width=True)
    n_kv_heads = mo.ui.number(value=6, label="KV heads", full_width=True)
    qk_norm = mo.ui.checkbox(label="QK-Norm", value=True)
    tie_weights = mo.ui.checkbox(label="Tie weights", value=True)
    return (
        context_len,
        drop_p,
        h_dim,
        n_block,
        n_heads,
        n_kv_heads,
        qk_norm,
        tie_weights,
    )


@app.cell
def _(
    context_len,
    drop_p,
    h_dim,
    mo,
    n_block,
    n_heads,
    n_kv_heads,
    qk_norm,
    tie_weights,
):
    batch_size = mo.ui.number(value=128, label="Batch size", full_width=True)
    epochs_per_session = mo.ui.number(value=4, label="Epochs/session", full_width=True)
    lr = mo.ui.number(value=3e-5, label="Learning rate", full_width=True)
    action_loss_weight = mo.ui.number(value=0.999, label="Action loss weight", full_width=True)
    state_loss_weight = mo.ui.number(value=0.002, label="State loss weight", full_width=True)
    return_loss_weight = mo.ui.number(value=0.0001, label="Return loss weight", full_width=True)
    use_json_config = mo.ui.checkbox(label="Use JSON config", value=False)
    _DFLT = '{\n  "state_dim": 18, "act_dim": 9, "max_timestep": 100000,\n  "discount_factor": 0.95, "val_split": 0.1, "return_scale": 1.0,\n  "weight_decay": 1e-4, "grad_clip_norm": 1.0,\n  "checkpoint_every_n_batches": 500, "max_training_seconds": 11 * 3600\n}'
    json_config = mo.ui.text_area(value=_DFLT, label="", full_width=True)

    def _snapshot_config(_):
        """Snapshot all UI config values when the user clicks Apply."""
        return {
            "n_block": n_block.value, "h_dim": h_dim.value,
            "n_heads": n_heads.value, "context_len": context_len.value,
            "drop_p": drop_p.value, "n_kv_heads": n_kv_heads.value,
            "qk_norm": qk_norm.value, "tie_weights": tie_weights.value,
            "batch_size": batch_size.value, "lr": lr.value,
            "epochs_per_session": epochs_per_session.value,
            "action_loss_weight": action_loss_weight.value,
            "state_loss_weight": state_loss_weight.value,
            "return_loss_weight": return_loss_weight.value,
            "use_json_config": use_json_config.value,
            "json_config": json_config.value,
        }


    commit_btn = mo.ui.button(
        label="Apply config", kind="success",
        on_click=_snapshot_config,
    )
    return (
        action_loss_weight,
        batch_size,
        commit_btn,
        epochs_per_session,
        json_config,
        lr,
        return_loss_weight,
        state_loss_weight,
        use_json_config,
    )


@app.cell
def _(mo, os):
    train_btn = mo.ui.run_button(label="Start Training", kind="success")
    upload_btn = mo.ui.run_button(label="Upload to HuggingFace", kind="info")
    hf_repo_id = mo.ui.text(value="mrvictoru/energydecision-dt-v2-impact", label="HF model repo", full_width=True)
    hf_token_input = mo.ui.text(value=os.environ.get("HF_TOKEN", ""), label="HF token", full_width=True)
    return hf_repo_id, hf_token_input, train_btn, upload_btn


@app.cell
def _(
    action_loss_weight,
    batch_size,
    commit_btn,
    context_len,
    drop_p,
    epochs_per_session,
    h_dim,
    json_config,
    lr,
    mo,
    n_block,
    n_heads,
    n_kv_heads,
    qk_norm,
    return_loss_weight,
    state_loss_weight,
    tie_weights,
    use_json_config,
):
    _commit_cfg = commit_btn.value  # None until first click, then a dict
    if isinstance(_commit_cfg, dict):
        _applied_info = (
            f"✅ **Applied** — {_commit_cfg['n_block']} blk, "
            f"{_commit_cfg['h_dim']} dim, ctx={_commit_cfg['context_len']}, "
            f"lr={_commit_cfg['lr']}, bs={_commit_cfg['batch_size']}"
        )
    else:
        _applied_info = "_Modify settings above, then click **Apply config** to lock them in._"

    mo.vstack([
        mo.md("### Architecture"),
        mo.hstack([n_block, h_dim, n_heads, context_len], justify="start", widths="equal", gap=1),
        mo.hstack([drop_p, n_kv_heads, qk_norm, tie_weights], justify="start", widths="equal", gap=1),
        mo.md("### Optimization"),
        mo.hstack([batch_size, epochs_per_session, lr], justify="start", widths="equal", gap=1),
        mo.hstack([action_loss_weight, state_loss_weight, return_loss_weight], justify="start", widths="equal", gap=1),
        mo.md("### JSON Config"),
        mo.hstack([use_json_config, mo.md("_Override all fields above with this JSON._")], gap=1),
        json_config,
        mo.md("---"),
        mo.hstack([commit_btn, mo.md(_applied_info)], gap=1),
    ])
    return


@app.cell
def _(
    ckpt_info,
    context_len,
    fresh_start,
    hf_data_repo,
    hf_repo_id,
    hf_token_input,
    mo,
    train_btn,
    upload_btn,
    use_pilot,
):
    mo.vstack([
        mo.md("### Dataset"),
        mo.hstack([use_pilot, fresh_start, hf_data_repo], widths=["auto", "auto", 1], gap=1),
        mo.md(f"## Impact-Aware DT — AEMO Training\n*ctx={context_len.value}, ckpt: {ckpt_info}*"),
        mo.md("### Action"),
        mo.hstack([train_btn, upload_btn], justify="start", gap=1),
        mo.hstack([hf_repo_id, hf_token_input], widths=[1, 1], gap=1),
    ])
    return


@app.cell
def _(Path, hf_data_repo, hf_hub_download, mo, pl, use_pilot):
    DATA_REPO = hf_data_repo.value.strip() or "mrvictoru/AEMO_simulated_trade_impact"
    CACHE = Path("/workspace")
    CACHE.mkdir(exist_ok=True)

    fn = "aemo_impact_dataset.parquet"
    fp = CACHE / fn
    if not fp.exists():
        hf_hub_download(repo_id=DATA_REPO, filename=fn, local_dir=str(CACHE),
                       local_dir_use_symlinks=False, repo_type="dataset")
    df_full = pl.read_parquet(fp)
    print(f"Loaded: {len(df_full):,} rows across {df_full['episode_id'].n_unique()} episodes")

    if use_pilot.value:
        _n_eps = df_full["episode_id"].n_unique()
        _pilot_eps = 12
        if _n_eps > _pilot_eps:
            import random as _random
            _ids = df_full["episode_id"].unique().to_list()
            _random.shuffle(_ids)
            _keep = set(_ids[:_pilot_eps])
            df = df_full.filter(pl.col("episode_id").is_in(_keep))
            print(f"🎯 Sampled to {_pilot_eps} episodes: {len(df):,} rows")
            del df_full
        else:
            df = df_full
    else:
        df = df_full

    mo.stop(df is None, mo.md("❌ No data"))
    print(f"✅ Loaded {len(df):,} rows, {df['episode_id'].n_unique()} episodes")
    return CACHE, df


@app.cell
def _(commit_btn, json):
    CFG = {
        "state_dim": 18, "act_dim": 9, "max_timestep": 100000,
        "discount_factor": 0.95, "val_split": 0.1, "return_scale": 1.0,
        "weight_decay": 1e-4, "grad_clip_norm": 1.0,
        "checkpoint_every_n_batches": 500, "max_training_seconds": 11 * 3600,
    }

    # Only update from the snapshot after user clicks "Apply config"
    _snapshot = commit_btn.value
    if isinstance(_snapshot, dict):
        if _snapshot.get("use_json_config"):
            try:
                CFG.update(json.loads(_snapshot["json_config"]))
            except Exception:
                pass
        else:
            CFG.update(dict(
                n_block=_snapshot["n_block"], h_dim=_snapshot["h_dim"],
                n_heads=_snapshot["n_heads"], context_len=_snapshot["context_len"],
                drop_p=_snapshot["drop_p"],
                batch_size=_snapshot["batch_size"], lr=_snapshot["lr"],
                n_kv_heads=int(_snapshot["n_kv_heads"]) if _snapshot["n_kv_heads"] else None,
                qk_norm=_snapshot["qk_norm"], tie_weights=_snapshot["tie_weights"],
                epochs_per_session=_snapshot["epochs_per_session"],
                action_loss_weight=_snapshot["action_loss_weight"],
                state_loss_weight=_snapshot["state_loss_weight"],
                return_loss_weight=_snapshot["return_loss_weight"],
            ))
        print(f"📋 [committed] {CFG['n_block']} blk, {CFG['h_dim']} dim, ctx={CFG['context_len']}")
    else:
        print("📋 [defaults] Click 'Apply config' to commit your settings")
    return (CFG,)


@app.cell
def _(CFG, TrajectoryDataset, df, torch):
    _stride = CFG["context_len"] // 2
    print(f"🔧 STRIDE = {_stride} (context_len={CFG['context_len']}) — expect ~{len(df) // _stride // CFG['batch_size']:,} batches/epoch")
    ds = TrajectoryDataset(
        data=df,
        context_length=CFG["context_len"],
        state_dim=CFG["state_dim"],
        act_dim=CFG["act_dim"],
        discount_factor=CFG["discount_factor"],
        stride=_stride,
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
    print(f"🎯 Actual batches/epoch: {len(train_ds) // CFG['batch_size']:,}")
    return train_loader, val_loader


@app.cell
def _(CFG, CHECKPOINT_DIR, CHECKPOINT_PATH, DecisionTransformer, time, torch):
    def load_or_create_model(cfg, device, fresh=False):
        model = DecisionTransformer(
            state_dim=cfg["state_dim"], act_dim=cfg["act_dim"],
            n_block=cfg["n_block"], h_dim=cfg["h_dim"],
            context_len=cfg["context_len"],
            n_heads=cfg["n_heads"], drop_p=cfg["drop_p"],
            max_timestep=cfg["max_timestep"],
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
            "return_scale": CFG["return_scale"], "timestamp": time.time()}
        if scaler: payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(payload, CHECKPOINT_PATH)
        freq = CFG.get("checkpoint_every_n_batches", 500)
        if freq and step and step % freq == 0:
            torch.save(payload, CHECKPOINT_DIR / f"checkpoint_step_{step}.pt")
            print(f"💾 Step checkpoint: step_{step}.pt")

    print("✅ Training helpers ready")
    return load_or_create_model, save_checkpoint


@app.cell
def _(
    BEST_MODEL_PATH,
    CFG,
    F,
    fresh_start,
    load_or_create_model,
    mo,
    save_checkpoint,
    time,
    torch,
    train_btn,
    train_loader,
    val_loader,
):
    session_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl, vl, bv, gs = [], [], float("inf"), 0

    if not train_btn.value:
        mo.stop(True, mo.md("> Click **Start Training**."))

    print("=" * 60)
    print(f"🎯 IMPACT DT — {device} | {CFG['n_block']} blk {CFG['h_dim']} dim")
    print("=" * 60)

    model, opt, sch, se, gs, tl, vl, bv, ss = load_or_create_model(CFG, device, fresh=fresh_start.value)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if scaler and ss:
        try: scaler.load_state_dict(ss)
        except: pass

    total_b = len(train_loader)
    ckpt_freq = CFG.get("checkpoint_every_n_batches", 500)
    rs = CFG["return_scale"]

    for epoch in range(se, se + CFG["epochs_per_session"]):
        model.train()
        loss_acc = al_acc = sl_acc = rl_acc = count = 0.0
        for bi, batch in enumerate(train_loader):
            st = batch["states"].to(device)
            ac = batch["actions"].to(device)
            rt = batch["rtgs"].to(device) / rs
            ts = batch["timesteps"].to(device)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                rp, sp, ap = model(st, rt, ts, ac)
                a_loss = F.mse_loss(ap, ac)
                s_loss = F.mse_loss(sp, st)
                r_loss = F.mse_loss(rp, rt)
                loss = (CFG["action_loss_weight"] * a_loss
                        + CFG["state_loss_weight"] * s_loss
                        + CFG["return_loss_weight"] * r_loss)

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
                vr = batch["rtgs"].to(device) / rs
                vt = batch["timesteps"].to(device)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                    rp, sp, ap = model(vs, vr, vt, va)
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
    return bv, device, gs, session_start, tl, vl


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
    | Model | `{CFG['n_block']} blk {CFG['h_dim']} dim ctx={CFG['context_len']}` |
    """)
    return


@app.cell
def _(
    CFG,
    CHECKPOINT_PATH,
    DecisionTransformer,
    HfApi,
    Path,
    hf_repo_id,
    hf_token_input,
    mo,
    os,
    torch,
    upload_btn,
):
    mo.stop(not upload_btn.value, mo.md("Press **Upload**."))
    repo_id = hf_repo_id.value.strip()
    token = (hf_token_input.value or os.environ.get("HF_TOKEN", "")).strip()
    if not repo_id: raise ValueError("Provide a HF repo ID.")
    if not token: raise ValueError("Provide a HF token.")

    src = (Path("/workspace/dt_checkpoints/best_model.pt") if Path("/workspace/dt_checkpoints/best_model.pt").exists()
           else CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else None)
    if src is None: raise FileNotFoundError("No checkpoint found.")

    upload_model = DecisionTransformer(
        state_dim=CFG["state_dim"], act_dim=CFG["act_dim"],
        n_block=CFG["n_block"], h_dim=CFG["h_dim"],
        context_len=CFG["context_len"],
        n_heads=CFG["n_heads"], drop_p=CFG["drop_p"],
        max_timestep=CFG["max_timestep"],
        n_kv_heads=CFG.get("n_kv_heads"), qk_norm=CFG.get("qk_norm", False),
        tie_weights=CFG.get("tie_weights", False),
    )
    ck = torch.load(src, map_location="cpu")
    upload_model.load_state_dict(ck.get("model_state_dict", ck))
    meta = {k: v for k, v in ck.items() if k != "model_state_dict"}
    meta["return_scale"] = CFG["return_scale"]

    upload_path = Path("/workspace/dt_checkpoints/upload_model.pt")
    torch.save({"model_state_dict": upload_model.state_dict(), **meta}, upload_path)
    HfApi().upload_file(path_or_fileobj=str(upload_path), path_in_repo="aemo_dt_fcas_model.pt",
                        repo_id=repo_id, repo_type="model", token=token)
    print(f"✅ Uploaded aemo_dt_fcas_model.pt → {repo_id}")
    return


@app.cell
def _(mo):
    mo.md(f"""
    > **Post-train:** copy the checkpoint to `models/aemo/dt/hf_v2_modern/` and run
    > `scripts/phase3_impact_eval.py` to compare the impact-aware DT vs the
    > pretrained modern v2 on identity + impact surfaces (incl. Oracle_MI).
    """)
    return


@app.cell
def _(TrajectoryDataset):
    import inspect, textwrap

    _src = inspect.getsource(TrajectoryDataset.__init__)
    print(textwrap.dedent(_src)[:4000])
    return


@app.cell
def _(mo):
    import inspect as _inspect
    print("marimo version:", mo.__version__)
    print("mo.ui.form sig:", _inspect.signature(mo.ui.form))
    print("mo.ui.button sig:", _inspect.signature(mo.ui.button))
    print("mo.ui.run_button sig:", _inspect.signature(mo.ui.run_button))
    print("has mo.ui.dictionary:", hasattr(mo.ui, "dictionary"))
    return


if __name__ == "__main__":
    app.run()
