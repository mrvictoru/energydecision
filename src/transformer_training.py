import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional, Any
import json
import os
import torch.nn.functional as F
from decision_transformer import DecisionTransformer
import datetime
import math
from tqdm.auto import tqdm




class TrajectoryDataset(Dataset):
    """
    Dataset for Decision Transformer training using normalized observations and trajectories, using sliding windows.
    Each item is a window of length `context_length` from a single episode, right-padded if at the start of an episode.
    """

    def __init__(
        self,
        data_path: str,
        context_length: int,
        state_dim: int,
        act_dim: int = 1,
        discount_factor: float = 0.99,
    ):
        """
        Loads the dataset from a parquet file, groups by episode, and builds a list of all possible sliding windows.
        Each window is a contiguous sequence of up to `context_length` steps from a single episode.
        Args:
            data_path: Path to a parquet file with trajectory data. Expects columns [episode_id, step, norm_observation, action, reward].
            context_length: Fixed length of each sequence/window (T).
            state_dim: Dimensionality of the normalized observation vector.
            act_dim: Dimensionality of the action vector.
            discount_factor: Gamma for computing discounted returns-to-go.
        """
        self.context_length = context_length
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = discount_factor

        # Load only the columns needed for DT training to reduce RAM usage.
        needed_cols = ["episode_id", "step", "norm_observation", "action", "reward"]
        try:
            df = pl.read_parquet(data_path, columns=needed_cols)
        except TypeError:
            # Older polars versions may not support `columns=` for parquet.
            df = pl.read_parquet(data_path).select(needed_cols)

        # Store all episode data and build sliding window indices
        self.episodes = []  # List of dicts, one per episode
        self.indices = []   # List of (episode_idx, start_idx) for each window

        for eid in df["episode_id"].unique().to_list():
            grp = df.filter(pl.col("episode_id") == eid)
            states = np.stack(grp["norm_observation"].to_list())  # [L, state_dim]
            actions = np.stack(grp["action"].to_list())           # [L, act_dim]
            rewards = np.array(grp["reward"].to_list(), dtype=np.float32)  # [L]
            timesteps = np.array(grp["step"].to_list(), dtype=np.int64)    # [L]
            rtgs = self._compute_rtgs(rewards)

            ep_len = states.shape[0]
            ep_dict = {
                "states": states,
                "actions": actions,
                "rtgs": rtgs,
                "timesteps": timesteps,
                "length": ep_len,
            }
            self.episodes.append(ep_dict)

            # For each possible window in this episode, store (episode_idx, start_idx)
            for start_idx in range(ep_len):
                self.indices.append((len(self.episodes)-1, start_idx))

    def _compute_rtgs(self, rewards: np.ndarray) -> np.ndarray:
        """
        Compute discounted cumulative rewards-to-go for a single episode.
        Args:
            rewards: np.ndarray of shape [L], the reward at each step in the episode.
        Returns:
            rtgs: np.ndarray of shape [L], where rtgs[i] = sum_{j=i}^L gamma^{j-i} * rewards[j]
        """
        rtgs = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.gamma * running
            rtgs[i] = running
        return rtgs


    def __len__(self) -> int:
        """
        Returns the total number of sliding windows (across all episodes) in the dataset.
        Each window is a possible training sample.
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a padded window of length `context_length` from a single episode.
        The window is right-aligned: if the window is shorter than `context_length`,
        the valid data is at the end and the beginning is zero-padded.
        Args:
            idx: Index into the list of all possible windows (across all episodes).
        Returns:
            dict with keys:
                states:   Tensor[T, state_dim]   (padded window of states)
                actions:  Tensor[T, act_dim]     (padded window of actions)
                rtgs:     Tensor[T, 1]           (padded window of returns-to-go)
                timesteps:Tensor[T]              (padded window of timesteps)
                mask:     Tensor[T]              (1 for valid, 0 for padding)
        """
        ep_idx, start_idx = self.indices[idx]
        ep = self.episodes[ep_idx]
        T = self.context_length
        ep_len = ep["length"]

        # Compute the window (may be partial at the end of the episode)
        end_idx = min(start_idx + T, ep_len)
        window_len = end_idx - start_idx

        # Initialize buffers (zero-padded)
        states = np.zeros((T, self.state_dim), dtype=np.float32)
        actions = np.zeros((T, self.act_dim), dtype=np.float32)
        rtgs = np.zeros((T, 1), dtype=np.float32)
        timesteps = np.zeros(T, dtype=np.int64)
        mask = np.zeros(T, dtype=np.float32)

        # Copy episode data to the right-aligned window
        states[-window_len:] = ep["states"][start_idx:end_idx]
        actions[-window_len:] = ep["actions"][start_idx:end_idx]
        rtgs[-window_len:, 0] = ep["rtgs"][start_idx:end_idx]
        timesteps[-window_len:] = ep["timesteps"][start_idx:end_idx]
        mask[-window_len:] = 1.0

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rtgs": torch.from_numpy(rtgs),
            "timesteps": torch.from_numpy(timesteps),
            "mask": torch.from_numpy(mask),
        }


class NonFiniteParameterError(RuntimeError):
    """Raised when model parameters contain NaN/Inf after an optimizer step."""
    pass

# --- Training Function for Decision Transformer ---
try:
    from torch import amp as _torch_amp  # type: ignore[attr-defined]

    GradScaler = _torch_amp.GradScaler  # type: ignore[attr-defined]
    autocast = _torch_amp.autocast  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - compatibility fallback
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

def train_decision_transformer(
    ds: TrajectoryDataset,
    model: DecisionTransformer,
    batch_size: int = 8,
    lr: float = 1e-4,
    epochs: int = 10,
    device: Optional[str] = None,
    save_path: str = "../models/dt_model.pt",
    best_model_path: Optional[str] = None,
    best_metrics_path: Optional[str] = None,
    checkpoint_path: Optional[str] = "../models/dt_checkpoint.pt",
    resume: bool = False,
    checkpoint_interval: int = 1,
    checkpoints_per_epoch: int = 0,
    val_ds: Optional[TrajectoryDataset] = None,
    action_loss_weight: float = 1.0,
    state_loss_weight: float = 0.1,
    return_loss_weight: float = 0.1,
    weight_decay: float = 1e-4,
    return_scale: float = 1.0,
    amp_mode: str = "auto",
    save_best_divergence_ratio: float = 4.0,
    save_best_min_delta: float = 1e-6,
    save_best_train_weight: float = 0.0,
    num_workers: int = 2,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    return_history: bool = False,
) -> tuple:
    
    # set best_model_path to be save_path with _best.pt suffix if not provided
    if best_model_path is None:
        best_model_path = save_path.replace(".pt", "_best.pt")

    # AMP setup method
    def _resolve_amp_settings(device_str: str, mode: str) -> tuple[bool, Optional[Any]]:
        applicable = device_str.startswith("cuda")
        if mode == "off":
            allowed = False
        elif mode == "on":
            allowed = applicable
        else:
            allowed = applicable
        scaler_instance = GradScaler('cuda') if allowed else None
        return allowed, scaler_instance
    
    GRAD_CLIP_NORM = 0.05

    # Helper to check for finite tensors
    def _is_finite(t: torch.Tensor) -> bool:
        # empty tensors are considered finite
        try:
            return torch.isfinite(t).all().item()
        except Exception:
            return False

    # Helper to decide if AMP should be used for this step
    def _should_use_amp_now() -> bool:
        return amp_allowed and amp_enabled and (scaler is not None)

    # Forward pass and loss computation
    def _forward_and_compute_loss(
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
        use_amp_now: bool,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None, str | None]:
        
        # Apply mask
        mask_bool = mask > 0.5
        mask_bool = mask_bool.to(dtype=torch.bool)
        m = mask_bool.unsqueeze(-1).float()
        valid_count = m.sum()
        # Ensure there is at least one valid entry
        if valid_count.item() == 0:
            return None, "zero valid mask entries"

        # Inner computation function
        def _compute():
            ret_pred, state_pred, act_pred = model(
                states, rtgs, timesteps, actions, attention_mask=mask_bool
            )
            if not (_is_finite(ret_pred) and _is_finite(state_pred) and _is_finite(act_pred)):
                return None, "NaN/Inf in model outputs"
            loss_r = (
                F.mse_loss(ret_pred * m, rtgs * m, reduction="sum") / valid_count
            )
            loss_s = (
                F.mse_loss(state_pred * m, states * m, reduction="sum") / valid_count
            )
            loss_a = (
                F.mse_loss(act_pred * m, actions * m, reduction="sum") / valid_count
            )
            loss = (
                action_loss_weight * loss_a
                + state_loss_weight * loss_s
                + return_loss_weight * loss_r
            )
            if not torch.isfinite(loss):
                return None, "non-finite loss"
            return (loss, loss_a, loss_s, loss_r, valid_count), None

        # Run with or without autocast
        if use_amp_now:
            device_type = "cuda" if device.startswith("cuda") else "cpu"
            with autocast(device_type=device_type):
                return _compute()
        return _compute()

    # Optimizer step function
    def _step_optimizer(loss: torch.Tensor, use_amp_now: bool) -> None:
        if use_amp_now:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

    def _run_validation(epoch: int, segment: int) -> Optional[dict[str, float]]:
        if val_loader is None or val_ds is None:
            return None
        model.eval()
        val_total_loss_sum = 0.0
        val_action_loss_sum = 0.0
        val_state_loss_sum = 0.0
        val_return_loss_sum = 0.0
        val_valid_count_sum = 0.0
        val_skipped_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                states = batch["states"].to(device, non_blocking=non_blocking).float()
                actions = batch["actions"].to(device, non_blocking=non_blocking).float()
                rtgs = batch["rtgs"].to(device, non_blocking=non_blocking).float()
                timesteps = batch["timesteps"].to(device, non_blocking=non_blocking)
                mask = batch["mask"].to(device, non_blocking=non_blocking)

                if return_scale != 1.0:
                    rtgs = rtgs / return_scale

                mask_bool = mask > 0.5
                mask_bool = mask_bool.to(dtype=torch.bool)
                m = mask_bool.unsqueeze(-1).float()
                valid_count = m.sum()
                if valid_count.item() == 0:
                    val_skipped_batches += 1
                    continue
                ret_pred, state_pred, act_pred = model(states, rtgs, timesteps, actions, attention_mask=mask_bool)
                loss_r = F.mse_loss(ret_pred * m, rtgs * m, reduction="sum") / valid_count
                loss_s = F.mse_loss(state_pred * m, states * m, reduction="sum") / valid_count
                loss_a = F.mse_loss(act_pred * m, actions * m, reduction="sum") / valid_count
                loss = action_loss_weight * loss_a + state_loss_weight * loss_s + return_loss_weight * loss_r
                vc = float(valid_count.detach().cpu().item())
                val_total_loss_sum += float(loss.item()) * vc
                val_action_loss_sum += float(loss_a.item()) * vc
                val_state_loss_sum += float(loss_s.item()) * vc
                val_return_loss_sum += float(loss_r.item()) * vc
                val_valid_count_sum += vc

        denom = max(1.0, val_valid_count_sum)
        avg_val_total = val_total_loss_sum / denom
        avg_val_action = val_action_loss_sum / denom
        avg_val_state = val_state_loss_sum / denom
        avg_val_return = val_return_loss_sum / denom

        # Do not append to the epoch-level val series here.
        # Validation is triggered by checkpoints; _save_checkpoint records these stats in `loss_history`
        # and (for segment=-1) also updates the epoch-level series.
        descriptor = f"epoch {epoch}"
        if segment >= 0:
            descriptor += f", segment {segment+1}"
        print(
            f"Validation ({descriptor}) — total: {avg_val_total:.6f} "
            f"(a:{avg_val_action:.6f}, s:{avg_val_state:.6f}, r:{avg_val_return:.6f}) "
            f"valid={int(val_valid_count_sum)} skipped_batches={val_skipped_batches}"
        )
        model.train()
        return {
            "val_total": avg_val_total,
            "val_action": avg_val_action,
            "val_state": avg_val_state,
            "val_return": avg_val_return,
            "val_valid": float(val_valid_count_sum),
            "val_skipped_batches": float(val_skipped_batches),
        }

    def _maybe_save_best(
        epoch: int,
        segment: int,
        train_loss_est: Optional[float],
        val_stats: Optional[dict[str, float]],
    ) -> None:
        nonlocal best_score, best_val_loss, best_train_loss_est
        if not best_model_path:
            return

        val_loss = None if val_stats is None else float(val_stats.get("val_total", float("nan")))
        if val_loss is not None and not math.isfinite(val_loss):
            val_loss = None

        # Choose the score used for "best" tracking.
        if val_loss is None:
            if train_loss_est is None:
                return
            score = float(train_loss_est)
        else:
            score = float(val_loss)
            if train_loss_est is not None:
                score += float(save_best_train_weight) * float(train_loss_est)

        # Divergence guard: don't save if validation is disproportionately worse than training.
        if val_loss is not None and train_loss_est is not None:
            train_floor = max(1e-12, float(train_loss_est))
            ratio = float(val_loss) / train_floor
            if ratio > float(save_best_divergence_ratio):
                return

        if score >= (best_score - float(save_best_min_delta)):
            return

        best_score = score
        if val_loss is not None:
            best_val_loss = float(val_loss)
        if train_loss_est is not None:
            best_train_loss_est = float(train_loss_est)

        best_dir = os.path.dirname(best_model_path)
        if best_dir:
            os.makedirs(best_dir, exist_ok=True)
        torch.save(model.state_dict(), best_model_path)
        if resolved_best_metrics_path:
            try:
                meta_dir = os.path.dirname(resolved_best_metrics_path)
                if meta_dir:
                    os.makedirs(meta_dir, exist_ok=True)
                with open(resolved_best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "best_score": best_score,
                            "best_val_loss": best_val_loss,
                            "best_train_loss_est": best_train_loss_est,
                            "epoch": epoch,
                            "segment": segment,
                            "timestamp": datetime.datetime.now().isoformat(),
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                    )
            except Exception as e:
                print(f"[WARN] Failed writing best-metric state to {resolved_best_metrics_path}: {e}")
        tag = f"epoch {epoch}" + (f", segment {segment+1}" if segment >= 0 else "")
        extra = []
        if train_loss_est is not None:
            extra.append(f"train≈{train_loss_est:.6f}")
        if val_loss is not None:
            extra.append(f"val={val_loss:.6f}")
        extra_str = (" (" + ", ".join(extra) + ")") if extra else ""
        print(f"[BEST] Saved best model weights to {best_model_path} at {tag}{extra_str}")

    # Checkpoint saving function
    def _save_checkpoint(epoch: int, segment: int = -1, train_loss_est: Optional[float] = None) -> None:
        if not checkpoint_path:
            return
        nonlocal amp_enabled, first_checkpoint_saved
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        val_stats = _run_validation(epoch, segment)
        _maybe_save_best(epoch, segment, train_loss_est=train_loss_est, val_stats=val_stats)

        # Record a combined snapshot (useful for plotting training progress).
        snap = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": float(epoch),
            "segment": float(segment),
        }
        if train_loss_est is not None:
            snap["train_total_ema"] = float(train_loss_est)
        if current_train_snapshot:
            snap.update(current_train_snapshot)
        if val_stats:
            snap.update(val_stats)
        loss_history.append(snap)

        # Epoch-level validation series: only update on the "epoch checkpoint" (segment == -1).
        if segment == -1 and val_stats is not None:
            val_losses.append(float(val_stats.get("val_total", float("nan"))))
            val_action_losses.append(float(val_stats.get("val_action", float("nan"))))
            val_state_losses.append(float(val_stats.get("val_state", float("nan"))))
            val_return_losses.append(float(val_stats.get("val_return", float("nan"))))

        checkpoint = {
            "epoch": epoch,
            "segment": segment,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "log_losses": log_losses,
            "train_action_losses": train_action_losses,
            "train_state_losses": train_state_losses,
            "train_return_losses": train_return_losses,
            "val_losses": val_losses,
            "val_action_losses": val_action_losses,
            "val_state_losses": val_state_losses,
            "val_return_losses": val_return_losses,
            "best_score": best_score,
            "best_val_loss": best_val_loss,
            "best_train_loss_est": best_train_loss_est,
            "batch_size": batch_size,
            "lr": lr,
            "use_amp": use_amp,
            "amp_enabled": amp_enabled,
            "loss_history": loss_history,
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)

        # Record that we have a checkpoint on disk and enable AMP if the device supports it.
        first_checkpoint_saved = True
        if amp_allowed and use_amp:
            amp_enabled = True
            print("[INFO] AMP enabled after checkpoint saved on GPU.")
        if segment >= 0:
            print(f"Checkpoint saved to {checkpoint_path} — epoch {epoch}, segment {segment+1}")
        else:
            print(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}")

        # Compact progress print that aligns stored + displayed values.
        msg = f"[PROGRESS] epoch {epoch}"
        if segment >= 0:
            msg += f" seg {segment+1}"
        if current_train_snapshot and "train_total_avg" in current_train_snapshot:
            msg += f" | train total(avg) {current_train_snapshot['train_total_avg']:.6f}"
        if train_loss_est is not None:
            msg += f" (ema {train_loss_est:.6f})"
        if val_stats and "val_total" in val_stats:
            msg += f" | val total {val_stats['val_total']:.6f}"
        print(msg)

    # Checker to ensure model parameters are finite after optimizer step
    def _assert_model_finite() -> None:
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise NonFiniteParameterError(f"Non-finite parameter detected in '{name}' after optimizer step")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # log the start time
    start_time = datetime.datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Epoch-level loss series (these are what we return).
    # Use "total" (combined) loss for training/validation so printed + stored match.
    log_losses: list[float] = []
    val_losses: list[float] = []

    # Component losses (helpful for debugging/plots). Stored in history + checkpoint.
    train_action_losses: list[float] = []
    train_state_losses: list[float] = []
    train_return_losses: list[float] = []
    val_action_losses: list[float] = []
    val_state_losses: list[float] = []
    val_return_losses: list[float] = []

    # Per-checkpoint / per-epoch snapshots (segment checkpoints + epoch checkpoints).
    loss_history: list[dict[str, Any]] = []
    current_train_snapshot: dict[str, float] = {}

    # Best-model tracking state
    best_score = float("inf")
    best_val_loss = float("inf")
    best_train_loss_est = float("inf")

    resolved_best_metrics_path: Optional[str] = None
    if best_model_path:
        resolved_best_metrics_path = best_metrics_path or (best_model_path + ".metrics.json")
        if resolved_best_metrics_path and os.path.exists(resolved_best_metrics_path):
            try:
                with open(resolved_best_metrics_path, "r", encoding="utf-8") as f:
                    best_meta = json.load(f)
                best_score = float(best_meta.get("best_score", best_score))
                best_val_loss = float(best_meta.get("best_val_loss", best_val_loss))
                best_train_loss_est = float(best_meta.get("best_train_loss_est", best_train_loss_est))
                print(f"[INFO] Restored best-metric state from {resolved_best_metrics_path}")
            except Exception as e:
                print(f"[WARN] Could not read best-metric state from {resolved_best_metrics_path}: {e}")

    # create DataLoader after device is chosen so pin_memory can be set appropriately
    pin_memory = True if device.startswith("cuda") else False
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": int(num_workers),
        "pin_memory": pin_memory,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds is not None else None

    # non_blocking transfers only help when using pinned host memory
    non_blocking = bool(pin_memory)

    model = model.to(device)
    
    # Save return_scale to model for inference consistency
    model.return_scale = return_scale
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Add a learning rate scheduler (StepLR: halve LR every 10 epochs by default)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # AMP setup
    amp_allowed, scaler = _resolve_amp_settings(device, amp_mode)
    use_amp = amp_allowed

    amp_enabled = False
    first_checkpoint_saved = False
    if amp_allowed:
        print("[INFO] AMP is enabled once the first checkpoint exists. Set --amp-mode=off to disable.")

    # Initialize training state    
    start_epoch = 1
    last_epoch_saved = 0
    last_segment_saved = -1
    resume_from_segment = 0

    # Load from checkpoint if resuming
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        log_losses = checkpoint.get("log_losses", [])  # type: ignore[assignment]
        train_action_losses = checkpoint.get("train_action_losses", [])  # type: ignore[assignment]
        train_state_losses = checkpoint.get("train_state_losses", [])  # type: ignore[assignment]
        train_return_losses = checkpoint.get("train_return_losses", [])  # type: ignore[assignment]
        val_losses = checkpoint.get("val_losses", [])  # type: ignore[assignment]
        val_action_losses = checkpoint.get("val_action_losses", [])  # type: ignore[assignment]
        val_state_losses = checkpoint.get("val_state_losses", [])  # type: ignore[assignment]
        val_return_losses = checkpoint.get("val_return_losses", [])  # type: ignore[assignment]
        loss_history = checkpoint.get("loss_history", [])  # type: ignore[assignment]
        best_score = float(checkpoint.get("best_score", best_score))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_train_loss_est = float(checkpoint.get("best_train_loss_est", best_train_loss_est))
        last_epoch_saved = checkpoint.get("epoch", 0)
        last_segment_saved = checkpoint.get("segment", -1)
        if amp_allowed and scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if checkpoint.get("amp_enabled", False):
                amp_enabled = True
                first_checkpoint_saved = True
                print("[INFO] AMP restored from checkpoint and enabled.")
        if checkpoints_per_epoch > 0 and last_segment_saved >= 0:
            if last_segment_saved + 1 >= checkpoints_per_epoch:
                start_epoch = last_epoch_saved + 1
                resume_from_segment = 0
            else:
                start_epoch = max(1, last_epoch_saved)
                resume_from_segment = last_segment_saved + 1
        else:
            start_epoch = last_epoch_saved + 1
        if start_epoch > epochs:
            print("Checkpoint epoch exceeds requested epochs; training will perform zero additional epochs.")
        else:
            print(f"Resuming from epoch {start_epoch} of {epochs}")
    elif resume and checkpoint_path:
        print(f"Resume requested but checkpoint not found at {checkpoint_path}; starting fresh.")


    # Training loop with recovery from non-finite parameters
    recovery_attempts = 0
    max_recovery_attempts = 3
    training_finished = False

    # Main training loop
    while not training_finished:
        try:
            for epoch in range(start_epoch, epochs + 1):
                # Set model to training mode
                model.train()
                train_total_loss_sum = 0.0
                train_action_loss_sum = 0.0
                train_state_loss_sum = 0.0
                train_return_loss_sum = 0.0
                train_valid_count_sum = 0.0
                skipped_batches = 0
                total_batches = len(loader)
                segment_boundaries: list[int] = []
                # Determine segment boundaries for checkpoints within epoch
                if checkpoints_per_epoch > 0 and total_batches > 0:
                    segment_size = max(1, math.ceil(total_batches / checkpoints_per_epoch))
                    segment_boundaries = [min(total_batches, segment_size * (i + 1)) for i in range(checkpoints_per_epoch)]
                segment_idx = 0
                skip_until_batch = 0
                # If resuming from a segment within the epoch, skip earlier batches
                if (resume
                    and epoch == start_epoch
                    and checkpoints_per_epoch > 0
                    and resume_from_segment > 0
                    and len(segment_boundaries) >= resume_from_segment
                ):
                    segment_idx = resume_from_segment
                    skip_until_batch = segment_boundaries[resume_from_segment - 1]
                progress_bar = tqdm(
                    loader,
                    desc=f"Epoch {epoch}/{epochs}",
                    leave=False,
                    unit="batch",
                )
                batch_count = 0
                train_loss_ema: Optional[float] = None
                ema_alpha = 0.05
                # Iterate over batches
                for batch_idx, batch in enumerate(progress_bar):
                    if skip_until_batch and batch_idx < skip_until_batch:
                        continue
                    states = batch["states"].to(device, non_blocking=non_blocking).float()
                    actions = batch["actions"].to(device, non_blocking=non_blocking).float()
                    rtgs = batch["rtgs"].to(device, non_blocking=non_blocking).float()
                    timesteps = batch["timesteps"].to(device, non_blocking=non_blocking)
                    mask = batch["mask"].to(device, non_blocking=non_blocking)

                    if return_scale != 1.0:
                        rtgs = rtgs / return_scale
                    # Check for finite inputs
                    if not (_is_finite(states) and _is_finite(actions) and _is_finite(rtgs)):
                        skipped_batches += 1
                        progress_bar.write(f"Skipping batch {batch_idx}: NaN/Inf in inputs")
                        continue
                    # reset gradients
                    optimizer.zero_grad(set_to_none=True)
                    use_amp_now = _should_use_amp_now()
                    # Forward pass and loss computation
                    batch_result, skip_reason = _forward_and_compute_loss(
                        states, actions, rtgs, timesteps, mask, use_amp_now
                    )
                    # Handle skipped batch
                    if batch_result is None:
                        skipped_batches += 1
                        progress_bar.write(f"Skipping batch {batch_idx}: {skip_reason}")
                        continue
                    loss, loss_a, loss_s, loss_r, valid_count = batch_result
                    # Optimizer step (Backward pass) with error handling
                    try:
                        _step_optimizer(loss, use_amp_now)
                        _assert_model_finite()
                    except NonFiniteParameterError:
                        raise
                    except Exception as e:
                        skipped_batches += 1
                        progress_bar.write(f"Skipping batch {batch_idx}: exception during forward/backward: {e}")
                        continue
                    # Logging
                    loss_value = float(loss.detach().cpu().item())
                    loss_a_value = float(loss_a.detach().cpu().item())
                    loss_s_value = float(loss_s.detach().cpu().item())
                    loss_r_value = float(loss_r.detach().cpu().item())
                    vc = float(valid_count.detach().cpu().item())

                    # Track a smooth estimate of training loss for checkpointing decisions.
                    if train_loss_ema is None:
                        train_loss_ema = float(loss_value)
                    else:
                        train_loss_ema = (1.0 - ema_alpha) * float(train_loss_ema) + ema_alpha * float(loss_value)

                    train_total_loss_sum += loss_value * vc
                    train_action_loss_sum += loss_a_value * vc
                    train_state_loss_sum += loss_s_value * vc
                    train_return_loss_sum += loss_r_value * vc
                    train_valid_count_sum += vc
                    batch_count += 1

                    denom = max(1.0, train_valid_count_sum)
                    avg_total_so_far = train_total_loss_sum / denom
                    progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "avg": f"{avg_total_so_far:.4f}", "skipped": skipped_batches})
                    # Checkpointing within epoch
                    if checkpoints_per_epoch > 0 and segment_boundaries:
                        while segment_idx < checkpoints_per_epoch and batch_idx + 1 >= segment_boundaries[segment_idx]:
                            if checkpoint_path:
                                denom = max(1.0, train_valid_count_sum)
                                current_train_snapshot.clear()
                                current_train_snapshot.update(
                                    {
                                        "train_total_avg": train_total_loss_sum / denom,
                                        "train_action_avg": train_action_loss_sum / denom,
                                        "train_state_avg": train_state_loss_sum / denom,
                                        "train_return_avg": train_return_loss_sum / denom,
                                        "train_valid": float(train_valid_count_sum),
                                        "batch_idx": float(batch_idx + 1),
                                    }
                                )
                                _save_checkpoint(epoch, segment_idx, train_loss_est=train_loss_ema)
                            segment_idx += 1
                # End of epoch logging
                denom = max(1.0, train_valid_count_sum)
                avg_total = train_total_loss_sum / denom
                avg_action = train_action_loss_sum / denom
                avg_state = train_state_loss_sum / denom
                avg_return = train_return_loss_sum / denom

                log_losses.append(avg_total)
                train_action_losses.append(avg_action)
                train_state_losses.append(avg_state)
                train_return_losses.append(avg_return)

                print(
                    f"Epoch {epoch}/{epochs} — train total: {avg_total:.6f} "
                    f"(a:{avg_action:.6f}, s:{avg_state:.6f}, r:{avg_return:.6f}) "
                    f"valid={int(train_valid_count_sum)} skipped_batches={skipped_batches}"
                )
                # Step the learning rate scheduler
                scheduler.step()

                if resume and epoch == start_epoch:
                    resume_from_segment = 0
                    resume = False

                if checkpoint_path and (epoch % checkpoint_interval == 0):
                    current_train_snapshot.clear()
                    current_train_snapshot.update(
                        {
                            "train_total_avg": avg_total,
                            "train_action_avg": avg_action,
                            "train_state_avg": avg_state,
                            "train_return_avg": avg_return,
                            "train_valid": float(train_valid_count_sum),
                            "batch_idx": float(total_batches),
                        }
                    )
                    _save_checkpoint(epoch, train_loss_est=train_loss_ema)

            training_finished = True

        # Handle non-finite parameter recovery (restart from last checkpoint)
        except NonFiniteParameterError as err:
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise
            recovery_attempts += 1
            print(f"Non-finite weights detected. Attempting recovery from {checkpoint_path} (attempt {recovery_attempts}).")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            log_losses = checkpoint.get("log_losses", [])  # type: ignore[assignment]
            train_action_losses = checkpoint.get("train_action_losses", [])  # type: ignore[assignment]
            train_state_losses = checkpoint.get("train_state_losses", [])  # type: ignore[assignment]
            train_return_losses = checkpoint.get("train_return_losses", [])  # type: ignore[assignment]
            val_losses = checkpoint.get("val_losses", [])  # type: ignore[assignment]
            val_action_losses = checkpoint.get("val_action_losses", [])  # type: ignore[assignment]
            val_state_losses = checkpoint.get("val_state_losses", [])  # type: ignore[assignment]
            val_return_losses = checkpoint.get("val_return_losses", [])  # type: ignore[assignment]
            loss_history = checkpoint.get("loss_history", [])  # type: ignore[assignment]
            best_score = float(checkpoint.get("best_score", best_score))
            best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
            best_train_loss_est = float(checkpoint.get("best_train_loss_est", best_train_loss_est))
            last_epoch_saved = checkpoint.get("epoch", 0)
            last_segment_saved = checkpoint.get("segment", -1)
            if amp_allowed and scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                if checkpoint.get("amp_enabled", False):
                    amp_enabled = True
                    first_checkpoint_saved = True
                    print("[INFO] AMP restored from checkpoint during recovery and enabled.")

            if checkpoints_per_epoch > 0 and last_segment_saved >= 0:
                if last_segment_saved + 1 >= checkpoints_per_epoch:
                    start_epoch = last_epoch_saved + 1
                    resume_from_segment = 0
                else:
                    start_epoch = max(1, last_epoch_saved)
                    resume_from_segment = last_segment_saved + 1
            else:
                start_epoch = last_epoch_saved + 1
                resume_from_segment = 0

            if start_epoch > epochs:
                print("Checkpoint epoch exceeds requested epochs; training will perform zero additional epochs.")
                training_finished = True
            else:
                print(f"Restarting from checkpoint — epoch {start_epoch} of {epochs}")
                resume = True

            if recovery_attempts >= max_recovery_attempts:
                raise RuntimeError("Exceeded maximum recovery attempts due to recurring non-finite parameters.") from err

            continue

    # log end time
    end_time = datetime.datetime.now()
    print(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")

    # Final checkpoint to capture finished state
    if checkpoint_path:
        # Use a distinct segment id so we don't append a duplicate "epoch validation" entry.
        _save_checkpoint(epochs, segment=-2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    if return_history:
        return (
            model,
            log_losses,
            val_losses,
            {
                "train_action_losses": train_action_losses,
                "train_state_losses": train_state_losses,
                "train_return_losses": train_return_losses,
                "val_action_losses": val_action_losses,
                "val_state_losses": val_state_losses,
                "val_return_losses": val_return_losses,
                "loss_history": loss_history,
            },
        )
    return (model, log_losses, val_losses)

def concat_trajectory_datasets(datasets: list[TrajectoryDataset]) -> ConcatDataset:
    """
    Concatenate multiple TrajectoryDataset instances into a single dataset for training.
    Returns a torch.utils.data.ConcatDataset.
    """
    return ConcatDataset(datasets)

