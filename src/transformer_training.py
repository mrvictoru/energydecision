import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional
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

        df = pl.read_parquet(data_path)

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
) -> tuple:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # log the start time
    start_time = datetime.datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    log_losses = []
    val_losses = []

    # create DataLoader after device is chosen so pin_memory can be set appropriately
    pin_memory = True if device.startswith("cuda") else False
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)

    model = model.to(device)
    
    # Save return_scale to model for inference consistency
    model.return_scale = return_scale

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Add a learning rate scheduler (StepLR: halve LR every 10 epochs by default)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    use_amp = device.startswith("cuda")
    if use_amp:
        scaler = GradScaler('cuda')
    else:
        scaler = None

    def _is_finite(t: torch.Tensor) -> bool:
        # empty tensors are considered finite
        try:
            return torch.isfinite(t).all().item()
        except Exception:
            return False

    start_epoch = 1

    def _save_checkpoint(epoch: int, segment: int = -1) -> None:
        if not checkpoint_path:
            return
        ckpt_dir = os.path.dirname(checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "segment": segment,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "log_losses": log_losses,
            "batch_size": batch_size,
            "lr": lr,
            "use_amp": use_amp,
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)
        if segment >= 0:
            print(f"Checkpoint saved to {checkpoint_path} — epoch {epoch}, segment {segment+1}")
        else:
            print(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}")

    last_epoch_saved = 0
    last_segment_saved = -1
    resume_from_segment = 0

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        log_losses = checkpoint.get("log_losses", [])  # type: ignore[assignment]
        last_epoch_saved = checkpoint.get("epoch", 0)
        last_segment_saved = checkpoint.get("segment", -1)
        if use_amp and scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
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

    for epoch in range(start_epoch, epochs+1):
        model.train()
        total_loss = 0.0
        skipped_batches = 0
        total_batches = len(loader)
        segment_boundaries: list[int] = []
        if checkpoints_per_epoch > 0 and total_batches > 0:
            segment_size = max(1, math.ceil(total_batches / checkpoints_per_epoch))
            segment_boundaries = [min(total_batches, segment_size * (i + 1)) for i in range(checkpoints_per_epoch)]
        segment_idx = 0
        skip_until_batch = 0
        if (
            resume
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
        batch_action_loss_sum = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(progress_bar):
            if skip_until_batch and batch_idx < skip_until_batch:
                continue
            # move tensors and ensure float dtype
            states    = batch["states"].to(device).float()
            actions   = batch["actions"].to(device).float()
            rtgs      = batch["rtgs"].to(device).float()
            timesteps = batch["timesteps"].to(device)
            mask      = batch["mask"].to(device)
            
            # Apply return_scale to RTGs if specified
            if return_scale != 1.0:
                rtgs = rtgs / return_scale

            # quick sanity checks on inputs
            if not (_is_finite(states) and _is_finite(actions) and _is_finite(rtgs)):
                skipped_batches += 1
                progress_bar.write(f"Skipping batch {batch_idx}: NaN/Inf in inputs")
                continue

            optimizer.zero_grad()
            m = mask.unsqueeze(-1).float()
            valid_count = m.sum()
            if valid_count.item() == 0:
                skipped_batches += 1
                progress_bar.write(f"Skipping batch {batch_idx}: zero valid mask entries")
                continue

            # compute predictions and losses inside AMP/autocast if available
            try:
                if use_amp and scaler is not None:
                    # device_type for autocast: 'cuda' or 'cpu'
                    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
                    with autocast(device_type=device_type):
                        ret_pred, state_pred, act_pred = model(states, rtgs, timesteps, actions, attention_mask=mask)
                        if not (_is_finite(ret_pred) and _is_finite(state_pred) and _is_finite(act_pred)):
                            skipped_batches += 1
                            progress_bar.write(f"Skipping batch {batch_idx}: NaN/Inf in model outputs")
                            continue
                        loss_r = F.mse_loss(ret_pred   * m, rtgs   * m, reduction="sum") / valid_count
                        loss_s = F.mse_loss(state_pred * m, states * m, reduction="sum") / valid_count
                        loss_a = F.mse_loss(act_pred   * m, actions* m, reduction="sum") / valid_count
                        loss = action_loss_weight * loss_a + state_loss_weight * loss_s + return_loss_weight * loss_r

                    if not torch.isfinite(loss):
                        skipped_batches += 1
                        progress_bar.write(f"Skipping batch {batch_idx}: non-finite loss")
                        continue

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()
                    loss_to_log = loss_a.detach().cpu().item()
                    loss_value = loss.detach().cpu().item()
                else:
                    ret_pred, state_pred, act_pred = model(states, rtgs, timesteps, actions, attention_mask=mask)
                    if not (_is_finite(ret_pred) and _is_finite(state_pred) and _is_finite(act_pred)):
                        skipped_batches += 1
                        progress_bar.write(f"Skipping batch {batch_idx}: NaN/Inf in model outputs")
                        continue
                    loss_r = F.mse_loss(ret_pred   * m, rtgs   * m, reduction="sum") / valid_count
                    loss_s = F.mse_loss(state_pred * m, states * m, reduction="sum") / valid_count
                    loss_a = F.mse_loss(act_pred   * m, actions* m, reduction="sum") / valid_count
                    loss = action_loss_weight * loss_a + state_loss_weight * loss_s + return_loss_weight * loss_r

                    if not torch.isfinite(loss):
                        skipped_batches += 1
                        progress_bar.write(f"Skipping batch {batch_idx}: non-finite loss")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    loss_to_log = loss_a.detach().cpu().item()
                    loss_value = loss.detach().cpu().item()
            except Exception as e:
                skipped_batches += 1
                progress_bar.write(f"Skipping batch {batch_idx}: exception during forward/backward: {e}")
                continue

            total_loss += loss_value * states.size(0)
            batch_action_loss_sum += loss_to_log
            batch_count += 1
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "skipped": skipped_batches})

            if checkpoints_per_epoch > 0 and segment_boundaries:
                while (
                    segment_idx < checkpoints_per_epoch
                    and batch_idx + 1 >= segment_boundaries[segment_idx]
                ):
                    if checkpoint_path:
                        _save_checkpoint(epoch, segment_idx)
                    segment_idx += 1

        avg_loss = total_loss / max(1, len(ds) - skipped_batches)
        avg_action_loss = batch_action_loss_sum / max(1, batch_count)
        log_losses.append(avg_action_loss)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f} — Action Loss: {avg_action_loss:.6f} — Skipped batches: {skipped_batches}")
        # Step the learning rate scheduler
        scheduler.step()

        # Validation loss check
        if val_ds is not None:
            model.eval()
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
            val_total_loss = 0.0
            val_skipped = 0
            with torch.no_grad():
                for batch in val_loader:
                    states    = batch["states"].to(device).float()
                    actions   = batch["actions"].to(device).float()
                    rtgs      = batch["rtgs"].to(device).float()
                    timesteps = batch["timesteps"].to(device)
                    mask      = batch["mask"].to(device)
                    
                    # Apply return_scale to RTGs if specified
                    if return_scale != 1.0:
                        rtgs = rtgs / return_scale
                    
                    m = mask.unsqueeze(-1).float()
                    valid_count = m.sum()
                    if valid_count.item() == 0:
                        val_skipped += 1
                        continue
                    ret_pred, state_pred, act_pred = model(states, rtgs, timesteps, actions, attention_mask=mask)
                    loss_r = F.mse_loss(ret_pred   * m, rtgs   * m, reduction="sum") / valid_count
                    loss_s = F.mse_loss(state_pred * m, states * m, reduction="sum") / valid_count
                    loss_a = F.mse_loss(act_pred   * m, actions* m, reduction="sum") / valid_count
                    loss = action_loss_weight * loss_a + state_loss_weight * loss_s + return_loss_weight * loss_r
                    val_total_loss += loss.item() * states.size(0)
            avg_val_loss = val_total_loss / max(1, len(val_ds) - val_skipped)
            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch}/{epochs} — Validation Loss: {avg_val_loss:.6f}")

        if resume and epoch == start_epoch:
            resume_from_segment = 0
            resume = False

        if checkpoint_path and (epoch % checkpoint_interval == 0):
            _save_checkpoint(epoch)

    # log end time
    end_time = datetime.datetime.now()
    print(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")

    # Final checkpoint to capture finished state
    if checkpoint_path:
        _save_checkpoint(epochs, segment=-1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return (model, log_losses, val_losses)

def concat_trajectory_datasets(datasets: list[TrajectoryDataset]) -> ConcatDataset:
    """
    Concatenate multiple TrajectoryDataset instances into a single dataset for training.
    Returns a torch.utils.data.ConcatDataset.
    """
    return ConcatDataset(datasets)

