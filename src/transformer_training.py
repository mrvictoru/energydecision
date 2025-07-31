import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Optional
import os
import torch.nn.functional as F
from decision_transformer import DecisionTransformer
import datetime


class TrajectoryDataset(Dataset):
    """Dataset for Decision Transformer training using normalized observations and trajectories."""

    def __init__(
        self,
        data_path: str,
        context_length: int,
        state_dim: int,
        act_dim: int = 1,
        discount_factor: float = 0.99,
    ):
        """
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

        # Load the entire parquet into memory via Polars
        df = pl.read_parquet(data_path)

        # Group by episode and preprocess
        self.episodes = []
        # iterate unique episode_ids
        for eid in df["episode_id"].unique().to_list():
            grp = df.filter(pl.col("episode_id") == eid)
            states = np.stack(grp["norm_observation"].to_list())  # [L, state_dim]
            actions = np.stack(grp["action"].to_list())           # [L, act_dim]
            rewards = np.array(grp["reward"].to_list(), dtype=np.float32)  # [L]
            timesteps = np.array(grp["step"].to_list(), dtype=np.int64)    # [L]

            # Compute returns-to-go
            rtgs = self._compute_rtgs(rewards)

            self.episodes.append({
                "states": states,
                "actions": actions,
                "rtgs": rtgs,
                "timesteps": timesteps,
            })

    def _compute_rtgs(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted cumulative rewards-to-go for a single episode."""
        rtgs = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.gamma * running
            rtgs[i] = running
        return rtgs

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a sequence of length `context_length`, padded on the left if needed.
        Output dict contains:
            states:   Tensor[T, state_dim]
            actions:  Tensor[T, act_dim]
            rtgs:     Tensor[T, 1]
            timesteps:Tensor[T]
            mask:     Tensor[T] (1=valid, 0=padding)
        """
        ep = self.episodes[idx]
        L = ep["states"].shape[0]
        T = self.context_length

        # Initialize buffers
        states = np.zeros((T, self.state_dim), dtype=np.float32)
        actions = np.zeros((T, self.act_dim), dtype=np.float32)
        rtgs = np.zeros((T, 1), dtype=np.float32)
        timesteps = np.zeros(T, dtype=np.int64)
        mask = np.zeros(T, dtype=np.float32)

        # Copy episode data to the right-aligned window
        start = max(T - L, 0)
        end = start + min(L, T)
        states[start:end] = ep["states"][-T:]
        actions[start:end] = ep["actions"][-T:]
        rtgs[start:end, 0] = ep["rtgs"][-T:]
        timesteps[start:end] = ep["timesteps"][-T:]
        mask[start:end] = 1.0

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rtgs": torch.from_numpy(rtgs),
            "timesteps": torch.from_numpy(timesteps),
            "mask": torch.from_numpy(mask),
        }

# --- Training Function for Decision Transformer ---
from torch.amp import GradScaler, autocast
def train_decision_transformer(
    ds: TrajectoryDataset,
    context_length: int,
    state_dim: int,
    act_dim: int,
    max_timestep: int,
    model: DecisionTransformer,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 10,
    device: Optional[str] = None,
    save_path: str = "../models/dt_model.pt",
) -> tuple:

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # log the start time
    start_time = datetime.datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    log_losses = []

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = device.startswith("cuda")
    if use_amp:
        scaler = GradScaler('cuda')
    else:
        scaler = None

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            states    = batch["states"].to(device)
            actions   = batch["actions"].to(device)
            rtgs      = batch["rtgs"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask      = batch["mask"].to(device)

            optimizer.zero_grad()
            if use_amp and scaler is not None:
                with autocast('cuda'):
                    ret_pred, state_pred, act_pred = model(states, rtgs, timesteps, actions)
                    m = mask.unsqueeze(-1).float()
                    loss_r = F.mse_loss(ret_pred   * m, rtgs   * m, reduction="sum") / m.sum()
                    loss_s = F.mse_loss(state_pred * m, states * m, reduction="sum") / m.sum()
                    loss_a = F.mse_loss(act_pred   * m, actions* m, reduction="sum") / m.sum()
                    loss = loss_r + loss_s + loss_a
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_to_log = loss_a.detach().cpu().item()
                loss_value = loss.detach().cpu().item()
            else:
                ret_pred, state_pred, act_pred = model(states, rtgs, timesteps, actions)
                m = mask.unsqueeze(-1).float()
                loss_r = F.mse_loss(ret_pred   * m, rtgs   * m, reduction="sum") / m.sum()
                loss_s = F.mse_loss(state_pred * m, states * m, reduction="sum") / m.sum()
                loss_a = F.mse_loss(act_pred   * m, actions* m, reduction="sum") / m.sum()
                loss = loss_r + loss_s + loss_a
                loss.backward()
                optimizer.step()
                loss_to_log = loss_a.detach().cpu().item()
                loss_value = loss.detach().cpu().item()
            total_loss += loss_value * states.size(0)
            # Log action losses
            log_losses.append(loss_to_log)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")

    # log end time
    end_time = datetime.datetime.now()
    print(f"Training completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {end_time - start_time}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return (model, log_losses)

def concat_trajectory_datasets(datasets: list[TrajectoryDataset]) -> ConcatDataset:
    """
    Concatenate multiple TrajectoryDataset instances into a single dataset for training.
    Returns a torch.utils.data.ConcatDataset.
    """
    return ConcatDataset(datasets)

