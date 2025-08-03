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
    # Add a learning rate scheduler (StepLR: halve LR every 10 epochs by default)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
                # Gradient clipping (max norm 0.1)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
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
                # Gradient clipping (max norm 0.1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                loss_to_log = loss_a.detach().cpu().item()
                loss_value = loss.detach().cpu().item()
            total_loss += loss_value * states.size(0)
            # Log action losses
            log_losses.append(loss_to_log)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")
        # Step the learning rate scheduler
        scheduler.step()

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

