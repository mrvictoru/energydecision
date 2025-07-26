import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


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
def train_decision_transformer(
    data_path: str,
    context_length: int,
    state_dim: int,
    act_dim: int,
    max_timestep: int,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 10,
    device: Optional[str] = None,
    save_path: str = "models/dt_model.pt",
) -> torch.nn.Module:
    import os
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from decision_transformer import DecisionTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = TrajectoryDataset(
        data_path=data_path,
        context_length=context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        discount_factor=0.99,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_block=2,
        h_dim=128,
        context_len=context_length,
        n_heads=8,
        drop_p=0.1,
        max_timestep=max_timestep,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            ret_pred, state_pred, act_pred = model(states, rtgs, timesteps, actions)

            m = mask.unsqueeze(-1).float()
            loss_r = F.mse_loss(ret_pred   * m, rtgs   * m, reduction="sum") / m.sum()
            loss_s = F.mse_loss(state_pred * m, states * m, reduction="sum") / m.sum()
            loss_a = F.mse_loss(act_pred   * m, actions* m, reduction="sum") / m.sum()
            loss = loss_r + loss_s + loss_a

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * states.size(0)

        avg_loss = total_loss / len(ds)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

if __name__ == "__main__":
    # Example usage (adjust paths & dims accordingly)
    train_decision_transformer(
        data_path="data/rule_all_episode_logs.parquet",
        context_length=16,
        state_dim=12,
        act_dim=1,
        max_timestep=4096,
        batch_size=64,
        lr=3e-4,
        epochs=20,
        save_path="models/dt_model.pt",
    )
