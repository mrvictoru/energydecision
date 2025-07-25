import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset


class DecisionTransformerDataset(Dataset):
    """Dataset for Decision Transformer training using normalized observations."""

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
