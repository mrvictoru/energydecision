"""
Forecast-conditioned Decision Transformer for AEMO energy trading.

Extends the modern v2 DT (8×768 GQA) with explicit forecast tokens:
- RoPE enabled by default
- forecast_len parameter controls how many future timesteps are prepended
- Inference-time forecast construction from aemo_data look-ahead
- Dataset class yields forecast windows alongside history windows

Usage:
    model = ForecastDecisionTransformer(
        state_dim=18, act_dim=9, context_len=210, forecast_len=48, ...
    )
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Building blocks (copied from modern DT)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x * self.scale


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop_p: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 4096, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires even dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, seq_len: int):
        if seq_len > self.cos.shape[0]:
            raise ValueError(f"seq_len {seq_len} exceeds RoPE cache {self.cos.shape[0]}")
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=q.dtype, device=q.device).unsqueeze(0).unsqueeze(0)
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    q = torch.cat([q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], dim=-1)
    k = torch.cat([k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], dim=-1)
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        h_dim: int,
        max_T: int,
        n_heads: int,
        drop_p: float,
        rope_enabled: bool = False,
        rope_max_position: int = 4096,
        rope_base: float = 10000.0,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert h_dim % n_heads == 0, "h_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads
        self.drop_p = drop_p
        self.rope_enabled = rope_enabled

        if n_kv_heads is None:
            n_kv_heads = n_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads

        if rope_enabled:
            if self.head_dim % 2 != 0:
                raise ValueError("RoPE requires even head_dim")
            self.rotary = RotaryEmbedding(
                self.head_dim, max_position=rope_max_position, base=rope_base
            )

        self.q_proj = nn.Linear(h_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h_dim, n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(h_dim, h_dim, bias=False)
        self.proj_drop = nn.Dropout(drop_p)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        mask = torch.tril(torch.ones(max_T, max_T)).view(1, 1, max_T, max_T)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope_enabled:
            cos, sin = self.rotary(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        causal = self.mask[:, :, :T, :T].bool()
        if key_padding_mask is not None:
            kp = key_padding_mask.view(B, 1, 1, T).to(dtype=torch.bool)
            combined = causal & kp
            attn_mask = torch.zeros((B, 1, T, T), device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~combined, float("-inf"))
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.drop_p if self.training else 0.0,
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.drop_p if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj_drop(self.proj(y))
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y


class ModernBlock(nn.Module):
    def __init__(
        self,
        h_dim: int,
        max_T: int,
        n_heads: int,
        drop_p: float,
        rope_enabled: bool = False,
        rope_max_position: int = 4096,
        rope_base: float = 10000.0,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(h_dim)
        self.attn = CausalSelfAttention(
            h_dim, max_T, n_heads, drop_p,
            rope_enabled=rope_enabled,
            rope_max_position=rope_max_position,
            rope_base=rope_base,
            n_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
        )
        self.norm2 = RMSNorm(h_dim)
        self.ffn = SwiGLU(h_dim, 4 * h_dim, drop_p)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Forecast Decision Transformer
# ---------------------------------------------------------------------------

class ForecastDecisionTransformer(nn.Module):
    """Decision Transformer with forecast token prefix and RoPE enabled by default.

    The model processes a sequence of (forecast_tokens, history_tokens) where:
    - forecast_tokens:  (rtg, state, pad) × forecast_len
    - history_tokens:   (rtg, state, action) × context_len

    Forecast tokens are prepended as a prefix so history tokens can attend
    to them via the existing causal mask.
    """

    def __init__(
        self,
        state_dim: int = 18,
        act_dim: int = 9,
        n_block: int = 8,
        h_dim: int = 768,
        context_len: int = 210,
        n_heads: int = 12,
        drop_p: float = 0.15,
        max_timestep: int = 100000,
        forecast_len: int = 48,
        rope_enabled: bool = True,
        rope_max_position: int = 4096,
        rope_base: float = 10000.0,
        n_kv_heads: int | None = 6,
        qk_norm: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.n_heads = n_heads
        self.n_block = n_block
        self.drop_p = drop_p
        self.max_timestep = max_timestep
        self.rope_enabled = rope_enabled

        total_seq_len = 3 * (context_len + forecast_len)

        # Transformer blocks
        self.transformer = nn.ModuleList([
            ModernBlock(
                h_dim, total_seq_len, n_heads, drop_p,
                rope_enabled=rope_enabled,
                rope_max_position=rope_max_position,
                rope_base=rope_base,
                n_kv_heads=n_kv_heads,
                qk_norm=qk_norm,
            )
            for _ in range(n_block)
        ])

        # Embeddings
        self.embed_ln = RMSNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_act = nn.Linear(act_dim, h_dim)
        self.embed_forecast_type = nn.Embedding(2, h_dim)  # 0=history, 1=forecast

        # Final norm
        self.ln_f = RMSNorm(h_dim)

        # Prediction heads
        self._tie_weights = tie_weights
        if tie_weights:
            self.pred_rtg = nn.Linear(1, h_dim, bias=False)
            self.pred_state = nn.Linear(state_dim, h_dim, bias=False)
            pred_act_linear = nn.Linear(act_dim, h_dim, bias=False)
            self.pred_rtg.weight = self.embed_rtg.weight
            self.pred_state.weight = self.embed_state.weight
            pred_act_linear.weight = self.embed_act.weight
            self.pred_act = nn.Sequential(pred_act_linear, nn.Tanh())
        else:
            self.pred_rtg = nn.Linear(h_dim, 1, bias=False)
            self.pred_state = nn.Linear(h_dim, state_dim, bias=False)
            self.pred_act = nn.Sequential(
                nn.Linear(h_dim, act_dim, bias=False), nn.Tanh()
            )

        self.return_scale = 1.0
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        forecast_states: torch.Tensor | None = None,
        forecast_rtgs: torch.Tensor | None = None,
        forecast_timesteps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            states:        [B, T, state_dim]  — history states
            actions:       [B, T, act_dim]    — history actions
            returns_to_go: [B, T, 1]          — history RTGs
            timesteps:     [B, T]             — history timesteps
            attention_mask:[B, T]             — history mask (1=valid, 0=pad)
            forecast_states:   [B, F, state_dim] or None
            forecast_rtgs:     [B, F, 1] or None
            forecast_timesteps:[B, F] or None

        Returns:
            (return_preds, state_preds, action_preds)
        """
        B, T, _ = states.shape
        fore_len = self.forecast_len
        device = states.device

        # Sanitize
        states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        returns_to_go = torch.nan_to_num(returns_to_go, nan=0.0, posinf=0.0, neginf=0.0)

        if returns_to_go.dim() == 1:
            returns_to_go = returns_to_go.unsqueeze(0).unsqueeze(-1)
        elif returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        elif returns_to_go.dim() == 3 and returns_to_go.shape[-1] != 1:
            returns_to_go = returns_to_go.unsqueeze(-1)

        timesteps = timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)

        # ------------------------------------------------------------------
        # Build history token stream (same as standard DT)
        # ------------------------------------------------------------------
        time_emb_h = self.embed_timestep(timesteps)
        state_emb = self.embed_state(states) + time_emb_h + self.embed_forecast_type(
            torch.zeros(B, T, dtype=torch.long, device=device)
        )
        rtg_emb = self.embed_rtg(returns_to_go) + time_emb_h + self.embed_forecast_type(
            torch.zeros(B, T, dtype=torch.long, device=device)
        )
        act_emb = self.embed_act(actions) + time_emb_h + self.embed_forecast_type(
            torch.zeros(B, T, dtype=torch.long, device=device)
        )

        h_history = torch.stack([rtg_emb, state_emb, act_emb], dim=1)
        h_history = h_history.permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        # ------------------------------------------------------------------
        # Build forecast token stream (if provided)
        # ------------------------------------------------------------------
        h_forecast = None
        if forecast_states is not None and fore_len > 0:
            f_state = torch.nan_to_num(forecast_states, nan=0.0, posinf=0.0, neginf=0.0)
            f_rtg = torch.nan_to_num(forecast_rtgs, nan=0.0, posinf=0.0, neginf=0.0)

            if f_rtg.dim() == 2:
                f_rtg = f_rtg.unsqueeze(-1)

            # Clamp timesteps
            f_ts = forecast_timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)
            f_time_emb = self.embed_timestep(f_ts)

            # Forecast type embedding (idx=1)
            f_type_idx = torch.ones(B, fore_len, dtype=torch.long, device=device)
            f_type_emb = self.embed_forecast_type(f_type_idx)

            f_state_emb = self.embed_state(f_state) + f_time_emb + f_type_emb
            f_rtg_emb = self.embed_rtg(f_rtg) + f_time_emb + f_type_emb
            # Forecast action slots: zeros (no action, no embedding)
            f_act_pad = torch.zeros(B, fore_len, self.h_dim, device=device)

            h_forecast = torch.stack([f_rtg_emb, f_state_emb, f_act_pad], dim=1)
            h_forecast = h_forecast.permute(0, 2, 1, 3).reshape(B, 3 * fore_len, self.h_dim)

        # ------------------------------------------------------------------
        # Concatenate forecast prefix + history
        # ------------------------------------------------------------------
        if h_forecast is not None:
            h = torch.cat([h_forecast, h_history], dim=1)
        else:
            h = h_history

        h = self.embed_ln(h)

        # Stacked attention mask
        total_seq = h.shape[1]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device)
            attention_mask = attention_mask > 0
            # Ensure mask has batch dim for stacking
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            # Stack three times for (rtg, state, action)
            stacked_mask = torch.stack(
                [attention_mask, attention_mask, attention_mask], dim=1
            )
            stacked_mask = stacked_mask.permute(0, 2, 1).reshape(B, 3 * T)

            if h_forecast is not None:
                # Forecast mask: all valid (1)
                f_mask = torch.ones(B, 3 * fore_len, dtype=torch.bool, device=device)
                full_mask = torch.cat([f_mask, stacked_mask], dim=1)
            else:
                full_mask = stacked_mask
        else:
            full_mask = torch.ones(B, total_seq, dtype=torch.bool, device=device)

        # Pass through blocks
        for block in self.transformer:
            h = block(h, key_padding_mask=full_mask)

        h = self.ln_f(h)

        # ------------------------------------------------------------------
        # Decode predictions
        # ------------------------------------------------------------------
        # h is [B, total_seq, h_dim]. Reshape to (B, T_total, 3, h_dim).
        T_total = total_seq // 3
        h = h.reshape(B, T_total, 3, self.h_dim).permute(0, 2, 1, 3)

        # We only decode predictions from the HISTORY positions.
        # History occupies the LAST T positions in the sequence.
        h_history_tokens = h[:, :, -T:, :]  # [B, 3, T, h_dim]

        if self._tie_weights:
            return_preds = F.linear(
                h_history_tokens[:, 2], self.pred_rtg.weight.t()
            )  # (B, T, 1)
            state_preds = F.linear(
                h_history_tokens[:, 2], self.pred_state.weight.t()
            )  # (B, T, state_dim)
            act_preds = torch.tanh(
                F.linear(h_history_tokens[:, 1], self.pred_act[0].weight.t())
            )  # (B, T, act_dim)
        else:
            return_preds = self.pred_rtg(h_history_tokens[:, 2])
            state_preds = self.pred_state(h_history_tokens[:, 2])
            act_preds = self.pred_act(h_history_tokens[:, 1])

        return_preds = torch.nan_to_num(return_preds, nan=0.0, posinf=0.0, neginf=0.0)
        state_preds = torch.nan_to_num(state_preds, nan=0.0, posinf=0.0, neginf=0.0)
        act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)

        return return_preds, state_preds, act_preds

    # ------------------------------------------------------------------
    # Inference: single-action API
    # ------------------------------------------------------------------

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        forecast_states: torch.Tensor | None = None,
        forecast_rtgs: torch.Tensor | None = None,
        forecast_timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if states.dim() == 2:
            states = states.unsqueeze(0)
            if actions.dim() == 2:
                actions = actions.unsqueeze(0)
            if returns_to_go.dim() == 2:
                returns_to_go = returns_to_go.unsqueeze(0)
            elif returns_to_go.dim() == 1:
                returns_to_go = returns_to_go.unsqueeze(0).unsqueeze(-1)
            if timesteps.dim() == 1:
                timesteps = timesteps.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if forecast_states is not None and forecast_states.dim() == 2:
                forecast_states = forecast_states.unsqueeze(0)
            if forecast_rtgs is not None and forecast_rtgs.dim() == 2:
                forecast_rtgs = forecast_rtgs.unsqueeze(0)
            if forecast_timesteps is not None and forecast_timesteps.dim() == 1:
                forecast_timesteps = forecast_timesteps.unsqueeze(0)

        # Ensure rtgs has trailing dim
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        elif returns_to_go.dim() == 3 and returns_to_go.shape[-1] != 1:
            returns_to_go = returns_to_go.unsqueeze(-1)
        if forecast_rtgs is not None:
            if forecast_rtgs.dim() == 2:
                forecast_rtgs = forecast_rtgs.unsqueeze(-1)
            elif forecast_rtgs.dim() == 3 and forecast_rtgs.shape[-1] != 1:
                forecast_rtgs = forecast_rtgs.unsqueeze(-1)

        _, _, act_preds = self.forward(
            states, actions, returns_to_go, timesteps,
            attention_mask=attention_mask,
            forecast_states=forecast_states,
            forecast_rtgs=forecast_rtgs,
            forecast_timesteps=forecast_timesteps,
        )
        act_preds = torch.nan_to_num(act_preds, nan=0.0, posinf=0.0, neginf=0.0)
        action = act_preds[0, -1]
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] > 1:
            action = torch.cat(
                [action[..., :1], torch.clamp(action[..., 1:], 0.0, 1.0)], dim=-1
            )
        return action

    def load_from_checkpoint(
        self, checkpoint_or_state: str | dict, map_location: str | None = None,
        strict: bool = True,
    ) -> None:
        if isinstance(checkpoint_or_state, (str, bytes)):
            ckpt_path = (checkpoint_or_state.decode() if isinstance(checkpoint_or_state, bytes)
                         else str(checkpoint_or_state))
            state = torch.load(ckpt_path, map_location=map_location)
        else:
            state = checkpoint_or_state
        meta = None
        if isinstance(state, dict) and "model_state_dict" in state:
            meta = state.get("meta")
            if meta is None and "return_scale" in state:
                meta = {"return_scale": state.get("return_scale")}
            state = state["model_state_dict"]
        if isinstance(meta, dict) and "return_scale" in meta:
            try:
                rs = float(meta["return_scale"])
                if rs == rs and abs(rs) >= 1e-12:
                    self.return_scale = rs
            except Exception:
                pass
        try:
            self.load_state_dict(state, strict=strict)
        except RuntimeError:
            if strict:
                self.load_state_dict(state, strict=False)
            else:
                raise


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ForecastTrajectoryDataset(Dataset):
    """Dataset for training the ForecastDecisionTransformer.

    Yields sliding windows over episodes. Each item contains:
    - states, actions, rtgs, timesteps, mask for the HISTORY window (T steps)
    - forecast_states, forecast_rtgs, forecast_timesteps for the FORECAST window (F steps after history)
    """

    def __init__(
        self,
        data_path: str | pl.DataFrame,
        context_length: int = 210,
        state_dim: int = 18,
        act_dim: int = 9,
        forecast_len: int = 48,
        discount_factor: float = 0.95,
        min_episode_length: int | None = None,
        forecast_npz_path: str | None = None,
    ):
        self.context_length = context_length
        self.forecast_len = forecast_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.discount_factor = discount_factor

        # Load TTM forecast data if provided
        self._forecast_map: np.ndarray | None = None
        if forecast_npz_path:
            fc = np.load(forecast_npz_path)
            self._forecast_map = fc["forecast_map"]  # [N, F, 6]
            print(f"[Dataset] TTM forecast map loaded: {self._forecast_map.shape}")

         # Load data
        if isinstance(data_path, pl.DataFrame):
            df = data_path
        else:
            needed_cols = ["episode_id", "step", "norm_observation", "action", "reward", "forecast"]
            try:
                df = pl.read_parquet(data_path, columns=needed_cols)
            except Exception:
                # Fallback: some columns may not exist
                df = pl.read_parquet(data_path, columns=needed_cols[:-1])

        self._has_forecast_column = "forecast" in df.columns
        self._has_episode_start = "episode_start" in df.columns

        # Filter rows with mismatched dimensions
        df_clean = df.filter(
            (pl.col("action").list.len() == act_dim)
            & (pl.col("norm_observation").list.len() == state_dim)
        )
        n_removed = len(df) - len(df_clean)
        if n_removed > 0 and n_removed < len(df):
            drop_pct = 100.0 * n_removed / len(df)
            print(f"⚠️ Filtered out {n_removed:,} rows ({drop_pct:.1f}%) with mismatched dims")

        self.episodes: list[dict[str, Any]] = []
        self.indices: list[tuple[int, int]] = []

        total_window = context_length + forecast_len
        min_len = min_episode_length or total_window

        for eid in df_clean["episode_id"].unique().to_list():
            grp = df_clean.filter(pl.col("episode_id") == eid)
            states = np.stack(grp["norm_observation"].to_list()).astype(np.float32)
            actions = np.stack(grp["action"].to_list()).astype(np.float32)
            rewards = np.array(grp["reward"].to_list(), dtype=np.float32)
            timesteps = np.array(grp["step"].to_list(), dtype=np.int64)
            rtgs = self._compute_rtgs(rewards)

            # Load pre-computed forecasts if available
            if self._has_forecast_column:
                forecasts = np.array(grp["forecast"].to_list(), dtype=np.float32)
                # forecasts shape: [L, forecast_len, n_channels]
            else:
                forecasts = None

            ep_len = states.shape[0]
            if ep_len < min_len:
                continue

            ep_dict: dict[str, Any] = {
                "states": states,
                "actions": actions,
                "rtgs": rtgs,
                "timesteps": timesteps,
                "length": ep_len,
                "forecasts": forecasts,
            }
            self.episodes.append(ep_dict)

            stride = max(1, context_length // 2)
            for start_idx in range(0, ep_len - total_window + 1, stride):
                self.indices.append((len(self.episodes) - 1, start_idx))

    @staticmethod
    def _compute_rtgs(rewards: np.ndarray, gamma: float = 0.95) -> np.ndarray:
        rtgs = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + gamma * running
            rtgs[i] = running
        return rtgs

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start_idx = self.indices[idx]
        ep = self.episodes[ep_idx]
        T = self.context_length
        F = self.forecast_len

        # History window
        h_end = start_idx + T
        h_len = min(T, ep["length"] - start_idx)
        h_actual = min(h_len, T)

        # Forecast window: F steps AFTER the history window
        f_start = h_end
        f_end = f_start + F
        f_actual = max(0, min(F, ep["length"] - f_start))

        # Build tensors
        states = np.zeros((T, self.state_dim), dtype=np.float32)
        actions = np.zeros((T, self.act_dim), dtype=np.float32)
        rtgs = np.zeros((T, 1), dtype=np.float32)
        timesteps = np.zeros(T, dtype=np.int64)
        mask = np.zeros(T, dtype=np.float32)

        # Right-align: the valid history goes at the END of the buffer
        states[-h_actual:] = ep["states"][start_idx:start_idx + h_actual]
        actions[-h_actual:] = ep["actions"][start_idx:start_idx + h_actual]
        rtgs[-h_actual:, 0] = ep["rtgs"][start_idx:start_idx + h_actual]
        timesteps[-h_actual:] = ep["timesteps"][start_idx:start_idx + h_actual]
        mask[-h_actual:] = 1.0

        # Build forecast window
        fc_map = self._forecast_map
        ep_start = ep.get("episode_start")

        if fc_map is not None and ep_start is not None and f_actual > 0:
            # TTM forecast from npz: global_idx = episode_start + step
            f_states = np.zeros((F, 18), dtype=np.float32)
            for fi in range(F):
                src_step = h_end + fi
                g_idx = int(ep_start) + int(src_step)
                if 0 <= g_idx < len(fc_map):
                    f_states[fi, 5:11] = fc_map[g_idx, 0, :6]  # [F, 6] → take first
            f_rtgs = np.zeros((F, 1), dtype=np.float32)
            f_timesteps = np.zeros(F, dtype=np.int64)
            f_mask = np.ones(F, dtype=np.float32)
        elif self._has_forecast_column and ep.get("forecasts") is not None:
            fc = ep["forecasts"]  # [L, F * 6] — flat per row
            n_chan = 6
            f_states = np.zeros((F, 18), dtype=np.float32)
            if f_actual > 0:
                for fi in range(F):
                    src_idx = min(f_start + fi, len(fc) - 1)
                    flat_row = fc[src_idx]
                    fc_vals = flat_row.reshape(-1, n_chan)[min(fi, flat_row.size // n_chan - 1)]
                    f_states[fi, 5:5 + n_chan] = fc_vals
            f_rtgs = np.zeros((F, 1), dtype=np.float32)
            f_timesteps = np.zeros(F, dtype=np.int64)
            f_mask = np.ones(F, dtype=np.float32) if f_actual > 0 else np.zeros(F, dtype=np.float32)
        else:
            # Fallback: perfect foresight from episode's own future steps
            f_states = np.zeros((F, self.state_dim), dtype=np.float32)
            f_rtgs = np.zeros((F, 1), dtype=np.float32)
            f_timesteps = np.zeros(F, dtype=np.int64)
            f_mask = np.zeros(F, dtype=np.float32)
            if f_actual > 0:
                f_states[:f_actual] = ep["states"][f_start:f_start + f_actual]
                f_rtgs[:f_actual, 0] = ep["rtgs"][f_start:f_start + f_actual]
                f_timesteps[:f_actual] = ep["timesteps"][f_start:f_start + f_actual]
                f_mask[:f_actual] = 1.0

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rtgs": torch.from_numpy(rtgs),
            "timesteps": torch.from_numpy(timesteps),
            "mask": torch.from_numpy(mask),
            "forecast_states": torch.from_numpy(f_states),
            "forecast_rtgs": torch.from_numpy(f_rtgs),
            "forecast_timesteps": torch.from_numpy(f_timesteps),
            "forecast_mask": torch.from_numpy(f_mask),
        }
