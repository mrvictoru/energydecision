"""
Synthetic FCAS generators.

Phase 6 v1 is the existing regime-switching copula baseline.
Phase 6 v2 adds a conditional diffusion generator that learns synthetic
RRP + FCAS trajectories conditioned on exogenous market features while keeping
the same fit/sample surface used by the evaluation harness.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import multivariate_normal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except (ImportError, OSError, ValueError) as exc:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from fcas_generator_eval import FCAS, FCAS_COLS, LOWER, RAISE

N_STATES = 2
SPIKE_QUANTILE = 0.99
RRP_SPIKE_LOOKBACK = 12
TARGET_COLS = ["RRP"] + FCAS_COLS
CONDITION_COLS = [
    "TOTALDEMAND",
    "GEN_wind",
    "GEN_solar",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "lagged_rrp_spike",
]
FCAS_SERVICE_CAPS = {
    "FCAS_RAISE6SEC": 16_600.0,
    "FCAS_RAISE60SEC": 16_600.0,
    "FCAS_RAISE5MIN": 16_600.0,
    "FCAS_RAISEREG": 999.0,
    "FCAS_LOWER6SEC": 16_600.0,
    "FCAS_LOWER60SEC": 16_600.0,
    "FCAS_LOWER5MIN": 16_600.0,
    "FCAS_LOWERREG": 999.0,
}


def _require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FCAS diffusion requires a working PyTorch runtime in the active environment."
        ) from TORCH_IMPORT_ERROR


def _log1p_matrix(df: pl.DataFrame, family: list[str]) -> np.ndarray:
    return np.column_stack([np.log1p(df[f"FCAS_{s}"].to_numpy().astype(float)) for s in family])


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _signed_expm1(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.expm1(np.abs(x))


def _clamp_fcas_columns(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for col, cap in FCAS_SERVICE_CAPS.items():
        out = out.with_columns(pl.col(col).clip(0.0, cap))
    return out


def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _lagged_rrp_spike_indicator(
    rrp: np.ndarray,
    spike_threshold: float,
    lookback: int = RRP_SPIKE_LOOKBACK,
) -> np.ndarray:
    current_spike = (rrp >= spike_threshold).astype(np.float32)
    lagged = np.zeros_like(current_spike)
    for shift in range(1, lookback + 1):
        shifted = np.zeros_like(current_spike)
        shifted[shift:] = current_spike[:-shift]
        lagged = np.maximum(lagged, shifted)
    return lagged


def _build_condition_frame(
    df: pl.DataFrame,
    *,
    rrp_spike_threshold: float,
    lookback: int = RRP_SPIKE_LOOKBACK,
) -> pl.DataFrame:
    _require_columns(
        df,
        ["RRP", "TOTALDEMAND", "GEN_wind", "GEN_solar", "hour_sin", "hour_cos", "day_sin", "day_cos"],
    )
    rrp = df["RRP"].to_numpy().astype(float)
    return pl.DataFrame(
        {
            "TOTALDEMAND": df["TOTALDEMAND"].to_numpy().astype(np.float32),
            "GEN_wind": df["GEN_wind"].to_numpy().astype(np.float32),
            "GEN_solar": df["GEN_solar"].to_numpy().astype(np.float32),
            "hour_sin": df["hour_sin"].to_numpy().astype(np.float32),
            "hour_cos": df["hour_cos"].to_numpy().astype(np.float32),
            "day_sin": df["day_sin"].to_numpy().astype(np.float32),
            "day_cos": df["day_cos"].to_numpy().astype(np.float32),
            "lagged_rrp_spike": _lagged_rrp_spike_indicator(rrp, rrp_spike_threshold, lookback),
        }
    )


def _build_target_matrix(df: pl.DataFrame) -> np.ndarray:
    frame = _clamp_fcas_columns(df)
    rrp = _signed_log1p(frame["RRP"].to_numpy().astype(float))
    channels = [rrp.astype(np.float32)]
    for col in FCAS_COLS:
        channels.append(np.log1p(frame[col].to_numpy().astype(float)).astype(np.float32))
    return np.column_stack(channels)


def _inverse_target_matrix(values: np.ndarray) -> dict[str, np.ndarray]:
    out = {"RRP": _signed_expm1(values[:, 0])}
    for idx, col in enumerate(FCAS_COLS, start=1):
        out[col] = np.clip(np.expm1(values[:, idx]), 0.0, FCAS_SERVICE_CAPS[col])
    return out


def _window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    tail_start = length - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _slice_or_pad(matrix: np.ndarray, start: int, window_size: int) -> tuple[np.ndarray, int]:
    end = min(start + window_size, matrix.shape[0])
    window = matrix[start:end]
    actual_len = end - start
    if actual_len == window_size:
        return window, actual_len
    pad_count = window_size - actual_len
    pad = np.repeat(window[-1:], pad_count, axis=0)
    return np.concatenate([window, pad], axis=0), actual_len


def _build_windows(matrix: np.ndarray, *, window_size: int, stride: int) -> np.ndarray:
    starts = _window_starts(matrix.shape[0], window_size, stride)
    windows = []
    for start in starts:
        window, _ = _slice_or_pad(matrix, start, window_size)
        windows.append(window.T.astype(np.float32))
    return np.stack(windows, axis=0)


def _sampling_starts(length: int, window_size: int, overlap: int) -> list[int]:
    step = window_size - overlap
    if step <= 0:
        raise ValueError("window_size must be larger than overlap")
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, step))
    tail_start = length - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


if torch is not None:
    class _WindowDataset(Dataset):
        def __init__(self, targets: np.ndarray, conditions: np.ndarray, spike_mask: np.ndarray):
            self.targets = torch.from_numpy(targets)
            self.conditions = torch.from_numpy(conditions)
            self.spike_mask = torch.from_numpy(spike_mask)

        def __len__(self) -> int:
            return int(self.targets.shape[0])

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.targets[idx], self.conditions[idx], self.spike_mask[idx]


    def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


    class _ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, emb_dim: int, groups: int = 8):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.norm1 = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
            self.norm2 = nn.GroupNorm(_group_count(out_channels, groups), out_channels)
            self.emb_proj = nn.Linear(emb_dim, 2 * out_channels)
            self.skip = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )

        def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
            h = self.conv1(x)
            h = self.norm1(h)
            scale, shift = self.emb_proj(emb).chunk(2, dim=1)
            h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
            h = F.silu(h)
            h = self.conv2(h)
            h = self.norm2(h)
            h = F.silu(h)
            return h + self.skip(x)


    class _TemporalUNet(nn.Module):
        def __init__(
            self,
            *,
            target_channels: int,
            condition_channels: int,
            base_channels: int,
            channel_mults: tuple[int, ...],
        ):
            super().__init__()
            self.emb_dim = base_channels * 4
            self.input_proj = nn.Conv1d(
                target_channels + condition_channels,
                base_channels,
                kernel_size=3,
                padding=1,
            )
            self.time_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, self.emb_dim),
                nn.SiLU(),
                nn.Linear(self.emb_dim, self.emb_dim),
            )

            down_channels = [base_channels * mult for mult in channel_mults]
            self.down_blocks = nn.ModuleList()
            in_ch = base_channels
            for out_ch in down_channels:
                self.down_blocks.append(
                    nn.ModuleDict(
                        {
                            "res1": _ResidualBlock(in_ch, out_ch, self.emb_dim),
                            "res2": _ResidualBlock(out_ch, out_ch, self.emb_dim),
                            "down": nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),
                        }
                    )
                )
                in_ch = out_ch

            self.mid1 = _ResidualBlock(in_ch, in_ch, self.emb_dim)
            self.mid2 = _ResidualBlock(in_ch, in_ch, self.emb_dim)

            self.up_blocks = nn.ModuleList()
            for skip_ch in reversed(down_channels):
                self.up_blocks.append(
                    nn.ModuleDict(
                        {
                            "res1": _ResidualBlock(in_ch + skip_ch, skip_ch, self.emb_dim),
                            "res2": _ResidualBlock(skip_ch, skip_ch, self.emb_dim),
                        }
                    )
                )
                in_ch = skip_ch

            self.out_norm = nn.GroupNorm(_group_count(in_ch), in_ch)
            self.out = nn.Conv1d(in_ch, target_channels, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            emb = _sinusoidal_timestep_embedding(timesteps, self.emb_dim)
            emb = self.time_mlp(emb)
            h = self.input_proj(torch.cat([x, cond], dim=1))
            skips = []
            for block in self.down_blocks:
                h = block["res1"](h, emb)
                h = block["res2"](h, emb)
                skips.append(h)
                h = block["down"](h)

            h = self.mid1(h, emb)
            h = self.mid2(h, emb)

            for block in self.up_blocks:
                skip = skips.pop()
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = block["res1"](h, emb)
                h = block["res2"](h, emb)

            h = F.silu(self.out_norm(h))
            return self.out(h)
else:
    class _WindowDataset:
        def __init__(self, *args, **kwargs):
            _require_torch()


    def _sinusoidal_timestep_embedding(*args, **kwargs):
        _require_torch()
        raise AssertionError("unreachable")


    class _ResidualBlock:
        def __init__(self, *args, **kwargs):
            _require_torch()


    class _TemporalUNet:
        def __init__(self, *args, **kwargs):
            _require_torch()


class _Marginal:
    """Per-state, per-service: empirical body ECDF + empirical tail above the global threshold."""

    def __init__(self, x: np.ndarray, threshold: float, global_tail: np.ndarray):
        x = np.sort(x)
        self.threshold = threshold
        self.body = x[x < threshold]
        self.cap = max(float(global_tail[-1]) if len(global_tail) else threshold, threshold)
        self._global_tail = np.sort(global_tail)

    def sample_body(self, u: np.ndarray) -> np.ndarray:
        if len(self.body) == 0:
            return np.full_like(u, self.threshold * 0.9)
        xp = (np.arange(len(self.body)) + 0.5) / len(self.body)
        return np.interp(u, xp, self.body)

    def sample_tail(self, u: np.ndarray) -> np.ndarray:
        if len(self._global_tail) == 0:
            return np.full_like(u, self.threshold)
        xp = (np.arange(len(self._global_tail)) + 0.5) / len(self._global_tail)
        return np.interp(u, xp, self._global_tail)


def _calibrate_rho(p1: float, p2: float, c_obs: float) -> float:
    """Gaussian rho making P(Z1>z1, Z2>z2) == c_obs with std-normal margins."""
    if p1 == 0 or p2 == 0:
        return 0.0
    z1, z2 = stats.norm.ppf(1 - p1), stats.norm.ppf(1 - p2)
    c_obs = min(c_obs, max(p1, p2) - 1e-9)

    def tail(rho: float) -> float:
        cdf2 = multivariate_normal.cdf([z1, z2], cov=[[1.0, rho], [rho, 1.0]])
        return p1 + p2 - 1.0 + cdf2

    lo, hi = tail(-0.99), tail(0.99)
    if c_obs <= lo:
        return -0.99
    if c_obs >= hi:
        return 0.99
    return brentq(lambda r: tail(r) - c_obs, -0.99, 0.99)


class FCASRegimeCopulaGenerator:
    """v1 generator: per-direction 2-state Markov regime + spike-coupled latent copula."""

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states
        self.marginals: dict[tuple[str, int, str], _Marginal] = {}
        self.spike_rate: dict[str, float] = {}
        self.threshold: dict[str, float] = {}
        self._cap: dict[str, float] = {}
        self._tail: dict[str, np.ndarray] = {}
        self.rho: dict[str, np.ndarray] = {}
        self._logit: dict[str, object] = {}

    def fit(self, df: pl.DataFrame, *, n_states: int | None = None) -> "FCASRegimeCopulaGenerator":
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LogisticRegression

        n_states = n_states or self.n_states
        self.n_states = n_states
        rrp = df["RRP"].to_numpy().astype(float)
        feats = self._build_features(df, float(np.quantile(rrp, 0.99)))

        for s in RAISE + LOWER:
            x = df[f"FCAS_{s}"].to_numpy().astype(float)
            self.threshold[s] = float(np.quantile(x, SPIKE_QUANTILE))
            self.spike_rate[s] = float(np.mean(x >= self.threshold[s]))
            self._cap[s] = float(np.max(x))
            self._tail[s] = x[x >= self.threshold[s]]

        for family_name, family in (("RAISE", RAISE), ("LOWER", LOWER)):
            X = _log1p_matrix(df, family)
            labels = KMeans(n_clusters=n_states, n_init=10, random_state=0).fit_predict(X.mean(axis=1, keepdims=True))
            order = np.argsort([X[labels == k].mean() for k in range(n_states)])
            mapping = {old: new for new, old in enumerate(order)}
            labels = np.array([mapping[l] for l in labels])

            lag = np.roll(labels, 1)
            lag[0] = labels[0]
            Xtr = np.column_stack([feats.to_numpy(), np.eye(n_states)[lag]])
            self._logit[family_name] = LogisticRegression(max_iter=500).fit(Xtr, labels)

            for k in range(n_states):
                idx = labels == k
                for s in family:
                    self.marginals[(family_name, k, s)] = _Marginal(
                        df[f"FCAS_{s}"].to_numpy()[idx], self.threshold[s], self._tail[s]
                    )

            spikes = {s: df[f"FCAS_{s}"].to_numpy() >= self.threshold[s] for s in family}
            m = len(family)
            R = np.eye(m)
            for a in range(m):
                for b in range(a + 1, m):
                    sa, sb = family[a], family[b]
                    c = float(np.mean(spikes[sa] & spikes[sb]))
                    R[a, b] = R[b, a] = _calibrate_rho(self.spike_rate[sa], self.spike_rate[sb], c)
            self.rho[family_name] = R
        return self

    @staticmethod
    def _build_features(df: pl.DataFrame, rrp_spike_threshold: float) -> pl.DataFrame:
        demand = df["TOTALDEMAND"].to_numpy().astype(float)
        wind = df["GEN_wind"].to_numpy().astype(float)
        solar = df["GEN_solar"].to_numpy().astype(float)
        rrp = df["RRP"].to_numpy().astype(float)
        return pl.DataFrame(
            {
                "hour_sin": df["hour_sin"].to_numpy(),
                "hour_cos": df["hour_cos"].to_numpy(),
                "demand_ramp": np.diff(demand, prepend=demand[0]) / (np.abs(demand).max() + 1e-9),
                "wind_delta": np.diff(wind, prepend=wind[0]),
                "solar_delta": np.diff(solar, prepend=solar[0]),
                "rrp_spike": (rrp >= rrp_spike_threshold).astype(float),
            }
        )

    def _simulate(
        self, context: pl.DataFrame, seed: int = 0
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """Run the Markov-regime + copula latent machinery and return the family
        state sequences, the copula uniforms, and the RRP-spike indicator."""
        n = context.height
        rrp = context["RRP"].to_numpy().astype(float)
        feats = self._build_features(context, float(np.quantile(rrp, 0.99)))
        Fm = feats.to_numpy()
        rrp_spike = Fm[:, 5].astype(bool)
        rng = np.random.default_rng(seed)

        states: dict[str, np.ndarray] = {}
        us: dict[str, np.ndarray] = {}
        for family_name, family in (("RAISE", RAISE), ("LOWER", LOWER)):
            state = np.zeros(n, dtype=int)
            for t in range(n):
                if t == 0:
                    state[t] = int(rng.integers(self.n_states))
                else:
                    p = self._logit[family_name].predict_proba(
                        np.concatenate([Fm[t], np.eye(self.n_states)[state[t - 1]]])[None, :]
                    )[0]
                    state[t] = int(rng.choice(self.n_states, p=p))

            m = len(family)
            cov = self.rho[family_name].copy()
            cov[np.diag_indices(m)] = 1.0
            z = np.empty((n, m))
            for k in range(self.n_states):
                idx = np.where(state == k)[0]
                if len(idx):
                    z[idx] = rng.multivariate_normal(np.zeros(m), cov, size=len(idx))
            us[family_name] = stats.norm.cdf(z)
            states[family_name] = state
        return states, us, rrp_spike

    def spike_booleans(self, context: pl.DataFrame, seed: int = 0) -> dict[str, np.ndarray]:
        """Per-service binary spike schedule over the context grid (timing only)."""
        _, us, rrp_spike = self._simulate(context, seed)
        spikes: dict[str, np.ndarray] = {}
        for family_name, family in (("RAISE", RAISE), ("LOWER", LOWER)):
            for i, s in enumerate(family):
                p_i = self.spike_rate[s]
                boost = 8.0 if family_name == "RAISE" and s in RAISE[:3] else 1.0
                p = np.where(rrp_spike, min(1.0, p_i * boost), p_i)
                spikes[f"FCAS_{s}"] = us[family_name][:, i] > (1.0 - p)
        return spikes

    def sample(self, context: pl.DataFrame) -> pl.DataFrame:
        states, us, rrp_spike = self._simulate(context, seed=0)
        n = context.height
        rng = np.random.default_rng(0)

        out = {}
        for family_name, family in (("RAISE", RAISE), ("LOWER", LOWER)):
            state = states[family_name]
            u = us[family_name]
            for i, s in enumerate(family):
                vals = np.empty(n)
                p_i = self.spike_rate[s]
                boost = 8.0 if family_name == "RAISE" and s in RAISE[:3] else 1.0
                for k in range(self.n_states):
                    msk = state == k
                    if not msk.any():
                        continue
                    marg = self.marginals[(family_name, k, s)]
                    uk = u[msk, i]
                    vals[msk] = marg.sample_body(uk)
                    if p_i > 0:
                        p = np.where(rrp_spike[msk], min(1.0, p_i * boost), p_i)
                        spike = uk > (1.0 - p)
                        if spike.any():
                            idx = np.where(msk)[0]
                            vals[idx[spike]] = marg.sample_tail(rng.random(spike.sum()))
                out[f"FCAS_{s}"] = vals
        return _clamp_fcas_columns(context.with_columns([pl.Series(name, out[name]) for name in out]))


class FCASDiffusionGenerator:
    """Conditional diffusion model for synthetic RRP + FCAS trajectories."""

    def __init__(
        self,
        *,
        window_size: int = 288,
        stride: int = 12,
        overlap: int = 48,
        diffusion_steps: int = 128,
        sample_steps: int = 32,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        epochs: int = 8,
        batch_size: int = 32,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        tail_quantile: float = 0.95,
        tail_weight: float = 4.0,
        spike_quantile: float = 0.99,
        sample_eta: float = 0.05,
        schedule_seed: int = 0,
        tail_mode: str = "schedule",
        seed: int = 0,
        device: str | None = None,
    ):
        _require_torch()
        if tail_mode not in ("diffusion", "schedule"):
            raise ValueError(f"tail_mode must be 'diffusion' or 'schedule', got {tail_mode!r}")
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")
        self.window_size = window_size
        self.stride = stride
        self.overlap = overlap
        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.tail_quantile = tail_quantile
        self.tail_weight = tail_weight
        self.spike_quantile = spike_quantile
        self.sample_eta = sample_eta
        self.schedule_seed = schedule_seed
        self.tail_mode = tail_mode
        self.seed = seed
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_name)

        self.rrp_spike_threshold: float | None = None
        self.cond_mean: np.ndarray | None = None
        self.cond_std: np.ndarray | None = None
        self.target_mean: np.ndarray | None = None
        self.target_std: np.ndarray | None = None
        self.target_min: np.ndarray | None = None
        self.target_max: np.ndarray | None = None
        self.tail_thresholds: np.ndarray | None = None
        self.train_loss_history: list[float] = []
        self.stage_a: FCASRegimeCopulaGenerator | None = None
        self._tail_knn_feats: np.ndarray | None = None
        self._tail_knn_spike_idx: dict[str, np.ndarray] = {}
        self._tail_knn_values: dict[str, np.ndarray] = {}
        self._tail_feat_mean: np.ndarray | None = None
        self._tail_feat_std: np.ndarray | None = None
        self._burst_templates: dict[str, list[np.ndarray]] = {}
        self._burst_event_rate: dict[str, float] = {}

        self.model = _TemporalUNet(
            target_channels=len(TARGET_COLS),
            condition_channels=len(CONDITION_COLS) + len(FCAS),
            base_channels=self.base_channels,
            channel_mults=self.channel_mults,
        ).to(self.device)

        betas = torch.linspace(1e-4, 0.02, diffusion_steps, dtype=torch.float32, device=self.device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=self.device), alpha_bars[:-1]], dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev

    def _observed_spike_schedule(self, frame: pl.DataFrame) -> np.ndarray:
        """Per-service spike indicators on the training frame (observed truth)."""
        cols = []
        for s in FCAS:
            x = frame[f"FCAS_{s}"].to_numpy().astype(np.float32)
            thr = float(np.quantile(x, self.spike_quantile))
            cols.append((x >= thr).astype(np.float32))
        return np.column_stack(cols)

    def _burst_expand_schedule(self, spikes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Stamp Stage A's copula onsets with real FCAS event templates.

        Real FCAS spikes are clustered multi-bar contingency events in which the
        services of a family co-spike with a specific rate/persistence profile.
        Stage A's copula couples co-onsets well but fires isolated single bars.
        This thins the copula event onsets down to the real event rate and stamps
        each surviving onset with a random event template sampled from the real
        training data, so the synthetic schedule reproduces the exact joint
        spike-rate / persistence / within-family co-occurrence structure.
        """
        n = len(spikes[f"FCAS_{FCAS[0]}"])
        rng = np.random.default_rng(self.schedule_seed)
        out: dict[str, np.ndarray] = dict(spikes)
        for family_name, family in (("RAISE", RAISE), ("LOWER", LOWER)):
            cols = [f"FCAS_{s}" for s in family]
            base = np.column_stack([spikes[c] for c in cols]).astype(bool)
            shifted = np.zeros_like(base)
            shifted[1:] = base[:-1]
            onsets = base & ~shifted
            event_onset = onsets.any(axis=1)
            base_event_rate = float(event_onset.mean())
            templates = self._burst_templates.get(family_name, [])
            target_rate = self._burst_event_rate.get(family_name, 0.0)
            if not templates or base_event_rate <= 0 or target_rate <= 0:
                continue
            keep_frac = min(1.0, target_rate / base_event_rate)
            expanded = np.zeros_like(base)
            n_templates = len(templates)
            for t in np.where(event_onset)[0]:
                if rng.random() > keep_frac:
                    continue
                template = templates[rng.integers(0, n_templates)]
                length = template.shape[0]
                end = min(t + length, n)
                expanded[t:end, :] |= template[: end - t, :]
            for k, s in enumerate(family):
                out[f"FCAS_{s}"] = expanded[:, k]
        return out

    def fit(self, df: pl.DataFrame) -> "FCASDiffusionGenerator":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        frame = _clamp_fcas_columns(df)
        self.rrp_spike_threshold = float(np.quantile(frame["RRP"].to_numpy().astype(float), 0.99))
        condition_frame = _build_condition_frame(frame, rrp_spike_threshold=self.rrp_spike_threshold)
        feature_matrix = condition_frame.select(CONDITION_COLS).to_numpy().astype(np.float32)

        # Stage A: per-service spike scheduler (Markov regime + Gaussian copula).
        self.stage_a = FCASRegimeCopulaGenerator(n_states=2).fit(frame)
        schedule = self._observed_spike_schedule(frame)

        # Real burst structure (from the observed schedule): per-family event
        # templates sampled from the real data, plus the real event rate. At
        # sample time Stage A's copula onsets are thinned to the real event rate
        # and each surviving onset is stamped with a random real event template,
        # reproducing the exact joint rate / persistence / co-occurrence of real
        # FCAS contingency events.
        self._burst_templates = {}
        self._burst_event_rate = {}
        for family_name, family in (("RAISE", RAISE), ("LOWER", LOWER)):
            family_cols = [FCAS.index(s) for s in family]
            m = schedule[:, family_cols] > 0
            any_s = m.any(axis=1)
            templates: list[np.ndarray] = []
            i = 0
            while i < schedule.shape[0]:
                if any_s[i]:
                    j = i
                    while j < schedule.shape[0] and any_s[j]:
                        j += 1
                    templates.append(m[i:j].copy())
                    i = j
                else:
                    i += 1
            self._burst_templates[family_name] = templates
            self._burst_event_rate[family_name] = len(templates) / max(schedule.shape[0], 1)

        # Conditional tail sampler: feature-conditioned empirical tail over the
        # fit window's own spike bars (per service). Used when tail_mode=schedule
        # to guarantee non-empty, feature-relevant tail magnitudes at spike bars.
        self._tail_knn_feats = feature_matrix
        self._tail_feat_mean = feature_matrix.mean(axis=0).astype(np.float32)
        self._tail_feat_std = np.clip(feature_matrix.std(axis=0).astype(np.float32), 1e-6, None)
        self._tail_knn_spike_idx = {}
        self._tail_knn_values = {}
        for i, s in enumerate(FCAS):
            spike_idx = np.where(schedule[:, i] > 0)[0]
            self._tail_knn_spike_idx[s] = spike_idx
            self._tail_knn_values[s] = frame[f"FCAS_{s}"].to_numpy().astype(np.float32)[spike_idx]

        condition_matrix = np.concatenate([feature_matrix, schedule], axis=1).astype(np.float32)
        target_matrix = _build_target_matrix(frame)

        target_windows = _build_windows(target_matrix, window_size=self.window_size, stride=self.stride)
        condition_windows = _build_windows(condition_matrix, window_size=self.window_size, stride=self.stride)
        schedule_windows = _build_windows(schedule, window_size=self.window_size, stride=self.stride)

        self.cond_mean = condition_windows.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.cond_std = np.clip(condition_windows.std(axis=(0, 2), keepdims=True).astype(np.float32), 1e-6, None)
        self.target_mean = target_windows.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.target_std = np.clip(target_windows.std(axis=(0, 2), keepdims=True).astype(np.float32), 1e-6, None)

        condition_windows = (condition_windows - self.cond_mean) / self.cond_std
        target_windows = (target_windows - self.target_mean) / self.target_std

        self.target_min = target_windows.min(axis=(0, 2), keepdims=True).astype(np.float32)
        self.target_max = target_windows.max(axis=(0, 2), keepdims=True).astype(np.float32)
        self.tail_thresholds = np.quantile(target_windows[:, 1:, :], self.tail_quantile, axis=(0, 2)).astype(np.float32)

        dataset = _WindowDataset(target_windows, condition_windows, schedule_windows)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        tail_thresholds = torch.from_numpy(self.tail_thresholds).to(self.device).view(1, -1, 1)

        self.model.train()
        self.train_loss_history = []
        for _ in range(self.epochs):
            epoch_loss = 0.0
            total = 0
            for clean, cond, spike in loader:
                clean = clean.to(self.device)
                cond = cond.to(self.device)
                spike = spike.to(self.device)
                timesteps = torch.randint(0, self.diffusion_steps, (clean.shape[0],), device=self.device)
                noise = torch.randn_like(clean)
                alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
                noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
                pred_noise = self.model(noisy, cond, timesteps)

                weights = torch.ones_like(clean)
                weights[:, 1:, :] = 1.0 + self.tail_weight * spike.float()
                weights[:, 1:, :] = weights[:, 1:, :] + self.tail_weight * (clean[:, 1:, :] >= tail_thresholds).float()
                loss = ((pred_noise - noise) ** 2 * weights).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch = clean.shape[0]
                epoch_loss += float(loss.detach().cpu()) * batch
                total += batch
            self.train_loss_history.append(epoch_loss / max(total, 1))
        return self

    def sample(self, context: pl.DataFrame) -> pl.DataFrame:
        self._ensure_fitted()
        frame = _clamp_fcas_columns(context)
        condition_frame = _build_condition_frame(frame, rrp_spike_threshold=float(self.rrp_spike_threshold))
        feature_matrix = condition_frame.select(CONDITION_COLS).to_numpy().astype(np.float32)
        spikes = self.stage_a.spike_booleans(frame, seed=self.schedule_seed)
        spikes = self._burst_expand_schedule(spikes)
        schedule = np.column_stack([spikes[f"FCAS_{s}"].astype(np.float32) for s in FCAS])
        condition_matrix = np.concatenate([feature_matrix, schedule], axis=1).astype(np.float32)
        normalized = (condition_matrix[None, :, :] - self.cond_mean.transpose(0, 2, 1)) / self.cond_std.transpose(0, 2, 1)
        normalized = normalized[0]

        out = np.zeros((frame.height, len(TARGET_COLS)), dtype=np.float32)
        filled_until = 0
        starts = _sampling_starts(frame.height, self.window_size, self.overlap)

        for start in starts:
            cond_window, actual_len = _slice_or_pad(normalized, start, self.window_size)
            generated = self._sample_window(cond_window.T[None, :, :])[0].T[:actual_len]
            if start == 0:
                out[:actual_len] = generated
                filled_until = actual_len
                continue

            overlap = max(0, filled_until - start)
            if overlap > 0:
                ramp = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1][:, None]
                out[start : start + overlap] = out[start : start + overlap] * (1.0 - ramp) + generated[:overlap] * ramp
            out[start + overlap : start + actual_len] = generated[overlap:actual_len]
            filled_until = max(filled_until, start + actual_len)

        restored = _inverse_target_matrix(out)
        if self.tail_mode == "schedule":
            restored = self._schedule_gated_tail(restored, feature_matrix, schedule)
        synth = frame.with_columns(
            [pl.Series("RRP", restored["RRP"].astype(float))]
            + [pl.Series(col, restored[col].astype(float)) for col in FCAS_COLS]
        )
        return _clamp_fcas_columns(synth)

    def save(self, path: str | Path) -> Path:
        self._ensure_fitted()
        path = Path(path)
        payload = {
            "config": {
                "window_size": self.window_size,
                "stride": self.stride,
                "overlap": self.overlap,
                "diffusion_steps": self.diffusion_steps,
                "sample_steps": self.sample_steps,
                "base_channels": self.base_channels,
                "channel_mults": self.channel_mults,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "tail_quantile": self.tail_quantile,
                "tail_weight": self.tail_weight,
                "spike_quantile": self.spike_quantile,
                "sample_eta": self.sample_eta,
                "schedule_seed": self.schedule_seed,
                "tail_mode": self.tail_mode,
                "seed": self.seed,
                "device": self.device_name,
            },
            "state_dict": self.model.state_dict(),
            "stage_a": self.stage_a,
            "tail_knn_feats": self._tail_knn_feats,
            "tail_knn_spike_idx": self._tail_knn_spike_idx,
            "tail_knn_values": self._tail_knn_values,
            "tail_feat_mean": self._tail_feat_mean,
            "tail_feat_std": self._tail_feat_std,
            "burst_templates": self._burst_templates,
            "burst_event_rate": self._burst_event_rate,
            "rrp_spike_threshold": self.rrp_spike_threshold,
            "cond_mean": self.cond_mean,
            "cond_std": self.cond_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "tail_thresholds": self.tail_thresholds,
            "train_loss_history": self.train_loss_history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | None = None) -> "FCASDiffusionGenerator":
        payload = torch.load(Path(path), map_location=map_location or "cpu", weights_only=False)
        gen = cls(**payload["config"])
        gen.model.load_state_dict(payload["state_dict"])
        gen.rrp_spike_threshold = float(payload["rrp_spike_threshold"])
        gen.cond_mean = payload["cond_mean"]
        gen.cond_std = payload["cond_std"]
        gen.target_mean = payload["target_mean"]
        gen.target_std = payload["target_std"]
        gen.target_min = payload["target_min"]
        gen.target_max = payload["target_max"]
        gen.tail_thresholds = payload["tail_thresholds"]
        gen.train_loss_history = list(payload.get("train_loss_history", []))
        gen.stage_a = payload.get("stage_a")
        gen._tail_knn_feats = payload.get("tail_knn_feats")
        gen._tail_knn_spike_idx = payload.get("tail_knn_spike_idx", {})
        gen._tail_knn_values = payload.get("tail_knn_values", {})
        gen._tail_feat_mean = payload.get("tail_feat_mean")
        gen._tail_feat_std = payload.get("tail_feat_std")
        gen._burst_templates = payload.get("burst_templates", {})
        gen._burst_event_rate = payload.get("burst_event_rate", {})
        gen.model.to(gen.device)
        gen.model.eval()
        return gen

    def _sample_window(self, condition_window: np.ndarray) -> np.ndarray:
        self.model.eval()
        schedule = self._sampling_schedule()
        with torch.no_grad():
            cond = torch.from_numpy(condition_window).to(self.device, dtype=torch.float32)
            x = torch.randn((cond.shape[0], len(TARGET_COLS), cond.shape[-1]), device=self.device)
            min_bound = torch.from_numpy(self.target_min).to(self.device, dtype=torch.float32)
            max_bound = torch.from_numpy(self.target_max).to(self.device, dtype=torch.float32)
            for idx, timestep in enumerate(schedule):
                t = torch.full((cond.shape[0],), timestep, device=self.device, dtype=torch.long)
                alpha_bar_t = self.alpha_bars[timestep]
                pred_noise = self.model(x, cond, t)
                pred_clean = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                pred_clean = torch.maximum(torch.minimum(pred_clean, max_bound), min_bound)
                next_t = schedule[idx + 1] if idx + 1 < len(schedule) else -1
                if next_t < 0:
                    x = pred_clean
                    break
                alpha_bar_next = self.alpha_bars[next_t]
                sigma = self.sample_eta * torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t))
                sigma = sigma * torch.sqrt(torch.clamp(1.0 - alpha_bar_t / alpha_bar_next, min=0.0))
                direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma ** 2, min=0.0)) * pred_noise
                noise = torch.randn_like(x) if self.sample_eta > 0.0 else torch.zeros_like(x)
                x = torch.sqrt(alpha_bar_next) * pred_clean + direction + sigma * noise

            x = x.cpu().numpy()
        x = x * self.target_std + self.target_mean
        return x

    def _schedule_gated_tail(
        self,
        restored: dict[str, np.ndarray],
        feature_matrix: np.ndarray,
        schedule: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Replace FCAS magnitudes at schedule-spike bars with feature-conditional
        samples from the fit window's own spike tail (per service).

        The diffusion under-shoots the heavy tail (regression-to-mean in log1p
        space), so spike bars otherwise contribute almost nothing above the
        holdout threshold. Gating the magnitudes on Stage A's schedule guarantees
        a non-empty, feature-relevant tail while keeping the diffusion's
        well-calibrated bulk for non-spike bars.
        """
        rng = np.random.default_rng(self.schedule_seed)
        feats = (feature_matrix - self._tail_feat_mean) / self._tail_feat_std
        out = dict(restored)
        k = 20
        for i, s in enumerate(FCAS):
            spike_idx = np.where(schedule[:, i] > 0)[0]
            pool_idx = self._tail_knn_spike_idx.get(s)
            if len(spike_idx) == 0 or pool_idx is None or len(pool_idx) == 0:
                continue
            pool_feats = (self._tail_knn_feats[pool_idx] - self._tail_feat_mean) / self._tail_feat_std
            pool_vals = self._tail_knn_values[s]
            d2 = ((feats[spike_idx][:, None, :] - pool_feats[None, :, :]) ** 2).sum(-1)
            nn = np.argsort(d2, axis=1)[:, : min(k, len(pool_idx))]
            pick = nn[np.arange(len(spike_idx)), rng.integers(0, nn.shape[1], size=len(spike_idx))]
            values = out[f"FCAS_{s}"].astype(np.float64)
            values[spike_idx] = pool_vals[pick].astype(np.float64)
            out[f"FCAS_{s}"] = values
        return out

    def _sampling_schedule(self) -> list[int]:
        raw = np.linspace(self.diffusion_steps - 1, 0, num=min(self.sample_steps, self.diffusion_steps))
        schedule = []
        seen = set()
        for value in raw.astype(int).tolist():
            if value not in seen:
                seen.add(value)
                schedule.append(value)
        if schedule[-1] != 0:
            schedule.append(0)
        return schedule

    def _ensure_fitted(self) -> None:
        if self.rrp_spike_threshold is None:
            raise RuntimeError("generator must be fit before sampling")
        if self.cond_mean is None or self.target_mean is None:
            raise RuntimeError("generator normalization statistics are missing")
        if self.stage_a is None:
            raise RuntimeError("Stage A spike scheduler is missing; fit must run first")
        if self.tail_mode == "schedule" and self._tail_knn_feats is None:
            raise RuntimeError("tail sampler is missing; fit must run first")
