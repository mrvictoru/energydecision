from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, TanhTransform

from decision_transformer import DecisionTransformer


def _reset_env(env, seed: int | None = None, options: dict[str, Any] | None = None):
    if options is None:
        return env.reset(seed=seed)

    reset_sig = inspect.signature(env.reset)
    if "options" in reset_sig.parameters:
        return env.reset(seed=seed, options=options)
    return env.reset(seed=seed, **options)


@dataclass(frozen=True)
class GRPOPrompt:
    seed: int | None = None
    options: dict[str, Any] | None = None
    rtg_value: float = 0.0
    max_steps: int | None = None


@dataclass
class GRPORolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rtgs: torch.Tensor
    timesteps: torch.Tensor
    attention_mask: torch.Tensor
    sampled_actions: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    env_rewards: torch.Tensor
    returns: torch.Tensor
    prompt_indices: torch.Tensor

    @property
    def num_steps(self) -> int:
        return int(self.sampled_actions.shape[0])

    @property
    def num_episodes(self) -> int:
        return int(self.returns.shape[0])


def _is_legacy_dt(model: DecisionTransformer) -> bool:
    """Detect if model was converted to LegacyDecisionTransformer by load_from_checkpoint."""
    return hasattr(model, "embed_return") and hasattr(model, "embed_action")


def load_pretrained_dt_for_grpo(
    model_kwargs: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[DecisionTransformer, DecisionTransformer]:
    model = DecisionTransformer(**model_kwargs)
    model.load_from_checkpoint(str(Path(checkpoint_path).resolve()), map_location=device)
    model.to(device)
    model.train()

    reference_model = copy.deepcopy(model)
    reference_model.to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    return model, reference_model


def compute_group_relative_advantages(
    returns: Sequence[float] | np.ndarray,
    group_size: int,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    values = np.asarray(list(returns), dtype=np.float32)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if values.size % group_size != 0:
        raise ValueError("returns length must be divisible by group_size.")

    grouped = values.reshape(-1, group_size)
    means = grouped.mean(axis=1, keepdims=True)
    stds = grouped.std(axis=1, keepdims=True)
    centered = grouped - means
    safe_stds = np.where(stds > eps, stds, 1.0)
    advantages = centered / safe_stds
    zero_var_mask = (stds <= eps).reshape(-1)
    if zero_var_mask.any():
        advantages[zero_var_mask] = centered[zero_var_mask]
    return advantages.reshape(-1).astype(np.float32)


MAX_ABS_GRPO_RTG = 1e6


def stable_rtg_update(
    current_rtg: float,
    reward: float,
    *,
    dt_gamma: float,
    initial_rtg: float,
) -> float:
    """Numerically stable return-to-go update for autoregressive DT rollout.

    The Decision Transformer is trained on the discounted convention
    ``rtg[t] = r[t] + gamma * rtg[t+1]`` (see ``helper._compute_rtgs_from_rewards``).
    Inverting it gives the exact recurrence ``rtg[t+1] = (rtg[t] - r[t]) / gamma``.

    That exact inverse is numerically unstable for ``gamma < 1`` on long horizons:
    every step multiplies the magnitude by ``1 / gamma > 1``, so RTG compounds as
    ``(1 / gamma) ** t`` and overflows (e.g. ~3e38 after 1728 steps at gamma=0.95).
    This collapsed every long-horizon modern-v2 GRPO run.

    We keep the exact undiscounted recurrence for ``gamma == 1.0`` and, for
    ``gamma < 1.0``, apply the discounted update while clamping the result to the
    trained RTG envelope so it can never blow up. The clamp bound is derived from
    the initial prompt magnitude (the target the model was actually conditioned
    on), which keeps the RTG signal inside the distribution the DT saw during
    pretraining instead of drifting to astronomical values.
    """
    if not np.isfinite(current_rtg) or not np.isfinite(reward):
        raise ValueError(
            f"Non-finite inputs to stable_rtg_update (rtg={current_rtg}, reward={reward})."
        )
    if dt_gamma == 1.0:
        return float(current_rtg - reward)

    next_rtg = (float(current_rtg) - float(reward)) / float(dt_gamma)
    # Clamp to the trained RTG envelope so the 1/gamma recurrence cannot explode.
    # The bound is a small multiple of the initial prompt magnitude; a floor keeps
    # small/zero prompts from collapsing the achievable RTG range.
    envelope = max(abs(float(initial_rtg)) * 4.0, 20.0)
    return float(np.clip(next_rtg, -envelope, envelope))


def _validate_grpo_rtg(value: float, *, step_count: int, dt_gamma: float) -> float:
    if not np.isfinite(value):
        raise ValueError(
            f"Non-finite GRPO RTG at step {step_count} with dt_gamma={dt_gamma}. "
            "Use dt_gamma=1.0 for long-horizon GRPO or reduce the episode length."
        )
    if abs(value) > MAX_ABS_GRPO_RTG:
        raise ValueError(
            f"Exploding GRPO RTG ({value}) at step {step_count} with dt_gamma={dt_gamma}. "
            f"Values above {MAX_ABS_GRPO_RTG:g} are treated as unstable. "
            "Use dt_gamma=1.0 for long-horizon GRPO or reduce the episode length."
        )
    return float(value)


def _build_dt_context(
    *,
    model: DecisionTransformer,
    state_buffer: list[np.ndarray],
    action_buffer: list[np.ndarray],
    rtg_buffer: list[float],
    timestep_buffer: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context_len = int(model.context_len)
    state_dim = int(model.state_dim)
    act_dim = int(model.act_dim)
    buffer_len = len(state_buffer)

    buffer_states = (
        np.array(state_buffer, dtype=np.float32)
        if buffer_len > 0
        else np.zeros((0, state_dim), dtype=np.float32)
    )
    buffer_actions = (
        np.array(action_buffer, dtype=np.float32)
        if buffer_len > 0
        else np.zeros((0, act_dim), dtype=np.float32)
    )
    if buffer_len > 0:
        buffer_rtgs64 = np.array(rtg_buffer, dtype=np.float64)
        if not np.all(np.isfinite(buffer_rtgs64)):
            raise ValueError("Encountered non-finite GRPO RTG values while building the DT context.")
        if float(np.max(np.abs(buffer_rtgs64))) > MAX_ABS_GRPO_RTG:
            raise ValueError(
                f"Encountered GRPO RTG magnitude above {MAX_ABS_GRPO_RTG:g} while building the DT context."
            )
        buffer_rtgs = buffer_rtgs64.astype(np.float32)
    else:
        buffer_rtgs = np.zeros(0, dtype=np.float32)
    buffer_timesteps = (
        np.array(timestep_buffer, dtype=np.int64)
        if buffer_len > 0
        else np.zeros(0, dtype=np.int64)
    )

    if buffer_len < context_len:
        pad_len = context_len - buffer_len
        states = np.vstack([np.zeros((pad_len, state_dim), dtype=np.float32), buffer_states])
        actions = np.vstack([np.zeros((pad_len, act_dim), dtype=np.float32), buffer_actions])
        rtgs = np.concatenate([np.zeros(pad_len, dtype=np.float32), buffer_rtgs])
        timesteps = np.concatenate([np.zeros(pad_len, dtype=np.int64), buffer_timesteps])
        attention_mask = np.concatenate(
            [np.zeros(pad_len, dtype=np.bool_), np.ones(buffer_len, dtype=np.bool_)]
        )
    else:
        states = buffer_states[-context_len:]
        actions = buffer_actions[-context_len:]
        rtgs = buffer_rtgs[-context_len:]
        timesteps = buffer_timesteps[-context_len:]
        attention_mask = np.ones(context_len, dtype=np.bool_)

    return_scale = float(getattr(model, "return_scale", 1.0))
    if not np.isfinite(return_scale) or abs(return_scale) < 1e-12:
        raise ValueError(f"Invalid Decision Transformer return_scale: {return_scale}")
    if return_scale != 1.0:
        rtgs = rtgs / return_scale

    max_time = getattr(model.embed_timestep, "num_embeddings", None)
    if max_time is not None and max_time > 0:
        timesteps = np.clip(timesteps, 0, int(max_time) - 1)

    states_t = torch.tensor(np.nan_to_num(states), dtype=torch.float32, device=device).unsqueeze(0)
    actions_t = torch.tensor(np.nan_to_num(actions), dtype=torch.float32, device=device).unsqueeze(0)
    rtgs_t = torch.tensor(np.nan_to_num(rtgs), dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    timesteps_t = torch.tensor(timesteps, dtype=torch.long, device=device).unsqueeze(0)
    mask_t = torch.tensor(attention_mask, dtype=torch.bool, device=device).unsqueeze(0)
    return states_t, actions_t, rtgs_t, timesteps_t, mask_t


def _diagonal_tanh_normal(action_mean: torch.Tensor, log_std: torch.Tensor) -> TransformedDistribution:
    std = log_std.exp().view(1, 1, -1).expand_as(action_mean)
    base = Independent(Normal(loc=action_mean, scale=std), 1)
    return TransformedDistribution(base, [TanhTransform(cache_size=1)])


def _mixed_action_distribution(
    action_mean: torch.Tensor,
    log_std: torch.Tensor,
    act_dim: int,
) -> TransformedDistribution:
    """Create a per-dimension mixed-bounds distribution.

    - Dim 0 (energy dispatch): ``TanhTransform`` → ``[-1, 1]``
    - Dims ``1..act_dim-1`` (FCAS bids): ``SigmoidTransform`` → ``[0, 1]``

    Falls back to ``_diagonal_tanh_normal`` when ``act_dim <= 1``.
    """
    if act_dim <= 1:
        return _diagonal_tanh_normal(action_mean, log_std)

    dim0_mean = action_mean[..., :1]
    dim0_std = log_std[:1].exp().view(1, 1, 1).expand_as(dim0_mean)
    dim0_base = Independent(Normal(loc=dim0_mean, scale=dim0_std), 1)
    dim0_dist = TransformedDistribution(dim0_base, [TanhTransform(cache_size=1)])

    fcas_mean = action_mean[..., 1:]
    fcas_std = log_std[1:].exp().view(1, 1, -1).expand_as(fcas_mean)
    fcas_base = Independent(Normal(loc=fcas_mean, scale=fcas_std), 1)
    fcas_dist = TransformedDistribution(fcas_base, [SigmoidTransform(cache_size=1)])

    class _Mixed:
        def __init__(self, d0, fc):
            self.d0 = d0
            self.fc = fc

        def rsample(self) -> torch.Tensor:
            return torch.cat([self.d0.rsample(), self.fc.rsample()], dim=-1)

        def log_prob(self, value: torch.Tensor) -> torch.Tensor:
            d0_lp = self.d0.log_prob(value[..., :1])
            fc_lp = self.fc.log_prob(value[..., 1:])
            return d0_lp + fc_lp

        @property
        def base_dist(self):
            return self

        def entropy(self) -> torch.Tensor:
            # Sample-based entropy: E[-log π(a)] ≈ -log π(a) for a ~ π
            # More stable than TransformedDistribution.entropy() which is often NotImplemented
            sample = self.rsample()
            return -self.log_prob(sample).detach()

    return _Mixed(dim0_dist, fcas_dist)


def sample_rtg_values(
    optimum: float,
    spread: float,
    count: int,
    distribution: str = "gaussian",
    seed: int | None = None,
) -> list[float]:
    """Sample ``count`` RTG values around ``optimum`` for group diversity.

    Always includes ``optimum`` itself as one of the values.  The remaining
    ``count - 1`` values are sampled from the chosen distribution.

    Args:
        optimum: The optimal / recommended RTG (e.g. from model calibration).
        spread: Spread parameter — standard deviation for ``'gaussian'``,
            half-width for ``'uniform'``.
        count: Number of RTG values to produce (including the optimum).
        distribution: ``'gaussian'`` (normal), ``'uniform'``, or ``'lognormal'``.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A shuffled list of ``count`` RTG values.
    """
    rng = np.random.default_rng(seed)
    n = max(0, int(count) - 1)

    if distribution == "lognormal" and optimum > 0:
        log_opt = np.log(optimum)
        raw = rng.normal(log_opt, spread, size=n).tolist()
        samples = [max(1e-6, np.exp(v)) for v in raw]
    elif distribution == "uniform":
        half = abs(float(spread))
        low = optimum - half
        high = optimum + half
        if low >= high:
            low, high = optimum - 1.0, optimum + 1.0
        samples = rng.uniform(low, high, size=n).tolist()
    else:
        # default: gaussian
        std = abs(float(spread))
        low = optimum - 3.0 * std if std > 0 else optimum - 1.0
        high = optimum + 3.0 * std if std > 0 else optimum + 1.0
        samples = rng.normal(optimum, std, size=n).tolist()
        samples = [max(low, min(high, v)) for v in samples]

    all_values = [float(optimum)] + samples
    rng.shuffle(all_values)
    return all_values


def _episode_max_steps(env, prompt: GRPOPrompt) -> int:
    if prompt.max_steps is not None:
        return max(1, int(prompt.max_steps))
    if hasattr(env, "max_step"):
        return max(1, int(getattr(env, "max_step")))
    if hasattr(env, "_max_episode_steps") and getattr(env, "_max_episode_steps") is not None:
        return max(1, int(getattr(env, "_max_episode_steps")))
    if hasattr(env, "df"):
        return max(1, int(len(env.df)))
    return 1_000


class GRPOTrainer:
    def __init__(
        self,
        model: DecisionTransformer,
        *,
        reference_model: DecisionTransformer | None = None,
        device: str = "cpu",
        lr: float = 1e-5,
        clip_ratio: float = 0.2,
        kl_coeff: float = 0.02,
        entropy_coeff: float = 0.0,
        initial_log_std: float = -1.0,
        trainable_log_std: bool = True,
        grad_clip_norm: float = 1.0,
        action_bounds: tuple[float, float] | None = None,
        degradation_penalty_weight: float = 1.0,
        mixed_precision: bool = False,
        cpu_rollout_buffer: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.reference_model = copy.deepcopy(model) if reference_model is None else reference_model
        self.reference_model = self.reference_model.to(device)
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

        self._act_dim = int(self.model.act_dim)
        self.log_std = nn.Parameter(
            torch.full((self._act_dim,), float(initial_log_std), device=self.device)
        )
        self.log_std.requires_grad_(bool(trainable_log_std))

        params: list[nn.Parameter] = [p for p in self.model.parameters() if p.requires_grad]
        if self.log_std.requires_grad:
            params.append(self.log_std)
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.clip_ratio = float(clip_ratio)
        self.kl_coeff = float(kl_coeff)
        self.entropy_coeff = float(entropy_coeff)
        self.grad_clip_norm = float(grad_clip_norm)
        self.degradation_penalty_weight = float(degradation_penalty_weight)
        self._last_prompts: list[GRPOPrompt] = []
        self._adaptive_rtg_ewma: float | None = None
        self.mixed_precision = bool(mixed_precision)
        self.cpu_rollout_buffer = bool(cpu_rollout_buffer)
        self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _sync_reference_model(self) -> None:
        self.reference_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.reference_model.to(self.device)
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    def _shape_reward(self, reward: float, info: dict[str, Any] | None) -> float:
        if self.degradation_penalty_weight <= 1.0:
            return float(reward)
        info_dict = info or {}
        degradation_cost = float(info_dict.get("degradation_cost", 0.0))
        extra_weight = self.degradation_penalty_weight - 1.0
        return float(reward) - extra_weight * degradation_cost

    def _resample_prompts(
        self,
        prompts: Sequence[GRPOPrompt],
        *,
        optimum: float,
        spread: float,
        distribution: str,
        seed: int | None,
    ) -> list[GRPOPrompt]:
        rtg_values = sample_rtg_values(
            optimum=optimum,
            spread=spread,
            count=len(prompts),
            distribution=distribution,
            seed=seed,
        )
        return [
            GRPOPrompt(
                seed=prompt.seed,
                options=prompt.options,
                rtg_value=float(rtg_values[idx]),
                max_steps=prompt.max_steps,
            )
            for idx, prompt in enumerate(prompts)
        ]

    @staticmethod
    def _forward_dt(model, states, rtgs, timesteps, actions, attention_mask, amp_context=None):
        """Call model.forward with correct argument order, normalizing return order.

        Modern:  forward(state, rtg, timestep, actions, mask) -> (return_preds, state_preds, act_preds)
        Legacy:  forward(states, actions, returns_to_go, timesteps, mask) -> (act_preds, state_preds, return_preds)
        """
        if amp_context is not None:
            with amp_context:
                if _is_legacy_dt(model):
                    act_preds, _, _ = model(states, actions, rtgs, timesteps, attention_mask=attention_mask)
                else:
                    _, _, act_preds = model(states, rtgs, timesteps, actions, attention_mask=attention_mask)
        else:
            if _is_legacy_dt(model):
                act_preds, _, _ = model(states, actions, rtgs, timesteps, attention_mask=attention_mask)
            else:
                _, _, act_preds = model(states, rtgs, timesteps, actions, attention_mask=attention_mask)
        return act_preds

    @staticmethod
    def _sanitize_action_preds(action_preds: torch.Tensor) -> torch.Tensor:
        if action_preds.dtype.is_floating_point:
            action_preds = torch.nan_to_num(action_preds, nan=0.0, posinf=1.0, neginf=-1.0)
        return action_preds

    def _action_distributions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[TransformedDistribution, TransformedDistribution]:
        amp_ctx = torch.cuda.amp.autocast(dtype=self._amp_dtype) if self.mixed_precision and 'cuda' in str(self.device) else None
        action_preds = self._forward_dt(
            self.model, states, rtgs, timesteps, actions, attention_mask, amp_context=amp_ctx
        )
        ref_preds = self._forward_dt(
            self.reference_model, states, rtgs, timesteps, actions, attention_mask, amp_context=amp_ctx
        )
        action_mean = self._sanitize_action_preds(action_preds[:, -1:, :])
        ref_mean = self._sanitize_action_preds(ref_preds[:, -1:, :])
        current_dist = _mixed_action_distribution(action_mean, self.log_std, self._act_dim)
        ref_dist = _mixed_action_distribution(ref_mean, self.log_std.detach(), self._act_dim)
        return current_dist, ref_dist

    def collect_rollouts(
        self,
        env_factory: Callable[[], Any],
        *,
        prompts: Sequence[GRPOPrompt],
        group_size: int = 4,
        dt_gamma: float = 0.99,
    ) -> GRPORolloutBatch:
        """Collect rollouts, grouping episodes by prompt (RTG) for advantage.

        The episode collection order is restructured so that each group
        contains one episode per prompt (i.e. one per RTG value).  This
        allows ``compute_group_relative_advantages`` to compare different
        RTG-conditioned behaviours within the same group.

        With ``len(prompts)`` RTG values and ``group_size`` repeats:
        Total episodes = ``group_size * len(prompts)``
        Groups of size ``len(prompts)`` where each group has one episode per RTG.
        """
        if not prompts:
            raise ValueError("prompts must be non-empty.")
        if group_size <= 0:
            raise ValueError("group_size must be positive.")

        states_records: list[torch.Tensor] = []
        actions_records: list[torch.Tensor] = []
        rtg_records: list[torch.Tensor] = []
        timestep_records: list[torch.Tensor] = []
        mask_records: list[torch.Tensor] = []
        sampled_action_records: list[torch.Tensor] = []
        old_log_prob_records: list[float] = []
        ref_log_prob_records: list[float] = []
        reward_records: list[float] = []
        env_reward_records: list[float] = []
        prompt_index_records: list[int] = []
        returns: list[float] = []
        episode_steps: list[int] = []

        num_rtgs = len(prompts)

        self.model.eval()
        # Restructured: outer loop over groups, inner loop over RTG prompts
        # Each group contains one episode per RTG value
        for group_idx in range(group_size):
            for prompt_index, prompt in enumerate(prompts):
                env = env_factory()
                obs, _ = _reset_env(env, seed=prompt.seed, options=prompt.options)
                obs = np.asarray(obs, dtype=np.float32)

                state_buffer = [obs.copy()]
                action_buffer = [np.zeros(int(self.model.act_dim), dtype=np.float32)]
                rtg_buffer = [float(prompt.rtg_value)]
                timestep_buffer = [int(getattr(env, "current_step", 0))]
                max_steps = _episode_max_steps(env, prompt)

                episode_return = 0.0
                step_count = 0
                terminated = False
                truncated = False

                while not (terminated or truncated) and step_count < max_steps:
                    context = _build_dt_context(
                        model=self.model,
                        state_buffer=state_buffer,
                        action_buffer=action_buffer,
                        rtg_buffer=rtg_buffer,
                        timestep_buffer=timestep_buffer,
                        device=self.device,
                    )
                    states_t, actions_t, rtgs_t, timesteps_t, mask_t = context
                    with torch.no_grad():
                        current_dist, ref_dist = self._action_distributions(
                            states_t, actions_t, rtgs_t, timesteps_t, mask_t
                        )
                        sampled_action = current_dist.rsample()
                        old_log_prob = float(current_dist.log_prob(sampled_action).item())
                        ref_log_prob = float(ref_dist.log_prob(sampled_action).item())

                    action_np = sampled_action.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                    # Clamp FCAS bids (dims 1+) to [0, 1] for full_fcas mode
                    if len(action_np) > 1:
                        action_np[1:] = np.clip(action_np[1:], 0.0, 1.0)
                    next_obs, reward, terminated, truncated, info = env.step(action_np)
                    next_obs = np.asarray(next_obs, dtype=np.float32)
                    env_reward = float(reward)
                    reward = self._shape_reward(env_reward, info)

                    states_records.append(states_t.squeeze(0).detach().cpu())
                    actions_records.append(actions_t.squeeze(0).detach().cpu())
                    rtg_records.append(rtgs_t.squeeze(0).detach().cpu())
                    timestep_records.append(timesteps_t.squeeze(0).detach().cpu())
                    mask_records.append(mask_t.squeeze(0).detach().cpu())
                    sampled_action_records.append(sampled_action.squeeze(0).squeeze(0).detach().cpu())
                    old_log_prob_records.append(old_log_prob)
                    ref_log_prob_records.append(ref_log_prob)
                    reward_records.append(float(reward))
                    env_reward_records.append(env_reward)
                    prompt_index_records.append(prompt_index)

                    episode_return += float(reward)
                    step_count += 1

                    action_buffer[-1] = action_np
                    next_rtg = stable_rtg_update(
                        rtg_buffer[-1],
                        float(reward),
                        dt_gamma=float(dt_gamma),
                        initial_rtg=float(prompt.rtg_value),
                    )
                    next_rtg = _validate_grpo_rtg(next_rtg, step_count=step_count, dt_gamma=float(dt_gamma))
                    state_buffer.append(next_obs.copy())
                    action_buffer.append(np.zeros(int(self.model.act_dim), dtype=np.float32))
                    rtg_buffer.append(float(next_rtg))
                    timestep_buffer.append(int(getattr(env, "current_step", step_count)))

                    if len(state_buffer) > int(self.model.context_len):
                        state_buffer = state_buffer[-int(self.model.context_len):]
                        action_buffer = action_buffer[-int(self.model.context_len):]
                        rtg_buffer = rtg_buffer[-int(self.model.context_len):]
                        timestep_buffer = timestep_buffer[-int(self.model.context_len):]

                returns.append(float(episode_return))
                episode_steps.append(step_count)
        # Compute advantages across RTG prompts within each group
        advantages_per_episode = compute_group_relative_advantages(returns, num_rtgs)
        step_advantages: list[float] = []
        for advantage, step_count in zip(advantages_per_episode.tolist(), episode_steps):
            step_advantages.extend([advantage] * step_count)

        batch_device = "cpu" if self.cpu_rollout_buffer else self.device
        batch = GRPORolloutBatch(
            states=torch.stack(states_records).to(batch_device),
            actions=torch.stack(actions_records).to(batch_device),
            rtgs=torch.stack(rtg_records).to(batch_device),
            timesteps=torch.stack(timestep_records).to(batch_device),
            attention_mask=torch.stack(mask_records).to(batch_device),
            sampled_actions=torch.stack(sampled_action_records).to(batch_device),
            old_log_probs=torch.tensor(old_log_prob_records, dtype=torch.float32, device=batch_device),
            ref_log_probs=torch.tensor(ref_log_prob_records, dtype=torch.float32, device=batch_device),
            advantages=torch.tensor(step_advantages, dtype=torch.float32, device=batch_device),
            rewards=torch.tensor(reward_records, dtype=torch.float32, device=batch_device),
            env_rewards=torch.tensor(env_reward_records, dtype=torch.float32, device=batch_device),
            returns=torch.tensor(returns, dtype=torch.float32, device=batch_device),
            prompt_indices=torch.tensor(prompt_index_records, dtype=torch.long, device=batch_device),
        )
        self.model.train()
        return batch

    def update(
        self,
        batch: GRPORolloutBatch,
        *,
        update_epochs: int = 1,
        minibatch_size: int = 128,
    ) -> dict[str, float]:
        if batch.num_steps == 0:
            raise ValueError("Cannot update with an empty rollout batch.")

        metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "entropy_bonus": 0.0,
        }
        update_steps = 0
        indices = torch.arange(batch.num_steps)
        _batch_on_cpu = self.cpu_rollout_buffer and str(batch.states.device) == "cpu"

        for _ in range(max(1, int(update_epochs))):
            shuffled = indices[torch.randperm(batch.num_steps)]
            for start in range(0, batch.num_steps, max(1, int(minibatch_size))):
                mb_idx = shuffled[start : start + max(1, int(minibatch_size))]
                if _batch_on_cpu:
                    # Move mini-batch to GPU on-the-fly — saves ~6 GB vs keeping full batch on GPU
                    states = batch.states[mb_idx].to(self.device)
                    actions = batch.actions[mb_idx].to(self.device)
                    rtgs = batch.rtgs[mb_idx].to(self.device)
                    timesteps = batch.timesteps[mb_idx].to(self.device)
                    attention_mask = batch.attention_mask[mb_idx].to(self.device)
                    sampled_actions = batch.sampled_actions[mb_idx].to(self.device)
                    old_log_probs = batch.old_log_probs[mb_idx].to(self.device)
                    ref_log_probs = batch.ref_log_probs[mb_idx].to(self.device)
                    advantages = batch.advantages[mb_idx].to(self.device)
                else:
                    states = batch.states[mb_idx]
                    actions = batch.actions[mb_idx]
                    rtgs = batch.rtgs[mb_idx]
                    timesteps = batch.timesteps[mb_idx]
                    attention_mask = batch.attention_mask[mb_idx]
                    sampled_actions = batch.sampled_actions[mb_idx]
                    old_log_probs = batch.old_log_probs[mb_idx]
                    ref_log_probs = batch.ref_log_probs[mb_idx]
                    advantages = batch.advantages[mb_idx]

                current_dist, _ = self._action_distributions(
                    states, actions, rtgs, timesteps, attention_mask
                )
                current_log_probs = current_dist.log_prob(sampled_actions.unsqueeze(1)).squeeze(-1)
                log_ratio = current_log_probs - old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                ref_gap = ref_log_probs - current_log_probs
                kl_loss = (torch.exp(ref_gap) - ref_gap - 1.0).mean()

                base_entropy = current_dist.base_dist.entropy().mean()
                loss = policy_loss + self.kl_coeff * kl_loss - self.entropy_coeff * base_entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.log_std.requires_grad:
                    torch.nn.utils.clip_grad_norm_([self.log_std], self.grad_clip_norm)
                self.optimizer.step()

                metrics["loss"] += float(loss.detach().cpu().item())
                metrics["policy_loss"] += float(policy_loss.detach().cpu().item())
                metrics["kl_loss"] += float(kl_loss.detach().cpu().item())
                metrics["entropy_bonus"] += float(base_entropy.detach().cpu().item())
                update_steps += 1

        if update_steps > 0:
            for key in metrics:
                metrics[key] /= float(update_steps)
        metrics["log_std_mean"] = float(self.log_std.detach().mean().cpu().item())
        return metrics

    def train(
        self,
        env_factory: Callable[[], Any],
        *,
        prompts: Sequence[GRPOPrompt],
        iterations: int = 1,
        group_size: int = 4,
        update_epochs: int = 1,
        minibatch_size: int = 128,
        dt_gamma: float = 0.99,
        sync_reference_every: int = 0,
        adaptive_rtg: bool = False,
        adaptive_rtg_spread: float = 3.0,
        adaptive_rtg_dist: str = "gaussian",
        adaptive_rtg_ewma_alpha: float = 0.1,
        adaptive_rtg_seed: int | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        prompts_for_iteration = list(prompts)
        self._last_prompts = list(prompts_for_iteration)
        self._adaptive_rtg_ewma = None
        for iteration in range(max(1, int(iterations))):
            batch = self.collect_rollouts(
                env_factory,
                prompts=prompts_for_iteration,
                group_size=group_size,
                dt_gamma=dt_gamma,
            )
            metrics = self.update(
                batch,
                update_epochs=update_epochs,
                minibatch_size=minibatch_size,
            )
            reference_synced = False
            if sync_reference_every and (iteration + 1) % int(sync_reference_every) == 0:
                self._sync_reference_model()
                reference_synced = True

            adaptive_rtg_optimum = None
            if adaptive_rtg:
                realized_mean_return = float(batch.returns.mean().detach().cpu().item())
                if self._adaptive_rtg_ewma is None:
                    self._adaptive_rtg_ewma = realized_mean_return
                else:
                    alpha = float(adaptive_rtg_ewma_alpha)
                    self._adaptive_rtg_ewma = ((1.0 - alpha) * self._adaptive_rtg_ewma) + (alpha * realized_mean_return)
                adaptive_rtg_optimum = float(self._adaptive_rtg_ewma)
                prompts_for_iteration = self._resample_prompts(
                    prompts_for_iteration,
                    optimum=adaptive_rtg_optimum,
                    spread=adaptive_rtg_spread,
                    distribution=adaptive_rtg_dist,
                    seed=None if adaptive_rtg_seed is None else int(adaptive_rtg_seed) + iteration + 1,
                )
            self._last_prompts = list(prompts_for_iteration)

            prompt_rtgs = [float(prompt.rtg_value) for prompt in prompts_for_iteration]
            metrics.update(
                {
                    "iteration": iteration + 1,
                    "episodes_collected": float(batch.num_episodes),
                    "steps_collected": float(batch.num_steps),
                    "mean_return": float(batch.returns.mean().detach().cpu().item()),
                    "max_return": float(batch.returns.max().detach().cpu().item()),
                    "min_return": float(batch.returns.min().detach().cpu().item()),
                    "mean_reward": float(batch.rewards.mean().detach().cpu().item()),
                    "mean_env_reward": float(batch.env_rewards.mean().detach().cpu().item()),
                    "mean_advantage": float(batch.advantages.mean().detach().cpu().item()),
                    "reference_synced": float(reference_synced),
                    "degradation_penalty_weight": float(self.degradation_penalty_weight),
                    "adaptive_rtg_enabled": float(adaptive_rtg),
                    "adaptive_rtg_optimum": float(adaptive_rtg_optimum) if adaptive_rtg_optimum is not None else float("nan"),
                    "prompt_rtg_min": float(min(prompt_rtgs)),
                    "prompt_rtg_max": float(max(prompt_rtgs)),
                    "prompt_rtg_mean": float(sum(prompt_rtgs) / len(prompt_rtgs)),
                }
            )
            history.append(metrics)
        return history
