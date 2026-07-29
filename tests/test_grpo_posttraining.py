import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_transformer import DecisionTransformer  # noqa: E402
from grpo_posttraining import (  # noqa: E402
    GRPOPrompt,
    GRPOTrainer,
    MAX_ABS_GRPO_RTG,
    compute_group_relative_advantages,
    sample_rtg_values,
    stable_rtg_update,
)


class TinyContinuousEnv:
    def __init__(self, target: float = 0.5, max_step: int = 3):
        self.default_target = float(target)
        self.max_step = int(max_step)
        self.current_step = 0
        self.target = self.default_target

    def reset(self, seed=None, options=None):
        _ = seed
        self.current_step = 0
        self.target = float((options or {}).get("target", self.default_target))
        obs = np.array([self.target, 0.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        action_value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        reward = 1.0 - abs(action_value - self.target)
        self.current_step += 1
        terminated = self.current_step >= self.max_step
        obs = np.array(
            [self.target, self.current_step / max(1, self.max_step)],
            dtype=np.float32,
        )
        return obs, reward, terminated, False, {}


class TinyContinuousEnvWithDegradation(TinyContinuousEnv):
    def __init__(self, target: float = 0.5, max_step: int = 3, degradation_cost: float = 0.25):
        super().__init__(target=target, max_step=max_step)
        self.degradation_cost = float(degradation_cost)

    def step(self, action):
        obs, reward, terminated, truncated, _ = super().step(action)
        return obs, reward, terminated, truncated, {"degradation_cost": self.degradation_cost}


def _build_model() -> DecisionTransformer:
    torch.manual_seed(0)
    return DecisionTransformer(
        state_dim=2,
        act_dim=1,
        n_block=1,
        h_dim=16,
        context_len=4,
        n_heads=4,
        drop_p=0.0,
        max_timestep=16,
    )


def test_compute_group_relative_advantages_normalizes_per_group():
    returns = [1.0, 3.0, 10.0, 14.0]
    advantages = compute_group_relative_advantages(returns, group_size=2)
    assert advantages.shape == (4,)
    np.testing.assert_allclose(advantages[:2], np.array([-1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(advantages[2:], np.array([-1.0, 1.0], dtype=np.float32))


def test_compute_group_relative_advantages_handles_zero_variance():
    advantages = compute_group_relative_advantages([2.0, 2.0], group_size=2)
    np.testing.assert_allclose(advantages, np.zeros(2, dtype=np.float32))


def test_grpo_collect_rollouts_returns_consistent_batch():
    model = _build_model()
    trainer = GRPOTrainer(model, device="cpu", trainable_log_std=False)

    batch = trainer.collect_rollouts(
        lambda: TinyContinuousEnv(),
        prompts=[GRPOPrompt(options={"target": 0.25})],
        group_size=3,
    )

    assert batch.num_episodes == 3
    assert batch.num_steps == 9
    assert batch.states.shape == (9, 4, 2)
    assert batch.actions.shape == (9, 4, 1)
    assert batch.rtgs.shape == (9, 4, 1)
    assert batch.sampled_actions.shape == (9, 1)
    assert batch.advantages.shape == (9,)
    assert torch.isfinite(batch.returns).all()


def test_grpo_train_updates_log_std_and_reports_metrics():
    model = _build_model()
    trainer = GRPOTrainer(
        model,
        device="cpu",
        lr=5e-3,
        trainable_log_std=True,
        initial_log_std=-0.5,
    )
    starting_log_std = trainer.log_std.detach().clone()

    history = trainer.train(
        lambda: TinyContinuousEnv(),
        prompts=[
            GRPOPrompt(options={"target": -0.4}),
            GRPOPrompt(options={"target": 0.6}),
        ],
        iterations=1,
        group_size=2,
        update_epochs=2,
        minibatch_size=4,
    )

    assert len(history) == 1
    metrics = history[0]
    assert metrics["episodes_collected"] == pytest.approx(4.0)
    assert metrics["steps_collected"] == pytest.approx(12.0)
    assert "mean_return" in metrics
    assert "kl_loss" in metrics
    assert not torch.allclose(starting_log_std, trainer.log_std.detach())


def test_grpo_collect_rollouts_applies_degradation_shaping():
    model = _build_model()
    trainer = GRPOTrainer(
        model,
        device="cpu",
        trainable_log_std=False,
        degradation_penalty_weight=1.5,
    )

    batch = trainer.collect_rollouts(
        lambda: TinyContinuousEnvWithDegradation(target=0.25, max_step=2, degradation_cost=0.4),
        prompts=[GRPOPrompt(options={"target": 0.25})],
        group_size=1,
    )

    expected_gap = torch.full_like(batch.rewards, 0.2)
    torch.testing.assert_close(batch.env_rewards - batch.rewards, expected_gap)


def test_grpo_train_syncs_reference_model_periodically():
    model = _build_model()
    trainer = GRPOTrainer(
        model,
        device="cpu",
        lr=5e-3,
        trainable_log_std=True,
        initial_log_std=-0.5,
    )

    history = trainer.train(
        lambda: TinyContinuousEnv(),
        prompts=[
            GRPOPrompt(options={"target": -0.4}),
            GRPOPrompt(options={"target": 0.6}),
        ],
        iterations=2,
        group_size=2,
        update_epochs=1,
        minibatch_size=4,
        sync_reference_every=1,
    )

    assert [row["reference_synced"] for row in history] == pytest.approx([1.0, 1.0])
    model_state = trainer.model.state_dict()
    reference_state = trainer.reference_model.state_dict()
    assert set(model_state.keys()) == set(reference_state.keys())
    for key in model_state:
        torch.testing.assert_close(model_state[key], reference_state[key])


def test_grpo_train_adapts_prompt_rtgs_between_iterations():
    model = _build_model()
    trainer = GRPOTrainer(model, device="cpu", trainable_log_std=False)
    initial_prompts = [
        GRPOPrompt(seed=10, options={"target": -0.4}, rtg_value=-1.0, max_steps=3),
        GRPOPrompt(seed=11, options={"target": 0.6}, rtg_value=1.0, max_steps=3),
    ]

    history = trainer.train(
        lambda: TinyContinuousEnv(),
        prompts=initial_prompts,
        iterations=2,
        group_size=1,
        update_epochs=1,
        minibatch_size=2,
        adaptive_rtg=True,
        adaptive_rtg_spread=0.25,
        adaptive_rtg_dist="gaussian",
        adaptive_rtg_seed=123,
    )

    assert len(history) == 2
    assert history[0]["adaptive_rtg_enabled"] == pytest.approx(1.0)
    assert trainer._last_prompts[0].seed == initial_prompts[0].seed
    assert trainer._last_prompts[0].options == initial_prompts[0].options
    assert len(trainer._last_prompts) == len(initial_prompts)
    final_rtgs = [prompt.rtg_value for prompt in trainer._last_prompts]
    initial_rtgs = [prompt.rtg_value for prompt in initial_prompts]
    assert final_rtgs != initial_rtgs


def test_sample_rtg_values_always_includes_optimum():
    values = sample_rtg_values(optimum=5.0, spread=1.0, count=4, distribution="gaussian", seed=42)
    assert len(values) == 4
    assert 5.0 in values
    assert all(isinstance(v, float) for v in values)


def test_sample_rtg_values_count_clamps_to_one():
    values = sample_rtg_values(optimum=2.0, spread=0.5, count=1, distribution="gaussian", seed=0)
    assert values == [2.0]


def test_sample_rtg_values_zero_count_returns_only_optimum():
    """count=0 still returns the optimum (always included) — no other samples generated."""
    values = sample_rtg_values(optimum=2.0, spread=0.5, count=0)
    assert values == [2.0]


def test_sample_rtg_values_gaussian_centers_on_optimum():
    """Samples should be roughly centered on the optimum (mean ≈ optimum)."""
    values = sample_rtg_values(
        optimum=10.0, spread=2.0, count=200, distribution="gaussian", seed=123
    )
    others = [v for v in values if v != 10.0]
    assert abs(np.mean(others) - 10.0) < 0.5  # sample mean within 0.5 of optimum
    # Standard deviation should be close to the input spread
    assert abs(np.std(others) - 2.0) < 0.5


def test_stable_rtg_update_undiscounted_is_exact():
    # gamma == 1.0 must remain the exact undiscounted recurrence R_{t+1} = R_t - r_t
    rtg = 10.0
    for _ in range(5):
        rtg = stable_rtg_update(rtg, 1.0, dt_gamma=1.0, initial_rtg=10.0)
    assert rtg == pytest.approx(5.0)


def test_stable_rtg_update_discounted_matches_exact_short_horizon():
    # For a few steps the clamp should not bind, so the discounted update must
    # match the exact inverse recurrence (R_t - r_t) / gamma.
    gamma = 0.95
    rtg = exact = 2.0
    for _ in range(3):
        exact = (exact - 0.1) / gamma
        rtg = stable_rtg_update(rtg, 0.1, dt_gamma=gamma, initial_rtg=2.0)
    assert rtg == pytest.approx(exact)


def test_stable_rtg_update_bounded_on_long_horizon():
    # The exact 1/gamma recurrence overflows on long horizons; the stable update
    # must stay finite and inside the guard bound for 1728 steps at gamma=0.95.
    rtg = 5.0
    for _ in range(1728):
        rtg = stable_rtg_update(rtg, -0.1, dt_gamma=0.95, initial_rtg=5.0)
    assert np.isfinite(rtg)
    assert abs(rtg) < MAX_ABS_GRPO_RTG
    # Bound is derived from the initial prompt envelope.
    assert abs(rtg) <= max(abs(5.0) * 4.0, 20.0) + 1e-6


def test_stable_rtg_update_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        stable_rtg_update(float("inf"), 0.0, dt_gamma=0.95, initial_rtg=1.0)
    with pytest.raises(ValueError):
        stable_rtg_update(0.0, float("nan"), dt_gamma=0.95, initial_rtg=1.0)
