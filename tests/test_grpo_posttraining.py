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
    compute_group_relative_advantages,
    sample_rtg_values,
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


def test_sample_rtg_values_uniform_stays_in_range():
    values = sample_rtg_values(
        optimum=0.0, spread=5.0, count=100, distribution="uniform", seed=7
    )
    others = [v for v in values if v != 0.0]
    assert all(-5.0 <= v <= 5.0 for v in others)


def test_sample_rtg_values_lognormal_positive():
    values = sample_rtg_values(
        optimum=5.0, spread=0.5, count=100, distribution="lognormal", seed=3
    )
    others = [v for v in values if v != 5.0]
    assert all(v > 0 for v in others)
    # Mean of lognormal is exp(mu + sigma^2/2) where mu = ln(5)
