"""Smoke tests for ForecastDecisionTransformer agent integration."""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision import AEMOAgent
from forecast_decision_transformer import ForecastDecisionTransformer


class _MockEnv:
    aemo_data = pl.DataFrame({"a": [1.0] * 50})
    episode_start_idx = 0
    current_step = 0
    battery_capacity = 10.0
    max_battery_flow = 1.0
    step_duration = 0.5
    action_mode = "full_fcas"
    normalize_obs = True
    base_deg_DoD = 80.0
    _raw_col_bounds = {"RRP": (-100, 500)}
    max_step = 10

    class _Space:
        shape = (9,)
        def sample(self):
            return np.zeros(9, dtype=np.float32)

    action_space = _Space()
    observation_space = _Space()

    def get_raw_obs(self):
        return np.zeros(18, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.episode_start_idx = 0
        return np.zeros(18, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_step
        return np.zeros(18, dtype=np.float32), 0.0, False, done, {}


@pytest.fixture
def tiny_model():
    m = ForecastDecisionTransformer(
        state_dim=18, act_dim=9, context_len=4, forecast_len=2,
        h_dim=32, n_block=2, n_heads=4, n_kv_heads=4,
        qk_norm=False, tie_weights=False, rope_enabled=False,
    )
    return m.eval()


@pytest.fixture
def mock_env():
    return _MockEnv()


def test_forecast_dt_choose_action_no_npz(tiny_model, mock_env):
    agent = AEMOAgent(mock_env, algorithm="dt", model=tiny_model)
    obs = np.zeros(18, dtype=np.float32)
    agent.dt_states_buffer = [obs.copy()]
    agent.dt_actions_buffer = [np.zeros(9, dtype=np.float32)]
    agent.dt_rtgs_buffer = [0.0]
    agent.dt_timesteps_buffer = [0]
    action = agent.choose_action(obs)
    assert isinstance(action, list)
    assert len(action) == 9
    for v in action:
        assert np.isfinite(v), f"Non-finite action value: {v}"


def test_forecast_dt_choose_action_with_npz(tiny_model, mock_env, tmp_path):
    f_npz = tmp_path / "test_forecast.npz"
    np.savez_compressed(f_npz, forecast_map=np.zeros((50, tiny_model.forecast_len, 6), dtype=np.float32))
    agent = AEMOAgent(mock_env, algorithm="dt", model=tiny_model, forecast_npz_path=str(f_npz))
    obs = np.zeros(18, dtype=np.float32)
    agent.dt_states_buffer = [obs.copy()]
    agent.dt_actions_buffer = [np.zeros(9, dtype=np.float32)]
    agent.dt_rtgs_buffer = [0.0]
    agent.dt_timesteps_buffer = [0]
    action = agent.choose_action(obs)
    assert isinstance(action, list)
    assert len(action) == 9
    for v in action:
        assert np.isfinite(v), f"Non-finite action value: {v}"


def test_forecast_dt_agent_run_episode(tiny_model, mock_env):
    agent = AEMOAgent(mock_env, algorithm="dt", model=tiny_model)
    episode_df, _ = agent.run_episode(display_progress=False)
    assert isinstance(episode_df, pl.DataFrame)
    assert episode_df.height > 0
