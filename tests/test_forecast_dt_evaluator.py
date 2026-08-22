"""Smoke tests for ForecastDecisionTransformer agent integration."""
import sys
from datetime import datetime
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


def test_aemo_agent_auto_rtg_mode_resolves_by_impact(tiny_model, mock_env):
    identity_agent = AEMOAgent(mock_env, algorithm="dt", model=tiny_model, rtg_mode="auto")
    assert identity_agent.rtg_mode == "j_t_soc"

    impact_env = _MockEnv()
    impact_env._impact = type("PiecewiseMeritOrderImpact", (), {})()
    impact_agent = AEMOAgent(impact_env, algorithm="dt", model=tiny_model, rtg_mode="auto")
    assert impact_agent.rtg_mode == "constant"


def test_jtsoc_init_passes_impact_model_and_market_metadata(monkeypatch: pytest.MonkeyPatch, tiny_model):
    env = _MockEnv()
    env.aemo_data = pl.DataFrame(
        {
            "REGIONID": ["NSW1", "NSW1"],
            "RRP": [100.0, 120.0],
            "TOTALDEMAND": [900.0, 950.0],
            "SETTLEMENTDATE": [datetime(2024, 1, 1, 0, 0), datetime(2024, 1, 1, 0, 5)],
        }
    )
    env._impact = type("PiecewiseMeritOrderImpact", (), {"intensity": 1.0})()

    captured: dict[str, object] = {}

    def fake_profile(cache_dir, region, step_h):
        return lambda month, hour: 77.0

    def fake_forecast(aemo_data, profile):
        return [{"RRP": profile(1, 0)}, {"RRP": profile(1, 0)}]

    def fake_compute(env_arg, forecast_arg, **kwargs):
        captured["env"] = env_arg
        captured["forecast"] = forecast_arg
        captured["kwargs"] = kwargs
        return np.zeros((3, 2), dtype=float), np.array([0.0, env_arg.battery_capacity], dtype=float)

    monkeypatch.setattr("aemo_sdp_executor.build_seasonal_rrp_profile", fake_profile)
    monkeypatch.setattr("aemo_sdp_executor.build_rrp_forecast", fake_forecast)
    monkeypatch.setattr("aemo_sdp_executor.compute_cost_to_go_table", fake_compute)

    agent = AEMOAgent(env, algorithm="dt", model=tiny_model, rtg_mode="j_t_soc")
    agent._init_jtsoc_table()

    assert captured["env"] is env
    assert captured["kwargs"]["impact_model"] is env._impact
    assert captured["forecast"][0]["TOTALDEMAND"] == pytest.approx(900.0)
    assert "SETTLEMENTDATE" in captured["forecast"][0]
