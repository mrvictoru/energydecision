import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from decision import AEMOAgent


def _make_stub_env(action_mode: str = 'multi_market', max_battery_flow: float = 5.0):
    grid_times = [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 30),
    ]
    return SimpleNamespace(
        aemo_data=pl.DataFrame({'SETTLEMENTDATE': grid_times}),
        step_duration=0.5,
        max_battery_flow=max_battery_flow,
        action_mode=action_mode,
        current_step=0,
    )


def test_single_duid_dispatch_replay_multimarket_uses_totalcleared():
    env = _make_stub_env(action_mode='multi_market')
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 0, 10),
            datetime(2024, 1, 1, 0, 35),
            datetime(2024, 1, 1, 0, 40),
        ],
        'DUID': ['KEPBG1'] * 4,
        'TOTALCLEARED': [2.0, 2.0, 4.0, 4.0],
        'RAISEREG': [1.0, 1.0, 2.0, 2.0],
        'LOWERREG': [0.5, 0.5, 1.0, 1.0],
    })

    agent = AEMOAgent(
        env,
        algorithm='dispatch',
        dispatch_data=dispatch_data,
        dispatch_duid='KEPBG1',
    )

    assert agent.dispatch_actions.shape == (2, 3)
    np.testing.assert_allclose(
        agent.dispatch_actions,
        np.array([
            [-0.4, 0.2, 0.1],
            [-0.8, 0.4, 0.2],
        ], dtype=np.float32),
    )


def test_single_duid_dispatch_replay_defaults_missing_fcas_to_zero():
    env = _make_stub_env(action_mode='multi_market')
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 0, 35),
        ],
        'DUID': ['KEPBG1', 'KEPBG1'],
        'TOTALCLEARED': [5.0, -5.0],
    })

    agent = AEMOAgent(
        env,
        algorithm='dispatch',
        dispatch_data=dispatch_data,
        dispatch_duid='KEPBG1',
        assume_single_duid_is_generator=False,
    )

    assert agent.dispatch_actions.shape == (2, 3)
    np.testing.assert_allclose(agent.dispatch_actions[:, 1:], 0.0)
    np.testing.assert_allclose(agent.dispatch_actions[:, 0], np.array([1.0, -1.0], dtype=np.float32))


def test_paired_duid_dispatch_replay_combines_gen_and_load_streams():
    env = _make_stub_env(action_mode='simple')
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 0, 35),
            datetime(2024, 1, 1, 0, 35),
        ],
        'DUID': ['BATG1', 'BATL1', 'BATG1', 'BATL1'],
        'TOTALCLEARED': [4.0, 1.0, 1.0, 4.0],
        'RAISEREG': [1.0, 0.5, 0.5, 0.25],
        'LOWERREG': [0.25, 0.5, 0.5, 1.0],
    })

    agent = AEMOAgent(
        env,
        algorithm='dispatch',
        dispatch_data=dispatch_data,
        dispatch_duid_gen='BATG1',
        dispatch_duid_load='BATL1',
    )

    assert agent.dispatch_actions.shape == (2, 1)
    np.testing.assert_allclose(
        agent.dispatch_actions[:, 0],
        np.array([-0.6, 0.6], dtype=np.float32),
    )