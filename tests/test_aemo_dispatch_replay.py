import os
import sys
import warnings
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from decision import AEMOAgent


def _make_stub_env(action_mode: str = 'multi_market', max_battery_flow: float = 5.0,
                   episode_start_idx: int = 0):
    # The dispatch data in this test file uses 30-min timestamps, so the stub grid
    # must match.  step_duration is intentionally kept at 0.5h here because the
    # dispatch replay tests exercise timestamp alignment, not step-duration physics.
    grid_times = [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 30),
        datetime(2024, 1, 1, 1, 0),
        datetime(2024, 1, 1, 1, 30),
    ]
    return SimpleNamespace(
        aemo_data=pl.DataFrame({'SETTLEMENTDATE': grid_times}),
        step_duration=0.5,
        max_battery_flow=max_battery_flow,
        action_mode=action_mode,
        current_step=0,
        episode_start_idx=episode_start_idx,
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

    assert agent.dispatch_actions.shape == (4, 3)
    np.testing.assert_allclose(
        agent.dispatch_actions[:2],
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

    assert agent.dispatch_actions.shape[0] == 4
    np.testing.assert_allclose(agent.dispatch_actions[:2, 1:], 0.0)
    np.testing.assert_allclose(agent.dispatch_actions[:2, 0], np.array([1.0, -1.0], dtype=np.float32))


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

    assert agent.dispatch_actions.shape[0] == 4
    np.testing.assert_allclose(
        agent.dispatch_actions[:2, 0],
        np.array([-0.6, 0.6], dtype=np.float32),
    )


def test_dispatch_action_uses_episode_start_idx():
    """dispatch_actions are indexed by episode_start_idx + current_step, not just current_step."""
    # Grid has 4 timestamps (30-min spacing); episode starts at index 2
    env = _make_stub_env(action_mode='multi_market', episode_start_idx=2)
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [
            datetime(2024, 1, 1, 0, 5),   # window 00:00 → index 0
            datetime(2024, 1, 1, 0, 35),  # window 00:30 → index 1
            datetime(2024, 1, 1, 1, 5),   # window 01:00 → index 2
            datetime(2024, 1, 1, 1, 35),  # window 01:30 → index 3
        ],
        'DUID': ['BAT1'] * 4,
        'TOTALCLEARED': [1.0, 2.0, 3.0, 4.0],
        'RAISEREG': [0.0] * 4,
        'LOWERREG': [0.0] * 4,
    })

    agent = AEMOAgent(
        env,
        algorithm='dispatch',
        dispatch_data=dispatch_data,
        dispatch_duid='BAT1',
    )

    # With episode_start_idx=2 and current_step=0, the action should come from index 2
    # (i.e., from the 3rd row: TOTALCLEARED=3.0 → NET_MW=-3.0 → a0=-3/5=-0.6)
    env.current_step = 0
    action = agent._dispatch_action()
    np.testing.assert_allclose(action[0], -3.0 / 5.0, atol=1e-6)

    # With current_step=1, index = 2+1 = 3 → TOTALCLEARED=4.0 → a0=-4/5=-0.8
    env.current_step = 1
    action = agent._dispatch_action()
    np.testing.assert_allclose(action[0], -4.0 / 5.0, atol=1e-6)


def test_dispatch_action_episode_start_zero_behaves_as_before():
    """When episode_start_idx=0 (default), behaviour is unchanged."""
    env = _make_stub_env(action_mode='multi_market', episode_start_idx=0)
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 0, 35),
        ],
        'DUID': ['BAT1', 'BAT1'],
        'TOTALCLEARED': [2.0, 4.0],
        'RAISEREG': [1.0, 2.0],
        'LOWERREG': [0.5, 1.0],
    })

    agent = AEMOAgent(
        env,
        algorithm='dispatch',
        dispatch_data=dispatch_data,
        dispatch_duid='BAT1',
    )

    env.current_step = 0
    action = agent._dispatch_action()
    np.testing.assert_allclose(action[0], -2.0 / 5.0, atol=1e-6)

    env.current_step = 1
    action = agent._dispatch_action()
    np.testing.assert_allclose(action[0], -4.0 / 5.0, atol=1e-6)


def test_dispatch_action_warns_when_all_zero():
    """A warning is emitted when the aligned dispatch actions are all zero."""
    env = _make_stub_env(action_mode='multi_market')
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 0, 35),
        ],
        'DUID': ['BAT1', 'BAT1'],
        'TOTALCLEARED': [0.0, 0.0],
        'RAISEREG': [0.0, 0.0],
        'LOWERREG': [0.0, 0.0],
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        AEMOAgent(
            env,
            algorithm='dispatch',
            dispatch_data=dispatch_data,
            dispatch_duid='BAT1',
        )
    assert any('all zero' in str(warning.message).lower() for warning in w)
