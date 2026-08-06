"""
Phase 1 invariant: Oracle_PT is a ceiling over replayed policies on shared episodes.

Cheap regression version of scripts/phase1_oracle_invariant.py: on a short synthetic
identity-impact episode with zero degradation, the perfect-foresight LP oracle must
net at least as much as the FCAS rule policy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import polars as pl
from datetime import datetime, timedelta

from AEMOBatteryEnv import AEMOBatteryTradingEnv
from decision import AEMOAgent


def _frame(n=288, seed=0):
    rng = np.random.default_rng(seed)
    ts = [datetime(2024, 6, 1) + timedelta(minutes=5 * i) for i in range(n)]
    cols = {'SETTLEMENTDATE': ts, 'RRP': rng.uniform(30, 80, n),
            'TOTALDEMAND': rng.uniform(1200, 1500, n),
            'hour_sin': np.sin(np.arange(n) / 12 * 2 * np.pi),
            'hour_cos': np.cos(np.arange(n) / 12 * 2 * np.pi)}
    for svc in ['RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN', 'RAISEREG',
                'LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN', 'LOWERREG']:
        cols[f'FCAS_{svc}'] = rng.uniform(0, 8, n)
    return pl.DataFrame(cols)


def _profit(agent):
    df, _ = agent.run_episode()
    infos = df['info'].to_list()
    return sum(i.get('energy_revenue', 0) + i.get('fcas_revenue', 0) for i in infos)


def test_oracle_pt_dominates_fcas_rule():
    frame = _frame()
    env = AEMOBatteryTradingEnv(aemo_data=frame, battery_capacity=10.0,
                                max_battery_flow=5.0, step_duration=1 / 12,
                                init_battery_level=5.0, max_step=frame.height,
                                action_mode='full_fcas', degradation_mode='none',
                                random_episode_start=False)
    oracle_profit = _profit(AEMOAgent(env, algorithm='aemo_oracle'))
    rule_profit = _profit(AEMOAgent(env, algorithm='fcas_rule'))
    assert oracle_profit >= rule_profit - 1e-6, (
        f"Oracle_PT ({oracle_profit:.2f}) must be >= FCAS rule ({rule_profit:.2f}) on the shared episode")
