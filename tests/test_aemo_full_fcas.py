"""Tests for the full_fcas (9-dim) action space in AEMOBatteryTradingEnv.

These tests verify:
1. Action space shape is (9,) for full_fcas mode
2. step() accepts 9-dim actions
3. Co-optimized enablement scales when over-committed
4. Reward includes all 8 FCAS services
5. Info dict has per-service FCAS fields
6. Legacy multi_market (3-dim) still works
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from AEMOBatteryEnv import AEMOBatteryTradingEnv


def _make_mock_aemo_data(n_steps: int = 100, step_duration: float = 0.5) -> pl.DataFrame:
    """Create a small synthetic AEMO dataset with all 8 FCAS price columns."""
    timestamps = [
        datetime(2024, 1, 1, 0, 0) + timedelta(hours=i * step_duration)
        for i in range(n_steps)
    ]

    rng = np.random.default_rng(42)
    hours_arr = np.array([t.hour for t in timestamps])
    days = np.array([t.timetuple().tm_yday for t in timestamps])

    df = pl.DataFrame({
        "SETTLEMENTDATE": timestamps,
        "RRP": rng.uniform(0, 500, n_steps),
        "TOTALDEMAND": rng.uniform(3000, 8000, n_steps),
        "RRP_normalized": rng.uniform(0, 1, n_steps),
        "DEMAND_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_RAISEREG": rng.uniform(0, 50, n_steps),
        "FCAS_LOWERREG": rng.uniform(0, 30, n_steps),
        "FCAS_RAISE6SEC": rng.uniform(0, 40, n_steps),
        "FCAS_LOWER6SEC": rng.uniform(0, 20, n_steps),
        "FCAS_RAISE60SEC": rng.uniform(0, 30, n_steps),
        "FCAS_LOWER60SEC": rng.uniform(0, 15, n_steps),
        "FCAS_RAISE5MIN": rng.uniform(0, 25, n_steps),
        "FCAS_LOWER5MIN": rng.uniform(0, 12, n_steps),
        "FCAS_RAISEREG_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_LOWERREG_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_RAISE6SEC_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_LOWER6SEC_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_RAISE60SEC_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_LOWER60SEC_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_RAISE5MIN_normalized": rng.uniform(0, 1, n_steps),
        "FCAS_LOWER5MIN_normalized": rng.uniform(0, 1, n_steps),
        "GEN_solar": rng.uniform(0, 0.5, n_steps),
        "GEN_wind": rng.uniform(0, 0.5, n_steps),
    })
    return df


@pytest.fixture
def env_full_fcas():
    data = _make_mock_aemo_data(n_steps=50)
    env = AEMOBatteryTradingEnv(
        aemo_data=data,
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_battery_level=5.0,
        max_step=20,
        step_duration=0.5,
        battery_life_cost=100_000,
        action_mode="full_fcas",
        degradation_mode="simple",
    )
    return env


@pytest.fixture
def env_multi_market_legacy():
    data = _make_mock_aemo_data(n_steps=50)
    env = AEMOBatteryTradingEnv(
        aemo_data=data,
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_battery_level=5.0,
        max_step=20,
        step_duration=0.5,
        battery_life_cost=100_000,
        action_mode="multi_market",
        degradation_mode="simple",
    )
    return env


class TestFullFCASActionSpace:
    """Action space shape and bounds for full_fcas mode."""

    def test_action_space_shape(self, env_full_fcas):
        assert env_full_fcas.action_space.shape == (9,)

    def test_action_space_bounds(self, env_full_fcas):
        low = env_full_fcas.action_space.low
        high = env_full_fcas.action_space.high
        assert low[0] == -1.0
        assert high[0] == 1.0
        for i in range(1, 9):
            assert low[i] == 0.0
            assert high[i] == 1.0


class TestFullFCASStep:
    """step() with 9-dim actions."""

    def test_step_accepts_9dim_action(self, env_full_fcas):
        env_full_fcas.reset()
        action = np.zeros(9, dtype=np.float32)
        obs, reward, terminated, truncated, info = env_full_fcas.step(action)
        assert obs.shape == (18,)
        assert isinstance(reward, float)

    def test_step_all_fcas_bids(self, env_full_fcas):
        """All 8 FCAS bids set to 1.0 with energy dispatch = 0."""
        env_full_fcas.reset()
        action = np.zeros(9, dtype=np.float32)
        action[1:] = 1.0  # all FCAS bids = 1.0
        obs, reward, terminated, truncated, info = env_full_fcas.step(action)
        # With all bids at max, co-optimization should scale them down
        # but FCAS revenue should be non-zero
        assert info["fcas_revenue"] > 0

    def test_step_smoke_multiple_steps(self, env_full_fcas):
        """Run several steps to ensure no exceptions."""
        env_full_fcas.reset()
        for _ in range(10):
            action = env_full_fcas.action_space.sample()
            obs, reward, terminated, truncated, info = env_full_fcas.step(action)
            if terminated:
                break
        assert True


class TestCoOptimization:
    """Co-optimized enablement scaling tests."""

    def test_proportional_scaling(self, env_full_fcas):
        """When all raise bids are 1.0 and SOC is at 50%, the total raise
        should be capped by available headroom (SOC / step_duration)."""
        env_full_fcas.reset()
        # SOC = 5 MWh, step_duration = 0.5h → raise headroom = min(5, 5/0.5) = 5 MW
        # But energy dispatch = 0, so all 5 MW available for FCAS
        # 4 raise services × 1.0 × 5 MW = 20 MW requested, but only 5 MW available
        # → scale factor = 5/20 = 0.25
        action = np.zeros(9, dtype=np.float32)
        action[1] = 1.0  # RAISEREG
        action[3] = 1.0  # RAISE6SEC
        action[5] = 1.0  # RAISE60SEC
        action[7] = 1.0  # RAISE5MIN
        obs, reward, terminated, truncated, info = env_full_fcas.step(action)
        # FCAS revenue should be non-zero (services were enabled, just scaled down)
        assert info["fcas_revenue"] > 0
        # Each service should have been scaled to 0.25 × 5 MW = 1.25 MW
        # Check per-service revenue is present
        for svc in ["RAISEREG", "RAISE6SEC", "RAISE60SEC", "RAISE5MIN"]:
            assert f"fcas_{svc}_revenue" in info
            assert info[f"fcas_{svc}_revenue"] > 0

    def test_no_scaling_when_within_headroom(self, env_full_fcas):
        """When bids are small enough, no scaling should occur."""
        env_full_fcas.reset()
        action = np.zeros(9, dtype=np.float32)
        action[1] = 0.1  # RAISEREG = 0.1 × 5 = 0.5 MW — well within headroom
        obs, reward, terminated, truncated, info = env_full_fcas.step(action)
        assert info["fcas_RAISEREG_revenue"] > 0


class TestInfoDict:
    """Verify the info dict has per-service FCAS fields."""

    def test_per_service_fields_present(self, env_full_fcas):
        env_full_fcas.reset()
        action = np.zeros(9, dtype=np.float32)
        action[1] = 0.5  # RAISEREG bid
        obs, reward, terminated, truncated, info = env_full_fcas.step(action)
        for svc in ["RAISEREG", "LOWERREG", "RAISE6SEC", "LOWER6SEC",
                     "RAISE60SEC", "LOWER60SEC", "RAISE5MIN", "LOWER5MIN"]:
            assert f"fcas_{svc}_bid" in info
            assert f"fcas_{svc}_revenue" in info

    def test_legacy_compatibility_fields(self, env_full_fcas):
        """fcas_raise_bid and fcas_lower_bid should still be in info."""
        env_full_fcas.reset()
        action = np.zeros(9, dtype=np.float32)
        obs, reward, terminated, truncated, info = env_full_fcas.step(action)
        assert "fcas_raise_bid" in info
        assert "fcas_lower_bid" in info


class TestLegacyMultiMarket:
    """Legacy 3-dim multi_market mode still works."""

    def test_legacy_action_space_shape(self, env_multi_market_legacy):
        assert env_multi_market_legacy.action_space.shape == (3,)

    def test_legacy_step(self, env_multi_market_legacy):
        env_multi_market_legacy.reset()
        action = np.array([0.0, 0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env_multi_market_legacy.step(action)
        assert obs.shape == (18,)
        assert info["fcas_revenue"] >= 0

    def test_legacy_info_has_per_service_fields(self, env_multi_market_legacy):
        """Legacy mode should also have per-service fields (RAISEREG/LOWERREG)."""
        env_multi_market_legacy.reset()
        action = np.array([0.0, 0.5, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env_multi_market_legacy.step(action)
        assert "fcas_RAISEREG_revenue" in info
        assert "fcas_LOWERREG_revenue" in info


class TestActDimMapping:
    """Verify pretrain_decision_transformer.py accepts full_fcas."""

    def test_act_dim_mapping(self):
        from pretrain_decision_transformer import ACTION_MODE_TO_ACT_DIM
        assert "full_fcas" in ACTION_MODE_TO_ACT_DIM
        assert ACTION_MODE_TO_ACT_DIM["full_fcas"] == 9
        assert ACTION_MODE_TO_ACT_DIM["multi_market"] == 3
        assert ACTION_MODE_TO_ACT_DIM["simple"] == 1