"""
Tests for EpisodeVisualizer in helper.py.

Validates the unified episode log visualizer for both SolarBatteryEnv
(household) and AEMOBatteryTradingEnv (AEMO) episode logs.
"""

import sys
import os
import pytest
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from helper import EpisodeVisualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def household_log():
    """Synthetic SolarBatteryEnv episode log (100 steps)."""
    np.random.seed(42)
    N = 100
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [
            [np.sin(i / 48 * 2 * np.pi), np.cos(i / 48 * 2 * np.pi),
             np.sin(i / 365 * 2 * np.pi), np.cos(i / 365 * 2 * np.pi),
             0.0,
             max(0, 3.0 * np.sin(i / 48 * np.pi)),    # solar [5]
             2.0 + 0.5 * np.sin(i / 48 * np.pi + 1),  # load [6]
             0.25,                                       # import price [7]
             0.05,                                       # export price [8]
             3.5 + 0.5 * np.sin(i / 10)]                # battery [-2], deg [-1]
            for i in range(N)
        ],
        'action': [[float(np.clip(np.random.normal(0, 0.5), -1, 1))]
                   for _ in range(N)],
        'reward': [float(-0.1 + np.random.normal(0, 0.02))
                   for _ in range(N)],
        'info': [
            {'battery_level': 3.5 + 0.5 * np.sin(i / 10),
             'grid_energy': float(np.random.normal(0, 1)),
             'step_degradation': 0.001,
             'total_degradation': 0.001 * i,
             'capacity_kwh': 7.0 - 0.001 * i}
            for i in range(N)
        ],
    })


@pytest.fixture
def aemo_simple_log():
    """Synthetic AEMO simple-mode episode log (100 steps, 1-element action)."""
    np.random.seed(42)
    N = 100
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [[0.0] * 18 for _ in range(N)],
        'action': [[float(np.clip(np.random.normal(0, 0.3), -1, 1))]
                   for _ in range(N)],
        'reward': [float(np.random.normal(0, 0.05)) for _ in range(N)],
        'info': [
            {'battery_soc': 5.0 + np.sin(i / 10),
             'energy_price': 50 + 30 * np.sin(i / 48 * np.pi),
             'energy_revenue': float(np.random.normal(0, 10)),
             'fcas_revenue': 0.0,
             'battery_dispatch': float(np.random.normal(0, 2)),
             'step_degradation': 0.0001,
             'total_degradation': 0.0001 * i,
             'capacity_mwh': 10.0 - 0.001 * i}
            for i in range(N)
        ],
    })


@pytest.fixture
def aemo_mm_log():
    """Synthetic AEMO multi-market log (100 steps, 3-element action)."""
    np.random.seed(42)
    N = 100
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [[0.0] * 18 for _ in range(N)],
        'action': [
            [float(np.clip(np.random.normal(0, 0.3), -1, 1)),
             float(np.clip(np.random.uniform(0, 0.3), 0, 1)),
             float(np.clip(np.random.uniform(0, 0.3), 0, 1))]
            for _ in range(N)
        ],
        'reward': [float(np.random.normal(0, 0.05)) for _ in range(N)],
        'info': [
            {'battery_soc': 5.0 + np.sin(i / 10),
             'energy_price': 50 + 30 * np.sin(i / 48 * np.pi),
             'energy_revenue': float(np.random.normal(0, 10)),
             'fcas_revenue': float(np.random.uniform(0, 5)),
             'fcas_raise_bid': float(np.random.uniform(0, 0.3)),
             'fcas_lower_bid': float(np.random.uniform(0, 0.3)),
             'battery_dispatch': float(np.random.normal(0, 2)),
             'step_degradation': 0.0001,
             'total_degradation': 0.0001 * i,
             'capacity_mwh': 10.0 - 0.001 * i}
            for i in range(N)
        ],
    })


# ---------------------------------------------------------------------------
# Tests — auto-detection
# ---------------------------------------------------------------------------

class TestEpisodeVisualizerDetection:
    """Verify environment type auto-detection."""

    def test_detects_household(self, household_log):
        vis = EpisodeVisualizer(household_log)
        assert vis.env_type == "household"

    def test_detects_aemo(self, aemo_simple_log):
        vis = EpisodeVisualizer(aemo_simple_log)
        assert vis.env_type == "aemo"

    def test_force_env_type(self, household_log):
        vis = EpisodeVisualizer(household_log, env_type="aemo")
        assert vis.env_type == "aemo"

    def test_empty_log_defaults_household(self):
        empty = pl.DataFrame({
            'step': [],
            'raw_observation': [],
            'action': [],
            'reward': [],
            'info': [],
        })
        vis = EpisodeVisualizer(empty)
        assert vis.env_type == "household"


# ---------------------------------------------------------------------------
# Tests — household plotting
# ---------------------------------------------------------------------------

class TestHouseholdPlot:
    """Household (SolarBatteryEnv) visualisation."""

    def test_plot_returns_figure(self, household_log):
        vis = EpisodeVisualizer(household_log, step_duration=0.5)
        fig = vis.plot(num_hours=24, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_has_four_axes(self, household_log):
        vis = EpisodeVisualizer(household_log, step_duration=0.5)
        fig = vis.plot(num_hours=24, show=False)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_plot_saves_file(self, household_log, tmp_path):
        vis = EpisodeVisualizer(household_log, step_duration=0.5)
        out = str(tmp_path / "household.png")
        fig = vis.plot(num_hours=24, show=False, save_path=out)
        assert os.path.exists(out)
        plt.close(fig)

    def test_plot_custom_window(self, household_log):
        vis = EpisodeVisualizer(household_log, step_duration=0.5)
        fig = vis.plot(start_step=10, num_hours=12, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — AEMO plotting
# ---------------------------------------------------------------------------

class TestAEMOPlot:
    """AEMO (AEMOBatteryTradingEnv) visualisation."""

    def test_simple_mode_three_panels(self, aemo_simple_log):
        vis = EpisodeVisualizer(aemo_simple_log, step_duration=0.5)
        fig = vis.plot(num_hours=24, show=False)
        assert len(fig.axes) == 3  # SOC, dispatch, price
        plt.close(fig)

    def test_multi_market_four_panels(self, aemo_mm_log):
        vis = EpisodeVisualizer(aemo_mm_log, step_duration=0.5)
        fig = vis.plot(num_hours=24, show=False)
        assert len(fig.axes) == 4  # SOC, dispatch, FCAS, price
        plt.close(fig)

    def test_aemo_saves_file(self, aemo_mm_log, tmp_path):
        vis = EpisodeVisualizer(aemo_mm_log, step_duration=0.5)
        out = str(tmp_path / "aemo_mm.png")
        fig = vis.plot(num_hours=24, show=False, save_path=out)
        assert os.path.exists(out)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — comparison
# ---------------------------------------------------------------------------

class TestCompare:
    """Side-by-side agent comparison."""

    def test_compare_returns_figure(self, aemo_simple_log, aemo_mm_log):
        fig = EpisodeVisualizer.compare(
            aemo_simple_log, aemo_mm_log,
            label1="A", label2="B",
            num_hours=24, step_duration=0.5, show=False,
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_compare_household(self, household_log):
        fig = EpisodeVisualizer.compare(
            household_log, household_log,
            label1="Run1", label2="Run2",
            num_hours=24, step_duration=0.5, show=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_compare_saves_file(self, aemo_simple_log, aemo_mm_log, tmp_path):
        out = str(tmp_path / "cmp.png")
        fig = EpisodeVisualizer.compare(
            aemo_simple_log, aemo_mm_log,
            num_hours=24, step_duration=0.5,
            show=False, save_path=out,
        )
        assert os.path.exists(out)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case handling."""

    def test_window_exceeds_data(self, household_log):
        """Window larger than data should not crash (clips to available)."""
        vis = EpisodeVisualizer(household_log, step_duration=0.5)
        fig = vis.plot(num_hours=200, show=False)  # 200h >> 50h of data
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_window_raises(self, household_log):
        """Starting past end of data raises ValueError."""
        vis = EpisodeVisualizer(household_log, step_duration=0.5)
        with pytest.raises(ValueError, match="empty"):
            vis.plot(start_step=9999, num_hours=24, show=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
