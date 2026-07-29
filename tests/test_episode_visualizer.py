"""
Smoke tests for EpisodeVisualizer in helper.py.
"""

import sys, os, pytest
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from helper import EpisodeVisualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def household_log():
    np.random.seed(42)
    N = 100
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [
            [np.sin(i/48*2*np.pi), np.cos(i/48*2*np.pi), np.sin(i/365*2*np.pi),
             np.cos(i/365*2*np.pi), 0.0, max(0, 3.0*np.sin(i/48*np.pi)),
             2.0+0.5*np.sin(i/48*np.pi+1), 0.25, 0.05, 3.5+0.5*np.sin(i/10)]
            for i in range(N)
        ],
        'action': [[float(np.clip(np.random.normal(0, 0.5), -1, 1))] for _ in range(N)],
        'reward': [float(-0.1+np.random.normal(0, 0.02)) for _ in range(N)],
        'info': [
            {'battery_level': 3.5+0.5*np.sin(i/10), 'grid_energy': float(np.random.normal(0,1)),
             'step_degradation': 0.001, 'total_degradation': 0.001*i, 'capacity_kwh': 7.0-0.001*i}
            for i in range(N)
        ],
    })


@pytest.fixture
def aemo_simple_log():
    np.random.seed(42)
    N = 100
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [[0.0]*18 for _ in range(N)],
        'action': [[float(np.clip(np.random.normal(0,0.3),-1,1))] for _ in range(N)],
        'reward': [float(np.random.normal(0,0.05)) for _ in range(N)],
        'info': [
            {'battery_soc': 5.0+np.sin(i/10), 'energy_price': 50+30*np.sin(i/48*np.pi),
             'energy_revenue': float(np.random.normal(0,10)), 'fcas_revenue': 0.0,
             'battery_dispatch': float(np.random.normal(0,2)), 'step_degradation': 0.0001,
             'total_degradation': 0.0001*i, 'capacity_mwh': 10.0-0.001*i}
            for i in range(N)
        ],
    })


@pytest.fixture
def aemo_mm_log():
    np.random.seed(42)
    N = 100
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [[0.0]*18 for _ in range(N)],
        'action': [
            [float(np.clip(np.random.normal(0,0.3),-1,1)),
             float(np.clip(np.random.uniform(0,0.3),0,1)),
             float(np.clip(np.random.uniform(0,0.3),0,1))]
            for _ in range(N)
        ],
        'reward': [float(np.random.normal(0,0.05)) for _ in range(N)],
        'info': [
            {'battery_soc': 5.0+np.sin(i/10), 'energy_price': 50+30*np.sin(i/48*np.pi),
             'energy_revenue': float(np.random.normal(0,10)),
             'fcas_revenue': float(np.random.uniform(0,5)), 'fcas_raise_bid': float(np.random.uniform(0,0.3)),
             'fcas_lower_bid': float(np.random.uniform(0,0.3)), 'battery_dispatch': float(np.random.normal(0,2)),
             'step_degradation': 0.0001, 'total_degradation': 0.0001*i, 'capacity_mwh': 10.0-0.001*i}
            for i in range(N)
        ],
    })


@pytest.fixture
def aemo_long_log():
    np.random.seed(0)
    N = 480
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [[0.0]*18 for _ in range(N)],
        'action': [[float(np.clip(np.random.normal(0,0.3),-1,1))] for _ in range(N)],
        'reward': [float(np.random.normal(0,0.05)) for _ in range(N)],
        'info': [
            {'battery_soc': 5.0+2.0*np.sin(i/48*2*np.pi), 'energy_price': 50+30*np.sin(i/48*np.pi),
             'energy_revenue': float(np.random.normal(0,10)), 'fcas_revenue': float(np.random.uniform(0,2)),
             'degradation_cost': 0.2+0.01*(i%24), 'battery_dispatch': float(np.random.normal(0,2)),
             'step_degradation': 0.0001, 'total_degradation': 0.0001*i, 'capacity_mwh': 10.0-0.0005*i}
            for i in range(N)
        ],
    })


@pytest.fixture
def household_long_log():
    np.random.seed(1)
    N = 480
    return pl.DataFrame({
        'step': list(range(N)),
        'raw_observation': [
            [0.0,0.0,0.0,0.0,0.0, max(0,3.0*np.sin(i/48*np.pi)), 2.0, 0.25, 0.05,
             3.5+0.5*np.sin(i/10), 0.001]
            for i in range(N)
        ],
        'action': [[float(np.clip(np.random.normal(0,0.3),-1,1))] for _ in range(N)],
        'reward': [float(np.random.normal(-0.05,0.02)) for _ in range(N)],
        'info': [
            {'battery_level': 3.5+0.5*np.sin(i/10), 'grid_energy': float(np.random.normal(0,1)),
             'step_degradation': 0.001, 'total_degradation': 0.001*i, 'capacity_kwh': 7.0-0.001*i}
            for i in range(N)
        ],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEpisodeVisualizerDetection:
    def test_detects_household(self, household_log):
        assert EpisodeVisualizer(household_log).env_type == "household"

    def test_detects_aemo(self, aemo_simple_log):
        assert EpisodeVisualizer(aemo_simple_log).env_type == "aemo"

    def test_force_env_type(self, household_log):
        assert EpisodeVisualizer(household_log, env_type="aemo").env_type == "aemo"

    def test_empty_log_defaults_household(self):
        empty = pl.DataFrame({'step':[],'raw_observation':[],'action':[],'reward':[],'info':[]})
        assert EpisodeVisualizer(empty).env_type == "household"


class TestHouseholdPlot:
    def test_plot_returns_figure(self, household_log):
        fig = EpisodeVisualizer(household_log, step_duration=0.5).plot(num_hours=24, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAEMOPlot:
    def test_simple_mode_three_panels(self, aemo_simple_log):
        fig = EpisodeVisualizer(aemo_simple_log, step_duration=0.5).plot(num_hours=24, show=False)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_multi_market_four_panels(self, aemo_mm_log):
        fig = EpisodeVisualizer(aemo_mm_log, step_duration=0.5).plot(num_hours=24, show=False)
        assert len(fig.axes) == 4
        plt.close(fig)


class TestEdgeCases:
    def test_window_exceeds_data(self, household_log):
        fig = EpisodeVisualizer(household_log, step_duration=0.5).plot(num_hours=200, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_window_raises(self, household_log):
        with pytest.raises(ValueError, match="empty"):
            EpisodeVisualizer(household_log, step_duration=0.5).plot(start_step=9999, num_hours=24, show=False)


class TestLongHorizonPlot:
    def test_aemo_returns_figure(self, aemo_long_log):
        fig = EpisodeVisualizer(aemo_long_log, step_duration=0.5).plot_long_horizon(period_hours=24, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_aemo_has_six_panels(self, aemo_long_log):
        fig = EpisodeVisualizer(aemo_long_log, step_duration=0.5).plot_long_horizon(period_hours=24, show=False)
        assert len(fig.axes) == 6
        plt.close(fig)

    def test_household_has_five_panels(self, household_long_log):
        fig = EpisodeVisualizer(household_long_log, step_duration=0.5).plot_long_horizon(period_hours=24, show=False)
        assert len(fig.axes) == 5
        plt.close(fig)

    def test_empty_window_raises(self, aemo_long_log):
        with pytest.raises(ValueError, match="empty"):
            EpisodeVisualizer(aemo_long_log, step_duration=0.5).plot_long_horizon(period_hours=24, start_step=9999, show=False)
