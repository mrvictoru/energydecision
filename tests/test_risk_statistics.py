"""
Tests for risk-sensitive evaluation metrics (CVaR, VaR) and
statistical comparison utilities (bootstrap CIs, paired comparisons).
"""

import sys
import os
import math

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from helper import (
    evaluate_experiment_logs,
    evaluate_experiments,
    bootstrap_confidence_intervals,
    paired_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_episode(rewards: list[float]) -> pl.DataFrame:
    """Create a minimal episode DataFrame with the given per-step rewards."""
    return pl.DataFrame({
        "step": list(range(len(rewards))),
        "reward": rewards,
        "raw_observation": [None] * len(rewards),
        "info": [{}] * len(rewards),
    })


@pytest.fixture
def simple_logs():
    """10 episodes with linearly spaced total rewards from -5 to +4."""
    episodes = []
    for total in np.linspace(-5, 4, 10):
        # Single-step episode whose reward equals the total
        episodes.append(_make_episode([float(total)]))
    return episodes


@pytest.fixture
def deterministic_logs():
    """5 identical episodes with total reward = 10."""
    return [_make_episode([10.0]) for _ in range(5)]


@pytest.fixture
def paired_logs():
    """Two experiment log lists with matched episode pairs."""
    rng = np.random.default_rng(42)
    n = 20
    base = rng.normal(5.0, 2.0, n)
    logs_a = [_make_episode([float(v)]) for v in base]
    logs_b = [_make_episode([float(v + 1.0)]) for v in base]  # shifted +1
    return logs_a, logs_b


# ---------------------------------------------------------------------------
# CVaR / VaR tests
# ---------------------------------------------------------------------------

class TestCVaR:
    def test_var_cvar_present_in_metrics(self, simple_logs):
        """evaluate_experiment_logs returns var_5 and cvar_5 keys."""
        metrics = evaluate_experiment_logs(simple_logs)
        assert "var_5" in metrics
        assert "cvar_5" in metrics

    def test_cvar_leq_var(self, simple_logs):
        """CVaR (expected shortfall) should be <= VaR (the quantile threshold) at 5%."""
        metrics = evaluate_experiment_logs(simple_logs)
        assert metrics["cvar_5"] <= metrics["var_5"]

    def test_cvar_leq_mean(self, simple_logs):
        """CVaR at 5% should always be <= the mean reward."""
        metrics = evaluate_experiment_logs(simple_logs)
        assert metrics["cvar_5"] <= metrics["mean_reward"]

    def test_var_matches_percentile(self, simple_logs):
        """VaR at 5 % should equal the 5th percentile of episode returns."""
        metrics = evaluate_experiment_logs(simple_logs)
        assert metrics["var_5"] == pytest.approx(metrics["pct_5_reward"], abs=1e-9)

    def test_deterministic_episodes(self, deterministic_logs):
        """With identical episodes, VaR = CVaR = mean."""
        metrics = evaluate_experiment_logs(deterministic_logs)
        assert metrics["var_5"] == pytest.approx(10.0)
        assert metrics["cvar_5"] == pytest.approx(10.0)

    def test_empty_logs(self):
        """Empty logs should return 0 for risk metrics."""
        metrics = evaluate_experiment_logs([])
        assert metrics["var_5"] == 0.0
        assert metrics["cvar_5"] == 0.0

    def test_single_episode(self):
        """Single episode: VaR and CVaR both equal the single return."""
        logs = [_make_episode([3.0])]
        metrics = evaluate_experiment_logs(logs)
        assert metrics["var_5"] == pytest.approx(3.0)
        assert metrics["cvar_5"] == pytest.approx(3.0)

    def test_cvar_in_experiments_dataframe(self, simple_logs):
        """evaluate_experiments DataFrame contains var_5 and cvar_5 columns."""
        all_logs = {"algo_a": simple_logs}
        df = evaluate_experiments(all_logs, make_plots=False)
        assert "var_5" in df.columns
        assert "cvar_5" in df.columns


# ---------------------------------------------------------------------------
# Bootstrap confidence interval tests
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_basic_keys(self, simple_logs):
        """bootstrap_confidence_intervals returns expected keys."""
        result = bootstrap_confidence_intervals(
            {"test": simple_logs}, n_bootstrap=200, seed=0,
        )
        assert "test" in result
        for key in ("mean", "ci_lower", "ci_upper", "std"):
            assert key in result["test"]

    def test_ci_contains_mean(self, simple_logs):
        """The point estimate mean should lie within the CI."""
        result = bootstrap_confidence_intervals(
            {"test": simple_logs}, n_bootstrap=2000, seed=42,
        )
        ci = result["test"]
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_deterministic_ci(self, deterministic_logs):
        """With identical episodes the CI should collapse to a single point."""
        result = bootstrap_confidence_intervals(
            {"det": deterministic_logs}, n_bootstrap=500, seed=0,
        )
        ci = result["det"]
        assert ci["ci_lower"] == pytest.approx(10.0)
        assert ci["ci_upper"] == pytest.approx(10.0)
        assert ci["std"] == pytest.approx(0.0, abs=1e-12)

    def test_higher_confidence_wider(self, simple_logs):
        """A 99% CI should be at least as wide as a 90% CI."""
        ci90 = bootstrap_confidence_intervals(
            {"a": simple_logs}, n_bootstrap=2000, confidence_level=0.90, seed=1,
        )["a"]
        ci99 = bootstrap_confidence_intervals(
            {"a": simple_logs}, n_bootstrap=2000, confidence_level=0.99, seed=1,
        )["a"]
        width90 = ci90["ci_upper"] - ci90["ci_lower"]
        width99 = ci99["ci_upper"] - ci99["ci_lower"]
        assert width99 >= width90 - 1e-9

    def test_empty_logs(self):
        """Empty experiment should produce zero CI."""
        result = bootstrap_confidence_intervals({"empty": []}, n_bootstrap=100)
        ci = result["empty"]
        assert ci["mean"] == 0.0

    def test_custom_metric(self, simple_logs):
        """A custom metric_fn should be used for bootstrap resampling."""
        # Metric: number of episodes (constant under resampling with replacement)
        def ep_count(logs):
            return float(len(logs))
        result = bootstrap_confidence_intervals(
            {"a": simple_logs}, metric_fn=ep_count, n_bootstrap=100, seed=0,
        )
        # Resampled sets always have the same length
        assert result["a"]["mean"] == pytest.approx(len(simple_logs))

    def test_multiple_experiments(self, simple_logs, deterministic_logs):
        """Multiple experiments are handled independently."""
        result = bootstrap_confidence_intervals(
            {"varied": simple_logs, "det": deterministic_logs},
            n_bootstrap=500, seed=0,
        )
        assert "varied" in result
        assert "det" in result
        # Deterministic CI should be tighter
        assert result["det"]["std"] < result["varied"]["std"] + 1e-9


# ---------------------------------------------------------------------------
# Paired comparison tests
# ---------------------------------------------------------------------------

class TestPairedComparison:
    def test_basic_keys(self, paired_logs):
        logs_a, logs_b = paired_logs
        result = paired_comparison(logs_a, logs_b)
        for key in ("mean_diff", "median_diff", "std_diff", "wilcoxon_stat", "wilcoxon_p"):
            assert key in result

    def test_shift_detected(self, paired_logs):
        """logs_b = logs_a + 1, so mean_diff should be approximately -1."""
        logs_a, logs_b = paired_logs
        result = paired_comparison(logs_a, logs_b)
        assert result["mean_diff"] == pytest.approx(-1.0, abs=0.01)

    def test_significant_p(self, paired_logs):
        """A consistent +1 shift should yield a small p-value."""
        logs_a, logs_b = paired_logs
        result = paired_comparison(logs_a, logs_b)
        if not math.isnan(result["wilcoxon_p"]):
            assert result["wilcoxon_p"] < 0.05

    def test_identical_no_diff(self, deterministic_logs):
        """Comparing identical logs should produce zero mean_diff."""
        result = paired_comparison(deterministic_logs, deterministic_logs)
        assert result["mean_diff"] == pytest.approx(0.0)

    def test_empty_logs(self):
        """Empty inputs should return zero diff and NaN p-value."""
        result = paired_comparison([], [])
        assert result["mean_diff"] == 0.0
        assert math.isnan(result["wilcoxon_p"])

    def test_mismatched_length(self):
        """Unequal lists should use the min length without error."""
        a = [_make_episode([5.0]) for _ in range(10)]
        b = [_make_episode([3.0]) for _ in range(7)]
        result = paired_comparison(a, b)
        assert result["mean_diff"] == pytest.approx(2.0)

    def test_custom_metric(self, paired_logs):
        """A custom per-episode metric should be used for comparison."""
        logs_a, logs_b = paired_logs
        # Metric that always returns 0 → no difference
        result = paired_comparison(logs_a, logs_b, metric_fn=lambda df: 0.0)
        assert result["mean_diff"] == pytest.approx(0.0)
