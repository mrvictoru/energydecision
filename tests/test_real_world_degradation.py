"""
Tests for the RealWorldBESSDegradationModel and its integration into
AEMOBatteryTradingEnv ('real_world' degradation mode).

The 'real_world' model (doi:10.3390/batteries11110392) differs from the
Muenzel et al. (2015) model used in 'rainflow' mode in three key ways:

1. Calendar aging — accounted for every timestep (absent in Muenzel).
2. Arrhenius temperature dependency — physically grounded (vs. empirical cubic).
3. Power-law DoD / C-rate for cycle aging — compact & chemistry-specific.
"""

import sys
import os
import math
import pytest
import numpy as np
import polars as pl
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from batterydeg import RealWorldBESSDegradationModel, BESS_CHEMISTRY_PRESETS
from AEMOBatteryEnv import AEMOBatteryTradingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aemo_test_data():
    """Synthetic AEMO market data DataFrame (200 steps)."""
    np.random.seed(42)
    n = 200
    ts = [datetime(2024, 6, 1) + timedelta(minutes=30 * i) for i in range(n)]
    return pl.DataFrame({
        'SETTLEMENTDATE': ts,
        'Time': ts,
        'RRP': np.random.uniform(20, 100, n),
        'TOTALDEMAND': np.random.uniform(5000, 8000, n),
        'FCAS_RAISEREG': np.random.uniform(5, 20, n),
        'FCAS_LOWERREG': np.random.uniform(5, 20, n),
        'FCAS_RAISE6SEC': np.random.uniform(10, 30, n),
        'FCAS_LOWER6SEC': np.random.uniform(10, 30, n),
        'FCAS_RAISE60SEC': np.random.uniform(8, 25, n),
        'FCAS_LOWER60SEC': np.random.uniform(8, 25, n),
        'FCAS_RAISE5MIN': np.random.uniform(7, 22, n),
        'FCAS_LOWER5MIN': np.random.uniform(7, 22, n),
        'GEN_solar': np.random.uniform(0, 500, n),
        'GEN_wind': np.random.uniform(0, 300, n),
        'hour_sin': np.sin(2 * np.pi * np.arange(n) / 48),
        'hour_cos': np.cos(2 * np.pi * np.arange(n) / 48),
        'day_sin': np.zeros(n),
        'day_cos': np.ones(n),
        'is_peak': np.ones(n),
        'RRP_normalized': np.random.uniform(0, 1, n),
        'DEMAND_normalized': np.random.uniform(0, 1, n),
        'FCAS_RAISEREG_normalized': np.random.uniform(0, 1, n),
        'FCAS_LOWERREG_normalized': np.random.uniform(0, 1, n),
        'FCAS_RAISE6SEC_normalized': np.random.uniform(0, 1, n),
        'FCAS_LOWER6SEC_normalized': np.random.uniform(0, 1, n),
        'FCAS_RAISE60SEC_normalized': np.random.uniform(0, 1, n),
        'FCAS_LOWER60SEC_normalized': np.random.uniform(0, 1, n),
        'FCAS_RAISE5MIN_normalized': np.random.uniform(0, 1, n),
        'FCAS_LOWER5MIN_normalized': np.random.uniform(0, 1, n),
        'GEN_solar_pct': np.random.uniform(0, 0.3, n),
        'GEN_wind_pct': np.random.uniform(0, 0.2, n),
    })


@pytest.fixture
def real_world_env_nmc(aemo_test_data):
    env = AEMOBatteryTradingEnv(
        aemo_data=aemo_test_data,
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_battery_level=5.0,
        max_step=150,
        battery_life_cost=1_000_000.0,
        degradation_mode='real_world',
        degradation_chemistry='NMC',
        degradation_temperature=25.0,
    )
    env.reset(seed=0)
    return env


@pytest.fixture
def real_world_env_lfp(aemo_test_data):
    env = AEMOBatteryTradingEnv(
        aemo_data=aemo_test_data,
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_battery_level=5.0,
        max_step=150,
        battery_life_cost=1_000_000.0,
        degradation_mode='real_world',
        degradation_chemistry='LFP',
        degradation_temperature=25.0,
    )
    env.reset(seed=0)
    return env


# ===========================================================================
# Unit tests — RealWorldBESSDegradationModel
# ===========================================================================

class TestRealWorldBESSDegradationModel:
    """Unit tests for the standalone degradation model class."""

    def test_nmc_preset_instantiation(self):
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        assert model.chemistry == 'NMC'
        assert model.Ea_cal == BESS_CHEMISTRY_PRESETS['NMC']['Ea_cal']
        assert model.Ea_cyc == BESS_CHEMISTRY_PRESETS['NMC']['Ea_cyc']

    def test_lfp_preset_instantiation(self):
        model = RealWorldBESSDegradationModel(chemistry='lfp')  # lowercase
        assert model.chemistry == 'LFP'

    def test_invalid_chemistry_raises(self):
        with pytest.raises(ValueError, match="Unknown chemistry"):
            RealWorldBESSDegradationModel(chemistry='UNKNOWN')

    def test_custom_parameters_override_preset(self):
        model = RealWorldBESSDegradationModel(chemistry='NMC', k_cal_rate=1e-5)
        assert model.k_cal_rate == 1e-5

    # --- Calendar aging ---

    def test_calendar_aging_positive_per_step(self):
        """Calendar aging must be strictly positive for any active timestep."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        val = model.calendar_aging_per_step(T_celsius=25.0, soc_frac=0.5, dt_hours=0.5)
        assert val > 0.0

    def test_calendar_aging_zero_dt(self):
        """Zero duration step → zero calendar aging."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        val = model.calendar_aging_per_step(T_celsius=25.0, soc_frac=0.5, dt_hours=0.0)
        assert val == pytest.approx(0.0)

    def test_calendar_aging_increases_with_temperature(self):
        """Higher temperature → higher calendar aging (Arrhenius)."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        low = model.calendar_aging_per_step(T_celsius=15.0, soc_frac=0.5, dt_hours=1.0)
        high = model.calendar_aging_per_step(T_celsius=40.0, soc_frac=0.5, dt_hours=1.0)
        assert high > low

    def test_calendar_aging_increases_with_soc_nmc(self):
        """Higher SOC → more calendar aging for NMC (k_soc > 0)."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        low_soc = model.calendar_aging_per_step(T_celsius=25.0, soc_frac=0.1, dt_hours=1.0)
        high_soc = model.calendar_aging_per_step(T_celsius=25.0, soc_frac=0.9, dt_hours=1.0)
        assert high_soc > low_soc

    def test_calendar_aging_at_reference_conditions(self):
        """At 25 °C, 50 % SOC the rate equals k_cal_rate exactly."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        val = model.calendar_aging_per_step(T_celsius=25.0, soc_frac=0.5, dt_hours=1.0)
        # SOC stress at 0.5 → 1.0, Arrhenius at T_ref → 1.0
        assert val == pytest.approx(model.k_cal_rate, rel=1e-6)

    # --- Cycle aging ---

    def test_cycle_aging_positive_for_nonzero_dod(self):
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        val = model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=80.0, c_rate=0.5)
        assert val > 0.0

    def test_cycle_aging_zero_for_zero_dod(self):
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        val = model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=0.0, c_rate=0.5)
        assert val == pytest.approx(0.0)

    def test_cycle_aging_increases_with_dod(self):
        """Deeper cycling → more cycle aging."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        shallow = model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=20.0, c_rate=0.5)
        deep = model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=90.0, c_rate=0.5)
        assert deep > shallow

    def test_cycle_aging_increases_with_temperature(self):
        """Higher temperature → higher cycle aging (Arrhenius)."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        low = model.cycle_aging_per_cycle(T_celsius=15.0, dod_pct=80.0, c_rate=0.5)
        high = model.cycle_aging_per_cycle(T_celsius=45.0, dod_pct=80.0, c_rate=0.5)
        assert high > low

    def test_cycle_aging_increases_with_c_rate(self):
        """Higher C-rate → more cycle aging."""
        model = RealWorldBESSDegradationModel(chemistry='NMC')
        slow = model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=80.0, c_rate=0.2)
        fast = model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=80.0, c_rate=2.0)
        assert fast > slow

    def test_nmc_calendar_aging_greater_than_lfp(self):
        """NMC should age faster calendrically than LFP at equal conditions."""
        nmc = RealWorldBESSDegradationModel(chemistry='NMC')
        lfp = RealWorldBESSDegradationModel(chemistry='LFP')
        val_nmc = nmc.calendar_aging_per_step(T_celsius=25.0, soc_frac=0.5, dt_hours=1.0)
        val_lfp = lfp.calendar_aging_per_step(T_celsius=25.0, soc_frac=0.5, dt_hours=1.0)
        assert val_nmc > val_lfp

# ===========================================================================
# Integration tests — AEMOBatteryTradingEnv with 'real_world' mode
# ===========================================================================

class TestAEMORealWorldDegradation:
    """Tests for 'real_world' degradation mode in AEMOBatteryTradingEnv."""

    def test_real_world_mode_initialised_correctly(self, real_world_env_nmc):
        env = real_world_env_nmc
        assert env.degradation_mode == 'real_world'
        assert isinstance(env.degradation_model, RealWorldBESSDegradationModel)
        assert env.degradation_model.chemistry == 'NMC'

    def test_idle_battery_still_has_calendar_aging(self, real_world_env_nmc):
        """Idle battery (zero action) must incur calendar aging every step."""
        env = real_world_env_nmc
        _, _, _, _, info = env.step(np.array([0.0]))
        # Calendar degradation must be positive even when idle
        assert info['calendar_degradation'] > 0.0
        assert info['step_degradation'] > 0.0

    def test_cycling_increases_degradation_over_idle(self, aemo_test_data):
        """Cycling should produce more total degradation than idling."""
        def run_env(actions):
            env = AEMOBatteryTradingEnv(
                aemo_data=aemo_test_data,
                battery_capacity=10.0,
                max_battery_flow=5.0,
                max_step=100,
                degradation_mode='real_world',
                degradation_chemistry='NMC',
            )
            env.reset(seed=7)
            total_deg = 0.0
            for a in actions:
                _, _, t, tr, info = env.step(np.array([a]))
                total_deg = info['total_degradation']
                if t or tr:
                    break
            return total_deg

        idle_deg = run_env([0.0] * 50)
        cycling_deg = run_env([1.0, -1.0] * 25)
        assert cycling_deg > idle_deg

    def test_calendar_and_cycle_degradation_tracked_separately(self, real_world_env_nmc):
        """calendar_degradation and cycle_degradation must be reported in info."""
        env = real_world_env_nmc
        for _ in range(10):
            _, _, t, tr, info = env.step(np.array([1.0]))
            if t or tr:
                break
        for _ in range(10):
            _, _, t, tr, info = env.step(np.array([-1.0]))
            if t or tr:
                break

        assert 'calendar_degradation' in info
        assert 'cycle_degradation' in info
        assert info['calendar_degradation'] >= 0.0
        assert info['cycle_degradation'] >= 0.0

    def test_total_degradation_equals_calendar_plus_cycle(self, aemo_test_data):
        """total_degradation should equal calendar + cycle (before capping)."""
        env = AEMOBatteryTradingEnv(
            aemo_data=aemo_test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            max_step=60,
            degradation_mode='real_world',
            degradation_chemistry='NMC',
        )
        env.reset(seed=3)
        last_info = None
        for _ in range(40):
            _, _, t, tr, info = env.step(np.array([0.5]))
            last_info = info
            if t or tr:
                break

        assert last_info is not None
        expected = last_info['calendar_degradation'] + last_info['cycle_degradation']
        assert last_info['total_degradation'] == pytest.approx(expected, rel=1e-5)

    def test_capacity_fade_with_real_world_mode(self, real_world_env_nmc):
        """Repeated cycling must reduce battery capacity."""
        env = real_world_env_nmc
        initial_cap = env.initial_battery_capacity
        for _ in range(80):
            _, _, t, tr, _ = env.step(np.array([1.0]))
            if t or tr:
                break
        for _ in range(80):
            _, _, t, tr, _ = env.step(np.array([-1.0]))
            if t or tr:
                break
        assert env.battery_capacity < initial_cap

    def test_info_contains_expected_keys(self, real_world_env_nmc):
        """Step info dict must contain all required degradation fields."""
        _, _, _, _, info = real_world_env_nmc.step(np.array([0.3]))
        required = [
            'step_degradation', 'total_degradation', 'capacity_mwh',
            'rainflow_cumulative_deg', 'rainflow_num_cycles',
            'calendar_degradation', 'cycle_degradation', 'degradation_cost',
        ]
        for key in required:
            assert key in info, f"Missing key: {key}"

    def test_reset_clears_degradation_state(self, aemo_test_data):
        """Reset must zero all degradation accumulators."""
        env = AEMOBatteryTradingEnv(
            aemo_data=aemo_test_data,
            battery_capacity=10.0,
            max_battery_flow=5.0,
            max_step=50,
            degradation_mode='real_world',
        )
        env.reset(seed=0)
        for _ in range(20):
            _, _, t, tr, _ = env.step(np.array([0.8]))
            if t or tr:
                break

        env.reset(seed=1)
        assert env.total_degradation == 0.0
        assert env._calendar_degradation == 0.0
        assert env._cycle_degradation == 0.0
        assert env._rainflow_deg_cumulative == 0.0
        assert env._rainflow_num_cycles == 0

    def test_nmc_degrades_faster_than_lfp(self, aemo_test_data):
        """NMC should accumulate more degradation than LFP under deep cycling.

        At high DoD NMC's larger k_cyc dominates; at very shallow DoD the
        difference in alpha_dod exponents flips the ordering, so we use
        near-full-capacity cycling to keep DoD close to 100 %. The env defaults
        to 5-min steps (step_duration=0.0833 h), where a ±1.0 action at 5 MW /
        5 MWh moves only ~8 % SOC (shallow) — so we explicitly set a 1 h step so
        each ±1.0 action swings the full battery (~100 % DoD).
        """
        actions = [1.0, -1.0] * 25

        def run(chemistry):
            env = AEMOBatteryTradingEnv(
                aemo_data=aemo_test_data,
                battery_capacity=5.0,    # small → ±5 MWh per 1 h step ≈ 100 % DoD
                max_battery_flow=5.0,
                init_battery_level=0.0,  # start empty for full-swing cycling
                max_step=100,
                step_duration=1.0,       # 1 h steps → full-capacity swing per ±1.0 action
                degradation_mode='real_world',
                degradation_chemistry=chemistry,
            )
            env.reset(seed=5)
            last_deg = 0.0
            for a in actions:
                _, _, t, tr, info = env.step(np.array([a]))
                last_deg = info['total_degradation']
                if t or tr:
                    break
            return last_deg

        nmc_deg = run('NMC')
        lfp_deg = run('LFP')
        assert nmc_deg > lfp_deg, (
            f"NMC ({nmc_deg:.6f}) should exceed LFP ({lfp_deg:.6f}) under deep cycling"
        )

    def test_higher_temperature_increases_degradation(self, aemo_test_data):
        """Arrhenius: higher temperature must produce more total degradation."""
        actions = [0.0] * 30  # idle: only calendar aging varies

        def run(temp):
            env = AEMOBatteryTradingEnv(
                aemo_data=aemo_test_data,
                battery_capacity=10.0,
                max_battery_flow=5.0,
                max_step=60,
                degradation_mode='real_world',
                degradation_chemistry='NMC',
                degradation_temperature=temp,
            )
            env.reset(seed=9)
            last_deg = 0.0
            for a in actions:
                _, _, t, tr, info = env.step(np.array([a]))
                last_deg = info['total_degradation']
                if t or tr:
                    break
            return last_deg

        cool_deg = run(15.0)
        hot_deg = run(40.0)
        assert hot_deg > cool_deg


class TestAEMODegradationModes:
    """Verify all three degradation modes coexist and produce distinct results."""

    def test_real_world_differs_from_rainflow(self, aemo_test_data):
        actions = [np.array([v]) for v in [1.0, -1.0, 0.5, -0.5] * 10]

        def run(mode):
            env = AEMOBatteryTradingEnv(
                aemo_data=aemo_test_data,
                battery_capacity=10.0,
                max_battery_flow=5.0,
                max_step=80,
                degradation_mode=mode,
            )
            env.reset(seed=0)
            last_deg = 0.0
            for a in actions:
                _, _, t, tr, info = env.step(a)
                last_deg = info['total_degradation']
                if t or tr:
                    break
            return last_deg

        deg_rw = run('real_world')
        deg_rf = run('rainflow')
        # The two models use fundamentally different formulations;
        # results must be non-negative and differ from each other.
        assert deg_rw >= 0.0
        assert deg_rf >= 0.0
        assert deg_rw != pytest.approx(deg_rf)

    def test_all_three_modes_accepted(self, aemo_test_data):
        """All three mode strings must be accepted without error."""
        for mode in ('simple', 'rainflow', 'real_world'):
            env = AEMOBatteryTradingEnv(
                aemo_data=aemo_test_data,
                battery_capacity=10.0,
                max_battery_flow=5.0,
                degradation_mode=mode,
            )
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.array([0.5]))
            assert info['total_degradation'] >= 0.0
