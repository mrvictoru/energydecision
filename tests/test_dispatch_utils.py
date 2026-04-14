"""Tests for dispatch_utils.py — high-level dispatch replay helpers."""

import sys
import warnings
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, "src")

from dispatch_utils import (
    list_dispatch_candidates,
    resolve_dispatch_selection,
    run_dispatch_replay,
    scan_duid_availability,
    show_dispatch_table,
)


# ---------------------------------------------------------------------------
# show_dispatch_table
# ---------------------------------------------------------------------------

def test_show_dispatch_table_prints_subset(capsys):
    df = pl.DataFrame({
        "DUID": ["HPRG1", "LBBG1"],
        "Region": ["SA1", "SA1"],
        "Extra": [1, 2],
    })
    show_dispatch_table(df, ["DUID", "Region", "MISSING"], label="My label")
    out = capsys.readouterr().out
    assert "My label" in out
    assert "HPRG1" in out
    assert "MISSING" not in out   # silently skipped


def test_show_dispatch_table_empty(capsys):
    show_dispatch_table(pl.DataFrame(), ["DUID"], label="Empty test")
    out = capsys.readouterr().out
    assert "No rows" in out


def test_show_dispatch_table_limit(capsys):
    df = pl.DataFrame({"DUID": [f"D{i}" for i in range(10)]})
    show_dispatch_table(df, ["DUID"], limit=3)
    out = capsys.readouterr().out
    assert "D0" in out
    assert "D9" not in out   # limited to 3 rows


# ---------------------------------------------------------------------------
# resolve_dispatch_selection
# ---------------------------------------------------------------------------

def _make_battery_tables(*, has_paired=False, nonzero=500):
    static_cols = {
        "DUID": ["HPRG1", "LBBG1"],
        "Region": ["SA1", "SA1"],
        "DispatchType": ["Bidirectional Unit", "Bidirectional Unit"],
        "StorageCapacityMWh": [193.5, 52.0],
        "RegisteredCapacityMW": [150.0, 25.0],
    }
    active_cols = {
        **static_cols,
        "NonZeroIntervalCount": [nonzero, 100],
        "DispatchIntervalCount": [2016, 2016],
        "MaxEnergyMW": [140.0, 20.0],
        "PairedGenDUID": [None, None],
        "PairedLoadDUID": [None, None],
    }
    if has_paired:
        active_cols["PairedGenDUID"][0] = "KEPBG1"
        active_cols["PairedLoadDUID"][0] = "KEPBL1"

    return pl.DataFrame(static_cols), pl.DataFrame(active_cols)


def test_resolve_by_duid():
    battery_units, active = _make_battery_tables()
    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active,
        selected_duid="LBBG1",
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_soc=5.0,
        apply_unit_sizing=True,
    )
    assert sel["duid"] == "LBBG1"
    assert sel["battery_capacity"] == 52.0
    assert sel["max_battery_flow"] == 25.0


def test_resolve_by_index():
    battery_units, active = _make_battery_tables()
    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active,
        selected_duid=None,
        selected_index=0,
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_soc=5.0,
        apply_unit_sizing=True,
    )
    assert sel["duid"] == "HPRG1"


def test_resolve_apply_unit_sizing_false():
    battery_units, active = _make_battery_tables()
    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active,
        selected_duid="HPRG1",
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_soc=5.0,
        apply_unit_sizing=False,
    )
    assert sel["battery_capacity"] == 10.0
    assert sel["max_battery_flow"] == 5.0


def test_resolve_init_soc_ratio_applies_after_unit_sizing():
    battery_units, active = _make_battery_tables()
    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active,
        selected_duid="HPRG1",
        battery_capacity=10.0,
        max_battery_flow=5.0,
        init_soc=5.0,
        init_soc_ratio=0.25,
        apply_unit_sizing=True,
    )

    assert sel["battery_capacity"] == 193.5
    assert sel["init_battery_level"] == pytest.approx(193.5 * 0.25)


def test_resolve_paired_duid():
    battery_units, active = _make_battery_tables(has_paired=True)
    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active,
        selected_duid="HPRG1",
    )
    assert sel["dispatch_duid_gen"] == "KEPBG1"
    assert sel["dispatch_duid_load"] == "KEPBL1"


def test_resolve_warns_zero_nonzero_count():
    battery_units, active = _make_battery_tables(nonzero=0)
    with pytest.warns(UserWarning, match="NonZeroIntervalCount=0"):
        resolve_dispatch_selection(
            battery_units=battery_units,
            active_battery_units=active,
            selected_duid="HPRG1",
        )


def test_resolve_falls_back_to_static_when_active_empty():
    battery_units, _ = _make_battery_tables()
    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=pl.DataFrame(),
        selected_index=1,
        apply_unit_sizing=True,
    )
    assert sel["duid"] == "LBBG1"


def test_list_dispatch_candidates_keeps_static_rows_when_dispatch_empty(monkeypatch):
    import aemo_data as _aemo

    def fake_resolve(name_or_duid, start_date, end_date):
        assert name_or_duid == "hornsdale"
        return {
            "found": True,
            "key": "hornsdale",
            "station_name": "Hornsdale Power Reserve",
            "region": "SA1",
            "active_duids": [
                {
                    "duid": "HPR1",
                    "type": "bidirectional",
                    "valid_from": datetime(2022, 10, 1),
                    "valid_until": None,
                }
            ],
            "all_duids_in_range": ["HPR1"],
            "gen_duid": None,
            "load_duid": None,
            "bidi_duid": "HPR1",
            "spans_transition": False,
            "transition_dates": [],
        }

    monkeypatch.setattr(_aemo, "resolve_battery_duids", fake_resolve)
    monkeypatch.setattr(
        _aemo,
        "fetch_aemo_unit_dispatch",
        lambda **kwargs: pl.DataFrame(
            schema={
                "SETTLEMENTDATE": pl.Datetime,
                "DUID": pl.Utf8,
                "TOTALCLEARED": pl.Float64,
                "RAISE6SEC": pl.Float64,
                "RAISE60SEC": pl.Float64,
                "RAISE5MIN": pl.Float64,
                "RAISEREG": pl.Float64,
                "LOWER6SEC": pl.Float64,
                "LOWER60SEC": pl.Float64,
                "LOWER5MIN": pl.Float64,
                "LOWERREG": pl.Float64,
            }
        ),
    )

    battery_units, active_units = list_dispatch_candidates(
        region="SA1",
        start_date=datetime(2022, 4, 1),
        end_date=datetime(2022, 4, 2),
        station_name="hornsdale",
    )

    assert battery_units.height > 0
    assert active_units.height == 0


def test_resolve_raises_when_no_units():
    with pytest.raises(ValueError, match="No battery DUIDs"):
        resolve_dispatch_selection(
            battery_units=pl.DataFrame(),
            active_battery_units=pl.DataFrame(),
        )


def test_resolve_generator_with_paired_load_sets_dispatch_duid_gen():
    """For an old-model battery the primary DUID is the generator (discharge).
    When PairedLoadDUID is set and PairedGenDUID is null, resolve_dispatch_selection
    must set dispatch_duid_gen = primary_duid so both sides are fetched/replayed.
    Regression: without this fix, dispatch_duid_gen was None → no discharge actions.
    """
    battery_units = pl.DataFrame({
        "DUID": ["HPRG1", "HPRL1"],
        "Region": ["SA1", "SA1"],
        "DispatchType": ["generator", "load"],
        "StorageCapacityMWh": [194.0, 194.0],
        "RegisteredCapacityMW": [150.0, 120.0],
    })
    active_battery_units = pl.DataFrame({
        "DUID": ["HPRG1", "HPRL1"],
        "Region": ["SA1", "SA1"],
        "DispatchType": ["generator", "load"],
        "StorageCapacityMWh": [194.0, 194.0],
        "RegisteredCapacityMW": [150.0, 120.0],
        "NonZeroIntervalCount": [1728, 1728],
        "DispatchIntervalCount": [1728, 1728],
        "MaxEnergyMW": [140.0, 120.0],
        "PairedGenDUID": [None, "HPRG1"],  # load → gen=HPRG1; gen → gen=None
        "PairedLoadDUID": ["HPRL1", None],  # gen → load=HPRL1; load → load=None
    })

    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active_battery_units,
        selected_duid="HPRG1",
        apply_unit_sizing=True,
    )

    # Primary DUID is the generator; it must be carried as dispatch_duid_gen
    assert sel["duid"] == "HPRG1"
    assert sel["dispatch_duid_gen"] == "HPRG1", (
        "dispatch_duid_gen must equal the primary DUID when it is a generator "
        "with a paired load DUID (so discharge data is fetched and replayed)"
    )
    assert sel["dispatch_duid_load"] == "HPRL1"
    # Capacity should come from the static table
    assert sel["battery_capacity"] == 194.0


def test_resolve_registry_capacity_fallback(monkeypatch):
    """When StorageCapacityMWh is None in the active table, resolve_dispatch_selection
    must fall back to the BATTERY_REGISTRY capacity_mwh value.
    Regression: without this fix, capacity defaulted to 10.0 (notebook default).
    """
    import aemo_data as _aemo

    # Patch resolve_battery_duids to return a registry hit with known capacity
    def fake_resolve(name, start_date, end_date):
        return {
            "found": True,
            "key": "hornsdale",
            "station_name": "Hornsdale Power Reserve",
            "region": "SA1",
            "active_duids": [
                {
                    "duid": "HPRG1",
                    "type": "generator",
                    "valid_from": datetime(2017, 12, 1),
                    "valid_until": datetime(2022, 10, 1),
                }
            ],
            "all_duids_in_range": ["HPRG1", "HPRL1"],
            "gen_duid": "HPRG1",
            "load_duid": "HPRL1",
            "bidi_duid": None,
            "spans_transition": False,
            "transition_dates": [],
        }

    monkeypatch.setattr(_aemo, "resolve_battery_duids", fake_resolve)

    # Patch the availability check so no NEMOSIS call is made
    monkeypatch.setattr(_aemo, "check_aemo_dispatch_availability", lambda **kw: {
        "has_data": True,
        "row_count": 1728,
        "unique_intervals": 1728,
        "expected_intervals": 1728,
        "coverage_ratio": 1.0,
        "first_settlement": datetime(2021, 1, 1),
        "last_settlement": datetime(2021, 1, 7),
    })

    # Active table has null capacity (as in the station-name path before the fix)
    active = pl.DataFrame({
        "DUID": ["HPRG1"],
        "Region": ["SA1"],
        "DispatchType": ["generator"],
        "StorageCapacityMWh": [None],
        "RegisteredCapacityMW": [None],
        "NonZeroIntervalCount": [1728],
        "DispatchIntervalCount": [1728],
        "MaxEnergyMW": [140.0],
        "PairedGenDUID": [None],
        "PairedLoadDUID": ["HPRL1"],
    }, schema_overrides={"StorageCapacityMWh": pl.Float64, "RegisteredCapacityMW": pl.Float64})

    sel = resolve_dispatch_selection(
        battery_units=active,
        active_battery_units=active,
        selected_duid="HPRG1",
        battery_capacity=10.0,  # notebook default that should be overridden
        max_battery_flow=5.0,
        init_soc=5.0,
        apply_unit_sizing=True,
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2021, 1, 7),
    )

    # Must have picked up 194.0 MWh and 150.0 MW from the BATTERY_REGISTRY
    assert sel["battery_capacity"] == 194.0, (
        f"Expected registry fallback capacity 194.0 MWh, got {sel['battery_capacity']}"
    )
    assert sel["max_battery_flow"] == 150.0, (
        f"Expected registry fallback max_flow 150.0 MW, got {sel['max_battery_flow']}"
    )


def test_resolve_bidi_duid_does_not_get_gen_load_supplement():
    """For a bidirectional DUID (e.g. HPR1), the registry supplement must NOT
    set dispatch_duid_gen / dispatch_duid_load from historical gen/load DUIDs.
    Regression: without this fix, HPR1 selected over 2019-2022 would get
    dispatch_duid_gen=HPRG1, dispatch_duid_load=HPRL1, causing run_dispatch_replay
    to fetch only the old gen/load pair and miss HPR1's post-Oct-2022 data.
    """
    battery_units = pl.DataFrame({
        "DUID": ["HPRG1", "HPRL1", "HPR1"],
        "Region": ["SA1", "SA1", "SA1"],
        "DispatchType": ["generator", "load", "bidirectional"],
        "StorageCapacityMWh": [194.0, 194.0, 194.0],
        "RegisteredCapacityMW": [150.0, 150.0, 150.0],
    })
    active_battery_units = battery_units.with_columns([
        pl.lit(5000).alias("NonZeroIntervalCount"),
        pl.lit(5000).alias("DispatchIntervalCount"),
        pl.lit(140.0).alias("MaxEnergyMW"),
        pl.lit(datetime(2019, 6, 1)).alias("FirstDispatchInterval"),
        pl.lit(datetime(2022, 12, 31)).alias("LastDispatchInterval"),
        pl.lit(None, dtype=pl.Utf8).alias("PairedGenDUID"),
        pl.lit(None, dtype=pl.Utf8).alias("PairedLoadDUID"),
    ])

    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active_battery_units,
        selected_duid="HPR1",
        start_date=datetime(2019, 6, 1),
        end_date=datetime(2022, 12, 31),
    )

    assert sel["duid"] == "HPR1"
    assert sel["dispatch_duid_gen"] is None, (
        "Bidirectional DUID must not receive a gen-DUID supplement from registry "
        f"(got dispatch_duid_gen={sel['dispatch_duid_gen']!r})"
    )
    assert sel["dispatch_duid_load"] is None, (
        "Bidirectional DUID must not receive a load-DUID supplement from registry "
        f"(got dispatch_duid_load={sel['dispatch_duid_load']!r})"
    )
    assert sel["spans_transition"] is True
    assert set(sel["all_dispatch_duids"]) == {"HPR1", "HPRG1", "HPRL1"}


def test_resolve_bidi_includes_all_duids_in_transition_selection():
    """spans_transition=True and all_dispatch_duids contains all period DUIDs."""
    battery_units = pl.DataFrame({
        "DUID": ["HPRG1", "HPRL1", "HPR1"],
        "Region": ["SA1"] * 3,
        "DispatchType": ["generator", "load", "bidirectional"],
        "StorageCapacityMWh": [194.0, 194.0, 194.0],
        "RegisteredCapacityMW": [150.0, 150.0, 150.0],
    })
    active = battery_units.with_columns([
        pl.lit(1).alias("NonZeroIntervalCount"),
        pl.lit(1).alias("DispatchIntervalCount"),
        pl.lit(10.0).alias("MaxEnergyMW"),
        pl.lit(datetime(2019, 6, 1)).alias("FirstDispatchInterval"),
        pl.lit(datetime(2022, 12, 31)).alias("LastDispatchInterval"),
        pl.lit(None, dtype=pl.Utf8).alias("PairedGenDUID"),
        pl.lit(None, dtype=pl.Utf8).alias("PairedLoadDUID"),
    ])
    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active,
        selected_duid="HPR1",
        start_date=datetime(2019, 6, 1),
        end_date=datetime(2022, 12, 31),
    )
    assert sel["spans_transition"] is True
    assert len(sel["transition_dates"]) >= 1
    assert "HPRG1" in sel["all_dispatch_duids"]
    assert "HPR1" in sel["all_dispatch_duids"]
    assert sel["_registry_gen_duid"] == "HPRG1"
    assert sel["_registry_bidi_duid"] == "HPR1"


def test_resolve_selection_returns_region_metadata():
    battery_units, active = _make_battery_tables()

    sel = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active,
        selected_duid="HPRG1",
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2021, 1, 7),
    )

    assert sel["region"] == "SA1"
    assert sel["station_name"] == "Hornsdale Power Reserve"
    assert sel["station_key"] == "hornsdale"


def test_merge_transition_dispatch_gen_load_to_bidi():
    """_merge_transition_dispatch must produce a unified generator-convention
    TOTALCLEARED column: GEN-LOAD for the pre-transition period, and the
    bidirectional TOTALCLEARED for the post-transition period.
    """
    from dispatch_utils import _merge_transition_dispatch

    transition = datetime(2022, 10, 1)

    raw = pl.DataFrame({
        "SETTLEMENTDATE": [
            datetime(2021, 1, 1), datetime(2021, 1, 1),   # gen+load pre
            datetime(2022, 11, 1),                          # bidi post
        ],
        "DUID":          ["HPRG1", "HPRL1", "HPR1"],
        "TOTALCLEARED":  [100.0, 30.0, 80.0],
        "RAISEREG":      [5.0, 2.0, 8.0],
        "LOWERREG":      [0.0, 0.0, 4.0],
    })

    sel = {
        "duid": "HPR1",
        "_registry_gen_duid": "HPRG1",
        "_registry_load_duid": "HPRL1",
        "_registry_bidi_duid": "HPR1",
    }
    merged = _merge_transition_dispatch(raw, sel, [transition])

    # Pre-transition row: TOTALCLEARED = GEN - LOAD = 100 - 30 = 70
    pre_row = merged.filter(pl.col("SETTLEMENTDATE") == datetime(2021, 1, 1))
    assert pre_row.height == 1
    assert abs(pre_row["TOTALCLEARED"][0] - 70.0) < 1e-6

    # Post-transition row: TOTALCLEARED = bidi TOTALCLEARED = 80
    post_row = merged.filter(pl.col("SETTLEMENTDATE") == datetime(2022, 11, 1))
    assert post_row.height == 1
    assert abs(post_row["TOTALCLEARED"][0] - 80.0) < 1e-6

    # All rows should use the synthetic DUID name
    assert all("TRANSITION" in d for d in merged["DUID"].to_list())


def test_resolve_skips_availability_check_when_active_table_has_data(monkeypatch):
    """When active_battery_units already has NonZeroIntervalCount > 0,
    resolve_dispatch_selection must derive availability from the table and NOT call
    check_aemo_dispatch_availability (which would trigger an expensive NEMOSIS download).
    """
    import aemo_data as _aemo

    calls = []

    def _spy_avail(**kw):
        calls.append(kw)
        return {"has_data": True, "row_count": 1, "unique_intervals": 1,
                "expected_intervals": 1, "coverage_ratio": 1.0,
                "first_settlement": None, "last_settlement": None}

    monkeypatch.setattr(_aemo, "check_aemo_dispatch_availability", _spy_avail)

    active = pl.DataFrame({
        "DUID": ["HPRG1"],
        "Region": ["SA1"],
        "DispatchType": ["generator"],
        "StorageCapacityMWh": [194.0],
        "RegisteredCapacityMW": [150.0],
        "NonZeroIntervalCount": [2016],       # non-zero → should skip download
        "DispatchIntervalCount": [2016],
        "MaxEnergyMW": [140.0],
        "FirstDispatchInterval": [datetime(2021, 1, 1)],
        "LastDispatchInterval": [datetime(2021, 1, 7)],
        "PairedGenDUID": [None],
        "PairedLoadDUID": ["HPRL1"],
    })

    sel = resolve_dispatch_selection(
        battery_units=active,
        active_battery_units=active,
        selected_duid="HPRG1",
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2021, 1, 7),
    )

    assert sel["availability"]["has_data"] is True
    assert calls == [], (
        "check_aemo_dispatch_availability must NOT be called when the active "
        "table already confirms non-zero dispatch data"
    )




def test_scan_duid_availability_static_only(monkeypatch):
    """scan_duid_availability with no date window returns static rows + InDispatchLoad column."""
    # Patch get_available_battery_units inside the dispatch_utils module
    import dispatch_utils

    def _fake_get_available(cache_dir, region, refresh):
        data = {
            "SA1": pl.DataFrame({
                "DUID": ["HPRG1", "LBBG1"],
                "Region": ["SA1", "SA1"],
                "StorageCapacityMWh": [193.5, 52.0],
            }),
            "QLD1": pl.DataFrame({
                "DUID": ["TARBESS1"],
                "Region": ["QLD1"],
                "StorageCapacityMWh": [600.0],
            }),
        }
        return data.get(region, pl.DataFrame())

    import aemo_data as _aemo
    monkeypatch.setattr(_aemo, "get_available_battery_units", _fake_get_available)

    result = scan_duid_availability(
        regions=["SA1", "QLD1"],
        start_date=None,
        end_date=None,
        cache_dir="data/aemo",
    )
    assert result.height == 3  # 2 SA1 + 1 QLD1
    assert "InDispatchLoad" in result.columns


# ---------------------------------------------------------------------------
# run_dispatch_replay — unit test with fully mocked env + agent
# ---------------------------------------------------------------------------

def _make_mock_env(processed_data):
    """Create a minimal mock AEMOBatteryTradingEnv."""
    env = SimpleNamespace(
        aemo_data=processed_data,
        step_duration=0.5,
        max_battery_flow=5.0,
        battery_capacity=10.0,
        action_mode="multi_market",
        current_step=0,
        episode_start_idx=0,
    )
    return env


def test_run_dispatch_replay_returns_logs(monkeypatch, tmp_path):
    """run_dispatch_replay must return correctly shaped log lists and save parquet."""
    import dispatch_utils as du
    import aemo_data as _aemo

    # Synthetic processed_data (two timesteps)
    processed_data = pl.DataFrame({
        "SETTLEMENTDATE": [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 30)],
        "RRP": [50.0, 60.0],
    })

    # Fake dispatch data
    fake_dispatch = pl.DataFrame({
        "SETTLEMENTDATE": [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 30)],
        "DUID": ["HPRG1", "HPRG1"],
        "TOTALCLEARED": [80.0, -40.0],
        "RAISEREG": [10.0, 10.0],
        "LOWERREG": [0.0, 0.0],
    })

    # Patch fetch_aemo_unit_dispatch to return fake data
    monkeypatch.setattr(_aemo, "fetch_aemo_unit_dispatch", lambda **kw: fake_dispatch)

    # Patch AEMOBatteryTradingEnv and run_single
    fake_ep_log = pl.DataFrame({
        "step": [0, 1],
        "action": [[0.5, 0.1, 0.0], [0.3, 0.0, 0.0]],
        "reward": [10.0, -5.0],
    })
    fake_inc_log = pl.DataFrame()

    import AEMOBatteryEnv as _env_mod
    import decision as _dec

    class FakeEnv:
        def __init__(self, **kw):
            self.aemo_data = processed_data
            self.step_duration = kw.get("step_duration", 0.5)
            self.max_battery_flow = kw.get("max_battery_flow", 5.0)
            self.battery_capacity = kw.get("battery_capacity", 10.0)
            self.action_mode = kw.get("action_mode", "multi_market")

    monkeypatch.setattr(_env_mod, "AEMOBatteryTradingEnv", FakeEnv)
    monkeypatch.setattr(_dec, "run_single", lambda cls, env, agent_kwargs=None, **kw: (fake_ep_log, fake_inc_log))

    selection = {
        "duid": "HPRG1",
        "dispatch_type": "Bidirectional Unit",
        "dispatch_duid_gen": None,
        "dispatch_duid_load": None,
        "battery_capacity": 193.5,
        "max_battery_flow": 150.0,
        "init_battery_level": 96.75,
        "availability": None,
    }

    ep_logs, inc_logs, all_logs = du.run_dispatch_replay(
        processed_data=processed_data,
        selection=selection,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 7),
        region="SA1",
        cache_dir="data/aemo",
        num_episodes=2,
        output_dir=str(tmp_path),
        run_tag="test",
    )

    assert len(ep_logs) == 2
    assert len(inc_logs) == 2
    assert all_logs.height == 4  # 2 episodes × 2 steps
    assert "episode_id" in all_logs.columns

    # Check parquet saved
    assert (tmp_path / "test_dispatch_logs.parquet").exists()


def test_run_dispatch_replay_historical_duids_not_filtered_by_region(monkeypatch, tmp_path):
    """run_dispatch_replay must use duids= (not region=) for gen/load pairs.

    Regression: historical DUIDs (e.g. HPRG1/HPRL1 pre-2022) are absent from
    the current static generator table, so fetching with region= silently
    excluded them, causing "No dispatch data returned for DUID".  The fix is to
    pass duids=[gen, load] directly.
    """
    import dispatch_utils as du
    import aemo_data as _aemo

    processed_data = pl.DataFrame({
        "SETTLEMENTDATE": [datetime(2021, 1, 1, 0, 0), datetime(2021, 1, 1, 0, 30)],
        "RRP": [50.0, 60.0],
    })

    fake_dispatch = pl.DataFrame({
        "SETTLEMENTDATE": [datetime(2021, 1, 1, 0, 0), datetime(2021, 1, 1, 0, 30),
                           datetime(2021, 1, 1, 0, 0), datetime(2021, 1, 1, 0, 30)],
        "DUID": ["HPRG1", "HPRG1", "HPRL1", "HPRL1"],
        "TOTALCLEARED": [80.0, 80.0, -40.0, -40.0],
    })

    # Track what kwargs were passed to fetch_aemo_unit_dispatch
    call_kwargs: dict = {}

    def fake_fetch(**kw):
        call_kwargs.update(kw)
        return fake_dispatch

    monkeypatch.setattr(_aemo, "fetch_aemo_unit_dispatch", fake_fetch)

    import AEMOBatteryEnv as _env_mod
    import decision as _dec

    fake_ep_log = pl.DataFrame({"step": [0], "action": [[0.5]], "reward": [1.0]})
    fake_inc_log = pl.DataFrame()

    class FakeEnv:
        def __init__(self, **kw):
            self.aemo_data = processed_data
            self.step_duration = kw.get("step_duration", 0.5)
            self.max_battery_flow = kw.get("max_battery_flow", 5.0)
            self.battery_capacity = kw.get("battery_capacity", 10.0)
            self.action_mode = kw.get("action_mode", "multi_market")

    monkeypatch.setattr(_env_mod, "AEMOBatteryTradingEnv", FakeEnv)
    monkeypatch.setattr(_dec, "run_single", lambda cls, env, agent_kwargs=None, **kw: (fake_ep_log, fake_inc_log))

    selection = {
        "duid": "HPRG1",
        "dispatch_type": "generator",
        "dispatch_duid_gen": "HPRG1",
        "dispatch_duid_load": "HPRL1",
        "battery_capacity": 193.5,
        "max_battery_flow": 150.0,
        "init_battery_level": 96.75,
        "availability": None,
    }

    du.run_dispatch_replay(
        processed_data=processed_data,
        selection=selection,
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2021, 1, 7),
        region="SA1",
        cache_dir="data/aemo",
        num_episodes=1,
    )

    # Must NOT have used region= when gen/load DUIDs are specified
    assert "region" not in call_kwargs, (
        "run_dispatch_replay must not pass region= when gen/load DUIDs are set "
        "(historical DUIDs may be absent from the static table)"
    )
    # Must have passed the paired DUIDs directly
    assert "duids" in call_kwargs
    assert set(call_kwargs["duids"]) == {"HPRG1", "HPRL1"}


def test_run_dispatch_replay_rejects_selection_region_mismatch(monkeypatch):
    import dispatch_utils as du

    processed_data = pl.DataFrame({
        "SETTLEMENTDATE": [datetime(2025, 1, 1, 0, 0)],
        "REGIONID": ["NSW1"],
        "RRP": [50.0],
    })

    selection = {
        "duid": "HPR1",
        "region": "SA1",
        "dispatch_type": "Bidirectional Unit",
        "dispatch_duid_gen": None,
        "dispatch_duid_load": None,
        "battery_capacity": 193.5,
        "max_battery_flow": 150.0,
        "init_battery_level": 96.75,
        "availability": None,
    }

    with pytest.raises(ValueError, match="selection region='SA1'.*replay region='NSW1'"):
        du.run_dispatch_replay(
            processed_data=processed_data,
            selection=selection,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 7),
            region="NSW1",
            cache_dir="data/aemo",
            num_episodes=1,
        )


# ---------------------------------------------------------------------------
# find_duid_first_dispatch — unit test with mocked NEMOSIS
# ---------------------------------------------------------------------------

def test_find_duid_first_dispatch_returns_first_month(monkeypatch):
    """find_duid_first_dispatch returns the earliest settlement date when the DUID is found."""
    import aemo_data as _aemo

    call_count = [0]

    def fake_dynamic_data_compiler(start_time, end_time, table_name, raw_data_location):
        import pandas as pd
        call_count[0] += 1
        # Return data only on the second call (second month)
        if call_count[0] == 1:
            return pd.DataFrame()  # first month empty
        # Second month has data for target DUID
        return pd.DataFrame({
            "SETTLEMENTDATE": [datetime(2022, 2, 1, 0, 5), datetime(2022, 2, 1, 0, 10)],
            "DUID": ["HPR1", "HPR1"],
            "TOTALCLEARED": [100.0, -50.0],
        })

    monkeypatch.setattr(_aemo, "HAS_NEMOSIS", True)
    monkeypatch.setattr(_aemo, "dynamic_data_compiler", fake_dynamic_data_compiler, raising=False)

    result = _aemo.find_duid_first_dispatch(
        duid="HPR1",
        search_start=datetime(2022, 1, 1),
        search_end=datetime(2022, 3, 1),
        cache_dir="data/aemo",
        verbose=False,
    )
    assert result is not None
    assert result.year == 2022
    assert result.month == 2


def test_find_duid_first_dispatch_returns_none_when_not_found(monkeypatch):
    """find_duid_first_dispatch returns None when no records are found."""
    import aemo_data as _aemo
    import pandas as pd

    monkeypatch.setattr(_aemo, "HAS_NEMOSIS", True)
    monkeypatch.setattr(
        _aemo, "dynamic_data_compiler",
        lambda **kw: pd.DataFrame(),
        raising=False,
    )

    result = _aemo.find_duid_first_dispatch(
        duid="NONEXISTENT99",
        search_start=datetime(2024, 1, 1),
        search_end=datetime(2024, 2, 1),
        cache_dir="data/aemo",
        verbose=False,
    )
    assert result is None


# ---------------------------------------------------------------------------
# scan_duid_historical_availability — unit test with mocked dependencies
# ---------------------------------------------------------------------------

def test_scan_duid_historical_availability_returns_first_dispatch_dates(monkeypatch):
    """scan_duid_historical_availability returns one row per DUID."""
    import dispatch_utils as du
    import aemo_data as _aemo

    # Patch get_available_battery_units
    def _fake_get_avail(cache_dir, region, refresh):
        if region == "SA1":
            return pl.DataFrame({"DUID": ["HPR1", "LBB1"], "Region": ["SA1", "SA1"]})
        return pl.DataFrame()

    monkeypatch.setattr(_aemo, "get_available_battery_units", _fake_get_avail)

    # Patch find_duid_first_dispatch: HPR1 found, LBB1 not found
    def _fake_find_first(duid, search_start, search_end, cache_dir, refresh, verbose):
        if duid == "HPR1":
            return datetime(2022, 6, 1, 0, 5)
        return None  # LBB1 not found

    monkeypatch.setattr(_aemo, "find_duid_first_dispatch", _fake_find_first)

    result = du.scan_duid_historical_availability(
        regions=["SA1"],
        search_start=datetime(2022, 1, 1),
        cache_dir="data/aemo",
        verbose=False,
    )
    assert result.height == 2
    assert "DUID" in result.columns
    assert "FirstDispatchInHistory" in result.columns

    hpr = result.filter(pl.col("DUID") == "HPR1")
    assert hpr.height == 1
    assert hpr["FirstDispatchInHistory"][0].year == 2022

    lbb = result.filter(pl.col("DUID") == "LBB1")
    assert lbb.height == 1
    assert lbb["FirstDispatchInHistory"][0] is None


# ---------------------------------------------------------------------------
# BATTERY_REGISTRY / list_known_batteries / resolve_battery_duids
# ---------------------------------------------------------------------------

def test_list_known_batteries_returns_dataframe():
    """list_known_batteries returns a non-empty DataFrame with expected columns."""
    from dispatch_utils import list_known_batteries
    kb = list_known_batteries()
    assert kb.height > 0
    for col in ("Key", "StationName", "Region", "DUID", "DispatchType", "ValidFrom"):
        assert col in kb.columns, f"Expected column {col!r}"


def test_list_known_batteries_contains_hornsdale():
    """Hornsdale Power Reserve must appear with both old and new DUIDs."""
    from dispatch_utils import list_known_batteries
    kb = list_known_batteries()
    hpr_rows = kb.filter(pl.col("Key") == "hornsdale")
    assert hpr_rows.height >= 3  # HPR1 + HPRG1 + HPRL1
    duids = hpr_rows["DUID"].to_list()
    assert "HPR1" in duids
    assert "HPRG1" in duids
    assert "HPRL1" in duids


def test_resolve_battery_duids_pre_transition():
    """Querying Hornsdale before the transition returns old gen/load DUIDs."""
    import aemo_data as _aemo
    result = _aemo.resolve_battery_duids("hornsdale", datetime(2021, 1, 1), datetime(2021, 6, 30))
    assert result["found"] is True
    assert result["station_name"] == "Hornsdale Power Reserve"
    assert "HPRG1" in result["all_duids_in_range"]
    assert "HPRL1" in result["all_duids_in_range"]
    assert "HPR1" not in result["all_duids_in_range"]
    assert result["spans_transition"] is False


def test_resolve_battery_duids_post_transition():
    """Querying Hornsdale after transition returns only the bidirectional DUID."""
    import aemo_data as _aemo
    result = _aemo.resolve_battery_duids("HPR1", datetime(2023, 1, 1), datetime(2023, 6, 30))
    assert result["found"] is True
    assert result["all_duids_in_range"] == ["HPR1"]
    assert result["bidi_duid"] == "HPR1"
    assert result["spans_transition"] is False


def test_resolve_battery_duids_spans_transition():
    """A range that covers the transition returns all DUIDs and spans_transition=True."""
    import aemo_data as _aemo
    result = _aemo.resolve_battery_duids("hornsdale", datetime(2022, 1, 1), datetime(2023, 1, 1))
    assert result["found"] is True
    assert result["spans_transition"] is True
    assert "HPRG1" in result["all_duids_in_range"]
    assert "HPR1" in result["all_duids_in_range"]
    assert len(result["transition_dates"]) >= 1


def test_resolve_battery_duids_unknown_name():
    """An unknown name returns found=False."""
    import aemo_data as _aemo
    result = _aemo.resolve_battery_duids("nonexistent station xyz", datetime(2023, 1, 1), datetime(2023, 6, 30))
    assert result["found"] is False
    assert result["all_duids_in_range"] == []


def test_fetch_aemo_unit_dispatch_accepts_duids_list(monkeypatch):
    """fetch_aemo_unit_dispatch filters per-chunk when a duids list is given."""
    import aemo_data as _aemo
    import pandas as pd

    def fake_ddc(start_time, end_time, table_name, raw_data_location):
        # Return a fake DISPATCHLOAD chunk with multiple DUIDs
        return pd.DataFrame({
            "SETTLEMENTDATE": ["2025/01/01 00:05:00", "2025/01/01 00:05:00"],
            "DUID": ["HPR1", "OTHER_DUID"],
            "TOTALCLEARED": [100.0, 50.0],
        })

    monkeypatch.setattr(_aemo, "HAS_NEMOSIS", True)
    monkeypatch.setattr(_aemo, "dynamic_data_compiler", fake_ddc, raising=False)

    result = _aemo.fetch_aemo_unit_dispatch(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 2),
        duids=["HPR1"],
        cache_dir="data/aemo",
    )
    # Only HPR1 should be kept; OTHER_DUID should be filtered out
    assert result.height >= 1
    assert set(result["DUID"].to_list()) == {"HPR1"}


def test_fetch_aemo_unit_dispatch_region_prefilter(monkeypatch):
    """fetch_aemo_unit_dispatch pre-filters region DUIDs before the loop."""
    import aemo_data as _aemo
    import pandas as pd

    def fake_ddc(start_time, end_time, table_name, raw_data_location):
        return pd.DataFrame({
            "SETTLEMENTDATE": ["2025/01/01 00:05:00", "2025/01/01 00:05:00",
                               "2025/01/01 00:05:00"],
            "DUID": ["SA1_UNIT", "OTHER_REGION", "SA1_UNIT2"],
            "TOTALCLEARED": [100.0, 50.0, 75.0],
        })

    # Fake static table with Region column
    fake_gen_info = pl.DataFrame({
        "DUID": ["SA1_UNIT", "SA1_UNIT2", "OTHER_REGION"],
        "Region": ["SA1", "SA1", "NSW1"],
    })

    def fake_static(cache_path, refresh):
        return fake_gen_info.to_pandas()

    monkeypatch.setattr(_aemo, "HAS_NEMOSIS", True)
    monkeypatch.setattr(_aemo, "dynamic_data_compiler", fake_ddc, raising=False)
    monkeypatch.setattr(_aemo, "_get_generators_static_table", fake_static)

    result = _aemo.fetch_aemo_unit_dispatch(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 2),
        region="SA1",
        cache_dir="data/aemo",
    )
    # Only SA1 units should be kept
    assert result.height >= 1
    assert "OTHER_REGION" not in result["DUID"].to_list()
    assert "SA1_UNIT" in result["DUID"].to_list()
