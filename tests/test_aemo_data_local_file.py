"""
Tests for local-file fallback in aemo_data.py.

These tests exercise the path where NEMOSIS cannot download the static
'Generators and Scheduled Loads' table and the code falls back to the
bundled copy at src/data/aemo/NEM Registration and Exemption List.xls.

The tests are designed to pass even without an internet connection.
"""

import io
import os
import sys
import shutil
import struct
import tempfile
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aemo_data import (
    _is_zip_format,
    _read_excel_via_pandas,
    _read_generator_info_file,
    _auto_detect_generator_info_file,
    _find_column_by_candidates,
    get_available_battery_units,
    AEMO_REGIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUNDLED_XLS = Path(__file__).resolve().parent.parent / "src" / "data" / "aemo" / "NEM Registration and Exemption List.xls"


def _has_bundled_file() -> bool:
    return BUNDLED_XLS.exists()


skip_without_bundled = pytest.mark.skipif(
    not _has_bundled_file(),
    reason="Bundled NEM Registration and Exemption List.xls not present in src/data/aemo/",
)


# ---------------------------------------------------------------------------
# _is_zip_format
# ---------------------------------------------------------------------------

def test_is_zip_format_detects_xlsx_content(tmp_path):
    """A file that starts with ZIP magic bytes is identified as XLSX format."""
    f = tmp_path / "fake.xls"
    # Write a ZIP/XLSX magic header (PK\x03\x04)
    f.write_bytes(b"PK\x03\x04" + b"\x00" * 20)
    assert _is_zip_format(f) is True


def test_is_zip_format_returns_false_for_biff(tmp_path):
    """A genuine BIFF .xls file is NOT identified as ZIP format."""
    f = tmp_path / "fake.xls"
    # BIFF8 magic bytes: D0 CF 11 E0 A1 B1 1A E1
    f.write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 20)
    assert _is_zip_format(f) is False


def test_is_zip_format_missing_file(tmp_path):
    """Non-existent file returns False (does not raise)."""
    assert _is_zip_format(tmp_path / "nonexistent.xls") is False


# ---------------------------------------------------------------------------
# _read_excel_via_pandas  (CSV fallback created for this test)
# ---------------------------------------------------------------------------

def test_read_excel_via_pandas_reads_xlsx(tmp_path):
    """_read_excel_via_pandas can read a standard .xlsx file."""
    import pandas as pd

    df_input = pd.DataFrame({
        "DUID": ["TESTBAT1", "TESTBAT2"],
        "Region": ["NSW1", "QLD1"],
        "Fuel Source - Primary": ["Battery Storage", "Battery Storage"],
    })
    xlsx_path = tmp_path / "test.xlsx"
    df_input.to_excel(str(xlsx_path), index=False)

    result = _read_excel_via_pandas(xlsx_path)
    assert result.height == 2
    assert "DUID" in result.columns
    assert "Region" in result.columns


# ---------------------------------------------------------------------------
# _read_generator_info_file with the bundled XLSX-in-XLS file
# ---------------------------------------------------------------------------

@skip_without_bundled
def test_read_generator_info_file_bundled_xls():
    """The bundled .xls file (actually XLSX format) is read without error."""
    df = _read_generator_info_file(BUNDLED_XLS)
    assert df.height > 0, "Expected at least one data row"
    assert "DUID" in df.columns, "Expected DUID column"
    assert "Region" in df.columns, "Expected Region column"


@skip_without_bundled
def test_read_generator_info_file_returns_all_columns():
    """The bundled file exposes capacity columns (not just NEMOSIS defaults)."""
    df = _read_generator_info_file(BUNDLED_XLS)
    # Check that the full column list is present (not the truncated NEMOSIS default)
    assert "Maximum storage capacity" in df.columns, (
        "Expected 'Maximum storage capacity' column – got: " + str(df.columns)
    )
    assert "Reg Cap generation (MW)" in df.columns, (
        "Expected 'Reg Cap generation (MW)' column – got: " + str(df.columns)
    )


# ---------------------------------------------------------------------------
# _auto_detect_generator_info_file
# ---------------------------------------------------------------------------

@skip_without_bundled
def test_auto_detect_returns_bundled_file():
    """Auto-detection should find the bundled file in src/data/aemo."""
    detected = _auto_detect_generator_info_file()
    assert detected is not None, "Expected a file to be auto-detected"
    assert detected.exists(), f"Detected path does not exist: {detected}"


# ---------------------------------------------------------------------------
# _find_column_by_candidates – column matching for AEMO column names
# ---------------------------------------------------------------------------

def test_find_column_maximum_storage_capacity():
    """'Maximum storage capacity' is matched by exact candidate."""
    columns = [
        "DUID", "Region", "Fuel Source - Primary", "Fuel Source - Descriptor",
        "Technology Type - Primary", "Technology Type - Descriptor",
        "Reg Cap generation (MW)", "Maximum storage capacity",
    ]
    result = _find_column_by_candidates(
        columns,
        exact_candidates=[
            "storage capacity (mwh)",
            "maximum storage capacity",
            "storage capacity",
        ],
        keyword_groups=[["storage", "mwh"], ["storage", "capacity"]],
    )
    assert result == "Maximum storage capacity"


def test_find_column_reg_cap_generation_via_keyword():
    """'Reg Cap generation (MW)' is matched via ['reg', 'cap', 'mw'] keyword group."""
    columns = [
        "DUID", "Region", "Reg Cap generation (MW)", "Max Cap generation (MW)",
        "Maximum storage capacity",
    ]
    result = _find_column_by_candidates(
        columns,
        exact_candidates=["reg cap (mw)", "registered capacity (mw)", "reg cap generation (mw)"],
        keyword_groups=[["reg", "cap", "mw"]],
    )
    assert result == "Reg Cap generation (MW)"


# ---------------------------------------------------------------------------
# get_available_battery_units – end-to-end with bundled file
# ---------------------------------------------------------------------------

@skip_without_bundled
@pytest.mark.parametrize("region", ["NSW1", "QLD1", "SA1", "VIC1"])
def test_get_available_battery_units_region(region):
    """get_available_battery_units returns battery DUIDs with capacity data for each region."""
    result = get_available_battery_units(cache_dir="data/aemo", region=region)
    assert isinstance(result, pl.DataFrame)
    # Must have the expected output columns
    for col in ("DUID", "Region", "TechnologyType", "FuelType",
                "StorageCapacityMWh", "RegisteredCapacityMW"):
        assert col in result.columns, f"Missing column {col!r} for region {region}"

    if result.height == 0:
        pytest.skip(f"No battery units found for {region} (acceptable if region has none)")

    # All returned rows should be from the requested region
    assert result.filter(pl.col("Region") != region).height == 0, (
        f"Found rows from unexpected regions in {region} result"
    )


@skip_without_bundled
def test_get_available_battery_units_capacity_not_all_null():
    """At least some battery units should have non-null StorageCapacityMWh."""
    result = get_available_battery_units(cache_dir="data/aemo")
    assert result.height > 0, "Expected at least one battery unit across all regions"
    non_null_storage = result.filter(pl.col("StorageCapacityMWh").is_not_null())
    assert non_null_storage.height > 0, (
        "Expected at least some StorageCapacityMWh values to be non-null after fix"
    )
    non_null_mw = result.filter(pl.col("RegisteredCapacityMW").is_not_null())
    assert non_null_mw.height > 0, (
        "Expected at least some RegisteredCapacityMW values to be non-null after fix"
    )


@skip_without_bundled
def test_get_available_battery_units_has_dispatch_type():
    """get_available_battery_units now includes a DispatchType column."""
    result = get_available_battery_units(cache_dir="data/aemo")
    assert "DispatchType" in result.columns, (
        "Expected DispatchType column – got: " + str(result.columns)
    )
    # There should be at least some non-null DispatchType values
    non_null = result.filter(pl.col("DispatchType").is_not_null())
    assert non_null.height > 0, "Expected at least some non-null DispatchType values"


@skip_without_bundled
def test_get_available_battery_units_kep_dispatch_types():
    """KEPBG1 and KEPBL1 should have Generating Unit and Load dispatch types."""
    result = get_available_battery_units(cache_dir="data/aemo", region="QLD1")
    kepbg = result.filter(pl.col("DUID") == "KEPBG1")
    kepbl = result.filter(pl.col("DUID") == "KEPBL1")

    if kepbg.height > 0:
        dt = str(kepbg["DispatchType"][0]).lower()
        assert "generating" in dt, f"KEPBG1 should be 'Generating Unit', got {dt!r}"
    if kepbl.height > 0:
        dt = str(kepbl["DispatchType"][0]).lower()
        assert "load" in dt, f"KEPBL1 should be 'Load', got {dt!r}"


# ---------------------------------------------------------------------------
# dispatch_type sign correction in set_dispatch_data
# ---------------------------------------------------------------------------

def test_set_dispatch_data_load_duid_sign():
    """When dispatch_type='Load', actions should be positive (charging direction)."""
    from types import SimpleNamespace
    from datetime import datetime as _dt
    import numpy as _np
    from decision import AEMOAgent

    mock_aemo_data = pl.DataFrame({
        'SETTLEMENTDATE': [_dt(2024, 1, 1, 0, 0), _dt(2024, 1, 1, 0, 30)],
    })
    mock_env = SimpleNamespace(
        aemo_data=mock_aemo_data,
        step_duration=0.5,
        max_battery_flow=5.0,
        action_mode='simple',
        current_step=0,
        episode_start_idx=0,
    )
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [_dt(2024, 1, 1, 0, 0), _dt(2024, 1, 1, 0, 30)],
        'DUID': ['LOADUNIT1', 'LOADUNIT1'],
        'TOTALCLEARED': [3.0, 5.0],
        'RAISEREG': [0.0, 0.0],
        'LOWERREG': [0.0, 0.0],
    })
    agent = AEMOAgent(mock_env, algorithm='dispatch')

    # Default (assume_generator=True) → negative actions
    agent.set_dispatch_data(dispatch_data, dispatch_duid='LOADUNIT1')
    actions_gen = agent.dispatch_actions.flatten()
    assert _np.all(actions_gen < 0), f"Generator sign expected negative, got {actions_gen}"

    # With dispatch_type='Load' → positive actions
    agent.set_dispatch_data(dispatch_data, dispatch_duid='LOADUNIT1', dispatch_type='Load')
    actions_load = agent.dispatch_actions.flatten()
    assert _np.all(actions_load > 0), f"Load sign expected positive, got {actions_load}"


def test_set_dispatch_data_bidirectional_duid_unaffected():
    """dispatch_type='Bidirectional Unit' should not change the default sign convention."""
    from types import SimpleNamespace
    from datetime import datetime as _dt
    import numpy as _np
    from decision import AEMOAgent

    mock_aemo_data = pl.DataFrame({
        'SETTLEMENTDATE': [_dt(2024, 1, 1, 0, 0), _dt(2024, 1, 1, 0, 30)],
    })
    mock_env = SimpleNamespace(
        aemo_data=mock_aemo_data,
        step_duration=0.5,
        max_battery_flow=5.0,
        action_mode='simple',
        current_step=0,
        episode_start_idx=0,
    )
    dispatch_data = pl.DataFrame({
        'SETTLEMENTDATE': [_dt(2024, 1, 1, 0, 0), _dt(2024, 1, 1, 0, 30)],
        'DUID': ['BIDIBAT1', 'BIDIBAT1'],
        'TOTALCLEARED': [4.0, 2.0],
        'RAISEREG': [0.0, 0.0],
        'LOWERREG': [0.0, 0.0],
    })
    agent = AEMOAgent(mock_env, algorithm='dispatch')
    agent.set_dispatch_data(dispatch_data, dispatch_duid='BIDIBAT1', dispatch_type='Bidirectional Unit')
    actions = agent.dispatch_actions.flatten()
    # 'Bidirectional Unit' is neither "load"-only nor "generating"-only → treated as generator
    assert _np.all(actions < 0), f"Bidirectional should default to generator sign, got {actions}"


# ---------------------------------------------------------------------------
# NonZeroIntervalCount logic (unit test, no network needed)
# ---------------------------------------------------------------------------

def test_nonzero_interval_count_logic():
    """NonZeroIntervalCount correctly identifies all-zero vs active dispatch data."""
    from datetime import datetime as _dt

    # Simulate the expression used in get_dispatch_active_battery_units
    dispatch_zero = pl.DataFrame({
        'SETTLEMENTDATE': [_dt(2024, 1, 1, 0, 5), _dt(2024, 1, 1, 0, 10)],
        'DUID': ['KEPBL1', 'KEPBL1'],
        'TOTALCLEARED': [0.0, 0.0],
        'RAISEREG': [0.0, 0.0],
        'LOWERREG': [0.0, 0.0],
    })
    dispatch_active = pl.DataFrame({
        'SETTLEMENTDATE': [_dt(2024, 1, 1, 0, 5), _dt(2024, 1, 1, 0, 10)],
        'DUID': ['TARBESS1', 'TARBESS1'],
        'TOTALCLEARED': [100.0, -50.0],
        'RAISEREG': [10.0, 0.0],
        'LOWERREG': [0.0, 5.0],
    })

    def compute_nonzero_count(df: pl.DataFrame) -> int:
        from aemo_data import DISPATCH_NONZERO_THRESHOLD
        numeric_cols = [c for c in df.columns if c not in {'SETTLEMENTDATE', 'DUID'}]
        cond = None
        for col in numeric_cols:
            expr = pl.col(col).cast(pl.Float64, strict=False).abs() > DISPATCH_NONZERO_THRESHOLD
            cond = expr if cond is None else (cond | expr)
        flagged = df.with_columns(pl.when(cond).then(1).otherwise(0).alias('_active'))
        return int(flagged['_active'].sum())

    assert compute_nonzero_count(dispatch_zero) == 0, "All-zero data should have NonZeroIntervalCount=0"
    assert compute_nonzero_count(dispatch_active) == 2, "Active data should have NonZeroIntervalCount=2"

