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
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aemo_data import (
    AEMO_CACHE_ONLY_ENV_VAR,
    _is_zip_format,
    _read_excel_via_pandas,
    _read_generator_info_file,
    _auto_detect_generator_info_file,
    _get_generator_info,
    _build_aemo_month_archive_url,
    _get_nemosis_static_cache_dir,
    _dynamic_data_compiler_with_cache_control,
    fetch_aemo_monthly_cache_files,
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


def test_get_nemosis_static_cache_dir_uses_isolated_subdirectory(tmp_path):
    """Static-table downloads are isolated from the main data/aemo cache."""
    cache_dir = tmp_path / "aemo"
    static_dir = _get_nemosis_static_cache_dir(cache_dir)
    assert static_dir == cache_dir / "_nemosis_static"
    assert static_dir.exists()
    assert static_dir.is_dir()


def test_get_generator_info_uses_isolated_static_cache_and_falls_back(monkeypatch, tmp_path):
    """A failed NEMOSIS static fetch should not touch the local fallback file."""
    cache_dir = tmp_path / "aemo"
    local_csv = cache_dir / "NEM Registration and Exemption List.csv"
    local_csv.parent.mkdir(parents=True, exist_ok=True)
    local_csv.write_text(
        "DUID,Region,Fuel Source - Descriptor\n"
        "LOCALBAT1,NSW1,Battery Storage\n",
        encoding="utf-8",
    )

    calls = []

    def fake_static_table(*, table_name, raw_data_location, update_static_file, select_columns):
        calls.append(raw_data_location)
        static_dir = Path(raw_data_location)
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / "NEM Registration and Exemption List.xls").write_text(
            "Sorry, your request has failed. Please return to the home page and try again later.",
            encoding="utf-8",
        )
        raise RuntimeError("simulated fetch failure")

    monkeypatch.setattr("aemo_data.static_table", fake_static_table)
    result = _get_generator_info(cache_dir, generator_info_path=str(local_csv))

    assert result is not None
    assert "LOCALBAT1" in result["DUID"].to_list()
    assert calls == [str(cache_dir / "_nemosis_static")]
    assert local_csv.read_text(encoding="utf-8").startswith("DUID,Region")


def test_dynamic_data_compiler_cache_only_raises_for_missing_month(monkeypatch, tmp_path):
    """Cache-only mode should fail fast instead of attempting a network fetch."""
    monkeypatch.setenv(AEMO_CACHE_ONLY_ENV_VAR, "1")

    with pytest.raises(FileNotFoundError, match="2024-08"):
        _dynamic_data_compiler_with_cache_control(
            start_time="2024/08/01 00:00:00",
            end_time="2024/09/01 00:00:00",
            table_name="DISPATCHLOAD",
            raw_data_location=str(tmp_path),
        )


def test_dynamic_data_compiler_cache_only_uses_existing_local_month(monkeypatch, tmp_path):
    """Cache-only mode should delegate to NEMOSIS when the local month exists."""
    cache_dir = tmp_path / "aemo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_csv = cache_dir / "PUBLIC_ARCHIVE#DISPATCHLOAD#FILE01#202408010000.CSV"
    local_csv.write_text(
        "C,header row ignored by NEMOSIS\n"
        "SETTLEMENTDATE,DUID,TOTALCLEARED\n"
        "2024/08/01 00:05:00,TEST1,1.0\n"
        "C,footer row ignored by NEMOSIS\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(AEMO_CACHE_ONLY_ENV_VAR, "1")
    sentinel = object()
    calls = []

    def fake_dynamic_data_compiler(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr("aemo_data.dynamic_data_compiler", fake_dynamic_data_compiler)

    result = _dynamic_data_compiler_with_cache_control(
        start_time="2024/08/01 00:00:00",
        end_time="2024/09/01 00:00:00",
        table_name="DISPATCHLOAD",
        raw_data_location=str(cache_dir),
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["table_name"] == "DISPATCHLOAD"
    assert calls[0]["raw_data_location"] == str(cache_dir)


def test_build_aemo_month_archive_url_uses_double_encoded_hashes():
    """Post-transition monthly archives require the double-encoded %2523 URL form."""
    url = _build_aemo_month_archive_url("DISPATCHLOAD", 2025, 1)
    assert url.endswith(
        "PUBLIC_ARCHIVE%2523DISPATCHLOAD%2523FILE01%2523202501010000.zip"
    )


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.calls = []

    def get(self, url, timeout, headers):
        self.calls.append({"url": url, "timeout": timeout, "headers": headers})
        return _FakeResponse(self.payload)


def _build_zip_bytes(member_name: str, content: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, content)
    return buffer.getvalue()


def test_fetch_aemo_monthly_cache_files_downloads_expected_csv(tmp_path):
    """The manual downloader should write the exact CSV name NEMOSIS expects."""
    csv_name = "PUBLIC_ARCHIVE#DISPATCHLOAD#FILE01#202501010000.CSV"
    session = _FakeSession(
        _build_zip_bytes(
            csv_name,
            b"SETTLEMENTDATE,DUID,TOTALCLEARED\n2025/01/01 00:05:00,TEST1,1.0\n",
        )
    )

    manifest = fetch_aemo_monthly_cache_files(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 2, 1),
        tables=["DISPATCHLOAD"],
        cache_dir=str(tmp_path),
        session=session,
    )

    cached = tmp_path / csv_name
    assert cached.exists()
    assert "TEST1" in cached.read_text(encoding="utf-8")
    assert manifest == [
        {
            "table_name": "DISPATCHLOAD",
            "year": 2025,
            "month": 1,
            "path": str(cached),
            "url": _build_aemo_month_archive_url("DISPATCHLOAD", 2025, 1),
            "status": "downloaded",
        }
    ]
    assert session.calls[0]["url"].endswith("%2523202501010000.zip")


def test_fetch_aemo_monthly_cache_files_preserves_existing_file_on_bad_archive(tmp_path):
    """A bad web response must not overwrite a valid local cache file."""
    csv_name = "PUBLIC_ARCHIVE#DISPATCHLOAD#FILE01#202501010000.CSV"
    cached = tmp_path / csv_name
    cached.write_text("original-cache\n", encoding="utf-8")
    session = _FakeSession(
        _build_zip_bytes(
            "WRONG_NAME.CSV",
            b"broken\n",
        )
    )

    with pytest.raises(RuntimeError, match="did not contain expected member"):
        fetch_aemo_monthly_cache_files(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 2, 1),
            tables=["DISPATCHLOAD"],
            cache_dir=str(tmp_path),
            overwrite=True,
            session=session,
        )

    assert cached.read_text(encoding="utf-8") == "original-cache\n"


def test_fetch_aemo_monthly_cache_files_retries_transient_request_errors(tmp_path):
    """Transient request failures should be retried before giving up."""
    csv_name = "PUBLIC_ARCHIVE#DISPATCHLOAD#FILE01#202501010000.CSV"
    payload = _build_zip_bytes(
        csv_name,
        b"SETTLEMENTDATE,DUID,TOTALCLEARED\n2025/01/01 00:05:00,TEST1,1.0\n",
    )

    class _FlakySession:
        def __init__(self):
            self.calls = 0

        def get(self, url, timeout, headers):
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectionError("temporary dns failure")
            return _FakeResponse(payload)

    session = _FlakySession()

    manifest = fetch_aemo_monthly_cache_files(
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 2, 1),
        tables=["DISPATCHLOAD"],
        cache_dir=str(tmp_path),
        session=session,
        max_attempts=2,
    )

    assert session.calls == 2
    assert manifest[0]["status"] == "downloaded"


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
