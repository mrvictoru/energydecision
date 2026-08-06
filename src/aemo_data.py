"""
AEMO Data Fetching Module

This module provides utilities for fetching Australian Energy Market Operator (AEMO) 
datasets including FCAS prices, energy prices, and generation data by fuel type.

Data is cached locally to minimize API calls and improve performance.

This module uses NEMOSIS (https://github.com/UNSW-CEEM/NEMOSIS) to fetch actual 
AEMO market data from NEMWEB archives.

Public AEMO data sources:
- NEMWEB: http://nemweb.com.au/ (historical market data)
- MMS Data Model: Historical pricing and generation data

References:
- UNSW-CEEM/NEMOSIS: https://github.com/UNSW-CEEM/NEMOSIS
"""

import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterable, Mapping
from contextlib import contextmanager
import io
import os
import requests
import time
import warnings
import zipfile
from unittest.mock import patch


# Import NEMOSIS for actual AEMO data fetching
try:
    from nemosis import dynamic_data_compiler, static_table
    from nemosis import data_fetch_methods as _nemosis_data_fetch_methods
    HAS_NEMOSIS = True
except ImportError:
    HAS_NEMOSIS = False
    _nemosis_data_fetch_methods = None
    warnings.warn(
        "NEMOSIS not installed. Install with: pip install nemosis. "
        "Falling back to synthetic data generation for demonstration."
    )


# AEMO NEM Regions
AEMO_REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]

# FCAS Services
FCAS_SERVICES = [
    "RAISE6SEC", "LOWER6SEC",  # Fast raise/lower
    "RAISE60SEC", "LOWER60SEC",  # Slow raise/lower
    "RAISE5MIN", "LOWER5MIN",  # Delayed raise/lower
    "RAISEREG", "LOWERREG"  # Regulation raise/lower
]

# Generation fuel types available in AEMO data
FUEL_TYPES = [
    "solar", "wind", "coal_black", "coal_brown", 
    "gas_ccgt", "gas_ocgt", "gas_recip", "hydro", "battery_discharging"
]

# Minimum absolute MW value to consider a dispatch interval "non-zero".
# Values below this threshold (e.g. rounding artefacts) are treated as zero.
DISPATCH_NONZERO_THRESHOLD: float = 0.001

AEMO_CACHE_ONLY_ENV_VAR = "AEMO_CACHE_ONLY"
AEMO_MONTHLY_CACHE_TABLES = (
    "DISPATCHLOAD",
    "DISPATCHPRICE",
    "DISPATCHREGIONSUM",
    "DISPATCH_UNIT_SCADA",
)


# ---------------------------------------------------------------------------
# Battery station registry – historical DUID mappings
# ---------------------------------------------------------------------------
# Maps a short canonical key → station metadata + list of DUIDs ordered by
# registration period (newest first).  ``valid_until=None`` means the DUID is
# still active.  Dates are approximate — use ``find_duid_first_dispatch`` to
# determine the exact transition date for a specific battery.
#
# Dispatch types:
#   "bidirectional" — single DUID handles both charge and discharge (new model)
#   "generator"     — discharge-only DUID of a gen/load pair (old model)
#   "load"          — charge-only DUID of a gen/load pair (old model)
# ---------------------------------------------------------------------------

BATTERY_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- South Australia (SA1) ---
    "hornsdale": {
        "full_name": "Hornsdale Power Reserve",
        "region": "SA1",
        "aliases": ["hpr", "hornsdale power reserve", "hornsdale power"],
        "capacity_mwh": 194.0,   # 150 MW / 194 MWh after 2020 expansion
        "max_power_mw": 150.0,
        "duids": [
            {"duid": "HPR1",  "type": "bidirectional", "valid_from": datetime(2022, 10, 1), "valid_until": None},
            {"duid": "HPRG1", "type": "generator",     "valid_from": datetime(2017, 12, 1), "valid_until": datetime(2022, 10, 1)},
            {"duid": "HPRL1", "type": "load",          "valid_from": datetime(2017, 12, 1), "valid_until": datetime(2022, 10, 1)},
        ],
    },
    "lake_bonney": {
        "full_name": "Lake Bonney BESS1",
        "region": "SA1",
        "aliases": ["lbb", "lbb1", "lkbonny1", "lake bonney", "lake bonney bess"],
        "capacity_mwh": 25.0,
        "max_power_mw": 25.0,
        "duids": [
            {"duid": "LBB1",  "type": "bidirectional", "valid_from": datetime(2022, 6, 1),  "valid_until": None},
            {"duid": "LBBG1", "type": "generator",     "valid_from": datetime(2019, 8, 1),  "valid_until": datetime(2022, 6, 1)},
            {"duid": "LBBL1", "type": "load",          "valid_from": datetime(2019, 8, 1),  "valid_until": datetime(2022, 6, 1)},
            {"duid": "LKBONNY1", "type": "bidirectional", "valid_from": datetime(2019, 8, 1), "valid_until": None},
        ],
    },
    "dalrymple_north": {
        "full_name": "Dalrymple North BESS",
        "region": "SA1",
        "aliases": ["dalnth", "dalnth1", "dalrymple north"],
        "capacity_mwh": 8.0,
        "max_power_mw": 30.0,
        "duids": [
            {"duid": "DALNTH1",  "type": "bidirectional", "valid_from": datetime(2023, 1, 1), "valid_until": None},
            {"duid": "DALNTH01", "type": "bidirectional", "valid_from": datetime(2018, 4, 1), "valid_until": datetime(2023, 1, 1)},
        ],
    },
    "blyth": {
        "full_name": "Blyth Battery Energy Storage System",
        "region": "SA1",
        "aliases": ["blythb1", "blyth bess"],
        "capacity_mwh": 14.0,
        "max_power_mw": 10.0,
        "duids": [
            {"duid": "BLYTHB1", "type": "bidirectional", "valid_from": datetime(2023, 1, 1), "valid_until": None},
        ],
    },
    "bungama": {
        "full_name": "Bungama Battery Energy Storage System",
        "region": "SA1",
        "aliases": ["bungamb1"],
        "capacity_mwh": 50.0,
        "max_power_mw": 50.0,
        "duids": [
            {"duid": "BUNGAMB1", "type": "bidirectional", "valid_from": datetime(2023, 6, 1), "valid_until": None},
        ],
    },
    "torrens_island": {
        "full_name": "Torrens Island BESS",
        "region": "SA1",
        "aliases": ["tib1", "torrens island bess", "torrens"],
        "capacity_mwh": 250.0,
        "max_power_mw": 250.0,
        "duids": [
            {"duid": "TIB1", "type": "bidirectional", "valid_from": datetime(2024, 1, 1), "valid_until": None},
        ],
    },
    # --- Victoria (VIC1) ---
    "ballarat": {
        "full_name": "Ballarat Battery Energy Storage System",
        "region": "VIC1",
        "aliases": ["balb1", "ballarat bess"],
        "capacity_mwh": 30.0,
        "max_power_mw": 30.0,
        "duids": [
            {"duid": "BALB1",  "type": "bidirectional", "valid_from": datetime(2023, 1, 1), "valid_until": None},
            {"duid": "BALBG1", "type": "generator",     "valid_from": datetime(2019, 7, 1), "valid_until": datetime(2023, 1, 1)},
        ],
    },
    "gannawarra": {
        "full_name": "Gannawarra Energy Storage System",
        "region": "VIC1",
        "aliases": ["gannb1", "gannawarra ess"],
        "capacity_mwh": 25.0,
        "max_power_mw": 25.0,
        "duids": [
            {"duid": "GANNB1",  "type": "bidirectional", "valid_from": datetime(2023, 1, 1),  "valid_until": None},
            {"duid": "GANNBG1", "type": "generator",     "valid_from": datetime(2018, 12, 1), "valid_until": datetime(2023, 1, 1)},
            {"duid": "GANNBL1", "type": "load",          "valid_from": datetime(2018, 12, 1), "valid_until": datetime(2023, 1, 1)},
        ],
    },
    "victorian_big_battery": {
        "full_name": "Victorian Big Battery",
        "region": "VIC1",
        "aliases": ["vbb1", "vbb", "vbbg1", "vbbl1", "big battery"],
        "capacity_mwh": 450.0,
        "max_power_mw": 300.0,
        "duids": [
            {"duid": "VBBG1", "type": "generator", "valid_from": datetime(2021, 11, 1), "valid_until": None},
            {"duid": "VBBL1", "type": "load",      "valid_from": datetime(2021, 11, 1), "valid_until": None},
            {"duid": "VBB1", "type": "bidirectional", "valid_from": datetime(2021, 11, 1), "valid_until": None},
        ],
    },
    "bulgana": {
        "full_name": "Bulgana Green Power Hub",
        "region": "VIC1",
        "aliases": ["bulbes1", "bulgana green power"],
        "capacity_mwh": 20.0,
        "max_power_mw": 20.0,
        "duids": [
            {"duid": "BULBES1", "type": "bidirectional", "valid_from": datetime(2019, 12, 1), "valid_until": None},
        ],
    },
    # --- Queensland (QLD1) ---
    "kennedy_energy_park": {
        "full_name": "Kennedy Energy Park Battery",
        "region": "QLD1",
        "aliases": ["kennedy", "kepbg1", "kepbl1"],
        "capacity_mwh": 4.0,
        "max_power_mw": 2.0,
        "duids": [
            {"duid": "KEPBG1", "type": "generator", "valid_from": datetime(2019, 12, 1), "valid_until": None},
            {"duid": "KEPBL1", "type": "load",      "valid_from": datetime(2019, 12, 1), "valid_until": None},
        ],
    },
    "wandoan": {
        "full_name": "Wandoan Battery Energy Storage System",
        "region": "QLD1",
        "aliases": ["wandb1", "wandoan bess"],
        "capacity_mwh": 150.0,
        "max_power_mw": 100.0,
        "duids": [
            {"duid": "WANDB1", "type": "bidirectional", "valid_from": datetime(2021, 6, 1), "valid_until": None},
        ],
    },
    # --- New South Wales (NSW1) ---
    "waratah": {
        "full_name": "Waratah Super Battery",
        "region": "NSW1",
        "aliases": ["wtahb1", "waratah super"],
        "capacity_mwh": 850.0,
        "max_power_mw": 850.0,
        "duids": [
            {"duid": "WTAHB1", "type": "bidirectional", "valid_from": datetime(2024, 12, 1), "valid_until": None},
        ],
    },
    "wallgrove": {
        "full_name": "Wallgrove BESS 1",
        "region": "NSW1",
        "aliases": ["walgrv1", "wallgrove bess"],
        "capacity_mwh": 50.0,
        "max_power_mw": 50.0,
        "duids": [
            {"duid": "WALGRV1", "type": "bidirectional", "valid_from": datetime(2021, 4, 1), "valid_until": None},
        ],
    },
}

# Build a reverse lookup: DUID → registry key
_DUID_TO_REGISTRY_KEY: Dict[str, str] = {
    entry["duid"]: key
    for key, info in BATTERY_REGISTRY.items()
    for entry in info["duids"]
}


def list_known_batteries() -> pl.DataFrame:
    """Return a DataFrame of all batteries in the built-in ``BATTERY_REGISTRY``.

    Each row represents one DUID registration period.  Use the ``Key`` column
    to pass a station name to :func:`resolve_battery_duids`.

    Columns:

    * ``Key`` — canonical registry key (short, lowercase, usable as a name)
    * ``StationName`` — human-readable station name
    * ``Region`` — NEM region
    * ``DUID`` — dispatch unit ID
    * ``DispatchType`` — ``"bidirectional"``, ``"generator"``, or ``"load"``
    * ``ValidFrom`` — approximate start of this DUID registration
    * ``ValidUntil`` — approximate end (``null`` = still active)

    Example::

        from aemo_data import list_known_batteries
        list_known_batteries()
    """
    rows = []
    for key, info in BATTERY_REGISTRY.items():
        for entry in info["duids"]:
            rows.append({
                "Key":          key,
                "StationName":  info["full_name"],
                "Region":       info["region"],
                "DUID":         entry["duid"],
                "DispatchType": entry["type"],
                "ValidFrom":    entry.get("valid_from"),
                "ValidUntil":   entry.get("valid_until"),
            })
    return pl.DataFrame(rows, schema_overrides={
        "ValidFrom": pl.Datetime,
        "ValidUntil": pl.Datetime,
    }).sort(["Region", "StationName", "ValidFrom"])


def resolve_battery_duids(
    name_or_duid: str,
    start_date: datetime,
    end_date: datetime,
) -> Dict[str, Any]:
    """Resolve a station name or DUID to the correct DUID(s) for a date range.

    This is the recommended entry-point when you want to query a specific
    battery by name without worrying about which DUID it used in a given
    period.  The function looks up the station in :data:`BATTERY_REGISTRY`
    (supports fuzzy matching on the ``Key`` field and common aliases) and
    returns the DUIDs that were **active** during ``[start_date, end_date)``.

    When the date range **spans a transition** (e.g. a battery changed from a
    gen/load pair to a bidirectional DUID mid-range) both the old and new DUIDs
    are returned so callers can stitch the data together.

    Args:
        name_or_duid: A registry key (e.g. ``"hornsdale"``), a station name
            alias (e.g. ``"Hornsdale Power Reserve"``), or an exact DUID
            (e.g. ``"HPR1"``).  Matching is case-insensitive.
        start_date: Start of the date range you want to query.
        end_date: End of the date range.

    Returns:
        A dict with the following keys:

        ``found`` (bool)
            Whether the name / DUID was found in the registry.
        ``key`` (str | None)
            Registry key, or ``None`` if not found.
        ``station_name`` (str | None)
            Human-readable station name.
        ``region`` (str | None)
            NEM region.
        ``active_duids`` (list[dict])
            List of ``{"duid", "type", "valid_from", "valid_until"}`` entries
            that overlap with the requested date range.
        ``all_duids_in_range`` (list[str])
            Flat list of all active DUIDs (convenient for passing to
            :func:`fetch_aemo_unit_dispatch`).
        ``gen_duid`` (str | None)
            Generator DUID active in the range (for old-model batteries).
        ``load_duid`` (str | None)
            Load DUID active in the range.
        ``bidi_duid`` (str | None)
            Bidirectional DUID active in the range.
        ``spans_transition`` (bool)
            ``True`` when the range covers a DUID transition.
        ``transition_dates`` (list[datetime])
            Transition dates that fall within the range.

    Example::

        from aemo_data import resolve_battery_duids
        from datetime import datetime

        result = resolve_battery_duids("hornsdale", datetime(2021, 1, 1), datetime(2023, 6, 30))
        # spans_transition=True because Hornsdale moved from HPRG1/HPRL1 to HPR1 in Oct 2022
        print(result["all_duids_in_range"])   # ['HPRG1', 'HPRL1', 'HPR1']
    """
    # Normalise the query
    query = str(name_or_duid).strip().lower()

    # 1. Check if the query matches a registry key directly
    matched_key: Optional[str] = None
    if query in BATTERY_REGISTRY:
        matched_key = query
    else:
        # 2. Check aliases
        for key, info in BATTERY_REGISTRY.items():
            aliases_lower = [a.lower() for a in info.get("aliases", [])]
            if query in aliases_lower:
                matched_key = key
                break
        # 3. Check if it is a known DUID
        if matched_key is None:
            duid_upper = query.upper()
            if duid_upper in _DUID_TO_REGISTRY_KEY:
                matched_key = _DUID_TO_REGISTRY_KEY[duid_upper]
        # 4. Partial match on full station name or key
        if matched_key is None:
            for key, info in BATTERY_REGISTRY.items():
                if query in info["full_name"].lower() or query in key:
                    matched_key = key
                    break

    if matched_key is None:
        return {
            "found": False,
            "key": None,
            "station_name": None,
            "region": None,
            "active_duids": [],
            "all_duids_in_range": [],
            "gen_duid": None,
            "load_duid": None,
            "bidi_duid": None,
            "spans_transition": False,
            "transition_dates": [],
        }

    info = BATTERY_REGISTRY[matched_key]
    active_entries = []
    transition_dates_in_range = []

    for entry in info["duids"]:
        v_from = entry.get("valid_from") or datetime(2000, 1, 1)
        v_until = entry.get("valid_until") or datetime(9999, 12, 31)
        # Overlap: entry is active during [v_from, v_until) and query spans [start, end)
        if v_from < end_date and v_until > start_date:
            active_entries.append(entry)
        # Collect transitions within range
        if entry.get("valid_until") and start_date <= entry["valid_until"] <= end_date:
            transition_dates_in_range.append(entry["valid_until"])

    gen_duid: Optional[str] = None
    load_duid: Optional[str] = None
    bidi_duid: Optional[str] = None
    for entry in active_entries:
        t = entry.get("type", "")
        if t == "generator":
            gen_duid = entry["duid"]
        elif t == "load":
            load_duid = entry["duid"]
        elif t == "bidirectional":
            bidi_duid = entry["duid"]

    spans = len(transition_dates_in_range) > 0

    return {
        "found": True,
        "key": matched_key,
        "station_name": info["full_name"],
        "region": info["region"],
        "active_duids": active_entries,
        "all_duids_in_range": [e["duid"] for e in active_entries],
        "gen_duid": gen_duid,
        "load_duid": load_duid,
        "bidi_duid": bidi_duid,
        "spans_transition": spans,
        "transition_dates": sorted(transition_dates_in_range),
    }


def _as_polars(df: Any) -> pl.DataFrame:
    """Best-effort conversion of foreign DataFrames (e.g. pandas from NEMOSIS) to Polars."""
    if df is None:
        return pl.DataFrame()
    if isinstance(df, pl.DataFrame):
        return df
    try:
        return pl.from_pandas(df)
    except Exception as e:
        raise TypeError(f"Unsupported dataframe type for conversion to Polars: {type(df)!r}") from e


def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    for c in df.columns:
        c2 = str(c).strip()
        if c2 != c:
            rename_map[c] = c2
    return df.rename(rename_map) if rename_map else df


def _coerce_datetime(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df
    dtype = df.schema.get(col)
    if dtype == pl.Datetime:
        return df
    if dtype == pl.Utf8:
        return df.with_columns(pl.col(col).str.strptime(pl.Datetime, strict=False))
    return df.with_columns(pl.col(col).cast(pl.Datetime, strict=False))


def _coerce_f64(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df
    return df.with_columns(pl.col(col).cast(pl.Float64, strict=False))


def _iter_month_windows(start_date: datetime, end_date: datetime) -> list[tuple[datetime, datetime]]:
    """Split a date range into month-bounded windows to keep NEMOSIS fetches smaller."""
    if end_date <= start_date:
        return [(start_date, end_date)]

    windows: list[tuple[datetime, datetime]] = []
    window_start = start_date
    while window_start < end_date:
        if window_start.month == 12:
            next_month = datetime(window_start.year + 1, 1, 1)
        else:
            next_month = datetime(window_start.year, window_start.month + 1, 1)
        window_end = min(end_date, next_month)
        if window_end <= window_start:
            break
        windows.append((window_start, window_end))
        window_start = window_end

    return windows or [(start_date, end_date)]


def _rows_to_polars(headers: list[str], rows: Iterable[Iterable[Any]]) -> pl.DataFrame:
    normalized_headers = [str(h).strip() if h is not None else "" for h in headers]
    out_rows: list[dict[str, Any]] = []
    for r in rows:
        vals = list(r)
        row_dict = {}
        for i, h in enumerate(normalized_headers):
            if not h:
                continue
            row_dict[h] = vals[i] if i < len(vals) else None
        if row_dict:
            out_rows.append(row_dict)
    df = pl.DataFrame(out_rows) if out_rows else pl.DataFrame()
    return _normalize_columns(df)


def _is_zip_format(file_path: Path) -> bool:
    """Return True if the file starts with the ZIP magic bytes (``PK\\x03\\x04``), indicating
    it is actually in XLSX/OOXML format regardless of its file extension."""
    try:
        with open(file_path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def _read_excel_via_pandas(file_path: Path) -> pl.DataFrame:
    """Read an Excel workbook (any extension) using pandas and return a Polars DataFrame.

    Selects the first sheet that contains both a 'DUID' and 'Region' column.
    Falls back to the first non-empty sheet if no such sheet is found.
    Column names are stripped of leading/trailing whitespace before conversion.
    """
    import pandas as pd

    required = {"duid", "region"}
    pd_sheets: dict = pd.read_excel(str(file_path), sheet_name=None, dtype=str)

    chosen_df = None
    for _sheet_name, df in pd_sheets.items():
        cols_lower = {str(c).strip().lower() for c in df.columns}
        if required.issubset(cols_lower):
            chosen_df = df
            break

    if chosen_df is None:
        # Fall back to the first non-empty sheet
        for df in pd_sheets.values():
            if not df.empty:
                chosen_df = df
                break

    if chosen_df is None:
        return pl.DataFrame()

    # Normalise column names (strip surrounding whitespace)
    chosen_df = chosen_df.copy()
    chosen_df.columns = [str(c).strip() for c in chosen_df.columns]
    return _normalize_columns(_as_polars(chosen_df))


def _read_generator_info_file(file_path: Path) -> pl.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _normalize_columns(pl.read_csv(file_path))

    # Support reading old .xls by converting to .xlsx first (xls2xlsx is in requirements).
    # However, AEMO sometimes distributes files with a .xls extension that are actually in
    # XLSX/OOXML (ZIP) format.  xls2xlsx cannot convert these files, so detect the format
    # via magic bytes and use the pandas path directly when the file is already XLSX.
    if suffix == ".xls":
        if _is_zip_format(file_path):
            # File is XLSX format despite the .xls extension – use pandas which handles
            # the actual binary content rather than relying on the extension.
            return _read_excel_via_pandas(file_path)

        # Genuine BIFF/XLS format – convert to .xlsx with xls2xlsx then continue.
        try:
            from xls2xlsx import XLS2XLSX
        except Exception as e:
            raise ImportError(
                "Reading .xls generator info requires the optional package 'xls2xlsx'. "
                "Install it in your environment or convert the file to .xlsx/.csv."
            ) from e

        xlsx_path = file_path.with_suffix(".xlsx")
        if not xlsx_path.exists():
            x2x = XLS2XLSX(str(file_path))
            x2x.to_xlsx(str(xlsx_path))
        file_path = xlsx_path
        suffix = ".xlsx"

    if suffix in {".xlsx", ".xlsm"}:
        # Try pandas first – it reliably handles multi-sheet workbooks and all column
        # names regardless of edge cases in the Polars/openpyxl readers.
        try:
            return _read_excel_via_pandas(file_path)
        except Exception:
            pass

        # Fallback 1: Polars' native Excel reader.
        if hasattr(pl, "read_excel"):
            try:
                sheets = pl.read_excel(file_path, sheet_name=None)  # type: ignore[attr-defined]
                if isinstance(sheets, dict) and sheets:
                    required = {"duid", "region"}
                    for _, sheet in sheets.items():
                        sheet = _normalize_columns(sheet)
                        cols_lower = {c.lower() for c in sheet.columns}
                        if required.issubset(cols_lower):
                            return sheet
                    return _normalize_columns(next(iter(sheets.values())))
            except Exception:
                pass

        # Fallback 2: openpyxl.
        try:
            from openpyxl import load_workbook
        except Exception as e:
            raise ImportError(
                "Reading .xlsx generator info requires 'openpyxl'. Install it or provide a CSV."
            ) from e

        wb = load_workbook(filename=str(file_path), read_only=True, data_only=True)
        required = {"duid", "region"}
        chosen_headers: list[str] | None = None
        chosen_rows: Iterable[Iterable[Any]] | None = None

        for ws in wb.worksheets:
            values = ws.values
            try:
                headers = next(values)
            except StopIteration:
                continue
            headers_list = [str(h).strip() if h is not None else "" for h in headers]
            cols_lower = {h.lower() for h in headers_list if h}
            if required.issubset(cols_lower):
                chosen_headers = headers_list
                chosen_rows = values
                break

            if chosen_headers is None:
                chosen_headers = headers_list
                chosen_rows = values

        if chosen_headers is None or chosen_rows is None:
            return pl.DataFrame()

        return _rows_to_polars(chosen_headers, chosen_rows)

    raise ValueError(f"Unsupported generator info file type: {file_path}")


def _auto_detect_generator_info_file() -> Optional[Path]:
    """Best-effort lookup for a locally downloaded generator info XLS/XLSX/CSV.

    Searches ``src/data/aemo`` first (project-bundled copy), then
    ``data/aemo/manual`` (human-managed local copy), then ``data/aemo`` for
    backward compatibility. Returns the first matching file found so that a
    bundled or manual copy takes priority over the runtime cache.
    If no file is found, returns ``None``.
    """
    repo_root = Path(__file__).resolve().parent.parent
    search_dirs = [
        repo_root / "src/data/aemo",
        repo_root / "data/aemo/manual",
        repo_root / "data/aemo",
    ]

    for directory in search_dirs:
        if not (directory.exists() and directory.is_dir()):
            continue
        for ext in ("*.xls", "*.xlsx", "*.csv"):
            for candidate in sorted(directory.glob(ext)):
                return candidate  # return first match from highest-priority dir

    return None


def _get_nemosis_static_cache_dir(cache_path: Path) -> Path:
    """Return the dedicated cache directory for NEMOSIS static tables."""
    static_cache = cache_path / "_nemosis_static"
    static_cache.mkdir(parents=True, exist_ok=True)
    return static_cache


def _get_generator_info(cache_path: Path, generator_info_path: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Retrieve generator static info (DUID -> Region/Fuel descriptor).

    Tries NEMOSIS `static_table()` first; if that fails (e.g. AEMO blocks downloads),
    falls back to a user-provided local XLS/XLSX/CSV.
    """
    # 1) Try NEMOSIS static_table (preferred when it works)
    static_cache_path = _get_nemosis_static_cache_dir(cache_path)
    try:
        gen_info = static_table(
            table_name='Generators and Scheduled Loads',
            raw_data_location=str(static_cache_path),
            update_static_file=False,
            select_columns='all',
        )
        if gen_info is not None and len(gen_info) > 0:
            return _normalize_columns(_as_polars(gen_info))
    except Exception:
        pass

    # 2) Fall back to local file
    resolved: Optional[Path] = None
    if generator_info_path:
        resolved = Path(generator_info_path)
    else:
        env_path = os.getenv("AEMO_GENERATORS_FILE")
        if env_path:
            resolved = Path(env_path)

    if resolved is None:
        resolved = _auto_detect_generator_info_file()

    if resolved is None:
        return None

    if not resolved.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        resolved = (repo_root / resolved).resolve()

    if not resolved.exists():
        return None

    try:
        return _read_generator_info_file(resolved)
    except Exception:
        return None


def get_cache_dir(base_dir: str = "data/aemo") -> Path:
    """
    Get or create the cache directory for AEMO data.
    
    Args:
        base_dir: Base directory for caching (default: data/aemo)
        
    Returns:
        Path object for the cache directory
    """
    cache_path = Path(base_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _is_truthy_env(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_aemo_cache_only_enabled() -> bool:
    """Return True when dynamic AEMO fetches must use the local cache only."""
    return _is_truthy_env(os.getenv(AEMO_CACHE_ONLY_ENV_VAR))


def _build_nemosis_month_stub(table_name: str, year: int, month: int, chunk: int = 1) -> str:
    """Mirror the NEMOSIS on-disk filename stub for monthly MMS tables."""
    if year > 2024 or (year == 2024 and month >= 8):
        return f"PUBLIC_ARCHIVE#{table_name}#FILE{chunk:02d}#{year}{month:02d}010000"
    return f"PUBLIC_DVD_{table_name}_{year}{month:02d}010000"


def _build_aemo_month_archive_url(table_name: str, year: int, month: int, chunk: int = 1) -> str:
    """Return the NEMWeb monthly archive URL for a table/month pair."""
    archive_name = _build_nemosis_month_stub(table_name, year, month, chunk=chunk).replace(
        "#", "%2523"
    )
    return (
        "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
        f"{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/{archive_name}.zip"
    )


def _has_local_nemosis_month_file(cache_path: Path, table_name: str, year: int, month: int) -> bool:
    """Return True when the month exists locally as CSV/feather/parquet."""
    stub = _build_nemosis_month_stub(table_name, year, month)
    patterns = [
        cache_path / f"{stub}.csv",
        cache_path / f"{stub}.CSV",
        cache_path / f"{stub}.feather",
        cache_path / f"{stub}.parquet",
    ]
    return any(path.exists() for path in patterns)


def _download_aemo_month_archive_file(
    *,
    url: str,
    expected_csv_name: str,
    destination_path: Path,
    session: Optional[requests.Session] = None,
    timeout: int = 180,
    max_attempts: int = 5,
) -> None:
    """Download one monthly archive zip and atomically write the expected CSV."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(f"{destination_path.suffix}.tmp")
    client = session or requests.Session()
    last_error: Optional[Exception] = None
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            last_error = None
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(min(2 * attempt, 10))

    if response is None:
        raise RuntimeError(f"Failed to download {url}") from last_error

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if expected_csv_name not in archive.namelist():
                raise RuntimeError(
                    f"{url} did not contain expected member {expected_csv_name}; "
                    f"found {archive.namelist()[:5]}"
                )
            with archive.open(expected_csv_name) as src, temporary_path.open("wb") as dst:
                dst.write(src.read())
        temporary_path.replace(destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def fetch_aemo_monthly_cache_files(
    *,
    start_date: datetime,
    end_date: datetime,
    tables: Iterable[str] = AEMO_MONTHLY_CACHE_TABLES,
    cache_dir: str = "data/aemo",
    overwrite: bool = False,
    session: Optional[requests.Session] = None,
    timeout: int = 180,
    max_attempts: int = 5,
) -> list[dict[str, Any]]:
    """Download monthly MMS archives into NEMOSIS-compatible cache files."""
    if end_date <= start_date:
        raise ValueError("end_date must be later than start_date.")

    cache_path = get_cache_dir(cache_dir)
    manifest: list[dict[str, Any]] = []
    requested_tables = [str(table).strip().upper() for table in tables if str(table).strip()]
    for window_start, _ in _iter_month_windows(start_date, end_date):
        for table_name in requested_tables:
            stub = _build_nemosis_month_stub(table_name, window_start.year, window_start.month)
            csv_name = f"{stub}.CSV"
            destination_path = cache_path / csv_name
            if destination_path.exists() and not overwrite:
                manifest.append(
                    {
                        "table_name": table_name,
                        "year": window_start.year,
                        "month": window_start.month,
                        "path": str(destination_path),
                        "url": _build_aemo_month_archive_url(
                            table_name, window_start.year, window_start.month
                        ),
                        "status": "existing",
                    }
                )
                continue

            url = _build_aemo_month_archive_url(table_name, window_start.year, window_start.month)
            _download_aemo_month_archive_file(
                url=url,
                expected_csv_name=csv_name,
                destination_path=destination_path,
                session=session,
                timeout=timeout,
                max_attempts=max_attempts,
            )
            manifest.append(
                {
                    "table_name": table_name,
                    "year": window_start.year,
                    "month": window_start.month,
                    "path": str(destination_path),
                    "url": url,
                    "status": "downloaded",
                }
            )

    return manifest


def _missing_local_nemosis_months(
    start_time: str,
    end_time: str,
    table_name: str,
    raw_data_location: str,
) -> list[str]:
    """Return missing local monthly cache keys for the requested range."""
    cache_path = Path(raw_data_location)
    start_dt = datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S")
    end_dt = datetime.strptime(end_time, "%Y/%m/%d %H:%M:%S")
    missing: list[str] = []
    for window_start, _ in _iter_month_windows(start_dt, end_dt):
        if not _has_local_nemosis_month_file(
            cache_path, table_name, window_start.year, window_start.month
        ):
            missing.append(f"{window_start.year}-{window_start.month:02d}")
    return missing


@contextmanager
def _nemosis_downloads_disabled():
    """Prevent NEMOSIS from downloading missing dynamic-table files."""
    if not HAS_NEMOSIS or _nemosis_data_fetch_methods is None:
        yield
        return
    with patch.object(_nemosis_data_fetch_methods, "_download_data", lambda *args, **kwargs: None):
        yield


def _dynamic_data_compiler_with_cache_control(
    *,
    start_time: str,
    end_time: str,
    table_name: str,
    raw_data_location: str,
    **kwargs,
):
    """Wrap NEMOSIS with an optional cache-only mode driven by env var."""
    cache_only = kwargs.pop("cache_only", None)
    if cache_only is None:
        cache_only = _is_aemo_cache_only_enabled()

    if cache_only:
        missing_months = _missing_local_nemosis_months(
            start_time=start_time,
            end_time=end_time,
            table_name=table_name,
            raw_data_location=raw_data_location,
        )
        if missing_months:
            first_year, first_month = map(int, missing_months[0].split("-"))
            expected = _build_nemosis_month_stub(table_name, first_year, first_month)
            raise FileNotFoundError(
                f"{AEMO_CACHE_ONLY_ENV_VAR}=1 and local cache is missing {table_name} month(s) "
                f"{', '.join(missing_months)} under {raw_data_location}. "
                f"Expected a file such as {expected}.CSV"
            )

        with _nemosis_downloads_disabled():
            return dynamic_data_compiler(
                start_time=start_time,
                end_time=end_time,
                table_name=table_name,
                raw_data_location=raw_data_location,
                **kwargs,
            )

    return dynamic_data_compiler(
        start_time=start_time,
        end_time=end_time,
        table_name=table_name,
        raw_data_location=raw_data_location,
        **kwargs,
    )


def fetch_aemo_dispatch_price(
    start_date: datetime,
    end_date: datetime,
    region: str = "NSW1",
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Fetch AEMO dispatch prices (5-minute intervals) for a specified region and date range.
    
    This function uses NEMOSIS to download dispatch price data from AEMO's public NEMWEB repository.
    The data includes Regional Reference Price (RRP) which is the spot price for energy.
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        region: AEMO region code (NSW1, QLD1, SA1, TAS1, VIC1)
        cache_dir: Directory to cache downloaded data
        refresh: If True, re-download even if cached data exists (NEMOSIS handles caching)
        
    Returns:
        Polars DataFrame with columns:
            - SETTLEMENTDATE: Datetime of the dispatch interval
            - REGIONID: Region identifier
            - RRP: Regional Reference Price ($/MWh)
            - TOTALDEMAND: Total demand in the region (MW)
            
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> prices = fetch_aemo_dispatch_price(start, end, region="NSW1")
        >>> print(prices.head())
    """
    if region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")
    
    if not HAS_NEMOSIS:
        raise ImportError(
            "NEMOSIS is required to fetch actual AEMO data. "
            "Install with: pip install nemosis"
        )
    
    cache_path = get_cache_dir(cache_dir)
    
    # Format datetime for NEMOSIS (requires specific format: 'YYYY/MM/DD HH:MM:SS')
    start_time = start_date.strftime('%Y/%m/%d %H:%M:%S')
    end_time = end_date.strftime('%Y/%m/%d %H:%M:%S')
    
    print(f"Fetching dispatch price data for {region} from {start_date.date()} to {end_date.date()}...")
    
    # Use NEMOSIS to fetch DISPATCHPRICE table
    # This table contains regional reference prices at 5-minute intervals
    try:
        price_data = _dynamic_data_compiler_with_cache_control(
            start_time=start_time,
            end_time=end_time,
            table_name='DISPATCHPRICE',
            raw_data_location=str(cache_path)
        )

        # Filter to the specified region
        price_pl = _as_polars(price_data)
        if price_pl.height > 0:
            price_pl = _normalize_columns(price_pl)
            if 'REGIONID' in price_pl.columns:
                price_pl = price_pl.filter(pl.col('REGIONID') == region)

            price_pl = _coerce_datetime(price_pl, 'SETTLEMENTDATE')
            price_pl = _coerce_f64(price_pl, 'RRP')

            # Also fetch DISPATCHREGIONSUM for demand data
            demand_data = _dynamic_data_compiler_with_cache_control(
                start_time=start_time,
                end_time=end_time,
                table_name='DISPATCHREGIONSUM',
                raw_data_location=str(cache_path)
            )

            demand_pl = _normalize_columns(_as_polars(demand_data))
            if demand_pl.height > 0 and 'REGIONID' in demand_pl.columns:
                demand_pl = demand_pl.filter(pl.col('REGIONID') == region)
            demand_pl = _coerce_datetime(demand_pl, 'SETTLEMENTDATE')
            demand_pl = _coerce_f64(demand_pl, 'TOTALDEMAND')

            if demand_pl.height > 0 and 'TOTALDEMAND' in demand_pl.columns:
                demand_pl = demand_pl.select(['SETTLEMENTDATE', 'TOTALDEMAND']).unique(subset=['SETTLEMENTDATE'])
                out = price_pl.join(demand_pl, on='SETTLEMENTDATE', how='left')
            else:
                out = price_pl.with_columns(pl.lit(0.0).alias('TOTALDEMAND'))

            out = out.select(['SETTLEMENTDATE', 'REGIONID', 'RRP', 'TOTALDEMAND']).sort('SETTLEMENTDATE')
            out = out.with_columns(pl.col('TOTALDEMAND').fill_null(0.0))
            print(f"Fetched {len(out)} price records")
            return out
        else:
            print("No data returned from NEMOSIS")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'RRP': pl.Float64,
                'TOTALDEMAND': pl.Float64
            })
            
    except Exception as e:
        print(f"Error fetching data from NEMOSIS: {e}")
        print("Note: NEMOSIS requires internet connection to download data from AEMO")
        raise


def fetch_aemo_fcas_price(
    start_date: datetime,
    end_date: datetime,
    region: str = "NSW1",
    service: str = "RAISEREG",
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Fetch AEMO FCAS (Frequency Control Ancillary Services) prices for a region and service.
    
    FCAS prices are for services that help maintain power system frequency at 50Hz.
    Prices are typically lower than energy prices but provide additional revenue streams.
    
    Note: This function returns regional FCAS prices only. For unit-specific enablement
    and dispatch data, use fetch_aemo_unit_dispatch().
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        region: AEMO region code (NSW1, QLD1, SA1, TAS1, VIC1)
        service: FCAS service type (RAISEREG, LOWERREG, RAISE6SEC, etc.)
        cache_dir: Directory to cache downloaded data
        refresh: If True, re-download even if cached data exists
        
    Returns:
        Polars DataFrame with columns:
            - SETTLEMENTDATE: Datetime of the dispatch interval
            - REGIONID: Region identifier
            - SERVICE: FCAS service type
            - PRICE: FCAS price ($/MW/h)
            
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> fcas = fetch_aemo_fcas_price(start, end, region="NSW1", service="RAISEREG")
        >>> print(fcas.head())
    """
    if region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")
    if service not in FCAS_SERVICES:
        raise ValueError(f"Service must be one of {FCAS_SERVICES}")
    
    if not HAS_NEMOSIS:
        raise ImportError(
            "NEMOSIS is required to fetch actual AEMO data. "
            "Install with: pip install nemosis"
        )
    
    cache_path = get_cache_dir(cache_dir)
    
    # Format datetime for NEMOSIS
    start_time = start_date.strftime('%Y/%m/%d %H:%M:%S')
    end_time = end_date.strftime('%Y/%m/%d %H:%M:%S')
    
    print(f"Fetching FCAS {service} price data for {region} from {start_date.date()} to {end_date.date()}...")
    
    try:
        # Fetch DISPATCHPRICE which contains FCAS prices
        price_data = _dynamic_data_compiler_with_cache_control(
            start_time=start_time,
            end_time=end_time,
            table_name='DISPATCHPRICE',
            raw_data_location=str(cache_path)
        )

        price_pl = _normalize_columns(_as_polars(price_data))
        if price_pl.height > 0:
            if 'REGIONID' in price_pl.columns:
                price_pl = price_pl.filter(pl.col('REGIONID') == region)
            price_pl = _coerce_datetime(price_pl, 'SETTLEMENTDATE')
            
            # Map service name to column name in DISPATCHPRICE table
            service_column_map = {
                'RAISE6SEC': 'RAISE6SECRRP',
                'RAISE60SEC': 'RAISE60SECRRP',
                'RAISE5MIN': 'RAISE5MINRRP',
                'RAISEREG': 'RAISEREGRRP',
                'LOWER6SEC': 'LOWER6SECRRP',
                'LOWER60SEC': 'LOWER60SECRRP',
                'LOWER5MIN': 'LOWER5MINRRP',
                'LOWERREG': 'LOWERREGRRP',
            }
            
            price_col = service_column_map.get(service)
            if price_col and price_col in price_pl.columns:
                out = price_pl.select(['SETTLEMENTDATE', 'REGIONID', price_col]).with_columns([
                    pl.lit(service).alias('SERVICE'),
                    pl.col(price_col).cast(pl.Float64, strict=False).alias('PRICE'),
                ]).select(['SETTLEMENTDATE', 'REGIONID', 'SERVICE', 'PRICE']).sort('SETTLEMENTDATE')

                print(f"Fetched {len(out)} FCAS price records")
                return out
            else:
                print(f"Warning: Price column for service {service} not found in DISPATCHPRICE data")
                return pl.DataFrame(schema={
                    'SETTLEMENTDATE': pl.Datetime,
                    'REGIONID': pl.Utf8,
                    'SERVICE': pl.Utf8,
                    'PRICE': pl.Float64
                })
        else:
            print("No data returned from NEMOSIS")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'SERVICE': pl.Utf8,
                'PRICE': pl.Float64
            })
            
    except Exception as e:
        print(f"Error fetching FCAS data from NEMOSIS: {e}")
        print("Note: NEMOSIS requires internet connection to download data from AEMO")
        raise


def fetch_aemo_unit_dispatch(
    start_date: datetime,
    end_date: datetime,
    duid: Optional[str] = None,
    duids: Optional[List[str]] = None,
    region: Optional[str] = None,
    generator_info_path: Optional[str] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Fetch unit-specific dispatch data from AEMO DISPATCHLOAD table.

    This function provides detailed dispatch information for individual units (DUIDs),
    including energy dispatch targets and FCAS enablement across all services. This is
    essential for calculating actual FCAS revenue and understanding operational constraints.

    The DISPATCHLOAD table contains the dispatch targets set by AEMO for each unit in
    each 5-minute interval, including:
    - Energy dispatch (TOTALCLEARED)
    - FCAS enablement for each service (raise/lower, regulation/contingency)
    - Availability and bid data

    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        duid: Single Dispatch Unit ID to fetch (e.g., "LBBG1").
        duids: List of Dispatch Unit IDs to fetch.  Takes precedence over ``region``
            when both are specified.  More memory-efficient than ``region`` for large
            date windows because the per-month filter is applied immediately.
        region: Filter by AEMO region (NSW1, QLD1, SA1, TAS1, VIC1).  Only used when
            neither ``duid`` nor ``duids`` is provided.
        generator_info_path: Optional path to a local NEM registration spreadsheet.
        cache_dir: Directory to cache downloaded data
        refresh: If True, re-download even if cached data exists

    Returns:
        Polars DataFrame with columns:
            - SETTLEMENTDATE: Datetime of the dispatch interval
            - DUID: Dispatch Unit ID
            - TOTALCLEARED: Total energy dispatch target (MW)
            - RAISE6SEC / RAISE60SEC / RAISE5MIN / RAISEREG: FCAS raise enablement (MW)
            - LOWER6SEC / LOWER60SEC / LOWER5MIN / LOWERREG: FCAS lower enablement (MW)
            - AVAILABILITY: Unit availability (MW)
            - INITIALMW: Initial MW before dispatch
            - RAISEREGENABLEMENTMAX / RAISEREGENABLEMENTMIN: Regulation raise range
            - LOWERREGENABLEMENTMAX / LOWERREGENABLEMENTMIN: Regulation lower range
            - SEMIDISPATCHCAP: Semi-dispatch capacity flag
            - RAMPUPRATE / RAMPDOWNRATE: Ramp rate limits (MW/min)
            - AGCSTATUS: AGC control status

    Memory note:
        DISPATCHLOAD contains all NEM units (~1 000–2 000 DUIDs, ~850 K rows/month).
        When fetching a long date range without a ``duid``/``duids`` filter the raw
        data for each month is kept in memory briefly while the filter runs.  For
        windows longer than ~4 weeks, always provide ``duid`` or ``duids`` to keep
        per-chunk memory usage low.

    Example:
        >>> # Fetch dispatch data for a specific battery unit
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> dispatch = fetch_aemo_unit_dispatch(start, end, duid="LBBG1")
        >>> print(dispatch.head())
    """
    if region and region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")

    if not HAS_NEMOSIS:
        raise ImportError(
            "NEMOSIS is required to fetch actual AEMO data. "
            "Install with: pip install nemosis"
        )

    cache_path = get_cache_dir(cache_dir)

    # Build the per-chunk DUID filter set BEFORE the loop so we can trim each
    # raw month immediately (avoids accumulating ~850 K rows × N months).
    duid_filter: Optional[set] = None
    if duid:
        duid_filter = {str(duid).strip()}
    elif duids:
        duid_filter = {str(d).strip() for d in duids if d}
    elif region:
        # Get region DUIDs from the static table once, before the download loop
        try:
            gen_info_pre = _get_generators_static_table(cache_path, refresh)
        except Exception:
            gen_info_pre = _get_generator_info(
                cache_path, generator_info_path=generator_info_path
            )
        if gen_info_pre is not None:
            gen_info_pl_pre = _normalize_columns(_as_polars(gen_info_pre))
            if 'Region' in gen_info_pl_pre.columns and 'DUID' in gen_info_pl_pre.columns:
                duid_filter = set(
                    gen_info_pl_pre.filter(pl.col('Region') == region)
                    .select('DUID')
                    .unique()['DUID']
                    .to_list()
                )
                print(
                    f"Pre-filtering to {len(duid_filter)} DUIDs in region {region}"
                )
            else:
                print(
                    "Warning: Could not load generator static info to filter by region. "
                    "Provide `generator_info_path` (or set env var AEMO_GENERATORS_FILE)."
                )

    duid_str = f" for DUID {duid}" if duid else (f" for {len(duid_filter)} DUIDs" if duid_filter else "")
    region_str = f" in {region}" if region else ""
    print(
        f"Fetching unit dispatch data{duid_str}{region_str} "
        f"from {start_date.date()} to {end_date.date()}..."
    )

    columns_to_keep = ['SETTLEMENTDATE', 'DUID', 'TOTALCLEARED',
                       'RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN', 'RAISEREG',
                       'LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN', 'LOWERREG',
                       'AVAILABILITY', 'INITIALMW',
                       'RAISEREGENABLEMENTMAX', 'RAISEREGENABLEMENTMIN',
                       'LOWERREGENABLEMENTMAX', 'LOWERREGENABLEMENTMIN',
                       'SEMIDISPATCHCAP', 'RAMPUPRATE', 'RAMPDOWNRATE',
                       'AGCSTATUS']

    dispatch_frames: list[pl.DataFrame] = []
    windows = _iter_month_windows(start_date, end_date)

    try:
        for idx, (window_start, window_end) in enumerate(windows, start=1):
            window_start_time = window_start.strftime('%Y/%m/%d %H:%M:%S')
            window_end_time = window_end.strftime('%Y/%m/%d %H:%M:%S')
            if len(windows) > 1:
                print(
                    f"  DISPATCHLOAD chunk {idx}/{len(windows)}: "
                    f"{window_start.date()} to {window_end.date()}"
                )

            chunk_raw = _dynamic_data_compiler_with_cache_control(
                start_time=window_start_time,
                end_time=window_end_time,
                table_name='DISPATCHLOAD',
                raw_data_location=str(cache_path),
            )
            chunk_pl = _normalize_columns(_as_polars(chunk_raw))
            del chunk_raw  # free pandas memory immediately

            if chunk_pl.height == 0:
                continue

            # Apply DUID filter immediately to keep per-chunk memory low
            if duid_filter is not None and 'DUID' in chunk_pl.columns:
                chunk_pl = chunk_pl.filter(pl.col('DUID').is_in(duid_filter))
                if chunk_pl.height == 0:
                    continue

            chunk_pl = _coerce_datetime(chunk_pl, 'SETTLEMENTDATE')
            available_columns = list(
                dict.fromkeys(col for col in columns_to_keep if col in chunk_pl.columns)
            )
            if not available_columns:
                continue
            dispatch_frames.append(chunk_pl.select(available_columns))

    except Exception as e:
        print(f"Error fetching unit dispatch data from NEMOSIS: {e}")
        print("Note: NEMOSIS requires internet connection to download data from AEMO")
        raise

    if not dispatch_frames:
        print("No dispatch data returned from NEMOSIS")
        if duid:
            print(f"Hint: Check if DUID '{duid}' exists in DISPATCHLOAD table for this date range")
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime, 'DUID': pl.Utf8, 'TOTALCLEARED': pl.Float64,
            'RAISE6SEC': pl.Float64, 'RAISE60SEC': pl.Float64, 'RAISE5MIN': pl.Float64,
            'RAISEREG': pl.Float64, 'LOWER6SEC': pl.Float64, 'LOWER60SEC': pl.Float64,
            'LOWER5MIN': pl.Float64, 'LOWERREG': pl.Float64,
            'AVAILABILITY': pl.Float64, 'INITIALMW': pl.Float64,
            'RAISEREGENABLEMENTMAX': pl.Float64, 'RAISEREGENABLEMENTMIN': pl.Float64,
            'LOWERREGENABLEMENTMAX': pl.Float64, 'LOWERREGENABLEMENTMIN': pl.Float64,
            'SEMIDISPATCHCAP': pl.Float64, 'RAMPUPRATE': pl.Float64,
            'RAMPDOWNRATE': pl.Float64, 'AGCSTATUS': pl.Utf8,
        })

    dispatch_pl = pl.concat(dispatch_frames, how='vertical_relaxed')
    dispatch_pl = dispatch_pl.unique(subset=['SETTLEMENTDATE', 'DUID'], keep='last').sort('SETTLEMENTDATE')

    # Cast numeric columns
    available_columns = list(
        dict.fromkeys(col for col in columns_to_keep if col in dispatch_pl.columns)
    )
    select_exprs = []
    for c in available_columns:
        if c in ('SETTLEMENTDATE', 'DUID'):
            select_exprs.append(pl.col(c))
        else:
            select_exprs.append(pl.col(c).cast(pl.Float64, strict=False))
    out = dispatch_pl.select(select_exprs).sort('SETTLEMENTDATE')
    print(f"Fetched {len(out)} dispatch records for {out['DUID'].n_unique()} unique units")
    return out


def check_aemo_dispatch_availability(
    start_date: datetime,
    end_date: datetime,
    duid: str,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> Dict[str, Any]:
    """Return a lightweight availability summary for a DUID over a date range."""
    if not duid:
        raise ValueError("duid must be provided")

    if not HAS_NEMOSIS:
        raise ImportError(
            "NEMOSIS is required to fetch actual AEMO data. "
            "Install with: pip install nemosis"
        )

    cache_path = get_cache_dir(cache_dir)
    windows = _iter_month_windows(start_date, end_date)
    total_rows = 0
    unique_intervals = 0
    months_with_data = 0
    first_settlement = None
    last_settlement = None

    expected_intervals = max(
        0,
        int((end_date - start_date).total_seconds() // (5 * 60))
    )

    for window_start, window_end in windows:
        chunk = _dynamic_data_compiler_with_cache_control(
            start_time=window_start.strftime('%Y/%m/%d %H:%M:%S'),
            end_time=window_end.strftime('%Y/%m/%d %H:%M:%S'),
            table_name='DISPATCHLOAD',
            raw_data_location=str(cache_path),
        )
        chunk_pl = _normalize_columns(_as_polars(chunk))
        if chunk_pl.height == 0 or 'DUID' not in chunk_pl.columns:
            continue

        filtered = chunk_pl.filter(pl.col('DUID') == duid)
        if filtered.height == 0:
            continue

        filtered = _coerce_datetime(filtered, 'SETTLEMENTDATE')
        if 'SETTLEMENTDATE' not in filtered.columns:
            continue

        months_with_data += 1
        total_rows += filtered.height
        settlement_dates = filtered['SETTLEMENTDATE']
        unique_intervals += settlement_dates.n_unique()

        chunk_first = settlement_dates.min()
        chunk_last = settlement_dates.max()
        if first_settlement is None or (chunk_first is not None and chunk_first < first_settlement):
            first_settlement = chunk_first
        if last_settlement is None or (chunk_last is not None and chunk_last > last_settlement):
            last_settlement = chunk_last

    has_data = total_rows > 0
    coverage_ratio = (
        float(unique_intervals) / float(expected_intervals)
        if expected_intervals > 0 else (1.0 if has_data else 0.0)
    )

    return {
        'duid': duid,
        'start_date': start_date,
        'end_date': end_date,
        'has_data': has_data,
        'row_count': int(total_rows),
        'unique_intervals': int(unique_intervals),
        'expected_intervals': int(expected_intervals),
        'coverage_ratio': float(coverage_ratio),
        'months_checked': len(windows),
        'months_with_data': int(months_with_data),
        'first_settlement': first_settlement,
        'last_settlement': last_settlement,
    }


def find_duid_first_dispatch(
    duid: str,
    search_start: datetime,
    search_end: Optional[datetime] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
    verbose: bool = True,
) -> Optional[datetime]:
    """Find the earliest date a DUID appears in the AEMO DISPATCHLOAD table.

    Scans DISPATCHLOAD month-by-month forward from *search_start* until the
    DUID is found or *search_end* is reached.  Stops at the first month that
    contains at least one row for *duid*.  Use this to discover when a battery
    was first commissioned or when its current DUID became active.

    .. note::
        The current ``get_available_battery_units`` static table only lists
        *current* AEMO registrations.  Many batteries have been re-registered
        under new **Bidirectional Unit** DUIDs (e.g. ``HPR1``, ``LBB1``) after
        previously operating as separate **Generating Unit** + **Load** pairs
        (e.g. ``HPRG1``/``HPRL1``).  The new DUIDs appear in DISPATCHLOAD only
        from the date of re-registration.  If this function returns a date in
        2022–2025 for a battery that has been operating since 2017–2020, the
        battery almost certainly has historical data under its older DUID pair.

    Args:
        duid: Dispatch Unit ID to search for.
        search_start: Earliest date to scan (e.g. ``datetime(2018, 1, 1)``).
        search_end: Latest date to scan (defaults to today).
        cache_dir: Path to the NEMOSIS data cache directory.
        refresh: When ``True``, force re-download of cached files.
        verbose: Print progress messages.

    Returns:
        The earliest ``datetime`` with at least one DISPATCHLOAD record for
        *duid*, or ``None`` if no records were found in the search range.
    """
    if not HAS_NEMOSIS:
        raise ImportError(
            "NEMOSIS is required to fetch actual AEMO data. "
            "Install with: pip install nemosis"
        )

    if search_end is None:
        search_end = datetime.now()

    cache_path = get_cache_dir(cache_dir)
    windows = _iter_month_windows(search_start, search_end)

    if verbose:
        print(
            f"Scanning {len(windows)} month(s) for DUID {duid!r}: "
            f"{search_start.date()} → {search_end.date()}"
        )

    for window_start, window_end in windows:
        if verbose:
            print(f"  Checking {window_start.date()} …", end=" ", flush=True)
        try:
            chunk = _dynamic_data_compiler_with_cache_control(
                start_time=window_start.strftime('%Y/%m/%d %H:%M:%S'),
                end_time=window_end.strftime('%Y/%m/%d %H:%M:%S'),
                table_name='DISPATCHLOAD',
                raw_data_location=str(cache_path),
            )
            chunk_pl = _normalize_columns(_as_polars(chunk))
            if chunk_pl.height == 0 or 'DUID' not in chunk_pl.columns:
                if verbose:
                    print("no data")
                continue

            filtered = chunk_pl.filter(pl.col('DUID') == duid)
            if filtered.height == 0:
                if verbose:
                    print("not found")
                continue

            filtered = _coerce_datetime(filtered, 'SETTLEMENTDATE')
            first_dt = filtered['SETTLEMENTDATE'].min()
            if verbose:
                print(f"FOUND — first interval: {first_dt}")
            return first_dt
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"error ({exc})")
            continue

    if verbose:
        print(f"DUID {duid!r} not found in search range.")
    return None


def get_dispatch_active_battery_units(
    start_date: datetime,
    end_date: datetime,
    region: Optional[str] = None,
    cache_dir: str = "data/aemo",
    generator_info_path: Optional[str] = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Return battery DUIDs from the static table that appear in DISPATCHLOAD for the date window.

    Returned columns include activity statistics so callers can distinguish
    batteries that were actually dispatched from those with only zero-valued records:

      - All columns from ``get_available_battery_units``
      - ``DispatchIntervalCount`` – number of unique 5-min intervals with any record
      - ``NonZeroIntervalCount`` – intervals where at least one dispatch column is non-zero
      - ``MaxEnergyMW`` – peak absolute TOTALCLEARED across all intervals
      - ``FirstDispatchInterval`` / ``LastDispatchInterval`` – timestamp range
      - ``PairedGenDUID`` / ``PairedLoadDUID`` – populated for old-model batteries
        registered as separate "Generating Unit" + "Load" DUIDs at the same station

    Rows are sorted by ``NonZeroIntervalCount`` descending so the most-active
    batteries appear first.  If all found batteries have ``NonZeroIntervalCount=0``
    (i.e. none were dispatched during the window), a warning is printed and the
    rows are still returned with zero activity stats so callers can surface this
    information to the user.
    """
    battery_units = get_available_battery_units(
        cache_dir=cache_dir,
        generator_info_path=generator_info_path,
        region=region,
        refresh=refresh,
    )
    if battery_units.height == 0 or 'DUID' not in battery_units.columns:
        return battery_units

    # Pass the battery DUID list so fetch_aemo_unit_dispatch can filter per-chunk.
    # This is critical for long date windows: without a DUID filter the full
    # DISPATCHLOAD (~850 K rows/month) is kept in memory for every month.
    battery_duid_list = battery_units['DUID'].drop_nulls().to_list()

    dispatch_df = fetch_aemo_unit_dispatch(
        start_date=start_date,
        end_date=end_date,
        duids=battery_duid_list,
        generator_info_path=generator_info_path,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    if dispatch_df.height == 0 or 'DUID' not in dispatch_df.columns:
        return battery_units.head(0)

    # Compute per-DUID activity statistics
    dispatch_numeric_cols = [c for c in dispatch_df.columns if c not in {'SETTLEMENTDATE', 'DUID'}]

    # Build a per-row "any column is non-zero" flag
    activity_df = dispatch_df.with_columns(
        pl.col('SETTLEMENTDATE').cast(pl.Datetime, strict=False)
    )
    if dispatch_numeric_cols:
        nonzero_condition: Optional[pl.Expr] = None
        for col in dispatch_numeric_cols:
            expr = pl.col(col).cast(pl.Float64, strict=False).abs() > DISPATCH_NONZERO_THRESHOLD
            nonzero_condition = expr if nonzero_condition is None else (nonzero_condition | expr)
        activity_df = activity_df.with_columns(
            pl.when(nonzero_condition).then(1).otherwise(0).alias('_has_activity')
        )
        tc_expr = pl.col('TOTALCLEARED').cast(pl.Float64, strict=False).abs() if 'TOTALCLEARED' in dispatch_numeric_cols else pl.lit(0.0)
        dispatch_summary = (
            activity_df
            .group_by('DUID')
            .agg([
                pl.len().alias('DispatchRowCount'),
                pl.col('SETTLEMENTDATE').n_unique().alias('DispatchIntervalCount'),
                pl.col('SETTLEMENTDATE').min().alias('FirstDispatchInterval'),
                pl.col('SETTLEMENTDATE').max().alias('LastDispatchInterval'),
                pl.col('_has_activity').sum().cast(pl.UInt32).alias('NonZeroIntervalCount'),
                tc_expr.max().alias('MaxEnergyMW'),
            ])
        )
    else:
        dispatch_summary = (
            activity_df
            .group_by('DUID')
            .agg([
                pl.len().alias('DispatchRowCount'),
                pl.col('SETTLEMENTDATE').n_unique().alias('DispatchIntervalCount'),
                pl.col('SETTLEMENTDATE').min().alias('FirstDispatchInterval'),
                pl.col('SETTLEMENTDATE').max().alias('LastDispatchInterval'),
                pl.lit(0, dtype=pl.UInt32).alias('NonZeroIntervalCount'),
                pl.lit(None, dtype=pl.Float64).alias('MaxEnergyMW'),
            ])
        )

    result = battery_units.join(dispatch_summary, on='DUID', how='inner')

    # ------------------------------------------------------------------
    # Detect paired gen/load DUID batteries (old AEMO registration model).
    # A "Generating Unit" DUID and a "Load" DUID form a logical battery
    # pair.  We expose PairedGenDUID / PairedLoadDUID so callers can pass
    # both to set_dispatch_data() for a correct net-energy replay.
    #
    # Pairing strategy (safest first):
    #   1. If a StationName column exists, pair within (StationName, Region)
    #      to get exactly one gen+one load per physical station.
    #   2. Otherwise, pair within Region – but only if there is exactly
    #      one gen DUID and one load DUID in that region, to avoid creating
    #      a spurious Cartesian product when there are multiple gen or load
    #      DUIDs in the same region.
    # ------------------------------------------------------------------
    if 'DispatchType' in result.columns:
        gen_duids = result.filter(
            pl.col('DispatchType').cast(pl.Utf8, strict=False).str.to_lowercase().str.contains('generating')
        )
        load_duids = result.filter(
            pl.col('DispatchType').cast(pl.Utf8, strict=False).str.to_lowercase().str.contains('load')
        )

        gen_load_pairs: Optional[pl.DataFrame] = None
        if gen_duids.height > 0 and load_duids.height > 0:
            station_col = 'StationName' if 'StationName' in result.columns else None
            if station_col:
                # Most precise: pair within the same station and region
                gen_sel = gen_duids.select(['DUID', 'Region', station_col]).rename({'DUID': 'GenDUID'})
                load_sel = load_duids.select(['DUID', 'Region', station_col]).rename({'DUID': 'LoadDUID'})
                gen_load_pairs = gen_sel.join(load_sel, on=['Region', station_col], how='inner').select(['GenDUID', 'LoadDUID', 'Region'])
            else:
                # Fallback: pair by region, but only when there is exactly
                # one gen and one load per region to avoid Cartesian products.
                gen_per_region = gen_duids.group_by('Region').agg(pl.count().alias('_n_gen'))
                load_per_region = load_duids.group_by('Region').agg(pl.count().alias('_n_load'))
                safe_regions = (
                    gen_per_region
                    .join(load_per_region, on='Region', how='inner')
                    .filter((pl.col('_n_gen') == 1) & (pl.col('_n_load') == 1))
                    .select('Region')
                )
                if safe_regions.height > 0:
                    gen_sel = gen_duids.select(['DUID', 'Region']).rename({'DUID': 'GenDUID'})
                    load_sel = load_duids.select(['DUID', 'Region']).rename({'DUID': 'LoadDUID'})
                    gen_load_pairs = (
                        gen_sel
                        .join(load_sel, on='Region', how='inner')
                        .join(safe_regions, on='Region', how='inner')
                        .select(['GenDUID', 'LoadDUID', 'Region'])
                    )

        if gen_load_pairs is not None and gen_load_pairs.height > 0:
            # Add PairedLoadDUID column to gen rows
            result = result.join(
                gen_load_pairs.select(['GenDUID', 'LoadDUID']).rename({'GenDUID': 'DUID', 'LoadDUID': 'PairedLoadDUID'}),
                on='DUID', how='left',
            )
            # Add PairedGenDUID column to load rows
            result = result.join(
                gen_load_pairs.select(['LoadDUID', 'GenDUID']).rename({'LoadDUID': 'DUID', 'GenDUID': 'PairedGenDUID'}),
                on='DUID', how='left',
            )
        else:
            result = result.with_columns([
                pl.lit(None, dtype=pl.Utf8).alias('PairedGenDUID'),
                pl.lit(None, dtype=pl.Utf8).alias('PairedLoadDUID'),
            ])
    else:
        result = result.with_columns([
            pl.lit(None, dtype=pl.Utf8).alias('PairedGenDUID'),
            pl.lit(None, dtype=pl.Utf8).alias('PairedLoadDUID'),
        ])

    # Sort: batteries with actual non-zero dispatch come first
    result = result.sort(
        ['NonZeroIntervalCount', 'DispatchIntervalCount', 'MaxEnergyMW'],
        descending=[True, True, True],
        nulls_last=True,
    )

    # Warn if no battery was actually dispatched during this window
    if result['NonZeroIntervalCount'].max() == 0:
        import warnings as _warnings
        _warnings.warn(
            f"All battery DUIDs found in DISPATCHLOAD for {'region ' + region + ' ' if region else ''}"
            f"{start_date.date()} to {end_date.date()} have zero dispatch values "
            f"(TOTALCLEARED=0, all FCAS=0).\n"
            f"Found DUIDs: {result['DUID'].to_list()}\n"
            "The dispatch replay will produce no actions for this period.\n"
            "Possible fixes:\n"
            "  1. Try a different (wider) date range when the battery was actively dispatched.\n"
            "  2. Try a different region (e.g. SA1 has Hornsdale Power Reserve which is often active).\n"
            "  3. Use a rule-based or RL agent instead of the dispatch replay agent.",
            stacklevel=2,
        )

    return result


def _get_generators_static_table(cache_path: Path, refresh: bool):
    """
    Helper to robustly load the 'Generators and Scheduled Loads' static table.
    Tries once with the requested refresh flag, and on Excel-format errors
    tries to convert any cached .xls files to .xlsx using xls2xlsx, then retries
    the read. Raises a helpful error if still failing.
    
    Also detects if AEMO's server returned an HTML error page instead of the Excel file.
    """
    static_cache_path = _get_nemosis_static_cache_dir(cache_path)

    # Helper to check if a file is an AEMO error payload instead of a spreadsheet.
    def _is_aemo_error_file(file_path):
        """Check if file contains an AEMO error page/message instead of Excel data."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024)
                content_lower = content.lower()
                has_error_text = (
                    b'sorry' in content_lower
                    and b'failed' in content_lower
                ) or b'please return to the home page' in content_lower
                is_html_error = (
                    (b'<html' in content_lower or b'<!doctype' in content_lower)
                    and (b'sorry' in content_lower or b'failed' in content_lower or b'error' in content_lower)
                )
                has_excel_signature = (
                    content.startswith(b'PK')
                    or content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1')
                )
                if (has_error_text or is_html_error) and not has_excel_signature:
                    return True
        except Exception:
            pass
        return False
    
    # Check if any existing cached files are HTML error pages
    try:
        cache_files = list(static_cache_path.glob('*'))
        for cache_file in cache_files:
            if cache_file.suffix.lower() in ['.xls', '.xlsx'] and _is_aemo_error_file(cache_file):
                print(f"Detected corrupted/error file in cache: {cache_file.name}")
                print(f"Deleting corrupted file and forcing re-download...")
                cache_file.unlink()
                refresh = True  # Force refresh if we deleted a corrupted file
    except Exception as e:
        print(f"Warning: Could not check cache for corrupted files: {e}")
    
    try:
        return static_table(
            table_name='Generators and Scheduled Loads',
            raw_data_location=str(static_cache_path),
            update_static_file=bool(refresh),
            select_columns='all',
        )
    except Exception as excel_error:
        msg = str(excel_error)
        if 'Excel file format cannot be determined' in msg:
            # Inspect cache for .xls files and attempt conversion to .xlsx
            try:
                files = sorted([p.name for p in static_cache_path.iterdir()])
            except Exception:
                files = []

            xls_files = [f for f in files if f.lower().endswith('.xls')]
            if xls_files:
                converted = []
                conversion_errors = []
                try:
                    from xls2xlsx import XLS2XLSX
                except Exception:
                    raise ImportError(
                        "Found .xls cached files in the NEMOSIS cache but 'xls2xlsx' is not installed.\n"
                        "Install it in the Jupyter environment, e.g.:\n"
                        "  docker compose exec app python3 -m pip install xls2xlsx\n"
                        "Then restart the notebook kernel (or reload this module) and retry.\n\n"
                        f"Cache directory: {static_cache_path}\n"
                        f"Cache dir listing (first 50 entries): {files[:50]}"
                    ) from excel_error

                for f in xls_files:
                    xls_path = static_cache_path / f
                    xlsx_name = xls_path.stem + '.xlsx'
                    xlsx_path = static_cache_path / xlsx_name
                    # Skip conversion if xlsx already exists
                    if xlsx_path.exists():
                        converted.append(str(xlsx_path.name))
                        continue
                    try:
                        x2x = XLS2XLSX(str(xls_path))
                        x2x.to_xlsx(str(xlsx_path))
                        converted.append(str(xlsx_path.name))
                    except Exception as conv_err:
                        conversion_errors.append((str(xls_path.name), str(conv_err)))

                # If we converted at least one file, try reading again
                if converted:
                    try:
                        return static_table(
                            table_name='Generators and Scheduled Loads',
                            raw_data_location=str(static_cache_path),
                            update_static_file=False,
                            select_columns='all',
                        )
                    except Exception:
                        # fall through to forced refresh below
                        pass

                # If conversions failed and no success, raise informative error
                if conversion_errors and not converted:
                    raise ValueError(
                        "Failed to convert cached .xls files to .xlsx.\n"
                        "Conversion errors: " + ", ".join([f"{n}: {e}" for n, e in conversion_errors]) + "\n\n"
                        f"Cache directory: {static_cache_path}\n"
                        f"Cache dir listing (first 50 entries): {files[:50]}\n\n"
                        "Fix: Install 'xls2xlsx' or delete the cache directory and retry (rm -rf data/aemo),"
                    ) from excel_error

            # Try a forced refresh (redownload) in case cached file is corrupt or was HTML
            print("Attempting forced refresh to re-download static table...")
            try:
                result = static_table(
                    table_name='Generators and Scheduled Loads',
                    raw_data_location=str(static_cache_path),
                    update_static_file=True,
                    select_columns='all',
                )
                # Check if the newly downloaded file is an HTML error
                try:
                    files = sorted([p.name for p in static_cache_path.iterdir()])
                    for f in files:
                        file_path = static_cache_path / f
                        if file_path.suffix.lower() in ['.xls', '.xlsx']:
                            if _is_aemo_error_file(file_path):
                                raise ValueError(
                                    "AEMO server returned an error payload instead of the Excel file. "
                                    "The server may be experiencing issues or rate-limiting requests.\n\n"
                                    "The downloaded file contains: 'Sorry, your request has failed...'\n\n"
                                    "Fixes:\n"
                                    " - Wait a few minutes and try again (AEMO server may be temporarily unavailable)\n"
                                    " - Delete the corrupted cache file manually:\n"
                                    f"     rm {file_path}\n"
                                    " - Delete the entire cache directory and retry:\n"
                                    f"     rm -rf {static_cache_path}\n"
                                    " - Try using a different date range or region to reduce load on AEMO servers\n"
                                )
                except Exception:
                    pass  # Continue with the result if check failed
                return result
            except Exception as excel_error2:
                # Provide helpful diagnostics including cache listing
                try:
                    files = sorted([p.name for p in static_cache_path.iterdir()])
                except Exception:
                    files = []
                raise ValueError(
                    "Failed to read NEMOSIS static table 'Generators and Scheduled Loads'. "
                    "Pandas could not determine the Excel format for the cached file. "
                    "This usually means:\n"
                    " 1. AEMO's server returned an HTML error page instead of the Excel file\n"
                    "    (error message: 'Sorry, your request has failed...')\n"
                    " 2. The cached file is corrupt, empty, or has wrong extension\n"
                    " 3. The file format is .xls (old Excel) instead of .xlsx\n\n"
                    f"Cache directory: {static_cache_path}\n"
                    f"Cache dir listing (first 50 entries): {files[:50]}\n\n"
                    "Fixes:\n"
                    " - Wait a few minutes and try again (AEMO server may be temporarily unavailable)\n"
                    " - Delete the cache directory and retry:\n"
                    f"     rm -rf {static_cache_path}\n"
                    " - If files end with .xls, install 'xls2xlsx' in the Jupyter env:\n"
                    "     pip install xls2xlsx\n"
                    " - Call fetch_aemo_generation_by_fuel(..., refresh=True) to force re-download\n"
                ) from excel_error2
        # Re-raise other exceptions
        raise


def _find_column_by_candidates(
    columns: List[str],
    exact_candidates: List[str],
    keyword_groups: Optional[List[List[str]]] = None,
) -> Optional[str]:
    """Find a best-effort matching column name using exact names first, then keyword groups."""
    cols_map = {c.lower(): c for c in columns}

    for cand in exact_candidates:
        if cand in cols_map:
            return cols_map[cand]

    if keyword_groups:
        for group in keyword_groups:
            for c in columns:
                lc = c.lower()
                if all(k in lc for k in group):
                    return c

    return None


def _numeric_from_mixed_text_expr(col_name: str) -> pl.Expr:
    """Parse floats from mixed text columns (e.g. '100 MW', '1,200.5')."""
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.replace_all(',', '')
        .str.extract(r'(-?\d+(?:\.\d+)?)', 1)
        .cast(pl.Float64, strict=False)
    )


def get_available_battery_units(
    cache_dir: str = "data/aemo",
    generator_info_path: Optional[str] = None,
    region: Optional[str] = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Return available battery units from the static table with enriched metadata.

    Output columns (when available in the source static table):
      - DUID
      - Region
      - DispatchType (e.g. "Bidirectional Unit", "Generating Unit", "Load")
      - TechnologyType
      - FuelType
      - StorageCapacityMWh
      - RegisteredCapacityMW

    ``DispatchType`` is important for paired gen/load batteries registered under
    the old AEMO model: a "Generating Unit" DUID handles discharge and its
    corresponding "Load" DUID handles charging.  Most modern batteries are
    "Bidirectional Unit" (single DUID for both directions).

    Args:
        cache_dir: path to cache directory used by NEMOSIS (default: data/aemo)
        generator_info_path: optional local XLS/XLSX/CSV file to use instead of downloading
        region: optional region filter (NSW1/QLD1/SA1/TAS1/VIC1)
        refresh: force re-download of static table when True

    Returns:
        Polars DataFrame (possibly empty) containing battery units and metadata.
    """
    if region and region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")

    cache_path = get_cache_dir(cache_dir)

    try:
        gen_info = _get_generators_static_table(cache_path, refresh)
    except Exception:
        gen_info = _get_generator_info(cache_path, generator_info_path=generator_info_path)

    if gen_info is None:
        return pl.DataFrame()

    gen_info_pl = _normalize_columns(_as_polars(gen_info))
    if gen_info_pl.height == 0:
        return pl.DataFrame()

    columns = gen_info_pl.columns
    duid_col = _find_column_by_candidates(
        columns,
        exact_candidates=["duid"],
        keyword_groups=[["duid"]],
    )
    if duid_col is None:
        return pl.DataFrame()

    region_col = _find_column_by_candidates(
        columns,
        exact_candidates=["region", "regionid"],
        keyword_groups=[["region"]],
    )
    tech_col = _find_column_by_candidates(
        columns,
        exact_candidates=[
            "technology type - descriptor",
            "technology type - primary",
            "technology type",
            "technology",
            "tech type",
        ],
        keyword_groups=[["technology"], ["tech", "type"]],
    )
    fuel_col = _find_column_by_candidates(
        columns,
        exact_candidates=[
            "fuel source - descriptor",
            "fuel source - primary",
            "fuel source",
            "fuel",
        ],
        keyword_groups=[["fuel"]],
    )
    dispatch_type_col = _find_column_by_candidates(
        columns,
        exact_candidates=["dispatch type", "dispatchtype"],
        keyword_groups=[["dispatch", "type"]],
    )

    storage_mwh_col = _find_column_by_candidates(
        columns,
        exact_candidates=[
            "storage capacity (mwh)",
            "storage capacity mwh",
            "storage_mwh",
            "energy capacity (mwh)",
            "energy capacity mwh",
            "max storage (mwh)",
            "battery storage (mwh)",
            # AEMO NEM Registration and Exemption List actual column name
            "maximum storage capacity",
            "storage capacity",
        ],
        keyword_groups=[["storage", "mwh"], ["energy", "mwh"], ["capacity", "mwh"], ["storage", "capacity"]],
    )
    reg_cap_mw_col = _find_column_by_candidates(
        columns,
        exact_candidates=[
            "reg cap (mw)",
            "registered capacity (mw)",
            "registered capacity mw",
            "nameplate capacity (mw)",
            "max capacity (mw)",
            "capacity (mw)",
            "maxcap mw",
            # AEMO NEM Registration and Exemption List actual column names
            "reg cap generation (mw)",
            "max cap generation (mw)",
        ],
        keyword_groups=[["reg", "cap", "mw"], ["registered", "capacity"], ["capacity", "mw"]],
    )

    # Battery identification mask from technology/fuel columns
    battery_mask = None
    for col in [tech_col, fuel_col]:
        if col is None:
            continue
        cond = (
            pl.col(col)
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .str.contains("battery")
        )
        battery_mask = cond if battery_mask is None else (battery_mask | cond)

    # Final fallback: scan all text-like columns for "battery"
    if battery_mask is None:
        for c in columns:
            cond = (
                pl.col(c)
                .cast(pl.Utf8, strict=False)
                .str.to_lowercase()
                .str.contains("battery")
            )
            battery_mask = cond if battery_mask is None else (battery_mask | cond)

    if battery_mask is None:
        return pl.DataFrame()

    batteries = gen_info_pl.filter(battery_mask)
    if batteries.height == 0:
        return pl.DataFrame()

    select_exprs = [pl.col(duid_col).alias("DUID")]
    if region_col:
        select_exprs.append(pl.col(region_col).alias("Region"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("Region"))

    if dispatch_type_col:
        select_exprs.append(pl.col(dispatch_type_col).cast(pl.Utf8, strict=False).str.strip_chars().alias("DispatchType"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("DispatchType"))

    if tech_col:
        select_exprs.append(pl.col(tech_col).cast(pl.Utf8, strict=False).alias("TechnologyType"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("TechnologyType"))

    if fuel_col:
        select_exprs.append(pl.col(fuel_col).cast(pl.Utf8, strict=False).alias("FuelType"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("FuelType"))

    if storage_mwh_col:
        select_exprs.append(_numeric_from_mixed_text_expr(storage_mwh_col).alias("StorageCapacityMWh"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Float64).alias("StorageCapacityMWh"))

    if reg_cap_mw_col:
        select_exprs.append(_numeric_from_mixed_text_expr(reg_cap_mw_col).alias("RegisteredCapacityMW"))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Float64).alias("RegisteredCapacityMW"))

    out = (
        batteries
        .select(select_exprs)
        .with_columns(pl.col("DUID").cast(pl.Utf8, strict=False).str.strip_chars())
        .filter(pl.col("DUID").is_not_null() & (pl.col("DUID") != ""))
        .unique(subset=["DUID"], keep="first")
    )

    if region:
        out = out.filter(pl.col("Region") == region)

    return out.sort(["Region", "DUID"])

def get_available_battery_duids(
    cache_dir: str = "data/aemo",
    generator_info_path: Optional[str] = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Return available battery DUIDs (and their Region if available).

    This helper loads the 'Generators and Scheduled Loads' static table (via the
    robust loader `_get_generators_static_table` with fallbacks) and returns a
    Polars DataFrame with columns:
      - DUID
      - Region (if present in the static table)

    Args:
        cache_dir: path to cache directory used by NEMOSIS (default: data/aemo)
        generator_info_path: optional local XLS/XLSX/CSV file to use instead of downloading
        refresh: force re-download of the static table when True

    Returns:
        Polars DataFrame (possibly empty) with unique battery DUIDs.
    """
    out = get_available_battery_units(
        cache_dir=cache_dir,
        generator_info_path=generator_info_path,
        region=None,
        refresh=refresh,
    )
    if out.height == 0:
        return out
    return out.select(["DUID", "Region"])

def fetch_aemo_generation_by_fuel(
    start_date: datetime,
    end_date: datetime,
    region: str = "NSW1",
    fuel_types: Optional[List[str]] = None,
    generator_info_path: Optional[str] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Fetch AEMO generation data by fuel type (solar, wind, coal, gas, etc.).
    
    This provides insight into the generation mix and renewable penetration,
    which affects price volatility and trading strategies.
    
    Note: This function uses DISPATCH_UNIT_SCADA data and joins with static 
    generator information to aggregate generation by fuel type.
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        region: AEMO region code (NSW1, QLD1, SA1, TAS1, VIC1)
        fuel_types: List of fuel types to fetch (default: ["solar", "wind"])
        cache_dir: Directory to cache downloaded data
        refresh: If True, re-download even if cached data exists
        
    Returns:
        Polars DataFrame with columns:
            - SETTLEMENTDATE: Datetime of the interval
            - REGIONID: Region identifier
            - FUEL_TYPE: Type of generation fuel
            - GENERATION: Power output (MW)
            
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> gen = fetch_aemo_generation_by_fuel(start, end, region="NSW1", fuel_types=["solar", "wind"])
        >>> print(gen.head())
    """
    if region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")
    
    if fuel_types is None:
        fuel_types = ["solar", "wind"]
    
    if not HAS_NEMOSIS:
        raise ImportError(
            "NEMOSIS is required to fetch actual AEMO data. "
            "Install with: pip install nemosis"
        )
    
    cache_path = get_cache_dir(cache_dir)
    
    # Format datetime for NEMOSIS
    start_time = start_date.strftime('%Y/%m/%d %H:%M:%S')
    end_time = end_date.strftime('%Y/%m/%d %H:%M:%S')
    
    print(f"Fetching generation data for {fuel_types} in {region} from {start_date.date()} to {end_date.date()}...")
    
    try:
        # First, get static generator information to map DUIDs to fuel types
        try:
            gen_info = _get_generators_static_table(cache_path, refresh)
        except Exception:
            # If robust static-table loading fails (HTML/corrupt cache, conversion issues),
            # fall back to a local file if available (AEMO_GENERATORS_FILE or src/data/aemo autodetect).
            gen_info = _get_generator_info(cache_path, generator_info_path=generator_info_path)
        
        if gen_info is None or len(gen_info) == 0:
            print(
                "Warning: Could not load generator static info (DUID/Region/Fuel). "
                "Provide `generator_info_path` (or set env var AEMO_GENERATORS_FILE)."
            )
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })
        
        gen_info_pl = _normalize_columns(_as_polars(gen_info))

        # Filter generators by region
        if 'Region' not in gen_info_pl.columns or 'DUID' not in gen_info_pl.columns:
            print(
                "Warning: Generator info file is missing required columns (need 'DUID' and 'Region')."
            )
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })

        gen_info_pl = gen_info_pl.filter(pl.col('Region') == region)
        
        # Map fuel source descriptors to our fuel types
        fuel_mapping = {
            'solar': ['Solar'],
            'wind': ['Wind'],
            'coal_black': ['Black Coal'],
            'coal_brown': ['Brown Coal'],
            'gas_ccgt': ['Natural Gas / Fuel Oil', 'Natural Gas (Pipeline)'],
            'gas_ocgt': ['Natural Gas / Fuel Oil'],
            'gas_recip': ['Natural Gas / Fuel Oil'],
            'hydro': ['Hydro', 'Water'],
            'battery_discharging': ['Battery']
        }
        
        # Get SCADA data for generation
        scada_data = _dynamic_data_compiler_with_cache_control(
            start_time=start_time,
            end_time=end_time,
            table_name='DISPATCH_UNIT_SCADA',
            raw_data_location=str(cache_path)
        )

        scada_pl = _normalize_columns(_as_polars(scada_data))
        if scada_pl.height == 0:
            print("Warning: No SCADA data returned")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })

        scada_pl = _coerce_datetime(scada_pl, 'SETTLEMENTDATE')
        scada_pl = _coerce_f64(scada_pl, 'SCADAVALUE')
        
        # Merge SCADA data with generator info
        fuel_col = 'Fuel Source - Descriptor'
        if fuel_col not in gen_info_pl.columns:
            print(
                "Warning: Generator info is missing 'Fuel Source - Descriptor' column; "
                "cannot aggregate by fuel type."
            )
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })

        merged = scada_pl.join(
            gen_info_pl.select(['DUID', fuel_col]),
            on='DUID',
            how='left'
        )

        fuel_expr = None
        for ft in fuel_types:
            sources = fuel_mapping.get(ft)
            if not sources:
                continue
            cond = pl.col(fuel_col).is_in(sources)
            if fuel_expr is None:
                fuel_expr = pl.when(cond).then(pl.lit(ft))
            else:
                fuel_expr = fuel_expr.when(cond).then(pl.lit(ft))

        if fuel_expr is None:
            print("No generation data found for specified fuel types")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })

        merged = merged.with_columns(fuel_expr.otherwise(None).alias('FUEL_TYPE'))
        filtered = merged.filter(pl.col('FUEL_TYPE').is_not_null())
        if filtered.height == 0:
            print("No generation data found for specified fuel types")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })

        out = (
            filtered
            .group_by(['SETTLEMENTDATE', 'FUEL_TYPE'])
            .agg(pl.col('SCADAVALUE').sum().alias('GENERATION'))
            .with_columns(pl.lit(region).alias('REGIONID'))
            .select(['SETTLEMENTDATE', 'REGIONID', 'FUEL_TYPE', 'GENERATION'])
            .sort(['SETTLEMENTDATE', 'FUEL_TYPE'])
        )

        print(f"Fetched {len(out)} generation records")
        return out
            
    except Exception as e:
        print(f"Error fetching generation data from NEMOSIS: {e}")
        print("Note: NEMOSIS requires internet connection to download data from AEMO")
        raise


def fetch_aemo_data_bundle(
    start_date: datetime,
    end_date: datetime,
    region: str = "NSW1",
    fcas_services: Optional[List[str]] = None,
    fuel_types: Optional[List[str]] = None,
    generator_info_path: Optional[str] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    Convenience function to fetch multiple AEMO datasets at once.
    
    Returns a dictionary with keys: 'prices', 'fcas', 'generation'
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        region: AEMO region code
        fcas_services: List of FCAS services to fetch (default: ["RAISEREG", "LOWERREG"])
        fuel_types: List of fuel types to fetch (default: ["solar", "wind"])
        cache_dir: Directory to cache downloaded data
        refresh: If True, re-download even if cached data exists
        
    Returns:
        Dictionary containing:
            - 'prices': Energy price DataFrame
            - 'fcas': FCAS price DataFrame (combined services)
            - 'generation': Generation DataFrame (combined fuel types)
            
    Example:
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> data = fetch_aemo_data_bundle(start, end, region="NSW1")
        >>> print(data['prices'].head())
        >>> print(data['fcas'].head())
        >>> print(data['generation'].head())
    """
    if fcas_services is None:
        fcas_services = ["RAISEREG", "LOWERREG"]
    if fuel_types is None:
        fuel_types = ["solar", "wind"]
    
    print(f"Fetching AEMO data bundle for {region} from {start_date.date()} to {end_date.date()}...")
    
    # Fetch energy prices
    prices = fetch_aemo_dispatch_price(start_date, end_date, region, cache_dir, refresh)
    
    # Fetch FCAS prices for all requested services
    fcas_dfs = []
    for service in fcas_services:
        fcas_df = fetch_aemo_fcas_price(start_date, end_date, region, service, cache_dir, refresh)
        fcas_dfs.append(fcas_df)
    fcas = pl.concat(fcas_dfs)
    
    # Fetch generation data
    generation = fetch_aemo_generation_by_fuel(
        start_date, end_date, region, fuel_types, generator_info_path, cache_dir, refresh
    )
    
    return {
        'prices': prices,
        'fcas': fcas,
        'generation': generation,
    }


def fetch_aemo_data_bundle_with_dispatch(
    start_date: datetime,
    end_date: datetime,
    region: str = "NSW1",
    duid: Optional[str] = None,
    fcas_services: Optional[List[str]] = None,
    fuel_types: Optional[List[str]] = None,
    generator_info_path: Optional[str] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    Comprehensive function to fetch AEMO market data including unit-specific dispatch.
    
    This function fetches regional market prices, regional generation mix, and
    unit-specific dispatch/enablement data in one call. Use this for analyzing
    individual unit performance and FCAS participation.
    
    Returns a dictionary with keys: 'prices', 'fcas', 'generation', 'unit_dispatch'
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        region: AEMO region code
        duid: Specific Dispatch Unit ID to fetch (optional). If None, fetches all units in region
        fcas_services: List of FCAS services to fetch (default: ["RAISEREG", "LOWERREG"])
        fuel_types: List of fuel types to fetch (default: ["solar", "wind"])
        cache_dir: Directory to cache downloaded data
        refresh: If True, re-download even if cached data exists
        
    Returns:
        Dictionary containing:
            - 'prices': Energy price DataFrame (regional)
            - 'fcas': FCAS price DataFrame (regional, combined services)
            - 'generation': Generation DataFrame (regional, combined fuel types)
            - 'unit_dispatch': Unit-specific dispatch and FCAS enablement DataFrame
            
    Example:
        >>> # Fetch data for a specific battery unit
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> data = fetch_aemo_data_bundle_with_dispatch(
        ...     start, end, region="NSW1", duid="LBBG1",
        ...     fcas_services=["RAISEREG", "LOWERREG"]
        ... )
        >>> print(data['prices'].head())
        >>> print(data['unit_dispatch'].head())
        >>> 
        >>> # Calculate FCAS revenue for the unit
        >>> # Join unit_dispatch with fcas prices on SETTLEMENTDATE and SERVICE
    """
    if fcas_services is None:
        fcas_services = ["RAISEREG", "LOWERREG"]
    if fuel_types is None:
        fuel_types = ["solar", "wind"]
    
    duid_str = f" for DUID {duid}" if duid else ""
    print(f"Fetching comprehensive AEMO data bundle for {region}{duid_str} from {start_date.date()} to {end_date.date()}...")
    
    # Fetch energy prices
    prices = fetch_aemo_dispatch_price(start_date, end_date, region, cache_dir, refresh)
    
    # Fetch FCAS prices for all requested services
    fcas_dfs = []
    for service in fcas_services:
        fcas_df = fetch_aemo_fcas_price(start_date, end_date, region, service, cache_dir, refresh)
        fcas_dfs.append(fcas_df)
    fcas = pl.concat(fcas_dfs) if fcas_dfs else pl.DataFrame()
    
    # Fetch generation data
    generation = fetch_aemo_generation_by_fuel(
        start_date, end_date, region, fuel_types, generator_info_path, cache_dir, refresh
    )
    
    # Fetch unit-specific dispatch data (use keyword args to avoid position mismatch)
    unit_dispatch = fetch_aemo_unit_dispatch(
        start_date, end_date,
        duid=duid, region=region,
        generator_info_path=generator_info_path,
        cache_dir=cache_dir, refresh=refresh,
    )
    
    return {
        'prices': prices,
        'fcas': fcas,
        'generation': generation,
        'unit_dispatch': unit_dispatch,
    }


def aggregate_fcas_market_depth(
    region: str,
    start_date: datetime,
    end_date: datetime,
    demand_series: Optional[pl.DataFrame] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Aggregate FCAS market depth (total cleared MW per service) per 5-min interval.

    Strategy (auto-detected):
      1.  First tries to sum per-unit cleared enablement from DISPATCHLOAD
          (columns RAISE6SEC … LOWERREG).  If any are > 0 across the date range,
          those sums are the depth.
      2.  If DISPATCHLOAD enablement is uniformly zero (common for regions such as
          SA1 that import FCAS via interconnectors), falls back to demand-ratio
          heuristics based on TOTALDEMAND, provided via *demand_series*.

    Args:
        region: AEMO region code (NSW1, QLD1, SA1, TAS1, VIC1).
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.
        demand_series: Optional DataFrame with ``SETTLEMENTDATE``, ``TOTALDEMAND``
            columns.  Required when DISPATCHLOAD enablement is zero (fallback path).
        cache_dir: Directory to cache downloaded data.
        refresh: If True, re-download even if cached data exists.

    Returns:
        Polars DataFrame with ``SETTLEMENTDATE`` and ``FCAS_DEPTH_{SERVICE}_MW``
        columns for all 8 services.

    Notes on APPROXIMATE heuristics (fallback path):
    ::

          Regulation      = max(30 MW, 0.03 × TOTALDEMAND)
          Contingency 6s  = max(50 MW, 0.10 × TOTALDEMAND)
          Contingency 60s  = max(30 MW, 0.05 × TOTALDEMAND)
          Contingency 5min = max(20 MW, 0.02 × TOTALDEMAND)

        These are rough per-region minimums intended to give the impact model
        a realistic denominator.  Replace with MMSDM aggregate-table data
        when per-service precision matters (Phase 3+).
    """
    region = region.upper()
    if region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")

    fcas_services = ['RAISE6SEC', 'RAISE60SEC', 'RAISE5MIN', 'RAISEREG',
                     'LOWER6SEC', 'LOWER60SEC', 'LOWER5MIN', 'LOWERREG']

    # -------- 1. Try DISPATCHLOAD per-unit enablement first ----------
    dispatch = fetch_aemo_unit_dispatch(
        start_date=start_date, end_date=end_date,
        region=region, duid=None, duids=None,
        cache_dir=cache_dir, refresh=refresh,
    )
    if dispatch.height > 0:
        # Check if at least one FCAS-enablement column has meaningful nonzero values.
        cols_present = [s for s in fcas_services if s in dispatch.columns]
        if cols_present:
            max_mean = max(dispatch[c].mean() for c in cols_present)
            if max_mean > 0.01:
                agg = {s: pl.col(s).sum().alias(f'FCAS_DEPTH_{s}_MW')
                       for s in cols_present}
                return (
                    dispatch
                    .group_by('SETTLEMENTDATE')
                    .agg(*agg.values())
                    .sort('SETTLEMENTDATE')
                )

    # -------- 2. Fallback: demand-ratio heuristics ----------
    # Heuristic ratios derived from typical NEM FCAS requirements per region.
    # Per-service depth = max(min_mw, ratio * TOTALDEMAND).
    HEURISTIC: dict[str, tuple[float, float]] = {
        'RAISE6SEC':   (50.0, 0.10),
        'RAISE60SEC':  (30.0, 0.05),
        'RAISE5MIN':   (15.0, 0.02),
        'RAISEREG':    (30.0, 0.03),
        'LOWER6SEC':   (50.0, 0.10),
        'LOWER60SEC':  (30.0, 0.05),
        'LOWER5MIN':   (15.0, 0.02),
        'LOWERREG':    (30.0, 0.03),
    }

    if demand_series is None or 'TOTALDEMAND' not in demand_series.columns:
        print("Warning: No demand_series provided (TOTALDEMAND required). "
              "Returning empty FCAS depth.")
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime,
            **{f'FCAS_DEPTH_{s}_MW': pl.Float64 for s in fcas_services},
        })

    dem = demand_series.select('SETTLEMENTDATE', 'TOTALDEMAND').drop_nulls().sort('SETTLEMENTDATE')
    if dem.height == 0:
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime,
            **{f'FCAS_DEPTH_{s}_MW': pl.Float64 for s in fcas_services},
        })

    expr = [pl.col('SETTLEMENTDATE')]
    for srv, (min_mw, ratio) in HEURISTIC.items():
        colname = f'FCAS_DEPTH_{srv}_MW'
        expr.append(
            pl.max_horizontal(
                pl.lit(min_mw),
                pl.col('TOTALDEMAND') * ratio
            ).alias(colname)
        )
    result = dem.select(expr)
    print("FCAS depth: using TOTALDEMAND heuristic (DISPATCHLOAD enablement was zero for this region)")
    return result


def aggregate_residual_supply(
    region: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Compute total available generation (MW) per 5-min interval in a region,
    calculated as sum(AVAILABILITY) across all DISPATCHLOAD DUIDs.

    The consumer (e.g. AEMODataPreprocessor or impact model) can subtract
    TOTALDEMAND from the returned AVAILABILITY_SUM to get residual supply.

    Args:
        region: AEMO region code.
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.
        cache_dir: Directory to cache downloaded data.
        refresh: If True, re-download even if cached data exists.

    Returns:
        Polars DataFrame with SETTLEMENTDATE and AVAILABILITY_SUM_MW.
    """
    region = region.upper()
    if region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")

    dispatch = fetch_aemo_unit_dispatch(
        start_date=start_date, end_date=end_date,
        region=region, duid=None, duids=None,
        cache_dir=cache_dir, refresh=refresh,
    )
    if dispatch.height == 0 or 'AVAILABILITY' not in dispatch.columns:
        print("Warning: No AVAILABILITY data in DISPATCHLOAD result")
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime,
            'AVAILABILITY_SUM_MW': pl.Float64,
        })
    return (
        dispatch
        .group_by('SETTLEMENTDATE')
        .agg(pl.col('AVAILABILITY').sum().alias('AVAILABILITY_SUM_MW'))
        .sort('SETTLEMENTDATE')
    )


# Marginal-cost tier labels for NEM fuel types (AUD/MWh, approximate order).
# These determine the position in the merit-order supply curve.
FUEL_MARGINAL_COST_TIERS: dict[str, float] = {
    'solar': 0.0,
    'wind': 0.0,
    'hydro': 5.0,
    'battery_discharging': 10.0,
    'coal_brown': 15.0,
    'coal_black': 30.0,
    'gas_ccgt': 70.0,
    'gas_recip': 120.0,
    'gas_ocgt': 180.0,
    'gas_steam': 200.0,
    'diesel': 300.0,
    'other': 999.0,
}

# Inverse mapping from NEM "Fuel Source - Descriptor" strings to our keys.
_FUEL_SOURCE_TO_KEY: dict[str, str] = {
    'Solar': 'solar',
    'Wind': 'wind',
    'Water': 'hydro',
    'Hydro': 'hydro',
    'Battery': 'battery_discharging',
    'Brown Coal': 'coal_brown',
    'Black Coal': 'coal_black',
    'Natural Gas / Fuel Oil': 'gas_ocgt',
    'Natural Gas (Pipeline)': 'gas_ccgt',
    'Gas / Oil': 'gas_recip',
    'Gas / Diesel': 'gas_recip',
    'Diesel / Oil': 'diesel',
    'Diesel': 'diesel',
    'Other': 'other',
}


def _infer_marginal_cost(
    gen_info_df: pl.DataFrame,
    fuel_col: str = 'Fuel Source - Descriptor',
) -> pl.DataFrame:
    """Add MARGINAL_COST column to a DataFrame of static generator info."""
    if fuel_col not in gen_info_df.columns:
        return gen_info_df.with_columns(pl.lit(999.0).alias('MARGINAL_COST'))
    return gen_info_df.with_columns(
        pl.col(fuel_col)
        .replace_strict(_FUEL_SOURCE_TO_KEY, default='other')
        .replace_strict(FUEL_MARGINAL_COST_TIERS, default=999.0)
        .alias('MARGINAL_COST')
    )


def build_supply_curve(
    region: str,
    start_date: datetime,
    end_date: datetime,
    generator_info_path: Optional[str] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """
    Build a merit-order supply curve for a region over a date range.

    For each 5-min DISPATCHLOAD interval, generators are sorted by inferred
    marginal cost (from fuel type) and accumulated available MW.  The result
    is a DataFrame where every (SETTLEMENTDATE, row-index) is a step on the
    supply ladder: price tier and cumulative MW at that tier.

    Args:
        region: AEMO region code.
        start_date: Start date for data retrieval.
        end_date: End date for data retrieval.
        generator_info_path: Optional path to NEM registration spreadsheet.
        cache_dir: Directory to cache downloaded data.
        refresh: If True, re-download even if cached data exists.

    Returns:
        Polars DataFrame with columns:
            - SETTLEMENTDATE: Datetime of the dispatch interval
            - MARGINAL_COST: $/MWh cost tier for this supply step
            - AVAILABILITY_MW: MW available at this tier (per-interval)
            - CUMULATIVE_MW: Cumulative MW up to this tier
    """
    region = region.upper()
    if region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")

    cache_path = get_cache_dir(cache_dir)

    # 1. Load static generator info for fuel-type → marginal cost mapping.
    try:
        gen_info = _get_generators_static_table(cache_path, refresh)
    except Exception:
        gen_info = _get_generator_info(cache_path, generator_info_path=generator_info_path)

    if gen_info is None or len(gen_info) == 0:
        print("Warning: Could not load generator static info; supply curve cannot be built.")
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime,
            'MARGINAL_COST': pl.Float64,
            'AVAILABILITY_MW': pl.Float64,
            'CUMULATIVE_MW': pl.Float64,
        })

    gen_pl = _normalize_columns(_as_polars(gen_info))
    fuel_col = 'Fuel Source - Descriptor'
    if 'DUID' not in gen_pl.columns or 'Region' not in gen_pl.columns:
        print("Warning: Generator info missing 'DUID' or 'Region' columns.")
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime,
            'MARGINAL_COST': pl.Float64,
            'AVAILABILITY_MW': pl.Float64,
            'CUMULATIVE_MW': pl.Float64,
        })

    # Filter to region, then infer marginal cost on the subset.
    region_gen = gen_pl.filter(pl.col('Region') == region)
    duid_costs = (
        _infer_marginal_cost(region_gen, fuel_col)
        .select(['DUID', 'MARGINAL_COST'])
        .unique(subset=['DUID'])
    )
    if duid_costs.height == 0:
        print(f"Warning: No generators found for region {region}")
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime,
            'MARGINAL_COST': pl.Float64,
            'AVAILABILITY_MW': pl.Float64,
            'CUMULATIVE_MW': pl.Float64,
        })

    # 2. Fetch DISPATCHLOAD with AVAILABILITY for the region.
    dispatch = fetch_aemo_unit_dispatch(
        start_date=start_date, end_date=end_date,
        region=region, duid=None, duids=None,
        cache_dir=cache_dir, refresh=refresh,
    )
    if dispatch.height == 0 or 'AVAILABILITY' not in dispatch.columns:
        print("Warning: No AVAILABILITY data in DISPATCHLOAD result; supply curve empty.")
        return pl.DataFrame(schema={
            'SETTLEMENTDATE': pl.Datetime,
            'MARGINAL_COST': pl.Float64,
            'AVAILABILITY_MW': pl.Float64,
            'CUMULATIVE_MW': pl.Float64,
        })

    # 3. Join DUID → marginal cost, group by interval + cost tier, sum MW.
    supply = (
        dispatch
        .join(duid_costs, on='DUID', how='inner')
        .group_by(['SETTLEMENTDATE', 'MARGINAL_COST'])
        .agg(pl.col('AVAILABILITY').sum().alias('AVAILABILITY_MW'))
        .sort(['SETTLEMENTDATE', 'MARGINAL_COST'])
    )
    # Cumulative sum per interval.
    supply = supply.with_columns(
        pl.col('AVAILABILITY_MW')
        .cum_sum()
        .over('SETTLEMENTDATE')
        .alias('CUMULATIVE_MW')
    )
    print(f"Built supply curve: {supply.shape[0]} rows, "
          f"{supply['SETTLEMENTDATE'].n_unique()} intervals")
    return supply


def example_fetch_short_range() -> Dict[str, pl.DataFrame]:
    """
    Example function demonstrating how to fetch a short date range of AEMO data.
    
    This fetches actual AEMO market data using NEMOSIS library. The data is downloaded
    from AEMO's NEMWEB archives and cached locally for future use.
    
    Note: This function requires internet connectivity and may take a few minutes on
    first run as NEMOSIS downloads data from AEMO servers.
    
    For better reliability, use historical data from at least 2-3 months ago, as recent
    data may still be in preliminary status or not yet archived.
    
    Returns:
        Dictionary with 'prices', 'fcas', and 'generation' DataFrames containing
        actual AEMO market data
        
    Example:
        >>> data = example_fetch_short_range()
        >>> print(f"Fetched {len(data['prices'])} price records")
        >>> print(f"Fetched {len(data['fcas'])} FCAS records")
        >>> print(f"Fetched {len(data['generation'])} generation records")
    """
    # Use historical data from several months ago for better reliability
    # AEMO archives data progressively, so older data is more stable
    end_date = datetime(2023, 6, 1, 12, 0, 0)  # Use a specific historical date
    start_date = datetime(2023, 6, 1, 0, 0, 0)  # 12 hours of data
    
    print("=" * 60)
    print("AEMO Data Fetch Example (using NEMOSIS)")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Region: NSW1")
    print()
    print("NOTE: This downloads ACTUAL data from AEMO via NEMOSIS")
    print("      First run may take 1-2 minutes to download and cache")
    print("      Subsequent runs will use cached data (much faster)")
    print("=" * 60)
    
    try:
        data = fetch_aemo_data_bundle(
            start_date=start_date,
            end_date=end_date,
            region="NSW1",
            fcas_services=["RAISEREG", "LOWERREG"],
            fuel_types=["solar", "wind"],
        )
        
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Energy prices: {len(data['prices'])} records")
        print(f"  FCAS prices: {len(data['fcas'])} records")
        print(f"  Generation: {len(data['generation'])} records")
        print("=" * 60)
        
        return data
    except Exception as e:
        print(f"\nError fetching data: {e}")
        print("\nPossible causes:")
        print("  - No internet connectivity")
        print("  - AEMO servers temporarily unavailable")
        print("  - Requested data not yet archived")
        print("\nTip: Try using a date range from 2-3 months ago for better reliability")
        raise


if __name__ == "__main__":
    # Run example when module is executed directly
    data = example_fetch_short_range()
    print("\nSample energy prices:")
    print(data['prices'].head())
    print("\nSample FCAS prices:")
    print(data['fcas'].head())
    print("\nSample generation data:")
    print(data['generation'].head())
