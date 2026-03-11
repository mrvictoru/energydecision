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
import os
import warnings


# Import NEMOSIS for actual AEMO data fetching
try:
    from nemosis import dynamic_data_compiler, static_table
    HAS_NEMOSIS = True
except ImportError:
    HAS_NEMOSIS = False
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


def _read_generator_info_file(file_path: Path) -> pl.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _normalize_columns(pl.read_csv(file_path))

    # Support reading old .xls by converting to .xlsx first (xls2xlsx is in requirements).
    if suffix == ".xls":
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
        # Prefer Polars' Excel reader if available; otherwise use openpyxl.
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
    """Best-effort lookup for a locally downloaded generator info XLS/XLSX/CSV."""
    candidates: List[Path] = []
    repo_root = Path(__file__).resolve().parent.parent
    for p in [repo_root / "src/data/aemo", repo_root / "data/aemo"]:
        if p.exists() and p.is_dir():
            for ext in ("*.xls", "*.xlsx", "*.csv"):
                candidates.extend(sorted(p.glob(ext)))

    if len(candidates) == 1:
        return candidates[0]
    return None


def _get_generator_info(cache_path: Path, generator_info_path: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Retrieve generator static info (DUID -> Region/Fuel descriptor).

    Tries NEMOSIS `static_table()` first; if that fails (e.g. AEMO blocks downloads),
    falls back to a user-provided local XLS/XLSX/CSV.
    """
    # 1) Try NEMOSIS static_table (preferred when it works)
    try:
        gen_info = static_table(
            table_name='Generators and Scheduled Loads',
            raw_data_location=str(cache_path),
            update_static_file=False
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
        price_data = dynamic_data_compiler(
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
            demand_data = dynamic_data_compiler(
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
        price_data = dynamic_data_compiler(
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
        duid: Specific Dispatch Unit ID to fetch (e.g., "LBBG1"). If None, fetches all units
        region: Filter by AEMO region (NSW1, QLD1, SA1, TAS1, VIC1). If None, fetches all regions
        cache_dir: Directory to cache downloaded data
        refresh: If True, re-download even if cached data exists
        
    Returns:
        Polars DataFrame with columns:
            - SETTLEMENTDATE: Datetime of the dispatch interval
            - DUID: Dispatch Unit ID
            - TOTALCLEARED: Total energy dispatch target (MW)
            - RAISE6SEC: Enabled capacity for 6-second raise (MW)
            - RAISE60SEC: Enabled capacity for 60-second raise (MW)
            - RAISE5MIN: Enabled capacity for 5-minute raise (MW)
            - RAISEREG: Enabled capacity for regulation raise (MW)
            - LOWER6SEC: Enabled capacity for 6-second lower (MW)
            - LOWER60SEC: Enabled capacity for 60-second lower (MW)
            - LOWER5MIN: Enabled capacity for 5-minute lower (MW)
            - LOWERREG: Enabled capacity for regulation lower (MW)
            - Additional columns: AVAILABILITY, RAMPUPRATE, RAMPDOWNRATE, etc.
            
    Example:
        >>> # Fetch dispatch data for a specific battery unit
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> dispatch = fetch_aemo_unit_dispatch(start, end, duid="LBBG1")
        >>> print(dispatch.head())
        >>> 
        >>> # Calculate FCAS revenue for a unit
        >>> # Revenue = Enablement (MW) * Price ($/MW/h) * interval duration (hours)
    """
    if region and region not in AEMO_REGIONS:
        raise ValueError(f"Region must be one of {AEMO_REGIONS}")
    
    if not HAS_NEMOSIS:
        raise ImportError(
            "NEMOSIS is required to fetch actual AEMO data. "
            "Install with: pip install nemosis"
        )
    
    cache_path = get_cache_dir(cache_dir)
    
    # Format datetime for NEMOSIS
    start_time = start_date.strftime('%Y/%m/%d %H:%M:%S')
    end_time = end_date.strftime('%Y/%m/%d %H:%M:%S')
    
    duid_str = f" for DUID {duid}" if duid else ""
    region_str = f" in {region}" if region else ""
    print(f"Fetching unit dispatch data{duid_str}{region_str} from {start_date.date()} to {end_date.date()}...")
    
    try:
        columns_to_keep = ['SETTLEMENTDATE', 'DUID', 'TOTALCLEARED']

        fcas_columns = [
            'RAISE6SEC',
            'RAISE60SEC',
            'RAISE5MIN',
            'RAISEREG',
            'LOWER6SEC',
            'LOWER60SEC',
            'LOWER5MIN',
            'LOWERREG',
        ]
        columns_to_keep.extend(fcas_columns)

        dispatch_frames: list[pl.DataFrame] = []
        windows = _iter_month_windows(start_date, end_date)
        for idx, (window_start, window_end) in enumerate(windows, start=1):
            window_start_time = window_start.strftime('%Y/%m/%d %H:%M:%S')
            window_end_time = window_end.strftime('%Y/%m/%d %H:%M:%S')
            if len(windows) > 1:
                print(
                    f"  DISPATCHLOAD chunk {idx}/{len(windows)}: "
                    f"{window_start.date()} to {window_end.date()}"
                )

            chunk = dynamic_data_compiler(
                start_time=window_start_time,
                end_time=window_end_time,
                table_name='DISPATCHLOAD',
                raw_data_location=str(cache_path)
            )
            chunk_pl = _normalize_columns(_as_polars(chunk))
            if chunk_pl.height == 0:
                continue

            if duid and 'DUID' in chunk_pl.columns:
                before_filter = chunk_pl.height
                chunk_pl = chunk_pl.filter(pl.col('DUID') == duid)
                print(
                    f"    DUID filter: {before_filter} records before filter, "
                    f"{chunk_pl.height} after filtering for '{duid}'"
                )
                if chunk_pl.height == 0:
                    continue

            chunk_pl = _coerce_datetime(chunk_pl, 'SETTLEMENTDATE')
            available_columns = list(dict.fromkeys(col for col in columns_to_keep if col in chunk_pl.columns))
            if not available_columns:
                continue
            dispatch_frames.append(chunk_pl.select(available_columns))

        if not dispatch_frames:
            print("No dispatch data returned from NEMOSIS")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'DUID': pl.Utf8,
                'TOTALCLEARED': pl.Float64,
                'RAISE6SEC': pl.Float64,
                'RAISE60SEC': pl.Float64,
                'RAISE5MIN': pl.Float64,
                'RAISEREG': pl.Float64,
                'LOWER6SEC': pl.Float64,
                'LOWER60SEC': pl.Float64,
                'LOWER5MIN': pl.Float64,
                'LOWERREG': pl.Float64,
            })
        dispatch_pl = pl.concat(dispatch_frames, how='vertical_relaxed')

        if duid and dispatch_pl.height == 0:
            print(f"Warning: No data found for DUID {duid}")
            print(f"Hint: Check if DUID '{duid}' exists in DISPATCHLOAD table for this date range")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'DUID': pl.Utf8,
                'TOTALCLEARED': pl.Float64,
                'RAISE6SEC': pl.Float64,
                'RAISE60SEC': pl.Float64,
                'RAISE5MIN': pl.Float64,
                'RAISEREG': pl.Float64,
                'LOWER6SEC': pl.Float64,
                'LOWER60SEC': pl.Float64,
                'LOWER5MIN': pl.Float64,
                'LOWERREG': pl.Float64,
            })

        dispatch_pl = dispatch_pl.unique(subset=['SETTLEMENTDATE', 'DUID'], keep='last').sort('SETTLEMENTDATE')
        
        # Filter by region if specified (requires generator static info)
        # NOTE: Only filter by region if DUID was NOT specified (DUID already implies region)
        if region and not duid:
            # Prefer the robust static-table reader which attempts conversions and forced refreshes
            gen_info = None
            try:
                gen_info = _get_generators_static_table(cache_path, refresh)
            except Exception:
                # Fall back to local file or env var when the static_table path fails
                gen_info = _get_generator_info(cache_path, generator_info_path=generator_info_path)

            gen_info_pl = _normalize_columns(_as_polars(gen_info)) if gen_info is not None else None

            if gen_info_pl is not None and 'Region' in gen_info_pl.columns and 'DUID' in gen_info_pl.columns:
                region_duids = gen_info_pl.filter(pl.col('Region') == region).select('DUID').unique()
                before_region_filter = dispatch_pl.height
                if region_duids.height > 0:
                    dispatch_pl = dispatch_pl.join(region_duids, on='DUID', how='inner')
                after_region_filter = dispatch_pl.height
                print(f"Region filter: {before_region_filter} records before filter, {after_region_filter} after filtering for region '{region}'")
            else:
                print(
                    "Warning: Could not load generator static info to filter by region. "
                    "Provide `generator_info_path` (or set env var AEMO_GENERATORS_FILE)."
                )

        # Select only the columns that exist and cast numerics
        available_columns = list(dict.fromkeys(col for col in columns_to_keep if col in dispatch_pl.columns))
        select_exprs = []
        for c in available_columns:
            if c == 'SETTLEMENTDATE' or c == 'DUID':
                select_exprs.append(pl.col(c))
            else:
                select_exprs.append(pl.col(c).cast(pl.Float64, strict=False))

        out = dispatch_pl.select(select_exprs).sort('SETTLEMENTDATE')
        print(f"Fetched {len(out)} dispatch records for {out['DUID'].n_unique()} unique units")
        return out
        
    except Exception as e:
        print(f"Error fetching unit dispatch data from NEMOSIS: {e}")
        print("Note: NEMOSIS requires internet connection to download data from AEMO")
        raise


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
        chunk = dynamic_data_compiler(
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


def get_dispatch_active_battery_units(
    start_date: datetime,
    end_date: datetime,
    region: Optional[str] = None,
    cache_dir: str = "data/aemo",
    generator_info_path: Optional[str] = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Return battery DUIDs from the static table that also appear in DISPATCHLOAD for the date window."""
    battery_units = get_available_battery_units(
        cache_dir=cache_dir,
        generator_info_path=generator_info_path,
        region=region,
        refresh=refresh,
    )
    if battery_units.height == 0 or 'DUID' not in battery_units.columns:
        return battery_units

    dispatch_df = fetch_aemo_unit_dispatch(
        start_date=start_date,
        end_date=end_date,
        region=region,
        generator_info_path=generator_info_path,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    if dispatch_df.height == 0 or 'DUID' not in dispatch_df.columns:
        return battery_units.head(0)

    dispatch_summary = (
        dispatch_df
        .with_columns(pl.col('SETTLEMENTDATE').cast(pl.Datetime, strict=False))
        .group_by('DUID')
        .agg([
            pl.len().alias('DispatchRowCount'),
            pl.col('SETTLEMENTDATE').n_unique().alias('DispatchIntervalCount'),
            pl.col('SETTLEMENTDATE').min().alias('FirstDispatchInterval'),
            pl.col('SETTLEMENTDATE').max().alias('LastDispatchInterval'),
        ])
    )

    return (
        battery_units
        .join(dispatch_summary, on='DUID', how='inner')
        .sort(['DispatchIntervalCount', 'DispatchRowCount'], descending=[True, True])
    )


def _get_generators_static_table(cache_path: Path, refresh: bool):
    """
    Helper to robustly load the 'Generators and Scheduled Loads' static table.
    Tries once with the requested refresh flag, and on Excel-format errors
    tries to convert any cached .xls files to .xlsx using xls2xlsx, then retries
    the read. Raises a helpful error if still failing.
    
    Also detects if AEMO's server returned an HTML error page instead of the Excel file.
    """
    # Helper to check if a file is an HTML error page from AEMO
    def _is_html_error_file(file_path):
        """Check if file contains AEMO's HTML error message."""
        try:
            with open(file_path, 'rb') as f:
                # Read first 1KB to check for HTML markers
                content = f.read(1024)
                content_lower = content.lower()
                # Check for HTML tags and AEMO error message
                if (b'<html' in content_lower or b'<!doctype' in content_lower) and \
                   (b'sorry' in content_lower or b'failed' in content_lower or b'error' in content_lower):
                    return True
        except Exception:
            pass
        return False
    
    # Check if any existing cached files are HTML error pages
    try:
        cache_files = list(cache_path.glob('*'))
        for cache_file in cache_files:
            if cache_file.suffix.lower() in ['.xls', '.xlsx'] and _is_html_error_file(cache_file):
                print(f"Detected corrupted/HTML error file in cache: {cache_file.name}")
                print(f"Deleting corrupted file and forcing re-download...")
                cache_file.unlink()
                refresh = True  # Force refresh if we deleted a corrupted file
    except Exception as e:
        print(f"Warning: Could not check cache for corrupted files: {e}")
    
    try:
        return static_table(
            table_name='Generators and Scheduled Loads',
            raw_data_location=str(cache_path),
            update_static_file=bool(refresh),
        )
    except Exception as excel_error:
        msg = str(excel_error)
        if 'Excel file format cannot be determined' in msg:
            # Inspect cache for .xls files and attempt conversion to .xlsx
            try:
                files = sorted([p.name for p in Path(cache_path).iterdir()])
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
                        f"Cache directory: {cache_path}\n"
                        f"Cache dir listing (first 50 entries): {files[:50]}"
                    ) from excel_error

                for f in xls_files:
                    xls_path = Path(cache_path) / f
                    xlsx_name = xls_path.stem + '.xlsx'
                    xlsx_path = Path(cache_path) / xlsx_name
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
                            raw_data_location=str(cache_path),
                            update_static_file=False,
                        )
                    except Exception:
                        # fall through to forced refresh below
                        pass

                # If conversions failed and no success, raise informative error
                if conversion_errors and not converted:
                    raise ValueError(
                        "Failed to convert cached .xls files to .xlsx.\n"
                        "Conversion errors: " + ", ".join([f"{n}: {e}" for n, e in conversion_errors]) + "\n\n"
                        f"Cache directory: {cache_path}\n"
                        f"Cache dir listing (first 50 entries): {files[:50]}\n\n"
                        "Fix: Install 'xls2xlsx' or delete the cache directory and retry (rm -rf data/aemo),"
                    ) from excel_error

            # Try a forced refresh (redownload) in case cached file is corrupt or was HTML
            print("Attempting forced refresh to re-download static table...")
            try:
                result = static_table(
                    table_name='Generators and Scheduled Loads',
                    raw_data_location=str(cache_path),
                    update_static_file=True,
                )
                # Check if the newly downloaded file is an HTML error
                try:
                    files = sorted([p.name for p in Path(cache_path).iterdir()])
                    for f in files:
                        file_path = Path(cache_path) / f
                        if file_path.suffix.lower() in ['.xls', '.xlsx']:
                            if _is_html_error_file(file_path):
                                raise ValueError(
                                    "AEMO server returned an HTML error page instead of the Excel file. "
                                    "The server may be experiencing issues or rate-limiting requests.\n\n"
                                    "The downloaded file contains: 'Sorry, your request has failed...'\n\n"
                                    "Fixes:\n"
                                    " - Wait a few minutes and try again (AEMO server may be temporarily unavailable)\n"
                                    " - Delete the corrupted cache file manually:\n"
                                    f"     rm {file_path}\n"
                                    " - Delete the entire cache directory and retry:\n"
                                    f"     rm -rf {cache_path}\n"
                                    " - Try using a different date range or region to reduce load on AEMO servers\n"
                                )
                except Exception:
                    pass  # Continue with the result if check failed
                return result
            except Exception as excel_error2:
                # Provide helpful diagnostics including cache listing
                try:
                    files = sorted([p.name for p in Path(cache_path).iterdir()])
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
                    f"Cache directory: {cache_path}\n"
                    f"Cache dir listing (first 50 entries): {files[:50]}\n\n"
                    "Fixes:\n"
                    " - Wait a few minutes and try again (AEMO server may be temporarily unavailable)\n"
                    " - Delete the cache directory and retry:\n"
                    f"     rm -rf {cache_path}\n"
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
      - TechnologyType
      - FuelType
      - StorageCapacityMWh
      - RegisteredCapacityMW

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
        ],
        keyword_groups=[["storage", "mwh"], ["energy", "mwh"], ["capacity", "mwh"]],
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
        scada_data = dynamic_data_compiler(
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
    
    # Fetch unit-specific dispatch data
    unit_dispatch = fetch_aemo_unit_dispatch(
        start_date, end_date, duid, region, generator_info_path, cache_dir, refresh
    )
    
    return {
        'prices': prices,
        'fcas': fcas,
        'generation': generation,
        'unit_dispatch': unit_dispatch,
    }


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
