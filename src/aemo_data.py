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
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _pick_sheet_with_columns(sheets: Dict[str, pd.DataFrame], required_columns: List[str]) -> Optional[pd.DataFrame]:
    required_lower = {c.lower() for c in required_columns}
    for _, sheet in sheets.items():
        sheet = _normalize_columns(sheet)
        cols_lower = {c.lower() for c in sheet.columns}
        if required_lower.issubset(cols_lower):
            return sheet
    return None


def _read_generator_info_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
        return _normalize_columns(df)

    # Excel: load all sheets, then pick the one that looks like generator info.
    sheets = pd.read_excel(file_path, sheet_name=None)
    if isinstance(sheets, dict) and sheets:
        picked = _pick_sheet_with_columns(sheets, ["DUID", "Region"])
        if picked is not None:
            return picked
        # Fall back to first sheet
        first_sheet = next(iter(sheets.values()))
        return _normalize_columns(first_sheet)

    # pandas can return a DataFrame for some excel engines; normalize and return.
    if isinstance(sheets, pd.DataFrame):
        return _normalize_columns(sheets)

    raise ValueError(f"Could not read generator info file: {file_path}")


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


def _get_generator_info(cache_path: Path, generator_info_path: Optional[str] = None) -> Optional[pd.DataFrame]:
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
            return _normalize_columns(gen_info)
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
        if price_data is not None and len(price_data) > 0:
            price_data = price_data[price_data['REGIONID'] == region].copy()
            
            # Convert to polars DataFrame with standardized column names
            # NEMOSIS returns: SETTLEMENTDATE, REGIONID, RRP, and other columns
            price_data['SETTLEMENTDATE'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
            price_data['RRP'] = pd.to_numeric(price_data['RRP'], errors='coerce')
            
            # Also fetch DISPATCHREGIONSUM for demand data
            demand_data = dynamic_data_compiler(
                start_time=start_time,
                end_time=end_time,
                table_name='DISPATCHREGIONSUM',
                raw_data_location=str(cache_path)
            )
            
            if demand_data is not None and len(demand_data) > 0:
                demand_data = demand_data[demand_data['REGIONID'] == region].copy()
                demand_data['SETTLEMENTDATE'] = pd.to_datetime(demand_data['SETTLEMENTDATE'])
                demand_data['TOTALDEMAND'] = pd.to_numeric(demand_data['TOTALDEMAND'], errors='coerce')
                
                # Merge price and demand data
                result = price_data.merge(
                    demand_data[['SETTLEMENTDATE', 'TOTALDEMAND']],
                    on='SETTLEMENTDATE',
                    how='left'
                )
            else:
                # If no demand data, add placeholder column
                result = price_data.copy()
                result['TOTALDEMAND'] = 0.0
            
            # Select and order columns
            result = result[['SETTLEMENTDATE', 'REGIONID', 'RRP', 'TOTALDEMAND']]
            
            # Convert to Polars
            df = pl.from_pandas(result)
            
            print(f"Fetched {len(df)} price records")
            return df
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
        
        if price_data is not None and len(price_data) > 0:
            price_data = price_data[price_data['REGIONID'] == region].copy()
            price_data['SETTLEMENTDATE'] = pd.to_datetime(price_data['SETTLEMENTDATE'])
            
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
            if price_col and price_col in price_data.columns:
                price_data['PRICE'] = pd.to_numeric(price_data[price_col], errors='coerce')
                
                result = price_data[['SETTLEMENTDATE', 'REGIONID']].copy()
                result['SERVICE'] = service
                result['PRICE'] = price_data['PRICE']
                
                # Convert to Polars
                df = pl.from_pandas(result)
                
                print(f"Fetched {len(df)} FCAS price records")
                return df
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
            - RAISE6SECACTUALAVAILABILITY: Enabled capacity for 6-second raise (MW)
            - RAISE60SECACTUALAVAILABILITY: Enabled capacity for 60-second raise (MW)
            - RAISE5MINACTUALAVAILABILITY: Enabled capacity for 5-minute raise (MW)
            - RAISEREGACTUALAVAILABILITY: Enabled capacity for regulation raise (MW)
            - LOWER6SECACTUALAVAILABILITY: Enabled capacity for 6-second lower (MW)
            - LOWER60SECACTUALAVAILABILITY: Enabled capacity for 60-second lower (MW)
            - LOWER5MINACTUALAVAILABILITY: Enabled capacity for 5-minute lower (MW)
            - LOWERREGACTUALAVAILABILITY: Enabled capacity for regulation lower (MW)
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
        # Fetch DISPATCHLOAD which contains unit-specific dispatch and FCAS enablement
        dispatch_data = dynamic_data_compiler(
            start_time=start_time,
            end_time=end_time,
            table_name='DISPATCHLOAD',
            raw_data_location=str(cache_path)
        )
        
        if dispatch_data is None or len(dispatch_data) == 0:
            print("No dispatch data returned from NEMOSIS")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'DUID': pl.Utf8,
                'TOTALCLEARED': pl.Float64,
                'RAISE6SECACTUALAVAILABILITY': pl.Float64,
                'RAISE60SECACTUALAVAILABILITY': pl.Float64,
                'RAISE5MINACTUALAVAILABILITY': pl.Float64,
                'RAISEREGACTUALAVAILABILITY': pl.Float64,
                'LOWER6SECACTUALAVAILABILITY': pl.Float64,
                'LOWER60SECACTUALAVAILABILITY': pl.Float64,
                'LOWER5MINACTUALAVAILABILITY': pl.Float64,
                'LOWERREGACTUALAVAILABILITY': pl.Float64,
            })
        
        # Filter by DUID if specified
        if duid:
            dispatch_data = dispatch_data[dispatch_data['DUID'] == duid].copy()
            if len(dispatch_data) == 0:
                print(f"Warning: No data found for DUID {duid}")
                return pl.DataFrame(schema={
                    'SETTLEMENTDATE': pl.Datetime,
                    'DUID': pl.Utf8,
                    'TOTALCLEARED': pl.Float64,
                    'RAISE6SECACTUALAVAILABILITY': pl.Float64,
                    'RAISE60SECACTUALAVAILABILITY': pl.Float64,
                    'RAISE5MINACTUALAVAILABILITY': pl.Float64,
                    'RAISEREGACTUALAVAILABILITY': pl.Float64,
                    'LOWER6SECACTUALAVAILABILITY': pl.Float64,
                    'LOWER60SECACTUALAVAILABILITY': pl.Float64,
                    'LOWER5MINACTUALAVAILABILITY': pl.Float64,
                    'LOWERREGACTUALAVAILABILITY': pl.Float64,
                })
        
        # Convert timestamp
        dispatch_data['SETTLEMENTDATE'] = pd.to_datetime(dispatch_data['SETTLEMENTDATE'])
        
        # Select relevant columns (all FCAS enablement columns and energy dispatch)
        # ACTUALAVAILABILITY columns represent the enabled capacity for each FCAS service
        columns_to_keep = ['SETTLEMENTDATE', 'DUID']
        
        # Energy dispatch
        if 'TOTALCLEARED' in dispatch_data.columns:
            columns_to_keep.append('TOTALCLEARED')
            dispatch_data['TOTALCLEARED'] = pd.to_numeric(dispatch_data['TOTALCLEARED'], errors='coerce')
        
        # FCAS enablement columns
        fcas_columns = [
            'RAISE6SECACTUALAVAILABILITY',
            'RAISE60SECACTUALAVAILABILITY',
            'RAISE5MINACTUALAVAILABILITY',
            'RAISEREGACTUALAVAILABILITY',
            'LOWER6SECACTUALAVAILABILITY',
            'LOWER60SECACTUALAVAILABILITY',
            'LOWER5MINACTUALAVAILABILITY',
            'LOWERREGACTUALAVAILABILITY',
        ]
        
        for col in fcas_columns:
            if col in dispatch_data.columns:
                columns_to_keep.append(col)
                dispatch_data[col] = pd.to_numeric(dispatch_data[col], errors='coerce')
        
        # Filter by region if specified (requires generator static info)
        if region:
            # Prefer the robust static-table reader which attempts conversions and forced refreshes
            gen_info = None
            try:
                gen_info = _get_generators_static_table(cache_path, refresh)
            except Exception:
                # Fall back to local file or env var when the static_table path fails
                gen_info = _get_generator_info(cache_path, generator_info_path=generator_info_path)

            if gen_info is not None and 'Region' in gen_info.columns and 'DUID' in gen_info.columns:
                region_duids = gen_info[gen_info['Region'] == region]['DUID'].tolist()
                dispatch_data = dispatch_data[dispatch_data['DUID'].isin(region_duids)].copy()
            else:
                print(
                    "Warning: Could not load generator static info to filter by region. "
                    "Provide `generator_info_path` (or set env var AEMO_GENERATORS_FILE)."
                )
        
        # Select only the columns that exist
        available_columns = [col for col in columns_to_keep if col in dispatch_data.columns]
        result = dispatch_data[available_columns].copy()
        
        # Convert to Polars
        df = pl.from_pandas(result)
        
        print(f"Fetched {len(df)} dispatch records for {df['DUID'].n_unique()} unique units")
        return df
        
    except Exception as e:
        print(f"Error fetching unit dispatch data from NEMOSIS: {e}")
        print("Note: NEMOSIS requires internet connection to download data from AEMO")
        raise


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
        
        # Filter generators by region
        if 'Region' not in gen_info.columns or 'DUID' not in gen_info.columns:
            print(
                "Warning: Generator info file is missing required columns (need 'DUID' and 'Region')."
            )
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })

        gen_info = gen_info[gen_info['Region'] == region].copy()
        
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
        
        if scada_data is None or len(scada_data) == 0:
            print("Warning: No SCADA data returned")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })
        
        # Convert timestamp
        scada_data['SETTLEMENTDATE'] = pd.to_datetime(scada_data['SETTLEMENTDATE'])
        scada_data['SCADAVALUE'] = pd.to_numeric(scada_data['SCADAVALUE'], errors='coerce')
        
        # Merge SCADA data with generator info
        fuel_col = 'Fuel Source - Descriptor'
        if fuel_col not in gen_info.columns:
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

        merged = scada_data.merge(
            gen_info[['DUID', fuel_col]],
            on='DUID',
            how='left'
        )
        
        # Map to our fuel types
        all_data = []
        for fuel_type in fuel_types:
            if fuel_type in fuel_mapping:
                fuel_sources = fuel_mapping[fuel_type]
                fuel_data = merged[merged[fuel_col].isin(fuel_sources)].copy()
                
                # Aggregate by timestamp
                if len(fuel_data) > 0:
                    aggregated = fuel_data.groupby('SETTLEMENTDATE').agg({
                        'SCADAVALUE': 'sum'
                    }).reset_index()
                    
                    aggregated['REGIONID'] = region
                    aggregated['FUEL_TYPE'] = fuel_type
                    aggregated.rename(columns={'SCADAVALUE': 'GENERATION'}, inplace=True)
                    
                    all_data.append(aggregated)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = result[['SETTLEMENTDATE', 'REGIONID', 'FUEL_TYPE', 'GENERATION']]
            
            # Convert to Polars
            df = pl.from_pandas(result)
            
            print(f"Fetched {len(df)} generation records")
            return df
        else:
            print("No generation data found for specified fuel types")
            return pl.DataFrame(schema={
                'SETTLEMENTDATE': pl.Datetime,
                'REGIONID': pl.Utf8,
                'FUEL_TYPE': pl.Utf8,
                'GENERATION': pl.Float64
            })
            
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
