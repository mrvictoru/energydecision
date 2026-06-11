"""
Dispatch Replay Utilities for AEMO Battery Trading Environments
==============================================================

This module provides high-level helpers for selecting, resolving, and running
*dispatch replay* simulations using real AEMO DISPATCHLOAD data.  It bridges
the raw data-fetching functions in ``aemo_data`` with the ``AEMOBatteryTradingEnv``
and ``AEMOAgent`` so that notebooks stay clean and readable.

Typical workflow
----------------
1. Call :func:`list_known_batteries` to see all batteries registered in the
   built-in registry with their historical DUIDs.
2. Call :func:`list_dispatch_candidates` to see which battery DUIDs are available
   and which were actually dispatched during a chosen date window.  Pass a
   station name (e.g. ``"hornsdale"``) instead of a DUID to automatically use
   the correct historical DUID(s) for the requested period.
3. Call :func:`resolve_dispatch_selection` to pick a DUID (by name or index) and
   obtain sizing information and DUID metadata.
4. Call :func:`run_dispatch_replay` to run N episodes with the dispatch replay
   agent and save the logs to parquet files.

Public API
----------
- :func:`list_known_batteries` — show all batteries in the built-in registry
- :func:`show_dispatch_table` — pretty-print a subset of columns from a DataFrame
- :func:`list_dispatch_candidates` — list available battery DUIDs and their
  dispatch-activity statistics for a given region and date window
- :func:`resolve_dispatch_selection` — select a DUID and resolve env-sizing /
  paired gen–load metadata
- :func:`run_dispatch_replay` — run dispatch replay episodes and save logs

See ``docs/aemo/dispatch-replay.md`` for a detailed usage guide.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Default SOC ratio used when battery_capacity is zero (50 %)
_DEFAULT_SOC_RATIO: float = 0.5

# AEMO dispatch type strings that indicate a bidirectional (combined charge/discharge) DUID
_BIDI_DISPATCH_TYPES: frozenset = frozenset({"bidirectional", "bidirectional unit"})

# Suffix appended to the synthetic DUID name created when merging a transition-spanning
# dispatch signal (pre-transition gen/load + post-transition bidi).
_TRANSITION_DUID_SUFFIX: str = "__TRANSITION__"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def list_known_batteries() -> pl.DataFrame:
    """Return a DataFrame of all batteries in the built-in registry.

    Convenience re-export of :func:`aemo_data.list_known_batteries`.
    Each row represents one DUID registration period.  The ``Key`` column
    can be passed to :func:`list_dispatch_candidates` or
    :func:`resolve_dispatch_selection` as a station name.

    Example::

        from dispatch_utils import list_known_batteries
        list_known_batteries()
    """
    from aemo_data import list_known_batteries as _lkb
    return _lkb()


def show_dispatch_table(
    df: pl.DataFrame,
    columns: List[str],
    label: str = "",
    limit: int = 20,
) -> None:
    """Print a labelled, column-filtered view of *df*.

    Args:
        df: Polars DataFrame to display.
        columns: Ordered list of column names to show.  Missing columns are
            silently skipped.
        label: Optional heading printed above the table.
        limit: Maximum number of rows to print.
    """
    if label:
        print(f"\n{label}")
    if df.height == 0:
        print("[INFO] No rows found.")
        return
    shown = [c for c in columns if c in df.columns]
    print(df.select(shown).head(limit))


# ---------------------------------------------------------------------------

def _dispatch_type_priority(value: Any) -> int:
    text = str(value or "").strip().lower()
    if text in {"generator", "generating unit"}:
        return 0
    if text in _BIDI_DISPATCH_TYPES:
        return 1
    if text == "load":
        return 2
    return 3


def _sort_station_candidates(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df
    sort_cols = [c for c in ["NonZeroIntervalCount", "DispatchIntervalCount", "MaxEnergyMW"] if c in df.columns]
    descending = [True] * len(sort_cols)
    out = df
    if "DispatchType" in out.columns:
        out = out.with_columns(
            pl.col("DispatchType")
            .map_elements(_dispatch_type_priority, return_dtype=pl.Int64)
            .alias("_dispatch_type_priority")
        )
        sort_cols.append("_dispatch_type_priority")
        descending.append(False)
    if "DUID" in out.columns:
        sort_cols.append("DUID")
        descending.append(False)
    out = out.sort(sort_cols, descending=descending, nulls_last=True)
    return out.drop("_dispatch_type_priority") if "_dispatch_type_priority" in out.columns else out


def _attach_station_pair_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df
    if "PairedGenDUID" not in df.columns or "PairedLoadDUID" not in df.columns:
        df = df.with_columns([
            pl.lit(None, dtype=pl.Utf8).alias("PairedGenDUID"),
            pl.lit(None, dtype=pl.Utf8).alias("PairedLoadDUID"),
        ])
    if "DispatchType" not in df.columns:
        return df

    gen_rows = df.filter(
        pl.col("DispatchType").cast(pl.Utf8, strict=False).str.to_lowercase().is_in(["generator", "generating unit"])
    )
    load_rows = df.filter(
        pl.col("DispatchType").cast(pl.Utf8, strict=False).str.to_lowercase() == "load"
    )
    if gen_rows.height == 1 and load_rows.height == 1:
        gen_duid = str(gen_rows["DUID"][0])
        load_duid = str(load_rows["DUID"][0])
        df = df.with_columns([
            pl.when(pl.col("DUID") == gen_duid)
            .then(pl.lit(load_duid))
            .otherwise(pl.col("PairedLoadDUID"))
            .alias("PairedLoadDUID"),
            pl.when(pl.col("DUID") == load_duid)
            .then(pl.lit(gen_duid))
            .otherwise(pl.col("PairedGenDUID"))
            .alias("PairedGenDUID"),
        ])
    return df


# ---------------------------------------------------------------------------

def list_dispatch_candidates(
    region: str,
    start_date: datetime,
    end_date: datetime,
    station_name: Optional[str] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Return all battery DUIDs and the subset that was active in DISPATCHLOAD.

    Queries the AEMO static table for batteries in *region* and then checks
    DISPATCHLOAD to find those that were actually dispatched between *start_date*
    and *end_date*.

    You can optionally pass a ``station_name`` (e.g. ``"hornsdale"`` or
    ``"Hornsdale Power Reserve"``) to restrict the query to a single battery.
    The function will automatically resolve the correct DUID(s) for the
    requested date range using :func:`aemo_data.resolve_battery_duids`, so
    you get the right historical DUID(s) even for pre-transition periods.

    .. note::
        For date windows longer than ~1 month the underlying data download is
        automatically memory-optimised by filtering to only the battery DUIDs
        found in the static table (typically 10–15 DUIDs per region), rather
        than downloading all ~1 000+ NEM units per month.

    Args:
        region: AEMO region code (e.g. ``"SA1"``, ``"QLD1"``).
        start_date: Start of the date window (inclusive).
        end_date: End of the date window (inclusive).
        station_name: Optional station name or registry key (e.g.
            ``"hornsdale"``, ``"lake bonney"``, ``"HPR1"``).  When provided,
            only that battery is queried — and its historical DUIDs are used
            automatically for pre-transition periods.
        cache_dir: Path to the NEMOSIS data cache directory.
        refresh: When ``True``, force re-download of cached files.

    Returns:
        ``(battery_units, active_battery_units)``

        *battery_units* — all batteries registered in *region* with columns
        ``DUID``, ``Region``, ``DispatchType``, ``TechnologyType``,
        ``FuelType``, ``StorageCapacityMWh``, ``RegisteredCapacityMW``.

        *active_battery_units* — the subset with DISPATCHLOAD rows, enriched
        with ``NonZeroIntervalCount``, ``MaxEnergyMW``, ``DispatchIntervalCount``,
        ``FirstDispatchInterval``, ``LastDispatchInterval``,
        ``PairedGenDUID``, ``PairedLoadDUID``.
        Sorted by ``NonZeroIntervalCount`` descending so the most-active
        batteries appear first.
    """
    from aemo_data import (
        get_available_battery_units,
        get_dispatch_active_battery_units,
        resolve_battery_duids,
        fetch_aemo_unit_dispatch,
        DISPATCH_NONZERO_THRESHOLD,
        BATTERY_REGISTRY,
    )

    # -------------------------------------------------------------------------
    # Station-name path: resolve to historical DUIDs automatically
    # -------------------------------------------------------------------------
    if station_name:
        resolution = resolve_battery_duids(station_name, start_date, end_date)
        if not resolution["found"]:
            print(
                f"[WARN] Station name {station_name!r} not found in the battery registry.\n"
                "       Available stations:\n"
                + "\n".join(
                    f"  {k}: {v['full_name']}"
                    for k, v in __import__("aemo_data").BATTERY_REGISTRY.items()
                )
            )
            return pl.DataFrame(), pl.DataFrame()

        duids_in_range = resolution["all_duids_in_range"]
        station_info = BATTERY_REGISTRY.get(resolution["key"], {})
        all_known_station_duids = list(dict.fromkeys(
            str(entry["duid"]).strip()
            for entry in station_info.get("duids", [])
            if entry.get("duid")
        ))
        print(
            f"Station: {resolution['station_name']!r} ({resolution['region']})\n"
            f"DUIDs active in {start_date.date()} → {end_date.date()}: {duids_in_range}"
        )
        if all_known_station_duids and all_known_station_duids != duids_in_range:
            print(f"All known DUIDs checked for this station: {all_known_station_duids}")

        # Guard: battery not registered in this period at all
        if not duids_in_range and not all_known_station_duids:
            # Show what DUIDs exist and when, to help user pick the right range
            all_periods = [
                f"  {e['duid']} ({e['type']}): "
                f"{e.get('valid_from','?').date() if e.get('valid_from') else '?'}"
                f" → "
                f"{e['valid_until'].date() if e.get('valid_until') else 'present'}"
                for e in __import__("aemo_data").BATTERY_REGISTRY[resolution["key"]]["duids"]
            ]
            print(
                f"[WARN] {resolution['station_name']!r} has no registered DUID(s) "
                f"for {start_date.date()} → {end_date.date()}.\n"
                f"       Known registration periods:\n" + "\n".join(all_periods) + "\n"
                f"       Call list_known_batteries() to review all registration details."
            )
            return pl.DataFrame(), pl.DataFrame()

        if resolution["spans_transition"]:
            td = [str(d.date()) for d in resolution["transition_dates"]]
            print(
                f"[NOTE] The date range spans a DUID transition on {td}.\n"
                "       Data will be fetched for all relevant DUIDs and merged."
            )

        # Fetch dispatch for the resolved DUIDs
        dispatch_df = fetch_aemo_unit_dispatch(
            start_date=start_date,
            end_date=end_date,
            duids=all_known_station_duids or duids_in_range,
            cache_dir=cache_dir,
            refresh=refresh,
        )

        # Build a minimal activity summary table matching the normal output format.
        # If no dispatch rows exist for the selected station/window, keep the
        # registry-backed static table so callers can still resolve sizing and
        # surface a clearer replay-time error.
        dispatch_numeric_cols = [c for c in dispatch_df.columns if c not in {"SETTLEMENTDATE", "DUID"}]
        activity_df = dispatch_df.with_columns(
            pl.col("SETTLEMENTDATE").cast(pl.Datetime, strict=False)
        )
        nonzero_condition = None
        for col in dispatch_numeric_cols:
            expr = pl.col(col).cast(pl.Float64, strict=False).abs() > DISPATCH_NONZERO_THRESHOLD
            nonzero_condition = expr if nonzero_condition is None else (nonzero_condition | expr)

        if nonzero_condition is not None:
            activity_df = activity_df.with_columns(
                pl.when(nonzero_condition).then(1).otherwise(0).alias("_has_activity")
            )
        else:
            activity_df = activity_df.with_columns(pl.lit(0).alias("_has_activity"))

        tc_expr = (
            pl.col("TOTALCLEARED").cast(pl.Float64, strict=False).abs()
            if "TOTALCLEARED" in dispatch_numeric_cols
            else pl.lit(0.0)
        )
        summary = (
            activity_df.group_by("DUID").agg([
                pl.col("SETTLEMENTDATE").n_unique().alias("DispatchIntervalCount"),
                pl.col("SETTLEMENTDATE").min().alias("FirstDispatchInterval"),
                pl.col("SETTLEMENTDATE").max().alias("LastDispatchInterval"),
                pl.col("_has_activity").sum().cast(pl.UInt32).alias("NonZeroIntervalCount"),
                tc_expr.max().alias("MaxEnergyMW"),
            ])
        )

        # Build a minimal battery_units table for the resolved station.
        # Populate capacity from the BATTERY_REGISTRY as a reliable fallback for
        # historical gen/load DUIDs that are deregistered from the NEMOSIS static
        # table (e.g. HPRG1/HPRL1 before Hornsdale's 2022 transition to HPR1).
        registry_capacity_mwh: Optional[float] = station_info.get("capacity_mwh")
        registry_max_power_mw: Optional[float] = station_info.get("max_power_mw")
        static_rows = [{
            "DUID": e["duid"],
            "Region": resolution["region"],
            "DispatchType": e["type"],
            "TechnologyType": "Battery and Inverter",
            "FuelType": "Grid",
            "StorageCapacityMWh": registry_capacity_mwh,
            "RegisteredCapacityMW": registry_max_power_mw,
        } for e in station_info.get("duids", [])]
        battery_units = pl.DataFrame(static_rows, schema_overrides={
            "StorageCapacityMWh": pl.Float64,
            "RegisteredCapacityMW": pl.Float64,
        })

        active_battery_units = battery_units.join(summary, on="DUID", how="inner")
        battery_units = _sort_station_candidates(_attach_station_pair_columns(battery_units))
        active_battery_units = _sort_station_candidates(_attach_station_pair_columns(active_battery_units))

        show_dispatch_table(
            active_battery_units,
            ["DUID", "Region", "DispatchType", "NonZeroIntervalCount",
             "DispatchIntervalCount", "MaxEnergyMW",
             "FirstDispatchInterval", "LastDispatchInterval",
             "PairedGenDUID", "PairedLoadDUID"],
            label=f"Dispatch data for {resolution['station_name']!r}",
        )
        return battery_units, active_battery_units

    # -------------------------------------------------------------------------
    # Normal path: region-based query
    # -------------------------------------------------------------------------
    battery_units = get_available_battery_units(
        cache_dir=cache_dir,
        region=region,
        refresh=refresh,
    )

    if battery_units.height == 0:
        print(f"[SKIP] No battery units found in static table for region={region}")
        return battery_units, pl.DataFrame()

    battery_units = battery_units.sort(
        ["StorageCapacityMWh", "RegisteredCapacityMW"],
        descending=[True, True],
        nulls_last=True,
    )
    print(f"Available battery DUIDs in {region}: {battery_units.height}")
    show_dispatch_table(
        battery_units,
        [
            "DUID",
            "Region",
            "DispatchType",
            "TechnologyType",
            "FuelType",
            "StorageCapacityMWh",
            "RegisteredCapacityMW",
        ],
        label="Static-table battery units",
    )

    print(
        f"\nFinding dispatch-active battery DUIDs in {region} for "
        f"{start_date.date()} to {end_date.date()}…"
    )
    active_battery_units = get_dispatch_active_battery_units(
        start_date=start_date,
        end_date=end_date,
        region=region,
        cache_dir=cache_dir,
        refresh=refresh,
    )

    if active_battery_units.height == 0:
        print(
            f"[SKIP] No battery DUIDs from the static table appear in DISPATCHLOAD for "
            f"{region} during {start_date.date()} to {end_date.date()}."
        )
        return battery_units, active_battery_units

    active_battery_units = active_battery_units.sort(
        ["NonZeroIntervalCount", "DispatchIntervalCount", "MaxEnergyMW"],
        descending=[True, True, True],
        nulls_last=True,
    )
    print(f"Dispatch-active battery DUIDs in {region}: {active_battery_units.height}")
    show_dispatch_table(
        active_battery_units,
        [
            "DUID",
            "Region",
            "DispatchType",
            "TechnologyType",
            "FuelType",
            "NonZeroIntervalCount",
            "DispatchIntervalCount",
            "MaxEnergyMW",
            "FirstDispatchInterval",
            "LastDispatchInterval",
            "PairedGenDUID",
            "PairedLoadDUID",
        ],
        label="Dispatch-active battery units (sorted by activity)",
    )
    print(
        "\nPick a DUID from the dispatch-active table above and pass it "
        "to resolve_dispatch_selection() or run_dispatch_replay()."
    )
    return battery_units, active_battery_units


# ---------------------------------------------------------------------------

def resolve_dispatch_selection(
    battery_units: pl.DataFrame,
    active_battery_units: pl.DataFrame,
    selected_duid: Optional[str] = None,
    selected_index: int = 0,
    battery_capacity: float = 10.0,
    max_battery_flow: float = 5.0,
    init_soc: float = 5.0,
    init_soc_ratio: Optional[float] = None,
    apply_unit_sizing: bool = True,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    cache_dir: str = "data/aemo",
) -> Dict[str, Any]:
    """Select a DUID and resolve environment-sizing parameters.

    Args:
        battery_units: Full static-table battery listing (from
            :func:`list_dispatch_candidates`).
        active_battery_units: Dispatch-active subset (from
            :func:`list_dispatch_candidates`). May be empty — falls back to
            *battery_units* in that case.
        selected_duid: Explicit DUID to use.  When *None*, uses
            *selected_index* to pick from the dispatch-active table
            (or the full static table if no active units were found).
        selected_index: Zero-based row index to use when *selected_duid* is
            *None*.
        battery_capacity: Default battery capacity (MWh) used when the static
            table does not provide ``StorageCapacityMWh`` or when
            *apply_unit_sizing* is ``False``.
        max_battery_flow: Default max discharge/charge rate (MW).
        init_soc: Default initial state-of-charge (MWh).
        init_soc_ratio: Optional initial state-of-charge ratio applied after
            unit sizing is resolved. When provided, this takes precedence over
            the legacy ``init_soc`` / ``battery_capacity`` ratio fallback.
        apply_unit_sizing: When ``True``, override *battery_capacity* and
            *max_battery_flow* with values from the static table (if available
            and finite).
        start_date: Start of the dispatch window (used for availability check).
            May be ``None`` to skip the availability check.
        end_date: End of the dispatch window (used for availability check).
        cache_dir: Path to the NEMOSIS data cache directory.

    Returns:
        A dictionary with the following keys:

        ``duid`` (str)
            The selected DUID.
        ``dispatch_type`` (str | None)
            AEMO dispatch type string (``"Bidirectional Unit"``,
            ``"Generating Unit"``, ``"Load"``, or ``None``).
        ``dispatch_duid_gen`` (str | None)
            The paired generator DUID (discharge direction) for old-model
            batteries, or ``None`` for bidirectional units.
        ``dispatch_duid_load`` (str | None)
            The paired load DUID (charge direction), or ``None``.
        ``region`` (str | None)
            Resolved NEM region for the selected unit / station.
        ``station_name`` (str | None)
            Registry station name when the DUID could be resolved.
        ``station_key`` (str | None)
            Registry station key when the DUID could be resolved.
        ``battery_capacity`` (float)
            Resolved battery capacity (MWh).
        ``max_battery_flow`` (float)
            Resolved max battery flow (MW).
        ``init_battery_level`` (float)
            Resolved initial SOC (MWh).
        ``availability`` (dict | None)
            Availability summary from
            :func:`aemo_data.check_aemo_dispatch_availability`, or ``None``
            if *start_date* / *end_date* were not provided.
    """
    from aemo_data import check_aemo_dispatch_availability, resolve_battery_duids, BATTERY_REGISTRY

    source_table = active_battery_units if active_battery_units.height > 0 else battery_units
    source_label = "dispatch-active" if active_battery_units.height > 0 else "static"

    if source_table.height == 0:
        raise ValueError(
            "No battery DUIDs are available for dispatch replay. "
            "Check the station name / DUID and the requested date range."
        )

    if selected_duid and str(selected_duid).strip():
        query = str(selected_duid).strip()
        # First try direct DUID match
        duid = query
        chosen = source_table.filter(pl.col("DUID") == duid).head(1)
        if chosen.height == 0:
            chosen = battery_units.filter(pl.col("DUID") == duid).head(1)

        # If not found as a DUID, try station-name resolution
        if chosen.height == 0 and start_date and end_date:
            resolution = resolve_battery_duids(query, start_date, end_date)
            if resolution["found"]:
                station_info = BATTERY_REGISTRY.get(resolution["key"], {})
                candidate_duids = list(dict.fromkeys(
                    str(entry["duid"]).strip()
                    for entry in station_info.get("duids", [])
                    if entry.get("duid")
                ))
                if candidate_duids:
                    chosen = _sort_station_candidates(
                        source_table.filter(pl.col("DUID").cast(pl.Utf8, strict=False).is_in(candidate_duids))
                    ).head(1)
                    if chosen.height == 0:
                        chosen = _sort_station_candidates(
                            battery_units.filter(pl.col("DUID").cast(pl.Utf8, strict=False).is_in(candidate_duids))
                        ).head(1)
                if chosen.height > 0:
                    duid = str(chosen["DUID"][0])
                else:
                    # Fall back to the registry primary DUID for this period: prefer bidi, then gen.
                    bidi = resolution["bidi_duid"]
                    gen = resolution["gen_duid"]
                    duid = bidi or gen or duid
                    chosen = source_table.filter(pl.col("DUID") == duid).head(1)
                    if chosen.height == 0:
                        chosen = battery_units.filter(pl.col("DUID") == duid).head(1)
                print(
                    f"Resolved station name {query!r} → DUID {duid!r} "
                    f"for {start_date.date()} to {end_date.date()}"
                )

        print(f"Selected dispatch DUID from manual input: {duid}")
    else:
        idx = max(0, min(int(selected_index), source_table.height - 1))
        duid = source_table["DUID"][idx]
        chosen = source_table.filter(pl.col("DUID") == duid).head(1)
        print(f"Selected dispatch DUID from {source_label} table row {idx}: {duid}")

    # Warn if the selected DUID has no non-zero dispatch
    if chosen.height > 0 and "NonZeroIntervalCount" in chosen.columns:
        nonzero = chosen["NonZeroIntervalCount"][0]
        if nonzero == 0:
            warnings.warn(
                f"Selected DUID {duid!r} has NonZeroIntervalCount=0 for the chosen date range.\n"
                "The dispatch replay will produce all-zero actions unless you choose a "
                "different unit or date range.",
                stacklevel=2,
            )

    selected_region: Optional[str] = None
    selected_station_name: Optional[str] = None
    selected_station_key: Optional[str] = None

    # Resolve paired gen/load DUIDs and check for DUID transitions
    dispatch_type: Optional[str] = None
    dispatch_duid_gen: Optional[str] = None
    dispatch_duid_load: Optional[str] = None
    spans_transition: bool = False
    transition_dates: list = []
    all_dispatch_duids: list = [duid]  # DUIDs needed across the full date range
    _registry_gen_duid: Optional[str] = None
    _registry_load_duid: Optional[str] = None
    _registry_bidi_duid: Optional[str] = None

    if chosen.height > 0:
        if "Region" in chosen.columns:
            val = chosen["Region"][0]
            if val is not None and str(val).strip():
                selected_region = str(val)
        if "DispatchType" in chosen.columns:
            dispatch_type = chosen["DispatchType"][0]
        if "PairedGenDUID" in chosen.columns:
            val = chosen["PairedGenDUID"][0]
            if val is not None and str(val).strip():
                dispatch_duid_gen = str(val)
        if "PairedLoadDUID" in chosen.columns:
            val = chosen["PairedLoadDUID"][0]
            if val is not None and str(val).strip():
                dispatch_duid_load = str(val)

    # Consult registry for transition info and paired DUIDs.
    # IMPORTANT: only supplement gen/load DUIDs when the primary DUID is NOT bidirectional.
    # For a bidi DUID (e.g. HPR1), adding the old gen/load pair as "paired DUIDs" would
    # cause run_dispatch_replay to fetch only those old DUIDs, skipping the bidi period.
    if start_date and end_date:
        _res = resolve_battery_duids(duid, start_date, end_date)
        if _res["found"]:
            selected_region = _res["region"] or selected_region
            selected_station_name = _res["station_name"]
            selected_station_key = _res["key"]
            spans_transition = _res["spans_transition"]
            transition_dates = _res["transition_dates"]
            _registry_gen_duid = _res["gen_duid"]
            _registry_load_duid = _res["load_duid"]
            _registry_bidi_duid = _res["bidi_duid"]
            all_dispatch_duids = _res["all_duids_in_range"] or [duid]

            _is_bidi = dispatch_type and dispatch_type.lower() in _BIDI_DISPATCH_TYPES
            if spans_transition:
                print(
                    f"[NOTE] The date range spans a DUID transition. "
                    f"All DUIDs will be fetched: {all_dispatch_duids}"
                )
            # Only supplement gen/load from registry when primary DUID is not bidirectional
            if not _is_bidi and dispatch_duid_gen is None and dispatch_duid_load is None:
                dispatch_duid_gen = dispatch_duid_gen or _registry_gen_duid
                dispatch_duid_load = dispatch_duid_load or _registry_load_duid

    # For old-model gen+load pairs the primary DUID is typically a "generator"
    # (discharge direction) and PairedLoadDUID points to the charge unit.
    # In this case PairedGenDUID is null (the DUID *is* the gen unit), so we
    # set dispatch_duid_gen = duid to ensure discharge data is fetched and
    # passed to the replay agent alongside the load data.
    if dispatch_duid_load is not None and dispatch_duid_gen is None:
        if dispatch_type and dispatch_type.lower() in {"generator", "generating unit"}:
            dispatch_duid_gen = duid

    if dispatch_duid_gen or dispatch_duid_load:
        print(
            f"Paired gen/load DUID detected: gen={dispatch_duid_gen!r}, load={dispatch_duid_load!r}\n"
            "The replay will fetch both units and combine them."
        )

    # Availability check — skip the expensive full re-download when the
    # active_battery_units table already tells us there is dispatch data.
    # This avoids re-reading 35 M+ rows for long date windows.
    availability: Optional[Dict[str, Any]] = None
    _skip_avail_check = False
    _chosen_nonzero = 0
    if chosen.height > 0 and "NonZeroIntervalCount" in chosen.columns:
        _chosen_nonzero = int(chosen["NonZeroIntervalCount"][0] or 0)
    if _chosen_nonzero > 0 or (active_battery_units.height > 0 and spans_transition):
        # Two conditions allow us to skip the expensive NEMOSIS download:
        # 1. NonZeroIntervalCount > 0 — list_dispatch_candidates already confirmed
        #    non-zero dispatch activity for this DUID in the date range.
        # 2. spans_transition — active_battery_units was built from all sub-period DUIDs;
        #    even if any single DUID shows NonZeroIntervalCount=0 (e.g. HPR1 has data only
        #    in the post-transition sub-period), the combined table has_data=True.
        _first = chosen["FirstDispatchInterval"][0] if (
            chosen.height > 0 and "FirstDispatchInterval" in chosen.columns
        ) else None
        _last = chosen["LastDispatchInterval"][0] if (
            chosen.height > 0 and "LastDispatchInterval" in chosen.columns
        ) else None
        _cnt = int(chosen["DispatchIntervalCount"][0]) if (
            chosen.height > 0 and "DispatchIntervalCount" in chosen.columns
            and chosen["DispatchIntervalCount"][0] is not None
        ) else _chosen_nonzero
        _expected = max(
            0,
            int((end_date - start_date).total_seconds() // (5 * 60))
        ) if start_date and end_date else 0
        _cov = float(_cnt) / float(_expected) if _expected > 0 else 1.0
        availability = {
            "duid": duid,
            "start_date": start_date,
            "end_date": end_date,
            "has_data": True,
            "row_count": int(_cnt),
            "unique_intervals": int(_cnt),
            "expected_intervals": _expected,
            "coverage_ratio": _cov,
            "months_checked": 0,
            "months_with_data": 1,
            "first_settlement": _first,
            "last_settlement": _last,
        }
        print(
            f"Dispatch availability for {duid} (from cached table): has_data=True, "
            f"intervals={_cnt}/{_expected}, coverage={_cov:.2%}"
        )
        print(f"  First: {_first}, Last: {_last}")
        _skip_avail_check = True

    if not _skip_avail_check and start_date is not None and end_date is not None:
        availability = check_aemo_dispatch_availability(
            start_date=start_date,
            end_date=end_date,
            duid=duid,
            cache_dir=cache_dir,
        )
        print(
            f"Dispatch availability for {duid}: has_data={availability['has_data']}, "
            f"rows={availability['row_count']}, "
            f"intervals={availability['unique_intervals']}/{availability['expected_intervals']}, "
            f"coverage={availability['coverage_ratio']:.2%}"
        )
        print(
            f"  First: {availability['first_settlement']}, "
            f"Last: {availability['last_settlement']}"
        )
        if not availability["has_data"]:
            raise ValueError(
                f"DUID {duid!r} has no DISPATCHLOAD rows in the chosen range "
                f"{start_date.date()} to {end_date.date()}."
            )

    # Sizing — prefer static-table values; fall back to BATTERY_REGISTRY when
    # the DUID is historical and absent from (or has null values in) the table.
    suggested_mwh: Optional[float] = None
    suggested_mw: Optional[float] = None
    if chosen.height > 0:
        if "StorageCapacityMWh" in chosen.columns:
            val = chosen["StorageCapacityMWh"][0]
            if val is not None:
                suggested_mwh = float(val)
        if "RegisteredCapacityMW" in chosen.columns:
            val = chosen["RegisteredCapacityMW"][0]
            if val is not None:
                suggested_mw = float(val)

    # Registry fallback for historical / deregistered DUIDs
    if (suggested_mwh is None or suggested_mw is None) and start_date and end_date:
        _reg_res = resolve_battery_duids(duid, start_date, end_date)
        if _reg_res["found"]:
            _station_info = BATTERY_REGISTRY.get(_reg_res["key"], {})
            if suggested_mwh is None:
                suggested_mwh = _station_info.get("capacity_mwh")
            if suggested_mw is None:
                suggested_mw = _station_info.get("max_power_mw")

    print(f"Suggested env sizing from table → capacity={suggested_mwh} MWh, max_flow={suggested_mw} MW")

    resolved_capacity = battery_capacity
    resolved_max_flow = max_battery_flow
    if apply_unit_sizing:
        if suggested_mwh is not None and np.isfinite(float(suggested_mwh)) and float(suggested_mwh) > 0:
            resolved_capacity = float(suggested_mwh)
        if suggested_mw is not None and np.isfinite(float(suggested_mw)) and float(suggested_mw) > 0:
            resolved_max_flow = float(suggested_mw)

    if init_soc_ratio is None:
        resolved_init_soc_ratio = (init_soc / battery_capacity) if battery_capacity > 0 else _DEFAULT_SOC_RATIO
    else:
        resolved_init_soc_ratio = float(np.clip(init_soc_ratio, 0.0, 1.0))
    resolved_init_soc = float(
        np.clip(resolved_capacity * resolved_init_soc_ratio, 0.0, resolved_capacity)
    )
    print(
        f"Dispatch replay env params → capacity={resolved_capacity:.3f} MWh, "
        f"max_flow={resolved_max_flow:.3f} MW, init_soc={resolved_init_soc:.3f} MWh"
    )

    return {
        "duid": duid,
        "region": selected_region,
        "station_name": selected_station_name,
        "station_key": selected_station_key,
        "dispatch_type": dispatch_type,
        "dispatch_duid_gen": dispatch_duid_gen,
        "dispatch_duid_load": dispatch_duid_load,
        # Transition-aware fields (populated when the date range crosses a DUID
        # registration boundary, e.g. Hornsdale HPRG1/HPRL1 → HPR1 in Oct 2022)
        "spans_transition": spans_transition,
        "transition_dates": transition_dates,
        "all_dispatch_duids": all_dispatch_duids,
        # Registry gen/load/bidi DUIDs (used by run_dispatch_replay for transitions)
        "_registry_gen_duid": _registry_gen_duid,
        "_registry_load_duid": _registry_load_duid,
        "_registry_bidi_duid": _registry_bidi_duid,
        "battery_capacity": resolved_capacity,
        "max_battery_flow": resolved_max_flow,
        "init_battery_level": resolved_init_soc,
        "availability": availability,
    }


# ---------------------------------------------------------------------------

def _merge_transition_dispatch(
    dispatch_df: pl.DataFrame,
    selection: Dict[str, Any],
    transition_dates: list,
) -> pl.DataFrame:
    """Merge a multi-DUID dispatch DataFrame across a DUID transition into a
    single generator-convention TOTALCLEARED series.

    For periods before each transition date the gen/load pair is used:
    ``TOTALCLEARED = GEN_MW - LOAD_MW`` (positive = discharging, negative = charging).
    For periods after the last transition date the bidirectional DUID is used directly
    (TOTALCLEARED in the same generator convention).

    The resulting DataFrame has a single synthetic DUID column and can be passed to
    ``AEMOAgent`` with ``assume_single_duid_is_generator=True``.
    """
    _registry_gen = selection.get("_registry_gen_duid")
    _registry_load = selection.get("_registry_load_duid")
    _registry_bidi = selection.get("_registry_bidi_duid")
    duid_primary = selection.get("duid", "MERGED")

    # Use first transition date to split pre/post periods
    transition_date = min(transition_dates) if transition_dates else None

    numeric_cols = [c for c in dispatch_df.columns if c not in {"SETTLEMENTDATE", "DUID"}]

    def _get(duid: Optional[str]) -> pl.DataFrame:
        if not duid or dispatch_df.height == 0:
            return pl.DataFrame()
        return dispatch_df.filter(pl.col("DUID") == duid)

    # --- Pre-transition: gen/load pair → TOTALCLEARED = GEN_MW - LOAD_MW ---
    parts: list[pl.DataFrame] = []
    if _registry_gen or _registry_load:
        gen_raw = _get(_registry_gen)
        load_raw = _get(_registry_load)

        if gen_raw.height > 0 and load_raw.height > 0:
            gen_s = gen_raw.select(
                ["SETTLEMENTDATE"] + numeric_cols
            ).rename({c: f"_gen_{c}" for c in numeric_cols})
            load_s = load_raw.select(
                ["SETTLEMENTDATE"] + numeric_cols
            ).rename({c: f"_load_{c}" for c in numeric_cols})
            pre_merged = gen_s.join(load_s, on="SETTLEMENTDATE", how="full", coalesce=True)
            # Generator convention: positive = generating (discharging)
            tc_expr = (
                pl.col(f"_gen_TOTALCLEARED").fill_null(0.0)
                - pl.col(f"_load_TOTALCLEARED").fill_null(0.0)
            ).alias("TOTALCLEARED") if "TOTALCLEARED" in numeric_cols else pl.lit(0.0).alias("TOTALCLEARED")
            reg_r = (
                pl.col(f"_gen_RAISEREG").fill_null(0.0)
                + pl.col(f"_load_RAISEREG").fill_null(0.0)
            ).alias("RAISEREG") if "RAISEREG" in numeric_cols else pl.lit(0.0).alias("RAISEREG")
            reg_l = (
                pl.col(f"_gen_LOWERREG").fill_null(0.0)
                + pl.col(f"_load_LOWERREG").fill_null(0.0)
            ).alias("LOWERREG") if "LOWERREG" in numeric_cols else pl.lit(0.0).alias("LOWERREG")
            pre = pre_merged.with_columns([tc_expr, reg_r, reg_l]).select(
                ["SETTLEMENTDATE", "TOTALCLEARED", "RAISEREG", "LOWERREG"]
            )
            if transition_date is not None:
                pre = pre.filter(pl.col("SETTLEMENTDATE") < transition_date)
            parts.append(pre)
        elif gen_raw.height > 0:
            pre = gen_raw.select(
                ["SETTLEMENTDATE"] + [c for c in ["TOTALCLEARED", "RAISEREG", "LOWERREG"] if c in numeric_cols]
            )
            if transition_date is not None:
                pre = pre.filter(pl.col("SETTLEMENTDATE") < transition_date)
            parts.append(pre)

    # --- Post-transition: bidirectional DUID (already in generator convention) ---
    if _registry_bidi:
        bidi_raw = _get(_registry_bidi)
        if bidi_raw.height > 0:
            post = bidi_raw.select(
                ["SETTLEMENTDATE"] + [c for c in ["TOTALCLEARED", "RAISEREG", "LOWERREG"] if c in bidi_raw.columns]
            )
            if transition_date is not None:
                post = post.filter(pl.col("SETTLEMENTDATE") >= transition_date)
            parts.append(post)

    if not parts:
        # Fallback: return a pass-through single DUID with TOTALCLEARED=0
        return dispatch_df.filter(pl.col("DUID") == (
            _registry_bidi or _registry_gen or duid_primary
        ))

    unified = pl.concat(parts, how="diagonal_relaxed").sort("SETTLEMENTDATE")
    # Fill any gaps in RAISEREG/LOWERREG
    for col in ["RAISEREG", "LOWERREG"]:
        if col not in unified.columns:
            unified = unified.with_columns(pl.lit(0.0).alias(col))
    unified = unified.with_columns(
        pl.lit(f"{duid_primary}{_TRANSITION_DUID_SUFFIX}").alias("DUID")
    )
    print(
        f"[Transition] Unified dispatch signal: {unified.height} intervals "
        f"across {len(parts)} sub-periods "
        f"(transition at {transition_date.date() if transition_date else 'N/A'})"
    )
    return unified


# ---------------------------------------------------------------------------

def run_dispatch_replay(
    processed_data: pl.DataFrame,
    selection: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    region: str,
    cache_dir: str = "data/aemo",
    num_episodes: int = 1,
    step_duration: float = 0.5,
    battery_life_cost: float = 1_000_000.0,
    max_step: int = 288,
    output_dir: Optional[str] = None,
    run_tag: str = "dispatch",
    action_mode: str = "multi_market",
    degradation_mode: str = "rainflow",
    degradation_chemistry: str = "NMC",
    degradation_temperature: float = 25.0,
) -> Tuple[List[pl.DataFrame], List[pl.DataFrame], pl.DataFrame]:
    """Run dispatch replay episodes and save logs to parquet.

    Creates *num_episodes* independent ``AEMOBatteryTradingEnv`` instances and
    runs the ``AEMOAgent`` in ``algorithm='dispatch'`` mode for each one.  The
    dispatch actions are sourced from real AEMO DISPATCHLOAD data.

    Args:
        processed_data: Preprocessed AEMO market data (output of
            ``AEMODataPreprocessor.preprocess_aemo_data``).
        selection: Dictionary returned by :func:`resolve_dispatch_selection`.
        start_date: Start of the dispatch window to fetch.
        end_date: End of the dispatch window to fetch.
        region: AEMO region code used when fetching paired gen/load data.
        cache_dir: Path to the NEMOSIS data cache directory.
        num_episodes: Number of parallel episodes to run.
        step_duration: Step duration in hours (default 0.5 = 30 min).
        battery_life_cost: Battery degradation cost constant.
        max_step: Maximum steps per episode.
        output_dir: If provided, save ``{run_tag}_dispatch_logs.parquet``
            and ``{run_tag}_dispatch_incident_logs.parquet`` here.  If
            *None*, the parquet files are not written.
        run_tag: Prefix for output file names.
        action_mode: Environment action mode (default ``"multi_market"``).
        degradation_mode: Battery degradation model (default ``"rainflow"``).
        degradation_chemistry: Cell chemistry preset for the ``'real_world'``
            degradation model.  One of ``'NMC'`` or ``'LFP'``.
        degradation_temperature: Ambient / cell temperature in °C for
            degradation calculations.

    Returns:
        ``(episode_logs, incident_logs, all_logs_combined)``

        *episode_logs* — list of per-episode log DataFrames.
        *incident_logs* — list of per-episode incident DataFrames.
        *all_logs_combined* — all episode logs concatenated with an
        ``episode_id`` column.
    """
    from AEMOBatteryEnv import AEMOBatteryTradingEnv
    from aemo_data import fetch_aemo_unit_dispatch
    from decision import AEMOAgent, run_single

    duid = selection["duid"]
    selection_region = selection.get("region")
    dispatch_duid_gen = selection.get("dispatch_duid_gen")
    dispatch_duid_load = selection.get("dispatch_duid_load")
    dispatch_type = selection.get("dispatch_type")
    spans_transition = selection.get("spans_transition", False)
    all_dispatch_duids = selection.get("all_dispatch_duids") or [duid]
    transition_dates = selection.get("transition_dates") or []

    if selection_region and selection_region != region:
        raise ValueError(
            f"Dispatch replay region mismatch for {duid!r}: "
            f"selection region={selection_region!r}, replay region={region!r}."
        )

    if "REGIONID" in processed_data.columns:
        processed_regions = sorted(
            {
                str(value)
                for value in processed_data.get_column("REGIONID").drop_nulls().unique().to_list()
                if str(value).strip()
            }
        )
        if len(processed_regions) == 1 and processed_regions[0] != region:
            raise ValueError(
                f"Processed market data region mismatch for {duid!r}: "
                f"processed_data region={processed_regions[0]!r}, replay region={region!r}."
            )
        if selection_region and processed_regions and selection_region not in processed_regions:
            raise ValueError(
                f"Dispatch replay selection region mismatch for {duid!r}: "
                f"selection region={selection_region!r}, processed_data regions={processed_regions!r}."
            )

    # Create independent env instances
    dispatch_envs = [
        AEMOBatteryTradingEnv(
            aemo_data=processed_data,
            battery_capacity=selection["battery_capacity"],
            max_battery_flow=selection["max_battery_flow"],
            init_battery_level=selection["init_battery_level"],
            max_step=max_step,
            step_duration=step_duration,
            battery_life_cost=battery_life_cost,
            action_mode=action_mode,
            degradation_mode=degradation_mode,
            degradation_chemistry=degradation_chemistry,
            degradation_temperature=degradation_temperature,
        )
        for _ in range(num_episodes)
    ]

    # Fetch dispatch data
    if spans_transition and len(all_dispatch_duids) > 1:
        # Date range spans a DUID transition (e.g. Hornsdale HPRG1/HPRL1 → HPR1 Oct 2022).
        # Fetch ALL DUIDs for the full range and merge them into a unified dispatch signal
        # so the replay covers the complete episode without gaps at the transition boundary.
        print(
            f"[Transition] Fetching all DUIDs for transition-spanning range: {all_dispatch_duids}"
        )
        raw_all = fetch_aemo_unit_dispatch(
            start_date=start_date,
            end_date=end_date,
            duids=all_dispatch_duids,
            cache_dir=cache_dir,
        )
        dispatch_df = _merge_transition_dispatch(
            dispatch_df=raw_all,
            selection=selection,
            transition_dates=transition_dates,
        )
        # Reset paired gen/load — the merged signal is a single generator-convention series
        dispatch_duid_gen = None
        dispatch_duid_load = None
        dispatch_type = "generator"
        duid = f"{selection['duid']}{_TRANSITION_DUID_SUFFIX}"
    elif dispatch_duid_gen or dispatch_duid_load:
        # Use paired DUIDs directly — do NOT use region= here, because historical
        # DUIDs (e.g. HPRG1/HPRL1 before the Hornsdale re-registration) may no
        # longer appear in the current static generator table, causing the region
        # pre-filter to silently exclude them.
        paired_duids = [d for d in [dispatch_duid_gen, dispatch_duid_load] if d]
        dispatch_df = fetch_aemo_unit_dispatch(
            start_date=start_date,
            end_date=end_date,
            duids=paired_duids,
            cache_dir=cache_dir,
        )
    else:
        dispatch_df = fetch_aemo_unit_dispatch(
            start_date=start_date,
            end_date=end_date,
            duid=selection["duid"],
            cache_dir=cache_dir,
        )

    print(f"Dispatch rows fetched for {duid}: {dispatch_df.height}")
    if dispatch_df.height > 0:
        print(dispatch_df.head(5))
    else:
        raise ValueError(f"No dispatch data returned for DUID={duid!r}")

    episode_logs: List[pl.DataFrame] = []
    incident_logs: List[pl.DataFrame] = []

    for idx, env in enumerate(dispatch_envs):
        agent_kwargs: Dict[str, Any] = {
            "algorithm": "dispatch",
            "dispatch_data": dispatch_df,
        }
        if dispatch_duid_gen or dispatch_duid_load:
            agent_kwargs["dispatch_duid_gen"] = dispatch_duid_gen
            agent_kwargs["dispatch_duid_load"] = dispatch_duid_load
        else:
            agent_kwargs["dispatch_duid"] = duid
            agent_kwargs["dispatch_type"] = dispatch_type

        ep_df, inc_df = run_single(
            AEMOAgent,
            env,
            agent_kwargs=agent_kwargs,
            render=False,
            display_progress=False,
        )
        episode_logs.append(ep_df)
        incident_logs.append(inc_df)
        print(f"  Dispatch episode {idx}: {ep_df.height} steps, reward={ep_df['reward'].sum():.2f}")

    # Combine all episodes
    dfs_with_id = [
        df.with_columns(pl.lit(i).alias("episode_id"))
        for i, df in enumerate(episode_logs)
    ]
    all_logs_combined = pl.concat(dfs_with_id)

    # Save logs if output_dir is provided
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        out_file = out_path / f"{run_tag}_dispatch_logs.parquet"
        all_logs_combined.write_parquet(str(out_file))
        print(f"Saved dispatch logs → {out_file} ({all_logs_combined.height} rows)")

        incident_dfs = [
            df.with_columns(pl.lit(i).alias("episode_id"))
            for i, df in enumerate(incident_logs)
            if df.height > 0
        ]
        if incident_dfs:
            dispatch_inc = pl.concat(incident_dfs)
            inc_file = out_path / f"{run_tag}_dispatch_incident_logs.parquet"
            dispatch_inc.write_parquet(str(inc_file))
            print(f"Saved dispatch incident logs → {inc_file}")
        else:
            print("[INFO] No dispatch incident logs to save.")

    return episode_logs, incident_logs, all_logs_combined


# ---------------------------------------------------------------------------
# DUID availability exploration
# ---------------------------------------------------------------------------

def scan_duid_availability(
    regions: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> pl.DataFrame:
    """Return a summary DataFrame of battery DUID activity across regions.

    For each battery unit found in the AEMO static table (filtered to
    *regions*), the function reports:

    - Static metadata (``DispatchType``, ``StorageCapacityMWh``,
      ``RegisteredCapacityMW``).
    - If *start_date* and *end_date* are provided:
      ``DispatchIntervalCount``, ``NonZeroIntervalCount``, ``MaxEnergyMW``,
      ``FirstDispatchInterval``, ``LastDispatchInterval``.

    This is useful for quickly discovering which batteries were actively
    dispatched during a given period across all NEM regions.

    Args:
        regions: List of AEMO region codes to scan.  Defaults to all five
            NEM regions (``["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]``).
        start_date: Start of the date window for activity stats.  When
            ``None`` (the default), no DISPATCHLOAD data is fetched —
            the function returns only the static-table metadata.
        end_date: End of the date window.  Required when *start_date* is set.
        cache_dir: Path to the NEMOSIS data cache directory.
        refresh: When ``True``, force re-download of cached files.

    Returns:
        Polars DataFrame with one row per battery DUID.  Includes columns
        from the static table plus dispatch-activity columns (when
        *start_date*/*end_date* are provided).  An ``InDispatchLoad``
        boolean column indicates whether the unit appeared in DISPATCHLOAD
        during the window.
    """
    from aemo_data import AEMO_REGIONS, get_available_battery_units, get_dispatch_active_battery_units

    if regions is None:
        regions = list(AEMO_REGIONS)

    all_static: List[pl.DataFrame] = []
    for region in regions:
        units = get_available_battery_units(
            cache_dir=cache_dir,
            region=region,
            refresh=refresh,
        )
        if units.height > 0:
            all_static.append(units)

    if not all_static:
        return pl.DataFrame()

    static_df = pl.concat(all_static, how="diagonal_relaxed").unique(subset=["DUID"], keep="first")

    if start_date is None or end_date is None:
        return static_df.with_columns(pl.lit(None, dtype=pl.Boolean).alias("InDispatchLoad"))

    # Fetch activity stats per region and merge
    activity_frames: List[pl.DataFrame] = []
    for region in regions:
        active = get_dispatch_active_battery_units(
            start_date=start_date,
            end_date=end_date,
            region=region,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        if active.height > 0:
            activity_frames.append(active)

    if not activity_frames:
        return static_df.with_columns(pl.lit(False).alias("InDispatchLoad"))

    activity_df = pl.concat(activity_frames, how="diagonal_relaxed").unique(subset=["DUID"], keep="first")

    activity_cols = [
        c for c in [
            "DUID",
            "NonZeroIntervalCount",
            "DispatchIntervalCount",
            "MaxEnergyMW",
            "FirstDispatchInterval",
            "LastDispatchInterval",
            "PairedGenDUID",
            "PairedLoadDUID",
        ]
        if c in activity_df.columns
    ]

    result = static_df.join(
        activity_df.select(activity_cols),
        on="DUID",
        how="left",
    ).with_columns(
        pl.col("DispatchIntervalCount").is_not_null().alias("InDispatchLoad")
    )

    # Sort: active first, then by region and storage capacity
    return result.sort(
        ["InDispatchLoad", "NonZeroIntervalCount", "Region", "StorageCapacityMWh"],
        descending=[True, True, False, True],
        nulls_last=True,
    )


# ---------------------------------------------------------------------------
# Historical DUID availability
# ---------------------------------------------------------------------------

def scan_duid_historical_availability(
    duids: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    search_start: Optional[datetime] = None,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """Find the earliest DISPATCHLOAD date for each battery DUID across a historical range.

    This is useful for understanding when batteries became operational (or when
    their current DUID was first used after an AEMO re-registration).

    .. note:: **Why historical data might be missing for some DUIDs**

        AEMO's current static table (*NEM Registration and Exemption List*)
        only contains **current** DUID registrations.  Many batteries have
        changed their registration over time:

        * **Old model (pre-2021)**: Batteries were registered as *two* separate
          DUIDs — a *Generating Unit* (discharge, e.g. ``HPRG1``) and a *Load*
          (charge, e.g. ``HPRL1``).
        * **New model (2021+)**: AEMO introduced *Bidirectional Unit* DUIDs
          (e.g. ``HPR1``) where a single DUID handles both charge and discharge.

        When a battery transitions from the old to the new model, the old DUIDs
        are retired and the new Bidirectional DUID starts appearing in
        DISPATCHLOAD.  Therefore:

        * Scanning for ``HPR1`` (Hornsdale Power Reserve) in 2022 DISPATCHLOAD
          will find no results because the data at that time used ``HPRG1`` /
          ``HPRL1``.
        * The ``FirstDispatchInterval`` returned by this function is the first
          date the *current* DUID appears in DISPATCHLOAD — not the battery's
          commissioning date.

        To find historical data for a battery that has transitioned, you need to
        use its old DUID pair (gen + load) directly with
        :func:`aemo_data.fetch_aemo_unit_dispatch`.

    Args:
        duids: Explicit list of DUIDs to scan.  When *None*, all battery DUIDs
            from the static table for *regions* are used.
        regions: Region codes to use when *duids* is ``None``.  Defaults to all
            five NEM regions.
        search_start: Earliest date to scan.  Defaults to ``datetime(2019, 1, 1)``
            (around when the first grid-scale batteries came online in the NEM).
        cache_dir: Path to the NEMOSIS data cache directory.
        refresh: When ``True``, force re-download of cached files.
        verbose: Print month-by-month progress messages.

    Returns:
        Polars DataFrame with one row per DUID and columns:

        * ``DUID``
        * ``Region`` (from the static table, if *duids* was ``None``)
        * ``FirstDispatchInHistory`` — earliest ``SETTLEMENTDATE`` found, or
          ``null`` if the DUID was not found in the search range.
        * ``SearchStart`` / ``SearchEnd`` — the date range that was scanned.
    """
    from aemo_data import AEMO_REGIONS, get_available_battery_units, find_duid_first_dispatch

    if search_start is None:
        search_start = datetime(2019, 1, 1)

    search_end = datetime.now()

    # Build the DUID list
    duid_region_map: Dict[str, Optional[str]] = {}

    if duids is not None:
        for d in duids:
            duid_region_map[d] = None
    else:
        if regions is None:
            regions = list(AEMO_REGIONS)
        for region in regions:
            units = get_available_battery_units(
                cache_dir=cache_dir,
                region=region,
                refresh=refresh,
            )
            for row in units.iter_rows(named=True):
                duid_region_map[row["DUID"]] = row.get("Region")

    if not duid_region_map:
        print("[INFO] No DUIDs to scan.")
        return pl.DataFrame({
            "DUID": pl.Series([], dtype=pl.Utf8),
            "Region": pl.Series([], dtype=pl.Utf8),
            "FirstDispatchInHistory": pl.Series([], dtype=pl.Datetime),
            "SearchStart": pl.Series([], dtype=pl.Datetime),
            "SearchEnd": pl.Series([], dtype=pl.Datetime),
        })

    print(
        f"Scanning {len(duid_region_map)} DUID(s) for earliest DISPATCHLOAD record "
        f"from {search_start.date()} onward…"
    )

    rows: List[Dict[str, Any]] = []
    for duid, region in duid_region_map.items():
        print(f"\n[DUID={duid!r}]")
        first_dt = find_duid_first_dispatch(
            duid=duid,
            search_start=search_start,
            search_end=search_end,
            cache_dir=cache_dir,
            refresh=refresh,
            verbose=verbose,
        )
        rows.append({
            "DUID": duid,
            "Region": region,
            "FirstDispatchInHistory": first_dt,
            "SearchStart": search_start,
            "SearchEnd": search_end,
        })

    result = pl.DataFrame(rows, schema_overrides={
        "FirstDispatchInHistory": pl.Datetime,
        "SearchStart": pl.Datetime,
        "SearchEnd": pl.Datetime,
    })

    # Sort: DUIDs with earliest first dispatch first, then by region
    return result.sort(
        ["FirstDispatchInHistory", "Region", "DUID"],
        descending=[False, False, False],
        nulls_last=True,
    )
