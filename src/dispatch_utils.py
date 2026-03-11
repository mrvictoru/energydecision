"""
Dispatch Replay Utilities for AEMO Battery Trading Environments
==============================================================

This module provides high-level helpers for selecting, resolving, and running
*dispatch replay* simulations using real AEMO DISPATCHLOAD data.  It bridges
the raw data-fetching functions in ``aemo_data`` with the ``AEMOBatteryTradingEnv``
and ``AEMOAgent`` so that notebooks stay clean and readable.

Typical workflow
----------------
1. Call :func:`list_dispatch_candidates` to see which battery DUIDs are available
   and which were actually dispatched during a chosen date window.
2. Call :func:`resolve_dispatch_selection` to pick a DUID (by name or index) and
   obtain sizing information and DUID metadata.
3. Call :func:`run_dispatch_replay` to run N episodes with the dispatch replay
   agent and save the logs to parquet files.

Public API
----------
- :func:`show_dispatch_table` — pretty-print a subset of columns from a DataFrame
- :func:`list_dispatch_candidates` — list available battery DUIDs and their
  dispatch-activity statistics for a given region and date window
- :func:`resolve_dispatch_selection` — select a DUID and resolve env-sizing /
  paired gen–load metadata
- :func:`run_dispatch_replay` — run dispatch replay episodes and save logs

See ``docs/AEMO_DISPATCH_UTILS.md`` for a detailed usage guide.
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


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

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

def list_dispatch_candidates(
    region: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: str = "data/aemo",
    refresh: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Return all battery DUIDs and the subset that was active in DISPATCHLOAD.

    Queries the AEMO static table for batteries in *region* and then checks
    DISPATCHLOAD to find those that were actually dispatched between *start_date*
    and *end_date*.

    Args:
        region: AEMO region code (e.g. ``"SA1"``, ``"QLD1"``).
        start_date: Start of the date window (inclusive).
        end_date: End of the date window (inclusive).
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
    from aemo_data import get_available_battery_units, get_dispatch_active_battery_units

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
    from aemo_data import check_aemo_dispatch_availability

    source_table = active_battery_units if active_battery_units.height > 0 else battery_units
    source_label = "dispatch-active" if active_battery_units.height > 0 else "static"

    if source_table.height == 0:
        raise ValueError("No battery DUIDs are available for dispatch replay.")

    if selected_duid and str(selected_duid).strip():
        duid = str(selected_duid).strip()
        chosen = source_table.filter(pl.col("DUID") == duid).head(1)
        if chosen.height == 0:
            chosen = battery_units.filter(pl.col("DUID") == duid).head(1)
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

    # Resolve paired gen/load DUIDs
    dispatch_type: Optional[str] = None
    dispatch_duid_gen: Optional[str] = None
    dispatch_duid_load: Optional[str] = None
    if chosen.height > 0:
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

    if dispatch_duid_gen or dispatch_duid_load:
        print(
            f"Paired gen/load DUID detected: gen={dispatch_duid_gen!r}, load={dispatch_duid_load!r}\n"
            "The replay will fetch both units and combine them."
        )

    # Availability check
    availability: Optional[Dict[str, Any]] = None
    if start_date is not None and end_date is not None:
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

    # Sizing
    suggested_mwh: Optional[float] = None
    suggested_mw: Optional[float] = None
    if chosen.height > 0:
        if "StorageCapacityMWh" in chosen.columns:
            suggested_mwh = chosen["StorageCapacityMWh"][0]
        if "RegisteredCapacityMW" in chosen.columns:
            suggested_mw = chosen["RegisteredCapacityMW"][0]
    print(f"Suggested env sizing from table → capacity={suggested_mwh} MWh, max_flow={suggested_mw} MW")

    resolved_capacity = battery_capacity
    resolved_max_flow = max_battery_flow
    if apply_unit_sizing:
        if suggested_mwh is not None and np.isfinite(float(suggested_mwh)) and float(suggested_mwh) > 0:
            resolved_capacity = float(suggested_mwh)
        if suggested_mw is not None and np.isfinite(float(suggested_mw)) and float(suggested_mw) > 0:
            resolved_max_flow = float(suggested_mw)

    init_soc_ratio = (init_soc / battery_capacity) if battery_capacity > 0 else _DEFAULT_SOC_RATIO
    resolved_init_soc = float(np.clip(resolved_capacity * init_soc_ratio, 0.0, resolved_capacity))
    print(
        f"Dispatch replay env params → capacity={resolved_capacity:.3f} MWh, "
        f"max_flow={resolved_max_flow:.3f} MW, init_soc={resolved_init_soc:.3f} MWh"
    )

    return {
        "duid": duid,
        "dispatch_type": dispatch_type,
        "dispatch_duid_gen": dispatch_duid_gen,
        "dispatch_duid_load": dispatch_duid_load,
        "battery_capacity": resolved_capacity,
        "max_battery_flow": resolved_max_flow,
        "init_battery_level": resolved_init_soc,
        "availability": availability,
    }


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
    dispatch_duid_gen = selection.get("dispatch_duid_gen")
    dispatch_duid_load = selection.get("dispatch_duid_load")
    dispatch_type = selection.get("dispatch_type")

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
        )
        for _ in range(num_episodes)
    ]

    # Fetch dispatch data
    if dispatch_duid_gen or dispatch_duid_load:
        raw_dispatch = fetch_aemo_unit_dispatch(
            start_date=start_date,
            end_date=end_date,
            region=region,
            cache_dir=cache_dir,
        )
        paired_duids = [d for d in [dispatch_duid_gen, dispatch_duid_load] if d]
        dispatch_df = raw_dispatch.filter(pl.col("DUID").is_in(paired_duids))
    else:
        dispatch_df = fetch_aemo_unit_dispatch(
            start_date=start_date,
            end_date=end_date,
            duid=duid,
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
