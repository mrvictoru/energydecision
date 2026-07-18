# AEMO dispatch replay guide and API reference (`dispatch_utils`)

This is the **dispatch replay guide and API reference** for `src/dispatch_utils.py`.

Use this document when you need:

- to find candidate battery DUIDs
- to resolve station sizing and dispatch metadata
- to run replay episodes from real AEMO dispatch schedules
- the `dispatch_utils` function reference

## New: Direct Episode Converter

For full 8-service FCAS support, use the new **direct episode converter** instead of the env-simulated dispatch replay. It reads AEMO DISPATCHLOAD + DISPATCHPRICE data directly and produces DT-compatible episode logs without running the environment:

```bash
python3 scripts/convert_dispatch_to_episodes.py \
    --station dalrymple_north \
    --start-date 2024-01-01 \
    --end-date 2024-07-01 \
    --output data/aemo_dispatch_episodes/dalrymple_north_2024h1.parquet
```

The converter captures all 8 FCAS services from actual AEMO-cleared values, reconstructs the SoC trajectory from energy dispatch, and computes revenue using actual market prices. See `scripts/convert_dispatch_to_episodes.py` for details.

**Note on FCAS coverage:** The legacy `multi_market` dispatch replay (via `AEMOAgent` with `algorithm='dispatch'`) only captures RAISEREG and LOWERREG. The direct converter (`full_fcas` output) captures all 8 services.

If you want the broader notebook workflow, read [workflow.md](workflow.md). If you want the full AEMO docs map, start with [README.md](README.md).

The `dispatch_utils` module (``src/dispatch_utils.py``) provides high-level
helpers for **dispatch replay** simulations.  It bridges the raw data-fetching
functions in `aemo_data` with `AEMOBatteryTradingEnv` and `AEMOAgent` so that
notebooks stay clean and readable.

---

## Overview

Dispatch replay re-runs a battery's real AEMO dispatch schedule inside the
simulation environment.  The actual MW targets and FCAS enablement are loaded
from the `DISPATCHLOAD` table via NEMOSIS and fed to the `AEMOAgent` as
pre-computed actions.

### Typical workflow

```python
from dispatch_utils import (
    list_dispatch_candidates,
    resolve_dispatch_selection,
    run_dispatch_replay,
    scan_duid_availability,
)
```

1. **Discover** which batteries were dispatched during a window:

   ```python
   battery_units, active_units = list_dispatch_candidates(
       region="SA1",
       start_date=datetime(2025, 1, 1),
       end_date=datetime(2025, 1, 7),
       cache_dir="data/aemo",
   )
   ```

2. **Select** a DUID and resolve environment sizing:

   ```python
   selection = resolve_dispatch_selection(
       battery_units=battery_units,
       active_battery_units=active_units,
       selected_duid="HPRG1",        # or use selected_index=0
       battery_capacity=10.0,        # default fallback (MWh)
       max_battery_flow=5.0,         # default fallback (MW)
       init_soc=5.0,
       apply_unit_sizing=True,       # use static-table sizing if available
       start_date=datetime(2025, 1, 1),
       end_date=datetime(2025, 1, 7),
   )
   ```

3. **Run** dispatch replay episodes:

   ```python
   ep_logs, inc_logs, all_logs = run_dispatch_replay(
       processed_data=processed_data,   # from AEMODataPreprocessor
       selection=selection,
       start_date=datetime(2025, 1, 1),
       end_date=datetime(2025, 1, 7),
       region="SA1",
       cache_dir="data/aemo",
       num_episodes=3,
       output_dir="data/aemo_sim_output",
       run_tag="aemo_mm",
    )
    ```

For evaluator baselines, pick stations that actually have `DISPATCHLOAD` coverage in the
chosen window. In the SA1 `2024-07-01` → `2024-07-14` window, `dalrymple_north` and
`torrens_island` both replay successfully, while `hornsdale` and `lake_bonney` do not.

---

## API Reference

### `show_dispatch_table(df, columns, label="", limit=20)`

Pretty-print a labelled, column-filtered view of a Polars DataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pl.DataFrame` | DataFrame to display |
| `columns` | `list[str]` | Columns to show (missing columns skipped silently) |
| `label` | `str` | Optional heading line printed before the table |
| `limit` | `int` | Maximum number of rows (default 20) |

---

### `list_dispatch_candidates(region, start_date, end_date, cache_dir, refresh)`

List all battery DUIDs in a region and the subset that appeared in
`DISPATCHLOAD` during a date window.

Returns `(battery_units, active_battery_units)` — both Polars DataFrames.

The `active_battery_units` table includes:

| Column | Description |
|--------|-------------|
| `DUID` | Dispatch Unit ID |
| `Region` | NEM region |
| `DispatchType` | `"Bidirectional Unit"`, `"Generating Unit"`, or `"Load"` |
| `StorageCapacityMWh` | Battery storage capacity from static table |
| `RegisteredCapacityMW` | Max power rating from static table |
| `NonZeroIntervalCount` | Number of 5-min slots with any non-zero dispatch value |
| `DispatchIntervalCount` | Total 5-min intervals in DISPATCHLOAD |
| `MaxEnergyMW` | Peak absolute `TOTALCLEARED` value |
| `FirstDispatchInterval` | Timestamp of first dispatch record |
| `LastDispatchInterval` | Timestamp of last dispatch record |
| `PairedGenDUID` | Generator DUID of a paired gen/load battery (old model) |
| `PairedLoadDUID` | Load DUID of a paired gen/load battery (old model) |

> **Tip**: Sort by `NonZeroIntervalCount` descending to find the most active
> batteries first.  A unit with `NonZeroIntervalCount=0` was registered but
> not dispatched during the window — replay will produce all-zero actions.

---

### `resolve_dispatch_selection(...)`

Select a DUID from the candidate tables and return a `selection` dict ready
for `run_dispatch_replay`.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `battery_units` | — | Full static-table listing |
| `active_battery_units` | — | Dispatch-active subset (may be empty) |
| `selected_duid` | `None` | Explicit DUID string; if `None`, uses `selected_index` |
| `selected_index` | `0` | Row index from the active (or static) table |
| `apply_unit_sizing` | `True` | Override env params with static-table sizing |
| `start_date` / `end_date` | `None` | Run availability check if provided |

Returns a dict with keys:

```python
{
    "duid": str,                   # selected DUID
    "region": str | None,          # resolved station / unit region
    "station_name": str | None,    # resolved registry station name
    "station_key": str | None,     # resolved registry key
    "dispatch_type": str | None,   # "Bidirectional Unit" / "Generating Unit" / "Load"
    "dispatch_duid_gen": str|None, # paired generator DUID (old-model batteries)
    "dispatch_duid_load": str|None,# paired load DUID (old-model batteries)
    "battery_capacity": float,     # resolved MWh
    "max_battery_flow": float,     # resolved MW
    "init_battery_level": float,   # resolved initial SOC (MWh)
    "availability": dict | None,   # output of check_aemo_dispatch_availability()
}
```

---

### `run_dispatch_replay(processed_data, selection, start_date, end_date, region, ...)`

Run N dispatch replay episodes.  Returns `(episode_logs, incident_logs, all_logs_combined)`.

Dispatch replay is region-bound. The requested `region`, the resolved `selection["region"]`,
and any single-region provenance carried by `processed_data` must agree. If they do not,
`run_dispatch_replay(...)` raises `ValueError` instead of replaying one station against a
different market region.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `processed_data` | — | Preprocessed market data (from `AEMODataPreprocessor`) |
| `selection` | — | Dict returned by `resolve_dispatch_selection` |
| `num_episodes` | `1` | Number of independent episodes |
| `output_dir` | `None` | Directory for parquet log files (skipped if `None`) |
| `run_tag` | `"dispatch"` | Prefix for output file names |
| `action_mode` | `"multi_market"` or `"full_fcas"` | Environment action mode |

Output files (when `output_dir` is provided):
- `{run_tag}_dispatch_logs.parquet` — all episode logs with `episode_id` column
- `{run_tag}_dispatch_incident_logs.parquet` — incident logs (if any)

---

### `scan_duid_availability(regions, start_date, end_date, cache_dir, refresh)`

Return a single DataFrame spanning **all NEM regions** with static-table metadata
and (optionally) dispatch-activity statistics.

| Column | Description |
|--------|-------------|
| `InDispatchLoad` | `True` if the unit appeared in DISPATCHLOAD during the window |
| `NonZeroIntervalCount` | Non-zero dispatch intervals |
| `MaxEnergyMW` | Peak energy dispatch |

Use this to quickly discover which batteries across all regions are suitable for
dispatch replay during a given period.

```python
summary = scan_duid_availability(
    regions=["SA1", "VIC1"],
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 7),
)
summary.filter(pl.col("InDispatchLoad")).sort("NonZeroIntervalCount", descending=True)
```

---

### `scan_duid_historical_availability(duids, regions, search_start, cache_dir, refresh, verbose)`

Scan DISPATCHLOAD month-by-month forward from `search_start` and return the
**earliest dispatch date** for each DUID.  This answers *"when was this battery
first operational under its current DUID?"*.

Returns a Polars DataFrame with columns `DUID`, `Region`,
`FirstDispatchInHistory`, `SearchStart`, `SearchEnd`.

```python
from dispatch_utils import scan_duid_historical_availability
from datetime import datetime

history = scan_duid_historical_availability(
    regions=["SA1"],
    search_start=datetime(2022, 1, 1),
    cache_dir="data/aemo",
)
print(history)
```

See also `aemo_data.find_duid_first_dispatch()` for the per-DUID function that
underlies this scan.

---

## Why older dates show zero dispatch activity

When running `scan_duid_availability` or `list_dispatch_candidates` for dates
before 2024, many battery DUIDs will show **zero activity** or not appear in
DISPATCHLOAD at all.  This is expected and has two root causes:

### 1. Newly commissioned batteries

Most batteries in the current AEMO static table were built between 2022 and 2025:

| Battery | DUID | Commissioned ≈ |
|---------|------|----------------|
| Waratah Super Battery (NSW) | `WTAHB1` | Dec 2024 |
| Torrens Island BESS (SA) | `TIB1` | 2024 |
| Blyth BESS (SA) | `BLYTHB1` | 2023 |
| Western Downs BESS (QLD) | `WDBESS1/2` | 2023–2024 |
| Victorian Big Battery (VIC) | `VBB1` | Nov 2021 |

### 2. DUID re-registration (gen/load pairs → Bidirectional Units)

Before 2021–2022, AEMO registered batteries as *two* separate DUIDs:
- a **Generating Unit** DUID for the discharge direction (e.g. `HPRG1`)
- a **Load** DUID for the charge direction (e.g. `HPRL1`)

Starting from 2021, AEMO introduced **Bidirectional Unit** DUIDs where a single
DUID handles both directions (e.g. `HPR1`).  When a battery re-registers, the
old DUIDs are retired and the new Bidirectional DUID starts appearing in
DISPATCHLOAD from the transition date.

**Known transitions**:

| Current DUID | Old Gen DUID | Old Load DUID | Re-registered ≈ |
|-------------|-------------|--------------|-----------------|
| `HPR1` (Hornsdale, SA1) | `HPRG1` | `HPRL1` | 2022 |
| `LBB1` (Lake Bonney, SA1) | `LBBG1` | `LBBL1` | 2022 |
| `BALB1` (Ballarat, VIC1) | `BALBG1` | — | 2023 |
| `GANNB1` (Gannawarra, VIC1) | `GANNBG1` | `GANNBL1` | 2023 |
| `DALNTH1` (Dalrymple North, SA1) | `DALNTH01` | — | 2023 |

To fetch data for a battery before its re-registration date, use the older DUID
pair directly:

```python
from aemo_data import fetch_aemo_unit_dispatch
from datetime import datetime

# Hornsdale historical data using the OLD generator DUID
old_dispatch = fetch_aemo_unit_dispatch(
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 1, 7),
    duid="HPRG1",
    cache_dir="data/aemo",
)
```

---

## Paired gen/load batteries (old AEMO model)

Some batteries (particularly older registrations) are modelled as **two separate
DUIDs**:

| DUID suffix | Direction | `DispatchType` |
|-------------|-----------|----------------|
| `*G1` | Discharge → grid | `"Generating Unit"` |
| `*L1` | Charge ← grid | `"Load"` |

For example `KEPBG1` (gen) + `KEPBL1` (load) form the Kennedy Energy Park
battery.

`list_dispatch_candidates` and `resolve_dispatch_selection` automatically
detect these pairs and populate `PairedGenDUID` / `PairedLoadDUID`.  When
both are present, `run_dispatch_replay` fetches data for both DUIDs and passes
them as `dispatch_duid_gen` / `dispatch_duid_load` to `AEMOAgent.set_dispatch_data`
so the net energy action is computed correctly (`LOAD_MW − GEN_MW`).

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/aemo_simrun.ipynb` | Configurable simulation pipeline with rule / dispatch / SB3 runs and battery-size sweeps |
| `notebooks/test_aemo_data.ipynb` | Data exploration; section 5 demonstrates `list_dispatch_candidates`, `scan_duid_availability`, and `scan_duid_historical_availability` |
