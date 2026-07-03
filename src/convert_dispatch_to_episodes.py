"""Convert raw AEMO DISPATCHLOAD + DISPATCHPRICE data directly into
episode logs (Parquet) **without running the simulator environment**.

Unlike ``generate_dispatch_replays.py`` (legacy, energy-only), this
converter captures **all 8 FCAS services** by reading the actual AEMO
cleared enablement values from DISPATCHLOAD and the actual FCAS prices
from DISPATCHPRICE.

The output Parquet uses the same schema as environment-generated logs:
``episode_id``, ``step``, ``norm_observation`` (18-dim), ``action`` (9-dim
full_fcas) and ``reward``.

Usage::

    python3 src/convert_dispatch_to_episodes.py \
        --station dalrymple_north \
        --start-date 2024-01-01 \
        --end-date 2024-07-01 \
        --output data/aemo_dispatch_episodes/dalrymple_north_2024h1.parquet

    # Multiple stations
    python3 src/convert_dispatch_to_episodes.py \
        --station dalrymple_north --station hornsdale \
        --start-date 2024-01-01 --end-date 2024-07-01 \
        --output-dir data/aemo_dispatch_episodes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


# All 8 FCAS services ordered to match AEMOBatteryTradingEnv._fcas_services
FCAS_SERVICES = [
    "RAISEREG", "LOWERREG", "RAISE6SEC", "LOWER6SEC",
    "RAISE60SEC", "LOWER60SEC", "RAISE5MIN", "LOWER5MIN",
]

# Default round-trip efficiency (if not in BATTERY_REGISTRY)
DEFAULT_RTE = 0.88


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# SoC reconstruction
# ---------------------------------------------------------------------------

def reconstruct_soc(
    net_mw: np.ndarray,
    step_duration: float,
    capacity_mwh: float,
    init_soc_mwh: float,
    rte: float,
) -> np.ndarray:
    """Integrate energy dispatch to produce a SoC trajectory (MWh).

    ``net_mw`` > 0 means charging, < 0 means discharging.
    """
    soc = np.zeros(len(net_mw) + 1, dtype=np.float64)
    soc[0] = init_soc_mwh
    eff_charge = np.sqrt(rte)
    eff_discharge = np.sqrt(rte)
    for i, mw in enumerate(net_mw):
        energy = mw * step_duration
        if energy > 0:  # charging
            soc[i + 1] = soc[i] + energy * eff_charge
        else:  # discharging
            soc[i + 1] = soc[i] + energy / eff_discharge
        soc[i + 1] = np.clip(soc[i + 1], 0.0, capacity_mwh)
    return soc


# ---------------------------------------------------------------------------
# Revenue computation
# ---------------------------------------------------------------------------

def compute_episode_reward(
    aligned: pl.DataFrame,
    soc: np.ndarray,
    step_duration: float,
    capacity_mwh: float,
    battery_life_cost: float,
    degradation_model: str = "rainflow",
    degradation_temperature: float = 30.0,
) -> pl.DataFrame:
    """Compute per-step revenue, degradation, and reward for the episode.

    Returns a DataFrame with columns:
    ``reward, energy_revenue, fcas_revenue, fcas_<SERVICE>_revenue (×8),
    step_degradation, total_degradation, battery_soc, actual_energy``.
    """
    from batterydeg import DegradationModel, RainflowCounter

    n = aligned.height

    # Energy revenue
    rrp = aligned["RRP"].to_numpy()
    # Net MW > 0 = charging (buy), < 0 = discharging (sell)
    net_mw = aligned["NET_MW"].to_numpy()
    energy_mwh = net_mw * step_duration
    energy_revenue = np.where(
        net_mw < 0,
        np.abs(energy_mwh) * rrp,   # discharging → sell
        -np.abs(energy_mwh) * rrp,  # charging → buy
    ).astype(np.float64)

    # FCAS revenue (all 8 services)
    fcas_revenue_by_service: dict[str, np.ndarray] = {}
    fcas_total = np.zeros(n, dtype=np.float64)
    for svc in FCAS_SERVICES:
        enabled_col = f"{svc}_MW"
        price_col = f"FCAS_{svc}"
        if enabled_col in aligned.columns and price_col in aligned.columns:
            enabled_mw = aligned[enabled_col].to_numpy().astype(np.float64)
            price = aligned[price_col].to_numpy().astype(np.float64)
        else:
            enabled_mw = np.zeros(n, dtype=np.float64)
            price = np.zeros(n, dtype=np.float64)
        svc_rev = enabled_mw * price * step_duration
        fcas_revenue_by_service[svc] = svc_rev
        fcas_total += svc_rev

    # Degradation
    soc_pct = (soc[:-1] / capacity_mwh) * 100.0
    step_degradation = np.zeros(n, dtype=np.float64)
    total_deg = 0.0

    if degradation_model == "rainflow":
        deg_model = DegradationModel()
        max_c_rate = float(aligned["max_power_mw"][0]) / capacity_mwh if "max_power_mw" in aligned.columns else 1.0
        rf = RainflowCounter(step_duration=step_duration, max_c_rate=max_c_rate)
        for i in range(n):
            cycles = rf.update(float(soc_pct[i]))
            for SoC_avg, DoD, Id, Ich in cycles:
                inc, _ = deg_model.degradation_per_cycle(
                    T=degradation_temperature, Id=Id, Ich=Ich,
                    SOCav=SoC_avg, DOD=DoD,
                )
                step_degradation[i] += float(inc)
            total_deg = min(1.0, total_deg + step_degradation[i])
    else:
        # Simple linear degradation
        for i in range(n):
            dod = abs(energy_mwh[i]) / capacity_mwh
            step_degradation[i] = dod * 0.0001
            total_deg = min(1.0, total_deg + step_degradation[i])

    degradation_cost = step_degradation * battery_life_cost

    # Total reward (same normalization as env: / 1000)
    reward = (energy_revenue + fcas_total - degradation_cost) / 1000.0

    result = pl.DataFrame({
        "reward": reward,
        "energy_revenue": energy_revenue,
        "fcas_revenue": fcas_total,
        "step_degradation": step_degradation,
        "total_degradation": np.minimum(np.cumsum(step_degradation), 1.0),
        "battery_soc": soc[:-1],
        "actual_energy": energy_mwh,
    })
    for svc in FCAS_SERVICES:
        result = result.with_columns(
            pl.Series(f"fcas_{svc}_revenue", fcas_revenue_by_service[svc])
        )
    return result


# ---------------------------------------------------------------------------
# Observation construction
# ---------------------------------------------------------------------------

def build_observations(aligned: pl.DataFrame, soc: np.ndarray, capacity_mwh: float) -> np.ndarray:
    """Build 18-dim normalized observations matching AEMOBatteryTradingEnv.

    Layout: [hour_sin, hour_cos, day_sin, day_cos, is_peak,
             RRP_normalized, DEMAND_normalized,
             8 × FCAS_normalized,
             solar_pct, wind_pct,
             SOC_normalized]
    """
    n = aligned.height

    def _to_datetime(value: Any) -> datetime:
        if isinstance(value, np.datetime64):
            try:
                return datetime.utcfromtimestamp(int(value.astype("datetime64[us]").astype("int64")) / 1e6)
            except Exception:
                return datetime.utcfromtimestamp(int(value.astype("int64")) / 1e9)
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        return value

    timestamps = aligned["SETTLEMENTDATE"].to_numpy()
    datetimes = [_to_datetime(t) for t in timestamps]
    hours = np.array([t.hour for t in datetimes], dtype=np.float32)
    days_of_year = np.array([t.timetuple().tm_yday for t in datetimes], dtype=np.float32)

    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    day_sin = np.sin(2 * np.pi * days_of_year / 365.25)
    day_cos = np.cos(2 * np.pi * days_of_year / 365.25)
    is_peak = ((hours >= 7) & (hours < 10) | (hours >= 16) & (hours < 21)).astype(np.float32)

    # Normalize market data
    def _norm(col, default_min=0.0, default_max=1.0):
        if col not in aligned.columns:
            return np.zeros(n, dtype=np.float32)
        vals = aligned[col].to_numpy().astype(np.float64)
        vmin = float(np.nanmin(vals)) if len(vals) > 0 else default_min
        vmax = float(np.nanmax(vals)) if len(vals) > 0 else default_max
        if vmax - vmin < 1e-9:
            return np.zeros(n, dtype=np.float32)
        return ((vals - vmin) / (vmax - vmin)).clip(0.0, 1.0).astype(np.float32)

    rrp_norm = _norm("RRP")
    demand_norm = _norm("TOTALDEMAND")
    fcas_norms = [_norm(f"FCAS_{svc}") for svc in FCAS_SERVICES]

    gen_solar = _norm("GEN_solar")
    gen_wind = _norm("GEN_wind")

    soc_norm = (soc[:-1] / capacity_mwh).clip(0.0, 1.0).astype(np.float32)

    obs = np.stack([
        hour_sin, hour_cos, day_sin, day_cos, is_peak,
        rrp_norm, demand_norm,
        *fcas_norms,
        gen_solar, gen_wind,
        soc_norm,
    ], axis=1).astype(np.float32)
    return obs


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------

def convert_station_to_episode(
    station_name: str,
    start_date: datetime,
    end_date: datetime,
    cache_dir: str,
    processed_data: pl.DataFrame | None = None,
    step_duration: float = 0.5,
    degradation_mode: str = "rainflow",
    degradation_temperature: float = 30.0,
    battery_life_cost: float | None = None,
) -> pl.DataFrame | None:
    """Convert AEMO dispatch data for one station into an episode DataFrame.

    Returns a DataFrame with columns matching DT trajectory log schema:
    ``episode_id, step, norm_observation, action, reward``
    plus additional ``info`` columns for analysis.

    Returns ``None`` if no dispatch data is available.
    """
    sys.path.insert(0, str(repo_root() / "src"))
    from aemo_data import (
        BATTERY_REGISTRY,
        fetch_aemo_unit_dispatch,
        fetch_aemo_data_bundle,
    )
    from AEMOBatteryEnv import AEMODataPreprocessor
    from dispatch_utils import list_dispatch_candidates, resolve_dispatch_selection

    info = BATTERY_REGISTRY.get(station_name)
    if not info:
        print(f"  Station '{station_name}' not found in BATTERY_REGISTRY")
        return None

    region = info.get("region", "?")
    capacity_mwh = float(info.get("capacity_mwh", 10.0))
    max_power_mw = float(info.get("max_power_mw", 5.0))
    rte = float(info.get("efficiency", DEFAULT_RTE))
    init_soc_mwh = capacity_mwh * 0.5

    if battery_life_cost is None:
        battery_life_cost = 400_000.0  # typical BESS replacement cost

    print(f"\n  [{station_name}] ({region}) — fetching dispatch data...")
    try:
        battery_units, active_units = list_dispatch_candidates(
            region=region,
            start_date=start_date,
            end_date=end_date,
            station_name=station_name,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"  ERROR checking {station_name}: {e}")
        return None

    if active_units.height == 0:
        print(f"  [{station_name}]: no dispatch data in this window — skipping")
        return None

    print(f"  [{station_name}]: ACTIVE — building dispatch selection...")
    selection = resolve_dispatch_selection(
        battery_units=battery_units,
        active_battery_units=active_units,
        selected_index=0,
        apply_unit_sizing=True,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )

    # Fetch DISPATCHLOAD data
    dispatch_duid = selection.get("dispatch_duid") or selection.get("duid")
    dispatch_duid_gen = selection.get("dispatch_duid_gen")
    dispatch_duid_load = selection.get("dispatch_duid_load")

    duid_list: list[str] = []
    if dispatch_duid:
        duid_list = [str(dispatch_duid)]
    elif dispatch_duid_gen or dispatch_duid_load:
        duid_list = [str(d) for d in [dispatch_duid_gen, dispatch_duid_load] if d]
    if not duid_list:
        fallback_duids = selection.get("all_dispatch_duids") or []
        if fallback_duids:
            duid_list = [str(d) for d in fallback_duids if d]

    if not duid_list:
        print(f"  [{station_name}]: no DUIDs to fetch — skipping")
        return None

    print(f"  [{station_name}]: fetching DISPATCHLOAD for {duid_list}...")
    dispatch_data = fetch_aemo_unit_dispatch(
        start_date=start_date,
        end_date=end_date,
        duids=duid_list,
        cache_dir=cache_dir,
    )
    if dispatch_data is None or dispatch_data.height == 0:
        print(f"  [{station_name}]: no DISPATCHLOAD rows returned — skipping")
        return None

    # Fetch/regenerate processed market data
    if processed_data is None:
        print(f"  [{station_name}]: fetching market data for {region}...")
        data = fetch_aemo_data_bundle(
            start_date=start_date,
            end_date=end_date,
            region=region,
            fcas_services=FCAS_SERVICES,
            fuel_types=["solar", "wind"],
            cache_dir=cache_dir,
        )
        preprocessor = AEMODataPreprocessor(step_duration_hours=step_duration)
        processed_data = preprocessor.preprocess_aemo_data(
            prices=data["prices"],
            fcas=data["fcas"],
            generation=data["generation"],
        )

    # Align dispatch data to processed market data timestamps
    every_minutes = int(round(step_duration * 60))
    every = f"{every_minutes}m"

    def _prep(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(pl.col("SETTLEMENTDATE").cast(pl.Datetime, strict=False))
        df = df.sort("SETTLEMENTDATE")
        numeric_cols = [c for c in df.columns if c not in {"SETTLEMENTDATE", "DUID"}]
        aggs = [pl.col(c).mean().alias(c) for c in numeric_cols]
        return df.group_by_dynamic("SETTLEMENTDATE", every=every, label="left", closed="left").agg(aggs)

    # Build dispatch actions (same logic as _build_dispatch_actions in decision.py)
    df = dispatch_data
    if "DUID" in df.columns and (dispatch_duid_gen or dispatch_duid_load):
        gen_df = None
        load_df = None
        if dispatch_duid_gen:
            gen_rename = {"TOTALCLEARED": "GEN_MW"}
            gen_rename.update({svc: f"GEN_{svc}" for svc in FCAS_SERVICES})
            gen_df = _prep(df.filter(pl.col("DUID") == dispatch_duid_gen)).rename(gen_rename)
        if dispatch_duid_load:
            load_rename = {"TOTALCLEARED": "LOAD_MW"}
            load_rename.update({svc: f"LOAD_{svc}" for svc in FCAS_SERVICES})
            load_df = _prep(df.filter(pl.col("DUID") == dispatch_duid_load)).rename(load_rename)

        merged = gen_df if gen_df is not None else load_df
        if merged is None:
            return None
        if gen_df is not None and load_df is not None:
            merged = gen_df.join(load_df, on="SETTLEMENTDATE", how="full", coalesce=True)

        required_cols = {"GEN_MW": 0.0, "LOAD_MW": 0.0}
        for svc in FCAS_SERVICES:
            required_cols[f"GEN_{svc}"] = 0.0
            required_cols[f"LOAD_{svc}"] = 0.0
        missing_exprs = [pl.lit(v).alias(c) for c, v in required_cols.items() if c not in merged.columns]
        if missing_exprs:
            merged = merged.with_columns(missing_exprs)
        merged = merged.fill_null(0.0)

        sum_exprs = [(pl.col("LOAD_MW").fill_null(0.0) - pl.col("GEN_MW").fill_null(0.0)).alias("NET_MW")]
        for svc in FCAS_SERVICES:
            sum_exprs.append(
                (pl.col(f"GEN_{svc}").fill_null(0.0) + pl.col(f"LOAD_{svc}").fill_null(0.0)).alias(f"{svc}_MW")
            )
        select_cols = ["SETTLEMENTDATE", "NET_MW"] + [f"{svc}_MW" for svc in FCAS_SERVICES]
        dispatch_res = merged.with_columns(sum_exprs).select(select_cols)
    else:
        if "DUID" in df.columns and dispatch_duid:
            df = df.filter(pl.col("DUID") == dispatch_duid)
            if df.height == 0:
                return None
        dispatch_res = _prep(df)
        if "TOTALCLEARED" not in dispatch_res.columns:
            return None
        sum_exprs = [(pl.lit(-1.0) * pl.col("TOTALCLEARED")).alias("NET_MW")]
        for svc in FCAS_SERVICES:
            if svc in dispatch_res.columns:
                sum_exprs.append(pl.col(svc).fill_null(0.0).alias(f"{svc}_MW"))
            else:
                sum_exprs.append(pl.lit(0.0).alias(f"{svc}_MW"))
        select_cols = ["SETTLEMENTDATE", "NET_MW"] + [f"{svc}_MW" for svc in FCAS_SERVICES]
        dispatch_res = dispatch_res.with_columns(sum_exprs).select(select_cols)

    # Align to processed market data timeline
    grid = (
        processed_data.select(["SETTLEMENTDATE"])
        .with_columns(pl.col("SETTLEMENTDATE").cast(pl.Datetime("us"), strict=False))
        .sort("SETTLEMENTDATE")
    )
    dispatch_res = dispatch_res.with_columns(
        pl.col("SETTLEMENTDATE").cast(pl.Datetime("us"), strict=False)
    )
    aligned = grid.join(dispatch_res, on="SETTLEMENTDATE", how="left").fill_null(0.0)

    # Merge market columns from processed_data
    market_cols = ["RRP", "TOTALDEMAND"] + [f"FCAS_{svc}" for svc in FCAS_SERVICES]
    gen_cols = [c for c in processed_data.columns if c.startswith("GEN_")]
    market_cols += gen_cols
    market_df = processed_data.select(["SETTLEMENTDATE"] + [c for c in market_cols if c in processed_data.columns])
    aligned = aligned.join(market_df, on="SETTLEMENTDATE", how="left").fill_null(0.0)
    aligned = aligned.with_columns(pl.lit(max_power_mw).alias("max_power_mw"))

    n = aligned.height
    if n == 0:
        print(f"  [{station_name}]: aligned dataset is empty — skipping")
        return None

    print(f"  [{station_name}]: {n} steps, reconstructing SoC and computing reward...")

    # SoC reconstruction
    net_mw = aligned["NET_MW"].to_numpy()
    soc = reconstruct_soc(net_mw, step_duration, capacity_mwh, init_soc_mwh, rte)

    # Revenue + degradation
    reward_df = compute_episode_reward(
        aligned, soc, step_duration, capacity_mwh, battery_life_cost,
        degradation_model=degradation_mode,
        degradation_temperature=degradation_temperature,
    )

    # Observations (18-dim, normalized)
    observations = build_observations(aligned, soc, capacity_mwh)

    # Actions (9-dim): [energy_norm, 8 × fcas_bid_fraction]
    a0 = np.clip(net_mw / max_power_mw, -1.0, 1.0).astype(np.float32)
    fcas_cols = []
    for svc in FCAS_SERVICES:
        bid = np.clip(aligned[f"{svc}_MW"].to_numpy() / max_power_mw, 0.0, 1.0).astype(np.float32)
        fcas_cols.append(bid)
    actions = np.stack([a0] + fcas_cols, axis=1).astype(np.float32)

    # Assemble episode DataFrame
    episode_id = f"{station_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    result = pl.DataFrame({
        "episode_id": [episode_id] * n,
        "step": list(range(n)),
        "norm_observation": [row for row in observations],
        "action": [row for row in actions],
        "reward": reward_df["reward"].to_list(),
    })

    # Add info columns
    for col in reward_df.columns:
        if col != "reward":
            result = result.with_columns(reward_df[col].alias(col))
    result = result.with_columns(pl.lit(station_name).alias("station_name"))

    print(f"  [{station_name}]: done — {n} steps, "
          f"reward={float(reward_df['reward'].sum()):.2f}, "
          f"FCAS={float(reward_df['fcas_revenue'].sum()):.0f}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Convert AEMO dispatch data to episode logs (full 8-service FCAS)."
    )
    parser.add_argument("--station", action="append", required=True,
                        help="Station name(s) from BATTERY_REGISTRY (can be repeated)")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output parquet path (single station only)")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "data" / "aemo_dispatch_episodes",
                        help="Output directory (multi-station mode)")
    parser.add_argument("--cache-dir", type=Path,
                        default=root / "data" / "aemo",
                        help="AEMO data cache directory")
    parser.add_argument("--step-duration", type=float, default=0.5,
                        help="Step duration in hours (default 0.5 = 30 min)")
    parser.add_argument("--degradation-mode", type=str, default="rainflow",
                        choices=["rainflow", "real_world", "simple"],
                        help="Degradation model")
    parser.add_argument("--battery-life-cost", type=float, default=None,
                        help="Battery replacement cost in USD (default 400k)")
    args = parser.parse_args()

    sys.path.insert(0, str(root / "src"))

    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    cache_dir = str(args.cache_dir.resolve())

    for station in args.station:
        result = convert_station_to_episode(
            station_name=station,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            step_duration=args.step_duration,
            degradation_mode=args.degradation_mode,
            battery_life_cost=args.battery_life_cost,
        )
        if result is not None:
            if args.output and len(args.station) == 1:
                output_path = args.output
            else:
                output_dir = args.output_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                tag = f"{station}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                output_path = output_dir / f"{tag}.parquet"
            output_path = output_path if isinstance(output_path, Path) else Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.write_parquet(str(output_path))
            print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()