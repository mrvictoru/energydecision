"""
Household real-data ingestion (FUTURE_PLAN §6b, H0).

Normalizes raw portal CSV exports (one week per file, high resolution) into
the schema expected by `transform_polars_df` / `SolarBatteryEnv`:

    Timestamp, SolarGen, HouseLoad, ImportEnergyPrice, ExportEnergyPrice,
    [FutureSolar, FutureLoad], [BatterySOC, BatteryPower]  (+ provenance cols)

Portal column names vary; a JSON ``--column-map`` maps them onto canonical
names. Reasonable auto-detection heuristics are applied when no map is given.

Privacy: raw and normalized telemetry stay local-only under data/household/real/
(gitignored). Only the manifest — checksums + validation stats, no metering
values — is designed to be shareable.

CLI entrypoint: scripts/ingest_household_portal_csv.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
import datetime as dt_
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

CANONICAL_COLUMNS = [
    "Timestamp", "SolarGen", "HouseLoad",
    "ImportEnergyPrice", "ExportEnergyPrice",
]
OPTIONAL_COLUMNS = ["FutureSolar", "FutureLoad", "BatterySOC", "BatteryPower"]

# Heuristics used when no explicit column map is provided. First regex match wins.
DEFAULT_COLUMN_HINTS = {
    "Timestamp": [r"(?i)^(timestamp|datetime|date[_ ]?time|time)$"],
    "SolarGen": [r"(?i)(solar|pv|generation).*(kw|kwh)|^(solar|pv|generation)$"],
    "HouseLoad": [r"(?i)(consumption|load|demand|usage).*(kw|kwh)|^(consumption|load|demand)$"],
    "BatterySOC": [r"(?i)(soc|state.of.charge|battery.charge)"],
    "BatteryPower": [r"(?i)(battery)(?!.*(soc|charge)).*(kw|kwh)|^(battery)(?!.*(soc|state))$"],
}


@dataclass
class IngestReport:
    """Validation statistics for one ingested file. Shareable (no metering values)."""

    source_file: str
    sha256: str
    rows_in: int
    rows_out: int
    resolution_minutes: Optional[int]
    duplicate_timestamps: int = 0
    gap_count: int = 0
    max_gap_minutes: Optional[float] = None
    null_values: int = 0
    negative_solar_rows: int = 0
    dst_anomaly_suspected: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_column_map(columns: List[str]) -> Dict[str, str]:
    """Best-effort mapping of portal columns onto canonical names via regex hints."""
    import re

    mapping: Dict[str, str] = {}
    for canonical, patterns in DEFAULT_COLUMN_HINTS.items():
        for col in columns:
            if col in mapping.values():
                continue
            if any(re.search(p, col) for p in patterns):
                mapping[canonical] = col
                break
    # ImportEnergyPrice/ExportEnergyPrice are optional (flat-tariff defaults are filled);
    # only the physical channels are truly required from the portal.
    required = ["Timestamp", "SolarGen", "HouseLoad"]
    missing = [c for c in required if c not in mapping]
    if missing:
        raise ValueError(
            f"Could not auto-detect required columns {missing}; "
            f"provide --column-map explicitly. Source columns: {columns}"
        )
    return mapping


def _sniff_separator(path: Path) -> str:
    """Detect ',' vs ';' delimiters (SMA/ennexos exports use ';')."""
    import csv as _csv
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = "".join(f.readline() for _ in range(5))
    try:
        return _csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except _csv.Error:
        return ","


def load_csv(
    path: Path,
    column_map: Optional[Dict[str, str]] = None,
    decimal_comma: bool = False,
) -> pl.DataFrame:
    """Load one portal CSV and rename mapped columns to canonical names.

    Handles SMA/ennexos conventions when present: ';' separators (sniffed),
    BOM, and decimal commas (``decimal_comma=True`` converts '2,5' -> 2.5).
    """
    sep = _sniff_separator(path)
    df = pl.read_csv(path, separator=sep, infer_schema_length=10_000)
    # decimal-comma locales: numerics arrive as strings like '2,5'
    if decimal_comma:
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                stripped = df[col].str.strip_chars()
                if stripped.str.contains(",").any() and stripped.str.replace_all(",", ".", literal=True).str.contains(r"^-?\d+(\.\d+)?$").any():
                    df = df.with_columns(
                        stripped.str.replace_all(",", ".", literal=True)
                        .str.replace_all("\u00a0", "", literal=False)
                        .cast(pl.Float64, strict=False)
                        .alias(col)
                    )
    if column_map is None:
        column_map = detect_column_map(df.columns)
    # invert: portal-name -> canonical-name rename
    rename = {portal: canon for canon, portal in column_map.items() if portal in df.columns}
    df = df.rename(rename)
    return df


def convert_watts_to_kilo(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """SMA portals report power in W; convert selected channels to kW."""
    renames = {}
    for c in columns:
        if c in df.columns and df[c].dtype.is_numeric():
            renames[c] = (pl.col(c) / 1000.0).alias(c)
    return df.with_columns(list(renames.values())) if renames else df


# ---------------------------------------------------------------------------
# SMA ennexos "Energy balance - Day" exports
# ---------------------------------------------------------------------------
# Format (verified against real exports):
#   line 1:  sep=;
#   lines 2..k: metadata ("Version", "System ID", ...) — skipped
#   header row begins with "Time period"
#   data rows: "12.05 AM";"0";"1,200";...  (12-h dotted clock, no date;
#   date must come from the filename), powers in W with thousands commas,
#   battery SOC in percent.


def _parse_sma_time(t: str) -> dt_.time:
    """'12.05 AM' -> 00:05, '12.05 PM' -> 12:05, '01.30 PM' -> 13:30."""
    normalized = t.strip().replace(".", ":")
    return dt_.datetime.strptime(normalized, "%I:%M %p").time()


def is_sma_energy_balance_csv(path: Path) -> bool:
    """Detect the SMA 'Energy balance - Day' export format ([W] or [kW] variants)."""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            head = f.read(4096)
    except OSError:
        return False
    return '"Time period"' in head and ("[W]" in head or "[kW]" in head)


def _sma_header_index(header: List[str], base: str) -> Optional[int]:
    """Find e.g. 'Total consumption' as 'Total consumption [W]' or 'Total consumption [kW]'."""
    for i, h in enumerate(header):
        if h.strip() == base or re.match(rf"^{re.escape(base)}\s*\[(k?)W\]$", h.strip()):
            return i
    return None


def load_sma_energy_balance_csv(path: Path, date_iso: str,
                                decimal_comma: bool = True) -> pl.DataFrame:
    """
    Parse one SMA energy-balance daily export into canonical channels:

      HouseLoad   <- "Total consumption [W]"       (kW)
      SolarGen    <- "Total generation [W]"        (kW)
      BatteryPower<- "Charge battery [W]" - "Discharge battery [W]"  (net, kW, +charge)
      BatterySOC  <- "Battery state of charge [%]" (fraction 0..1)
    """
    import csv as _csv

    rows_started = False
    header: List[str] = []
    records: List[List[str]] = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        # first line may be a 'sep=;' pragma
        lines = f.read().splitlines()
    for line in lines:
        if line.startswith("sep="):
            continue
        if not rows_started:
            if line.startswith('"Time period"'):
                header = next(_csv.reader([line], delimiter=";", quotechar='"'))
                rows_started = True
            continue  # metadata preamble
        records.append(next(_csv.reader([line], delimiter=";", quotechar='"')))
    if not rows_started:
        raise ValueError(f"No 'Time period' header found in {path.name}")

    def col(base: str) -> List[str]:
        idx = _sma_header_index(header, base)
        if idx is None:
            raise ValueError(f"Missing SMA column '{base}' in {path.name}")
        return [r[idx] if len(r) > idx else "" for r in records]

    def to_float(cell: str) -> float:
        v = cell.strip().replace(",", "") if decimal_comma else cell.strip()
        try:
            return float(v)
        except ValueError:
            return float("nan")

    day = dt_.date.fromisoformat(date_iso)

    def to_timestamp(t: str) -> str:
        tod = _parse_sma_time(t)
        # SMA exports midnight as "12.00 AM"; a row showing 00:xx on the same
        # calendar day is unambiguous because each file covers exactly one day.
        return f"{day.isoformat()}T{tod.strftime('%H:%M')}"

    timestamps = [to_timestamp(t) for t in col("Time period")]

    def channel_scaled(base: str) -> List[float]:
        """Channel values scaled to kW regardless of [W]/[kW] header unit."""
        idx = _sma_header_index(header, base)
        scale = 1.0 / 1000.0 if (idx is not None and "[W]" in header[idx]) else 1.0
        return [v * scale if v == v else None for v in (to_float(c) for c in col(base))]

    total_cons = channel_scaled("Total consumption")
    total_gen = channel_scaled("Total generation")
    charge = channel_scaled("Charge battery")
    discharge = channel_scaled("Discharge battery")

    df = pl.DataFrame({
        "Timestamp": timestamps,
        "HouseLoad": total_cons,
        "SolarGen": total_gen,
        "BatteryPower": [
            ((c - dch) if (c is not None and dch is not None) else None)
            for c, dch in zip(charge, discharge)
        ],
        "BatterySOC": [
            (to_float(c) * 0.01 if to_float(c) == to_float(c) else None)
            for c in col("Battery state of charge [%]")
        ],
    })
    return df


def ingest_sma_file(
    path: Path,
    output_dir: Path,
    date_iso: Optional[str] = None,
    tariff_import: float = 0.30,
    tariff_export: float = 0.05,
) -> tuple[Path, IngestReport]:
    """
    Ingest one SMA 'Energy balance - Day' CSV. The date is taken from the
    trailing YYYY-MM-DD in the filename (which also keeps owner/system names
    OUT of the shareable manifest — source_file is recorded as sma_<date>).
    """
    if date_iso is None:
        matches = re.findall(r"(\d{4}-\d{2}-\d{2})", path.stem)
        if not matches:
            raise ValueError(
                f"Cannot derive date from filename '{path.name}'; "
                "expected a trailing YYYY-MM-DD"
            )
        date_iso = matches[-1]

    digest = sha256_of(path)
    df = load_sma_energy_balance_csv(path, date_iso=date_iso)
    norm, report = validate_and_normalize(
        df,
        source_file=f"sma_{date_iso}",  # anonymized: never record owner/system names
        source_sha256=digest,
        expected_resolution_minutes=5,
        tariff_import=tariff_import,
        tariff_export=tariff_export,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"sma_{date_iso}_normalized.parquet"
    norm.write_parquet(out_path)
    return out_path, report


def validate_and_normalize(
    df: pl.DataFrame,
    source_file: str,
    source_sha256: str,
    expected_resolution_minutes: int = 5,
    tariff_import: float = 0.30,
    tariff_export: float = 0.05,
) -> tuple[pl.DataFrame, IngestReport]:
    """
    Validate one week of telemetry and normalize it toward the env schema.

    Returns (normalized_df, report). Never silently drops problems: gaps,
    duplicates and anomalies are counted in the report and surfaced as warnings.
    """
    warnings: List[str] = []

    if "Timestamp" not in df.columns:
        raise ValueError("Column 'Timestamp' is required after mapping")

    ts_col = "Timestamp"
    if df[ts_col].dtype == pl.Utf8:
        df = df.with_columns(pl.col(ts_col).str.strptime(pl.Datetime, strict=False))
    df = df.with_columns(pl.col(ts_col).alias("Time"))
    n_in = df.height

    # --- duplicates ---
    dup = df.height - df.unique(subset=[ts_col]).height
    if dup > 0:
        warnings.append(f"{dup} duplicate timestamps dropped")
        df = df.unique(subset=[ts_col], keep="first").sort(ts_col)

    # --- null handling in value columns ---
    value_cols = [c for c in ["SolarGen", "HouseLoad", "BatterySOC", "BatteryPower"] if c in df.columns]
    nulls = df.select([pl.col(c).is_null().sum() for c in value_cols]).sum_horizontal()[0] if value_cols else 0

    # --- gap detection vs expected resolution ---
    step_diffs = df.select(pl.col(ts_col).diff().drop_nulls().dt.total_minutes())
    gaps = step_diffs.filter(pl.col(ts_col) > expected_resolution_minutes)
    gap_count = gaps.height
    max_gap = float(gaps[ts_col].max()) if gap_count else float(expected_resolution_minutes)
    if gap_count > 0:
        warnings.append(f"{gap_count} gaps > {expected_resolution_minutes} min (largest {max_gap:.0f} min)")

    # --- DST anomaly heuristic: local-time offsets of e.g. 60 min around the hour grid ---
    odd_steps = step_diffs.filter(
        (pl.col(ts_col) != expected_resolution_minutes)
        & ((pl.col(ts_col) + expected_resolution_minutes) % 60 == 0)
    )
    dst_suspect = odd_steps.height > 0
    if dst_suspect:
        warnings.append("Possible DST transition detected — inspect timestamps around the anomaly")

    # --- sanity checks ---
    neg_solar = 0
    if "SolarGen" in df.columns:
        neg_solar = int(df.filter(pl.col("SolarGen") < 0).height)
        if neg_solar:
            warnings.append(f"{neg_solar} rows with negative SolarGen (check sign convention)")

    # --- fill defaults: flat tariffs when portal has none ---
    if "ImportEnergyPrice" not in df.columns:
        df = df.with_columns(pl.lit(tariff_import).alias("ImportEnergyPrice"))
    if "ExportEnergyPrice" not in df.columns:
        df = df.with_columns(pl.lit(tariff_export).alias("ExportEnergyPrice"))

    # interpolate small nulls in value channels (<=2 consecutive steps handled here;
    # larger holes are reported, never fabricated silently)
    for c in value_cols:
        df = df.with_columns(pl.col(c).interpolate())

    keep = [c for c in ["Time"] + CANONICAL_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]
    out = df.select(keep).sort(ts_col)

    report = IngestReport(
        source_file=source_file,
        sha256=source_sha256,
        rows_in=n_in,
        rows_out=out.height,
        resolution_minutes=expected_resolution_minutes,
        duplicate_timestamps=int(dup),
        gap_count=int(gap_count),
        max_gap_minutes=max_gap,
        null_values=int(nulls or 0),
        negative_solar_rows=neg_solar,
        dst_anomaly_suspected=bool(dst_suspect),
        warnings=warnings,
    )
    return out, report


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_file(
    path: Path,
    output_dir: Path,
    column_map: Optional[Dict[str, str]] = None,
    expected_resolution_minutes: int = 5,
    tariff_import: float = 0.30,
    tariff_export: float = 0.05,
    decimal_comma: bool = False,
    watts_to_kilo: bool = False,
) -> tuple[Path, IngestReport]:
    """Ingest one raw portal CSV -> normalized parquet + report."""
    digest = sha256_of(path)
    raw = load_csv(path, column_map=column_map, decimal_comma=decimal_comma)
    if watts_to_kilo:
        raw = convert_watts_to_kilo(raw, ["SolarGen", "HouseLoad", "BatteryPower"])
    norm, report = validate_and_normalize(
        raw,
        source_file=path.name,
        source_sha256=digest,
        expected_resolution_minutes=expected_resolution_minutes,
        tariff_import=tariff_import,
        tariff_export=tariff_export,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (path.stem + "_normalized.parquet")
    norm.write_parquet(out_path)
    return out_path, report


def update_manifest(manifest_path: Path, reports: List[IngestReport]) -> Dict[str, Any]:
    """Merge ingest reports into the shareable manifest (checksums + stats only)."""
    manifest: Dict[str, Any] = {"schema": CANONICAL_COLUMNS + OPTIONAL_COLUMNS, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("files", {})
    for r in reports:
        manifest["files"][r.source_file] = r.to_dict()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    from scripts.ingest_household_portal_csv import main  # type: ignore

    main(sys.argv[1:])
