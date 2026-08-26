"""Tests for household real-data ingestion (FUTURE_PLAN §6b H0)."""

import json
import os
import sys

import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from household_ingest import (  # noqa: E402
    convert_watts_to_kilo,
    detect_column_map,
    ingest_file,
    load_csv,
    update_manifest,
    validate_and_normalize,
)


def _sha(path):
    from household_ingest import sha256_of
    return sha256_of(path)


def _write_week_csv(path, n=2016, minutes=5, solar=2.0, load=1.0, portal_names=False, start="2023-06-01T00:00"):
    """Synthetic one-week CSV at 5-min resolution (n steps)."""
    import datetime as dt

    base = dt.datetime.fromisoformat(start)
    times = [base + dt.timedelta(minutes=i * minutes) for i in range(n)]
    cols = {
        ("Date Time" if portal_names else "Timestamp"): [t.isoformat() for t in times],
        ("Solar (kW)" if portal_names else "SolarGen"): [solar] * n,
        ("Consumption (kW)" if portal_names else "HouseLoad"): [load] * n,
    }
    df = pl.DataFrame(cols)
    df.write_csv(path)
    return path


def test_detect_column_map_portal_style_names(tmp_path):
    p = tmp_path / "w.csv"
    _write_week_csv(p, portal_names=True)
    raw = pl.read_csv(p)
    m = detect_column_map(raw.columns)
    assert m["Timestamp"] == "Date Time"
    assert m["SolarGen"] == "Solar (kW)"
    assert m["HouseLoad"] == "Consumption (kW)"


def test_detect_column_map_raises_on_missing_required():
    with pytest.raises(ValueError, match="Could not auto-detect"):
        detect_column_map(["foo", "bar"])


def test_validate_and_normalize_happy_path():
    import datetime as dt

    base = dt.datetime(2023, 6, 1)
    n = 100
    df = pl.DataFrame({
        "Timestamp": [base + dt.timedelta(minutes=5 * i) for i in range(n)],
        "SolarGen": [2.0] * n,
        "HouseLoad": [1.0] * n,
    })
    out, report = validate_and_normalize(df, source_file="x.csv", source_sha256="abc")
    assert report.rows_out == n
    assert report.gap_count == 0
    assert report.duplicate_timestamps == 0
    assert not report.warnings
    # flat tariffs filled in
    assert out["ImportEnergyPrice"].unique().to_list() == [0.30]
    assert out["ExportEnergyPrice"].unique().to_list() == [0.05]
    assert set(["Timestamp", "SolarGen", "HouseLoad", "Time"]).issubset(out.columns)


def test_gap_detection_counts_and_reports():
    import datetime as dt

    base = dt.datetime(2023, 6, 1)
    times = [base + dt.timedelta(minutes=5 * i) for i in range(50)]
    times += [times[-1] + dt.timedelta(hours=3)] + [
        times[-1] + dt.timedelta(hours=3) + dt.timedelta(minutes=5 * i) for i in range(1, 20)
    ]
    n = len(times)
    df = pl.DataFrame({
        "Timestamp": times,
        "SolarGen": [1.0] * n,
        "HouseLoad": [1.0] * n,
    })
    _, report = validate_and_normalize(df, source_file="g.csv", source_sha256="x")
    assert report.gap_count >= 1
    assert report.max_gap_minutes >= 180.0
    assert any("gap" in w.lower() for w in report.warnings)


def test_duplicate_timestamps_dropped_and_reported():
    import datetime as dt

    base = dt.datetime(2023, 6, 1)
    times = [base + dt.timedelta(minutes=5 * i) for i in range(40)]
    times.insert(7, times[7])  # duplicate
    n = len(times)
    df = pl.DataFrame({"Timestamp": times, "SolarGen": [1.0] * n, "HouseLoad": [1.0] * n})
    out, report = validate_and_normalize(df, source_file="d.csv", source_sha256="x")
    assert report.duplicate_timestamps == 1
    assert out.height == 40


def test_negative_solar_reported_not_fixed():
    import datetime as dt

    base = dt.datetime(2023, 6, 1)
    n = 10
    df = pl.DataFrame({
        "Timestamp": [base + dt.timedelta(minutes=5 * i) for i in range(n)],
        "SolarGen": [-1.0] * n,
        "HouseLoad": [1.0] * n,
    })
    _, report = validate_and_normalize(df, source_file="n.csv", source_sha256="x")
    assert report.negative_solar_rows == n
    assert any("negative SolarGen" in w for w in report.warnings)


def test_ingest_file_end_to_end_with_explicit_map(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write_week_csv(raw_dir / "week_2023-06-01.csv", portal_names=True)

    colmap = {"Timestamp": "Date Time", "SolarGen": "Solar (kW)", "HouseLoad": "Consumption (kW)"}
    out_path, report = ingest_file(p, tmp_path / "normalized", column_map=colmap)

    assert out_path.exists()
    assert report.rows_out == 2016
    assert report.sha256 == _sha(p)
    got = pl.read_parquet(out_path)
    assert {"Timestamp", "SolarGen", "HouseLoad", "ImportEnergyPrice", "ExportEnergyPrice", "Time"}.issubset(got.columns)


def test_manifest_tracks_checksums_and_stats_only(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    p = _write_week_csv(raw_dir / "week.csv")
    colmap = {"Timestamp": "Timestamp", "SolarGen": "SolarGen", "HouseLoad": "HouseLoad"}
    _, report = ingest_file(p, tmp_path / "norm", column_map=colmap)
    manifest = update_manifest(tmp_path / "manifest.json", [report])
    entry = manifest["files"]["week.csv"]
    assert entry["sha256"]
    assert "rows_out" in entry
    serialized = json.dumps(manifest)
    # privacy: no metering values may leak into the manifest
    assert "2.0" not in serialized and "1.0" not in serialized


def test_load_csv_applies_column_map(tmp_path):
    p = tmp_path / "w.csv"
    _write_week_csv(p, portal_names=True)
    df = load_csv(p, column_map={"Timestamp": "Date Time", "SolarGen": "Solar (kW)", "HouseLoad": "Consumption (kW)"})
    assert {"Timestamp", "SolarGen", "HouseLoad"}.issubset(df.columns)


def test_sma_ennexos_semicolon_decimal_comma_watts(tmp_path):
    """SMA/ennexos exports: ';' separator, decimal commas, power in W."""
    import datetime as dt

    base = dt.datetime(2023, 6, 1)
    n = 20
    times = [(base + dt.timedelta(minutes=5 * i)).isoformat() for i in range(n)]
    lines = ["Time;PV-Generator (W);Consumption (W)"]
    lines += [f"{t};2500,5;1100,3" for t in times]
    p = tmp_path / "sma_week.csv"
    p.write_text("\n".join(lines), encoding="utf-8")

    colmap = {"Timestamp": "Time", "SolarGen": "PV-Generator (W)", "HouseLoad": "Consumption (W)"}
    df = load_csv(p, column_map=colmap, decimal_comma=True)
    assert df["SolarGen"].dtype.is_numeric()
    assert abs(df["SolarGen"][0] - 2500.5) < 1e-6

    converted = convert_watts_to_kilo(df, ["SolarGen", "HouseLoad"])
    assert abs(converted["SolarGen"][0] - 2.5005) < 1e-6
    assert abs(converted["HouseLoad"][0] - 1.1003) < 1e-6

    out_path, report = ingest_file(
        p, tmp_path / "norm", column_map=colmap, decimal_comma=True, watts_to_kilo=True
    )
    got = pl.read_parquet(out_path)
    assert abs(got["SolarGen"][0] - 2.5005) < 1e-6
    assert report.rows_out == n


def test_sniff_separator_semicolon(tmp_path):
    from household_ingest import _sniff_separator
    p = tmp_path / "s.csv"
    p.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    assert _sniff_separator(p) == ";"
    p2 = tmp_path / "c.csv"
    p2.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert _sniff_separator(p2) == ","


SMA_CSV = '''sep=;
"Version";"4";;;;
"System ID";"10574124";;;;
"Time period";"Direct consumption [W]";"Discharge battery [W]";"Grid-supplied power [W]";"Charge battery [W]";"Grid feed-in [W]";"Battery state of charge [%]";"Total consumption [W]";"Total generation [W]"
"12.00 AM";"500";"0";"700";"0";"0";"10";"1,200";"500"
"12.05 AM";"600";"200";"800";"300";"0";"15";"1,200";"600"
"12.10 AM";"700";"0";"0";"400";"900";"20";"700";"2,000"
"01.30 PM";"2,500";"0";"0";"0";"1,000";"80";"2,500";"3,500"
'''


def test_is_sma_energy_balance_detection(tmp_path):
    from household_ingest import is_sma_energy_balance_csv
    p = tmp_path / "sma.csv"
    p.write_text(SMA_CSV, encoding="utf-8")
    assert is_sma_energy_balance_csv(p)
    plain = tmp_path / "plain.csv"
    _write_week_csv(plain)
    assert not is_sma_energy_balance_csv(plain)


def test_sma_energy_balance_full_pipeline(tmp_path):
    from household_ingest import ingest_sma_file
    p = tmp_path / "Energy balance - Day - Some System Name - 2026-05-21.csv"
    p.write_text(SMA_CSV, encoding="utf-8")
    out_path, report = ingest_sma_file(p, tmp_path / "norm")
    df = pl.read_parquet(out_path)

    # date + owner name anonymized in the manifest identity
    assert report.source_file == "sma_2026-05-21"

    # 12-h dotted clock parsed correctly (12.00 AM -> 00:00, 12.05 -> 00:05,
    # 01.30 PM -> 13:30); timestamps carry the file date
    ts = df["Timestamp"].dt.to_string("%Y-%m-%dT%H:%M").to_list()
    assert ts[0] == "2026-05-21T00:00"
    assert ts[1] == "2026-05-21T00:05"
    assert ts[2] == "2026-05-21T00:10"
    assert ts[3] == "2026-05-21T13:30"

    # W -> kW with thousands commas stripped
    assert abs(df["HouseLoad"][0] - 1.2) < 1e-9
    assert abs(df["SolarGen"][3] - 3.5) < 1e-9

    # net battery power = charge - discharge (+charge convention)
    assert abs(df["BatteryPower"][1] - (0.3 - 0.2)) < 1e-9
    assert abs(df["BatteryPower"][0]) < 1e-12

    # SOC percent -> fraction 0..1
    assert abs(df["BatterySOC"][3] - 0.8) < 1e-9

    assert report.rows_out == 4


SMA_CSV_KW = SMA_CSV.replace("[W]", "[kW]").replace(
    '"12.00 AM";"500";"0";"700";"0";"0";"10";"1,200";"500"',
    '"12.00 AM";"0.50";"0.00";"0.70";"0.00";"0.00";"10";"1.20";"0.50"'
)


def test_sma_energy_balance_kilowatt_variant(tmp_path):
    """Newer portal exports use [kW] columns with decimal-point values."""
    from household_ingest import ingest_sma_file
    p = tmp_path / "Energy balance - Day - Name - 2026-05-21.csv"
    p.write_text(SMA_CSV_KW, encoding="utf-8")
    out_path, report = ingest_sma_file(p, tmp_path / "norm")
    df = pl.read_parquet(out_path)
    # values already in kW: no /1000 applied
    assert abs(df["HouseLoad"][0] - 1.2) < 1e-9
    assert abs(df["SolarGen"][0] - 0.5) < 1e-9
    assert report.rows_out == 4


def test_sma_energy_balance_requires_date_in_filename(tmp_path):
    from household_ingest import ingest_sma_file
    p = tmp_path / "no-date-here.csv"
    p.write_text(SMA_CSV, encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot derive date"):
        ingest_sma_file(p, tmp_path / "norm")


def _write_norm_day(tmp_path, date_iso, load_kw=2.0, solar_kw=0.0):
    import datetime as dt
    base = dt.datetime.fromisoformat(date_iso)
    n = 288
    df = pl.DataFrame({
        "Timestamp": [base + dt.timedelta(minutes=5 * i) for i in range(n)],
        "HouseLoad": [load_kw] * n,
        "SolarGen": [solar_kw] * n,
        "BatteryPower": [0.0] * n,
        "BatterySOC": [0.5] * n,
        "ImportEnergyPrice": [0.3] * n,
        "ExportEnergyPrice": [0.05] * n,
        "Time": [base + dt.timedelta(minutes=5 * i) for i in range(n)],
    })
    df.write_parquet(tmp_path / f"sma_{date_iso}_normalized.parquet")


def test_build_year_dataset_merges_and_converts(tmp_path):
    from household_ingest import build_year_dataset, env_view
    d = tmp_path / "normalized"
    d.mkdir()
    _write_norm_day(d, "2026-05-20", load_kw=2.0)
    _write_norm_day(d, "2026-05-21", load_kw=4.0)

    full = build_year_dataset(d)
    assert full.height == 576  # 2 days x 288

    # kW -> kWh conversion: 2.0 kW over 5 min = 2.0 * 5/60 kWh
    first = full.filter(pl.col("Timestamp") == pl.datetime(2026, 5, 20, 0, 0))
    assert abs(first["SolarGen"][0]) < 1e-12
    assert abs(first["HouseLoad"][0] - 2.0 * 5 / 60) < 1e-9
    # day-2 rows use their own (4.0 kW) scale
    day2 = full.filter(pl.col("Timestamp") == pl.datetime(2026, 5, 21, 0, 0))
    assert abs(day2["HouseLoad"][0] - 4.0 * 5 / 60) < 1e-9

    # day-ahead persistence: on day 2, FutureSolar == day 1's SolarGen at same slot;
    # leading 24h falls back to current value
    assert (full.filter(pl.col("Timestamp") < pl.datetime(2026, 5, 21))["FutureSolar"].abs().sum() == 0)
    day2_fut = full.filter(pl.col("Timestamp") >= pl.datetime(2026, 5, 21))
    assert day2_fut["FutureLoad"].null_count() == 0

    # env view has exactly the env columns and drops battery channels
    ev = env_view(full)
    assert set(ev.columns) == {
        "Timestamp", "Time", "SolarGen", "HouseLoad",
        "FutureSolar", "FutureLoad", "ImportEnergyPrice", "ExportEnergyPrice",
    }
    assert ev["Time"].dtype == pl.Datetime


def test_build_year_dataset_dedupes_timestamps(tmp_path):
    from household_ingest import build_year_dataset
    d = tmp_path / "normalized"
    d.mkdir()
    _write_norm_day(d, "2026-05-20")
    # duplicate file for same day -> identical timestamps; keep-first dedupes
    _write_norm_day(d, "2026-05-20", load_kw=99.0)
    full = build_year_dataset(d)
    assert full.height == 288
    assert full["HouseLoad"].max() <= 99.0 * 5 / 60 + 1e-9


def test_build_year_dataset_empty_dir_raises(tmp_path):
    from household_ingest import build_year_dataset
    with pytest.raises(FileNotFoundError):
        build_year_dataset(tmp_path)


def test_split_segments_detects_month_gap(tmp_path):
    from household_ingest import build_year_dataset, find_gap_boundaries, split_segments
    d = tmp_path / "normalized"
    d.mkdir()
    _write_norm_day(d, "2024-01-24")
    _write_norm_day(d, "2024-01-25")
    # three-month hole in the raw exports
    _write_norm_day(d, "2025-08-01")

    full = build_year_dataset(d)
    bounds = find_gap_boundaries(full)
    assert bounds == [576]  # seam right after day 2

    segs = split_segments(full)
    assert len(segs) == 2
    assert segs[0].height == 576 and segs[1].height == 288
    assert segs[0]["SegmentID"][0] == 0 and segs[1]["SegmentID"][0] == 1

    # each segment is internally contiguous at 5-min steps
    for s in segs:
        dt = s["Timestamp"].diff().dt.total_seconds().drop_nulls()
        assert dt.max() <= 15 * 60


def test_seam_row_energy_conversion_capped_at_nominal_step(tmp_path):
    """A month-scale gap must not turn the seam row into weeks of kWh."""
    from household_ingest import build_year_dataset
    d = tmp_path / "normalized"
    d.mkdir()
    _write_norm_day(d, "2024-01-24", load_kw=2.0)
    _write_norm_day(d, "2025-08-01", load_kw=3.0)

    full = build_year_dataset(d)
    seam = full.filter(pl.col("Timestamp") == pl.datetime(2025, 8, 1, 0, 0))
    # 3.0 kW x 5 min, NOT 3.0 kW x ~19 months
    assert abs(seam["HouseLoad"][0] - 3.0 * 5 / 60) < 1e-9


def _with_dead_stretch(df, start_idx, n_rows):
    """Zero out HouseLoad/SolarGen for n_rows from start_idx (offline sim)."""
    idx = list(range(start_idx, start_idx + n_rows))
    return df.with_columns([
        pl.when(pl.arange(0, df.height).is_in(idx))
        .then(0.0).otherwise(pl.col("HouseLoad")).alias("HouseLoad"),
        pl.when(pl.arange(0, df.height).is_in(idx))
        .then(0.0).otherwise(pl.col("SolarGen")).alias("SolarGen"),
    ])


def test_drop_dead_runs_removes_sustained_offline():
    from household_ingest import drop_dead_runs
    df = _write_norm_day.__wrapped__ if False else None
    # reuse helper: one day, then zero out 3h in the middle
    import datetime as dt
    base = dt.datetime.fromisoformat("2026-05-20")
    n = 288
    day = pl.DataFrame({
        "Timestamp": [base + dt.timedelta(minutes=5 * i) for i in range(n)],
        "HouseLoad": [2.0] * n,
        "SolarGen": [1.0] * n,
    })
    out, dropped = drop_dead_runs(_with_dead_stretch(day, 100, 36), min_run_minutes=120)
    assert dropped == 36
    assert out.height == n - 36


def test_drop_dead_runs_keeps_short_blips():
    from household_ingest import drop_dead_runs
    import datetime as dt
    base = dt.datetime.fromisoformat("2026-05-20")
    n = 288
    day = pl.DataFrame({
        "Timestamp": [base + dt.timedelta(minutes=5 * i) for i in range(n)],
        "HouseLoad": [2.0] * n,
        "SolarGen": [1.0] * n,
    })
    out, dropped = drop_dead_runs(_with_dead_stretch(day, 100, 12), min_run_minutes=120)
    assert dropped == 0  # 60-min blip < 120-min threshold
    assert out.height == n


def test_build_year_dataset_splits_at_offline_stretch(tmp_path):
    from household_ingest import build_year_dataset, split_segments
    d = tmp_path / "normalized"
    d.mkdir()
    _write_norm_day(d, "2026-05-20", load_kw=2.0)
    _write_norm_day(d, "2026-05-21", load_kw=3.0)

    raw = pl.concat([pl.read_parquet(f) for f in sorted(d.glob("*.parquet"))])
    dead = _with_dead_stretch(raw, 288 + 100, 60)  # 5h dead inside day 2
    dead.write_parquet(d / "sma_2026-05-21_normalized.parquet")

    full = build_year_dataset(d)
    segs = split_segments(full)
    # the offline stretch must not survive as a fake idle episode
    zero_rows = full.filter((pl.col("HouseLoad") == 0) & (pl.col("SolarGen") == 0))
    assert zero_rows.height <= 23  # only the kept short tail of the run
    assert len(segs) >= 2


def test_validate_flags_exact_zero_rows():
    import datetime as dt
    from household_ingest import validate_and_normalize
    base = dt.datetime.fromisoformat("2026-05-20")
    n = 288
    df = pl.DataFrame({
        "Timestamp": [base + dt.timedelta(minutes=5 * i) for i in range(n)],
        "HouseLoad": [2.0] * n,
        "SolarGen": [1.0] * n,
        "BatterySOC": [0.5] * n,
        "BatteryPower": [0.0] * n,
    })
    dead = _with_dead_stretch(df, 50, 48)  # 4h all-zero
    _, report = validate_and_normalize(dead, source_file="t", source_sha256="x")
    assert report.exact_zero_rows == 48
    assert any("offline" in w for w in report.warnings)
