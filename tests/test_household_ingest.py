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
