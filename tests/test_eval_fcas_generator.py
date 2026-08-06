import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import eval_fcas_generator as mod  # noqa: E402


def _write_processed(path: Path, start: datetime, rows: int) -> None:
    idx = np.arange(rows, dtype=np.float32)
    ts = [start + timedelta(minutes=5 * i) for i in range(rows)]
    df = pl.DataFrame(
        {
            "SETTLEMENTDATE": ts,
            "RRP": idx - 10.0,
            "TOTALDEMAND": 1000.0 + idx,
            "GEN_wind": 50.0 + idx,
            "GEN_solar": 20.0 + idx,
            "hour_sin": np.sin(idx),
            "hour_cos": np.cos(idx),
            "day_sin": np.sin(idx / 10.0),
            "day_cos": np.cos(idx / 10.0),
            "FCAS_RAISE6SEC": 1.0 + idx,
            "FCAS_RAISE60SEC": 2.0 + idx,
            "FCAS_RAISE5MIN": 3.0 + idx,
            "FCAS_RAISEREG": 4.0 + idx,
            "FCAS_LOWER6SEC": 5.0 + idx,
            "FCAS_LOWER60SEC": 6.0 + idx,
            "FCAS_LOWER5MIN": 7.0 + idx,
            "FCAS_LOWERREG": 8.0 + idx,
        }
    )
    df.write_parquet(path)


def test_parse_dataset_spec():
    assert mod._parse_dataset_spec("NSW1:2024-01-01:2024-07-01") == (
        "NSW1",
        "2024-01-01",
        "2024-07-01",
    )


def test_load_interval_can_slice_larger_cached_file(tmp_path, monkeypatch):
    data_dir = tmp_path / "aemo"
    data_dir.mkdir()
    path = data_dir / "processed_NSW1_2023-07-01_2024-01-01_0.0833h.parquet"
    _write_processed(path, datetime(2023, 7, 1), rows=288)
    monkeypatch.setattr(mod, "DATA", data_dir)

    span = mod.load_interval("NSW1", "2023-07-01", "2023-07-02")
    assert span.height == 288


def test_train_specs_default_to_calendar_2024():
    args = mod.parse_args([])
    specs = mod._train_specs(args)
    assert ("SA1", "2024-01-01", "2024-07-01") in specs
    assert ("SA1", "2024-07-01", "2025-01-01") in specs
