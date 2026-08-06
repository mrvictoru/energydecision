import os
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic_fcas import (  # noqa: E402
    FCASDiffusionGenerator,
    FCAS_SERVICE_CAPS,
    TORCH_IMPORT_ERROR,
    _lagged_rrp_spike_indicator,
)


def _processed_frame(n_rows: int = 72) -> pl.DataFrame:
    base = datetime(2024, 1, 1, 0, 0)
    ts = [base + timedelta(minutes=5 * i) for i in range(n_rows)]
    idx = np.arange(n_rows, dtype=np.float32)
    rrp = -20.0 + 30.0 * np.sin(idx / 4.0)
    rrp[18] = 300.0
    rrp[19] = 180.0

    data = {
        "SETTLEMENTDATE": ts,
        "RRP": rrp,
        "TOTALDEMAND": 8_000.0 + 600.0 * np.sin(idx / 6.0),
        "GEN_wind": 1_500.0 + 120.0 * np.cos(idx / 7.0),
        "GEN_solar": 900.0 + 350.0 * np.maximum(np.sin((idx - 12.0) / 8.0), 0.0),
        "hour_sin": np.sin(2.0 * np.pi * idx / 288.0),
        "hour_cos": np.cos(2.0 * np.pi * idx / 288.0),
        "day_sin": np.sin(2.0 * np.pi * idx / (288.0 * 7.0)),
        "day_cos": np.cos(2.0 * np.pi * idx / (288.0 * 7.0)),
        "FCAS_RAISE6SEC": 20.0 + 8.0 * np.maximum(np.sin(idx / 5.0), 0.0),
        "FCAS_RAISE60SEC": 18.0 + 6.0 * np.maximum(np.sin(idx / 5.5), 0.0),
        "FCAS_RAISE5MIN": 15.0 + 7.0 * np.maximum(np.sin(idx / 6.0), 0.0),
        "FCAS_RAISEREG": 12.0 + 3.0 * np.maximum(np.sin(idx / 7.0), 0.0),
        "FCAS_LOWER6SEC": 21.0 + 9.0 * np.maximum(np.cos(idx / 5.0), 0.0),
        "FCAS_LOWER60SEC": 19.0 + 5.0 * np.maximum(np.cos(idx / 5.5), 0.0),
        "FCAS_LOWER5MIN": 16.0 + 6.0 * np.maximum(np.cos(idx / 6.0), 0.0),
        "FCAS_LOWERREG": 11.0 + 4.0 * np.maximum(np.cos(idx / 7.0), 0.0),
    }
    return pl.DataFrame(data)


def test_lagged_rrp_spike_indicator_excludes_current_bar():
    rrp = np.array([0.0, 0.0, 150.0, 0.0, 0.0, 0.0], dtype=np.float32)
    lagged = _lagged_rrp_spike_indicator(rrp, 100.0, lookback=2)
    assert lagged.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]


@pytest.mark.skipif(TORCH_IMPORT_ERROR is not None, reason="PyTorch runtime is unavailable in this environment")
def test_diffusion_generator_samples_capped_series_on_cpu():
    df = _processed_frame()
    gen = FCASDiffusionGenerator(
        window_size=24,
        stride=12,
        overlap=6,
        diffusion_steps=8,
        sample_steps=4,
        base_channels=16,
        channel_mults=(1, 2),
        epochs=1,
        batch_size=4,
        lr=1e-3,
        seed=7,
        device="cpu",
    )

    synth = gen.fit(df).sample(df)

    assert synth.height == df.height
    assert "RRP" in synth.columns
    for col, cap in FCAS_SERVICE_CAPS.items():
        values = synth[col].to_numpy()
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        assert (values <= cap + 1e-6).all()
