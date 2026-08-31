import datetime as dt

import numpy as np
import polars as pl
import pytest

from household_forecast import (
    apply_forecast_sidecar,
    forecast_quality,
    generate_causal_forecasts,
)


def _frame(rows: int = 12) -> pl.DataFrame:
    timestamps = [
        dt.datetime(2026, 1, 1) + dt.timedelta(minutes=5 * index)
        for index in range(rows)
    ]
    return pl.DataFrame({
        "Timestamp": timestamps,
        "SolarGen": np.arange(rows, dtype=float),
        "HouseLoad": np.arange(rows, dtype=float) + 100.0,
        "FutureSolar": np.full(rows, -1.0),
        "FutureLoad": np.full(rows, -1.0),
    })


def _predict(contexts: np.ndarray) -> np.ndarray:
    last = contexts[:, -1:, :]
    return np.repeat(last, 3, axis=1) + np.arange(1, 4)[None, :, None]


def test_causal_forecasts_align_issue_and_target_times():
    forecast = generate_causal_forecasts(
        _frame(), _predict, context_length=4, prediction_length=3, lead_steps=2
    )

    assert forecast["ForecastValid"].to_list()[:4] == [False, False, False, True]
    assert forecast["FutureSolar"][3] == pytest.approx(5.0)
    assert forecast["FutureLoad"][3] == pytest.approx(105.0)
    assert forecast["ForecastIssuedAt"][3] == dt.datetime(2026, 1, 1, 0, 15)
    assert forecast["ForecastTargetTime"][3] == dt.datetime(2026, 1, 1, 0, 25)


def test_forecasts_before_cutoff_do_not_depend_on_future_actuals():
    original = _frame()
    changed = original.with_columns([
        pl.when(pl.int_range(pl.len()) > 7)
        .then(pl.lit(9999.0))
        .otherwise(pl.col(name))
        .alias(name)
        for name in ("SolarGen", "HouseLoad")
    ])
    kwargs = dict(context_length=4, prediction_length=3, lead_steps=1)

    first = generate_causal_forecasts(original, _predict, **kwargs)
    second = generate_causal_forecasts(changed, _predict, **kwargs)

    assert first.slice(0, 8).equals(second.slice(0, 8))


def test_apply_sidecar_replaces_forecasts_and_requires_full_coverage():
    frame = _frame()
    sidecar = generate_causal_forecasts(
        frame, _predict, context_length=4, prediction_length=3, lead_steps=1
    )
    result = apply_forecast_sidecar(frame, sidecar)

    assert result["FutureSolar"][3] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="does not cover"):
        apply_forecast_sidecar(frame, sidecar.slice(1))


def test_forecast_quality_uses_future_target_without_leakage():
    frame = _frame()
    sidecar = generate_causal_forecasts(
        frame, _predict, context_length=4, prediction_length=3, lead_steps=2
    )

    quality = forecast_quality(frame, sidecar, lead_steps=2)

    assert quality["SolarGen"]["n"] == 7
    assert quality["SolarGen"]["ttm_mae"] == pytest.approx(0.0)
    assert quality["SolarGen"]["current_persistence_mae"] == pytest.approx(2.0)


def test_singleton_segment_is_preserved_as_invalid_forecast():
    forecast = generate_causal_forecasts(
        _frame(1), _predict, context_length=4, prediction_length=3, lead_steps=1
    )

    assert len(forecast) == 1
    assert not forecast["ForecastValid"][0]
    assert forecast_quality(_frame(1), forecast, lead_steps=1)["SolarGen"]["ttm_mae"] is None
