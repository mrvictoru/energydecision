"""Causal offline forecast helpers for household environment data."""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import numpy as np
import polars as pl

TARGET_COLUMNS = ("SolarGen", "HouseLoad")
FORECAST_COLUMNS = ("FutureSolar", "FutureLoad")


def generate_causal_forecasts(
    frame: pl.DataFrame,
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    context_length: int,
    prediction_length: int,
    lead_steps: int,
    batch_size: int = 256,
) -> pl.DataFrame:
    """Forecast from history through row ``t`` and align predictions to row ``t``.

    The forecast stored on row ``t`` targets ``t + lead_steps``. Rows without a
    full context are marked invalid and receive zero-valued forecasts.
    """
    if context_length < 1 or prediction_length < 1 or batch_size < 1:
        raise ValueError("context_length, prediction_length, and batch_size must be positive")
    if not 1 <= lead_steps <= prediction_length:
        raise ValueError("lead_steps must be within the model prediction length")
    missing = [name for name in ("Timestamp", *TARGET_COLUMNS) if name not in frame.columns]
    if missing:
        raise ValueError(f"Forecast frame missing columns: {missing}")
    timestamps = frame["Timestamp"].to_list()
    if len(frame) > 1:
        deltas = np.diff(frame["Timestamp"].to_numpy()).astype("timedelta64[ns]").astype(np.int64)
        positive = deltas[deltas > 0]
        if len(positive) == 0:
            raise ValueError("Forecast timestamps must increase")
        step = dt.timedelta(microseconds=int(np.median(positive)) / 1_000)
    else:
        step = dt.timedelta(minutes=5)
    values = frame.select(TARGET_COLUMNS).to_numpy().astype(np.float32)
    forecasts = np.zeros((len(frame), len(TARGET_COLUMNS)), dtype=np.float32)
    valid = np.zeros(len(frame), dtype=bool)

    issue_indices = np.arange(context_length - 1, len(frame), dtype=np.int64)
    for start in range(0, len(issue_indices), batch_size):
        batch_indices = issue_indices[start:start + batch_size]
        contexts = np.stack([
            values[index - context_length + 1:index + 1]
            for index in batch_indices
        ])
        outputs = np.asarray(predict(contexts), dtype=np.float32)
        expected = (len(batch_indices), prediction_length, len(TARGET_COLUMNS))
        if outputs.shape != expected:
            raise ValueError(f"Predictor returned {outputs.shape}; expected {expected}")
        forecasts[batch_indices] = np.maximum(outputs[:, lead_steps - 1, :], 0.0)
        valid[batch_indices] = True

    target_times = [
        timestamp + step * lead_steps
        for timestamp in timestamps
    ]
    return pl.DataFrame({
        "Timestamp": timestamps,
        "ForecastIssuedAt": timestamps,
        "ForecastTargetTime": target_times,
        "ForecastValid": valid,
        FORECAST_COLUMNS[0]: forecasts[:, 0],
        FORECAST_COLUMNS[1]: forecasts[:, 1],
    })


def forecast_quality(
    frame: pl.DataFrame,
    sidecar: pl.DataFrame,
    *,
    lead_steps: int,
) -> dict[str, dict[str, float | int | None]]:
    """Compare valid forecasts with their future targets and current persistence."""
    valid_indices = np.flatnonzero(sidecar["ForecastValid"].to_numpy())
    valid_indices = valid_indices[valid_indices + lead_steps < len(frame)]
    result: dict[str, dict[str, float | int]] = {}
    for actual_name, forecast_name in zip(TARGET_COLUMNS, FORECAST_COLUMNS):
        actual = frame[actual_name].to_numpy()[valid_indices + lead_steps]
        predicted = sidecar[forecast_name].to_numpy()[valid_indices]
        persistence = frame[actual_name].to_numpy()[valid_indices]
        result[actual_name] = {
            "n": int(len(valid_indices)),
            "ttm_mae": float(np.mean(np.abs(predicted - actual))) if len(actual) else None,
            "ttm_rmse": (
                float(np.sqrt(np.mean(np.square(predicted - actual))))
                if len(actual) else None
            ),
            "current_persistence_mae": (
                float(np.mean(np.abs(persistence - actual))) if len(actual) else None
            ),
        }
    return result


def apply_forecast_sidecar(frame: pl.DataFrame, sidecar: pl.DataFrame) -> pl.DataFrame:
    """Replace environment forecast columns using a timestamp-keyed sidecar.

    Replaced columns keep their original positions; the observation layout of
    the environment (``DataFrame.columns`` order) must not depend on which
    forecast provider produced the values.
    """
    required = {"Timestamp", *FORECAST_COLUMNS}
    missing = sorted(required - set(sidecar.columns))
    if missing:
        raise ValueError(f"Forecast sidecar missing columns: {missing}")
    original_order = frame.columns
    replacement = sidecar.select(["Timestamp", *FORECAST_COLUMNS])
    joined = (
        frame.drop([name for name in FORECAST_COLUMNS if name in frame.columns])
        .join(replacement, on="Timestamp", how="left", validate="1:1")
    )
    null_rows = joined.filter(
        pl.any_horizontal([pl.col(name).is_null() for name in FORECAST_COLUMNS])
    ).height
    if null_rows:
        raise ValueError(f"Forecast sidecar does not cover {null_rows} environment rows")
    return joined.select([name for name in original_order if name in joined.columns])
