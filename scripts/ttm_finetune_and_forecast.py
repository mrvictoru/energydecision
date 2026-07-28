#!/usr/bin/env python3
"""
Few-shot fine-tune IBM Granite TTM-R3 on AEMO price data, then generate
forecast-augmented training data for the ForecastDecisionTransformer.

Usage:
    python3 scripts/ttm_finetune_and_forecast.py --finetune --generate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from tsfm_public import (
    TinyTimeMixerForPrediction,
    TimeSeriesPreprocessor,
    TimeSeriesForecastingPipeline,
    TrackingCallback,
    get_datasets,
)
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits

# ── Constants ───────────────────────────────────────────────────────────

FORECAST_LEN = 48
CONTEXT_LEN = 512
TTM_MODEL = "ibm-granite/granite-timeseries-ttm-r3"
TTM_REVISION = "512-48-dec-512-r3"

# All 10 price channels for multivariate fine-tuning
TARGET_COLUMNS = [
    "RRP", "TOTALDEMAND",
    "FCAS_RAISEREG", "FCAS_LOWERREG",
    "FCAS_RAISE5MIN", "FCAS_LOWER5MIN",
]

FINETUNE_DIR = Path("models/ttm_aemo_finetuned")
FINETUNE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NPZ = Path("data/aemo_dt_forecast/ttm_forecasts.npz")


# ── Data loading ─────────────────────────────────────────────────────────

def load_aemo_data(aemo_dir: str = "data/aemo") -> pd.DataFrame:
    """Load and combine all 5-min AEMO data into a single pandas DataFrame."""
    aemo_files = sorted(Path(aemo_dir).glob("processed_*_0.0833h.parquet"))
    all_dfs = []
    for f in aemo_files:
        all_dfs.append(
            pl.read_parquet(f, columns=["SETTLEMENTDATE"] + TARGET_COLUMNS)
            .with_columns(pl.col("SETTLEMENTDATE").cast(pl.Datetime("us")))
        )
    df = pl.concat(all_dfs).unique(subset=["SETTLEMENTDATE"]).sort("SETTLEMENTDATE")
    df = df.drop_nulls(subset=TARGET_COLUMNS)
    pdf = df.to_pandas().rename(columns={"SETTLEMENTDATE": "time"})
    pdf = pdf.set_index("time").asfreq("5min").reset_index()  # ensure regular 5-min grid
    # Clamp extreme values to prevent NaN in student-t NLL loss
    for col in TARGET_COLUMNS:
        if col in pdf.columns:
            lo, hi = pdf[col].quantile(0.01), pdf[col].quantile(0.99)
            pdf[col] = pdf[col].clip(lo, hi).fillna(0.0)
    print(f"[Data] {len(pdf)} rows, {pdf['time'].min()} → {pdf['time'].max()}")
    return pdf


# ── Fine-tuning ──────────────────────────────────────────────────────────

def _build_diverse_fewshot_indices(
    pdf: pd.DataFrame,
    n_train: int,
    fewshot_fraction: float,
) -> np.ndarray:
    """Return evenly-spaced indices across the training period.

    Instead of ``fewshot_location="first"`` (which clusters all few-shot
    samples at the very beginning of the training data), this spreads them
    uniformly across the full training range so the TTM sees multiple
    seasons and market regimes.
    """
    n_samples = max(1, int(n_train * fewshot_fraction))
    indices = np.linspace(0, n_train - 1, n_samples, dtype=int)
    print(f"[FT] Diverse few-shot: {n_samples} indices spread evenly across {n_train} training rows")
    return indices


def finetune_ttm(
    pdf: pd.DataFrame,
    device: str,
    fewshot_location: str = "first",
) -> TinyTimeMixerForPrediction:
    """Few-shot fine-tune TTM on AEMO price data, using 5% of data.

    Args:
        fewshot_location: How to sample the few-shot subset.
            ``"first"`` — original behaviour, takes 5% from the beginning of
            the training split (14K rows from Jan-Feb 2021).
            ``"diverse"`` — evenly spaced indices across the full training
            period, exposing the TTM to multiple seasons and market regimes.
    """
    print(f"\n=== Fine-tuning TTM on AEMO data (fewshot_location={fewshot_location}) ===")

    # Prepare data splits (train/val/test)
    split_config = {
        "train": 0.7,
        "test": 0.2,
    }

    time_column = "time"
    tsp = TimeSeriesPreprocessor(
        timestamp_column=time_column,
        target_columns=TARGET_COLUMNS,
        context_length=CONTEXT_LEN,
        prediction_length=FORECAST_LEN,
        scaling=True,
        center=True,
    )

    # Fit preprocessor on full data
    tsp.train(pdf)

    if fewshot_location == "diverse":
        # Build custom few-shot dataset with evenly-spaced indices
        n_train = int(len(pdf) * split_config["train"])
        fewshot_indices = _build_diverse_fewshot_indices(
            pdf, n_train, fewshot_fraction=0.05,
        )
        train_dataset, valid_dataset, test_dataset = get_datasets(
            tsp,
            pdf,
            split_config,
            fewshot_fraction=0.05,
            fewshot_location="first",  # fallback, overridden below
        )
        # Replace the training dataset with our custom few-shot subset
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, fewshot_indices)
        print(f"[FT] Custom few-shot dataset: {len(train_dataset)} samples")
    else:
        # Original behaviour: few-shot from the beginning
        train_dataset, valid_dataset, test_dataset = get_datasets(
            tsp,
            pdf,
            split_config,
            fewshot_fraction=0.05,
            fewshot_location=fewshot_location,
        )
    print(f"[FT] Datasets: train={len(train_dataset)}, val={len(valid_dataset)}, test={len(test_dataset)}")

    # Load base model
    model = TinyTimeMixerForPrediction.from_pretrained(
        TTM_MODEL, revision=TTM_REVISION,
    )
    model.to(device)
    print(f"[FT] Base model: {model.config}")

    # Training args — batch_size tuned for 22 GB GPU
    batch_size = 32
    num_epochs = 100
    learning_rate = 4e-4
    patience = 10

    args = TrainingArguments(
        output_dir=str(FINETUNE_DIR / "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=4,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=str(FINETUNE_DIR / "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        use_cpu=(device == "cpu"),
    )

    early_stop = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=0.00001,
    )
    tracking = TrackingCallback()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / batch_size),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stop, tracking],
        optimizers=(optimizer, scheduler),
    )

    print("[FT] Starting fine-tuning...")
    t0 = time.time()
    trainer.train()
    print(f"[FT] Fine-tuning done in {time.time() - t0:.1f}s")

    # Save fine-tuned model
    save_path = str(FINETUNE_DIR / "model")
    trainer.save_model(save_path)
    print(f"[FT] Model saved to {save_path}")

    return model


# ── Forecast generation ──────────────────────────────────────────────────

def generate_forecasts(
    pdf: pd.DataFrame, model: TinyTimeMixerForPrediction, device: str
) -> dict[str, np.ndarray]:
    """Generate 48-step forecasts for every valid starting position.

    For each channel, slides a window over the data and produces
    forecasts at stride intervals, then fills gaps via interpolation.
    """
    print("\n=== Generating forecasts ===")
    n_total = len(pdf)
    stride = FORECAST_LEN // 2  # 24 — enough overlap
    results: dict[str, np.ndarray] = {}

    for channel in TARGET_COLUMNS:
        t0 = time.time()
        series_data = pdf[["time", channel]].copy()
        series_data[channel] = series_data[channel].fillna(0.0).astype(np.float32)

        # Preprocessor for this channel
        tsp = TimeSeriesPreprocessor(
            timestamp_column="time",
            target_columns=[channel],
            context_length=CONTEXT_LEN,
            prediction_length=FORECAST_LEN,
            scaling=True,
            center=True,
        )
        tsp.train(series_data)

        # Build pipeline
        pipeline = TimeSeriesForecastingPipeline(
            model=model,
            device=device,
            batch_size=64,
            preprocessor=tsp,
        )

        # Generate forecasts at stride intervals
        forecasts = np.zeros((n_total, FORECAST_LEN), dtype=np.float32)

        # For each position with enough context, get the forecast
        positions = list(range(CONTEXT_LEN, n_total - FORECAST_LEN + 1, stride))

        for pos in positions:
            chunk = series_data.iloc[:pos + FORECAST_LEN]
            try:
                pred_df = pipeline(chunk)
                # Extract the prediction columns
                pred_col = [c for c in pred_df.columns if channel in c]
                if pred_col:
                    vals = pred_df[pred_col[-1]].values  # last prediction column
                    if len(vals) >= FORECAST_LEN:
                        forecasts[pos] = vals[:FORECAST_LEN]
            except Exception as e:
                print(f"  [WARN] pos {pos}: {e}")

        # Fill gaps between stride windows via linear interpolation
        for col in range(FORECAST_LEN):
            col_vals = forecasts[:, col]
            valid = col_vals != 0
            if valid.any():
                idx = np.where(valid)[0]
                for i in range(len(idx) - 1):
                    start, end = idx[i], idx[i + 1]
                    if end - start > stride:
                        forecasts[start:end, col] = np.linspace(
                            forecasts[start, col], forecasts[end, col], end - start
                        )

        results[channel] = forecasts
        elapsed = time.time() - t0
        print(f"[FC] {channel}: {len(positions)} forecasts in {elapsed:.1f}s")

    return results


def assemble_forecast_states(
    forecasts: dict[str, np.ndarray], n_total: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble per-step forecast states from individual channel forecasts.

    Returns:
        forecast_states: [n_total, FORECAST_LEN, 18] — 18-dim observation
        forecast_rtgs:   [n_total, FORECAST_LEN] — discounted RTGs
    """
    # Build the 18-dim observations from available forecast channels
    # The 18-dim observation is: [time_feats(5), RRP(1), demand(1),
    #   FCAS(8), gen(2), SOC(1)]
    # We forecast: RRP, demand, 8 FCAS = 10 dims
    # Time features (hour_sin/cos, day_sin/cos, is_peak), gen(2), SOC are
    #   not forecast (they follow deterministic patterns).
    # For those, we'll use the last observed value or a persistence forecast.

    n_channels = len(TARGET_COLUMNS)  # 10
    f_states = np.zeros((n_total, FORECAST_LEN, 18), dtype=np.float32)

    for ci, ch in enumerate(TARGET_COLUMNS):
        if ch in forecasts:
            f_states[:, :, 5 + ci] = forecasts[ch]  # offset 5 for time features
    # Remaining dims (time features, gen mix, SOC) stay 0
    # These will be masked or filled by the dataset

    # RTGs: cumulative discounted return from forecast point
    f_rtgs = np.zeros((n_total, FORECAST_LEN), dtype=np.float32)
    for t in range(n_total):
        for fi in range(FORECAST_LEN):
            # Simple heuristic: RTG decays exponentially from current step
            f_rtgs[t, fi] = 0.95 ** fi

    return f_states, f_rtgs


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", help="Run fine-tuning")
    parser.add_argument("--generate", action="store_true", help="Generate forecasts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--aemo-data-dir", default="data/aemo")
    parser.add_argument(
        "--fewshot-location",
        default="first",
        choices=["first", "diverse"],
        help="Few-shot sampling strategy: 'first' (original, 14K rows from Jan-Feb 2021) "
             "or 'diverse' (evenly spaced across training period for multi-season exposure). "
             "Default: 'first' for backward compatibility.",
    )
    args = parser.parse_args()

    pdf = load_aemo_data(args.aemo_data_dir)

    model = None
    if args.finetune:
        model = finetune_ttm(pdf, args.device, fewshot_location=args.fewshot_location)
    else:
        # Load pre-finetuned model
        model_path = str(FINETUNE_DIR / "model")
        if os.path.exists(model_path):
            model = TinyTimeMixerForPrediction.from_pretrained(model_path)
            model.to(args.device)
            print(f"[Main] Loaded fine-tuned model from {model_path}")
        else:
            print("[Main] No fine-tuned model found. Use --finetune first.")
            return

    if args.generate:
        forecasts = generate_forecasts(pdf, model, args.device)
        f_states, f_rtgs = assemble_forecast_states(forecasts, len(pdf))

        OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUTPUT_NPZ,
            forecast_states=f_states,
            forecast_rtgs=f_rtgs,
            n_total=len(pdf),
            forecast_len=FORECAST_LEN,
            channels=TARGET_COLUMNS,
        )
        print(f"[Main] Forecast data saved to {OUTPUT_NPZ}")
        print(f"   forecast_states: {f_states.shape}")
        print(f"   forecast_rtgs:   {f_rtgs.shape}")

    print("[Main] Done")


if __name__ == "__main__":
    main()
