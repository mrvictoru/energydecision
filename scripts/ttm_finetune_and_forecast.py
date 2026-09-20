#!/usr/bin/env python3
"""
Few-shot fine-tune IBM Granite TTM-R3 on AEMO price data.

Usage:
    python3 scripts/ttm_finetune_and_forecast.py --finetune
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
        # First get the full training dataset to know its actual size
        train_dataset, valid_dataset, test_dataset = get_datasets(
            tsp, pdf, split_config,
            fewshot_fraction=1.0, fewshot_location="first",
        )
        n_actual = len(train_dataset)
        fewshot_indices = _build_diverse_fewshot_indices(
            pdf, n_actual, fewshot_fraction=0.05,
        )
        from torch.utils.data import Dataset as _Dataset
        class _IndexDataset(_Dataset):
            def __init__(self, base_dataset, indices):
                self.base = base_dataset
                self.indices = list(indices)
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.base[self.indices[idx]]
        train_dataset = _IndexDataset(train_dataset, fewshot_indices)
        print(f"[FT] Custom diverse few-shot dataset: {len(train_dataset)} samples (from {n_actual} total)")
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
        dataloader_num_workers=0 if fewshot_location == "diverse" else 4,
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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", help="Run fine-tuning")
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

    print("[Main] Done")


if __name__ == "__main__":
    main()
