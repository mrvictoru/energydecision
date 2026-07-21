#!/usr/bin/env python3
"""Generate TTM-based forecast lookup for the full AEMO dataset."""
import sys
import numpy as np
import pandas as pd
import polars as pl
import torch
import time
from pathlib import Path

from tsfm_public import TinyTimeMixerForPrediction

print = lambda *a, **kw: sys.stdout.write(" ".join(str(x) for x in a) + "\n") and sys.stdout.flush()

FORECAST_LEN = 48
CONTEXT_LEN = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # process one window at a time for reliability

TARGET_COLUMNS = [
    "RRP", "TOTALDEMAND",
    "FCAS_RAISEREG", "FCAS_LOWERREG",
    "FCAS_RAISE5MIN", "FCAS_LOWER5MIN",
]
N_CHANNELS = len(TARGET_COLUMNS)

# Load fine-tuned model
print("Loading fine-tuned TTM...")
model = TinyTimeMixerForPrediction.from_pretrained("models/ttm_aemo_finetuned/model")
_ = model.to(DEVICE)
model.eval()
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")

# Load AEMO data
print("Loading AEMO price data...")
aemo_files = sorted(Path("data/aemo").glob("processed_*_0.0833h.parquet"))
all_dfs = []
for f in aemo_files:
    all_dfs.append(
        pl.read_parquet(f, columns=["SETTLEMENTDATE"] + TARGET_COLUMNS)
        .with_columns(pl.col("SETTLEMENTDATE").cast(pl.Datetime("us")))
    )
df = pl.concat(all_dfs).unique(subset=["SETTLEMENTDATE"]).sort("SETTLEMENTDATE")
df = df.drop_nulls(subset=TARGET_COLUMNS)
pdf = df.to_pandas().rename(columns={"SETTLEMENTDATE": "time"})
pdf = pdf.set_index("time").asfreq("5min").reset_index()
for col in TARGET_COLUMNS:
    lo, hi = pdf[col].quantile(0.01), pdf[col].quantile(0.99)
    pdf[col] = pdf[col].clip(lo, hi).fillna(0.0)
N = len(pdf)
print(f"Data: {N} rows, {pdf['time'].min()} to {pdf['time'].max()}")

# Generate forecasts at stride intervals
stride = FORECAST_LEN // 2  # 24
positions = list(range(CONTEXT_LEN, N - FORECAST_LEN + 1, stride))
print(f"Generating {len(positions)} forecasts at stride={stride}...")

forecast_map = np.zeros((N, FORECAST_LEN, N_CHANNELS), dtype=np.float32)
t0 = time.time()

for i, pos in enumerate(positions):
    context = pdf[TARGET_COLUMNS].iloc[pos - CONTEXT_LEN:pos].values  # [512, 6]
    inp = torch.tensor(context, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, 512, 6]
    with torch.no_grad():
        out = model(past_values=inp)
    forecast_map[pos] = out.prediction_outputs[0].cpu().numpy()
    
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  [{elapsed:.0f}s] {i+1}/{len(positions)} ({100*(i+1)//len(positions)}%)")

# Fill gaps — for any position without a forecast, use the nearest prior forecast
print("Filling gaps...")
last_valid = np.zeros((N_CHANNELS,), dtype=np.float32)
for t in range(N):
    if (forecast_map[t] != 0).any():
        last_valid = forecast_map[t, -1].copy()  # last step of the forecast from position t
    elif t > CONTEXT_LEN:
        # Propagate last valid forecast forward
        forecast_map[t] = forecast_map[t - 1]

elapsed = time.time() - t0
covered = (forecast_map != 0).any(axis=(1, 2)).sum()
print(f"\nDone in {elapsed:.1f}s. Coverage: {covered}/{N} ({100*covered//N}%)")

out_dir = Path("data/aemo_dt_forecast")
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_dir / "ttm_forecasts.npz",
    forecast_map=forecast_map,
    timestamps=np.array([t.timestamp() for t in pdf["time"]]),
    channels=TARGET_COLUMNS,
    forecast_len=FORECAST_LEN,
)
print(f"Saved to {out_dir / 'ttm_forecasts.npz'}")
print(f"  forecast_map shape: {forecast_map.shape}")

