"""Temporal-fidelity audit: synthetic vs real FCAS spike dynamics (NSW1 H2 2024).

Uses the synthetic frame the downstream DT was trained on
(data/aemo_dt_synth/synth_NSW1_2024-07-01_2025-01-01.parquet) vs the real frame.

Measures the temporal properties the harness's aggregate metrics miss:
  - spike burst lengths (real spikes cluster; do synthetic ones?)
  - lag-1 spike persistence P(spike_t | spike_{t-1})
  - lagged cross-service co-movement (does a spike drag other services up?)
  - spike-onset magnitude trajectory (t .. t+3 after onset)
  - hour-of-day spike-rate profile
  - per-service ACF at several lags
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcas_generator_eval import FCAS  # noqa: E402

SYNTH = ROOT / "data/aemo_dt_synth/synth_NSW1_2024-07-01_2025-01-01.parquet"
REAL = ROOT / "data/aemo/processed_NSW1_2024-07-01_2025-01-01_0.0833h.parquet"


def threshold_and_bool(x, thr):
    return x >= thr


def burst_lengths(flags: np.ndarray) -> np.ndarray:
    lengths = []
    run = 0
    for v in flags:
        if v:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return np.array(lengths, dtype=int)


def lag_persist(flags: np.ndarray) -> float:
    pos = np.where(flags)[0]
    pos = pos[pos > 0]
    if len(pos) == 0:
        return float("nan")
    return float(flags[pos - 1].mean())


def onset_trajectory(x: np.ndarray, flags: np.ndarray, max_lag: int = 4) -> list[float]:
    onsets = np.where(flags & ~np.r_[False, flags[:-1]])[0]
    out = []
    for lag in range(max_lag):
        idx = onsets + lag
        idx = idx[idx < len(x)]
        out.append(float(x[idx].mean()) if len(idx) else float("nan"))
    return out


def acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    if x.std() == 0:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / (len(x) - lag) / x.var())


def hour_spike_rate(flags: np.ndarray, hour: np.ndarray) -> list[float]:
    return [float(flags[hour == h].mean()) for h in range(24)]


def main() -> None:
    synth = pl.read_parquet(SYNTH)
    real = pl.read_parquet(REAL)
    n = min(synth.height, real.height)
    hour = np.arange(n) % 288 // 12  # 5-min bars -> hour of day

    print(f"{'service':18s} {'burst2+ synth/real':>18s} {'mean_burst synth/real':>20s} {'lag1 persist synth/real':>20s}")
    summary: dict[str, dict] = {}
    for s in FCAS:
        col = f"FCAS_{s}"
        rv = real[col].to_numpy()[:n]
        sv = synth[col].to_numpy()[:n]
        thr = float(real[col].quantile(0.99))
        rf, sf = threshold_and_bool(rv, thr), threshold_and_bool(sv, thr)
        rb, sb = burst_lengths(rf), burst_lengths(sf)
        p2r = float(np.mean(rb >= 2)) if len(rb) else float("nan")
        p2s = float(np.mean(sb >= 2)) if len(sb) else float("nan")
        mbr = float(rb.mean()) if len(rb) else float("nan")
        mbs = float(sb.mean()) if len(sb) else float("nan")
        l1r, l1s = lag_persist(rf), lag_persist(sf)
        print(f"{s:18s} {p2s:9.3f}/{p2r:7.3f} {mbs:10.2f}/{mbr:7.2f} {l1s:9.3f}/{l1r:7.3f}")
        summary[s] = {
            "thr": thr,
            "burst_p2_synth": p2s, "burst_p2_real": p2r,
            "mean_burst_synth": mbs, "mean_burst_real": mbr,
            "lag1_persist_synth": l1s, "lag1_persist_real": l1r,
            "onset_synth": onset_trajectory(sv, sf),
            "onset_real": onset_trajectory(rv, rf),
            "acf1_synth": acf(sv, 1), "acf1_real": acf(rv, 1),
            "acf12_synth": acf(sv, 12), "acf12_real": acf(rv, 12),
            "hour_spike_synth": hour_spike_rate(sf, hour),
            "hour_spike_real": hour_spike_rate(rf, hour),
        }

    print("\nSpike-onset magnitude trajectory (mean value at t, t+1, t+2, t+3 after onset):")
    for s in FCAS:
        o = summary[s]
        print(f"  {s:18s} real  : {[f'{v:8.1f}' for v in o['onset_real']]}")
        print(f"  {'':18s} synth : {[f'{v:8.1f}' for v in o['onset_synth']]}")

    print("\nACF lag1 (log-space) real vs synth:")
    for s in FCAS:
        o = summary[s]
        print(f"  {s:18s} real={o['acf1_real']:+.3f} synth={o['acf1_synth']:+.3f}  | lag12 real={o['acf12_real']:+.3f} synth={o['acf12_synth']:+.3f}")

    print("\nHour-of-day spike rate |real - synth| max/mean:")
    for s in FCAS:
        o = summary[s]
        r = np.array(o["hour_spike_real"]); sy = np.array(o["hour_spike_synth"])
        print(f"  {s:18s} max_diff={np.abs(r - sy).max():.4f} mean_diff={np.abs(r - sy).mean():.4f}")


if __name__ == "__main__":
    main()
