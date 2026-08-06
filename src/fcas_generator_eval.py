"""
FCAS generator evaluation harness (Phase 6).

Compares a synthetic FCAS series against a real held-out series on the metrics
the plan specifies: per-service MAE/RMSE, tail KS test, spike-event recall,
joint spike co-occurrence, discriminative score — plus autocorrelation and
diurnal profile as cheap sanity checks.

Inputs are polars DataFrames with the same schema as the processed AEMO
parquets (SETTLEMENTDATE, RRP, TOTALDEMAND, FCAS_<8 services>, GEN_wind/solar).
Synthetic series are generated on the SAME time grid as the real exogenous
features, so interval-aligned comparisons (recall, co-occurrence) are valid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

RAISE = ["RAISE6SEC", "RAISE60SEC", "RAISE5MIN", "RAISEREG"]
LOWER = ["LOWER6SEC", "LOWER60SEC", "LOWER5MIN", "LOWERREG"]
FCAS = RAISE + LOWER
FCAS_COLS = [f"FCAS_{s}" for s in FCAS]


@dataclass
class EvalResult:
    per_service: dict
    tail_ks: dict
    spike_recall: dict
    spike_cooccurrence: dict
    acf: dict
    diurnal_mae: dict
    discriminator_auc: float | None
    spike_rate: dict


def _to_np(df: pl.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy().astype(float)


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    if x.std() == 0:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / (len(x) - lag) / x.var())


def _spike_thresholds(real: pl.DataFrame, p99: float = 0.99) -> dict[str, float]:
    return {s: float(real[f"FCAS_{s}"].quantile(p99)) for s in FCAS}


def _ks_tail(real: np.ndarray, synth: np.ndarray, thr: float) -> float:
    r = real[real >= thr]
    s = synth[synth >= thr]
    if len(r) < 5 or len(s) < 5:
        return 1.0
    return float(stats.ks_2samp(r, s).pvalue)


def _spike_bool(x: np.ndarray, thr: float) -> np.ndarray:
    return x >= thr


def compare(real: pl.DataFrame, synth: pl.DataFrame, *, seed: int = 42) -> EvalResult:
    """Compare real vs synthetic FCAS series on the shared time grid."""
    thresh = _spike_thresholds(real)
    n = real.height
    assert synth.height == n, "real and synthetic must share the time grid"

    per_service, tail_ks, spike_recall, spike_rate, acf = {}, {}, {}, {}, {}
    for s in FCAS:
        r, y = _to_np(real, f"FCAS_{s}"), _to_np(synth, f"FCAS_{s}")
        per_service[s] = {
            "mae": float(np.mean(np.abs(r - y))),
            "rmse": float(np.sqrt(np.mean((r - y) ** 2))),
        }
        tail_ks[s] = _ks_tail(r, y, thresh[s])
        rb, sb = _spike_bool(r, thresh[s]), _spike_bool(y, thresh[s])
        spike_recall[s] = float(sb[rb].mean()) if rb.any() else 1.0  # frac of real spikes also spikes in synth
        spike_rate[s] = {"real": float(rb.mean()), "synth": float(sb.mean())}
        acf[s] = {
            "lag1": {"real": _acf(r, 1), "synth": _acf(y, 1)},
            "lag12": {"real": _acf(r, 12), "synth": _acf(y, 12)},
            "lag288": {"real": _acf(r, min(288, n // 2)), "synth": _acf(y, min(288, n // 2))},
        }

    # Joint spike co-occurrence: within-direction and across-direction.
    rb = {s: _spike_bool(_to_np(real, f"FCAS_{s}"), thresh[s]) for s in FCAS}
    sb = {s: _spike_bool(_to_np(synth, f"FCAS_{s}"), thresh[s]) for s in FCAS}

    def cooc(bools: dict[str, np.ndarray], family: list[str], given: list[str]) -> float:
        any_given = np.zeros(n, dtype=bool)
        for g in given:
            any_given |= bools[g]
        if not any_given.any():
            return 1.0
        frac = []
        for f in family:
            frac.append(float(bools[f][any_given].mean()))
        return float(np.mean(frac))

    spike_cooccurrence = {
        "within_raise": {
            "real": cooc(rb, RAISE[1:], RAISE[:1]),
            "synth": cooc(sb, RAISE[1:], RAISE[:1]),
        },
        "within_lower": {
            "real": cooc(rb, LOWER[1:], LOWER[:1]),
            "synth": cooc(sb, LOWER[1:], LOWER[:1]),
        },
        "cross": {  # RAISE services spike when LOWER spikes (and vice versa) — should stay low
            "real_raise_given_lower": cooc(rb, RAISE, LOWER[:1]),
            "synth_raise_given_lower": cooc(sb, RAISE, LOWER[:1]),
            "real_lower_given_raise": cooc(rb, LOWER, RAISE[:1]),
            "synth_lower_given_raise": cooc(sb, LOWER, RAISE[:1]),
        },
    }

    # Diurnal profile: mean price by hour-of-day.
    hour = real["SETTLEMENTDATE"].dt.hour()
    diurnal_mae = {}
    for s in FCAS:
        r_mean = _to_np(real, f"FCAS_{s}").reshape(-1).copy()
        y_mean = _to_np(synth, f"FCAS_{s}").reshape(-1).copy()
        r_h = np.array([r_mean[hour == h].mean() for h in range(24)])
        y_h = np.array([y_mean[hour == h].mean() for h in range(24)])
        diurnal_mae[s] = float(np.mean(np.abs(r_h - y_h)))

    # Discriminative score: logistic regression on log1p price features (+1 lag).
    auc = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        feats = []
        for s in FCAS:
            v = _to_np(real, f"FCAS_{s}")
            feats.append(np.log1p(v))
            feats.append(np.log1p(np.roll(v, 1)))
        X = np.column_stack(feats)
        y = np.zeros(2 * n)
        y[n:] = 1
        X2 = np.vstack([X, X])
        for i, s in enumerate(FCAS):
            v = _to_np(synth, f"FCAS_{s}")
            X2[n:, 2 * i] = np.log1p(v)
            X2[n:, 2 * i + 1] = np.log1p(np.roll(v, 1))
        auc = float(np.mean(cross_val_score(
            LogisticRegression(max_iter=500), X2, y, cv=3, scoring="roc_auc")))
    except Exception as exc:  # sklearn unavailable or degenerate data
        print(f"  [harness] discriminator skipped: {exc}")

    return EvalResult(
        per_service=per_service, tail_ks=tail_ks, spike_recall=spike_recall,
        spike_cooccurrence=spike_cooccurrence, acf=acf,
        diurnal_mae=diurnal_mae, discriminator_auc=auc, spike_rate=spike_rate,
    )


def summarize(res: EvalResult) -> dict:
    """Collapse EvalResult into a compact report dict for the diary/log."""
    return {
        "mae_mean": float(np.mean([v["mae"] for v in res.per_service.values()])),
        "rmse_mean": float(np.mean([v["rmse"] for v in res.per_service.values()])),
        "tail_ks_min_pvalue": float(np.min(list(res.tail_ks.values()))),
        "tail_ks_fail_count": int(sum(p < 0.05 for p in res.tail_ks.values())),
        "spike_recall_min": float(np.min(list(res.spike_recall.values()))),
        "spike_recall_mean": float(np.mean(list(res.spike_recall.values()))),
        "within_raise": res.spike_cooccurrence["within_raise"],
        "within_lower": res.spike_cooccurrence["within_lower"],
        "cross": res.spike_cooccurrence["cross"],
        "acf_lag1_mean_abs_err": float(np.mean([
            abs(v["lag1"]["real"] - v["lag1"]["synth"]) for v in res.acf.values()])),
        "diurnal_mae_mean": float(np.mean(list(res.diurnal_mae.values()))),
        "discriminator_auc": res.discriminator_auc,
        "spike_rate_error": float(np.mean([
            abs(v["real"] - v["synth"]) for v in res.spike_rate.values()])),
    }
