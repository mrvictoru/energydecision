#!/usr/bin/env python3
"""
Phase 6 v2 gate achievability diagnostic (real-data only, no model training).

Answers three questions before investing GPU hours in the generator restructure:

1. ACHIEVABILITY: The tail-KS gate demands p >= 0.05 on all 8 services x 3
   regions. Even a perfect generator (real-vs-real) may fail this: KS on a few
   hundred tail samples per service is noisy, and 24 tests inflate the
   false-failure rate. This measures the real-vs-real bound directly by
   splitting each region's same-period holdout into two independent real samples
   (even/odd bars, plus randomized 50% splits) and running the harness tail-KS
   on each service.

2. REGIME DRIFT: tail-KS between the fit window's tail and the holdout's tail
   per service. This bounds any empirical/parametric tail model trained on the
   fit window (e.g. v1's empirical-tail replay, or a GPD fitted on fit data).

3. DATA PROFILE: per-service tail sample counts, max values, and cap hits in
   fit vs holdout, so we know which services dominate the failures and whether
   they are cap-event artifacts or genuine distribution drift.

Pure numpy/scipy/polars -- no torch required, so it runs on the host or in any
Distrobox. Output: eval_output/phase6_fcas/gate_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fcas_generator_eval import FCAS, _ks_tail, _spike_thresholds  # noqa: E402
from eval_fcas_generator import load_interval  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "eval_output" / "phase6_fcas"
HOLDOUT = ["SA1", "NSW1", "VIC1"]
SAME_PERIOD_SPAN = "2024-01-01:2024-07-01"
FIT_FRACTION = 0.55
N_RANDOM_SPLITS = 5

SERVICE_CAPS = {
    "RAISE6SEC": 16_600.0,
    "RAISE60SEC": 16_600.0,
    "RAISE5MIN": 16_600.0,
    "RAISEREG": 999.0,
    "LOWER6SEC": 16_600.0,
    "LOWER60SEC": 16_600.0,
    "LOWER5MIN": 16_600.0,
    "LOWERREG": 999.0,
}


FEATURE_COLS = [
    "TOTALDEMAND",
    "GEN_wind",
    "GEN_solar",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


def _service_series(df: pl.DataFrame, service: str) -> np.ndarray:
    return df[f"FCAS_{service}"].to_numpy().astype(float)


def _lagged_spike(x: np.ndarray, threshold: float, lookback: int = 12) -> np.ndarray:
    cur = (x >= threshold).astype(np.float32)
    out = np.zeros_like(cur)
    for shift in range(1, lookback + 1):
        sh = np.zeros_like(cur)
        sh[shift:] = cur[:-shift]
        out = np.maximum(out, sh)
    return out


def _holdout_split(df: pl.DataFrame, fraction: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    split = int(df.height * fraction)
    return df.head(split), df.tail(df.height - split)


def real_vs_real_tail_ks(hold: pl.DataFrame, n_random: int) -> dict:
    """Per-service KS p between two independent real samples of the holdout."""
    n = hold.height
    thr = _spike_thresholds(hold)
    rng = np.random.default_rng(0)
    even, odd, rand = {}, {}, {}
    for s in FCAS:
        x = _service_series(hold, s)
        even[s] = float(_ks_tail(x[0::2], x[1::2], thr[s]))
        odd[s] = even[s]
        ps = []
        for _ in range(n_random):
            perm = rng.permutation(n)
            a = x[perm[: n // 2]]
            b = x[perm[n // 2:]]
            ps.append(float(_ks_tail(a, b, thr[s])))
        rand[s] = ps
    return {"even_odd": even, "random": rand}


def fit_vs_holdout_tail_ks(fit: pl.DataFrame, hold: pl.DataFrame) -> dict:
    """Per-service KS p between the fit window's tail and the holdout's tail."""
    thr = _spike_thresholds(hold)
    out = {}
    for s in FCAS:
        xf = _service_series(fit, s)
        xh = _service_series(hold, s)
        out[s] = float(_ks_tail(xf, xh, thr[s]))
    return out


def conditional_predictability(fit: pl.DataFrame, hold: pl.DataFrame) -> dict:
    """Per-service: can P(x >= hold_p99 | features) learned on fit generalize to hold?

    AUC > ~0.65 means tail events are at least partly predictable from the
    generator's conditioning features, so a well-calibrated conditional
    generator has a realistic shot at the marginal-tail-KS gate despite the
    fit-vs-holdout regime drift. AUC ~0.5 means the events are hidden shocks
    (grid disturbances, dispatch events) that the exogenous features do not
    encode -- in which case the marginal-tail gate is effectively unreachable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    thr = _spike_thresholds(hold)
    rrp_fit = fit["RRP"].to_numpy().astype(float)
    rrp_hold = hold["RRP"].to_numpy().astype(float)
    spike_thr = float(np.quantile(rrp_fit, 0.99))
    lag_fit = _lagged_spike(rrp_fit, spike_thr)
    lag_hold = _lagged_spike(rrp_hold, spike_thr)

    Xf = np.column_stack([fit[c].to_numpy().astype(float) for c in FEATURE_COLS] + [lag_fit])
    Xh = np.column_stack([hold[c].to_numpy().astype(float) for c in FEATURE_COLS] + [lag_hold])
    scaler = StandardScaler().fit(Xf)
    Xf = scaler.transform(Xf)
    Xh = scaler.transform(Xh)

    out = {}
    for s in FCAS:
        yf = (fit[f"FCAS_{s}"].to_numpy().astype(float) >= thr[s]).astype(int)
        yh = (hold[f"FCAS_{s}"].to_numpy().astype(float) >= thr[s]).astype(int)
        if yf.sum() < 20 or yf.sum() >= len(yf) * 0.5:
            out[s] = {"auc": None, "note": f"insufficient positives in fit ({yf.sum()})"}
            continue
        clf = LogisticRegression(max_iter=1000)
        try:
            clf.fit(Xf, yf)
            auc = roc_auc_score(yh, clf.predict_proba(Xh)[:, 1])
            out[s] = {
                "auc": float(auc),
                "hold_real_spike_rate": float(yh.mean()),
                "hold_pred_spike_rate": float(clf.predict_proba(Xh)[:, 1].mean()),
            }
        except Exception as exc:  # sklearn degenerate data
            out[s] = {"auc": None, "note": str(exc)}
    return out


def profile(fit: pl.DataFrame, hold: pl.DataFrame) -> dict:
    """Per-service tail sample counts, maxima, and cap hits in fit vs holdout."""
    thr = _spike_thresholds(hold)
    out = {}
    for s in FCAS:
        xf = _service_series(fit, s)
        xh = _service_series(hold, s)
        cap = SERVICE_CAPS[s]
        out[s] = {
            "hold_p99": float(thr[s]),
            "n_fit_ge_thr": int(np.sum(xf >= thr[s])),
            "n_hold_ge_thr": int(np.sum(xh >= thr[s])),
            "max_fit": float(xf.max()),
            "max_hold": float(xh.max()),
            "n_fit_at_cap": int(np.sum(xf >= cap)),
            "n_hold_at_cap": int(np.sum(xh >= cap)),
        }
    return out


def _min_p(ps: dict) -> float:
    return float(min(ps.values()))


def _pass_fraction(ps: dict) -> float:
    return float(np.mean([p >= 0.05 for p in ps.values()]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--same-period-span", default=SAME_PERIOD_SPAN)
    parser.add_argument("--fit-fraction", type=float, default=FIT_FRACTION)
    parser.add_argument("--random-splits", type=int, default=N_RANDOM_SPLITS)
    args = parser.parse_args(argv)

    start, end = args.same_period_span.split(":", 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"_meta": {"same_period_span": [start, end], "fit_fraction": args.fit_fraction, "random_splits": args.random_splits}}

    global_min_even = 1.0
    global_min_drift = 1.0
    all_pass_24 = True
    n_tests = 0

    for region in HOLDOUT:
        full = load_interval(region, start, end)
        fit, hold = _holdout_split(full, args.fit_fraction)
        achievable = real_vs_real_tail_ks(hold, args.random_splits)
        drift = fit_vs_holdout_tail_ks(fit, hold)
        cond = conditional_predictability(fit, hold)
        prof = profile(fit, hold)

        even_p = achievable["even_odd"]
        min_even = _min_p(even_p)
        global_min_even = min(global_min_even, min_even)
        min_drift = _min_p(drift)
        global_min_drift = min(global_min_drift, min_drift)
        pass_fraction_even = _pass_fraction(even_p)
        all_pass_24 = all_pass_24 and pass_fraction_even == 1.0
        n_tests += len(even_p)

        rand_ps = achievable["random"]
        rand_flat = [p for ps in rand_ps.values() for p in ps]
        rand_pass = float(np.mean([p >= 0.05 for p in rand_flat]))

        print(f"\n== {region}: fit {fit.height} bars, holdout {hold.height} bars ==")
        print(f"  real-vs-real (even/odd) tail KS: min p = {min_even:.4g}, services passing >= 0.05: {int(pass_fraction_even * len(even_p))}/{len(even_p)}")
        print(f"  real-vs-real (random splits, {args.random_splits} per service): pass fraction >= 0.05 = {rand_pass:.2f}")
        print(f"  fit-vs-holdout regime drift tail KS: min p = {min_drift:.4g}")
        cond_aucs = [v["auc"] for v in cond.values() if v.get("auc") is not None]
        mean_cond_auc = float(np.mean(cond_aucs)) if cond_aucs else float("nan")
        print(f"  conditional predictability: mean AUC = {mean_cond_auc:.2f} "
              f"({len(cond_aucs)}/{len(FCAS)} services fittable)")
        for s in FCAS:
            v = cond[s]
            if v.get("auc") is not None:
                print(f"    {s:20s} AUC={v['auc']:.2f}  hold real spike rate={v['hold_real_spike_rate']:.4f}  pred={v['hold_pred_spike_rate']:.4f}")
            else:
                print(f"    {s:20s} {v.get('note', 'skipped')}")

        report[region] = {
            "achievable_even_odd": even_p,
            "achievable_random": rand_ps,
            "achievable_random_pass_fraction": rand_pass,
            "regime_drift_fit_vs_holdout": drift,
            "conditional_predictability": cond,
            "profile": prof,
        }

    print("\n=== GATE ACHIEVABILITY SUMMARY ===")
    print(f"real-vs-real (even/odd) min tail-KS p across all services x regions: {global_min_even:.4g}")
    print(f"  -> every service passes (24/24) at p>=0.05: {all_pass_24}")
    print(f"fit-vs-holdout regime-drift min tail-KS p across all services x regions: {global_min_drift:.4g}")
    print(f"  (bound for empirical/parametric tail models trained on the fit window)")

    cond_aucs_all = [v["auc"] for reg in HOLDOUT for v in report[reg]["conditional_predictability"].values() if v.get("auc") is not None]
    report["_summary"] = {
        "achievable_min_p_even_odd": global_min_even,
        "achievable_all_services_pass": all_pass_24,
        "achievable_n_tests": n_tests,
        "regime_drift_min_p": global_min_drift,
        "conditional_mean_auc": float(np.mean(cond_aucs_all)) if cond_aucs_all else None,
    }

    out_path = args.output_dir / "gate_diagnostic.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
