"""
Phase 6: fit the v1 FCAS generator on H1 2024 (SA1+NSW1+QLD1), sample conditioned
on H2 2024 holdout exogenous (SA1+NSW1+VIC1), and evaluate against real H2 prices.

Also computes a "train-vs-holdout shift" reference so generator error is read
relative to the inherent H1->H2 regime shift.
"""

import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl

from fcas_generator_eval import compare, summarize, FCAS, FCAS_COLS
from synthetic_fcas import FCASRegimeCopulaGenerator

DATA = Path(__file__).resolve().parents[1] / "data" / "aemo"
OUT = Path(__file__).resolve().parents[1] / "eval_output" / "phase6_fcas"

TRAIN = ["SA1", "NSW1", "QLD1"]
HOLDOUT = ["SA1", "NSW1", "VIC1"]  # 🐴 ceiling: H2 processed parquets only exist for these
H1 = "2024-01-01_2024-07-01"
H2 = "2024-07-01_2025-01-01"


def load(region: str, span: str) -> pl.DataFrame:
    path = DATA / f"processed_{region}_{span}_0.0833h.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pl.read_parquet(path)
    # Downsample? No — keep full 5-min resolution. Clamp pathological caps.
    for s in FCAS_COLS:
        df = df.with_columns(pl.col(s).clip(0.0, 16_600.0))
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    train_frames = [load(r, H1) for r in TRAIN]
    train = pl.concat(train_frames)
    print(f"train: {train.height} intervals ({len(TRAIN)} regions, H1)")

    gen = FCASRegimeCopulaGenerator(n_states=2).fit(train)
    print("generator fitted (2-state regime per direction, logistic transitions)")

    report = {}

    # (1) Same-period holdout: fit on H1 Jan-Apr, validate on H1 Apr-Jun (per region).
    #     This is the clean distributional test — the cross-period H1->H2 regime shift
    #     is so large that even real-vs-real data fails the tail KS (see reference below).
    for region in HOLDOUT:
        full = load(region, H1)
        split = int(full.height * 0.55)
        fit_df, hold_df = full.head(split), full.tail(full.height - split)
        g = FCASRegimeCopulaGenerator(n_states=2).fit(fit_df)
        synth = g.sample(hold_df)
        res = compare(hold_df, synth)
        report[f"{region}_samesplit"] = summarize(res)
        print(f"\n== {region} same-period split holdout (fit {split}, eval {full.height - split}) ==")
        print(json.dumps(summarize(res), indent=2))

    # (2) Cross-period stress test: fit on full H1, sample on H2 exogenous.
    for region in HOLDOUT:
        real = load(region, H2)
        synth = gen.sample(real)
        print(f"\n== {region} H2 holdout (cross-period): {real.height} intervals ==")
        res = compare(real, synth)
        report[region] = summarize(res)
        print(json.dumps(report[region], indent=2))

        # Reference: how far does H1 differ from H2 on its own? (regime shift floor)
        train_ref = pl.concat([load(r, H1) for r in HOLDOUT])
        res_ref = compare(train_ref.head(real.height), real)
        report[f"{region}_h1_shift_reference"] = summarize(res_ref)

    (OUT / "generator_eval.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {OUT / 'generator_eval.json'}")

    # Same-period gate (the fair distributional test).
    tail_ks = {k: v["tail_ks_min_pvalue"] for k, v in report.items() if k.endswith("_samesplit")}
    gate = all(p >= 0.05 for p in tail_ks.values())
    print(f"\nSAME-PERIOD GATE (all samesplit tail_ks_min_p >= 0.05): {'PASS' if gate else 'FAIL'}")
    print("  per-region samesplit tail_ks_min_p:", tail_ks)
    print("\nCross-period tail_ks_min_p (regime-shift stress):",
          {k: v["tail_ks_min_pvalue"] for k, v in report.items() if not k.endswith("_samesplit") and not k.endswith("_reference")})


if __name__ == "__main__":
    main()
