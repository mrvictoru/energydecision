"""
Phase 6 FCAS generator evaluation.

Supports both the legacy v1 regime-copula generator and the v2 conditional
diffusion generator, using the same same-period and cross-period harness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fcas_generator_eval import FCAS_COLS, compare, summarize
from synthetic_fcas import FCASDiffusionGenerator, FCASRegimeCopulaGenerator, FCAS_SERVICE_CAPS

DATA = Path(__file__).resolve().parents[1] / "data" / "aemo"
OUT = Path(__file__).resolve().parents[1] / "eval_output" / "phase6_fcas"

DEFAULT_TRAIN_SPECS = [
    ("SA1", "2024-01-01", "2024-07-01"),
    ("NSW1", "2024-01-01", "2024-07-01"),
    ("QLD1", "2024-01-01", "2024-07-01"),
    ("SA1", "2024-07-01", "2025-01-01"),
    ("NSW1", "2024-07-01", "2025-01-01"),
    ("QLD1", "2024-07-01", "2025-01-01"),
]
HOLDOUT = ["SA1", "NSW1", "VIC1"]
H1 = "2024-01-01_2024-07-01"
H2 = "2024-07-01_2025-01-01"
SPEC_PATTERN = re.compile(r"^(?P<region>[A-Z0-9]+):(?P<start>\d{4}-\d{2}-\d{2}):(?P<end>\d{4}-\d{2}-\d{2})$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=["v1", "v2"], default="v2")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument(
        "--train-spec",
        action="append",
        default=[],
        help="Repeatable REGION:YYYY-MM-DD:YYYY-MM-DD slice for the global train set.",
    )
    parser.add_argument(
        "--same-period-span",
        default="2024-01-01:2024-07-01",
        help="Shared holdout-region span used for same-period fit/holdout splits.",
    )
    parser.add_argument(
        "--cross-period-span",
        default="2024-07-01:2025-01-01",
        help="Shared holdout-region span used for cross-period evaluation.",
    )
    parser.add_argument("--same-period-fit-fraction", type=float, default=0.55)
    parser.add_argument("--window-size", type=int, default=288)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--overlap", type=int, default=48)
    parser.add_argument("--diffusion-steps", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tail-quantile", type=float, default=0.95)
    parser.add_argument("--tail-weight", type=float, default=4.0)
    parser.add_argument("--sample-eta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _parse_dataset_spec(spec: str) -> tuple[str, str, str]:
    match = SPEC_PATTERN.match(spec)
    if not match:
        raise ValueError(f"invalid dataset spec {spec!r}; expected REGION:YYYY-MM-DD:YYYY-MM-DD")
    region = match.group("region")
    start = match.group("start")
    end = match.group("end")
    if start >= end:
        raise ValueError(f"dataset spec start must be before end: {spec!r}")
    return region, start, end


def _parse_span(spec: str) -> tuple[str, str]:
    regionless = _parse_dataset_spec(f"X:{spec}")
    return regionless[1], regionless[2]


def _candidate_processed_files(region: str) -> list[tuple[str, str, Path]]:
    pattern = re.compile(
        rf"^processed_{re.escape(region)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})_0\.0833h\.parquet$"
    )
    candidates = []
    for path in sorted(DATA.glob(f"processed_{region}_*_0.0833h.parquet")):
        match = pattern.match(path.name)
        if match:
            candidates.append((match.group(1), match.group(2), path))
    return candidates


def load(region: str, span: str) -> pl.DataFrame:
    start, end = span.split("_", 1)
    return load_interval(region, start, end)


def load_interval(region: str, start: str, end: str) -> pl.DataFrame:
    exact = DATA / f"processed_{region}_{start}_{end}_0.0833h.parquet"
    path = exact if exact.exists() else None
    if path is None:
        req_start_dt = datetime.fromisoformat(start)
        req_end_dt = datetime.fromisoformat(end)
        candidates = [
            (cand_start, cand_end, cand_path)
            for cand_start, cand_end, cand_path in _candidate_processed_files(region)
            if cand_start <= start and cand_end >= end
        ]
        if not candidates:
            raise FileNotFoundError(
                f"no processed parquet covers {region}:{start}:{end} under {DATA}"
            )
        path = min(
            candidates,
            key=lambda item: (
                datetime.fromisoformat(item[1]) - datetime.fromisoformat(item[0]),
                abs((datetime.fromisoformat(item[0]) - req_start_dt).total_seconds()),
                abs((datetime.fromisoformat(item[1]) - req_end_dt).total_seconds()),
            ),
        )[2]

    df = pl.read_parquet(path)
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    if "SETTLEMENTDATE" in df.columns:
        df = df.filter(
            (pl.col("SETTLEMENTDATE") >= pl.lit(start_dt)) & (pl.col("SETTLEMENTDATE") < pl.lit(end_dt))
        )
    for col in FCAS_COLS:
        df = df.with_columns(pl.col(col).clip(0.0, FCAS_SERVICE_CAPS[col]))
    return df


def build_generator(args: argparse.Namespace):
    if args.generator == "v1":
        return FCASRegimeCopulaGenerator(n_states=2)
    return FCASDiffusionGenerator(
        window_size=args.window_size,
        stride=args.stride,
        overlap=args.overlap,
        diffusion_steps=args.diffusion_steps,
        sample_steps=args.sample_steps,
        base_channels=args.base_channels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tail_quantile=args.tail_quantile,
        tail_weight=args.tail_weight,
        sample_eta=args.sample_eta,
        seed=args.seed,
        device=args.device,
    )


def _fit_and_sample(args: argparse.Namespace, fit_df: pl.DataFrame, context_df: pl.DataFrame) -> pl.DataFrame:
    generator = build_generator(args)
    generator.fit(fit_df)
    return generator.sample(context_df)


def _train_specs(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    return [_parse_dataset_spec(spec) for spec in args.train_spec] or list(DEFAULT_TRAIN_SPECS)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    same_period_start, same_period_end = _parse_span(args.same_period_span)
    cross_period_start, cross_period_end = _parse_span(args.cross_period_span)
    train_specs = _train_specs(args)

    train = pl.concat([load_interval(region, start, end) for region, start, end in train_specs])
    print(f"train: {train.height} intervals ({len(train_specs)} slices)")
    print(f"generator: {args.generator}")

    global_generator = build_generator(args)
    global_generator.fit(train)
    print("global generator fitted on configured train specs")

    report: dict[str, object] = {
        "_meta": {
            "generator": args.generator,
            "train_specs": [f"{region}:{start}:{end}" for region, start, end in train_specs],
            "holdout_regions": HOLDOUT,
            "same_period_span": [same_period_start, same_period_end],
            "cross_period_span": [cross_period_start, cross_period_end],
            "window_size": args.window_size,
            "stride": args.stride,
            "overlap": args.overlap,
            "diffusion_steps": args.diffusion_steps,
            "sample_steps": args.sample_steps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "tail_quantile": args.tail_quantile,
            "tail_weight": args.tail_weight,
            "sample_eta": args.sample_eta,
            "seed": args.seed,
            "device": args.device,
            "same_period_fit_fraction": args.same_period_fit_fraction,
        }
    }

    for region in HOLDOUT:
        full = load_interval(region, same_period_start, same_period_end)
        split = int(full.height * args.same_period_fit_fraction)
        fit_df, hold_df = full.head(split), full.tail(full.height - split)
        synth = _fit_and_sample(args, fit_df, hold_df)
        res = compare(hold_df, synth)
        summary = summarize(res)
        report[f"{region}_samesplit"] = summary
        print(f"\n== {region} same-period split holdout (fit {split}, eval {full.height - split}) ==")
        print(json.dumps(summary, indent=2))

    for region in HOLDOUT:
        real = load_interval(region, cross_period_start, cross_period_end)
        synth = global_generator.sample(real)
        print(
            f"\n== {region} cross-period holdout ({cross_period_start} -> {cross_period_end}): "
            f"{real.height} intervals =="
        )
        res = compare(real, synth)
        report[region] = summarize(res)
        print(json.dumps(report[region], indent=2))

        train_ref = pl.concat([load_interval(r, same_period_start, same_period_end) for r in HOLDOUT])
        ref_res = compare(train_ref.head(real.height), real)
        report[f"{region}_h1_shift_reference"] = summarize(ref_res)

    out_path = args.output_dir / f"generator_eval_{args.generator}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}")

    tail_ks = {k: v["tail_ks_min_pvalue"] for k, v in report.items() if k.endswith("_samesplit")}
    gate = all(p >= 0.05 for p in tail_ks.values())
    print(f"\nSAME-PERIOD GATE (all samesplit tail_ks_min_p >= 0.05): {'PASS' if gate else 'FAIL'}")
    print("  per-region samesplit tail_ks_min_p:", tail_ks)
    print(
        "\nCross-period tail_ks_min_p (regime-shift stress):",
        {
            k: v["tail_ks_min_pvalue"]
            for k, v in report.items()
            if not k.endswith("_samesplit") and not k.endswith("_reference") and not k.startswith("_")
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
