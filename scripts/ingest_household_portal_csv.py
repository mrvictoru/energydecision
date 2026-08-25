#!/usr/bin/env python3
"""
Ingest raw household portal CSVs into the SolarBatteryEnv schema.

FUTURE_PLAN §6b (H0). One command per batch of weekly downloads:

    python3 scripts/ingest_household_portal_csv.py \
      --input "data/household/real/raw/*.csv" \
      --resolution-minutes 5

Outputs normalized parquet files + a shareable manifest (checksums and
validation stats only — no metering values) under data/household/real/.
Raw telemetry never leaves data/household/real/ (gitignored).

If the portal's column names are not auto-detected, pass an explicit map:

    python3 scripts/ingest_household_portal_csv.py --input ... \
      --column-map '{"Timestamp": "Date Time", "SolarGen": "Solar (kW)", "HouseLoad": "Consumption (kW)"}'
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from household_ingest import (  # noqa: E402
    ingest_file,
    ingest_sma_file,
    is_sma_energy_balance_csv,
    update_manifest,
)

DEFAULT_REAL_DIR = Path("data/household/real")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Glob or directory of raw portal CSVs (e.g. 'data/household/real/raw/*.csv')")
    p.add_argument("--output-dir", default=str(DEFAULT_REAL_DIR / "normalized"),
                   help="Where normalized parquet files are written (default: %(default)s)")
    p.add_argument("--manifest", default="household_ingest_manifest.json",
                   help="Shareable manifest path (checksums + stats, no metering values). "
                        "Default lives at repo root because data/ is fully gitignored.")
    p.add_argument("--column-map", default=None,
                   help="JSON mapping canonical -> portal column names; omit for auto-detection")
    p.add_argument("--resolution-minutes", type=int, default=5,
                   help="Expected telemetry resolution in minutes (default: %(default)s)")
    p.add_argument("--tariff-import", type=float, default=0.30,
                   help="Flat import price $/kWh used when the portal has no tariff channel")
    p.add_argument("--tariff-export", type=float, default=0.05,
                   help="Flat export/FiT price $/kWh used when the portal has no tariff channel")
    p.add_argument("--decimal-comma", action="store_true", default=False,
                   help="CSV numerics use decimal commas (European locale, e.g. SMA/ennexos exports)")
    p.add_argument("--watts-to-kilo", action="store_true", default=False,
                   help="Portal reports power in W (typical for SMA/ennexos); convert to kW")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
    else:
        import glob as _glob
        csv_files = sorted(_glob.glob(args.input))
    if not csv_files:
        print(f"No CSVs matched: {args.input}", file=sys.stderr)
        return 1

    column_map = json.loads(args.column_map) if args.column_map else None
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    reports = []
    failures = 0
    csv_files = [Path(f) for f in csv_files]
    for f in csv_files:
        try:
            if is_sma_energy_balance_csv(f):
                out_path, report = ingest_sma_file(
                    f, output_dir,
                    tariff_import=args.tariff_import,
                    tariff_export=args.tariff_export,
                )
            else:
                out_path, report = ingest_file(
                    f, output_dir,
                    column_map=column_map,
                    expected_resolution_minutes=args.resolution_minutes,
                    tariff_import=args.tariff_import,
                    tariff_export=args.tariff_export,
                    decimal_comma=args.decimal_comma,
                    watts_to_kilo=args.watts_to_kilo,
                )
            reports.append(report)
            status = "OK" if not report.warnings else f"OK ({len(report.warnings)} warnings)"
            print(f"[{status}] {f.name} -> {out_path} ({report.rows_out} rows)")
            for w in report.warnings:
                print(f"    ! {w}")
        except Exception as exc:  # keep going: one bad week must not kill a batch
            failures += 1
            print(f"[FAIL] {f.name}: {exc}", file=sys.stderr)

    manifest = update_manifest(manifest_path, reports)
    print(f"\nManifest updated: {manifest_path} ({len(manifest['files'])} files tracked)")
    print("NOTE: raw + normalized telemetry stay local-only (gitignored). "
          "The manifest contains checksums/stats only and is safe to commit.")
    return 1 if failures and not reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
