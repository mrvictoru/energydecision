from __future__ import annotations

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

from aemo_data import AEMO_MONTHLY_CACHE_TABLES, fetch_aemo_monthly_cache_files


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_date(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download AEMO monthly MMS archive zips into local NEMOSIS-compatible cache files."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root() / "data" / "aemo",
        help="Destination cache directory (default: <repo_root>/data/aemo).",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(AEMO_MONTHLY_CACHE_TABLES),
        help="MMS table names to download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing cache files after validating the downloaded archive.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Retry count for transient download or DNS failures.",
    )

    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument(
        "--year",
        type=int,
        help="Fetch a full calendar year (for example: --year 2025).",
    )
    window.add_argument(
        "--start-date",
        type=_parse_date,
        help="Fetch all months touched by [start-date, end-date). Requires --end-date.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="Exclusive end of the fetch window when using --start-date.",
    )
    args = parser.parse_args(argv)
    if args.start_date is not None and args.end_date is None:
        parser.error("--end-date is required when --start-date is used.")
    if args.start_date is None and args.end_date is not None:
        parser.error("--start-date is required when --end-date is used.")
    return args


def _resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    if args.year is not None:
        return datetime(args.year, 1, 1), datetime(args.year + 1, 1, 1)
    assert args.start_date is not None and args.end_date is not None
    return args.start_date, args.end_date


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_date, end_date = _resolve_window(args)
    manifest = fetch_aemo_monthly_cache_files(
        start_date=start_date,
        end_date=end_date,
        tables=args.tables,
        cache_dir=str(args.cache_dir.resolve()),
        overwrite=bool(args.overwrite),
        timeout=int(args.timeout),
        max_attempts=int(args.max_attempts),
    )

    downloaded = sum(1 for entry in manifest if entry["status"] == "downloaded")
    existing = sum(1 for entry in manifest if entry["status"] == "existing")
    print(
        f"Fetched monthly cache entries for {start_date.date()} to {end_date.date()} "
        f"into {args.cache_dir.resolve()}."
    )
    print(f"Downloaded: {downloaded}")
    print(f"Existing: {existing}")
    for entry in manifest:
        print(
            f"{entry['status'].upper():10s} "
            f"{entry['year']}-{entry['month']:02d} "
            f"{entry['table_name']} -> {entry['path']}"
        )


if __name__ == "__main__":
    main()
