#!/usr/bin/env python3
"""
Fetch daily energy-balance measurements from the ennexos/SMA UI API.

Endpoint observed from the portal UI (FUTURE_PLAN §6b H0):

    https://uiapi.sunnyportal.com/api/v1/measurements/{PLANT_ID}/energybalance\
        ?dateBeginLocal=YYYY-MM-DD&interval=Day

Authentication: the UI authenticates via session tokens. This script does NOT
handle login itself — copy your browser session once from DevTools:

  1. F12 -> Network -> pick the energybalance request
  2. Copy the request's `Cookie:` header value   -> export ENNXOS_COOKIE='...'
     and/or its `Authorization: Bearer ...` value -> export ENNXOS_BEARER='...'
  3. Re-copy when the session expires (script exits with code 2 on 401)

Usage:

    # single-day probe (prints status + response head; use before any loop)
    python3 scripts/fetch_ennexos.py --plant-id 10574124 \
        --start 2026-08-20 --end 2026-08-20 --probe

    # full range, one file per day, resumable, polite delay
    python3 scripts/fetch_ennexos.py --plant-id 10574124 \
        --start 2019-01-01 --end 2026-08-01 \
        --interval Day --out-dir data/household/real/raw/ennexos_json

Responses are saved byte-for-byte (raw JSON) — parsing/conversion happens in a
separate step once the schema is known. Credentials are read from environment
variables only and never written anywhere.
"""

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://uiapi.sunnyportal.com/api/v1/measurements/{plant}/energybalance"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def build_url(plant_id: str, date_iso: str, interval: str) -> str:
    return (
        f"{BASE_URL.format(plant=plant_id)}"
        f"?dateBeginLocal={date_iso}&interval={interval}"
    )


def build_headers(cookie: str = "", bearer: str = "") -> dict:
    h = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://ennexos.sunnyportal.com/",
    }
    if bearer:
        h["Authorization"] = f"Bearer {bearer}"
    if cookie:
        h["Cookie"] = cookie
    return h


def iter_dates(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def fetch_day(session: requests.Session, plant_id: str, day: dt.date,
              interval: str, timeout: int = 30) -> requests.Response:
    url = build_url(plant_id, day.isoformat(), interval)
    return session.get(url, timeout=timeout)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plant-id", required=True, help="SMA plant id (numeric path segment)")
    p.add_argument("--start", required=True, help="First date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="Last date YYYY-MM-DD (inclusive)")
    p.add_argument("--interval", default="Day",
                   help="interval query param as used by the UI (default: %(default)s). "
                        "Try other values (e.g. Minute5 / QuarterHour / Week) during probing.")
    p.add_argument("--out-dir", default="data/household/real/raw/ennexos_json",
                   help="Directory for raw per-day JSON captures")
    p.add_argument("--delay", type=float, default=1.5,
                   help="Seconds between requests (politeness throttle)")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the output file exists")
    p.add_argument("--probe", action="store_true",
                   help="Fetch only the FIRST date and print diagnostics; no files written")
    args = p.parse_args(argv)

    cookie = os.environ.get("ENNXOS_COOKIE", "").strip()
    bearer = os.environ.get("ENNXOS_BEARER", "").strip()
    if not cookie and not bearer:
        print("ERROR: set ENNXOS_COOKIE and/or ENNXOS_BEARER (copy from browser "
              "DevTools; see --help). Never commit them.", file=sys.stderr)
        return 1

    try:
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
    except ValueError as e:
        print(f"ERROR: bad date: {e}", file=sys.stderr)
        return 1
    if end < start:
        print("ERROR: --end precedes --start", file=sys.stderr)
        return 1

    session = requests.Session()
    session.headers.update(build_headers(cookie, bearer))

    dates = list(iter_dates(start, end))
    print(f"{len(dates)} day(s) to fetch | plant {args.plant_id} | interval={args.interval}")

    ok = skipped = failed = 0
    for i, day in enumerate(dates):
        out_dir = Path(args.out_dir)
        out_path = out_dir / f"energybalance_{day.isoformat()}_{args.interval}.json"

        if args.probe:
            resp = fetch_day(session, args.plant_id, day, args.interval)
            print(f"PROBE {args.start}: HTTP {resp.status_code}, "
                  f"{len(resp.content)} bytes, content-type={resp.headers.get('content-type')}")
            head = resp.text[:600].replace("\n", " ")
            print(f"HEAD: {head}")
            if resp.status_code == 401:
                print("\n401 Unauthorized — refresh ENNXOS_COOKIE / ENNXOS_BEARER "
                      "from DevTools and retry.", file=sys.stderr)
                return 2
            return 0

        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            skipped += 1
            continue

        try:
            resp = fetch_day(session, args.plant_id, day, args.interval)
            if resp.status_code == 401:
                print(f"\n401 at {day} — session expired. Refresh ENNXOS_COOKIE/"
                      "ENNXOS_BEARER and rerun (existing files are kept/resumed).",
                      file=sys.stderr)
                return 2
            resp.raise_for_status()
            if len(resp.content.strip()) == 0:
                print(f"[EMPTY] {day}", flush=True)
                failed += 1
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(resp.content)
                ok += 1
                if ok % 25 == 0 or i == len(dates) - 1:
                    print(f"...{ok} fetched / {skipped} skipped / {failed} empty-failed "
                          f"(at {day})", flush=True)
        except requests.RequestException as exc:
            failed += 1
            print(f"[FAIL] {day}: {exc}", file=sys.stderr, flush=True)

        time.sleep(args.delay)

    print(f"\nDone: {ok} fetched, {skipped} skipped (resume), {failed} failed.")
    print("Next: inspect the JSON schema, then wire a converter into "
          "scripts/ingest_household_portal_csv.py.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
