#!/usr/bin/env python3
"""
ennexos fetcher via Playwright browser automation.

Bypasses the Keycloak/OIDC auth dance by driving a real browser. The user logs
in once per run (or the browser session is reused from a persistent profile);
Playwright handles the redirect, the token-exchange step, and the session
cookies that uiapi actually checks.

Why pure-requests (fetch_ennexos.py) wasn't enough:
  - The Authorization Bearer token shown in DevTools is the Keycloak account
    token (aud=account, ~15 min lifetime) — not what uiapi validates.
  - uiapi checks an internal session cookie set after the OIDC callback.
  - Manually replaying both pieces is brittle; Playwright handles the whole
    SSO flow correctly.

Usage:
    python3 scripts/fetch_ennexos_playwright.py --plant-id 10574124 \
        --start 2019-01-01 --end 2026-08-01 \
        --out-dir data/household/real/raw/ennexos_json
"""

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

# Playwright is optional — only required for this entrypoint. Document the
# install command so users know how to enable it.
try:
    from playwright.async_api import async_playwright
except ImportError:
    print(
        "ERROR: playwright not installed. Install with:\n"
        "  pip install playwright\n"
        "  python -m playwright install chromium",
        file=sys.stderr,
    )
    sys.exit(2)


def build_url(plant_id: str, date_iso: str, interval: str) -> str:
    return (
        f"https://uiapi.sunnyportal.com/api/v1/measurements/{plant_id}/energybalance"
        f"?dateBeginLocal={date_iso}&interval={interval}"
    )


def iter_dates(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


async def probe_and_print(page, plant_id: str, day: dt.date, interval: str) -> int:
    url = build_url(plant_id, day.isoformat(), interval)
    resp = await page.request.get(url, headers={"Accept": "application/json"})
    print(f"PROBE {day.isoformat()}: HTTP {resp.status}, "
          f"{len(await resp.body())} bytes, content-type={resp.headers.get('content-type')}")
    body = await resp.text()
    print(f"HEAD: {body[:600].replace(chr(10), ' ')}")
    return resp.status


async def fetch_range(plant_id: str, start: dt.date, end: dt.date, interval: str,
                      out_dir: Path, delay: float, force: bool, headless: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        # Login: navigate to the energy-balance view; the browser follows the
        # OIDC redirect to login.sma.energy, user authenticates once.
        login_url = f"https://ennexos.sunnyportal.com/{plant_id}/monitoring/view-energy-balance"
        print(f"Opening login flow: {login_url}")
        print(">>> If prompted, log in with your ennexos credentials in the browser window. <<<")
        await page.goto(login_url, wait_until="domcontentloaded")
        # Wait until we're actually logged in (URL contains /monitoring/ again,
        # not the auth redirect path). Login may take a minute for 2FA.
        for _ in range(120):
            await page.wait_for_timeout(1000)
            if "/monitoring/" in page.url:
                break
        else:
            print("ERROR: login timed out after 2 minutes.", file=sys.stderr)
            await browser.close()
            return 2
        print("Logged in. Starting fetch loop.\n")

        ok = skipped = failed = 0
        for i, day in enumerate(iter_dates(start, end)):
            out_path = out_dir / f"energybalance_{day.isoformat()}_{interval}.json"
            if out_path.exists() and out_path.stat().st_size > 0 and not force:
                skipped += 1
                continue
            try:
                url = build_url(plant_id, day.isoformat(), interval)
                resp = await page.request.get(url, headers={"Accept": "application/json"})
                if resp.status == 401:
                    print(f"\n401 at {day} — session expired mid-loop. "
                          "Re-run with --headed and re-authenticate.", file=sys.stderr)
                    await browser.close()
                    return 2
                resp.raise_for_status()
                body = await resp.body()
                if not body.strip():
                    print(f"[EMPTY] {day.isoformat()}", flush=True)
                    failed += 1
                else:
                    out_path.write_bytes(body)
                    ok += 1
                    if ok % 25 == 0 or i == (end - start).days:
                        print(f"...{ok} fetched / {skipped} skipped / {failed} empty "
                              f"(at {day.isoformat()})", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {day.isoformat()}: {exc}", file=sys.stderr, flush=True)
            await page.wait_for_timeout(int(delay * 1000))

        await browser.close()
    print(f"\nDone: {ok} fetched, {skipped} skipped (resume), {failed} failed.")
    return 0 if failed == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plant-id", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="Day",
                   help="interval query param; probe Day/Minute5/QuarterHour/Week to find 5-min rows")
    p.add_argument("--out-dir", default="data/household/real/raw/ennexos_json")
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--force", action="store_true")
    p.add_argument("--headed", action="store_true",
                   help="Run with a visible browser window (default headless)")
    p.add_argument("--probe", action="store_true",
                   help="Login once, fetch a single day, print diagnostics, exit")
    args = p.parse_args()

    try:
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
    except ValueError as e:
        print(f"ERROR: bad date: {e}", file=sys.stderr)
        return 1
    if end < start:
        print("ERROR: --end precedes --start", file=sys.stderr)
        return 1

    async def _run():
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=not args.headed)
            context = await browser.new_context()
            page = await context.new_page()
            login_url = f"https://ennexos.sunnyportal.com/{args.plant_id}/monitoring/view-energy-balance"
            print(f"Opening login flow: {login_url}")
            if args.headed:
                print(">>> Log in with your ennexos credentials in the browser window. <<<")
            await page.goto(login_url, wait_until="domcontentloaded")
            for _ in range(180):
                await page.wait_for_timeout(1000)
                if "/monitoring/" in page.url:
                    break
            else:
                print("ERROR: login timed out.", file=sys.stderr)
                await browser.close()
                return 2
            if args.probe:
                rc = await probe_and_print(page, args.plant_id, start, args.interval)
                await browser.close()
                return 0 if 200 <= rc < 300 else 1
            await browser.close()
        return await fetch_range(args.plant_id, start, end, args.interval,
                                 Path(args.out_dir), args.delay, args.force,
                                 headless=not args.headed)

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nInterrupted. Existing files preserved for resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())