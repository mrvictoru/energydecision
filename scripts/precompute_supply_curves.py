"""Precompute supply curves + FCAS depth for the Phase 4 (2021-2023) training windows.

The impact model needs per-interval supply curves + FCAS depth for every region
over its full training window. DISPATCHLOAD is only cached from 2022-12 onward,
so this downloads the missing months once (~7 hr total) and caches the result.

Caches to /tmp/scenario_cache/<REGION>_supply.pkl as (curves_df, depth_df).
Usage: python3 scripts/precompute_supply_curves.py [--regions NSW1,QLD1,SA1,TAS1,VIC1]
"""
import sys, time, pickle, argparse
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_data import build_supply_curve, aggregate_fcas_market_depth
import polars as pl

REGIONS = {
    "NSW1": ("2021-01-01", "2023-04-01", "data/aemo/processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet"),
    "QLD1": ("2021-01-01", "2023-04-01", "data/aemo/processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet"),
    "SA1":  ("2022-04-01", "2023-12-01", "data/aemo/processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet"),
    "TAS1": ("2021-01-01", "2023-04-01", "data/aemo/processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet"),
    "VIC1": ("2021-04-01", "2023-12-01", "data/aemo/processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', default='NSW1,QLD1,SA1,TAS1,VIC1')
    ap.add_argument('--cache', default='/tmp/scenario_cache')
    args = ap.parse_args()
    cache_dir = Path(args.cache); cache_dir.mkdir(exist_ok=True)

    for region in [r for r in args.regions.split(',') if r]:
        s, e, proc_path = REGIONS[region]
        out = cache_dir / f"{region}_supply.pkl"
        if out.exists():
            print(f"{region}: cached, skip"); continue
        start = datetime.strptime(s, '%Y-%m-%d'); end = datetime.strptime(e, '%Y-%m-%d')
        t0 = time.time()
        print(f"{region}: building supply curves {s}..{e} ...", flush=True)
        curves = build_supply_curve(region, start, end)
        print(f"{region}: supply {curves.height if curves.height>0 else 0} rows in {time.time()-t0:.0f}s; fcas depth...", flush=True)
        dem = pl.read_parquet(proc_path).select(['SETTLEMENTDATE', 'TOTALDEMAND'])
        depth = aggregate_fcas_market_depth(region, start, end, demand_series=dem)
        with open(out, 'wb') as f:
            pickle.dump((curves, depth), f)
        print(f"{region}: cached ({time.time()-t0:.0f}s)", flush=True)

    print("All regions precomputed.")


if __name__ == '__main__':
    main()
