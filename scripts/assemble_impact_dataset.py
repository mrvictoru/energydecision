"""
Final assembly + observation reconstruction for the Phase 4 impact dataset.

The generated raw episodes store (step, action, reward, info) but not the
env observation. This script reconstructs the exact 18-dim normalized
observation from the source aemo_data rows + recorded SOC, then assembles
all episodes into the DT training schema:
    step, norm_observation(List<f32>), action(List<f64>), reward,
    episode_id, source_policy

start_idx per episode is recovered by matching the first ~10 recorded
energy_price values against the region's RRP series (unique under the
short horizon; verified across all sources).

Usage: python3 scripts/assemble_impact_dataset.py [--out data/aemo_dt_impact/aemo_impact_dataset.parquet]
"""

import sys, re, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl

RAW_DIR = Path("data/aemo_dt_impact/raw_logs")
REGION_PROC = {
    "NSW1": "data/aemo/processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet",
    "QLD1": "data/aemo/processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet",
    "SA1":  "data/aemo/processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet",
    "TAS1": "data/aemo/processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet",
    "VIC1": "data/aemo/processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet",
}
BAT_CAP = {"b08": 8.0, "b50": 50.0, "b150": 194.0, "b250": 250.0}
FCAS_SERVICES = ['RAISEREG', 'LOWERREG', 'RAISE6SEC', 'LOWER6SEC',
                 'RAISE60SEC', 'LOWER60SEC', 'RAISE5MIN', 'LOWER5MIN']
GEN_FUELS = ['solar', 'wind']


def recover_start(prices_ep, rrp, n_check=10):
    prices = np.asarray(prices_ep[:n_check], dtype=float)
    rrp = np.asarray(rrp, dtype=float)
    cand = np.where(np.abs(rrp[:-n_check] - prices[0]) < 0.5)[0]
    for c in cand:
        if np.all(np.abs(rrp[c:c+n_check] - prices) < 0.5):
            return int(c)
    return None


def build_obs_frame(proc, start_idx, socs_after, cap, init_soc_ratio=0.5):
    """Build the 18-dim normalized observation per step from aemo_data + SOC.

    obs at step t uses the SOC BEFORE step t: init_soc for t=0, else the SOC
    recorded after step t-1 (info[t-1]['battery_soc']).
    """
    n = len(socs_after)
    # socs_before[t] = init_soc if t==0 else socs_after[t-1]
    socs_before = np.empty(n, dtype=np.float64)
    socs_before[0] = cap * init_soc_ratio
    socs_before[1:] = np.asarray(socs_after[:-1], dtype=np.float64)
    sub = proc.slice(start_idx, n)
    obs_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_peak',
                'RRP_normalized', 'DEMAND_normalized']
    for s in FCAS_SERVICES:
        obs_cols.append(f'FCAS_{s}_normalized')
    for f in GEN_FUELS:
        obs_cols.append(f'GEN_{f}_pct')
    # Build obs robustly: missing columns (e.g. GEN_solar_pct) default to 0,
    # matching env .get(col, 0).
    obs = np.zeros((n, len(obs_cols)), dtype=np.float32)
    for j, col in enumerate(obs_cols):
        if col in sub.columns:
            obs[:, j] = sub[col].to_numpy()
    soc_norm = (socs_before.astype(np.float32) / cap).reshape(-1, 1)
    obs = np.concatenate([obs, soc_norm], axis=1)  # (n, 18)
    return obs


def main():
    import argparse, shutil
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/aemo_dt_impact/aemo_impact_dataset.parquet')
    ap.add_argument('--staging', default='data/aemo_dt_impact/assemble_stage')
    args = ap.parse_args()

    proc_cache = {}
    files = sorted(glob.glob(str(RAW_DIR / "*.parquet")))
    print(f"processing {len(files)} episodes", flush=True)
    staging = Path(args.staging)
    shutil.rmtree(staging, ignore_errors=True); staging.mkdir(parents=True, exist_ok=True)

    ep_id = 0
    for f in files:
        name = Path(f).stem
        parts = name.split('__')
        region, policy, horizon, battery = parts[0], parts[1], parts[2], parts[3]
        cap = BAT_CAP.get(battery, 8.0)
        if region not in proc_cache:
            proc_cache[region] = pl.read_parquet(REGION_PROC[region])
        proc = proc_cache[region]
        rrp = proc['RRP'].to_numpy()

        df = pl.read_parquet(f)
        infos = df['info'].to_list()
        prices_ep = [i['energy_price'] for i in infos]
        socs = [i['battery_soc'] for i in infos]
        start = recover_start(prices_ep, rrp)
        if start is None or start < 0 or start + len(df) > len(proc):
            print(f"  [SKIP] {name}: start_idx not recoverable", flush=True)
            continue
        obs = build_obs_frame(proc, start, socs, cap)
        n = len(df)
        ep_frame = pl.DataFrame({
            'episode_id': pl.Series([ep_id] * n, dtype=pl.Int32),
            'step': pl.Series(np.arange(n), dtype=pl.Int64),
            'norm_observation': pl.Series([obs[t].tolist() for t in range(n)], dtype=pl.List(pl.Float32)),
            'action': pl.Series([[float(x) for x in df['action'][t]] for t in range(n)], dtype=pl.List(pl.Float64)),
            'reward': pl.Series(df['reward'].to_list(), dtype=pl.Float32),
            'source_policy': pl.Series([policy] * n, dtype=pl.Utf8),
        })
        ep_frame.write_parquet(staging / f"{ep_id:05d}.parquet")
        ep_id += 1
        if ep_id % 100 == 0:
            print(f"  {ep_id} episodes staged", flush=True)

    # Stream-concat all staged parquets
    print(f"concatenating {ep_id} staged episodes ...", flush=True)
    lazy = pl.scan_parquet(str(staging / "*.parquet"))
    lazy.sink_parquet(args.out)
    out = pl.read_parquet(args.out)
    print(f"\nWrote {args.out}: {out.height} rows, {ep_id} episodes", flush=True)
    print(f"Schema: {out.schema}", flush=True)
    print(f"Source policy counts: {out.group_by('source_policy').len().sort('source_policy')}", flush=True)


if __name__ == '__main__':
    main()
