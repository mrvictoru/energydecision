"""
Phase 4: generate an impact-aware FCAS training dataset (modern v2 era).

Produces episodes rolled out under the piecewise-linear market-impact model
with real_world degradation, so a retrained Decision Transformer learns to
avoid self-impact (and degradation) across horizons and battery scales.

Sources (each episode = a randomly sampled sub-window within the region's
2021-2023 date range):
  - oracle_mi : impact-aware optimal LP over the sub-window  (short/medium only)
  - oracle_pt : price-taking optimal LP, replayed under impact (contrast set)
  - ppo       : existing SB3 PPO under impact
  - a2c       : existing SB3 A2C under impact
  - dt_v2     : modern v2 DT (self-generated) under impact
  - fcas_rule : fcas_rule under impact

Batteries: 8/50/150/250 MWh. Horizons: short(12d)/medium(8wk)/long(26wk)/xlong(~9mo).
Degradation: real_world (LFP, 30C).

Diversity for the deterministic Oracle LPs comes from sampling different
sub-windows (each solve is a distinct impact-optimal trajectory).

Requires precomputed supply curves + FCAS depth per region over the full
2021-2023 windows (see precompute_supply_curves.py).

Usage:
    python3 scripts/generate_impact_dataset.py [--regions NSW1,QLD1,SA1,TAS1,VIC1]
                                              [--n-episodes 50] [--out data/aemo_dt_impact]
"""

import sys, time, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl
import torch

STEP_DURATION = 5 / 60  # hours (5-min)

# Region -> (cached processed parquet path, supply/depth cache path)
REGION_WINDOWS = {
    "NSW1": ("data/aemo/processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet", "NSW1"),
    "QLD1": ("data/aemo/processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet", "QLD1"),
    "SA1":  ("data/aemo/processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet",  "SA1"),
    "TAS1": ("data/aemo/processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet", "TAS1"),
    "VIC1": ("data/aemo/processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet", "VIC1"),
}

# Battery configs: name -> (capacity_mwh, max_power_mw, init_soc_ratio)
BATTERIES = {
    "b08":  (8.0,  30.0, 0.5),
    "b50":  (50.0, 35.0, 0.5),
    "b150": (194.0, 150.0, 0.5),   # Hornsdale-class
    "b250": (250.0, 250.0, 0.5),   # Torrens-class
}

# Horizon -> max_steps (5-min intervals). Short matches the 14-day eval length.
HORIZONS = {
    "short":  3456,    # 12 days
    "medium": 16128,   # 8 weeks
    "long":   74880,   # 26 weeks
    "xlong":  151200,  # ~9 months (0.0833h * 151200 = 9 months)
}

# Source config: (policy_kind, weight). oracle_mi/oracle_pt restricted to short/medium.
SOURCES = ["oracle_mi", "oracle_pt", "ppo", "a2c", "dt_v2", "fcas_rule"]
SOURCE_WEIGHTS = {"oracle_mi": 0.33, "ppo": 0.22, "dt_v2": 0.17,
                  "a2c": 0.11, "oracle_pt": 0.08, "fcas_rule": 0.08}
# oracle LP sources: short/medium only (LP size grows with horizon)
ORACLE_HORIZONS = ["short", "medium"]

MODEL_CKPT = "models/aemo/dt/hf_v2_modern/aemo_dt_fcas_model.pt"
MODEL_CONFIG = "configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json"


def load_dt_v2():
    from decision_transformer import DecisionTransformer
    from aemo_dt_hf import load_model_kwargs
    config = load_model_kwargs(MODEL_CONFIG)
    init = {k: v for k, v in config.items()
            if k in {'state_dim','act_dim','n_block','h_dim','context_len','n_heads','drop_p',
                     'max_timestep','rope_enabled','rope_max_position','rope_base',
                     'n_kv_heads','qk_norm','tie_weights'}}
    m = DecisionTransformer(**init)
    ckpt = torch.load(MODEL_CKPT, map_location='cuda', weights_only=False)
    m.load_from_checkpoint(ckpt); m.to('cuda'); m.eval()
    if not hasattr(m, 'return_scale') or m.return_scale is None:
        m.return_scale = float(config.get('return_scale', 1.0))
    return m


def make_env(processed, battery, max_step, impact, curves, depth, start_idx, real_world=True):
    from AEMOBatteryEnv import AEMOBatteryTradingEnv
    cap, flow, ratio = battery
    deg = dict(degradation_mode='real_world', degradation_chemistry='LFP',
               degradation_temperature=30.0) if real_world else dict(degradation_mode='none')
    env = AEMOBatteryTradingEnv(
        aemo_data=processed, battery_capacity=cap, max_battery_flow=flow,
        step_duration=STEP_DURATION, init_battery_level=cap * ratio,
        max_step=max_step, action_mode='full_fcas', random_episode_start=False,
        **deg,
        impact_model=impact, impact_intensity=1.0,
        supply_curves=curves if impact != 'identity' else None,
        fcas_depth=depth if impact != 'identity' else None,
    )
    env.reset(options={'episode_start_idx': start_idx})
    return env


def run_oracle(processed, battery, max_step, impact, curves, depth, start_idx, is_mi):
    """Solve Oracle LP over the sub-window [start_idx, start_idx+max_step)."""
    from aemo_oracle_algo import AEMOOracleSolver
    cap, flow, ratio = battery
    solver = AEMOOracleSolver(battery_capacity=cap, max_battery_flow=flow,
                              step_duration=STEP_DURATION, init_soc=cap * ratio,
                              min_soc=0.0, max_soc=cap)
    sub = processed.slice(start_idx, max_step)
    if is_mi:
        res = solver.solve_mi(sub, curves, depth, impact_intensity=1.0, max_iter=3, verbose=False)
    else:
        res = solver.solve(sub, verbose=False)
    return res


def generate_one(policy, processed, battery, max_step, impact, curves, depth, start_idx, model=None):
    """Run one episode of `policy`; returns a per-step polars frame (reward + info)."""
    from AEMOBatteryEnv import AEMOBatteryTradingEnv
    cap, flow, ratio = battery

    if policy in ('oracle_mi', 'oracle_pt'):
        res = run_oracle(processed, battery, max_step, impact, curves, depth,
                         start_idx, is_mi=(policy == 'oracle_mi'))
        # Replay the optimal trajectory through the env to log reward/info with degradation.
        env = make_env(processed, battery, max_step, impact, curves, depth, start_idx)
        actions = np.zeros((res.n_intervals, 9))
        actions[:, 0] = np.clip(-res.optimal_dispatch / flow, -1.0, 1.0)
        # FCAS bid order must match env._fcas_services: RAISEREG,LOWERREG,6S,60S,5MIN...
        env_services = env._fcas_services
        o_raise = {'RAISE6SEC': 0, 'RAISE60SEC': 1, 'RAISE5MIN': 2, 'RAISEREG': 3}
        o_lower = {'LOWER6SEC': 0, 'LOWER60SEC': 1, 'LOWER5MIN': 2, 'LOWERREG': 3}
        for i, svc in enumerate(env_services):
            arr = res.optimal_raise_bids if svc.startswith('RAISE') else res.optimal_lower_bids
            oidx = o_raise[svc] if svc.startswith('RAISE') else o_lower[svc]
            actions[:, 1 + i] = np.clip(arr[:, oidx] / flow, 0.0, 1.0)
        logs = []
        for t in range(res.n_intervals):
            obs, reward, done, _, info = env.step(actions[t].tolist())
            logs.append({'step': t, 'action': actions[t].tolist(),
                         'reward': reward, 'info': info})
        return pl.DataFrame(logs)

    if policy in ('ppo', 'a2c'):
        from aemo_notebook_utils import get_sb3_model_class
        env = make_env(processed, battery, max_step, impact, curves, depth, start_idx)
        mdl = get_sb3_model_class(policy.upper())
        mdl_path = f"models/aemo_sb3/{policy}_aemo_fcas_model.zip"
        m = mdl.load(mdl_path, device='cuda')
        logs = []
        done = False
        step = 0
        while not done and step < max_step:
            obs = env._get_observation()
            act, _ = m.predict(obs, deterministic=True)
            if isinstance(act, np.ndarray) and act.ndim > 1: act = act.flatten()
            obs, reward, done, _, info = env.step(act)
            logs.append({'step': step, 'action': act.tolist(), 'reward': reward, 'info': info})
            step += 1
        return pl.DataFrame(logs)

    if policy == 'dt_v2':
        from decision import AEMOAgent
        env = make_env(processed, battery, max_step, impact, curves, depth, start_idx)
        agent = AEMOAgent(env, algorithm='dt', model=model, rtg_value=10.0)
        ep_df, _ = agent.run_episode()
        return ep_df.select(['step', 'action', 'reward', 'info'])

    if policy == 'fcas_rule':
        from decision import AEMOAgent
        env = make_env(processed, battery, max_step, impact, curves, depth, start_idx)
        agent = AEMOAgent(env, algorithm='fcas_rule')
        ep_df, _ = agent.run_episode()
        return ep_df.select(['step', 'action', 'reward', 'info'])

    raise ValueError(f"Unknown policy {policy}")


def build_episode_plan(regions, n_total):
    """Sample (region, battery, horizon, policy, start_idx) tuples."""
    import random
    rng = random.Random(42)
    plan = []
    horizon_names = list(HORIZONS)
    for _ in range(n_total):
        region = rng.choice(regions)
        battery = rng.choice(list(BATTERIES))
        # Oracle sources: short/medium only
        policy = rng.choices(SOURCES, weights=[SOURCE_WEIGHTS[s] for s in SOURCES])[0]
        if policy in ('oracle_mi', 'oracle_pt'):
            # Bias to short (fast LP); a minority at medium (slower 8-wk LP).
            h = rng.choices(ORACLE_HORIZONS, weights=[0.75, 0.25])[0]
        else:
            h = rng.choices(horizon_names, weights=[0.4, 0.3, 0.2, 0.1])[0]
        max_step = HORIZONS[h]
        processed_path = REGION_WINDOWS[region][0]
        n_rows = pl.scan_parquet(processed_path).select(pl.len()).collect().item()
        max_start = max(0, n_rows - max_step - 1)
        start_idx = rng.randint(0, max_start) if max_start > 0 else 0
        plan.append({'region': region, 'battery': battery, 'horizon': h,
                     'policy': policy, 'max_step': max_step, 'start_idx': start_idx})
    return plan


def worker(task):
    """Process a chunk of episodes in a spawned process. Returns list of written paths."""
    import time as _t
    from pathlib import Path
    out = Path(task['out'])
    scenario = task['scenario_data']
    written = []
    model = None
    for i, ep in enumerate(task['episodes']):
        region = ep['region']
        if region not in scenario:
            continue
        proc = scenario[region]['processed']; curves = scenario[region]['curves']
        depth = scenario[region]['depth']
        bat = BATTERIES[ep['battery']]
        if ep['policy'] == 'dt_v2' and model is None:
            model = load_dt_v2()
        t0 = _t.time()
        try:
            df = generate_one(ep['policy'], proc, bat, ep['max_step'], 'piecewise_merit_order',
                              curves, depth, ep['start_idx'], model=model)
            tag = f"{region}__{ep['policy']}__{ep['horizon']}__{ep['battery']}__ep{i:04d}"
            (out / 'raw_logs').mkdir(parents=True, exist_ok=True)
            df.write_parquet(out / 'raw_logs' / f"{tag}.parquet")
            written.append(tag)
            print(f"    [{ep['region']}] {tag} ({len(df)} steps, {_t.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"    [ERR] {region}/{ep['policy']}/{ep['battery']}/{ep['horizon']}: {e}", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', default='NSW1,QLD1,SA1,TAS1,VIC1')
    ap.add_argument('--n-episodes', type=int, default=50)
    ap.add_argument('--out', default='data/aemo_dt_impact')
    ap.add_argument('--supply-cache', default='/tmp/scenario_cache')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    regions = [r for r in args.regions.split(',') if r]
    out = Path(args.out); (out / 'raw_logs').mkdir(parents=True, exist_ok=True)

    import pickle, os
    scenario = {}
    for region in regions:
        proc_path = Path(REGION_WINDOWS[region][0])
        if not proc_path.exists():
            print(f"[skip] missing processed data for {region}: {proc_path}")
            continue
        processed = pl.read_parquet(proc_path)
        sc = Path(args.supply_cache) / f"{REGION_WINDOWS[region][1]}_supply.pkl"
        if not sc.exists():
            print(f"[WARN] no supply cache for {region} ({sc}); skipping region")
            continue
        with open(sc, 'rb') as f:
            curves, depth = pickle.load(f)
        scenario[region] = {'processed': processed, 'curves': curves, 'depth': depth}
        print(f"{region}: {processed.shape[0]} rows")

    if not scenario:
        print("No regions have both processed data and supply caches. "
              "Wait for precompute_supply_curves.py to finish.")
        return

    plan = build_episode_plan(list(scenario), args.n_episodes)
    print(f"plan: {len(plan)} episodes, {args.workers} workers")

    # Split plan across workers
    import concurrent.futures as cf
    import multiprocessing as mp
    n_workers = min(args.workers, len(plan))
    chunks = [plan[i::n_workers] for i in range(n_workers)]
    tasks = [{'episodes': ch, 'scenario_data': scenario, 'out': str(out)} for ch in chunks]

    t_start = time.time()
    all_written = []
    with cf.ProcessPoolExecutor(max_workers=n_workers,
                                mp_context=mp.get_context('spawn')) as pool:
        for result in pool.map(worker, tasks):
            all_written.extend(result)
    total = time.time() - t_start
    print(f"\nWrote {len(all_written)} episodes to {out}/raw_logs in {total:.0f}s")

    # Assemble into a single parquet (mirror generate_fcas_dataset.py schema)
    frames = []
    for tag in all_written:
        frames.append(pl.read_parquet(out / 'raw_logs' / f"{tag}.parquet"))
    if frames:
        combined = pl.concat(frames, how='vertical_relaxed')
        combined.write_parquet(out / 'aemo_impact_dataset.parquet')
        print(f"Assembled {out / 'aemo_impact_dataset.parquet'} ({len(combined)} rows)")
    print("Next: upload to HF (mrvictoru/AEMO_simulated_impact_trade), then MoLab retrain.")


if __name__ == '__main__':
    main()
