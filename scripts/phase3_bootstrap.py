"""
Phase 3 bootstrap CI + paired Wilcoxon evaluation (modern v2 model).

Runs N episodes per (scenario, battery, impact) cell with random episode
starts, pairing every policy on the same seed (so Wilcoxon signed-rank is
valid). Writes per-episode profits to a CSV, then computes bootstrap CIs
and paired comparisons using src/helper.py.

Focus cells (compute-manageable):
  scenario: sa1_oct_2024 (headline)
  battery:  small, hornsdale
  impact:   identity, piecewise_merit_order
  policies: dt (best RTG per cell), ppo, oracle, fcasrule; oracle_mi in impact
  N episodes per cell (--n-episodes)

Usage:
    python3 scripts/phase3_bootstrap.py [--n-episodes 5] [--out /tmp/phase3_boot.csv]
"""

import sys, time, csv, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl
import torch

DEVICE = "cuda"
MODEL_CKPT = str(Path(__file__).resolve().parents[1] / "models" / "aemo" / "dt" / "hf_v2_modern" / "aemo_dt_fcas_model.pt")
MODEL_CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json")
SCENARIO = ("sa1_oct_2024", "SA1", "2024-10-01", "2024-10-14")
BATTERIES = [
    dict(name="small", capacity=8.0, max_flow=30.0, step_h=0.08333, init_soc=4.0),
    dict(name="hornsdale", capacity=194.0, max_flow=150.0, step_h=0.08333, init_soc=97.0),
]
# Best RTG per cell from the v2 phase3 sweep (SA1 Oct, average-friendly):
# identity peaks at rtg=10 for small, rtg=5-10 for hornsdale; impact varies.
DT_RTG = {
    ('identity', 'small'): 10.0,
    ('identity', 'hornsdale'): 10.0,
    ('piecewise_merit_order', 'small'): 10.0,   # 22,518 vs 14,920 at rtg=0
    ('piecewise_merit_order', 'hornsdale'): 50.0,  # 77,761 at rtg=50 in Oct
}


def load_dt_model():
    from decision_transformer import DecisionTransformer
    from aemo_dt_hf import load_model_kwargs
    config = load_model_kwargs(MODEL_CONFIG)
    init = {k: v for k, v in config.items()
            if k in {'state_dim','act_dim','n_block','h_dim','context_len','n_heads','drop_p',
                     'max_timestep','rope_enabled','rope_max_position','rope_base',
                     'n_kv_heads','qk_norm','tie_weights'}}
    m = DecisionTransformer(**init)
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE, weights_only=False)
    m.load_from_checkpoint(ckpt)
    m.to(DEVICE); m.eval()
    if not hasattr(m, 'return_scale') or m.return_scale is None:
        m.return_scale = float(config.get('return_scale', 1.0))
    return m


def load_scenario():
    from aemo_data import (fetch_aemo_dispatch_price, fetch_aemo_fcas_price,
                           fetch_aemo_generation_by_fuel, build_supply_curve,
                           aggregate_fcas_market_depth)
    from AEMOBatteryEnv import AEMODataPreprocessor
    from datetime import datetime
    label, region, s, e = SCENARIO
    cache = Path("/tmp/scenario_cache") / f"{label}.pkl"
    if cache.exists():
        import pickle
        with open(cache, 'rb') as f:
            return pickle.load(f)
    start = datetime.strptime(s, '%Y-%m-%d'); end = datetime.strptime(e, '%Y-%m-%d')
    prices = fetch_aemo_dispatch_price(start, end, region)
    fl = []
    for svc in ['RAISE6SEC','RAISE60SEC','RAISE5MIN','RAISEREG',
                'LOWER6SEC','LOWER60SEC','LOWER5MIN','LOWERREG']:
        d = fetch_aemo_fcas_price(start, end, region, svc)
        if d.height > 0: fl.append(d)
    fcas = pl.concat(fl)
    gen = fetch_aemo_generation_by_fuel(start, end, region)
    prep = AEMODataPreprocessor(step_duration_hours=0.08332, add_normalized_features=True)
    processed = prep.preprocess_aemo_data(prices, fcas, gen)
    curves = build_supply_curve(region, start, end)
    depth = aggregate_fcas_market_depth(region, start, end, demand_series=processed)
    entry = {'processed': processed, 'curves': curves, 'depth': depth}
    import pickle
    Path("/tmp/scenario_cache").mkdir(exist_ok=True)
    with open(cache, 'wb') as f:
        pickle.dump(entry, f)
    return entry


def run_cell(model, sd, battery, impact, seed, dt_rtg):
    """Run all policies on one (battery, impact) episode seeded by `seed`.
    Returns dict policy -> profit."""
    from AEMOBatteryEnv import AEMOBatteryTradingEnv
    from decision import AEMOAgent
    from aemo_notebook_utils import get_sb3_model_class
    proc = sd['processed']; curves = sd['curves']; depth = sd['depth']
    max_step = proc.shape[0]

    def _mk_env():
        return AEMOBatteryTradingEnv(
            aemo_data=proc, battery_capacity=battery['capacity'], max_battery_flow=battery['max_flow'],
            step_duration=battery['step_h'], init_battery_level=battery['init_soc'],
            max_step=max_step, action_mode='full_fcas', degradation_mode='none',
            battery_life_cost=0.0, random_episode_start=True,
            impact_model=impact, impact_intensity=1.0,
            supply_curves=curves if impact != 'identity' else None,
            fcas_depth=depth if impact != 'identity' else None,
        )

    def _run(agent, env, reset_seed):
        env.reset(seed=reset_seed)
        ep_df, _ = agent.run_episode()
        infos = ep_df['info'].to_list()
        energy = sum(i.get('energy_revenue', 0) for i in infos)
        fcas = sum(i.get('fcas_revenue', 0) for i in infos)
        return energy + fcas

    out = {}
    # DT
    env = _mk_env(); agent = AEMOAgent(env, algorithm='dt', model=model, rtg_value=dt_rtg,
                                       reset_seed=seed)
    out['dt'] = _run(agent, env, seed)
    # FCAS rule
    env = _mk_env(); agent = AEMOAgent(env, algorithm='fcas_rule', reset_seed=seed)
    out['fcasrule'] = _run(agent, env, seed)
    # PPO
    env = _mk_env()
    PPO = get_sb3_model_class('PPO')
    ppo_model = PPO.load(str(Path(__file__).resolve().parents[1] / "models" / "aemo_sb3" / "ppo_aemo_fcas_model.zip"),
                         device=DEVICE)
    env.reset(seed=seed)
    done = False
    infos = []
    while not done:
        obs = env._get_observation()
        act, _ = ppo_model.predict(obs, deterministic=True)
        if isinstance(act, np.ndarray) and act.ndim > 1: act = act.flatten()
        obs, r, done, _, info = env.step(act)
        infos.append(info)
    out['ppo'] = sum(i.get('energy_revenue',0) for i in infos) + sum(i.get('fcas_revenue',0) for i in infos)
    # Oracle_PT
    from aemo_oracle_algo import AEMOOracleSolver
    solver = AEMOOracleSolver(battery_capacity=battery['capacity'], max_battery_flow=battery['max_flow'],
                              step_duration=battery['step_h'], init_soc=battery['init_soc'],
                              min_soc=0.0, max_soc=battery['capacity'])
    r_pt = solver.solve(proc, verbose=False)
    out['oracle'] = r_pt.total_profit
    if impact != 'identity':
        r_mi = solver.solve_mi(proc, curves, depth, impact_intensity=1.0, max_iter=5, verbose=False)
        out['oracle_mi'] = r_mi.total_profit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-episodes', type=int, default=5)
    ap.add_argument('--out', default='/tmp/phase3_boot.csv')
    args = ap.parse_args()

    model = load_dt_model()
    sd = load_scenario()
    print(f"scenario loaded: {sd['processed'].shape[0]} intervals", flush=True)

    rows = []
    for battery in BATTERIES:
        for impact in ['identity', 'piecewise_merit_order']:
            dt_rtg = DT_RTG[(impact, battery['name'])]
            print(f"\ncell: {battery['name']} x {impact} (dt_rtg={dt_rtg})", flush=True)
            for seed in range(args.n_episodes):
                t0 = time.time()
                res = run_cell(model, sd, battery, impact, seed, dt_rtg)
                for pol, prof in res.items():
                    rows.append({'battery': battery['name'], 'impact': impact,
                                 'policy': pol, 'seed': seed, 'profit': prof})
                print(f"  seed {seed}: " + " ".join(f"{k}=${v:,.0f}" for k,v in res.items()) +
                      f"  ({time.time()-t0:.0f}s)", flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['battery','impact','policy','seed','profit'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}", flush=True)

    # Bootstrap CIs
    from helper import bootstrap_confidence_intervals, paired_comparison
    import polars as pl
    df = pl.read_csv(args.out)
    print("\n=== Bootstrap 95% CI (mean profit) ===")
    for impact in ['identity', 'piecewise_merit_order']:
        for battery in [b['name'] for b in BATTERIES]:
            sub = df.filter((pl.col('impact')==impact) & (pl.col('battery')==battery))
            all_logs = {}
            for pol in ['dt','ppo','oracle','fcasrule'] + (['oracle_mi'] if impact!='identity' else []):
                psub = sub.filter(pl.col('policy')==pol)
                # build fake per-episode DataFrames with a reward column
                logs = []
                for r in psub['profit']:
                    logs.append(pl.DataFrame({'reward': [r/1000.0]}))  # profit/1000 ~ reward scale
                all_logs[pol] = logs
            cis = bootstrap_confidence_intervals(all_logs, n_bootstrap=1000, seed=42)
            for pol, c in cis.items():
                print(f"  {impact:>20} {battery:>10} {pol:>10}: ${c['mean']*1000:>10,.0f} "
                      f"[${c['ci_lower']*1000:>9,.0f}, ${c['ci_upper']*1000:>9,.0f}]")

    # Paired Wilcoxon DT vs PPO under impact
    print("\n=== Paired Wilcoxon: DT vs PPO (impact) ===")
    for battery in [b['name'] for b in BATTERIES]:
        sub = df.filter((pl.col('impact')=='piecewise_merit_order') & (pl.col('battery')==battery))
        dt = sub.filter(pl.col('policy')=='dt').sort('seed')['profit'].to_list()
        ppo = sub.filter(pl.col('policy')=='ppo').sort('seed')['profit'].to_list()
        if len(dt) == len(ppo) and len(dt) >= 2:
            dt_logs = [pl.DataFrame({'reward':[d/1000.0]}) for d in dt]
            ppo_logs = [pl.DataFrame({'reward':[p/1000.0]}) for p in ppo]
            res = paired_comparison(dt_logs, ppo_logs)
            print(f"  {battery:>10}: DT ${np.mean(dt):,.0f} vs PPO ${np.mean(ppo):,.0f} "
                  f"mean_diff=${res.get('mean_diff',0)*1000:,.0f} p={res.get('wilcoxon_p','n/a')}")


if __name__ == '__main__':
    main()
