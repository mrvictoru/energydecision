"""
Phase 3: Re-evaluate existing policies under market impact.

Reports profit, timing, and GPU/CPU utilisation for {identity, piecewise_merit_order}
× {DT, PPO, Oracle_PT, dispatch, FCAS rule} on SA1 Oct/Nov 2024.
"""

import sys, time, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import polars as pl
import torch

from aemo_data import (
    fetch_aemo_data_bundle,
    fetch_aemo_dispatch_price,
    fetch_aemo_fcas_price,
    fetch_aemo_generation_by_fuel,
    aggregate_fcas_market_depth,
    aggregate_residual_supply,
    build_supply_curve,
)
from AEMOBatteryEnv import AEMODataPreprocessor, AEMOBatteryTradingEnv
from decision import AEMOAgent
from aemo_dt_hf import (
    modern_v2_model_config_path,
    load_model_kwargs,
)
from market_impact import create_impact_model, PiecewiseMeritOrderImpact
from aemo_oracle_algo import AEMOOracleSolver

# ── Config ──────────────────────────────────────────────────────────────────
SCENARIOS = [
    ("sa1_oct_2024", "SA1", "2024-10-01", "2024-10-14"),
    ("sa1_nov_2024", "SA1", "2024-11-01", "2024-11-14"),
    ("vic1_oct_2024", "VIC1", "2024-10-01", "2024-10-14"),
]
BATTERIES = [
    dict(name="small", capacity=8.0, max_flow=30.0, step_h=0.08333, init_soc=4.0),
    dict(name="hornsdale", capacity=194.0, max_flow=150.0, step_h=0.08333, init_soc=97.0),
    dict(name="torrens", capacity=250.0, max_flow=250.0, step_h=0.08333, init_soc=125.0),
]
RTG_VALUES = [0.0, 5.0, 10.0, 20.0, 50.0]  # DT prompt sweep
PPO_RTG = [0.0]  # PPO doesn't use RTG, but we add it as a policy for comparison
DEVICE = "cuda"
FP16 = False  # skip FP16 — the AEMOAgent code doesn't cast inputs to match model dtype

def now_s():
    return time.perf_counter()

def report_util(label):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  [util] {label}: GPU mem={mem:.2f} GB")

# ── Build price data for a scenario ─────────────────────────────────────────
def fetch_scenario(region, start, end, step_h=0.08332):
    print(f"\n  Fetching {region} {start}–{end} …")
    t0 = now_s()

    prices = fetch_aemo_dispatch_price(start, end, region)
    fcas_list = []
    for svc in ['RAISE6SEC','RAISE60SEC','RAISE5MIN','RAISEREG',
                'LOWER6SEC','LOWER60SEC','LOWER5MIN','LOWERREG']:
        d = fetch_aemo_fcas_price(start, end, region, svc)
        if d.height > 0:
            fcas_list.append(d)
    fcas = pl.concat(fcas_list)
    gen = fetch_aemo_generation_by_fuel(start, end, region)

    prep = AEMODataPreprocessor(step_duration_hours=step_h, add_normalized_features=True)
    processed = prep.preprocess_aemo_data(prices, fcas, gen)
    print(f"  → {processed.shape[0]} intervals, {time.time()-t0:.1f}s")
    return processed

# ── Build supply curves + FCAS depth for a scenario ────────────────────────
def build_market_data(region, start, end, processed):
    print(f"  Building supply curves …")
    t0 = now_s()
    curves = build_supply_curve(region, start, end)
    depth = aggregate_fcas_market_depth(region, start, end, demand_series=processed)
    print(f"  → {curves.shape[0] if curves.height>0 else 0} curve pts, {depth.shape[0]} depth intervals, {time.time()-t0:.1f}s")
    return curves, depth

# ── Run a policy on one env instance ────────────────────────────────────────
def run_policy(env, agent, label):
    t0 = now_s()
    episode_df, _ = agent.run_episode()
    elapsed = now_s() - t0
    infos = episode_df['info'].to_list()
    energy = sum(i.get('energy_revenue', 0) for i in infos)
    fcas = sum(i.get('fcas_revenue', 0) for i in infos)
    deg = sum(i.get('degradation_cost', 0) for i in infos)
    profit = energy + fcas - deg
    return dict(label=label, profit=profit, energy=energy, fcas=fcas, deg=deg, steps=len(infos), time_s=elapsed)

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Phase 3 — Market Impact Evaluation")
    print(f"  Device: {DEVICE}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    report_util("start")

    # 1. Load DT model
    print("\n── Loading modern v2 DT model ──")
    t0 = now_s()
    from decision_transformer import DecisionTransformer
    config_path = modern_v2_model_config_path()
    model_kwargs = load_model_kwargs(config_path)
    # Filter to only constructor args
    init_keys = {'state_dim','act_dim','n_block','h_dim','context_len','n_heads',
                 'drop_p','max_timestep','rope_enabled','rope_max_position','rope_base',
                 'n_kv_heads','qk_norm','tie_weights'}
    model_init_kwargs = {k: v for k, v in model_kwargs.items() if k in init_keys}
    checkpoint_path = str(Path(__file__).resolve().parents[1] / "models" / "aemo" / "dt" / "hf_v2_modern" / "aemo_dt_fcas_model.pt")
    print(f"  Loading modern v2 checkpoint from: {checkpoint_path}")
    dt_model = DecisionTransformer(**model_init_kwargs)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    dt_model.load_from_checkpoint(state)
    dt_model.to(DEVICE)
    if not hasattr(dt_model, 'return_scale') or dt_model.return_scale is None:
        dt_model.return_scale = float(model_kwargs.get('return_scale', 1.0))
    dt_model.eval()
    if FP16:
        dt_model = dt_model.half()
    print(f"  Model loaded: {time.time()-t0:.1f}s, params: {sum(p.numel() for p in dt_model.parameters())/1e6:.1f}M")
    report_util("after DT model load")

    # 1b. Load PPO reference model
    print("\n── Loading PPO reference model ──")
    t0 = now_s()
    from aemo_notebook_utils import get_sb3_model_class
    PPO = get_sb3_model_class('PPO')
    ppo_path = str(Path(__file__).resolve().parents[1] / "models" / "aemo_sb3" / "ppo_aemo_fcas_model.zip")
    ppo_model = PPO.load(ppo_path, device=DEVICE)
    print(f"  PPO loaded: {time.time()-t0:.1f}s")
    report_util("after PPO model load")

    # 2. Build supply curves + depth (once, shared across all runs)
    print("\n── Building market data for scenarios ──")
    scenario_data = {}
    for label, region, start_str, end_str in SCENARIOS:
        from datetime import datetime
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        processed = fetch_scenario(region, start, end)
        curves, depth = build_market_data(region, start, end, processed)
        scenario_data[label] = dict(processed=processed, curves=curves, depth=depth,
                                    region=region, start=start, end=end)
    report_util("after data build")

    # 4. Run evaluations
    results = []
    IMPACT_KINDS = ['identity', 'piecewise_merit_order']

    for impact_kind in IMPACT_KINDS:
        print(f"\n{'='*60}")
        print(f"  Impact model: {impact_kind}")
        print(f"{'='*60}")

        for label, sd in scenario_data.items():
            processed = sd['processed']
            curves = sd['curves']
            depth = sd['depth']
            region = sd['region']
            max_step = processed.shape[0]

            print(f"\n  Scenario: {label} ({max_step} intervals)")

            for battery in BATTERIES:
                bname = battery['name']
                bcap = battery['capacity']
                bflow = battery['max_flow']
                binit = battery['init_soc']
                print(f"    ── battery {bname} ({bcap} MWh / {bflow} MW) ──")

                def _mk_env():
                    return AEMOBatteryTradingEnv(
                        aemo_data=processed,
                        battery_capacity=bcap,
                        max_battery_flow=bflow,
                        step_duration=battery['step_h'],
                        init_battery_level=binit,
                        max_step=max_step,
                        action_mode='full_fcas', degradation_mode='none', battery_life_cost=0.0,
                        random_episode_start=False,
                        impact_model=impact_kind, impact_intensity=1.0,
                        supply_curves=curves if impact_kind != 'identity' else None,
                        fcas_depth=depth if impact_kind != 'identity' else None,
                    )

                # ── Oracle ──
                env = _mk_env()
                agent = AEMOAgent(env, algorithm='aemo_oracle')
                result = run_policy(env, agent, f"{impact_kind}_oracle_{bname}_{label}")
                results.append(result)
                print(f"    {'oracle':>10}: ${result['profit']:>9,.0f}  (E=${result['energy']:>6,.0f}  F=${result['fcas']:>6,.0f})  {result['time_s']:.1f}s")
                del env, agent

                # ── Oracle_MI (impact-aware, only under piecewise_merit_order) ──
                if impact_kind != 'identity':
                    solver = AEMOOracleSolver(
                        battery_capacity=bcap, max_battery_flow=bflow,
                        step_duration=battery['step_h'], init_soc=binit,
                        min_soc=0.0, max_soc=bcap,
                    )
                    result_mi = solver.solve_mi(processed, curves, depth, impact_intensity=1.0,
                                                max_iter=5, verbose=False)
                    results.append(dict(label=f"{impact_kind}_oraclemi_{bname}_{label}",
                                        profit=result_mi.total_profit,
                                        energy=result_mi.energy_revenue,
                                        fcas=result_mi.fcas_revenue,
                                        deg=0.0, steps=result_mi.n_intervals, time_s=0.0))
                    print(f"    {'oracle_mi':>10}: ${result_mi.total_profit:>9,.0f}  (E=${result_mi.energy_revenue:>6,.0f}  F=${result_mi.fcas_revenue:>6,.0f})  0.0s")

                # ── DT with RTG sweep ──
                for rtg in RTG_VALUES:
                    env = _mk_env()
                    agent = AEMOAgent(env, algorithm='dt', model=dt_model, rtg_value=rtg)
                    result = run_policy(env, agent, f"{impact_kind}_dt_rtg{rtg}_{bname}_{label}")
                    results.append(result)
                    print(f"    {'dt_rtg'+str(rtg):>10}: ${result['profit']:>9,.0f}  (E=${result['energy']:>6,.0f}  F=${result['fcas']:>6,.0f})  {result['time_s']:.1f}s")
                    del env, agent

                # ── FCAS rule ──
                env = _mk_env()
                agent = AEMOAgent(env, algorithm='fcas_rule')
                result = run_policy(env, agent, f"{impact_kind}_fcasrule_{bname}_{label}")
                results.append(result)
                print(f"    {'fcas_rule':>10}: ${result['profit']:>9,.0f}  (E=${result['energy']:>6,.0f}  F=${result['fcas']:>6,.0f})  {result['time_s']:.1f}s")
                del env, agent

                # ── PPO reference ──
                env = _mk_env()
                t_ep = now_s()
                env.reset()
                done = False
                infos_ppo = []
                while not done:
                    obs = env._get_observation()
                    act, _ = ppo_model.predict(obs, deterministic=True)
                    if isinstance(act, np.ndarray) and act.ndim > 1:
                        act = act.flatten()
                    obs, reward, done, _, info = env.step(act)
                    infos_ppo.append(info)
                elapsed = now_s() - t_ep
                energy = sum(i.get('energy_revenue', 0) for i in infos_ppo)
                fcas = sum(i.get('fcas_revenue', 0) for i in infos_ppo)
                deg = sum(i.get('degradation_cost', 0) for i in infos_ppo)
                profit = energy + fcas - deg
                result = dict(label=f"{impact_kind}_ppo_{bname}_{label}", profit=profit,
                              energy=energy, fcas=fcas, deg=deg, steps=len(infos_ppo), time_s=elapsed)
                results.append(result)
                print(f"    {'ppo':>10}: ${profit:>9,.0f}  (E=${energy:>6,.0f}  F=${fcas:>6,.0f})  {elapsed:.1f}s")
                del env

    # 5. Summary
    print("\n\n── SUMMARY ──")
    print(f"{'Impact':>10} {'Policy':>14} {'Scenario':>16} {'Profit':>10} {'Energy':>10} {'FCAS':>10} {'Deg':>8} {'Time':>6}")
    print("-"*90)
    for r in sorted(results, key=lambda x: (x['label'].split('_')[0], x['label'].split('_')[1], x['label'].split('_')[2])):
        parts = r['label'].split('_')
        imp = parts[0]
        pol = parts[1]
        sce = '_'.join(parts[2:])
        print(f"{imp:>10} {pol:>14} {sce:>16} ${r['profit']:>8,.0f} ${r['energy']:>8,.0f} ${r['fcas']:>8,.0f} ${r['deg']:>7,.0f} {r['time_s']:>5.1f}s")

    report_util("end")
