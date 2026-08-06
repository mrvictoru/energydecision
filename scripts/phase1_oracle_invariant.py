"""
Phase 1 invariant test: Oracle_PT >= any replayed policy on shared episodes.

Runs Oracle_PT, DT (v2 + impact), PPO, FCAS rule, and dispatch replay on the same
identity-impact episodes with degradation_mode='none' (clean price-taker ceiling:
net profit == revenue, which is exactly what the LP optimizes). Asserts the
oracle dominates every policy per cell, and dumps results to eval_output/phase1_invariant/.

The real_world-degradation counterpart is analysed separately from the existing
eval_output/phase3_impact/results.json (revenue 9/9, net 7/9 — the LP is
degradation-blind; see plan diary).
"""

import sys, time, json, pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from AEMOBatteryEnv import AEMOBatteryTradingEnv
from decision import AEMOAgent
from aemo_dt_hf import modern_v2_model_config_path, load_model_kwargs
from phase3_impact_eval import (
    SCENARIOS,
    BATTERIES,
    RTG_VALUES,
    fetch_scenario,
    build_market_data,
    run_policy,
)

REPO = Path(__file__).resolve().parents[1]
DEVICE = "cuda"
RESULTS = REPO / "eval_output" / "phase3_impact" / "results.json"
OUT = REPO / "eval_output" / "phase1_invariant" / "results.json"
V2_CKPT = REPO / "models" / "aemo" / "dt" / "hf_v2_modern" / "aemo_dt_fcas_model.pt"
IMPACT_CKPT = REPO / "models" / "aemo" / "dt" / "hf_v2_impact" / "aemo_dt_fcas_model.pt"
CACHE = Path("/tmp/scenario_cache")


def load_dt(path):
    from decision_transformer import DecisionTransformer
    kw = load_model_kwargs(modern_v2_model_config_path())
    keys = {'state_dim', 'act_dim', 'n_block', 'h_dim', 'context_len', 'n_heads',
            'drop_p', 'max_timestep', 'rope_enabled', 'rope_max_position', 'rope_base',
            'n_kv_heads', 'qk_norm', 'tie_weights'}
    model = DecisionTransformer(**{k: v for k, v in kw.items() if k in keys})
    model.load_from_checkpoint(torch.load(path, map_location=DEVICE, weights_only=False))
    if not hasattr(model, 'return_scale') or model.return_scale is None:
        model.return_scale = float(kw.get('return_scale', 1.0))
    model.to(DEVICE).eval()
    return model


def best_rtg_per_cell(label_tag):
    """Best identity RTG per (battery, scenario) for a DT label, from the real_world run."""
    data = json.loads(RESULTS.read_text())
    best = {}
    for e in data:
        lbl = e['label']
        if not lbl.startswith(f"identity_{label_tag}_rtg"):
            continue
        body = lbl[len(f"identity_{label_tag}_rtg"):]  # e.g. "10.0_small_sa1_oct_2024"
        rtg, _, bat, scen = body.split('_', 3) if '_' in body else (None, None, None, None)
        if rtg is None:
            continue
        key = (bat, scen)
        if key not in best or e['profit'] > best[key][1]:
            best[key] = (float(rtg), e['profit'])
    return {k: v[0] for k, v in best.items()}


def main():
    v2 = load_dt(V2_CKPT)
    impact = load_dt(IMPACT_CKPT)
    print(f"models loaded: v2 {sum(p.numel() for p in v2.parameters())/1e6:.1f}M, "
          f"impact {sum(p.numel() for p in impact.parameters())/1e6:.1f}M")

    from aemo_notebook_utils import get_sb3_model_class
    ppo_model = get_sb3_model_class('PPO').load(
        str(REPO / 'models' / 'aemo_sb3' / 'ppo_aemo_fcas_model.zip'), device=DEVICE)

    v2_rtg = best_rtg_per_cell('dt_v2')
    imp_rtg = best_rtg_per_cell('dt_impact')
    print("best RTG/cell v2:", v2_rtg)
    print("best RTG/cell impact:", imp_rtg)

    scenario_data = {}
    for label, region, start_s, end_s in SCENARIOS:
        cache_file = CACHE / f"{label}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                scenario_data[label] = pickle.load(f)
            print(f"  {label}: loaded from cache")
        else:
            from datetime import datetime
            start, end = datetime.strptime(start_s, "%Y-%m-%d"), datetime.strptime(end_s, "%Y-%m-%d")
            processed = fetch_scenario(region, start, end)
            curves, depth = build_market_data(region, start, end, processed)
            scenario_data[label] = dict(processed=processed, curves=curves, depth=depth,
                                        region=region, start=start, end=end)
            with open(cache_file, 'wb') as f:
                pickle.dump(scenario_data[label], f)

    results = []
    violations = []
    for label, sd in scenario_data.items():
        processed, region = sd['processed'], sd['region']
        max_step = processed.shape[0]
        for battery in BATTERIES:
            b = battery['name']
            def mk_env():
                return AEMOBatteryTradingEnv(
                    aemo_data=processed, battery_capacity=battery['capacity'],
                    max_battery_flow=battery['max_flow'], step_duration=battery['step_h'],
                    init_battery_level=battery['init_soc'], max_step=max_step,
                    action_mode='full_fcas', degradation_mode='none',
                    random_episode_start=False,
                    impact_model='identity', impact_intensity=1.0,
                    supply_curves=None, fcas_depth=None)
            cell = f"{b}_{label}"
            # Oracle_PT (price-taking LP) — the ceiling under test.
            env = mk_env(); res = run_policy(env, AEMOAgent(env, algorithm='aemo_oracle'), f"oracle_{cell}")
            results.append(res); oracle = res['profit']
            print(f"  oracle {cell}: ${oracle:,.0f} ({res['time_s']:.1f}s)")
            del env

            policies = []
            env = mk_env(); policies.append(run_policy(env, AEMOAgent(env, algorithm='dt', model=v2, rtg_value=v2_rtg.get((b, label), 10.0)), f"dt_v2_{cell}"))
            del env
            env = mk_env(); policies.append(run_policy(env, AEMOAgent(env, algorithm='dt', model=impact, rtg_value=imp_rtg.get((b, label), 10.0)), f"dt_impact_{cell}"))
            del env
            env = mk_env(); policies.append(run_policy(env, AEMOAgent(env, algorithm='fcas_rule'), f"fcasrule_{cell}"))
            del env
            # PPO (single-obs deterministic rollout)
            env = mk_env(); env.reset(); done = False; infos = []
            while not done:
                act, _ = ppo_model.predict(env._get_observation(), deterministic=True)
                act = np.asarray(act).flatten() if hasattr(act, 'flatten') else act
                _, _, done, _, info = env.step(act)
                infos.append(info)
            energy = sum(i.get('energy_revenue', 0) for i in infos)
            fcas = sum(i.get('fcas_revenue', 0) for i in infos)
            policies.append(dict(label=f"ppo_{cell}", profit=energy + fcas, energy=energy,
                                 fcas=fcas, deg=0.0, steps=len(infos), time_s=0.0))
            del env

            if region == "SA1":
                from aemo_data import fetch_aemo_unit_dispatch
                disp = fetch_aemo_unit_dispatch(sd['start'], sd['end'],
                                                duid="DALNTH1", region=region)
                env = mk_env()
                agent = AEMOAgent(env, algorithm='dispatch', dispatch_data=disp,
                                  dispatch_duid="DALNTH1", dispatch_duid_gen="DALNTH1")
                policies.append(run_policy(env, agent, f"dispatch_{cell}"))
                del env

            results.extend(policies)
            worst = max(policies, key=lambda p: p['profit'])
            ok = oracle >= worst['profit'] - 1e-6
            print(f"    best other: {worst['label']} ${worst['profit']:,.0f}  invariant={ok}")
            if not ok:
                violations.append((cell, oracle, worst['profit'], worst['label']))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} results -> {OUT}")
    print(f"\nINVARIANT RESULT: {'PASS' if not violations else 'FAIL'}")
    if violations:
        for cell, o, w, lbl in violations:
            print(f"  VIOLATION {cell}: oracle ${o:,.0f} < {lbl} ${w:,.0f}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
