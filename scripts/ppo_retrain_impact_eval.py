"""Re-run the nine-cell market-impact gate with a *retrained* PPO baseline.

``scripts/phase3_impact_eval.py`` hard-codes the PPO checkpoint
(``models/aemo_sb3/ppo_aemo_fcas_model.zip``) and writes to
``eval_output/phase3_impact/results.json``. This script imports its scenario
builders and policy runners so the gate is reproduced exactly (same scenarios,
same batteries, same impact model, same DT inference mode) but:

  * the PPO checkpoint is a CLI argument;
  * only the policies you ask for are run (default: DT + PPO);
  * results are written to a caller-supplied path that must NOT be the
    canonical ``eval_output/phase3_impact/results.json``.

The DT leg reproduces the shipped gate exactly (``rtg_mode='auto'``,
``rtg_value=0.0`` => j_t_soc under identity, constant 0.0 under
piecewise-merit-order), so the DT cells double as a regression check against
``eval_output/phase3_impact/results.json``.

Usage (repo root, inside Distrobox):

    python3 scripts/ppo_retrain_impact_eval.py \
        --ppo-model models/aemo_sb3/ppo_retrain_20261125/tuned_seed0_....zip \
        --dt-checkpoint models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt \
        --dt-config models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json \
        --results-path eval_output/ppo_retrain_20261125/phase3_impact/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

CANONICAL_RESULTS = ROOT / "eval_output" / "phase3_impact" / "results.json"


def now_s() -> float:
    return time.perf_counter()


def main() -> None:
    import phase3_impact_eval as p3
    from aemo_notebook_utils import get_sb3_model_class
    from decision_transformer import DecisionTransformer
    from aemo_dt_hf import load_model_kwargs

    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-model", required=True)
    ap.add_argument("--dt-checkpoint", required=True)
    ap.add_argument("--dt-config", required=True)
    ap.add_argument("--dt-label", default="stagec_v2cal_auto")
    ap.add_argument("--results-path", required=True)
    ap.add_argument("--policies", default="dt,ppo")
    ap.add_argument("--rtg", type=float, default=0.0)
    ap.add_argument("--rtg-mode", default="auto", choices=["constant", "j_t_soc", "auto"])
    ap.add_argument("--impact-kinds", nargs="*", default=["identity", "piecewise_merit_order"])
    ap.add_argument("--resume", action="store_true", help="keep existing rows in --results-path")
    args = ap.parse_args()

    results_path = Path(args.results_path)
    if results_path.resolve() == CANONICAL_RESULTS.resolve():
        raise SystemExit("refusing to overwrite the canonical eval_output/phase3_impact/results.json")
    if not str(results_path.resolve()).startswith(str((ROOT / "eval_output").resolve())):
        raise SystemExit("--results-path must live under eval_output/")
    if not Path(args.ppo_model).is_file():
        raise SystemExit(f"PPO checkpoint not found: {args.ppo_model}")

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    results: list[dict] = []
    if args.resume and results_path.is_file():
        results = json.loads(results_path.read_text())
    done_labels = {r["label"] for r in results}

    def save() -> None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(results, indent=2))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- DT (same loading path as phase3_impact_eval.py) -------------------
    dt_model = None
    if "dt" in policies:
        model_kwargs = load_model_kwargs(args.dt_config)
        init_keys = {"state_dim", "act_dim", "n_block", "h_dim", "context_len", "n_heads",
                     "drop_p", "max_timestep", "rope_enabled", "rope_max_position", "rope_base",
                     "n_kv_heads", "qk_norm", "tie_weights", "action_head_mode"}
        init_kwargs = {k: v for k, v in model_kwargs.items() if k in init_keys}
        dt_model = DecisionTransformer(**init_kwargs)
        state = torch.load(args.dt_checkpoint, map_location=device, weights_only=False)
        dt_model.load_from_checkpoint(state)
        meta_path = Path(str(args.dt_checkpoint) + ".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("return_scale") is not None:
                dt_model.return_scale = float(meta["return_scale"])
        dt_model.to(device).eval()
        print(f"DT loaded: {args.dt_checkpoint}")

    ppo_model = None
    if "ppo" in policies:
        PPO = get_sb3_model_class("PPO")
        ppo_model = PPO.load(args.ppo_model, device=device)
        print(f"PPO loaded: {args.ppo_model}")

    # ---- scenario data (identical builders/caches to phase3_impact_eval) ---
    import pickle

    cache_dir = Path("/tmp/scenario_cache")
    cache_dir.mkdir(exist_ok=True)
    from datetime import datetime

    scenario_data = {}
    for label, region, start_str, end_str in p3.SCENARIOS:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        cache_file = cache_dir / f"{label}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as fh:
                scenario_data[label] = pickle.load(fh)
            print(f"scenario {label}: cache hit")
            continue
        processed = p3.fetch_scenario(region, start, end)
        curves, depth = p3.build_market_data(region, start, end, processed)
        entry = dict(processed=processed, curves=curves, depth=depth,
                     region=region, start=start, end=end)
        scenario_data[label] = entry
        with open(cache_file, "wb") as fh:
            pickle.dump(entry, fh)
        print(f"scenario {label}: built")

    def make_env(sd, battery, impact_kind):
        from AEMOBatteryEnv import AEMOBatteryTradingEnv

        return AEMOBatteryTradingEnv(
            aemo_data=sd["processed"],
            battery_capacity=battery["capacity"],
            max_battery_flow=battery["max_flow"],
            step_duration=battery["step_h"],
            init_battery_level=battery["init_soc"],
            max_step=sd["processed"].shape[0],
            action_mode="full_fcas",
            degradation_mode="real_world",
            degradation_chemistry="LFP",
            degradation_temperature=30.0,
            random_episode_start=False,
            impact_model=impact_kind,
            impact_intensity=1.0,
            supply_curves=sd["curves"] if impact_kind != "identity" else None,
            fcas_depth=sd["depth"] if impact_kind != "identity" else None,
        )

    def run_ppo(env):
        from decision import AEMOAgent  # noqa: F401  (kept for parity of imports)

        t0 = now_s()
        env.reset()
        done = False
        energy = fcas = deg = 0.0
        steps = 0
        while not done:
            obs = env._get_observation()
            act, _ = ppo_model.predict(obs, deterministic=True)
            if isinstance(act, np.ndarray) and act.ndim > 1:
                act = act.flatten()
            obs, _reward, done, _truncated, info = env.step(act)
            energy += float(info.get("energy_revenue", 0.0))
            fcas += float(info.get("fcas_revenue", 0.0))
            deg += float(info.get("degradation_cost", 0.0))
            steps += 1
        return dict(profit=energy + fcas - deg, energy=energy, fcas=fcas,
                    deg=deg, steps=steps, time_s=now_s() - t0)

    for impact_kind in args.impact_kinds:
        for label, sd in scenario_data.items():
            for battery in p3.BATTERIES:
                bname = battery["name"]
                if "dt" in policies:
                    lab = f"{impact_kind}_{args.dt_label}_rtg{args.rtg}_{bname}_{label}"
                    if lab in done_labels:
                        print(f"skip {lab}")
                    else:
                        from decision import AEMOAgent
                        env = make_env(sd, battery, impact_kind)
                        agent = AEMOAgent(env, algorithm="dt", model=dt_model,
                                          rtg_value=args.rtg, rtg_mode=args.rtg_mode)
                        res = p3.run_policy(env, agent, lab)
                        results.append(res)
                        save()
                        print(f"{lab}: ${res['profit']:,.0f} ({res['time_s']:.1f}s)", flush=True)
                        del env, agent
                if "ppo" in policies:
                    lab = f"{impact_kind}_ppo_{bname}_{label}"
                    if lab in done_labels:
                        print(f"skip {lab}")
                    else:
                        env = make_env(sd, battery, impact_kind)
                        res = run_ppo(env)
                        results.append(dict(label=lab, **res))
                        save()
                        print(f"{lab}: ${res['profit']:,.0f} ({res['time_s']:.1f}s)", flush=True)
                        del env

    print(f"saved {len(results)} rows -> {results_path}")


if __name__ == "__main__":
    main()
