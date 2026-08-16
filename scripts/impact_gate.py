"""Impact gate for the hierarchical dt_soc_oracle / dt_soc_sdp policies.

Runs the hierarchical policies under {identity, piecewise_merit_order} across
the canonical grid-scale batteries and compares against PPO, confirming the
policy does not regress under market impact (the final required gate for any
"best" claim — see docs/aemo_dt_preferred_policy_plan.md §2).

Usage:
  python3 scripts/impact_gate.py \
    --surface-manifest models/aemo/dt/soc_waypoint_dt_loss_surface_manifest.json \
    --model-path models/aemo/dt/soc_waypoint_dt_best.pt \
    --impact-config configs/impact_benchmark.json \
    --output eval_output/exp3a_impact_gate.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision import AEMOAgent
from decision_transformer import DecisionTransformer
from AEMOBatteryEnv import AEMOBatteryTradingEnv
from phase3_impact_eval import fetch_scenario, build_market_data


def load_waypoint_dt(manifest_path: Path, model_path: Path, device: str) -> DecisionTransformer:
    manifest = json.loads(manifest_path.read_text())
    model = DecisionTransformer(**manifest["model_kwargs"])
    model.load_from_checkpoint(str(model_path), map_location=device)
    model.to(device)
    model.eval()
    return model


def run_policy(env, agent, label):
    episode_df, _ = agent.run_episode()
    infos = episode_df["info"].to_list()
    energy = sum(i.get("energy_revenue", 0) for i in infos)
    fcas = sum(i.get("fcas_revenue", 0) for i in infos)
    deg = sum(i.get("degradation_cost", 0) for i in infos)
    profit = energy + fcas - deg
    return dict(label=label, profit=profit, energy=energy, fcas=fcas, deg=deg,
                steps=len(infos))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface-manifest", type=Path,
                    default=Path("models/aemo/dt/soc_waypoint_dt_loss_surface_manifest.json"))
    ap.add_argument("--model-path", type=Path,
                    default=Path("models/aemo/dt/soc_waypoint_dt_best.pt"))
    ap.add_argument("--impact-config", type=Path,
                    default=Path("configs/impact_benchmark.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("eval_output/exp3a_impact_gate.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--executors", default="lp,sdp")
    ap.add_argument("--standalone-dt", action="store_true",
                    help="Also evaluate the waypoint model as a plain 9-dim 'dt' "
                         "(for the Stage B standalone-DT impact gate).")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads(args.impact_config.read_text())
    scenarios = [tuple(s) for s in cfg["scenarios"]]
    batteries = [dict(b) for b in cfg["batteries"]]
    executors = [e.strip() for e in args.executors.split(",") if e.strip()]

    model = load_waypoint_dt(args.surface_manifest, args.model_path, device)

    cache_dir = Path("/tmp/scenario_cache")
    cache_dir.mkdir(exist_ok=True)

    results = []
    from datetime import datetime
    for label, region, start_str, end_str in scenarios:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")
        cache_file = cache_dir / f"{label}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                sd = pickle.load(f)
            print(f"[impact_gate] loaded {label} from cache")
        else:
            processed = fetch_scenario(region, start, end)
            curves, depth = build_market_data(region, start, end, processed)
            sd = dict(processed=processed, curves=curves, depth=depth, region=region)
            with open(cache_file, "wb") as f:
                pickle.dump(sd, f)

        processed = sd["processed"]
        for battery in batteries:
            bname = battery["name"]
            for impact in ["identity", "piecewise_merit_order"]:
                curves = sd["curves"] if impact != "identity" else None
                depth = sd["depth"] if impact != "identity" else None

                for ex in executors:
                    env = AEMOBatteryTradingEnv(
                        aemo_data=processed,
                        battery_capacity=battery["capacity"],
                        max_battery_flow=battery["max_flow"],
                        step_duration=battery["step_h"],
                        init_battery_level=battery["init_soc"],
                        max_step=processed.shape[0],
                        action_mode="full_fcas",
                        degradation_mode="real_world",
                        degradation_chemistry="LFP",
                        degradation_temperature=30.0,
                        random_episode_start=False,
                        impact_model=impact, impact_intensity=1.0,
                        supply_curves=curves, fcas_depth=depth,
                    )
                    agent = AEMOAgent(env, algorithm="dt_soc_oracle", model=model,
                                      rtg_value=0.0, executor=ex, deg_cost_per_mwh=50.0)
                    r = run_policy(env, agent, f"{impact}_dtsoc_{ex}_{bname}_{label}")
                    results.append(r)
                    print(f"  {r['label']}: profit=${r['profit']:>12,.0f} "
                          f"(E=${r['energy']:>9,.0f} F=${r['fcas']:>9,.0f} D=${r['deg']:>9,.0f})")

                if args.standalone_dt:
                    # Stage B standalone DT: the waypoint model is used as a
                    # plain 9-dim DT (its waypoint output is a 9-dim action).
                    env = AEMOBatteryTradingEnv(
                        aemo_data=processed,
                        battery_capacity=battery["capacity"],
                        max_battery_flow=battery["max_flow"],
                        step_duration=battery["step_h"],
                        init_battery_level=battery["init_soc"],
                        max_step=processed.shape[0],
                        action_mode="full_fcas",
                        degradation_mode="real_world",
                        degradation_chemistry="LFP",
                        degradation_temperature=30.0,
                        random_episode_start=False,
                        impact_model=impact, impact_intensity=1.0,
                        supply_curves=curves, fcas_depth=depth,
                    )
                    agent = AEMOAgent(env, algorithm="dt", model=model, rtg_value=0.0)
                    r = run_policy(env, agent, f"{impact}_dt_{bname}_{label}")
                    results.append(r)
                    print(f"  {r['label']}: profit=${r['profit']:>12,.0f} "
                          f"(E=${r['energy']:>9,.0f} F=${r['fcas']:>9,.0f} D=${r['deg']:>9,.0f})")

                # PPO reference (impact-aware env, same battery)
                from aemo_notebook_utils import get_sb3_model_class
                PPO = get_sb3_model_class("PPO")
                ppo = PPO.load("models/aemo_sb3/ppo_aemo_fcas_model.zip", device=device)
                env = AEMOBatteryTradingEnv(
                    aemo_data=processed,
                    battery_capacity=battery["capacity"],
                    max_battery_flow=battery["max_flow"],
                    step_duration=battery["step_h"],
                    init_battery_level=battery["init_soc"],
                    max_step=processed.shape[0],
                    action_mode="full_fcas",
                    degradation_mode="real_world",
                    degradation_chemistry="LFP",
                    degradation_temperature=30.0,
                    random_episode_start=False,
                    impact_model=impact, impact_intensity=1.0,
                    supply_curves=curves, fcas_depth=depth,
                )
                agent = AEMOAgent(env, algorithm="rl", model=ppo)
                r = run_policy(env, agent, f"{impact}_ppo_{bname}_{label}")
                results.append(r)
                print(f"  {r['label']}: profit=${r['profit']:>12,.0f} "
                      f"(E=${r['energy']:>9,.0f} F=${r['fcas']:>9,.0f} D=${r['deg']:>9,.0f})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} results to {args.output}")

    # Summary table: piecewise_merit_order, mean profit per policy per battery.
    print("\n=== piecewise_merit_order summary (per battery, mean over scenarios) ===")
    mi = [r for r in results if r["label"].startswith("piecewise_merit_order")]
    for bname in [b["name"] for b in batteries]:
        for pol in ["dtsoc_lp", "dtsoc_sdp", "ppo"]:
            vals = [r["profit"] for r in mi if pol in r["label"] and f"_{bname}_" in r["label"]]
            if vals:
                print(f"  {bname:>10} {pol:>10}: ${np.mean(vals):>12,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
