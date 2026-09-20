"""Focused verification of the impact-aware J_t(soc) path (known_issues B1).

Runs the shipped Stage C DT under piecewise merit-order impact on the three
canonical grid-scale batteries, comparing explicit ``rtg_mode="j_t_soc"``
against ``rtg_mode="constant"`` (rtg=0.0), and reports
``profit = energy + fcas - degradation``.

Why: the B1 fix corrected the dispatch sign used when re-pricing the J_t(soc)
table through the market-impact model. Before the fix, explicit j_t_soc was
reported to collapse on large batteries (hornsdale ~-$143k, torrens ~-$348k);
the constant fallback passed. This harness re-checks the post-fix behaviour.

It also loads ``return_scale`` from the checkpoint ``.meta.json`` sidecar
explicitly (``phase3_impact_eval.py`` passes a state dict and therefore leaves
the model default of 1.0), and can override it with ``--return-scale`` to test
that hypothesis.

Usage (inside distrobox, repo root):
    python3 scripts/verify_jtsoc_impact_sign.py --scenarios sa1_oct_2024
    python3 scripts/verify_jtsoc_impact_sign.py            # all 3 scenarios
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from phase3_impact_eval import BATTERIES, SCENARIOS, build_market_data, fetch_scenario  # noqa: E402
from AEMOBatteryEnv import AEMOBatteryTradingEnv  # noqa: E402
from decision import AEMOAgent  # noqa: E402
from decision_transformer import DecisionTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "models" / "aemo" / "dt" / "aemo_dt_sdp_jtsoc_fullcorpus.pt"
DEFAULT_CFG = ROOT / "configs" / "aemo_decision_transformer_model_kwargs_sdp_jtsoc_fullcorpus.json"
SCENARIO_CACHE = Path("/tmp/scenario_cache")

MODEL_INIT_KEYS = {
    "state_dim", "act_dim", "n_block", "h_dim", "context_len", "n_heads",
    "drop_p", "max_timestep", "rope_enabled", "rope_max_position", "rope_base",
    "n_kv_heads", "qk_norm", "tie_weights", "action_head_mode",
}


def load_stage_c(checkpoint: str, config: str, device: str, return_scale=None):
    kwargs = json.loads(Path(config).read_text())
    init = {k: kwargs[k] for k in MODEL_INIT_KEYS if k in kwargs}
    model = DecisionTransformer(**init)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_from_checkpoint(state)

    meta_path = Path(str(checkpoint) + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        model.return_scale = float(meta.get("return_scale", model.return_scale))
    if return_scale is not None:
        model.return_scale = float(return_scale)

    model.to(device).eval()
    return model


def get_scenario(label: str, region: str, start_str: str, end_str: str):
    SCENARIO_CACHE.mkdir(exist_ok=True)
    cache_file = SCENARIO_CACHE / f"{label}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    print(f"  building scenario {label} ({region}) ...", flush=True)
    processed = fetch_scenario(region, start, end)
    curves, depth = build_market_data(region, start, end, processed)
    entry = dict(processed=processed, curves=curves, depth=depth,
                 region=region, start=start, end=end)
    with open(cache_file, "wb") as f:
        pickle.dump(entry, f)
    return entry


def run_dt(entry, battery, model, rtg_mode: str, rtg_value: float):
    processed = entry["processed"]
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
        impact_model="piecewise_merit_order",
        impact_intensity=1.0,
        supply_curves=entry["curves"],
        fcas_depth=entry["depth"],
    )
    agent = AEMOAgent(env, algorithm="dt", model=model,
                      rtg_value=rtg_value, rtg_mode=rtg_mode)
    episode, _ = agent.run_episode()
    infos = episode["info"].to_list()
    energy = sum(i.get("energy_revenue", 0.0) for i in infos)
    fcas = sum(i.get("fcas_revenue", 0.0) for i in infos)
    deg = sum(i.get("degradation_cost", 0.0) for i in infos)
    return dict(energy=energy, fcas=fcas, deg=deg, profit=energy + fcas - deg,
                steps=len(infos))


def main():
    ap = argparse.ArgumentParser(description="Focused B1 j_t_soc impact verification")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="Subset of scenario labels (default: all).")
    ap.add_argument("--rtg-constant", type=float, default=0.0)
    ap.add_argument("--return-scale", type=float, default=None,
                    help="Override return_scale (default: checkpoint .meta.json).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "eval_output" / "phase3_impact"
                                          / "jtsoc_sign_verification.json"))
    args = ap.parse_args()

    model = load_stage_c(args.checkpoint, args.config, args.device, args.return_scale)
    print(f"Stage C loaded: return_scale={model.return_scale} device={args.device}")

    scenarios = [s for s in SCENARIOS
                 if args.scenarios is None or s[0] in set(args.scenarios)]
    if not scenarios:
        raise SystemExit(f"No scenarios matched {args.scenarios}")

    results = []
    for label, region, start_str, end_str in scenarios:
        print(f"\n=== {label} ({region}) ===", flush=True)
        entry = get_scenario(label, region, start_str, end_str)
        for battery in BATTERIES:
            print(f"  -- battery {battery['name']} --", flush=True)
            for mode, value in [("j_t_soc", 0.0), ("constant", args.rtg_constant)]:
                t0 = time.perf_counter()
                r = run_dt(entry, battery, model, mode, value)
                r.update(scenario=label, battery=battery["name"], mode=mode,
                         rtg_value=value, return_scale=float(model.return_scale),
                         time_s=round(time.perf_counter() - t0, 1))
                results.append(r)
                print(f"    {mode:<9} profit=${r['profit']:>11,.0f}  "
                      f"E=${r['energy']:>10,.0f}  F=${r['fcas']:>10,.0f}  "
                      f"deg=${r['deg']:>8,.0f}  {r['time_s']}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} rows -> {out_path}")


if __name__ == "__main__":
    main()
