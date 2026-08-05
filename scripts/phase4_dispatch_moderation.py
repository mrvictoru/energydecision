"""
Phase 4 follow-up: dispatch-moderation analysis.

Does the impact-aware DT bid less aggressively than the price-taking v2 DT
under market impact? Compares per-interval action magnitudes:

  - energy dispatch: |action[:, 0]| (0 = no charge/discharge, 1 = max)
  - FCAS bids:      sum(action[:, 1:]) (0 = no bids, 8 = max total)
  - fraction of steps with any dispatch / any FCAS bid

Runs both models on the SAME scenario/battery/impact setting (best RTG for
each model from results.json), extracts per-step actions from run_episode.
"""

import json
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from AEMOBatteryEnv import AEMOBatteryTradingEnv
from decision import AEMOAgent
from decision_transformer import DecisionTransformer
from aemo_dt_hf import load_model_kwargs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "eval_output" / "phase3_impact" / "results.json").read_text())

SCENARIO = "sa1_oct_2024"
CACHE_PKL = Path("/tmp/scenario_cache") / f"{SCENARIO}.pkl"

CHECKPOINTS = {
    "dt_impact": ROOT / "models" / "aemo" / "dt" / "hf_v2_impact" / "aemo_dt_fcas_model.pt",
    "dt_v2": ROOT / "models" / "aemo" / "dt" / "hf_v2_modern" / "aemo_dt_fcas_model.pt",
}
CONFIG = ROOT / "configs" / "aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json"

BATTERIES = [
    dict(name="small", capacity=8.0, max_flow=30.0, step_h=0.08333, init_soc=4.0),
    dict(name="hornsdale", capacity=194.0, max_flow=150.0, step_h=0.08333, init_soc=97.0),
    dict(name="torrens", capacity=250.0, max_flow=250.0, step_h=0.08333, init_soc=125.0),
]


def parse(lbl):
    m = re.match(r"(identity|piecewise_merit_order)_(.+?)_(small|hornsdale|torrens)_(.+)$", lbl)
    if not m:
        return None
    impact, pol, batt, scen = m.groups()
    return dict(impact=impact, pol=pol, batt=batt, scen=scen)


def best_rtg(model_key, batt, impact="piecewise_merit_order"):
    """Best RTG for a model on this scenario/battery under impact."""
    cands = []
    for r in RESULTS:
        p = parse(r["label"])
        if (p and p["impact"] == impact and p["pol"].startswith(model_key)
                and p["batt"] == batt and p["scen"] == SCENARIO):
            cands.append(r)
    if not cands:
        return None
    return max(cands, key=lambda x: x["profit"])


def load_model(model_key):
    init_keys = {"state_dim", "act_dim", "n_block", "h_dim", "context_len", "n_heads",
                 "drop_p", "max_timestep", "n_kv_heads", "qk_norm", "tie_weights"}
    kwargs = {k: v for k, v in load_model_kwargs(CONFIG).items() if k in init_keys}
    model = DecisionTransformer(**kwargs)
    state = torch.load(str(CHECKPOINTS[model_key]), map_location="cpu", weights_only=False)
    model.load_from_checkpoint(state)
    model.eval()
    return model


def action_stats(actions):
    """actions: np.ndarray [T, 9] -> moderation metrics."""
    a = np.asarray(actions, dtype=np.float32)
    energy = np.abs(a[:, 0])
    fcas = np.clip(a[:, 1:], 0.0, 1.0).sum(axis=1)
    return {
        "mean_abs_energy": float(energy.mean()),
        "median_abs_energy": float(np.median(energy)),
        "frac_energy_nonzero": float((energy > 1e-3).mean()),
        "mean_fcas_sum": float(fcas.mean()),
        "median_fcas_sum": float(np.median(fcas)),
        "frac_fcas_bid": float((fcas > 1e-3).mean()),
    }


def main():
    data = pickle.load(open(str(CACHE_PKL), "rb"))
    processed, curves, depth = data["processed"], data["curves"], data["depth"]
    max_step = processed.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {k: load_model(k).to(device) for k in CHECKPOINTS}

    print(f"Dispatch-moderation analysis — {SCENARIO}, impact=piecewise_merit_order\n")
    header = (f"{'battery':<10}{'model':<11}{'best_rtg':>8}{'mean|E|':>10}"
              f"{'med|E|':>9}{'%E>0':>7}{'meanFCAS':>10}{'medFCAS':>9}{'%bid':>7}")
    print(header)
    print("-" * len(header))

    summary = {}
    for battery in BATTERIES:
        bname = battery["name"]
        summary[bname] = {}
        for model_key, model in models.items():
            best = best_rtg(model_key, bname)
            rtg = float(best["label"].split("rtg")[1].split("_")[0])
            env = AEMOBatteryTradingEnv(
                aemo_data=processed, battery_capacity=battery["capacity"],
                max_battery_flow=battery["max_flow"], step_duration=battery["step_h"],
                init_battery_level=battery["init_soc"], max_step=max_step,
                action_mode="full_fcas", degradation_mode="real_world",
                degradation_chemistry="LFP", degradation_temperature=30.0,
                random_episode_start=False, impact_model="piecewise_merit_order",
                impact_intensity=1.0, supply_curves=curves, fcas_depth=depth,
            )
            agent = AEMOAgent(env, algorithm="dt", model=model, rtg_value=rtg)
            ep, _ = agent.run_episode()
            actions = np.array(ep["action"].to_list(), dtype=np.float32)
            stats = action_stats(actions)
            summary[bname][model_key] = stats
            print(f"{bname:<10}{model_key:<11}{rtg:>8.1f}"
                  f"{stats['mean_abs_energy']:>10.3f}{stats['median_abs_energy']:>9.3f}"
                  f"{stats['frac_energy_nonzero']:>7.1%}{stats['mean_fcas_sum']:>10.3f}"
                  f"{stats['median_fcas_sum']:>9.3f}{stats['frac_fcas_bid']:>7.1%}")

    print("\n── Moderation ratios (impact_dt / v2, <1 means impact-DT is more conservative) ──")
    for bname in BATTERIES:
        name = bname["name"]
        i, v = summary[name]["dt_impact"], summary[name]["dt_v2"]
        print(f"  {name:<10} mean|E| {i['mean_abs_energy']/max(v['mean_abs_energy'],1e-9):.2f}x   "
              f"meanFCAS {i['mean_fcas_sum']/max(v['mean_fcas_sum'],1e-9):.2f}x   "
              f"%bid {i['frac_fcas_bid']/max(v['frac_fcas_bid'],1e-9):.2f}x")


if __name__ == "__main__":
    main()
