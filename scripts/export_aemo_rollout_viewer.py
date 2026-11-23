#!/usr/bin/env python3
"""Export the website's six-day traces from saved v2cal dispatch-matched rollouts.

Run from the repository root with ``python3 scripts/export_aemo_rollout_viewer.py``.
This reads existing logs and metrics; it does not run the simulator.
"""

import csv
import json
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "eval_output/physics_v2/dispatch_year_v2cal"
CONFIG = ROOT / "configs/aemo_autoresearch_evaluator.sdp_teacher_dispatch_year.json"
OUT = ROOT / "website-assets/dispatch-v2cal.json"
POLICIES = {
    "dt": "candidate_dt",
    "ppo": "ppo_reference",
    "replay": "dispatch_dalrymple_north",
}
BIN_STEPS = 6  # Half-hour bins at the logged five-minute cadence.
METRICS = {
    "price": ("energy_price", "mean"),
    "dispatch": ("battery_dispatch", "mean"),
    "soc": ("battery_soc", "last"),
    "raiseBid": ("fcas_RAISEREG_bid", "mean"),
    "lowerBid": ("fcas_LOWERREG_bid", "mean"),
    "energy": ("energy_revenue", "cumulative"),
    "fcas": ("fcas_revenue", "cumulative"),
    "wear": ("degradation_cost", "cumulative"),
}


def load_episodes(policy):
    frame = pl.read_parquet(
        RUN / "heldout_logs" / f"{policy}_heldout_logs.parquet",
        columns=["scenario_label", "heldout_episode_index", "step", "info"],
    )
    episodes = {}
    for (label,), group in frame.group_by("scenario_label"):
        group = group.sort("step")
        if group["heldout_episode_index"].unique().to_list() != [0]:
            raise ValueError(f"Expected a single fixed start for {policy}/{label}")
        if group.height != 1728 or group["step"].to_list() != list(range(group.height)):
            raise ValueError(f"Incomplete six-day episode for {policy}/{label}")
        info = group["info"]
        episodes[label] = info.struct.unnest() if info.dtype == pl.Struct else pl.DataFrame(
            [json.loads(row) for row in info]
        )
    return episodes


def bin_trace(values, reducer):
    if not np.isfinite(values).all():
        raise ValueError("Trace contains non-finite values")
    if reducer == "cumulative":
        result = np.cumsum(values)[BIN_STEPS - 1 :: BIN_STEPS]
    elif reducer == "last":
        result = values[BIN_STEPS - 1 :: BIN_STEPS]
    else:
        result = values.reshape(-1, BIN_STEPS).mean(axis=1)
    return result.round(2).tolist()


def main():
    config = json.loads(CONFIG.read_text())
    summary = json.loads((RUN / "evaluation_summary.json").read_text())
    (asset,) = summary["heldout_evaluation"]["battery_variants"]
    with (RUN / "heldout_metrics_by_scenario.csv").open(newline="") as handle:
        profits = {
            (row["scenario_label"], row["policy_name"]): float(row["avg_profit_per_episode"])
            for row in csv.DictReader(handle)
        }
    raw = {key: load_episodes(name) for key, name in POLICIES.items()}
    payload = {
        "binMinutes": BIN_STEPS * 5,
        "capacityMwh": asset["battery_capacity"],
        "powerMw": asset["max_battery_flow"],
        "episodes": [],
    }
    for scenario in config["heldout"]["scenarios"]:
        label = scenario["label"]
        episode = {"month": scenario["start_date"][:7], "policies": {}}
        price = raw["dt"][label]["energy_price"].to_numpy()
        episode["price"] = bin_trace(price, "mean")
        for key, name in POLICIES.items():
            info = raw[key][label]
            if not np.array_equal(info["energy_price"].to_numpy(), price):
                raise ValueError(f"Unmatched price path for {name}/{label}")
            trace = {
                metric: bin_trace(info[column].to_numpy(), reducer)
                for metric, (column, reducer) in METRICS.items()
                if metric != "price"
            }
            # The reported profit metric excludes constraint penalties in the RL reward.
            profit = (info["energy_revenue"] + info["fcas_revenue"] - info["degradation_cost"]).to_numpy()
            if not np.isclose(profit.sum(), profits[label, name], rtol=0, atol=1e-5):
                raise ValueError(f"Trace profit differs from saved metrics for {name}/{label}")
            trace["profit"] = bin_trace(profit, "cumulative")
            episode["policies"][key] = trace
        payload["episodes"].append(episode)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported {len(payload['episodes'])} matched episodes to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
