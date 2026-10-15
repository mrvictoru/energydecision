"""Create Stage-A teacher configs (dt_soc_oracle/sdp) for the extended surfaces.

Copies each extended student config and swaps the `candidate_dt` policy for the
hierarchical teacher (waypoint DT + honest SDP executor) with the corpus's wear
calibration (deg_cost_per_mwh=50, deg_calibration=0.12). The reference cache dir
is kept so PPO baselines are identical to the student run.
"""
import json
from pathlib import Path

CFG = Path("configs")
PAIRS = {
    "sdp_teacher_standard_year": "soc_sdp_standard_year",
    "sdp_teacher_dispatch_year": "soc_sdp_dispatch_year",
    "sdp_teacher_expanded_full": "soc_sdp_expanded_full",
    "sdp_teacher_2025_fresh": "soc_sdp_2025_fresh",
}

for src, dst in PAIRS.items():
    d = json.loads((CFG / f"aemo_autoresearch_evaluator.{src}.json").read_text())
    d["policies"] = [
        {
            "name": "candidate_dt",
            "kind": "dt_soc_oracle",
            "executor": "sdp",
            "rtg_value": 0.0,
            "deg_cost_per_mwh": 50.0,
            "deg_calibration": 0.12,
        }
    ] + [p for p in d["policies"] if p.get("name") != "candidate_dt"]
    (CFG / f"aemo_autoresearch_evaluator.{dst}.json").write_text(json.dumps(d, indent=2) + "\n")
    print("wrote", dst, "scenarios:", len(d["heldout"]["scenarios"]))
