"""Generate the extended (small-n fix) evaluator configs from the canonical ones.

Creates three new configs that leave the frozen `preprint-v1` configs untouched:
  - standard_year : 5 regions x 6 periods (Jan/Mar/May/Jul/Sep/Nov 2024), medium_1c  -> n=30
  - dispatch_year : SA1 x 12 months 2024, dispatch_asset_template                     -> n=12
  - expanded_full : expanded 27 + TAS1 Jul/Sep/Nov 2024, medium                       -> n=30

Each uses a fresh reference_cache_dir so PPO rollouts for the new scenarios are
computed (and cached) rather than risking a key collision with the canonical runs.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs"

REGIONS = ["NSW1", "QLD1", "SA1", "VIC1", "TAS1"]
PERIODS = [("jan", "01"), ("mar", "03"), ("may", "05"), ("jul", "07"), ("sep", "09"), ("nov", "11")]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def load(name: str) -> dict:
    return json.loads((CFG / f"aemo_autoresearch_evaluator.{name}.json").read_text())


def dump(name: str, cfg: dict) -> None:
    out = CFG / f"aemo_autoresearch_evaluator.{name}.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    n = len(cfg["heldout"]["scenarios"])
    print(f"wrote {out.name}: {n} scenarios, batteries={[b['name'] for b in cfg['heldout']['battery_variants']]}")


def main() -> None:
    # --- standard_year: 5 regions x 6 periods, 144h, medium_1c ---
    cfg = load("sdp_teacher_standard")
    cfg["heldout"]["scenarios"] = [
        {"label": f"{r.lower()}_{m}_2024", "region": r,
         "start_date": f"2024-{mm}-01", "end_date": f"2024-{mm}-14"}
        for r in REGIONS for m, mm in PERIODS
    ]
    cfg["reference_cache_dir"] = "eval_output/reference_cache/tier_standard_year"
    dump("sdp_teacher_standard_year", cfg)

    # --- dispatch_year: SA1 x 12 months, 144h ---
    cfg = load("sdp_teacher_dispatch")
    cfg["heldout"]["scenarios"] = [
        {"label": f"sa1_{m}_2024", "region": "SA1",
         "start_date": f"2024-{i+1:02d}-01", "end_date": f"2024-{i+1:02d}-14"}
        for i, m in enumerate(MONTHS)
    ]
    cfg["reference_cache_dir"] = "eval_output/reference_cache/tier_dispatch_year"
    dump("sdp_teacher_dispatch_year", cfg)

    # --- expanded_full: canonical 27 + TAS1 Jul/Sep/Nov ---
    cfg = load("sdp_teacher_expanded")
    existing = {s["label"] for s in cfg["heldout"]["scenarios"]}
    for m, mm in [("jul", "07"), ("sep", "09"), ("nov", "11")]:
        lab = f"tas1_{m}_2024"
        if lab not in existing:
            cfg["heldout"]["scenarios"].append(
                {"label": lab, "region": "TAS1",
                 "start_date": f"2024-{mm}-01", "end_date": f"2024-{mm}-14"})
    cfg["reference_cache_dir"] = "eval_output/reference_cache/tier_expanded_full"
    dump("sdp_teacher_expanded_full", cfg)


if __name__ == "__main__":
    main()
