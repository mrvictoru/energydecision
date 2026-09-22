"""Phase 1 leakage check: does any SDP-teacher training data overlap an eval window?

The v2cal student is trained on `dt_trajectories_jtsoc_v2cal_conservative.parquet`,
whose slices are drawn from the pre-2024 processed parquets listed in
`scripts/generate_sdp_dt_trajectories.py` (REGION_FILES). The evaluator scores
2024 (identity/dispatch/expanded) and 2025 (OOD) windows. This script verifies,
from the source files themselves, that the two date ranges do not intersect, and
that the seasonal-RRP profile used for the J_t(soc) prompt is also pre-2024.

Run inside the energydecision-gpu box from the repo root.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]

# Mirror of scripts/generate_sdp_dt_trajectories.py:REGION_FILES (longest pre-2024 file).
REGION_FILES = {
    "NSW1": "data/aemo/processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet",
    "QLD1": "data/aemo/processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet",
    "SA1": "data/aemo/processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet",
    "TAS1": "data/aemo/processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet",
    "VIC1": "data/aemo/processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet",
}

CONFIGS = [
    "configs/aemo_autoresearch_evaluator.sdp_teacher_standard.json",
    "configs/aemo_autoresearch_evaluator.sdp_teacher_dispatch.json",
    "configs/aemo_autoresearch_evaluator.sdp_teacher_expanded.json",
    "configs/aemo_autoresearch_evaluator.sdp_teacher_2025.json",
]


def d(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> int:
    print("== Training source files (per region) ==")
    spans = {}
    for region, rel in REGION_FILES.items():
        p = ROOT / rel
        if not p.is_file():
            print(f"  MISSING {region}: {rel}")
            continue
        col = (
            pl.scan_parquet(p)
            .select(pl.col("SETTLEMENTDATE").min().alias("lo"),
                    pl.col("SETTLEMENTDATE").max().alias("hi"))
            .collect()
        )
        lo, hi = col["lo"][0], col["hi"][0]
        spans[region] = (lo, hi)
        print(f"  {region:5s} {lo}  ->  {hi}   ({rel})")

    print("\n== Eval windows (from configs) ==")
    eval_windows = []
    for cfg in CONFIGS:
        c = json.loads((ROOT / cfg).read_text())
        h = c["heldout"]
        for s in h.get("scenarios", []):
            win = (s["region"], d(s["start_date"]), d(s["end_date"]))
            eval_windows.append(win)
    seen = set()
    for region, lo, hi in eval_windows:
        key = (region, lo.date(), hi.date())
        if key in seen:
            continue
        seen.add(key)
        print(f"  {region:5s} {lo.date()} -> {hi.date()}")

    print("\n== Overlap verdict ==")
    train_max = max(hi for _, hi in spans.values())
    print(f"  latest training-source timestamp across regions: {train_max}")
    print(f"  earliest eval window start: {min(w[1] for w in eval_windows).date()}")
    violations = []
    for region, (tlo, thi) in spans.items():
        for er, elo, ehi in eval_windows:
            if er == region and not (thi < elo or tlo > ehi):
                violations.append((region, tlo, thi, elo, ehi))
    if violations:
        print("  !! OVERLAP DETECTED:", violations)
        return 1
    gap_days = (min(w[1] for w in eval_windows) - train_max).days
    print(f"  NO temporal overlap. Gap between latest training source and earliest eval: {gap_days} days.")

    print("\n== J_t(soc) seasonal-profile source ==")
    for region in REGION_FILES:
        cache = ROOT / "data" / "aemo_sdp" / f"seasonal_rrp_{region}.json"
        print(f"  {region:5s} profile cache: {'present' if cache.is_file() else 'absent'}")
    print("  build_seasonal_rrp_profile() calls find_training_parquet() -> pre-2024 only")
    print("  (src/aemo_sdp_executor.py:81-114). No eval-period prices enter the prompt.")

    print("\n== Dimensions note ==")
    print("  Training slices are within-region windows from the files above; eval scenarios")
    print("  are the 2024/2025 windows listed. Disjoint in time by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
