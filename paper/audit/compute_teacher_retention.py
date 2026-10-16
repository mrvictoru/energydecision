#!/usr/bin/env python3
"""Regenerate paper/audit/artifacts/teacher_retention.json.

The paper reports how much of its solver-in-the-loop teacher's aggregate profit
the distilled student retains on each extended identity surface. This script
recomputes that number from two canonical artifacts so the claim is traceable:

  * student / PPO means: paper/audit/artifacts/stagec_v2cal_extended_significance.json
    (surface set "v2cal_extended"; produced by paper/audit/reproduce_significance.py)
  * teacher mean profit: eval_output/physics_v2/teacher_best_<surface>/evaluation_summary.json
    (produced by paper/audit/run_teacher_eval.sh, i.e. the Stage-A waypoint DT
    + honest SDP executor at deg_calibration 0.12)

Retention = student / teacher. teacher_vs_ppo is reported for context only and
is not used in the manuscript.

Usage:
    python3 paper/audit/compute_teacher_retention.py            # write artifact
    python3 paper/audit/compute_teacher_retention.py --check    # verify only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIG = ROOT / "paper/audit/artifacts/stagec_v2cal_extended_significance.json"
OUT = ROOT / "paper/audit/artifacts/teacher_retention.json"

# label -> (significance surface key, teacher eval output dir)
SURFACES = {
    "standard 2024": ("standard_year_2024", "teacher_best_standard_year"),
    "dispatch-matched": ("dispatch_year_2024", "teacher_best_dispatch_year"),
    "expanded broad-2024": ("expanded_broad_2024", "teacher_best_expanded_full"),
    "2025 OOD": ("2025_ood", "teacher_best_2025_fresh"),
}

TEACHER_CKPT = "soc_waypoint_dt_best.pt"
DEG_CALIBRATION = 0.12


def _teacher_mean(out_dir: str) -> float:
    path = ROOT / f"eval_output/physics_v2/{out_dir}/evaluation_summary.json"
    summary = json.loads(path.read_text())
    return float(summary["heldout_evaluation"]["aggregate_metrics"][0]["avg_profit_per_episode"])


def build() -> dict:
    sig = json.loads(SIG.read_text())
    by_key = {s["surface"]: s for s in sig["surface_sets"]["v2cal_extended"]}

    out: dict = {}
    for label, (sig_key, teacher_dir) in SURFACES.items():
        s = by_key[sig_key]
        student = float(s["dt_mean"])
        ppo = float(s["ppo_mean"])
        teacher = _teacher_mean(teacher_dir)
        out[label] = {
            "student": student,
            "teacher": teacher,
            "ppo": ppo,
            "retention": student / teacher,
            "teacher_vs_ppo": teacher / ppo,
            "teacher_ckpt": TEACHER_CKPT,
            "deg_calibration": DEG_CALIBRATION,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare against the stored artifact; do not write")
    args = ap.parse_args()

    computed = build()

    if args.check:
        stored = json.loads(OUT.read_text())
        ok = True
        for label in SURFACES:
            r_new = computed[label]["retention"]
            r_old = stored[label]["retention"]
            flag = "OK " if abs(r_new - r_old) < 1e-9 else "DIFF"
            if flag == "DIFF":
                ok = False
            print(f"{flag} {label:22s} retention {r_new:.6f} (stored {r_old:.6f})")
        print("retention matches stored artifact" if ok else "retention MISMATCH")
        return 0 if ok else 1

    OUT.write_text(json.dumps(computed, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for label, rec in computed.items():
        print(f"  {label:22s} student {rec['student']:.2f}  teacher {rec['teacher']:.2f}  "
              f"retention {rec['retention']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
