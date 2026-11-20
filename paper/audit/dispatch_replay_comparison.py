#!/usr/bin/env python3
"""Reproduce the dispatch-replay comparison quoted in Section 5.1.

Compares the shipped v2cal Decision Transformer against the Dalrymple North
dispatch replay on the 12 matched dispatch-year cells. The bootstrap follows
``paper/audit/reproduce_significance.py`` (percentile interval, seed 42,
10,000 resamples) so the reported interval is reproducible.

Writes ``paper/audit/artifacts/dispatch_replay_comparison.json``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "eval_output/physics_v2/dispatch_year_v2cal/heldout_metrics_by_scenario.csv"
OUT = ROOT / "paper/audit/artifacts/dispatch_replay_comparison.json"

N_BOOT = 10_000
SEED = 42
CONF = 0.95
DT = "candidate_dt"
REPLAY = "dispatch_dalrymple_north"


def main() -> None:
    cells: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    with SOURCE.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_label"], row["battery_label"])
            cells[key][row["policy_name"]] = float(row["avg_profit_per_episode"])

    keys = sorted(cells)
    dt = np.array([cells[k][DT] for k in keys])
    replay = np.array([cells[k][REPLAY] for k in keys])
    diffs = dt - replay

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    means = diffs[idx].mean(axis=1)
    alpha = (1.0 - CONF) / 2.0
    ci = [float(np.percentile(means, 100 * alpha)),
          float(np.percentile(means, 100 * (1 - alpha)))]
    _, p_value = stats.wilcoxon(diffs)

    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "policies": {"dt": DT, "replay": REPLAY},
        "n": len(diffs),
        "dt_mean": float(dt.mean()),
        "replay_mean": float(replay.mean()),
        "diff_mean": float(diffs.mean()),
        "diff_ci": ci,
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "win_count": int((diffs > 0).sum()),
        "win_rate": float((diffs > 0).mean()),
        "wilcoxon_p": float(p_value),
        "method": {
            "bootstrap": "percentile over matched cells",
            "n_boot": N_BOOT,
            "seed": SEED,
            "confidence": CONF,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
