#!/usr/bin/env python3
"""Dump extra AEMO behaviour-chart series from Stage C held-out logs.

Reads the same parquet used to embed BEHAVIOR / BEHAVIOR_CUM in index.html and
emits raise/lower FCAS bid fractions plus energy/FCAS cumulative dollars,
binned to match BEHAVIOR_META (30-min / 1 h on 2024, 5 h / 10 h on 2025).

Usage (from the repository root):
    python3 scripts/dump_aemo_behavior.py \\
        --output eval_output/website/aemo_behavior_extras.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SURFACES = {
    "y2024": {
        "log_dir": ROOT / "eval_output/stagec_jtsoc_dispatch_rtgjtsoc/heldout_logs",
        "policies": {
            "dt": "candidate_dt_heldout_logs.parquet",
            "ppo": "ppo_reference_heldout_logs.parquet",
            "replay": "dispatch_dalrymple_north_heldout_logs.parquet",
        },
        "episodes": [
            "sa1_jul_2024",
            "sa1_aug_2024",
            "sa1_sep_2024",
            "sa1_oct_2024",
            "sa1_nov_2024",
            "sa1_dec_2024",
        ],
        "bin_steps": 6,  # 30-min means of 5-min steps
        "cum_bin_steps": 12,  # 1 h last-of-bin of the 5-min cumulative
    },
    "y2025": {
        "log_dir": ROOT / "eval_output/stagec_jtsoc_2025_rtgjtsoc/heldout_logs",
        "policies": {
            "dt": "candidate_dt_heldout_logs.parquet",
            "ppo": "ppo_reference_heldout_logs.parquet",
        },
        "episodes": [
            "nsw1_01_2025",
            "nsw1_02_2025",
            "sa1_01_2025",
            "sa1_02_2025",
            "qld1_01_2025",
            "qld1_02_2025",
        ],
        "bin_steps": 60,  # 5 h
        "cum_bin_steps": 120,  # 10 h
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval_output/website/aemo_behavior_extras.json",
    )
    return parser.parse_args()


def unnest_info(frame: pl.DataFrame) -> pl.DataFrame:
    col = frame["info"]
    if col.dtype == pl.Struct:
        return col.struct.unnest()
    rows = []
    for raw in col.to_list():
        if isinstance(raw, str):
            rows.append(json.loads(raw))
        elif isinstance(raw, dict):
            rows.append(raw)
        else:
            raise TypeError(f"unsupported info cell type: {type(raw)!r}")
    return pl.DataFrame(rows)


def series(info: pl.DataFrame, *names: str) -> np.ndarray:
    for name in names:
        if name in info.columns:
            return np.asarray(info[name].to_numpy(), dtype=np.float64)
    raise KeyError(f"none of {names} present in info columns {info.columns}")


def bin_mean(values: np.ndarray, bin_steps: int) -> list[float]:
    n = (len(values) // bin_steps) * bin_steps
    binned = values[:n].reshape(-1, bin_steps).mean(axis=1)
    return [round(float(x), 2) for x in binned]


def bin_cum_last(values: np.ndarray, bin_steps: int) -> list[float]:
    n = (len(values) // bin_steps) * bin_steps
    last = np.cumsum(values[:n]).reshape(-1, bin_steps)[:, -1]
    return [round(float(x), 1) for x in last]


def load_policy(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except pl.exceptions.SchemaError:
        return pl.read_parquet(
            path,
            schema_overrides={"action": pl.List(pl.Float64)},
        )


def episode_payload(frame: pl.DataFrame, episode: str, bin_steps: int, cum_bin_steps: int) -> dict:
    ep = frame.filter(pl.col("scenario_label") == episode).sort("step")
    if ep.height == 0:
        raise SystemExit(f"no rows for scenario {episode} in log")
    info = unnest_info(ep)
    raise_bid = series(info, "fcas_raise_bid", "fcas_RAISEREG_bid")
    lower_bid = series(info, "fcas_lower_bid", "fcas_LOWERREG_bid")
    energy = series(info, "energy_revenue")
    fcas = series(info, "fcas_revenue")
    deg = series(info, "degradation_cost")
    profit = energy + fcas - deg
    return {
        "raise": bin_mean(raise_bid, bin_steps),
        "lower": bin_mean(lower_bid, bin_steps),
        "energy": bin_cum_last(energy, cum_bin_steps),
        "fcas": bin_cum_last(fcas, cum_bin_steps),
        "split": {
            "profit": round(float(profit.sum()), 1),
            "energy": round(float(energy.sum()), 1),
            "fcas": round(float(fcas.sum()), 1),
            "deg": round(float(deg.sum()), 1),
        },
    }


def main() -> int:
    args = parse_args()
    out = {"fcas": {}, "cum": {}, "split": {}}
    for surface, spec in SURFACES.items():
        out["fcas"][surface] = {}
        out["cum"][surface] = {}
        out["split"][surface] = {}
        frames = {}
        for policy, filename in spec["policies"].items():
            path = spec["log_dir"] / filename
            if not path.exists():
                raise SystemExit(f"missing log {path}")
            frames[policy] = load_policy(path)
        for episode in spec["episodes"]:
            out["fcas"][surface][episode] = {}
            out["cum"][surface][episode] = {}
            out["split"][surface][episode] = {}
            for policy, frame in frames.items():
                payload = episode_payload(frame, episode, spec["bin_steps"], spec["cum_bin_steps"])
                out["fcas"][surface][episode][policy] = {
                    "raise": payload["raise"],
                    "lower": payload["lower"],
                }
                out["cum"][surface][episode][policy] = {
                    "energy": payload["energy"],
                    "fcas": payload["fcas"],
                }
                out["split"][surface][episode][policy] = payload["split"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
