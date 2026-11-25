"""Evaluate the shipped Decision Transformer on the PPO-retrain validation set.

Used only to interpret the PPO learning curve (how much of the DT's
in-distribution profit a trained PPO reaches). The paper's claims are tested on
the 2024/2025 held-out identity surfaces, not here.

Usage:
    python3 scripts/ppo_retrain_dt_validation.py \
        --out-dir models/aemo_sb3/ppo_retrain_20261125 \
        --model models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt \
        --config models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json \
        --out models/aemo_sb3/ppo_retrain_20261125/validate_dt.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rtg", type=float, default=0.0)
    ap.add_argument("--rtg-mode", default="j_t_soc", choices=["constant", "j_t_soc", "auto"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batteries", nargs="*", default=None,
                    help="battery labels to evaluate (default: all four)")
    args = ap.parse_args()

    import ppo_retrain_aemo as drv
    from aemo_notebook_utils import create_aemo_env, load_dt_model, resolve_battery_variants
    from decision import AEMOAgent

    manifest_path = Path(args.out_dir) / "data_split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    windows = manifest["validation_windows"]
    frames, _ = drv.load_region_frames()
    _train, valid_frames, _split = drv.split_frames(frames)

    model = load_dt_model(model_path=args.model, model_config_path=args.config, device=args.device)
    batteries = resolve_battery_variants(drv.BATTERIES)
    if args.batteries:
        wanted = set(args.batteries)
        batteries = [b for b in batteries if b["label"] in wanted]
        if not batteries:
            raise SystemExit(f"no battery matched {sorted(wanted)}")

    rows = []
    for window in windows:
        frame = valid_frames[window["region"]].slice(int(window["offset"]), drv.VALID_MAX_STEP)
        for battery in batteries:
            env = create_aemo_env(
                processed_data=frame,
                battery_variant=battery,
                max_step=drv.VALID_MAX_STEP,
                step_duration=drv.STEP_DURATION,
                action_mode="full_fcas",
                degradation_mode="real_world",
                degradation_chemistry="LFP",
                degradation_temperature=30.0,
                random_episode_start=False,
            )
            agent = AEMOAgent(env, algorithm="dt", model=model, rtg_value=args.rtg,
                              rtg_mode=args.rtg_mode)
            episode, _ = agent.run_episode()
            infos = episode["info"].to_list()
            energy = sum(float(i.get("energy_revenue", 0.0)) for i in infos)
            fcas = sum(float(i.get("fcas_revenue", 0.0)) for i in infos)
            deg = sum(float(i.get("degradation_cost", 0.0)) for i in infos)
            rows.append(
                {
                    "region": window["region"],
                    "window_index": window["window_index"],
                    "window_start": window["start"],
                    "battery": battery["label"],
                    "steps": len(infos),
                    "profit": energy + fcas - deg,
                    "energy_revenue": energy,
                    "fcas_revenue": fcas,
                    "degradation_cost": deg,
                }
            )

    profits = np.array([r["profit"] for r in rows], dtype=float)
    by_battery: dict[str, list[float]] = {}
    by_region: dict[str, list[float]] = {}
    for row in rows:
        by_battery.setdefault(row["battery"], []).append(row["profit"])
        by_region.setdefault(row["region"], []).append(row["profit"])
    summary = {
        "model": str(args.model),
        "rtg_value": args.rtg,
        "rtg_mode": args.rtg_mode,
        "n_episodes": len(rows),
        "mean_profit": float(profits.mean()),
        "median_profit": float(np.median(profits)),
        "std_profit": float(profits.std(ddof=1)),
        "mean_profit_by_battery": {k: float(np.mean(v)) for k, v in by_battery.items()},
        "mean_profit_by_region": {k: float(np.mean(v)) for k, v in by_region.items()},
        "episodes": rows,
    }
    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
