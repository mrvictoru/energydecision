#!/usr/bin/env python3
"""Generate SDP-teacher DT training trajectories for Stage B (standalone DT).

For each (region, horizon, battery) slot, slice the cached 2021-2023 processed
parquet into episode-length windows with random starts and replay the honest
SDP executor (AEMOAgent dt_soc_oracle executor='sdp') through the env,
capturing per-step (norm_observation, action[9], reward). These triples are
self-consistent (the reward is what the teacher earns from the action, the obs
are the env's normalized state), so a Decision Transformer retrained on them
is a standalone clone of the honest planner — no solver at inference.

Data source: data/aemo/processed_{REGION}_*.parquet (the SAME preprocessed
frames the evaluator uses at inference), so train/eval observation spaces are
identical.

Usage (pilot):
  python3 scripts/generate_sdp_dt_trajectories.py \
    --regions NSW1 QLD1 SA1 TAS1 VIC1 \
    --horizons short medium \
    --batteries medium_1c fast_375c \
    --episodes-per-slot 8 \
    --out data/aemo_dt_sdp/dt_trajectories.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_notebook_utils import create_aemo_env  # noqa: E402
from decision import AEMOAgent  # noqa: E402
from decision_transformer import DecisionTransformer  # noqa: E402

BATTERY_SPECS = {
    "medium_1c": {"capacity": 10.0, "max_flow": 10.0},
    "fast_375c": {"capacity": 8.0, "max_flow": 30.0},
    "large_07c": {"capacity": 50.0, "max_flow": 35.0},
    "small_05c": {"capacity": 2.0, "max_flow": 1.0},
}
# 5-min steps per episode by horizon (short=14d, medium=8wk)
HORIZON_STEPS = {"short": 3456, "medium": 16128, "long": 74880}

# Region -> longest pre-2024 processed parquet
REGION_FILES = {
    "NSW1": "data/aemo/processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet",
    "QLD1": "data/aemo/processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet",
    "SA1": "data/aemo/processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet",
    "TAS1": "data/aemo/processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet",
    "VIC1": "data/aemo/processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet",
}


def _dummy_env(cap: float, max_flow: float, n_steps: int):
    """Standalone env shim exposing only what the SDP cost-to-go needs."""
    from types import SimpleNamespace
    return SimpleNamespace(
        battery_capacity=cap, max_battery_flow=max_flow,
        step_duration=5.0 / 60.0, max_grid_energy=float("inf"), df=None,
        battery_life_cost=1312500.0,
        degradation_temperature=25.0,
    )


def _attach_jt_rtg(
    episode_df: pl.DataFrame,
    ctg: np.ndarray,
    soc_levels: np.ndarray,
    cap: float,
) -> pl.DataFrame:
    """Add an ``rtg_value`` column = -J_t(soc) for each step's SOC.

    SOC is recovered from the norm_observation index 17 (soc/capacity), the
    same indirection the DT rollout uses.
    """
    soc_list = []
    for obs in episode_df["norm_observation"].to_list():
        arr = np.asarray(obs, dtype=float)
        soc_kwh = float(arr[17]) * cap if arr.size > 17 else float(cap * 0.5)
        soc_list.append(soc_kwh)
    soc_arr = np.asarray(soc_list, dtype=float)
    s_idx = np.argmin(np.abs(soc_levels[:, None] - soc_arr[None, :]), axis=0)
    t_idx = np.arange(len(soc_arr), dtype=int)
    rtg = -ctg[t_idx, s_idx]
    return episode_df.with_columns(pl.Series("rtg_value", rtg.astype(np.float64)))


def _build_slot_ctg(sliced: pl.DataFrame, region: str, cap: float, max_flow: float,
                    n_steps: int, deg_cost_per_mwh: float):
    """Build the J_t(soc) table for THIS episode's actual time window.

    The SDP RRP forecast is a function of (month, hour), so the value table
    depends on the specific dates in ``sliced``. Building it from the episode's
    own window (rather than a slot-level first-window) keeps the stored RTG
    consistent with both the SDP executor's realised actions and the inference-
    time table, which is also built from the episode's aemo_data.
    """
    from aemo_sdp_executor import (
        build_rrp_forecast, build_seasonal_rrp_profile, compute_cost_to_go_table,
    )
    cache_dir = Path("data/aemo")
    try:
        profile = build_seasonal_rrp_profile(cache_dir, region, 5.0 / 60.0)
    except Exception as e:
        print(f"  [j_t_soc] profile failed {region}: {e}")
        profile = None
    if profile is None:
        print(f"  [j_t_soc] NO profile for {region}; returning None ctg")
        return None, None
    try:
        fc = build_rrp_forecast(sliced, profile)
        denv = _dummy_env(cap, max_flow, n_steps)
        return compute_cost_to_go_table(denv, fc, deg_cost_per_mwh=deg_cost_per_mwh)
    except Exception as e:
        print(f"  [j_t_soc] ctg failed {region} cap={cap} n={n_steps}: {e}")
        return None, None


def generate_slot(
    processed: pl.DataFrame,
    region: str,
    horizon: str,
    battery_name: str,
    n_eps: int,
    rng: np.random.Generator,
    model: DecisionTransformer,
    deg_cost_per_mwh: float,
    rtg_value: float,
    seed_base: int,
    rtg_mode: str = "constant",
) -> list[pl.DataFrame]:
    cap = BATTERY_SPECS[battery_name]["capacity"]
    max_flow = BATTERY_SPECS[battery_name]["max_flow"]
    n_steps = HORIZON_STEPS[horizon]

    max_start = processed.height - n_steps
    frames = []
    for ep in range(n_eps):
        if max_start <= 0:
            break
        start = int(rng.integers(0, max_start))
        sliced = processed.slice(start, n_steps)
        env = create_aemo_env(
            processed_data=sliced,
            battery_variant={"battery_capacity": cap, "max_battery_flow": max_flow,
                             "init_soc": cap * 0.5, "battery_life_cost": 1312500.0},
            max_step=n_steps, step_duration=5.0 / 60.0, action_mode="full_fcas",
            random_episode_start=False,
        )
        agent = AEMOAgent(env, algorithm="dt_soc_oracle", model=model,
                          rtg_value=rtg_value, executor="sdp",
                          deg_cost_per_mwh=deg_cost_per_mwh,
                          reset_seed=seed_base + ep)
        episode_df, _ = agent.run_episode()
        episode_df = episode_df.with_columns(
            pl.lit(f"{region}__{horizon}__{battery_name}__ep{ep:03d}").alias("episode_id"),
            pl.lit("sdp_teacher").alias("source_policy"),
            pl.lit(region).alias("region"),
            pl.lit(battery_name).alias("battery"),
        )
        if rtg_mode == "j_t_soc":
            ctg, soc_levels = _build_slot_ctg(sliced, region, cap, max_flow,
                                              n_steps, deg_cost_per_mwh)
            if ctg is not None and soc_levels is not None:
                episode_df = _attach_jt_rtg(episode_df, ctg, soc_levels, cap)
            else:
                # Table unavailable (e.g. seasonal profile failure): fall back to a
                # constant RTG so the output schema stays consistent (all episodes
                # carry an rtg_value column).
                episode_df = episode_df.with_columns(
                    pl.lit(float(rtg_value)).alias("rtg_value"))
        frames.append(episode_df)
    return frames


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", nargs="+", default=["NSW1", "QLD1", "SA1", "TAS1", "VIC1"])
    p.add_argument("--horizons", nargs="+", default=["short", "medium"])
    p.add_argument("--batteries", nargs="+",
                   default=["medium_1c", "fast_375c", "large_07c", "small_05c"])
    p.add_argument("--episodes-per-slot", type=int, default=8)
    p.add_argument("--model-manifest", type=Path,
                   default=Path("models/aemo/dt/soc_waypoint_dt_loss_surface_manifest.json"))
    p.add_argument("--model-path", type=Path,
                   default=Path("models/aemo/dt/soc_waypoint_dt_best.pt"))
    p.add_argument("--deg-cost-per-mwh", type=float, default=50.0)
    p.add_argument("--rtg-value", type=float, default=0.0)
    p.add_argument("--rtg-mode", type=str, default="constant",
                   choices=["constant", "j_t_soc"],
                   help="RTG source: 'constant' (scalar, Stage B) or 'j_t_soc' "
                        "(SDP cost-to-go, Stage C). j_t_soc stores a per-step "
                        "rtg_value column = -J_t(soc).")
    p.add_argument("--out", type=Path, default=Path("data/aemo_dt_sdp/dt_trajectories.parquet"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    import json
    manifest = json.loads(args.model_manifest.read_text())
    model = DecisionTransformer(**manifest["model_kwargs"])
    model.load_from_checkpoint(str(args.model_path), map_location="cpu")
    model.eval()

    rng = np.random.default_rng(args.seed)
    all_frames: list[pl.DataFrame] = []
    for region in args.regions:
        src = REGION_FILES.get(region)
        if src is None or not Path(src).exists():
            print(f"[stage_b] skip {region}: no training parquet")
            continue
        processed = pl.read_parquet(src)
        print(f"[stage_b] {region}: {processed.height} rows", flush=True)
        for horizon in args.horizons:
            for battery in args.batteries:
                frames = generate_slot(
                    processed, region, horizon, battery, args.episodes_per_slot,
                    rng, model, args.deg_cost_per_mwh, args.rtg_value, args.seed,
                    rtg_mode=args.rtg_mode,
                )
                n_rows = sum(f.height for f in frames)
                print(f"  {horizon}/{battery}: {len(frames)} eps, {n_rows} rows", flush=True)
                all_frames.extend(frames)

    combined = pl.concat(all_frames) if all_frames else pl.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(args.out)
    print(f"\nWrote {args.out} ({combined.height} rows, "
          f"{combined['episode_id'].n_unique()} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
