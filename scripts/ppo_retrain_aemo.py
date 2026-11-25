"""Leakage-free multi-seed PPO driver for the AEMO Decision Transformer paper.

Why this script exists
----------------------
``scripts/retrain_sb3_full_fcas.py`` reproduces the *original* PPO reference:
``N_TRIALS=0``, ``default_model=True``, 200k timesteps, no seed, and an
``eval_env_fn`` built from the *same* frame used for training (no held-out
validation split). ``sorted(CACHE_DIR.glob(...))[0]`` also silently picks the
first matching processed file per region.

This driver implements the protocol used for the ``ppo_retrain_<date>``
baseline:

* explicit per-region pre-2024 processed files (path + sha256 + span recorded);
* chronological per-region split: first 80 % train, last 20 % validation,
  both strictly pre-2024 (no 2024/2025 data is read at all);
* training envs: one per (region × battery variant) = 20 envs, 1-week episodes,
  random episode starts inside the train slice only;
* validation envs: fixed 144 h windows (2 per region, pre-specified offsets) on
  the validation slice, deterministic policy, raw profit read from ``info``
  (``total_revenue - total_degradation_cost``), not the normalised reward;
* milestone checkpoints (0.25M/0.5M/1M/2M/4M) + validation at each milestone;
* Optuna tuning on validation only (never on the 2024/2025 test surfaces).

Everything is written under ``--out-dir``. No existing file is modified.

Typical use (from the repo root, inside the Distrobox runtime):

    python3 scripts/ppo_retrain_aemo.py manifest \
        --out-dir models/aemo_sb3/ppo_retrain_20261125
    python3 scripts/ppo_retrain_aemo.py train \
        --out-dir models/aemo_sb3/ppo_retrain_20261125 \
        --tag default_seed0 --config default --seed 0 --timesteps 4000000
    python3 scripts/ppo_retrain_aemo.py tune \
        --out-dir models/aemo_sb3/ppo_retrain_20261125 \
        --trials 10 --timesteps 400000
    python3 scripts/ppo_retrain_aemo.py train \
        --out-dir models/aemo_sb3/ppo_retrain_20261125 \
        --tag tuned_seed0 --config-json models/aemo_sb3/ppo_retrain_20261125/optuna/best_params.json \
        --seed 0 --timesteps 4000000
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Protocol constants (identical to the original PPO comparison except where
# explicitly noted in the report: per-region envs instead of an interleaved
# 5-region frame, and a real validation split).
# ---------------------------------------------------------------------------

STEP_DURATION = 5.0 / 60.0            # 5-minute steps (paper step protocol)
TRAIN_EPISODE_HOURS = 24 * 7          # 1-week training episodes
TRAIN_MAX_STEP = int(round(TRAIN_EPISODE_HOURS / STEP_DURATION))   # 2016
VALID_HOURS = 144.0                   # matches the standard/dispatch identity surfaces
VALID_MAX_STEP = int(round(VALID_HOURS / STEP_DURATION))          # 1728
TRAIN_FRACTION = 0.80                 # chronological per-region split
VALID_WINDOW_FRACTIONS = (0.10, 0.55)  # pre-specified, inside the validation slice
N_ENVS = 20                           # 5 regions x 4 battery variants

BATTERIES = [
    {"name": "medium_1c", "capacity_mwh": 10.0, "max_power_mw": 10.0, "init_soc_ratio": 0.5},
    {"name": "large_07c", "capacity_mwh": 50.0, "max_power_mw": 35.0, "init_soc_ratio": 0.5},
    {"name": "small_05c", "capacity_mwh": 2.0, "max_power_mw": 1.0, "init_soc_ratio": 0.5},
    {"name": "fast_375c", "capacity_mwh": 8.0, "max_power_mw": 30.0, "init_soc_ratio": 0.5},
]

# Explicit processed files (no globbing). All end strictly before 2024-01-01.
REGION_FILES = {
    "NSW1": {
        "file": "processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet",
        "expected_start": "2021-01-01",
        "expected_end": "2023-04-01",
    },
    "QLD1": {
        "file": "processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet",
        "expected_start": "2021-01-01",
        "expected_end": "2023-04-01",
    },
    "TAS1": {
        "file": "processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet",
        "expected_start": "2021-01-01",
        "expected_end": "2023-04-01",
    },
    "SA1": {
        "file": "processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet",
        "expected_start": "2022-04-01",
        "expected_end": "2023-12-01",
    },
    "VIC1": {
        "file": "processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet",
        "expected_start": "2021-04-01",
        "expected_end": "2023-12-01",
    },
}

DEFAULT_PPO_KWARGS: dict[str, Any] = {}   # SB3 2.8 PPO defaults, as in the old checkpoint

DEFAULT_MILESTONES = (250_000, 500_000, 1_000_000, 2_000_000, 4_000_000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _rel(path: Path) -> str:
    """Repo-relative path when possible, absolute otherwise."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - git should exist
        return "unknown"


def _git_status_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _versions() -> dict[str, str]:
    import torch
    import stable_baselines3

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "polars": pl.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _fill_missing_market_columns(frame: pl.DataFrame, reference_columns: Sequence[str]) -> pl.DataFrame:
    """Add absent market columns as 0.0 (TAS1 has no GEN_solar / GEN_solar_pct).

    Mirrors the ``diagonal_relaxed`` + ``fill_null(0.0)`` step in
    ``retrain_sb3_full_fcas.py`` so every region yields the same observation
    vector, without concatenating regions into one interleaved frame.
    """
    missing = [c for c in reference_columns if c not in frame.columns and c != "SETTLEMENTDATE"]
    if not missing:
        return frame
    return frame.with_columns([pl.lit(0.0).alias(c) for c in missing]).select(reference_columns)


def load_region_frames() -> tuple[dict[str, pl.DataFrame], dict[str, dict[str, Any]]]:
    """Load the explicit pre-2024 files and return (frames, provenance)."""
    cache_dir = ROOT / "data" / "aemo"
    raw: dict[str, pl.DataFrame] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for region, spec in REGION_FILES.items():
        path = cache_dir / spec["file"]
        if not path.is_file():
            raise SystemExit(f"STOP: required pre-2024 processed data missing: {path}")
        frame = pl.read_parquet(str(path))
        if "SETTLEMENTDATE" not in frame.columns:
            raise SystemExit(f"STOP: {path} has no SETTLEMENTDATE column")
        starts = frame["SETTLEMENTDATE"].min()
        ends = frame["SETTLEMENTDATE"].max()
        if ends is None or starts is None:
            raise SystemExit(f"STOP: {path} has an empty date range")
        if ends >= dt.datetime(2024, 1, 1):
            raise SystemExit(
                f"STOP: {path} extends past 2023-12-31 ({ends}); refusing to train on test-era data"
            )
        expected_start = dt.datetime.fromisoformat(spec["expected_start"])
        expected_end = dt.datetime.fromisoformat(spec["expected_end"])
        if abs((starts - expected_start).total_seconds()) > 86_400 or abs(
            (ends - expected_end).total_seconds()
        ) > 86_400:
            raise SystemExit(
                f"STOP: {path} spans {starts}..{ends}, expected "
                f"{spec['expected_start']}..{spec['expected_end']}"
            )
        raw[region] = frame
        provenance[region] = {
            "path": _rel(path),
            "sha256": _sha256(path),
            "rows": int(frame.height),
            "columns": list(frame.columns),
            "min_settlementdate": str(starts),
            "max_settlementdate": str(ends),
            "expected_start": spec["expected_start"],
            "expected_end": spec["expected_end"],
        }

    reference_columns = list(raw["NSW1"].columns)
    frames = {r: _fill_missing_market_columns(f, reference_columns) for r, f in raw.items()}
    return frames, provenance


def split_frames(frames: dict[str, pl.DataFrame]) -> tuple[
    dict[str, pl.DataFrame], dict[str, pl.DataFrame], dict[str, dict[str, Any]]
]:
    """Chronological per-region train/validation split (both pre-2024)."""
    train: dict[str, pl.DataFrame] = {}
    valid: dict[str, pl.DataFrame] = {}
    split: dict[str, dict[str, Any]] = {}
    for region, frame in frames.items():
        cut = int(round(frame.height * TRAIN_FRACTION))
        if cut <= TRAIN_MAX_STEP or frame.height - cut <= VALID_MAX_STEP * 1.2:
            raise SystemExit(
                f"STOP: {region} has insufficient rows for a clean split "
                f"(rows={frame.height}, cut={cut})"
            )
        tr = frame.head(cut)
        va = frame.tail(frame.height - cut)
        # Enforce non-overlap of the timestamps themselves.
        if tr["SETTLEMENTDATE"].max() >= va["SETTLEMENTDATE"].min():
            raise SystemExit(f"STOP: {region} train/validation windows overlap")
        train[region] = tr
        valid[region] = va
        split[region] = {
            "rows_total": int(frame.height),
            "train_rows": int(tr.height),
            "valid_rows": int(va.height),
            "train_start": str(tr["SETTLEMENTDATE"].min()),
            "train_end": str(tr["SETTLEMENTDATE"].max()),
            "valid_start": str(va["SETTLEMENTDATE"].min()),
            "valid_end": str(va["SETTLEMENTDATE"].max()),
        }
    return train, valid, split


def validation_windows(valid_frames: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    """Pre-specified fixed 144 h validation windows (2 per region)."""
    windows: list[dict[str, Any]] = []
    for region, frame in valid_frames.items():
        max_start = frame.height - VALID_MAX_STEP
        if max_start <= 0:
            raise SystemExit(f"STOP: {region} validation slice shorter than one episode")
        for idx, frac in enumerate(VALID_WINDOW_FRACTIONS):
            offset = min(int(round(frame.height * frac)), max_start)
            offsets = frame["SETTLEMENTDATE"].to_list()
            windows.append(
                {
                    "region": region,
                    "window_index": idx,
                    "offset": offset,
                    "fraction": frac,
                    "start": str(offsets[offset]),
                    "end": str(offsets[min(offset + VALID_MAX_STEP - 1, frame.height - 1)]),
                }
            )
    return windows


def build_manifest(out_dir: Path) -> dict[str, Any]:
    frames, provenance = load_region_frames()
    train, valid, split = split_frames(frames)
    windows = validation_windows(valid)
    manifest = {
        "schema": "energydecision.ppo_retrain.protocol.v1",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_status_short": _git_status_short(),
        "versions": _versions(),
        "device_note": "PPO MlpPolicy trained on CPU (SB3 recommends CPU for MLPs); "
        "identical algorithm, faster wall clock than cuda for this workload.",
        "protocol": {
            "regions": list(REGION_FILES),
            "battery_variants": BATTERIES,
            "step_duration_hours": STEP_DURATION,
            "train_episode_hours": TRAIN_EPISODE_HOURS,
            "train_max_step": TRAIN_MAX_STEP,
            "validation_episode_hours": VALID_HOURS,
            "validation_max_step": VALID_MAX_STEP,
            "train_fraction": TRAIN_FRACTION,
            "validation_window_fractions": list(VALID_WINDOW_FRACTIONS),
            "n_envs": N_ENVS,
            "action_mode": "full_fcas",
            "degradation_mode": "real_world",
            "degradation_chemistry": "LFP",
            "degradation_temperature": 30.0,
            "market_impact": "identity",
            "random_episode_start": True,
            "train_data_era": "pre-2024 (2021-01-01 .. 2023-12-01)",
            "test_era_seen_in_training": False,
            "env_construction": "one env per (region x battery variant); regions are NOT "
            "concatenated, unlike retrain_sb3_full_fcas.py which interleaves all "
            "5 regions row-by-row after sorting by SETTLEMENTDATE.",
        },
        "data_files": provenance,
        "split": split,
        "validation_windows": windows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "data_split_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Environment / model construction
# ---------------------------------------------------------------------------

def _import_env_helpers():
    from aemo_notebook_utils import (  # noqa: WPS433 (runtime import by design)
        create_aemo_env,
        make_multi_scenario_aemo_env_fns,
        resolve_battery_variants,
    )

    return create_aemo_env, make_multi_scenario_aemo_env_fns, resolve_battery_variants


def build_vec_env(train_frames: dict[str, pl.DataFrame], seed: int):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    _, make_multi_scenario_aemo_env_fns, _ = _import_env_helpers()
    scenario_data = [({"label": region}, frame) for region, frame in train_frames.items()]
    env_fns = make_multi_scenario_aemo_env_fns(
        scenario_data=scenario_data,
        battery_variants=BATTERIES,
        episodes_per_variant=1,
        max_step=TRAIN_MAX_STEP,
        step_duration=STEP_DURATION,
        action_mode="full_fcas",
        degradation_mode="real_world",
        degradation_chemistry="LFP",
        degradation_temperature=30.0,
        random_episode_start=True,
    )
    assert len(env_fns) == N_ENVS, f"expected {N_ENVS} envs, got {len(env_fns)}"
    monitored = [
        (lambda fn=fn: Monitor(fn()))  # bind per-factory to avoid late binding
        for fn in env_fns
    ]
    vec_env = DummyVecEnv(monitored)
    vec_env.seed(seed)
    return vec_env


def build_ppo(vec_env, seed: int, overrides: dict[str, Any] | None = None, device: str = "cpu"):
    from stable_baselines3 import PPO

    kwargs = dict(DEFAULT_PPO_KWARGS)
    kwargs.update(overrides or {})
    kwargs.setdefault("verbose", 0)
    if "policy_kwargs" in kwargs and isinstance(kwargs["policy_kwargs"].get("net_arch"), tuple):
        kwargs["policy_kwargs"] = dict(kwargs["policy_kwargs"])
        kwargs["policy_kwargs"]["net_arch"] = list(kwargs["policy_kwargs"]["net_arch"])
    model = PPO("MlpPolicy", vec_env, seed=seed, device=device, **kwargs)
    model.set_random_seed(seed)
    return model


def validate_model(
    model,
    valid_frames: dict[str, pl.DataFrame],
    windows: list[dict[str, Any]],
    *,
    deterministic: bool = True,
    tag: str = "",
) -> dict[str, Any]:
    """Run the fixed validation episodes and return raw-$ and normalised metrics."""
    create_aemo_env, _, resolve_battery_variants = _import_env_helpers()
    resolved = resolve_battery_variants(BATTERIES)
    rows: list[dict[str, Any]] = []
    for window in windows:
        frame = valid_frames[window["region"]]
        start = int(window["offset"])
        sliced = frame.slice(start, VALID_MAX_STEP)
        for battery in resolved:
            env = create_aemo_env(
                processed_data=sliced,
                battery_variant=battery,
                max_step=VALID_MAX_STEP,
                step_duration=STEP_DURATION,
                action_mode="full_fcas",
                degradation_mode="real_world",
                degradation_chemistry="LFP",
                degradation_temperature=30.0,
                random_episode_start=False,
            )
            obs, _ = env.reset(seed=0)
            total_reward = 0.0
            info: dict[str, Any] = {}
            steps = 0
            energy = fcas = deg = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, _truncated, info = env.step(action)
                total_reward += float(reward)
                # info values are *per step*; accumulate to episode totals.
                energy += float(info.get("energy_revenue", 0.0))
                fcas += float(info.get("fcas_revenue", 0.0))
                deg += float(info.get("degradation_cost", 0.0))
                steps += 1
            rows.append(
                {
                    "region": window["region"],
                    "window_index": window["window_index"],
                    "window_start": window["start"],
                    "battery": battery["label"],
                    "capacity_mwh": float(battery["battery_capacity"]),
                    "max_power_mw": float(battery["max_battery_flow"]),
                    "steps": steps,
                    "profit": energy + fcas - deg,
                    "energy_revenue": energy,
                    "fcas_revenue": fcas,
                    "degradation_cost": deg,
                    # cross-check: cumulative environment counters at episode end
                    "cumulative_total_revenue": float(info.get("total_revenue", 0.0)),
                    "cumulative_total_degradation_cost": float(
                        info.get("total_degradation_cost", 0.0)
                    ),
                    "normalized_reward_sum": total_reward,
                }
            )
    profits = np.array([r["profit"] for r in rows], dtype=float)
    by_battery: dict[str, list[float]] = {}
    by_region: dict[str, list[float]] = {}
    for row in rows:
        by_battery.setdefault(str(row["battery"]), []).append(float(row["profit"]))
        by_region.setdefault(str(row["region"]), []).append(float(row["profit"]))
    summary = {
        "tag": tag,
        "deterministic": deterministic,
        "n_episodes": len(rows),
        "mean_profit": float(profits.mean()),
        "median_profit": float(np.median(profits)),
        "std_profit": float(profits.std(ddof=1)) if len(profits) > 1 else 0.0,
        "mean_profit_by_battery": {k: float(np.mean(v)) for k, v in by_battery.items()},
        "mean_profit_by_region": {k: float(np.mean(v)) for k, v in by_region.items()},
        "mean_normalized_reward_sum": float(
            np.mean([r["normalized_reward_sum"] for r in rows])
        ),
        "episodes": rows,
    }
    return summary


def read_ppo_metadata(model_path: Path) -> dict[str, Any]:
    import zipfile

    keys = [
        "num_timesteps",
        "_total_timesteps",
        "n_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "learning_rate",
        "policy_kwargs",
        "seed",
        "_n_updates",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "gae_lambda",
        "target_kl",
        "device",
    ]
    with zipfile.ZipFile(model_path) as archive:
        data = json.loads(archive.read("data"))
    meta: dict[str, Any] = {}
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict) and "value_schedule" in value:
            value = value["value_schedule"]
        meta[key] = value
    meta["sha256"] = _sha256(model_path)
    meta["size_bytes"] = model_path.stat().st_size
    return meta


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def cmd_manifest(args: argparse.Namespace) -> None:
    manifest = build_manifest(Path(args.out_dir))
    print(json.dumps({k: manifest[k] for k in ("git_sha", "split")}, indent=2))
    print(f"Wrote {Path(args.out_dir) / 'data_split_manifest.json'}")


def _milestones(requested: int, explicit: Sequence[int] | None) -> list[int]:
    if explicit:
        steps = sorted({int(s) for s in explicit})
    else:
        steps = [s for s in DEFAULT_MILESTONES if s <= requested]
        if requested not in steps:
            steps.append(requested)
    if steps[-1] != requested:
        steps.append(requested)
    return steps


def cmd_train(args: argparse.Namespace) -> None:
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.torch_threads))

    overrides: dict[str, Any] = {}
    if args.config_json:
        config = json.loads(Path(args.config_json).read_text())
        overrides = dict(config.get("overrides", config))
        for meta_key in ("name", "validation", "score", "trial", "net_arch_label"):
            overrides.pop(meta_key, None)
    elif args.config == "default":
        overrides = {}
    else:
        raise SystemExit(f"unknown --config {args.config!r}")

    manifest = build_manifest(out_dir)
    frames, _ = load_region_frames()
    train_frames, valid_frames, _split = split_frames(frames)
    windows = manifest["validation_windows"]

    vec_env = build_vec_env(train_frames, args.seed)
    model = build_ppo(vec_env, args.seed, overrides, device=args.device)

    steps = _milestones(int(args.timesteps), args.milestones)
    history: list[dict[str, Any]] = []
    t_start = time.time()
    trained = 0
    for target in steps:
        if trained < target:
            t0 = time.time()
            model.learn(total_timesteps=target - trained, reset_num_timesteps=False,
                        progress_bar=False)
            elapsed = time.time() - t0
            trained = int(model.num_timesteps)
        else:
            elapsed = 0.0
        # SB3 always completes the current rollout, so `trained` can exceed the
        # requested milestone. Record the *actual* step count and never emit the
        # same checkpoint twice.
        if any(h["model_num_timesteps"] == trained for h in history):
            continue
        ckpt = out_dir / f"{args.tag}_{trained // 1000:04d}k.zip"
        model.save(str(ckpt))
        val = validate_model(model, valid_frames, windows, tag=f"{args.tag}@{trained}")
        ep_info = {}
        if model.ep_info_buffer:
            rewards = [float(ep["r"]) for ep in model.ep_info_buffer]
            lengths = [float(ep["l"]) for ep in model.ep_info_buffer]
            ep_info = {
                "episodes_in_buffer": len(rewards),
                "mean_episode_normalized_reward": float(np.mean(rewards)),
                "mean_episode_length": float(np.mean(lengths)),
            }
        row = {
            "requested_milestone": target,
            "model_num_timesteps": trained,
            "wall_seconds": elapsed,
            "steps_per_second": (trained - (history[-1]["model_num_timesteps"] if history else 0)) / elapsed if elapsed > 0 else None,
            "checkpoint": _rel(ckpt),
            "checkpoint_sha256": _sha256(ckpt),
            "validation": {k: v for k, v in val.items() if k != "episodes"},
            "rollout_episode_stats": ep_info,
        }
        history.append(row)
        _write_json(
            out_dir / f"{args.tag}_validation_{trained // 1000:04d}k.json",
            {"requested_milestone": target, "model_num_timesteps": trained,
             "windows": windows, "validation": val},
        )
        _write_json(out_dir / f"{args.tag}_history.json", {"history": history})
        print(
            f"[{args.tag}] {trained:>9,} steps (target {target:,})  "
            f"{row['steps_per_second'] or 0:.0f} steps/s  "
            f"val mean profit ${val['mean_profit']:,.2f}  "
            f"({elapsed / 60:.1f} min)",
            flush=True,
        )
    total_wall = time.time() - t_start
    vec_env.close()

    final = history[-1]
    meta = {
        "tag": args.tag,
        "seed": args.seed,
        "config": args.config,
        "config_json": args.config_json,
        "overrides": overrides,
        "device": args.device,
        "requested_timesteps": int(args.timesteps),
        "actual_num_timesteps": final["model_num_timesteps"],
        "n_envs": N_ENVS,
        "total_wall_seconds": total_wall,
        "git_sha": manifest["git_sha"],
        "data_split_manifest": _rel(out_dir / "data_split_manifest.json"),
        "validation_mean_profit": final["validation"]["mean_profit"],
        "final_checkpoint": final["checkpoint"],
        "final_checkpoint_sha256": final["checkpoint_sha256"],
        "ppo_checkpoint_metadata": read_ppo_metadata(Path(final["checkpoint"])),
        "history": history,
    }
    _write_json(out_dir / f"{args.tag}_meta.json", meta)
    print(f"[{args.tag}] done in {total_wall / 3600:.2f} h; meta -> {out_dir / f'{args.tag}_meta.json'}")


def cmd_tune(args: argparse.Namespace) -> None:
    import optuna
    import torch

    out_dir = Path(args.out_dir) / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.torch_threads))

    manifest = build_manifest(Path(args.out_dir))
    frames, _ = load_region_frames()
    train_frames, valid_frames, _split = split_frames(frames)
    windows = manifest["validation_windows"]

    def objective(trial: "optuna.Trial") -> float:
        net_arch = trial.suggest_categorical("net_arch", ["64x64", "256x256", "400x300"])
        arch = {"64x64": [64, 64], "256x256": [256, 256], "400x300": [400, 300]}[net_arch]
        overrides = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "gamma": trial.suggest_float("gamma", 0.95, 0.9995),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
            "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
            "policy_kwargs": {"net_arch": arch},
        }
        t0 = time.time()
        vec_env = build_vec_env(train_frames, args.seed)
        model = build_ppo(vec_env, args.seed, overrides, device=args.device)
        model.learn(total_timesteps=int(args.timesteps), progress_bar=False)
        val = validate_model(model, valid_frames, windows, tag=f"trial{trial.number}")
        elapsed = time.time() - t0
        vec_env.close()
        trial.set_user_attr("wall_seconds", elapsed)
        trial.set_user_attr("mean_profit_by_battery", val["mean_profit_by_battery"])
        trial.set_user_attr("mean_profit_by_region", val["mean_profit_by_region"])
        trial.set_user_attr("n_episodes", val["n_episodes"])
        trial.set_user_attr("seed", args.seed)
        print(
            f"[optuna] trial {trial.number}: ${val['mean_profit']:,.2f} in {elapsed / 60:.1f} min "
            f"-> {trial.params}",
            flush=True,
        )
        return float(val["mean_profit"])

    # Fixed sampler seed => the search is reproducible and identical across runs.
    sampler = optuna.samplers.TPESampler(seed=int(args.optuna_seed), n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(args.trials), n_jobs=1)

    best = dict(study.best_trial.params)
    net_arch = best.pop("net_arch")
    best["policy_kwargs"] = {
        "net_arch": {"64x64": [64, 64], "256x256": [256, 256], "400x300": [400, 300]}[net_arch]
    }
    best["net_arch_label"] = net_arch
    payload = {
        "overrides": best,
        "score": float(study.best_value),
        "name": f"optuna_trial_{study.best_trial.number}",
        "validation": "fixed 40-episode pre-2024 validation set (see data_split_manifest.json)",
        "trial": study.best_trial.number,
    }
    _write_json(out_dir / "best_params.json", payload)

    trials_rows = []
    for trial in study.trials:
        trials_rows.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
                "wall_seconds": trial.user_attrs.get("wall_seconds"),
                "mean_profit_by_battery": trial.user_attrs.get("mean_profit_by_battery"),
                "mean_profit_by_region": trial.user_attrs.get("mean_profit_by_region"),
            }
        )
    _write_json(out_dir / "trials.json", {"seed": args.seed, "timesteps": int(args.timesteps),
                                          "trials": trials_rows, "best": payload})
    trials_df = study.trials_dataframe()
    csv_target = out_dir / "trials.csv"
    try:
        trials_df.write_csv(str(csv_target))
    except AttributeError:
        trials_df.to_pandas().to_csv(csv_target, index=False)
    print(f"[optuna] best ${study.best_value:,.2f} -> {out_dir / 'best_params.json'}")


def cmd_validate(args: argparse.Namespace) -> None:
    from stable_baselines3 import PPO

    frames, _ = load_region_frames()
    _train_frames, valid_frames, _split = split_frames(frames)
    manifest_path = Path(args.out_dir) / "data_split_manifest.json"
    if not manifest_path.is_file():
        manifest = build_manifest(Path(args.out_dir))
    else:
        manifest = json.loads(manifest_path.read_text())
    windows = manifest["validation_windows"]
    model = PPO.load(str(args.model), device=args.device)
    summary = validate_model(model, valid_frames, windows, deterministic=not args.stochastic,
                             tag=Path(args.model).stem)
    payload = {
        "model": _rel(args.model),
        "model_sha256": _sha256(Path(args.model)),
        "metadata": read_ppo_metadata(Path(args.model)),
        "validation": summary,
    }
    out = Path(args.out) if args.out else Path(args.out_dir) / f"validate_{Path(args.model).stem}.json"
    _write_json(out, payload)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", type=str, required=True)
    common.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    common.add_argument("--torch-threads", type=int, default=1)

    p_manifest = sub.add_parser("manifest", parents=[common])
    p_manifest.set_defaults(func=cmd_manifest)

    p_train = sub.add_parser("train", parents=[common])
    p_train.add_argument("--tag", type=str, required=True)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--timesteps", type=int, default=4_000_000)
    p_train.add_argument("--config", type=str, default="default", choices=["default"])
    p_train.add_argument("--config-json", type=str, default=None)
    p_train.add_argument("--milestones", type=int, nargs="*", default=None)
    p_train.set_defaults(func=cmd_train)

    p_tune = sub.add_parser("tune", parents=[common])
    p_tune.add_argument("--trials", type=int, default=10)
    p_tune.add_argument("--timesteps", type=int, default=400_000)
    p_tune.add_argument("--seed", type=int, default=0)
    p_tune.add_argument("--optuna-seed", type=int, default=42)
    p_tune.set_defaults(func=cmd_tune)

    p_val = sub.add_parser("validate", parents=[common])
    p_val.add_argument("--model", type=str, required=True)
    p_val.add_argument("--out", type=str, default=None)
    p_val.add_argument("--stochastic", action="store_true")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
