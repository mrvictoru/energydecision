#!/usr/bin/env python3
"""Precompute causal Granite-TTM household forecasts outside the simulator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from household_forecast import (
    apply_forecast_sidecar,
    forecast_quality,
    generate_causal_forecasts,
)
from household_ingest import build_year_dataset, env_view, split_segments

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r3"
MODEL_REVISION = "512-48-dec-512-r3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--normalized-dir", type=Path)
    source.add_argument("--synth-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--lead-steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args()


def _load_predictor(model_id: str, revision: str, device: str):
    try:
        from tsfm_public import TinyTimeMixerForDecomposedPrediction
    except ImportError as exc:
        raise RuntimeError(
            "granite-tsfm is unavailable. Run this script through "
            "scripts/run_household_ttm_forecasts.sh."
        ) from exc

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    model = TinyTimeMixerForDecomposedPrediction.from_pretrained(
        model_id,
        revision=revision,
    )
    model.to(resolved_device)
    model.eval()
    context_length = int(model.config.context_length)
    prediction_length = int(model.config.prediction_length)

    def predict(contexts: np.ndarray) -> np.ndarray:
        batch, context, channels = contexts.shape
        tensor = (
            torch.from_numpy(contexts)
            .permute(0, 2, 1)
            .reshape(batch * channels, context, 1)
            .to(resolved_device)
        )
        with torch.inference_mode():
            output = model(past_values=tensor).prediction_outputs
        return (
            output.reshape(batch, channels, prediction_length)
            .permute(0, 2, 1)
            .detach()
            .cpu()
            .numpy()
        )

    return predict, context_length, prediction_length, resolved_device


def _forecast_frame(
    frame: pl.DataFrame,
    predict,
    context_length: int,
    prediction_length: int,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, dict[str, float | int | None]]]:
    sidecar = generate_causal_forecasts(
        frame,
        predict,
        context_length=context_length,
        prediction_length=prediction_length,
        lead_steps=args.lead_steps,
        batch_size=args.batch_size,
    )
    return (
        apply_forecast_sidecar(frame, sidecar),
        sidecar,
        forecast_quality(frame, sidecar, lead_steps=args.lead_steps),
    )


def _regular_segments(frame: pl.DataFrame) -> list[pl.DataFrame]:
    """Split before cadence changes or missing intervals for valid TTM contexts."""
    timestamps = frame["Timestamp"].to_list()
    boundaries = [0]
    previous_nominal = None
    for index in range(1, len(timestamps)):
        minutes = (timestamps[index] - timestamps[index - 1]).total_seconds() / 60.0
        nominal = 5 if abs(minutes - 5) <= abs(minutes - 15) else 15
        if minutes > nominal * 1.5 or (
            previous_nominal is not None and nominal != previous_nominal
        ):
            boundaries.append(index)
        previous_nominal = nominal
    boundaries.append(len(frame))
    return [
        frame.slice(start, end - start)
        for start, end in zip(boundaries, boundaries[1:])
        if end > start
    ]


def _write_real(
    args: argparse.Namespace,
    predict,
    context_length: int,
    prediction_length: int,
) -> dict[str, object]:
    segments = [
        regular
        for segment in split_segments(build_year_dataset(args.normalized_dir))
        for regular in _regular_segments(segment)
    ]
    sidecars = []
    quality = []
    for index, segment in enumerate(segments):
        _, sidecar, segment_quality = _forecast_frame(
            env_view(segment), predict, context_length, prediction_length, args
        )
        sidecars.append(sidecar.with_columns(pl.lit(index).alias("SegmentID")))
        quality.append(segment_quality)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pl.concat(sidecars)
    result.write_parquet(args.output)
    return {"segments": len(segments), "rows": len(result), "quality_by_segment": quality}


def _write_synth(
    args: argparse.Namespace,
    predict,
    context_length: int,
    prediction_length: int,
) -> dict[str, object]:
    manifest_path = args.synth_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]
    if args.limit_episodes is not None:
        episodes = episodes[:args.limit_episodes]
    output_dir = args.output
    quality = []
    for index, entry in enumerate(episodes, start=1):
        source = args.synth_dir / entry["path"]
        enriched, _, episode_quality = _forecast_frame(
            pl.read_parquet(source), predict, context_length, prediction_length, args
        )
        quality.append({"episode_id": entry["episode_id"], **episode_quality})
        destination = output_dir / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        enriched.write_parquet(destination)
        print(f"[TTM] episode {index}/{len(episodes)}: {entry['path']}", flush=True)
    output_manifest = dict(manifest)
    output_manifest["episodes"] = episodes
    output_manifest["forecast"] = {
        "provider": "granite_ttm",
        "model": args.model,
        "revision": args.revision,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "lead_steps": args.lead_steps,
        "causal": True,
        "synthetic": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    return {"episodes": len(episodes), "quality_by_episode": quality}


def main() -> None:
    args = parse_args()
    predict, context_length, prediction_length, device = _load_predictor(
        args.model, args.revision, args.device
    )
    if args.lead_steps > prediction_length:
        raise ValueError(
            f"--lead-steps {args.lead_steps} exceeds model prediction length "
            f"{prediction_length}"
        )
    summary = (
        _write_real(args, predict, context_length, prediction_length)
        if args.normalized_dir is not None
        else _write_synth(args, predict, context_length, prediction_length)
    )
    metadata = {
        "schema": "energydecision.household_ttm_forecast.v1",
        "model": args.model,
        "revision": args.revision,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "lead_steps": args.lead_steps,
        "device": device,
        "causal_contract": "row t uses observations through t and predicts t + lead_steps",
        **summary,
    }
    metadata_path = (
        args.output.with_suffix(args.output.suffix + ".meta.json")
        if args.normalized_dir is not None
        else args.output / "forecast_metadata.json"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
