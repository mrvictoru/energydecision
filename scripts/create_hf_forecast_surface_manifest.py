from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_dt_hf import (
    build_surface_manifest,
    load_model_kwargs,
    repo_root,
    write_placeholder_loss_csv,
)

FORECAST_HF_REPO = "mrvictoru/energydecision-dt-v2-forecast"
FORECAST_HF_FILENAME = "aemo_dt_fcas_model.pt"
FORECAST_SURFACE_PRESET = "hf_modern_v2_forecast"
FORECAST_MODEL_CONFIG_RELATIVE = Path("configs") / "aemo_decision_transformer_model_kwargs_forecast.json"
DEFAULT_OUTPUT_DIR = repo_root() / "models" / "aemo" / "dt" / "hf_forecast"


def _forecast_model_config_path(root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / FORECAST_MODEL_CONFIG_RELATIVE


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an evaluator surface manifest for the HuggingFace Forecast Decision Transformer checkpoint."
    )
    parser.add_argument("--hf-repo", default=FORECAST_HF_REPO, help="HuggingFace repo ID.")
    parser.add_argument("--hf-filename", default=FORECAST_HF_FILENAME, help="Checkpoint filename in the repo.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=_forecast_model_config_path(),
        help="Path to the ForecastDecisionTransformer model kwargs JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the manifest and placeholder loss CSV.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local checkpoint path. When omitted, downloads from HuggingFace.",
    )
    parser.add_argument("--manifest-name", default="surface_manifest.json", help="Output manifest filename.")
    parser.add_argument("--loss-csv-name", default="dummy_loss.csv", help="Placeholder loss CSV filename.")
    parser.add_argument("--surface-preset", default=FORECAST_SURFACE_PRESET, help="surface_preset value.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        args.checkpoint_path.resolve()
        if args.checkpoint_path is not None
        else Path(hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename)).resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_kwargs = load_model_kwargs(args.model_config)
    model_kwargs["model_class"] = "ForecastDecisionTransformer"
    loss_csv_path = write_placeholder_loss_csv(output_dir / args.loss_csv_name)
    manifest = build_surface_manifest(
        model_kwargs=model_kwargs,
        save_path=checkpoint_path,
        loss_csv_path=loss_csv_path,
        surface_preset=args.surface_preset,
        model_variant="full_fcas_forecast",
        action_mode="full_fcas",
        hf_repo=args.hf_repo,
        hf_filename=args.hf_filename,
    )

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[HF forecast manifest] Checkpoint: {checkpoint_path}")
    print(f"[HF forecast manifest] Loss CSV:   {loss_csv_path}")
    print(f"[HF forecast manifest] Manifest:   {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
