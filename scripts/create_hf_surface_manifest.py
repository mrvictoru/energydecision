from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aemo_dt_hf import (
    MODERN_V2_HF_FILENAME,
    MODERN_V2_HF_REPO,
    MODERN_V2_SURFACE_PRESET,
    build_surface_manifest,
    load_model_kwargs,
    modern_v2_model_config_path,
    repo_root,
    write_placeholder_loss_csv,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an evaluator surface manifest for a HuggingFace AEMO DT checkpoint.")
    parser.add_argument("--hf-repo", default=MODERN_V2_HF_REPO, help="HuggingFace repo ID for the checkpoint.")
    parser.add_argument("--hf-filename", default=MODERN_V2_HF_FILENAME, help="Checkpoint filename in the HuggingFace repo.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=modern_v2_model_config_path(),
        help="Path to the Decision Transformer model kwargs JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "models" / "aemo" / "dt" / "hf_v2_modern",
        help="Directory where the manifest and placeholder loss CSV will be written.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional local checkpoint path. When omitted, the checkpoint is downloaded from HuggingFace.",
    )
    parser.add_argument("--manifest-name", default="hf_modern_surface_manifest.json", help="Output manifest filename.")
    parser.add_argument("--loss-csv-name", default="dummy_loss.csv", help="Placeholder loss CSV filename for evaluator compatibility.")
    parser.add_argument("--surface-preset", default=MODERN_V2_SURFACE_PRESET, help="surface_preset value written into the manifest.")
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
    loss_csv_path = write_placeholder_loss_csv(output_dir / args.loss_csv_name)
    manifest = build_surface_manifest(
        model_kwargs=model_kwargs,
        save_path=checkpoint_path,
        loss_csv_path=loss_csv_path,
        surface_preset=args.surface_preset,
        hf_repo=args.hf_repo,
        hf_filename=args.hf_filename,
    )

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[HF manifest] Checkpoint: {checkpoint_path}")
    print(f"[HF manifest] Loss CSV:   {loss_csv_path}")
    print(f"[HF manifest] Manifest:   {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
