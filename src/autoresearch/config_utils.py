from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Keys the agent may change in v1
ALLOWED_MUTABLE_KEYS_V1: frozenset[str] = frozenset(
    {
        "n_block",
        "h_dim",
        "n_heads",
        "drop_p",
        "context_len",
        "rope_enabled",
        "rope_base",
        "rope_max_position",
        "batch_size",
        "lr",
        "epochs",
        "return_scale",
        "action_loss_weight",
        "state_loss_weight",
        "return_loss_weight",
        "weight_decay",
        "rtg_value",
        "recommended_rtg_percentile",
        "action_mode",
        "degradation_mode",
        "degradation_chemistry",
        "step_duration_hours",
    }
)

# Keys that are NEVER mutable -- frozen per benchmark
FROZEN_KEYS: frozenset[str] = frozenset(
    {
        "state_dim",
        "act_dim",
        "max_timestep",
        "discount",
        "data_dir",
        "dataset_path",
        "env_kwargs",
        "eval_episodes",
        "eval_seed",
        "primary_metric",
        "guardrails",
        "stage_a_timeout",
        "stage_b_timeout",
    }
)


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object.")
    if not data:
        raise ValueError("Config must not be empty.")
    return data


def diff_configs(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    all_keys = sorted(set(baseline.keys()) | set(candidate.keys()))
    for key in all_keys:
        old = baseline.get(key)
        new = candidate.get(key)
        if old != new:
            diff[key] = {"old": old, "new": new}
    return diff


def validate_mutable_surface(
    candidate: dict[str, Any],
    allowed_keys: frozenset[str] = ALLOWED_MUTABLE_KEYS_V1,
) -> None:
    unknown = sorted(key for key in candidate.keys() if key not in allowed_keys)
    if unknown:
        raise ValueError(f"Candidate modifies disallowed keys: {unknown}")
    frozen = sorted(key for key in candidate.keys() if key in FROZEN_KEYS)
    if frozen:
        raise ValueError(f"Candidate attempted to modify frozen keys: {frozen}")


def _model_kwargs_from_config(config: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "state_dim": int(benchmark["state_dim"]),
        "act_dim": int(benchmark["act_dim"]),
        "n_block": int(config.get("n_block", 2 if benchmark.get("environment") == "household" else 4)),
        "h_dim": int(config.get("h_dim", 128)),
        "context_len": int(config.get("context_len", 60 if benchmark.get("environment") == "household" else 288)),
        "n_heads": int(config.get("n_heads", 8)),
        "drop_p": float(config.get("drop_p", 0.1)),
        "max_timestep": int(benchmark["max_timestep"]),
        "rope_enabled": bool(config.get("rope_enabled", benchmark.get("environment") == "aemo")),
        "rope_max_position": int(
            config.get(
                "rope_max_position",
                int(config.get("context_len", 60 if benchmark.get("environment") == "household" else 288)) * 3,
            )
        ),
        "rope_base": float(config.get("rope_base", 10000.0)),
    }
    return defaults


def write_model_kwargs(config: dict[str, Any], benchmark: dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_kwargs.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_model_kwargs_from_config(config, benchmark), fh, indent=2)
    return out_path


def build_training_cli_args(
    config: dict[str, Any],
    benchmark: dict[str, Any],
    output_dir: str,
    script: str,
    epochs_override: int | None = None,
) -> list[str]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_config_path = out_dir / "model_kwargs.json"
    save_path = out_dir / "model_final.pt"
    checkpoint_path = out_dir / "checkpoint.pt"
    loss_csv_path = out_dir / "loss.csv"

    epochs = int(epochs_override if epochs_override is not None else config.get("epochs", 1))
    batch_size = int(config.get("batch_size", 6))
    lr = float(config.get("lr", 2e-5))
    return_scale = float(config.get("return_scale", 1.0))
    action_loss_weight = float(config.get("action_loss_weight", 1.0))
    state_loss_weight = float(config.get("state_loss_weight", 0.01))
    return_loss_weight = float(config.get("return_loss_weight", 0.002))
    weight_decay = float(config.get("weight_decay", 1e-4))

    args = [
        sys.executable,
        str(Path(script).resolve()),
        "--model-config",
        str(model_config_path),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--save-path",
        str(save_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--loss-csv-path",
        str(loss_csv_path),
        "--return-scale",
        str(return_scale),
        "--action-loss-weight",
        str(action_loss_weight),
        "--state-loss-weight",
        str(state_loss_weight),
        "--return-loss-weight",
        str(return_loss_weight),
        "--weight-decay",
        str(weight_decay),
    ]

    if benchmark.get("environment") == "household":
        args.extend([
            "--data-dir",
            str(benchmark["data_dir"]),
            "--patterns",
        ])
        for pattern in benchmark.get("train_patterns", ["train"]):
            args.append(str(pattern))

        val_patterns = benchmark.get("val_patterns")
        if val_patterns:
            args.extend(["--val-data-dir", str(benchmark["data_dir"]), "--val-patterns"])
            for pattern in val_patterns:
                args.append(str(pattern))

        args.extend(["--discount", str(float(benchmark.get("discount", 0.99)))])

    elif benchmark.get("environment") == "aemo":
        args.extend(
            [
                "--dataset-path",
                str(benchmark["dataset_path"]),
            ]
        )
    else:
        raise ValueError(f"Unsupported environment: {benchmark.get('environment')}")

    return args
