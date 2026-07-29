from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MODERN_V2_HF_REPO = "mrvictoru/energydecision-dt-v2"
MODERN_V2_HF_FILENAME = "aemo_dt_fcas_model.pt"
MODERN_V2_SURFACE_PRESET = "hf_modern_v2"
MODERN_V2_MODEL_CONFIG_RELATIVE = Path("configs") / "aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def modern_v2_model_config_path(root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / MODERN_V2_MODEL_CONFIG_RELATIVE


def load_model_kwargs(model_config_path: str | Path) -> dict[str, Any]:
    path = Path(model_config_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model config must be a JSON object: {path}")
    return payload


def write_placeholder_loss_csv(path: str | Path) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "epoch,train_total,train_action,train_state,train_return,val_total,val_action,val_state,val_return\n"
        "1,0.0,0.0,0.0,0.0,,,,\n",
        encoding="utf-8",
    )
    return output_path


def build_surface_manifest(
    *,
    model_kwargs: dict[str, Any],
    save_path: str | Path,
    loss_csv_path: str | Path,
    surface_preset: str = MODERN_V2_SURFACE_PRESET,
    model_variant: str = "full_fcas",
    action_mode: str = "full_fcas",
    hf_repo: str | None = None,
    hf_filename: str | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "energydecision.dt_training_surface.v1",
        "surface_preset": surface_preset,
        "model_variant": model_variant,
        "action_mode": action_mode,
        "model_kwargs": dict(model_kwargs),
        "paths": {
            "save_path": str(Path(save_path).resolve()),
            "loss_csv_path": str(Path(loss_csv_path).resolve()),
        },
    }
    if hf_repo is not None or hf_filename is not None:
        manifest["source"] = {
            "kind": "huggingface",
            "repo_id": hf_repo,
            "filename": hf_filename,
        }
    return manifest


def set_dt_rtg_value(
    evaluation_config: dict[str, Any],
    *,
    rtg_value: float,
    candidate_policy_name: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(evaluation_config))
    matched = False
    for policy in updated.get("policies", []):
        if policy.get("kind") == "dt" and policy.get("name") == candidate_policy_name:
            policy["rtg_value"] = rtg_value
            matched = True
    if not matched:
        raise ValueError(f"Did not find DT policy named {candidate_policy_name!r} in evaluation config.")
    return updated


def extract_candidate_metrics(summary: dict[str, Any], candidate_policy_name: str) -> dict[str, Any]:
    heldout = summary.get("heldout_evaluation", {})
    metrics = next(
        (
            item for item in heldout.get("aggregate_metrics", [])
            if item.get("experiment") == candidate_policy_name
        ),
        None,
    )
    if metrics is None:
        raise ValueError(f"Summary did not include aggregate metrics for {candidate_policy_name!r}.")
    paired = heldout.get("paired_comparisons_vs_reference", {}).get(candidate_policy_name, {})
    return {
        "avg_reward_per_episode": metrics.get("avg_reward_per_episode"),
        "avg_profit_per_episode": metrics.get("avg_profit_per_episode"),
        "avg_fcas_revenue_per_episode": metrics.get("avg_fcas_revenue_per_episode"),
        "avg_degradation_cost_per_episode": metrics.get("avg_degradation_cost_per_episode"),
        "paired_mean_diff_vs_reference": paired.get("mean_diff"),
        "paired_p_value_vs_reference": paired.get("p_value"),
        "reference_policy": heldout.get("reference_policy"),
    }
