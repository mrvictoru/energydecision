import json
import os
import sys
import types
from importlib import util
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import aemo_dt_hf  # noqa: E402
import create_hf_surface_manifest as manifest_creator  # noqa: E402
import run_aemo_dt_rtg_sweep as rtg_sweep  # noqa: E402


def _load_module_with_stubs(module_name: str, path: Path):
    stubbed_modules = {
        "aemo_notebook_utils": types.SimpleNamespace(
            create_aemo_env=lambda *args, **kwargs: None,
            fetch_and_preprocess_aemo_scenarios=lambda *args, **kwargs: ({}, []),
            resolve_battery_variants=lambda variants: variants,
        ),
        "decision": types.SimpleNamespace(AEMOAgent=object),
        "decision_transformer": types.SimpleNamespace(DecisionTransformer=object),
        "grpo_posttraining": types.SimpleNamespace(
            GRPOPrompt=object,
            GRPOTrainer=object,
            load_pretrained_dt_for_grpo=lambda *args, **kwargs: None,
            sample_rtg_values=lambda *args, **kwargs: [],
        ),
        "huggingface_hub": types.SimpleNamespace(hf_hub_download=lambda *args, **kwargs: "dummy.pt"),
    }
    previous = {name: sys.modules.get(name) for name in stubbed_modules}
    sys.modules.update(stubbed_modules)
    try:
        spec = util.spec_from_file_location(module_name, path)
        module = util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def test_modern_v2_model_config_defaults_match_hf_artifact():
    model_kwargs = aemo_dt_hf.load_model_kwargs(aemo_dt_hf.modern_v2_model_config_path())

    assert model_kwargs["state_dim"] == 18
    assert model_kwargs["act_dim"] == 9
    assert model_kwargs["n_block"] == 8
    assert model_kwargs["h_dim"] == 768
    assert model_kwargs["context_len"] == 210
    assert model_kwargs["n_heads"] == 12
    assert model_kwargs["n_kv_heads"] == 6
    assert model_kwargs["qk_norm"] is True
    assert model_kwargs["tie_weights"] is True
    assert model_kwargs["rope_enabled"] is False


def test_build_surface_manifest_includes_hf_source(tmp_path: Path):
    manifest = aemo_dt_hf.build_surface_manifest(
        model_kwargs={"state_dim": 18, "act_dim": 9},
        save_path=tmp_path / "model.pt",
        loss_csv_path=tmp_path / "loss.csv",
        hf_repo="example/repo",
        hf_filename="model.pt",
    )

    assert manifest["surface_preset"] == aemo_dt_hf.MODERN_V2_SURFACE_PRESET
    assert manifest["paths"]["save_path"].endswith("model.pt")
    assert manifest["source"] == {
        "kind": "huggingface",
        "repo_id": "example/repo",
        "filename": "model.pt",
    }


def test_write_placeholder_loss_csv_matches_evaluator_columns(tmp_path: Path):
    loss_csv_path = aemo_dt_hf.write_placeholder_loss_csv(tmp_path / "dummy_loss.csv")

    content = loss_csv_path.read_text(encoding="utf-8").splitlines()
    assert content[0] == "epoch,train_total,train_action,train_state,train_return,val_total,val_action,val_state,val_return"
    assert content[1].startswith("1,0.0,0.0,0.0,0.0")


def test_create_hf_surface_manifest_main_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(manifest_creator, "hf_hub_download", lambda repo_id, filename: str(checkpoint_path))

    exit_code = manifest_creator.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--model-config",
            str(aemo_dt_hf.modern_v2_model_config_path()),
        ]
    )

    assert exit_code == 0
    manifest_path = tmp_path / "out" / "hf_modern_surface_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["repo_id"] == aemo_dt_hf.MODERN_V2_HF_REPO
    assert manifest["source"]["filename"] == aemo_dt_hf.MODERN_V2_HF_FILENAME
    assert manifest["model_kwargs"]["h_dim"] == 768
    assert (tmp_path / "out" / "dummy_loss.csv").exists()


def test_set_dt_rtg_value_updates_only_candidate_dt():
    config = {
        "policies": [
            {"name": "candidate_dt", "kind": "dt", "rtg_value": 0.0},
            {"name": "other_dt", "kind": "dt", "rtg_value": 5.0},
            {"name": "rule", "kind": "rule"},
        ]
    }

    updated = rtg_sweep.set_dt_rtg_value(config, rtg_value=1.5, candidate_policy_name="candidate_dt")

    assert updated["policies"][0]["rtg_value"] == 1.5
    assert updated["policies"][1]["rtg_value"] == 5.0


def test_extract_candidate_metrics_reads_summary_payload():
    summary = {
        "heldout_evaluation": {
            "reference_policy": "dispatch_dalrymple_north",
            "aggregate_metrics": [
                {"experiment": "candidate_dt", "avg_profit_per_episode": 123.0, "avg_reward_per_episode": 5.0},
            ],
            "paired_comparisons_vs_reference": {
                "candidate_dt": {"mean_diff": 1.2, "p_value": 0.03}
            },
        }
    }

    metrics = rtg_sweep.extract_candidate_metrics(summary, "candidate_dt")

    assert metrics["avg_profit_per_episode"] == 123.0
    assert metrics["paired_mean_diff_vs_reference"] == pytest.approx(1.2)
    assert metrics["reference_policy"] == "dispatch_dalrymple_north"


def test_run_grpo_posttraining_defaults_to_modern_v2_paths():
    module = _load_module_with_stubs(
        "run_grpo_posttraining_stub",
        Path(__file__).resolve().parents[1] / "src" / "run_grpo_posttraining.py",
    )

    args = module.parse_args([])

    assert args.hf_repo == aemo_dt_hf.MODERN_V2_HF_REPO
    assert args.hf_filename == aemo_dt_hf.MODERN_V2_HF_FILENAME
    assert args.model_config == aemo_dt_hf.modern_v2_model_config_path()


def test_run_grpo_multi_region_defaults_to_modern_v2_paths():
    module = _load_module_with_stubs(
        "run_grpo_multi_region_stub",
        Path(__file__).resolve().parents[1] / "src" / "run_grpo_multi_region.py",
    )

    args = module.parse_args([])

    assert args.hf_repo == aemo_dt_hf.MODERN_V2_HF_REPO
    assert args.hf_filename == aemo_dt_hf.MODERN_V2_HF_FILENAME
    assert args.model_config == aemo_dt_hf.modern_v2_model_config_path()
