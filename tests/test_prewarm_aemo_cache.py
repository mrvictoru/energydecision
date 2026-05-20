import json
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import prewarm_aemo_cache as prewarm  # noqa: E402


def test_main_writes_cache_manifest(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "eval.json"
    config_path.write_text(
        json.dumps(
            {
                "heldout": {
                    "step_duration": 0.5,
                    "fit_global_stats": False,
                    "scenarios": [
                        {
                            "label": "heldout_nsw1",
                            "region": "NSW1",
                            "start_date": "2024-01-01",
                            "end_date": "2024-01-02",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "cache_manifest.json"

    monkeypatch.setattr(prewarm, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        prewarm,
        "preflight_processed_cache_paths",
        lambda **kwargs: [{"label": "heldout_nsw1", "cache_exists": False}],
    )
    monkeypatch.setattr(
        prewarm,
        "fetch_and_preprocess_aemo_scenarios",
        lambda **kwargs: (
            {"heldout_nsw1": pl.DataFrame({"RRP": [1.0, 2.0]})},
            [
                {
                    "label": "heldout_nsw1",
                    "region": "NSW1",
                    "start_date": "2024-01-01 00:00:00",
                    "end_date": "2024-01-02 00:00:00",
                }
            ],
        ),
    )

    prewarm.main(["--evaluation-config", str(config_path), "--output-path", str(output_path)])

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "energydecision.aemo_cache_prewarm.v1"
    assert manifest["cache_preflight"][0]["label"] == "heldout_nsw1"
    assert manifest["scenario_manifest"][0]["row_count"] == 2
