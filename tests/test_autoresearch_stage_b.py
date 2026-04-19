import json
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autoresearch.stage_b import StageBEvaluator  # noqa: E402


def test_stage_b_compare_keep_and_discard(tmp_path: Path):
    benchmark = {
        "environment": "household",
        "data_dir": "data/household/logs",
        "state_dim": 12,
        "act_dim": 1,
        "max_timestep": 100,
        "primary_metric": "mean_reward",
        "higher_is_better": True,
    }
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(benchmark), encoding="utf-8")

    ev = StageBEvaluator(str(path), "household")
    keep, _ = ev.compare(
        {"guardrails_passed": True, "primary_metric_name": "mean_reward", "primary_metric_value": 2.0},
        {"primary_metric_name": "mean_reward", "primary_metric_value": 1.0},
    )
    discard, _ = ev.compare(
        {"guardrails_passed": False, "primary_metric_name": "mean_reward", "primary_metric_value": 3.0},
        {"primary_metric_name": "mean_reward", "primary_metric_value": 1.0},
    )
    assert keep == "keep"
    assert discard == "discard"


def test_stage_b_evaluate_reads_written_outputs(tmp_path: Path):
    benchmark = {
        "environment": "household",
        "data_dir": "data/household/logs",
        "state_dim": 12,
        "act_dim": 1,
        "max_timestep": 100,
        "primary_metric": "mean_reward",
        "higher_is_better": True,
    }
    bench_path = tmp_path / "bench.json"
    bench_path.write_text(json.dumps(benchmark), encoding="utf-8")

    fake = types.ModuleType("eval_household")

    def _main(argv):
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_metrics.json").write_text(json.dumps({"mean_reward": 1.23}), encoding="utf-8")
        (out_dir / "eval_summary.json").write_text(
            json.dumps(
                {
                    "primary_metric_name": "mean_reward",
                    "primary_metric_value": 1.23,
                    "guardrails_passed": True,
                }
            ),
            encoding="utf-8",
        )
        return 0

    fake.main = _main
    sys.modules["src.eval_household"] = fake
    sys.modules["eval_household"] = fake

    ev = StageBEvaluator(str(bench_path), "household")
    result = ev.evaluate(
        model_path=str(tmp_path / "model.pt"),
        model_config={"state_dim": 12, "act_dim": 1},
        rtg_value=0.0,
        return_scale=1.0,
        output_dir=str(tmp_path / "out"),
        device="cpu",
    )
    assert result["eval_summary"]["primary_metric_value"] == 1.23
