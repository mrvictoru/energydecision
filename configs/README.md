# Configuration files

Evaluator, model, impact-benchmark and training-tier configuration files. Run CLI entry points from the repository root, e.g. `python3 scripts/autoresearch_evaluator.py --evaluation-config configs/...`.

## Start here

- [Documentation hub](../docs/README.md)
- [Preprint reproduction guide](../docs/reproduction.md)
- [Evaluation guide](../docs/evaluation_guide.md)
- [Experiment protocol](../docs/experiment_protocol.md)

## Conventions

- AEMO DT evaluation configs use five-minute steps (`step_duration=0.083333`) to match the training data.
- Extended identity surfaces live in `configs/aemo_autoresearch_evaluator.sdp_teacher_*.json`.
- The mini evaluator uses `fcas_rule` as the reference baseline.
- Fixed run configs should be copied to a run-specific path outside the editable DT training surface.
- Cache prewarming: `python3 scripts/prewarm_aemo_cache.py --evaluation-config <config>`.

No linter or CI validates these configs; verify paths and hashes before long runs.
