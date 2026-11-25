# Scripts

Command-line entry points for training, evaluation, data generation, caching and audit artifacts. Run from the repository root as `python3 scripts/...`; Docker Compose uses bare names from `/code/src`.

## Start here

- [Documentation hub](../docs/README.md)
- [Development guide](../docs/development.md)
- [Repository guide](../REPOSITORY_GUIDE.md)
- [Experiment protocol](../docs/experiment_protocol.md)
- [Reproduction guide](../docs/reproduction.md)

## Key groups

- DT training: `pretrain_decision_transformer.py`, `pretrain_aemo_decision_transformer.py`, `launch_aemo_training.py`
- Evaluation: `autoresearch_evaluator.py`, `prewarm_aemo_cache.py`, `ppo_retrain_*.py`
- Data generation: `generate_fcas_dataset.py`, `generate_sdp_dt_trajectories.py`, `convert_dispatch_to_episodes.py`
- Paper artifacts: `stagec_paper_figures_v2.py`, `ppo_retrain_seed_aggregate.py`, `export_aemo_rollout_viewer.py`
- Long GPU runs: `run_full_learning_baseline.sh` writes telemetry under `system_logs/`

The editable autoresearch surface is `scripts/pretrain_decision_transformer.py`; everything outside it is stable during constrained runs.
