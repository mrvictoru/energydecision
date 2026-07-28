# AEMO Documentation Guide

This is the entrypoint for the grid-scale AEMO/NEM track.

If you are new to the repository, read [../README.md](../README.md), [../architecture.md](../architecture.md), and [../development.md](../development.md) first.

## Start Here

| If you want to... | Read this | What it covers |
| --- | --- | --- |
| Understand the AEMO track at a high level | [../architecture.md](../architecture.md) | Where AEMO modules, scripts, and artifacts live |
| Run the main workflow | [workflow.md](workflow.md) | Dataset creation, notebook flow, CLI entrypoints, and training artifacts |
| Understand the environment | [environment.md](environment.md) | Observation space, action space, reward, market data, and simulator behavior |
| Run held-out evaluation | [../evaluation_guide.md](../evaluation_guide.md) | Evaluator configs, metrics, and result interpretation |
| Work with dispatch replay | [dispatch-replay.md](dispatch-replay.md) | DUID discovery, sizing resolution, and replay helpers |
| Understand degradation modeling | [degradation.md](degradation.md) | Design rationale and implementation summary |

## Research Notes

Use these when you need time-specific experiment context rather than stable operating guidance.

- [../grpo_experiments.md](../grpo_experiments.md)
- [../dt_improvement_roadmap.md](../dt_improvement_roadmap.md)
- [recommended_data_generation.md](recommended_data_generation.md)

## Recommended Reading Order

1. [workflow.md](workflow.md)
2. [environment.md](environment.md)
3. [../evaluation_guide.md](../evaluation_guide.md)
4. [dispatch-replay.md](dispatch-replay.md) if your work depends on real dispatch traces
5. [degradation.md](degradation.md) if you need the battery-aging model details

## Notes

- Prefer `scripts/launch_aemo_training.py` as the canonical CLI training entrypoint.
- Treat `scripts/` as the executable surface and `src/` as reusable implementation.
- Keep volatile benchmark narratives in the research-note docs, not in workflow docs.
