# AEMO Documentation Guide

This is the entrypoint for the grid-scale AEMO/NEM track.

Start with the [preprint and released artifacts](../../README.md#the-preprint) or the [AEMO preprint reproduction guide](../reproduction.md). For new experiments and implementation details, use the references below.

## Current claims / canonical evidence

The corrected publication state (release tag `preprint-v2-baseline-corrected`) is the source of truth, not the historical research notes below: it compares the shipped policy against three tuned 4M-step PPO seeds (within-cell DT/PPO seed-mean ratios **1.70× / 1.20× / 1.69× / 1.60×**; dispatch's paired CI includes zero; one PPO seed reverses three surface means; under impact a **1.60×** seed-mean ratio with **8/9 wins** but seed-specific ratios of **0.91×–2.96×**; older 200k / 204,800-step default-PPO ratios are historical). Use these as the canonical references:

- [Preprint reproduction guide](../reproduction.md) — downloads, corpus and dependencies for the shipped checkpoint.
- [Known issues](../known_issues.md) — evaluation-artifact and comparator caveats.
- [PPO retrain audit](../../paper/audit/README.md) — the corrected 2026-11-25 three-seed comparison.
- [Seed-aggregate artifact](../../paper/audit/artifacts/ppo_retrain_seed_aggregate_20261125.json) — the DT/PPO ratios and impact-gate result.

## Start Here

| If you want to... | Read this | What it covers |
| --- | --- | --- |
| Understand the AEMO track at a high level | [../architecture.md](../architecture.md) | Where AEMO modules, scripts, and artifacts live |
| Run a new experiment | [workflow.md](workflow.md) | Dataset creation, notebook flow, CLI entry points and training artifacts |
| Understand the environment | [environment.md](environment.md) | Observation space, action space, reward, market data and simulator behaviour |
| Understand evaluator mechanics | [evaluation guide](../evaluation_guide.md) | Evaluator configs, metrics and historical evaluation tiers |
| Work with dispatch replay | [dispatch-replay.md](dispatch-replay.md) | DUID discovery, sizing resolution, and replay helpers |
| Understand degradation modeling | [degradation.md](degradation.md) | Design rationale and implementation summary |

## Research Notes

Use these when you need time-specific experiment context rather than stable operating guidance.

- [GRPO experiments](../grpo_experiments.md) (historical)
- [AEMO research plan](../aemo_research_plan.md) (superseded)
- [recommended_data_generation.md](recommended_data_generation.md) (historical / superseded — see the banner in that file)

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
