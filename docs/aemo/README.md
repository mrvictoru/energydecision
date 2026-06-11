# AEMO documentation guide

Use this page as the entry point for the AEMO-related docs in this repository.

## Which document should I read?

| If you want to... | Read this | What it covers |
| --- | --- | --- |
| Understand the environment itself | [`environment.md`](environment.md) | Observation space, action space, reward, market data, and environment behavior |
| Run notebook-based offline RL and Decision Transformer experiments | [`workflow.md`](workflow.md) | Dataset creation, notebook flow, SB3 training, DT dataset assembly, and DT training entrypoints |
| Regenerate the additive 2024 dispatch-heavy dataset | [`recommended_data_generation.md`](recommended_data_generation.md) | One-shot generation command, notebook config, and the current generated-vs-pending dispatch replay status |
| Run or inspect dispatch replay experiments | [`dispatch-replay.md`](dispatch-replay.md) | Dispatch replay workflow, DUID discovery, sizing resolution, and API reference |
| Understand battery degradation modeling choices | [`degradation.md`](degradation.md) | Design rationale, implementation summary, and model limitations |
| See the short-term AEMO DT roadmap | [`roadmap.md`](roadmap.md) | Research priorities and near-term milestones, not day-to-day operating instructions |

## Recommended reading order

1. Start with [`environment.md`](environment.md) if you are new to the AEMO environment.
2. Read [`workflow.md`](workflow.md) if you want to generate data, train models, or reproduce the notebook-first pipeline.
3. Read [`recommended_data_generation.md`](recommended_data_generation.md) if you want the additive 2024 dataset recipe or the current replay-generation status.
4. Use [`dispatch-replay.md`](dispatch-replay.md) when your workflow needs real dispatch replay data.
5. Read [`degradation.md`](degradation.md) when you need the modeling rationale behind the degradation setup.
6. Treat [`roadmap.md`](roadmap.md) as a roadmap note for future work rather than a setup guide.

## At a glance

- **Reference docs:** `environment.md`, `dispatch-replay.md`
- **Workflow guide:** `workflow.md`
- **Generation status + recipe:** `recommended_data_generation.md`
- **Design note:** `degradation.md`
- **Roadmap note:** `roadmap.md`
- **Manual static-table fallback:** see `workflow.md` and `environment.md` for the AEMO **NEM Registration and Exemption List** download URL and local override paths
