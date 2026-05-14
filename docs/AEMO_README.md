# AEMO documentation guide

Use this page as the entry point for the AEMO-related docs in this repository.

## Which document should I read?

| If you want to... | Read this | What it covers |
| --- | --- | --- |
| Understand the environment itself | [`AEMO_ENV_README.md`](AEMO_ENV_README.md) | Observation space, action space, reward, market data, and environment behavior |
| Run notebook-based offline RL and Decision Transformer experiments | [`AEMO_DT_WORKFLOW.md`](AEMO_DT_WORKFLOW.md) | Dataset creation, notebook flow, SB3 training, DT dataset assembly, and DT training entrypoints |
| Run or inspect dispatch replay experiments | [`AEMO_DISPATCH_UTILS.md`](AEMO_DISPATCH_UTILS.md) | Dispatch replay workflow, DUID discovery, sizing resolution, and API reference |
| Understand battery degradation modeling choices | [`AEMO_DEGRADATION_PLAN.md`](AEMO_DEGRADATION_PLAN.md) | Design rationale, implementation summary, and model limitations |
| See the short-term AEMO DT roadmap | [`AEMO_DT_SHORT_TERM_PROGRESS.md`](AEMO_DT_SHORT_TERM_PROGRESS.md) | Research priorities and near-term milestones, not day-to-day operating instructions |

## Recommended reading order

1. Start with [`AEMO_ENV_README.md`](AEMO_ENV_README.md) if you are new to the AEMO environment.
2. Read [`AEMO_DT_WORKFLOW.md`](AEMO_DT_WORKFLOW.md) if you want to generate data, train models, or reproduce the notebook-first pipeline.
3. Use [`AEMO_DISPATCH_UTILS.md`](AEMO_DISPATCH_UTILS.md) when your workflow needs real dispatch replay data.
4. Read [`AEMO_DEGRADATION_PLAN.md`](AEMO_DEGRADATION_PLAN.md) when you need the modeling rationale behind the degradation setup.
5. Treat [`AEMO_DT_SHORT_TERM_PROGRESS.md`](AEMO_DT_SHORT_TERM_PROGRESS.md) as a roadmap note for future work rather than a setup guide.

## At a glance

- **Reference docs:** `AEMO_ENV_README.md`, `AEMO_DISPATCH_UTILS.md`
- **Workflow guide:** `AEMO_DT_WORKFLOW.md`
- **Design note:** `AEMO_DEGRADATION_PLAN.md`
- **Roadmap note:** `AEMO_DT_SHORT_TERM_PROGRESS.md`
