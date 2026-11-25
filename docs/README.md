# Documentation hub

Audience: contributors, maintainers, and coding assistants. This is the main navigation page for the repository. For the current preprint's numerical claims, use [the reproduction guide](reproduction.md) and [the audit evidence index](../paper/audit/README.md).

## Current claims and source of truth

- Three seeded, validation-tuned PPO checkpoints trained for 4,003,840 steps are the corrected baseline.
- DT/PPO within-cell seed-mean ratios: **1.70× / 1.20× / 1.69× / 1.60×** on standard, dispatch-matched, expanded broad-2024 and 2025 OOD.
- Dispatch's paired interval includes zero; one PPO seed reverses three surface means.
- Market impact: seed-mean ratio **1.60×** with **8/9** wins; seed-specific ratios span **0.91×–2.96×**.
- Canonical evidence: [`ppo_retrain_20261125.md`](../paper/audit/ppo_retrain_20261125.md), [`ppo_retrain_seed_aggregate_20261125.json`](../paper/audit/artifacts/ppo_retrain_seed_aggregate_20261125.json), and [`provenance.tsv`](../paper/audit/provenance.tsv) rows P4/P5.
- Current release tag: `preprint-v2-baseline-corrected`.
- Historical results under `paper/audit/`, `report.md` and older plan documents are retained for provenance, not as current claims.

## Start here

- [Project overview](../README.md): current preprint and separate household benchmark
- [Repository guide](../REPOSITORY_GUIDE.md): conventions, runtime paths, GPU telemetry, data/caching
- [Reproduce the AEMO preprint](reproduction.md): CPU-only audit check and simulator prerequisites
- [Architecture](architecture.md): repository structure
- [Development guide](development.md): setup, runtime modes, tests and contributor workflow
- [Known issues](known_issues.md): modelling caveats and stale-command history

## By goal

### Learn the codebase

- [Architecture](architecture.md)
- [Components](../COMPONENTS.md)
- [Control policies reference](control_policies.md)
- [Helper reference](HELPER_README.md)
- [DP algorithm reference](DP_ALGORITHM_README.md)

### Reproduce the AEMO track

- [Preprint reproduction guide](reproduction.md)
- [AEMO track README](aemo/README.md)
- [AEMO workflow](aemo/workflow.md)
- [AEMO environment](aemo/environment.md)
- [Evaluation guide](evaluation_guide.md)
- [Audit evidence index](../paper/audit/README.md) and [artifact index](../paper/audit/artifacts/README.md)

### Reproduce the household track

- [Household README](household/README.md)
- [Household workflow](household/workflow.md)
- [Household environment](household/environment.md)
- [Household degradation](household/degradation.md)
- [Real-data protocol](household/real_data_protocol.md)
- [Replay battery-size analysis](household/replay_battery_size_analysis.md)

### Develop or run constrained experiments

- [Development guide](development.md)
- [Experiment protocol](experiment_protocol.md)
- [Repository guide](../REPOSITORY_GUIDE.md)
- [Components](../COMPONENTS.md)

### Maintain the website

- [Website maintenance](website_maintenance.md)
- [Website source](../index.html)
- [Website assets](../website-assets/README.md)

### Prepare the preprint

- [Paper directory](../paper/README.md)
- [Manuscript source notes](../paper/manuscript/README.md)
- [arXiv checklist](../paper/arxiv_submission_checklist.md)
- [Anticipated reviewer questions](../paper/anticipated_reviewer_questions.md)
- [Plain-text abstract](../paper/arxiv_abstract_plaintext.txt)

### Maintain Hugging Face cards

- [HF card sources](huggingface/README.md)
- [Upload handoff](huggingface/UPLOAD.md)

### Historical and status documents

- [Known issues](known_issues.md)
- [Research notes and experiment history](research/README.md)
- [Forward roadmap](FUTURE_PLAN.md)
- [Documentation audit](documentation_audit.md)
- [Companion technical report](../report.md)

## Documentation conventions

- Current claims live in the reproduction guide, audit evidence index and manuscript source.
- Historical results are explicitly labelled as historical or superseded.
- Tool-specific instruction files remain under `.github/` and `.opencode/`; they are not general documentation.
- Every directory with tracked code, configs, figures or evidence has a README or index.
