# Documentation and workflow artifact inventory

> Historical record from the 2026-10-24 documentation audit. It is not a current
> tracking or publication inventory. For current guidance, see
> [the documentation hub](README.md), [the development guide](development.md),
> and [the repository guide](../REPOSITORY_GUIDE.md).

This inventory identifies tracked documentation used in tool-assisted or
AI-assisted workflows. It separates intended audience from authorship: an
instruction file is not evidence that an AI wrote it, and unknown provenance
does not mean human-authored.

## Tool and workflow instruction files

These files instruct tools or define constrained workflows. The audience labels
are not authorship claims. The older “local-only” labels from the 2026-10-24
audit were incorrect: all files listed here are tracked.

| Document | Purpose |
|---|---|
| [`../REPOSITORY_GUIDE.md`](../REPOSITORY_GUIDE.md) | Neutral repository guide for contributors, maintainers, and coding assistants; formerly `AGENTS.md` |
| [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) | GitHub Copilot tool instructions; hidden tool-specific file |
| [`../.opencode/command/ponytail-*.md`](../.opencode/command/) | OpenCode slash-command prompts; hidden tool-specific files |
| [`experiment_protocol.md`](experiment_protocol.md) | Constrained Decision Transformer experiment protocol; formerly `program.md` |
| [`website_maintenance.md`](website_maintenance.md) | Website maintenance guide; formerly `website_requirements.md` |
| [`control_policies.md`](control_policies.md) | Software control-policy reference; formerly `AGENT_README.md` |

## Workflow, experiment, and status records

These plans, logs, handoffs, or research-status documents can be used in
AI-assisted workflows. Their authorship or degree of AI involvement is not
established by the document itself. Check current status before following any
historical command.

| Document | Artifact type / caveat |
|---|---|
| [`FUTURE_PLAN.md`](FUTURE_PLAN.md) | Living forward plan |
| [`session_2026-08_aemo_dt_reexamination.md`](session_2026-08_aemo_dt_reexamination.md) | Session diary and handoff |
| [`aemo_research_plan.md`](aemo_research_plan.md) | Historical research plan; marked stale |
| [`aemo_dt_preferred_policy_plan.md`](aemo_dt_preferred_policy_plan.md) | Archived plan and checklist |
| [`grpo_experiments.md`](grpo_experiments.md) | Experiment log |
| [`grpo_finetuning_guide.md`](grpo_finetuning_guide.md) | Historical operational guide; GRPO did not improve the modern v2 model |
| [`aemo/real-dispatch-data.md`](aemo/real-dispatch-data.md) | Dispatch-data generation plan; check current status before following |
| [`aemo/recommended_data_generation.md`](aemo/recommended_data_generation.md) | Historical FCAS-rich dataset workflow; shipped policy uses SDP-teacher trajectories |
| [`../data_fetch_progress.md`](../data_fetch_progress.md) | Historical fetch and dataset progress; figures may be stale |
| [`../paper/audit/audit_findings.md`](../paper/audit/audit_findings.md) | Dated provenance-audit record |
| [`../paper/audit/small_n_assessment.md`](../paper/audit/small_n_assessment.md) | Historical assessment, superseded by later results |
| [`../paper/audit/artifacts/cache_data_staleness_finding.md`](../paper/audit/artifacts/cache_data_staleness_finding.md) | Dated audit finding |

## Documented AI-assisted audit work

The following documents received AI-assisted edits during the documentation
audit dated 2026-10-24. This records the edits made in that audit; it does not
claim that the full documents were AI-authored.

- [`../README.md`](../README.md), [`../REPOSITORY_GUIDE.md`](../REPOSITORY_GUIDE.md), and [`../report.md`](../report.md)
- [`FUTURE_PLAN.md`](FUTURE_PLAN.md), [`aemo_dt_preferred_policy_plan.md`](aemo_dt_preferred_policy_plan.md), [`aemo_research_plan.md`](aemo_research_plan.md), [`known_issues.md`](known_issues.md), and [`website_maintenance.md`](website_maintenance.md)
- [`../index.html`](../index.html)
- [`../paper/audit/audit_findings.md`](../paper/audit/audit_findings.md) and [`../paper/audit/small_n_assessment.md`](../paper/audit/small_n_assessment.md)
- `../paper/outreach_emails.md` (local only; git-ignored)

This inventory was also drafted with AI assistance; no claim is made about the
authorship of the other indexed documents unless stated above.

## Scope notes

- The inventory covers tracked documentation and lists hidden tool-specific
  instruction/prompt files by path. It does not inventory code, notebooks,
  configuration, datasets, or generated evaluation outputs.
- All files listed in the first table are tracked. Hidden tool-specific files
  remain in place; the former “local-only / not tracked or published” labels
  were corrected.
- Documents absent from this inventory were not identified as workflow or
  instruction artifacts from the available evidence; absence is not proof of
  human authorship.
- [`control_policies.md`](control_policies.md) documents the repository's software
  control policies. Its former filename alone did not make it an AI-agent
  instruction document.
