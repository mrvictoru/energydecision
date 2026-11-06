# Agent-Facing and Workflow Artifacts

This index identifies tracked documentation that is written for AI agents, is
part of an agent workflow, or has documented AI-assisted edits. It separates
intended audience from authorship: an agent-facing prompt is not evidence that
an AI wrote it, and an unknown provenance does not mean human-authored.

## AI-agent-facing instructions and prompts

These documents direct an AI coding agent or define AI-tool commands. This is
an audience label, not an authorship claim.

| Document | Purpose |
|---|---|
| `AGENTS.md` | Repository guidance, including agent-specific workflow constraints; local-only |
| `.github/copilot-instructions.md` | Instructions for GitHub Copilot; local-only |
| `.opencode/command/ponytail-*.md` | Six OpenCode command prompts; local-only |
| `program.md` | Instructions for the autonomous autoresearch harness; local-only |
| `docs/website_requirements.md` | Website requirements written for an AI agent or developer; local-only |
| `paper/agent_audit_prompt.md` | Prompt for the documentation and website audit; local-only |

## Agent-workflow, experiment, and status records

These are plans, logs, handoffs, or research-status documents that can be used
in AI-assisted workflows. Their authorship or degree of AI involvement is not
established by the document itself.

| Document | Artifact type / caveat |
|---|---|
| [`FUTURE_PLAN.md`](FUTURE_PLAN.md) | Living forward plan |
| [`session_2026-08_aemo_dt_reexamination.md`](session_2026-08_aemo_dt_reexamination.md) | Session diary and handoff |
| [`aemo_research_plan.md`](aemo_research_plan.md) | Historical research plan; marked stale |
| [`aemo_dt_preferred_policy_plan.md`](aemo_dt_preferred_policy_plan.md) | Archived plan and checklist |
| [`grpo_experiments.md`](grpo_experiments.md) | Experiment log |
| [`grpo_finetuning_guide.md`](grpo_finetuning_guide.md) | Operational guide that also contains historical experiment context |
| [`aemo/real-dispatch-data.md`](aemo/real-dispatch-data.md) | Dispatch-data generation plan; check current status before following |
| [`aemo/recommended_data_generation.md`](aemo/recommended_data_generation.md) | Dataset-generation status and workflow notes |
| [`../data_fetch_progress.md`](../data_fetch_progress.md) | Historical fetch and dataset progress; figures may be stale |
| [`../paper/audit/audit_findings.md`](../paper/audit/audit_findings.md) | Dated provenance-audit record |
| [`../paper/audit/small_n_assessment.md`](../paper/audit/small_n_assessment.md) | Historical assessment, superseded by later results |
| [`../paper/audit/artifacts/cache_data_staleness_finding.md`](../paper/audit/artifacts/cache_data_staleness_finding.md) | Dated audit finding |
| [`../paper/manuscript/WRITING_PLAN.md`](../paper/manuscript/WRITING_PLAN.md) | Manuscript writing plan |
| [`../paper/phase2_mastery_guide.md`](../paper/phase2_mastery_guide.md) | Project handoff / work guide |

## Documented AI-assisted audit work

The following documents received AI-assisted edits during the documentation
audit dated 2026-10-24. This records the edits made in that audit; it does not
claim that the full documents were AI-authored.

- [`../README.md`](../README.md), `AGENTS.md` (local-only), and [`../report.md`](../report.md)
- [`FUTURE_PLAN.md`](FUTURE_PLAN.md), [`aemo_dt_preferred_policy_plan.md`](aemo_dt_preferred_policy_plan.md), [`aemo_research_plan.md`](aemo_research_plan.md), [`known_issues.md`](known_issues.md), and `docs/website_requirements.md` (local-only)
- [`../index.html`](../index.html)
- [`../paper/audit/audit_findings.md`](../paper/audit/audit_findings.md) and [`../paper/audit/small_n_assessment.md`](../paper/audit/small_n_assessment.md)
- [`../paper/manuscript/WRITING_PLAN.md`](../paper/manuscript/WRITING_PLAN.md), [`../paper/outreach_emails.md`](../paper/outreach_emails.md), and [`../paper/phase2_mastery_guide.md`](../paper/phase2_mastery_guide.md)
- [`../paper/audit/doc_audit_2026-10-24.md`](../paper/audit/doc_audit_2026-10-24.md), the audit report created during that work

This index was also drafted with AI assistance; no claim is made about the
authorship of the other indexed documents unless stated above.

## Scope notes

- The index covers tracked documentation and lists local-only agent
  instructions/prompts by path. It does not inventory code, notebooks,
  configuration, datasets, or generated evaluation outputs.
- Agent instructions and prompts marked local-only are intentionally not
  tracked or published; paths above identify their local purpose only.
- Documents absent from this index were not identified as agent artifacts from
  the available evidence; absence is not proof of human authorship.
- [`AGENT_README.md`](AGENT_README.md) documents the repository's software
  agents for human readers. Its filename alone does not make it an AI-agent
  instruction document.
