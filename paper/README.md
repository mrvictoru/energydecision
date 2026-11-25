# Paper directory

This directory contains the preprint manuscript, submission checklist, reviewer notes, outreach drafts and audit evidence for **Distilling a Planner into a Decision Transformer for Multi-Market Battery Dispatch**.

Current release tag: `preprint-v2-baseline-corrected`.

## Source of truth

- [Current claims and reproduction guide](../docs/reproduction.md)
- [Manuscript source](manuscript/main.tex) and [checked-in PDF](manuscript/draft_paper_NeurIPS_2026.pdf)
- [Audit evidence index](audit/README.md) and [artifact index](audit/artifacts/README.md)
- [Provenance table](audit/provenance.tsv), rows P4/P5
- [PPO retraining report](audit/ppo_retrain_20261125.md)

Use [the arXiv checklist](arxiv_submission_checklist.md) for submission packaging. `main.bbl` and `neurips_2026.sty` remain in the Overleaf project; no local arXiv source package is tracked here.

## Directory map

| Path | Purpose |
|---|---|
| [manuscript/](manuscript/README.md) | LaTeX source, bibliography, figures and checked-in PDF |
| [audit/](audit/README.md) | Current and historical provenance, significance and impact artifacts |
| [audit/artifacts/](audit/artifacts/README.md) | Machine-readable aggregates and per-seed statistics |
| [arxiv/](arxiv/) | arXiv packaging notes and drafts |
| [arxiv_submission_checklist.md](arxiv_submission_checklist.md) | Compile, metadata and source-package checklist |
| [anticipated_reviewer_questions.md](anticipated_reviewer_questions.md) | Short answers to likely reviewer questions |
| [outreach_emails.md](outreach_emails.md) | Draft emails; not evidence |
| [arxiv_abstract_plaintext.txt](arxiv_abstract_plaintext.txt) | Plain-text abstract for arXiv metadata |
| [environment.txt](environment.txt) | Frozen runtime environment |

## Scope notes

- The outreach emails are drafts and should not be cited as experimental evidence.
- Historical audit findings under `audit/` are labelled; use the P4/P5 rows in the provenance table for current claims.
- The public website is maintained from the repository root; see [website maintenance](../docs/website_maintenance.md).
