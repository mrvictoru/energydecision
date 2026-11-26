# arXiv submission checklist — Distilling a Planner into a Decision Transformer

Status: corrected three-seed PPO baseline in place; the **final Overleaf re-render (2026-11-26 22:27 HKT)** was verified: 16 pages, no undefined references, no Type 3 fonts, no stale old-PPO ratios, PDF abstract verbatim match at 1,854 ASCII characters, and Figure 2 now reads "Teacher (non-clairvoyant, hybrid)" with no "honest" anywhere. Repo-side wording, provenance, abstract, Figure 2 and checklist fixes are applied and committed, including a trim of the `main.tex` abstract to fit arXiv's 1,920-character limit. **The checked-in PDF matches `main.tex` and all six figures** — no further re-render is needed unless a source or figure file changes. The repo-side arXiv source package is intentionally omitted: Overleaf holds `main.bbl` and `neurips_2026.sty`, and the source package should be exported from Overleaf at submission time.

Source of truth: `paper/manuscript/main.tex`, `paper/manuscript/references.bib`, `paper/manuscript/figures/`, and the PPO baseline audit at `paper/audit/ppo_retrain_20261125.md`.

## A. Before you start (do these first)

- [ ] **Account + endorsement.** A first submission to a new category may require endorsement from an established author in that category. Register with the institutional/university email if possible (often grants automatic endorsement). Request the endorsement code early — endorsers are humans.
- [ ] **Choose primary category.** Recommended primary: **cs.LG** (Machine Learning). Cross-list: **eess.SY** (Systems and Control) and/or **math.OC** (Optimization and Control). Do not cross-list more than 2–3.
- [ ] **Licence.** Repository code is MIT (`LICENSE`); choose the arXiv licence (e.g. CC BY 4.0 for maximum reuse, or the default arXiv non-exclusive licence) at submission. This is permanent.
- [ ] **Author identity.** Decide the metadata author string. The manuscript uses "Hou Nam U (Victor)"; arXiv metadata usually wants the formal name ("Hou Nam U") with the nickname omitted.
- [ ] **Check prior posting / text overlap.** If any part was posted elsewhere, declare it. arXiv runs a text-overlap check against other submissions.

## B. Source package to upload

Upload a zip/tar.gz of **LaTeX source**, not just the PDF. arXiv rejects a PDF produced from TeX.

> The Overleaf project is the build environment. `main.bbl` and `neurips_2026.sty` are intentionally not tracked locally; export the arXiv source package from Overleaf at submission time.

Required files:

- [ ] `main.tex` (renamed if arXiv requires a specific root file — it accepts any single root `.tex`) — use the 2026-11-26 refreshed version
- [ ] `references.bib`
- [ ] **`main.bbl`** (the compiled bibliography). Generated and retained in Overleaf; export it with the arXiv source package at submission time.
- [ ] `neurips_2026.sty` (custom style). Retained in Overleaf; export it with the arXiv source package at submission time.
- [ ] `figures/fig1_main_results.pdf` … `fig6_ceiling_break.pdf` (all six)
- [ ] Only these; remove `.aux`, `.log`, `.out`, `.synctex.gz`, `.bib.bak`, etc.
- [ ] No hidden files/directories (arXiv deletes them).
- [x] File names use only `a-z A-Z 0-9 _ + - . , =` (all current names comply).
- [x] No absolute paths (checked: none in `main.tex`; all figures are `figures/...` relative).

## C. Compile and policy checks

> Verified on the final 2026-11-26 22:27 HKT PDF render, compiled from the refreshed `main.tex` and the corrected Figure 2. The 2026-11-26 edits are wording/provenance/abstract plus a Figure 2 label correction (no new citations, references or numbers). Reconfirm only if a source or figure file changes again.

- [x] Compile from a **clean** Overleaf project after a baseline refresh (pdfLaTeX, BibTeX, pdfLaTeX twice).
- [x] Confirm no undefined citations/references in the rendered PDF.
- [x] No line numbers, no watermarks obstructing text, no referee comments.
- [x] 10–14 pt type, single-spaced, ≥1 inch margins (NeurIPS preprint complies).
- [x] References are complete; the GitHub/Hugging Face links resolve publicly.
- [x] Figures embedded (not linked externally).

## D. Metadata

- [ ] **Title:** "Distilling a Planner into a Decision Transformer for Multi-Market Battery Dispatch"
- [ ] **Abstract:** paste the ASCII plain-text version from `paper/arxiv_abstract_plaintext.txt` (**1,854 characters**, verified under arXiv's **1,920-character hard limit**). It is a **verbatim plain-text rendering** of the trimmed `main.tex` abstract (no abridgement, no rewording): all headline numbers, the dispatch-CI-includes-zero caveat, the seed-ordering reversal and the impact seed spread are preserved. arXiv metadata fields accept **ASCII only** — do not paste the PDF's Unicode `×`, em/en dashes or curly quotes, and do not paste LaTeX (`$...$`, `\emph{}`).
- [ ] **Comments field:** e.g. "16 pages, 6 figures, 3 tables. Preprint."
- [ ] **Authors:** Hou Nam U (decide whether to include the "(Victor)" nickname).
- [ ] **Affiliation:** GitHub URL is currently in the author block; arXiv allows an affiliation field — decide what to put there.
- [ ] **MSC/ACM class:** optional (cs.LG needs none).

### Overleaf refresh — completed 2026-11-26 (22:27 HKT final render)

The checked-in PDF (creation 2026-11-26 22:27:44 HKT) is the re-render from the refreshed `main.tex` and the corrected Figure 2. The 2026-11-26 edits changed `main.tex` (wording, "gate" → "stress test" framing, Appendix C provenance, precision fixes, and the abstract trim to 1,854 characters) and `fig2_pipeline_schematic.pdf`/`.png`; tables, citations and numerical results are unchanged. Final verification: 16 pages, no undefined references or placeholders, no Type 3 fonts, abstract verbatim match, Figure 2 label corrected, Table 1 / Section 6 numbers unchanged, no "honest" or "gate" wording anywhere.

**Figure 2 label correction.** The schematic's teacher row read "Teacher (honest, hybrid)"; it now reads "Teacher (non-clairvoyant, hybrid)" to match the figure caption and the manuscript's "non-clairvoyant" wording. The change was made in `scripts/stagec_paper_figures_v2.py`, and the figure was regenerated from the project distrobox environment (a control regeneration was byte-identical to the shipped PNG apart from this label; the new figure has no Type 3 fonts).

The steps below were completed for the final render and are retained as the re-render recipe.

1. Upload the repository versions of `main.tex`, `references.bib` and every file under `figures/` (including the corrected `fig2_pipeline_schematic.pdf`). Keep the existing `neurips_2026.sty` unless deliberately updating the style.
2. Recompile from scratch (clear cached files if available): pdfLaTeX, BibTeX, then pdfLaTeX twice.
3. Confirm the rendered abstract matches `paper/arxiv_abstract_plaintext.txt` (1,854 ASCII characters); Figure 2 reads "Teacher (non-clairvoyant, hybrid)" (no "honest"); Table 1 still shows PPO 9,708 / 28,070 / 17,214 / 19,212; Figure 1 shows the three-seed PPO mean with white seed-mean dots; Figure 4 shows the seed-sensitive market-impact stress test; Section 6 is titled "Market-Impact Stress Test"; Appendix C lists the `ppo_retrain_*_20261125.json` artifacts as current and the `stagec_v2cal_*` artifacts as superseded.
4. Download the rendered PDF; keep `main.bbl` and `neurips_2026.sty` in Overleaf for the arXiv source export. Copy the PDF back to `paper/manuscript/draft_paper_NeurIPS_2026.pdf` for the checked-in release. No local `main.bbl`/`neurips_2026.sty` copy is required.

## E. Integrity checks already done (see `paper/audit/`)

- [x] Every corrected headline number traces to `paper/audit/provenance.tsv` and the 2026-11-25 PPO aggregate artifact (claims P4/P5 are `CURRENT`; E3/C3–C5 are `SUPERSEDED`).
- [x] All citations verified against the source papers (claims + metadata).
- [x] Overleaf recompile and PDF check (final 22:27 HKT render verified 2026-11-26): no unresolved refs or TODO/placeholder text in the rendered PDF.
- [x] Figures regenerated (Figure 2 label corrected); final rendered PDF contains no Type 3 fonts.

## F. After announcement

- [ ] Add the arXiv ID to the GitHub README and the Hugging Face model card.
- [ ] **Freeze the exact submitted state with a new tag at submission time.** The `preprint-v2-baseline-corrected` tag marks the corrected paper release (`dcbf6ce`); the later docs cleanup (`d1b50e9`), the PDF renders (`ec4dc7b`, `120be1d`, `6e2b4d1`), the 2026-11-26 wording/provenance/abstract fixes, the checklist updates (`a39bddb`, plus the final one below) and the Figure 2 label fix (`0f991b3`) are **not** under that tag. Tag the exact commit that contains the final PDF at submission (current candidate: the HEAD after this checklist update).
- [ ] Replace the preprint with the camera-ready if a venue accepts it (keep the arXiv v1 for priority).

## G. Outstanding external actions (not repo-side)

- **Overleaf:** export the arXiv source zip (`main.tex`, `references.bib`, `main.bbl`, `neurips_2026.sty`, all six figures — use the corrected Figure 2). The re-render and PDF replacement are complete (Section D).
- **arXiv form:** licence, primary/cross-list categories, author string, affiliation, abstract (1,854-char ASCII file), comments field, and endorsement if required.
- **Optional release supplement:** publish or link the retrained PPO checkpoint hashes and scenario-level CSVs (`models/aemo_sb3/ppo_retrain_20261125/`, `eval_output/ppo_retrain_20261125/` are local/ignored) for full raw reproducibility.
- **Optional research improvement (not an arXiv blocker):** a 5-seed PPO study to tighten the policy ranking; dispatch CI includes zero and one seed reverses three surfaces, so no SOTA/universal-win claim is made.
