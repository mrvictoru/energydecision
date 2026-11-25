# arXiv submission checklist — Distilling a Planner into a Decision Transformer

Status: baseline correction implemented and PDF render verified 2026-11-25. The repo-side arXiv source package is intentionally omitted: Overleaf holds `main.bbl` and `neurips_2026.sty`, and the source package should be exported from Overleaf at submission time.
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

- [ ] `main.tex` (renamed if arXiv requires a specific root file — it accepts any single root `.tex`)
- [ ] `references.bib`
- [ ] **`main.bbl`** (the compiled bibliography). Generated and retained in Overleaf; export it with the arXiv source package at submission time.
- [ ] `neurips_2026.sty` (custom style). Retained in Overleaf; export it with the arXiv source package at submission time.
- [ ] `figures/fig1_main_results.pdf` … `fig6_ceiling_break.pdf` (all six)
- [ ] Only these; remove `.aux`, `.log`, `.out`, `.synctex.gz`, `.bib.bak`, etc.
- [ ] No hidden files/directories (arXiv deletes them).
- [x] File names use only `a-z A-Z 0-9 _ + - . , =` (all current names comply).
- [x] No absolute paths (checked: none in `main.tex`; all figures are `figures/...` relative).

## C. Compile and policy checks

- [x] Compile from a **clean** Overleaf project after a baseline refresh (pdfLaTeX, BibTeX, pdfLaTeX twice).
- [x] Confirm no undefined citations/references in the rendered PDF.
- [x] No line numbers, no watermarks obstructing text, no referee comments.
- [x] 10–14 pt type, single-spaced, ≥1 inch margins (NeurIPS preprint complies).
- [x] References are complete; the GitHub/Hugging Face links resolve publicly.
- [x] Figures embedded (not linked externally).

## D. Metadata

- [ ] **Title:** "Distilling a Planner into a Decision Transformer for Multi-Market Battery Dispatch"
- [ ] **Abstract:** paste the corrected plain-text version from `paper/arxiv_abstract_plaintext.txt` (1,349 characters). Do **not** paste LaTeX (`$...$`, `\emph{}`). It matches the manuscript's seed-sensitive conclusion.
- [ ] **Comments field:** e.g. "16 pages, 6 figures, 3 tables. Preprint."
- [ ] **Authors:** Hou Nam U (decide whether to include the "(Victor)" nickname).
- [ ] **Affiliation:** GitHub URL is currently in the author block; arXiv allows an affiliation field — decide what to put there.
- [ ] **MSC/ACM class:** optional (cs.LG needs none).

### Overleaf refresh — completed 2026-11-25

1. Replace only `main.tex`, `references.bib` if changed, and every file under `figures/` with the repository versions. Keep the existing `neurips_2026.sty` unless deliberately updating the style.
2. Recompile from scratch (clear cached files if available): pdfLaTeX, BibTeX, then pdfLaTeX twice.
3. Confirm Table 1 shows PPO 9,708 / 28,070 / 17,214 / 19,212; Figure 4 shows the three-seed dot spread; Figure 6 describes the impact result as seed-sensitive.
4. Download the rendered PDF; keep `main.bbl` and `neurips_2026.sty` in Overleaf for the arXiv source export. Copy the PDF back to `paper/manuscript/draft_paper_NeurIPS_2026.pdf` for the checked-in release. No local `main.bbl`/`neurips_2026.sty` copy is required.

## E. Integrity checks already done (see `paper/audit/`)

- [x] Every corrected headline number traces to `paper/audit/provenance.tsv` and the 2026-11-25 PPO aggregate artifact.
- [x] All citations verified against the source papers (claims + metadata).
- [x] Overleaf recompile and PDF check: no unresolved refs or TODO/placeholder text in the rendered PDF.
- [x] Figures regenerated; rendered PDF contains no Type 3 fonts.

## F. After announcement

- [ ] Add the arXiv ID to the GitHub README and the Hugging Face model card.
- [x] Archive the exact paper state under the `preprint-v2-baseline-corrected` tag. It includes the refreshed PDF and release-hygiene changes; later documentation cleanup does not change numerical claims.
- [ ] Replace the preprint with the camera-ready if a venue accepts it (keep the arXiv v1 for priority).
