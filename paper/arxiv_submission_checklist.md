# arXiv submission checklist — Distilling a Planner into a Decision Transformer

Status: baseline correction implemented 2026-11-25; **render still pending in Overleaf**.
Source of truth: `paper/manuscript/main.tex`, `paper/manuscript/references.bib`,
`paper/manuscript/figures/`, and the PPO baseline audit at
`paper/audit/ppo_retrain_20261125.md`.

## A. Before you start (do these first)

- [ ] **Account + endorsement.** A first submission to a new category may require
      endorsement from an established author in that category. Register with the
      institutional/university email if possible (often grants automatic
      endorsement). Request the endorsement code early — endorsers are humans.
- [ ] **Choose primary category.** Recommended primary: **cs.LG** (Machine
      Learning). Cross-list: **eess.SY** (Systems and Control) and/or
      **math.OC** (Optimization and Control). Do not cross-list more than 2–3.
- [ ] **Licence.** Pick the arXiv licence you want (e.g. CC BY 4.0 for maximum
      reuse, or the default arXiv non-exclusive licence). This is permanent.
- [ ] **Author identity.** Decide the metadata author string. The manuscript uses
      "Hou Nam U (Victor)"; arXiv metadata usually wants the formal name
      ("Hou Nam U") with the nickname omitted.
- [ ] **Check prior posting / text overlap.** If any part was posted elsewhere,
      declare it. arXiv runs a text-overlap check against other submissions.

## B. Source package to upload

Upload a zip/tar.gz of **LaTeX source**, not just the PDF. arXiv rejects a PDF
produced from TeX.

Required files:

- [ ] `main.tex` (renamed if arXiv requires a specific root file — it accepts any
      single root `.tex`)
- [ ] `references.bib`
- [ ] **`main.bbl`** (the compiled bibliography). It is not tracked locally;
      download it from Overleaf after a clean successful compile so citations do
      not render as `[?]`.
- [ ] `neurips_2026.sty` (custom style). It is also not tracked locally; retain
      the copy already in the Overleaf project or obtain the exact style used by
      the prior render before replacing the source.
- [ ] `figures/fig1_main_results.pdf` … `fig6_ceiling_break.pdf` (all six)
- [ ] Only these; remove `.aux`, `.log`, `.out`, `.synctex.gz`, `.bib.bak`, etc.
- [ ] No hidden files/directories (arXiv deletes them).
- [ ] File names use only `a-z A-Z 0-9 _ + - . , =` (all current names comply).
- [ ] No absolute paths (checked: none in `main.tex`; all figures are
      `figures/...` relative).

## C. Compile and policy checks

- [ ] Compile from a **clean** directory (delete `.aux`/`.bbl` first), pdfLaTeX.
- [ ] Confirm no undefined citations/references in the log.
- [ ] No line numbers, no watermarks obstructing text, no referee comments.
- [ ] 10–14 pt type, single-spaced, ≥1 inch margins (NeurIPS preprint complies).
- [ ] References are complete; the GitHub/Hugging Face links resolve publicly.
- [ ] Figures embedded (not linked externally).

## D. Metadata

- [ ] **Title:** "Distilling a Planner into a Decision Transformer for
      Multi-Market Battery Dispatch"
- [ ] **Abstract:** paste the corrected plain-text version from
      `paper/arxiv_abstract_plaintext.txt` (1,349 characters). Do **not** paste
      LaTeX (`$...$`, `\emph{}`). It matches the manuscript's seed-sensitive
      conclusion.
- [ ] **Comments field:** e.g. "15 pages, 6 figures, 3 tables. Preprint."
- [ ] **Authors:** Hou Nam U (decide whether to include the "(Victor)" nickname).
- [ ] **Affiliation:** GitHub URL is currently in the author block; arXiv allows
      an affiliation field — decide what to put there.
- [ ] **MSC/ACM class:** optional (cs.LG needs none).

### Overleaf refresh after this baseline correction

1. Replace only `main.tex`, `references.bib` if changed, and every file under
   `figures/` with the repository versions. Keep the existing `neurips_2026.sty`
   unless you deliberately update the style.
2. Recompile from scratch (clear cached files if available): pdfLaTeX, BibTeX,
   then pdfLaTeX twice.
3. Confirm Table 1 shows PPO 9,708 / 28,070 / 17,214 / 19,212 and Fig. 1 shows
   the three-seed dot spread. Confirm Fig. 4 says the impact result is seed-sensitive.
4. Download the rendered PDF and `main.bbl`. Copy the PDF back to
   `paper/manuscript/draft_paper_NeurIPS_2026.pdf` for the checked-in release.

## E. Integrity checks already done (see `paper/audit/`)

- [x] Every corrected headline number traces to `paper/audit/provenance.tsv`
      and the 2026-11-25 PPO aggregate artifact.
- [x] All citations verified against the source papers (claims + metadata).
- [ ] Recompile in Overleaf and confirm no unresolved refs or duplicate labels.
- [x] Figures regenerated; local PDFs embed TrueType fonts.

## F. After announcement

- [ ] Add the arXiv ID to the GitHub README and the Hugging Face model card.
- [ ] Archive the exact source used: the `preprint-v1` tag plus the
      2026-09-22 extended-surface and cache-key commits through `ea558bf`.
- [ ] Replace the preprint with the camera-ready if a venue accepts it (keep the
      arXiv v1 for priority).
