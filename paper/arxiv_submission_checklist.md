# arXiv submission checklist — Planner-Distilled Decision Transformers

Status: prepared 2026-10-20. Source of truth: `paper/manuscript/main.tex`,
`paper/manuscript/references.bib`, `paper/manuscript/figures/`.

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
- [ ] **`main.bbl`** (the compiled bibliography). This is the classic failure
      point: generate it on Overleaf (or with `bibtex main`) and include it, so
      citations do not render as `[?]`.
- [ ] `neurips_2026.sty` (custom style; not in TeX Live, so it **must** travel
      with the source)
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

- [ ] **Title:** "Planner-Distilled Decision Transformers Break the Behaviour-Cloning
      Ceiling in Multi-Market BESS Dispatch"
- [ ] **Abstract:** paste the plain-text version from
      `paper/arxiv_abstract_plaintext.txt` (1592 characters, under arXiv's
      ~1920-character limit). Do **not** paste LaTeX (`$...$`, `\emph{}`).
- [ ] **Comments field:** e.g. "13 pages, 6 figures, 3 tables. Preprint."
- [ ] **Authors:** Hou Nam U (decide whether to include the "(Victor)" nickname).
- [ ] **Affiliation:** GitHub URL is currently in the author block; arXiv allows
      an affiliation field — decide what to put there.
- [ ] **MSC/ACM class:** optional (cs.LG needs none).

## E. Integrity checks already done (see `paper/audit/`)

- [x] Every headline number traces to `paper/audit/provenance.tsv` and the JSON
      artifacts.
- [x] All citations verified against the source papers (claims + metadata).
- [x] No TODO/placeholder text; no unresolved refs; no duplicate labels.
- [x] Figures regenerated and placed within their sections.

## F. After announcement

- [ ] Add the arXiv ID to the GitHub README and the Hugging Face model card.
- [ ] Tag the repo (`preprint-v1`) and archive the exact source used.
- [ ] Replace the preprint with the camera-ready if a venue accepts it (keep the
      arXiv v1 for priority).
