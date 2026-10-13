# Manuscript scaffold — how to build the preprint on Overleaf

This directory holds the LaTeX **skeleton** only. You (the author) write the prose;
the scaffold fixes the structure, the claim map, and the traced numbers.

```
paper/manuscript/
  main.tex        # skeleton: sections + placeholders + [FACT] table
  references.bib  # seed bibliography (VERIFY each entry before submitting)
  README.md       # this file
```

---

## 1. One-time Overleaf setup (~5 min)

1. Overleaf → **New Project → Template** → search **"NeurIPS 2024"** (or the newest
   NeurIPS template available) → **Create**.
   - This gives you the official `neurips_*.sty`. If the template ships a different
     year, open our `main.tex` and change `\usepackage[preprint]{neurips_2024}` to
     match the `.sty` you actually have (e.g. `neurips_2023`).
2. In the new project, **replace** the template's `main.tex` with ours, and
   **replace** `references.bib` with ours (Overleaf: upload, then rename/overwrite).
   Keep the template's `.sty` file — do not delete it.
3. Menu → **Compiler: pdfLaTeX**.
4. Click **Recompile**. You should get a compilable skeleton with a headline table
   and placeholder figure boxes.

> Prefer to keep the repo the source of truth: upload `main.tex` + `references.bib`
> from here, edit on Overleaf during the writing sprint, then download and commit.

---

## 2. LaTeX crash course (only what this paper needs)

| You want to… | Write |
|---|---|
| a section / subsection | `\section{Method}` / `\subsection{Teacher}` |
| bold / italic | `\textbf{...}` / `\textit{...}` |
| a bullet list | `\begin{itemize} \item ... \end{itemize}` |
| cite a reference | `\cite{chen2021decision}` (keys are in `references.bib`) |
| a table | copy the `table` block in `main.tex`; `booktabs` gives `\toprule \midrule \bottomrule` |
| a figure | uncomment/replace `\figph{...}` with `\includegraphics[width=\linewidth]{figures/fig1_main_results.pdf}` |
| an equation | `\begin{equation} L = w_a L_a + w_s L_s + w_r L_r \end{equation}` |
| a cross-reference | label it `\label{sec:method}` then `\ref{sec:method}` / `\autoref{}` |
| math inline | `$J_t(\mathrm{soc})$` |
| thousands in numbers | `16{,}501` (the `{,}` stops the space) |
| a comment | everything after `%` |

**Compile early and often.** A red error is normal; Overleaf shows the line number.
If a figure/table floats to a weird place, that's normal — LaTeX places floats.

---

## 3. Rules for this paper (anti-error)

1. **Never retype a number.** Every `[FACT]` value lives in
   `paper/audit/provenance.tsv`; copy from there and re-check the artifact.
2. **Verify every citation** against the original before submission. `references.bib`
   is transcribed from `report.md` and marked `TODO(author): VERIFY`.
3. **Every claim sentence should map to a claim:** C1 (cloning ceiling), C2
   (distillation + J_t(soc)), C3 (market-impact gate). If a sentence doesn't serve
   one of the three, cut it.
4. **Write the limitations yourself, unprompted** — see `\section{Limitations}`.
5. **Disclose AI assistance** in the `ack` block per venue policy.

---

## 4. Figures

Generated from the **extended-surface** artifacts by
`scripts/stagec_paper_figures_v2.py` (supersedes the v1 `stagec_paper_figures.py`):

| File | Content |
|---|---|
| `figures/fig1_main_results.pdf` | DT vs PPO across the 4 surfaces + paired-Δ forest plot |
| `figures/fig5_revenue_decomposition.pdf` | energy / FCAS / degradation decomposition (DT vs PPO) |
| `figures/fig4_impact_resilience.pdf` | market-impact gate, per-battery DT vs PPO |

Regenerate with `python3 scripts/stagec_paper_figures_v2.py`.
`main.tex` now uses `\includegraphics{figures/...}`, so **upload the whole
`figures/` folder to Overleaf**, or compilation will fail with "file not found".

Optional / not yet made: a J_t(soc) heatmap (needs a cost-to-go table build) and a
ceiling-break figure (historical stage ladder). Add only if space allows.

---

## 5. When the draft is ready

Run the Phase 6 checklist (in `paper/phase2_mastery_guide.md` → "When you're done"):
repro one-command eval on a fresh clone, arXiv endorsement check, tag the release.
