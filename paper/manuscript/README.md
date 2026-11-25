# Preprint source and PDF

**Distilling a Planner into a Decision Transformer for Multi-Market Battery Dispatch** is a simulator-based preprint.

Current release tag: `preprint-v2-baseline-corrected`. For current claims and the corrected PPO baseline, see the [reproduction guide](../../docs/reproduction.md) and [audit evidence index](../audit/README.md).

- [Latest checked-in PDF](draft_paper_NeurIPS_2026.pdf)
- [LaTeX manuscript](main.tex), [bibliography](references.bib) and [figures](figures/)
- [Provenance table](../audit/provenance.tsv), [audit evidence index](../audit/README.md), [audit artifacts](../audit/artifacts/) and [evaluation reproduction guide](../../docs/reproduction.md)

## Build the PDF

Use the NeurIPS 2026 LaTeX template in preprint mode. The required `neurips_2026.sty` is not tracked here. On Overleaf, keep the template style file and upload `main.tex`, `references.bib` and the complete `figures/` folder. Choose pdfLaTeX and compile `main.tex`. For a local build, put the style file beside `main.tex` and run `latexmk -pdf main.tex` from this directory.

The checked-in figures are sufficient to compile the paper. The historical figure-generation command for the extended surfaces is `python3 scripts/stagec_paper_figures_v2.py` from the repository root; it expects local evaluation inputs and is not part of building the PDF.
