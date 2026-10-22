#!/usr/bin/env bash
# Verify an Overleaf/arXiv source folder before zipping for arXiv upload.
# Usage: bash paper/arxiv/verify_package.sh /path/to/overleaf_source_dir
set -euo pipefail
DIR="${1:?usage: verify_package.sh <source-dir>}"
fail=0
need=(main.tex references.bib main.bbl neurips_2026.sty)
for f in "${need[@]}"; do
  if [[ -f "$DIR/$f" ]]; then echo "OK   $f"; else echo "MISS $f"; fail=1; fi
done
for f in fig1_main_results fig2_pipeline_schematic fig3_jtsoc_heatmap \
         fig4_impact_resilience fig5_revenue_decomposition fig6_ceiling_break; do
  if [[ -f "$DIR/figures/$f.pdf" ]]; then echo "OK   figures/$f.pdf"; else echo "MISS figures/$f.pdf"; fail=1; fi
done
# forbidden files
for pat in '*.aux' '*.log' '*.out' '*.synctex.gz' '*.bbl.bak'; do
  found=$(find "$DIR" -maxdepth 2 -name "$pat" -print -quit 2>/dev/null || true)
  [[ -n "$found" ]] && { echo "WARN remove $found"; }
done
if [[ $fail -eq 0 ]]; then
  echo "PACKAGE OK — zip with: (cd \"$DIR\" && zip -r ../arxiv_source.zip . -x '*.aux' '*.log' '*.out' '*.synctex.gz')"
else
  echo "PACKAGE INCOMPLETE — fix MISS entries above"; exit 1
fi
