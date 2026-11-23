# Maintaining the project website

The single-page site is [`index.html`](../index.html). Preview it locally by opening the file in a browser or serving the repository root with `python3 -m http.server`. It does not need a build step or a chart library.

## Content and evidence

- Use the [LaTeX manuscript](../paper/manuscript/main.tex) for the title, method, test scope and limitations. The site describes a simulator-based preprint and not a live system.
- Check every reported result against the [provenance table](../paper/audit/provenance.tsv) and the underlying artifact. Use `v2cal_extended` for price-taking results, `v2cal_shipped_fixed_rtg0` for the market-impact gate and the retention artifact for student/teacher comparisons. See the [reproduction guide](reproduction.md) for the exact sources. Ratios divide unrounded means; A$/episode is comparable within each row, not across rows.
- Keep the results as readable HTML tables. When updating a bar in the ratio chart, update its visual value and the visible number together. Avoid duplicating numerical claims in metadata and decorative copy.
- Link to the [household benchmark](household/README.md) separately. Do not frame its results as evidence for or against the AEMO preprint.
- Check the PDF, model, corpus and audit links as a reader without a login. Do not assume a local file or a repository listing proves the exact artifact is published.

## Before publishing

Check at desktop and narrow mobile widths, in both colour schemes, and with JavaScript disabled. Check keyboard navigation, table captions and any content that needs horizontal scrolling. Check `git diff --check` and verify links and anchors. The [GitHub Pages URL](https://mrvictoru.github.io/energydecision/) is the intended public address; confirm it is serving the current page before describing it as live.
