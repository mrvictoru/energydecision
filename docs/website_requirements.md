# Maintaining the project website

The single-page site is [`index.html`](../index.html). Serve the repository root with `python3 -m http.server` to preview the full page, including the interactive policy traces. The results tables and diagrams also work without JavaScript. No build step or chart library is required.

## Content and evidence

- Use the [LaTeX manuscript](../paper/manuscript/main.tex) for the title, method, test scope and limitations. The site describes a simulator-based preprint and not a live system.
- Check every reported result against the [provenance table](../paper/audit/provenance.tsv) and the underlying artifact. Use `v2cal_extended` for price-taking results, `v2cal_shipped_fixed_rtg0` for the market-impact gate and the retention artifact for student/teacher comparisons. See the [reproduction guide](reproduction.md) for the exact sources. Ratios divide unrounded means; A$/episode is comparable within each row, not across rows.
- Keep the results as readable HTML tables. When updating a bar in the ratio chart, update its visual value and the visible number together. Avoid duplicating numerical claims in metadata and decorative copy.
- The attention explorer illustrates head grouping and causal token visibility, not learned attention weights. Check architecture details against the [implementation](../src/decision_transformer.py) and preserve the default worked example without JavaScript.
- The [rollout explorer](../index.html#behavior) shows the shipped v2cal dispatch-matched policy traces. [`scripts/export_aemo_rollout_viewer.py`](../scripts/export_aemo_rollout_viewer.py) builds the [website JSON](../website-assets/dispatch-v2cal.json) from saved rollouts, verifying that the three policies see the same prices and their net profits match the evaluator metrics. Keep household results and earlier policy runs out of this dataset. The default vertical scale uses the central 96% with explicit clipped-point reporting; “fit every point” and physical-limit views remain available. Hover, touch and keyboard selection must expose exact values in the floating tooltip without relying on the screen-reader-only summary below the chart.
- Link to the [household benchmark](household/README.md) separately. Do not frame its results as evidence for or against the AEMO preprint.
- Check the PDF, model, corpus and audit links as a reader without a login. Do not assume a local file or a repository listing proves the exact artifact is published.

## Before publishing

Check at desktop and narrow mobile widths, in both colour schemes, and with JavaScript disabled. Check keyboard navigation, table captions and any content that needs horizontal scrolling. Check `git diff --check` and verify links and anchors. The [GitHub Pages URL](https://mrvictoru.github.io/energydecision/) is the intended public address. To use the [Pages workflow](../.github/workflows/pages.yml), select **GitHub Actions** as the source in **Settings → Pages**. The workflow publishes only the website and its linked local paper assets.
