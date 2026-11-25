# Maintaining the project website

Audience: maintainers and coding assistants. The single-page site is [`index.html`](../index.html). Serve the repository root with `python3 -m http.server` to preview the full page, including the interactive policy traces. The results tables and diagrams also work without JavaScript. No build step or chart library is required.

Current release: `preprint-v2-baseline-corrected`.

## Source of truth

- [Preprint reproduction guide](reproduction.md) for current claims and artifact availability
- [Audit evidence index](../paper/audit/README.md) and [artifact index](../paper/audit/artifacts/README.md)
- [`paper/audit/provenance.tsv`](../paper/audit/provenance.tsv), rows P4 and P5
- [`ppo_retrain_seed_aggregate_20261125.json`](../paper/audit/artifacts/ppo_retrain_seed_aggregate_20261125.json) for the current identity and impact tables
- [`ppo_retrain_impact_seeds_20261125.json`](../paper/audit/artifacts/ppo_retrain_impact_seeds_20261125.json) for per-seed impact sensitivity

The older `v2cal_extended` significance artifact and `v2cal_shipped_fixed_rtg0` impact artifact are historical comparisons against the 204,800-step default PPO reference. Do not use them as current sources for website numbers.

## Content and evidence

- Use the [LaTeX manuscript](../paper/manuscript/main.tex) for the title, method, test scope and limitations. The site describes a simulator-based preprint, not a live system.
- Check every reported result against the [provenance table](../paper/audit/provenance.tsv) and the underlying P4/P5 artifacts. Current identity ratios are 1.70× / 1.20× / 1.69× / 1.60×; dispatch's paired interval includes zero; one PPO seed reverses three surface means. Impact seed-mean ratio is 1.60× with 8/9 wins, but seed-specific ratios span 0.91×–2.96×. Ratios divide unrounded means; A$/episode is comparable within each row, not across rows.
- Keep results as readable HTML tables. When updating a bar in the ratio chart, update its visual value and the visible number together. Avoid duplicating numerical claims in metadata and decorative copy.
- The attention explorer illustrates head grouping and causal token visibility, not learned attention weights. Check architecture details against [`src/decision_transformer.py`](../src/decision_transformer.py) and preserve the default worked example without JavaScript.
- The [rollout explorer](../index.html#behavior) shows historical shipped-v2cal dispatch-matched policy traces. Their PPO trace is the old 204,800-step reference retained for qualitative inspection, not the corrected three-seed 4M PPO comparator used in the headline table. [`scripts/export_aemo_rollout_viewer.py`](../scripts/export_aemo_rollout_viewer.py) builds [`website-assets/dispatch-v2cal.json`](../website-assets/dispatch-v2cal.json) from saved rollouts, verifying that the three policies see the same prices and their net profits match the evaluator metrics. Keep household results and earlier policy runs out of this dataset. The default vertical scale uses the central 96% with explicit clipped-point reporting; “fit every point” and physical-limit views remain available. Hover, touch and keyboard selection must expose exact values in the floating tooltip without relying on the screen-reader-only summary below the chart.
- Link to the [household benchmark](household/README.md) separately. Do not frame its results as evidence for or against the AEMO preprint.
- Check the PDF, model, corpus and audit links as a reader without a login. Do not assume a local file or a repository listing proves the exact artifact is published.

## Before publishing

Check at desktop and narrow mobile widths, in both colour schemes, and with JavaScript disabled. Check keyboard navigation, table captions and any content that needs horizontal scrolling. Check `git diff --check` and verify links and anchors. The [GitHub Pages URL](https://mrvictoru.github.io/energydecision/) is the intended public address. To use the [Pages workflow](../.github/workflows/pages.yml), select **GitHub Actions** as the source in **Settings → Pages**. The workflow publishes only the website and its linked local paper assets.
