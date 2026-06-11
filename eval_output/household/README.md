# eval_output/household/ — Household Environment Evaluation Results

Contains all evaluation outputs for the household solar+battery environment.

## Files

- **`baseline/`** — Baseline comparison across all algorithms (Rule, SDP, MRDP, A2C, PPO, SAC, TD3, DDPG, DT, Oracle)
- **`dt_sensitivity/`** — Decision Transformer RTG sensitivity study
- **`risk_metrics.csv`** — Tail-risk summary (VaR, CVaR at 5%) for all algorithms
- **`pairwise_summary.csv`** — Wilcoxon signed-rank test results for all algorithm pairs
- **`pairwise_significance_heatmap.svg`** — Visual heatmap of pairwise significance
- **`wilcoxon_stat_vs_pvalue.svg`** — Wilcoxon statistic vs p-value diagnostic plot
