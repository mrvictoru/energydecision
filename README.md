# energydecision

energydecision is a research codebase for battery control using historical Australian data. It includes Gymnasium simulators, planning and reinforcement-learning baselines, offline Decision Transformers, and evaluation tools. The benchmarks have different objectives, data and conclusions; their results are not comparable.

- **AEMO / grid-scale — the preprint.** Battery dispatch in Australia's National Electricity Market (NEM), with energy arbitrage, eight frequency control ancillary services (FCAS) and battery wear. A standalone Decision Transformer distilled from a planner outperformed a tuned PPO baseline on four held-out price-taking evaluations and passed a separate simulated market-impact test.
- **Household / Ausgrid — a separate benchmark.** Residential solar-battery control under retail tariffs, including a legacy Ausgrid benchmark and a modern telemetry re-baseline. See the [household guide](docs/household/README.md) and [current results](docs/household/workflow.md#h49-long-horizon-pilot).

## The preprint

**Distilling a Planner into a Decision Transformer for Multi-Market Battery Dispatch** is a simulator-based preprint, not a peer-reviewed publication or evidence of live-market readiness.

- [Latest checked-in PDF](paper/manuscript/draft_paper_NeurIPS_2026.pdf) · [LaTeX source](paper/manuscript/main.tex) · [Website source](index.html) · [GitHub Pages](https://mrvictoru.github.io/energydecision/)
- [Shipped model on Hugging Face](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp): `aemo_dt_sdp_jtsoc_v2cal.pt` and its `.pt.meta.json` sidecar. Use the sidecar's scaling and the shipped `rtg_mode="auto"`.
- [Hugging Face dataset repository](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade_sdp): `dt_trajectories_jtsoc_v2cal_conservative.parquet` is the shipped v2cal training corpus. `dt_trajectories_jtsoc_v2_conservative.parquet` is a distinct earlier public corpus.
- [Provenance table](paper/audit/provenance.tsv) · [Audit artifacts](paper/audit/artifacts/) · [Dated audit findings](paper/audit/audit_findings.md) · [Frozen environment](paper/environment.txt)

## Reproduce the paper

Start with the [CPU-only audit walkthrough](docs/reproduction.md#verify-the-market-impact-headline), which recomputes the impact result from the released matched-cell values without training, credentials or market data. The [full evaluation notes](docs/reproduction.md#rerun-the-simulator-evaluations) point to the paper configs and list the extra inputs needed to rerun simulator rollouts. See the manuscript's limitations and [known issues](docs/known_issues.md) before interpreting the results.

## Developer quick start

From the repository root, use a Python environment set up per the [development guide](docs/development.md#local-python-install). This smoke test needs neither market data nor a GPU:

```bash
python3 -m pytest tests/test_market_impact.py -q
```

| Where | What to look for |
| --- | --- |
| [`src/`](src/) | Simulators, planners, policy and learning implementations |
| [`scripts/`](scripts/), [`configs/`](configs/), [`tests/`](tests/) | Runnable workflows, experiment settings and tests |
| [`docs/`](docs/), [`paper/`](paper/) | Track guides, preprint and audit evidence |
| [`notebooks/`](notebooks/) | Optional exploratory and teaching workflows |
| [`eval_output/`](eval_output/) | Selected historical outputs; most run outputs are local and ignored by Git |

**Historical and optional:** the [companion technical report](report.md) and [experiment notes](docs/research/README.md) preserve earlier work. The v1 planner-distilled model, behaviour-cloning and fine-tuning runs, and earlier small-sample evaluations are superseded as headline evidence. The [research roadmap](docs/FUTURE_PLAN.md) describes proposed work, not completed validation.
