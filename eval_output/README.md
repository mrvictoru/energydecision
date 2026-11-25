# Evaluation outputs

Simulator rollouts, metrics, plots and reference caches. A selection of historical outputs is tracked; most run outputs are local and ignored by Git.

## Start here

- [Preprint reproduction guide](../docs/reproduction.md)
- [Audit evidence index](../paper/audit/README.md)
- [Artifact index](../paper/audit/artifacts/README.md)
- [Evaluation guide](../docs/evaluation_guide.md)

## Notes

- Non-DT policy rollouts are cached in `autoresearch/reference_cache/`.
- Reused physics-v2 surfaces under `physics_v2/` were used for the original v2cal evaluation.
- Current corrected-PPO aggregates live in [paper/audit/artifacts](../paper/audit/artifacts/README.md), not in a single `evaluation_summary.json` here.
- Fresh external clones cannot regenerate raw rollouts without processed AEMO data, loss CSVs, model weights and the three tuned PPO checkpoints.
