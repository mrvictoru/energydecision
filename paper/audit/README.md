# Audit evidence index

This directory is the evidence trail for the preprint. It contains the current corrected-PPO baseline evidence, historical provenance records, dated findings and reproduction scripts.

Current release tag: `preprint-v2-baseline-corrected`.

## Current source of truth

| Path | Purpose |
|---|---|
| [ppo_retrain_20261125.md](ppo_retrain_20261125.md) | Full PPO retraining protocol, seed-level results and conclusions |
| [artifacts/ppo_retrain_seed_aggregate_20261125.json](artifacts/ppo_retrain_seed_aggregate_20261125.json) | Canonical three-seed aggregate used by the paper table (provenance P4/P5) |
| [artifacts/ppo_retrain_stats_20261125.json](artifacts/ppo_retrain_stats_20261125.json) | Per-seed identity-surface cells and statistics |
| [artifacts/ppo_retrain_impact_seeds_20261125.json](artifacts/ppo_retrain_impact_seeds_20261125.json) | Seed-specific impact results |
| [provenance.tsv](provenance.tsv) | Claim-by-claim map; rows P4/P5 are `CURRENT`, E3/C3–C5 are `SUPERSEDED` |
| [../../docs/reproduction.md](../../docs/reproduction.md) | Canonical reproduction guide and expected aggregate values |

Current corrected claims: DT/PPO within-cell seed-mean ratios **1.70× / 1.20× / 1.69× / 1.60×**; dispatch CI includes zero; impact seed-mean ratio **1.60×** with **8/9** wins; seed-specific impact ratios **0.91×–2.96×**.

## Historical evidence

These files are retained for provenance and are not current headline claims:

- [audit_findings.md](audit_findings.md): 2026-09-21 dated audit; superseded in scope by the 2026-11-25 PPO retrain
- [small_n_assessment.md](small_n_assessment.md): historical small-sample expansion; old-PPO numbers are historical
- [artifacts/cache_data_staleness_finding.md](artifacts/cache_data_staleness_finding.md): stale-cache finding
- [artifacts/stagec_v2cal_extended_significance.json](artifacts/stagec_v2cal_extended_significance.json): old-PPO identity artifact (provenance E3)
- [artifacts/stagec_v2cal_impact_analysis.json](artifacts/stagec_v2cal_impact_analysis.json): old-PPO impact artifact (provenance C3–C5)
- `results.json.pre_v2cal_backup`, dated logs and shell scripts under this directory

## Scripts

- `scripts/ppo_retrain_*.py` at the repository root regenerate the current artifacts and compact audit records.
- [upload_hf_cards.py](upload_hf_cards.py) uploads the local Hugging Face cards.
- Dated shell scripts (`run_extended_surfaces.sh`, `run_v2cal_impact.sh`, etc.) assume original paths; use separate output directories when rerunning.

## Navigation

- Up: [paper directory](../README.md)
- Up: [preprint reproduction guide](../../docs/reproduction.md)
- Down: [artifact index](artifacts/README.md)
