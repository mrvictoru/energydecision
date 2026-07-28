# AEMO Data Generation Status

This document is a status-heavy companion to [workflow.md](workflow.md).

Use it when you need:

- the current recommended FCAS-rich dataset recipe
- a summary of which generation steps produced the main corpus
- historical context on supplementary dispatch-replay data

If you are new to the AEMO track, start with [README.md](README.md) and [workflow.md](workflow.md) first.

## Primary dataset: FCAS-rich (2021-2023)

The main training dataset is the FCAS-rich assembly. It was generated in three phases:

**Phase 1 — Fetch AEMO data** (distrobox, ~30min):
```bash
python3 scripts/fetch_aemo_region.py --region NSW1
python3 scripts/fetch_aemo_region.py --region QLD1
python3 scripts/fetch_aemo_region.py --region SA1
python3 scripts/fetch_aemo_region.py --region TAS1
python3 scripts/fetch_aemo_region.py --region VIC1
```

**Phase 2 — Generate episodes** (distrobox, GPU recommended):
```bash
python3 scripts/generate_fcas_dataset.py --policies ppo      # 905 eps
python3 scripts/generate_fcas_dataset.py --policies td3,a2c,ddpg,sac  # 300 eps each
python3 scripts/generate_fcas_dataset.py --policies fcas_rule  # 300 eps
```

**Phase 3 — Assemble dataset**:
```bash
python3 memory_safe_assemble.py
```

**Output**: `data/aemo_dt_fcas/aemo_fcas_dataset.parquet` (2,425 eps, 78.4M rows)

## Dispatch replay data (2024, supplementary)

A 2024 dispatch replay generation was attempted but produced limited results.
The dispatch replay data (only replaying what actually happened) is inherently
limited in variety — it's not realistic to generate enough episodes to form a
significant fraction of the data mix.

Generated replay files exist in the raw logs:

| Scenario | Dispatch alias | Status |
|----------|---------------|--------|
| SA1 2024 H1 | Hornsdale | Done |
| SA1 2024 H1 | Lake Bonney | Done |
| VIC1 2024 H1 | Victorian Big Battery | Done |

These can be re-added to the dataset if needed by running `memory_safe_assemble.py`
with the raw log files included. The 2024 additive plan is otherwise superseded
by the 2021-2023 FCAS dataset.

This means the file should be treated as a research-status note, not as the primary operating guide for new contributors.

## FCAS rule algorithm

`fcas_rule` extends the original rule with percentile-based FCAS bidding:
- Energy dispatch: same as original rule (charge ≤ $30, discharge ≥ $120)
- FCAS bidding: bids 1.0 on RAISEREG/LOWERREG when price exceeds p80 threshold
- $2,941 FCAS rev / ep vs $0 for old rule

See `src/decision.py`, [../AGENT_README.md](../AGENT_README.md), and [../research/README.md](../research/README.md) for related context.
