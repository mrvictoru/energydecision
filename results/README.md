# Results directory

Scratch experiment ledgers. This directory is ignored by Git; the tracked historical ledger is [`results.tsv`](../results.tsv) at the repository root.

## Start here

- [Repository guide](../REPOSITORY_GUIDE.md)
- [Experiment protocol](../docs/experiment_protocol.md)
- [Documentation hub](../docs/README.md)

Each constrained autoresearch run writes a fresh ledger under `results/<tag>.tsv` with header `commit\ttrack\tmetric\tstatus\tdescription`. Never overwrite the tracked root ledger.
