# Historical GRPO v3 combined corpus

> **ARCHIVED / historical.** This directory contains a behaviour-cloning-era dataset used during GRPO experiments. It is not the shipped Stage C training corpus. GRPO did not improve the modern v2 model, and the shipped policy uses [SDP-teacher trajectories](../aemo_dt_sdp/README.md).

- [`dataset_card.md`](dataset_card.md): schema, composition and historical generation notes
- Generation commands that referenced `src/generate_fcas_dataset.py` or a missing `generate_grpo_episodes.py` are historical; current entry points live under [`scripts/`](../../scripts/README.md) or are documented in [docs/grpo_experiments.md](../../docs/grpo_experiments.md).
- The claim that this data “captures improved FCAS bidding behaviour” is historical/experimental only.

## Navigation

- Up: [data directory](../README.md)
- Current claims: [preprint reproduction guide](../../docs/reproduction.md)
- Historical GRPO context: [GRPO fine-tuning guide](../../docs/grpo_finetuning_guide.md)
