# Hugging Face card sources

This directory holds the locally maintained Hugging Face model/dataset cards used by the AEMO preprint. The [upload script](../../paper/audit/upload_hf_cards.py) copies these README files to the public repositories.

## Files

| File | Purpose | Status |
|---|---|---|
| [energydecision-dt-v2-sdp_README.md](energydecision-dt-v2-sdp_README.md) | Shipped physics-v2 Stage C DT model card | Canonical; corrected three-seed PPO numbers |
| [AEMO_simulated_trade_sdp_README.md](AEMO_simulated_trade_sdp_README.md) | Shipped v2cal trajectory corpus card | Current; historical v1 corpus labelled |
| [energydecision-waypoint-dt_README.md](energydecision-waypoint-dt_README.md) | SOC-waypoint helper model card | Corpus-generation helper, not shipped policy |
| [UPLOAD.md](UPLOAD.md) | Manual upload handoff and checksums | Reference |

## Public repositories

- Model: [mrvictoru/energydecision-dt-v2-sdp](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp)
- Dataset: [mrvictoru/AEMO_simulated_trade_sdp](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade_sdp)
- Waypoint helper: [mrvictoru/energydecision-waypoint-dt](https://huggingface.co/mrvictoru/energydecision-waypoint-dt)

## Current claims

The shipped model card uses the corrected three-seed 4M PPO baseline: DT/PPO ratios 1.70× / 1.20× / 1.69× / 1.60×, dispatch CI includes zero, impact seed-mean ratio 1.60× with 8/9 wins, and seed-specific impact ratios 0.91×–2.96×. Older 200k default-PPO ratios are historical. See the [reproduction guide](../reproduction.md) and [audit evidence index](../../paper/audit/README.md).

## Navigation

- Up: [documentation hub](../README.md)
- Up: [project overview](../../README.md)
- Related: [paper directory](../../paper/README.md)
