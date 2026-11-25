# Models directory

Model weights, checkpoints, sidecars and training manifests. Most contents are ignored by Git; public weights are hosted on Hugging Face.

## Start here

- [Preprint reproduction guide](../docs/reproduction.md)
- [Repository guide](../REPOSITORY_GUIDE.md)
- [Hugging Face card sources](../docs/huggingface/README.md)

## Public checkpoints

- Shipped physics-v2 Stage C DT: `aemo_dt_sdp_jtsoc_v2cal.pt` plus `.pt.meta.json` sidecar in [mrvictoru/energydecision-dt-v2-sdp](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp). The sidecar's `return_scale` is authoritative for inference.
- PPO retrain checkpoints under `aemo_sb3/ppo_retrain_20261125/` are local/ignored; hashes and split manifest are recorded in [the PPO retraining report](../paper/audit/ppo_retrain_20261125.md).

Verify architecture from weights and sidecars, not from prose documentation.
