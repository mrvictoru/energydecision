# Autoresearch pilot split

Small fixed pilot dataset for constrained Decision Transformer training experiments: 8 training and 4 validation episodes, stratified by source policy from the historical FCAS-rich corpus.

## Rebuild

```bash
python3 scripts/build_aemo_fcas_pilot_spec.py --build-pilot
```

This is experiment infrastructure, not the shipped preprint training corpus. For current paper claims, see [the reproduction guide](../../../docs/reproduction.md) and [SDP-teacher corpus](../../aemo_dt_sdp/README.md).

## Navigation

- Up: [historical FCAS-rich corpus](../README.md)
- Up: [data directory](../../README.md)
- [Experiment protocol](../../../docs/experiment_protocol.md)
