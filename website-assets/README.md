# Website assets

Static assets used by the single-page site at [`index.html`](../index.html).

## Files

- [`dispatch-v2cal.json`](dispatch-v2cal.json): saved dispatch-matched rollout traces for the interactive policy viewer
- [`rollout-viewer.js`](rollout-viewer.js): rollout chart and tooltip behaviour
- [`attention-explorer.js`](attention-explorer.js): attention-token explorer

## Start here

- [Website maintenance](../docs/website_maintenance.md)
- [Website source](../index.html)
- [Export script](../scripts/export_aemo_rollout_viewer.py)
- [Documentation hub](../docs/README.md)

The PPO trace in `dispatch-v2cal.json` is the historical 204,800-step reference retained for qualitative inspection; the corrected headline table uses three tuned 4M PPO seeds described in [the reproduction guide](../docs/reproduction.md).
