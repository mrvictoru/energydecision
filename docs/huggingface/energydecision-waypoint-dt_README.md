---
license: mit
language:
- en
tags:
- aemo
- nem
- battery-energy-storage
- decision-transformer
- reward-to-go
- teacher-model
- planner-distillation
pretty_name: AEMO SOC-Waypoint Decision Transformer (teacher helper)
---

# AEMO SOC-Waypoint Decision Transformer (teacher helper)

A small Decision Transformer that predicts **target-SOC waypoints** for the
hierarchical planning teacher used to build the
[AEMO_simulated_trade_sdp](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade_sdp)
corpus.

> **This is NOT the shipped policy.** The shipped standalone policy is
> [energydecision-dt-v2-sdp](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp)
> (`aemo_dt_sdp_jtsoc_v2cal.pt`), which needs **no** solver and **not** this model
> at inference. This checkpoint is a **corpus-generation input** only.

## What it does

The teacher is a two-stage hierarchy:

1. **This model** predicts a coarse target-SOC waypoint trajectory (K = 8 waypoints)
   over the episode.
2. The **honest SDP executor** (`src/aemo_sdp_executor.py`) plans energy dispatch
   between waypoints under a seasonal price forecast.
3. A **greedy FCAS bidder** allocates residual headroom to the highest-priced
   raise/lower service each step.

The resulting `(observation, action, reward)` triples form the SDP-teacher corpus.

## How it was trained

- **Labels:** `data/aemo_dt_soc_oracle/aemo_soc_waypoints.parquet` — SOC waypoint
  targets produced from **perfect-foresight Oracle LP** paths.
- **Label generator (GitHub, not shipped here):**
  [`scripts/generate_soc_waypoint_targets.py`](https://github.com/mrvictoru/energydecision/blob/main/scripts/generate_soc_waypoint_targets.py).
  The label dataset is large (~1.3 GB) and is regenerated from the repo + cached
  AEMO data, so it is not hosted here.
- **Loss:** masked MSE on the 8-dimensional sigmoid waypoint output
  (see the `*_loss_surface_manifest.json`).

> Note on foresight: perfect foresight appears **only** in the waypoint *labels*.
> The corpus-generation executor itself is honest (seasonal forecast, no realized
> future prices) — see the dataset card "How it was generated".

## Files

| File | Size | sha256 | Notes |
|---|---:|---|---|
| `soc_waypoint_dt_best.pt` | 603,209,936 B | `ca082bbd363f8b2040a927741b899eeebc15f4e7ff3cb26da08c9d8962737f07` | pure `state_dict` |
| `soc_waypoint_dt_best.pt.meta.json` | 214 B | — | arch + `return_scale` sidecar |
| `soc_waypoint_dt_best.pt.metrics.json` | 203 B | — | best val loss |
| `soc_waypoint_dt_loss_surface_manifest.json` | 5.4 KB | — | full arch + training kwargs |

## Architecture

| Parameter | Value |
|---|---|
| Blocks / hidden dim | 8 / 768 |
| Attention | GQA (12 Q / 6 KV), QK-Norm |
| Positional encoding | learned timestep embedding (`rope_enabled=false`) |
| Context length | 210 |
| State / action dim | 18 / 8 (`act_dim=8` SOC waypoints) |
| Action head | sigmoid |
| `return_scale` | 1.0 (unused for waypoint regression) |

## Usage — regenerate the SDP-teacher corpus

```bash
# in the energydecision repo checkout
python3 scripts/generate_sdp_dt_trajectories.py \
  --regions NSW1 QLD1 SA1 TAS1 VIC1 \
  --horizons short medium \
  --batteries medium_1c fast_375c large_07c small_05c \
  --episodes-per-slot 8 \
  --model-path <path to>/soc_waypoint_dt_best.pt \
  --deg-cost-per-mwh 50.0 --deg-calibration 0.12 \
  --rtg-mode j_t_soc \
  --out data/aemo_dt_sdp/dt_trajectories_jtsoc_v2cal_conservative.parquet
```

The default `--model-path` is `models/aemo/dt/soc_waypoint_dt_best.pt`; download this
checkpoint there (or pass the path explicitly).

## Related

- Shipped policy: [energydecision-dt-v2-sdp](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp)
- Training corpus: [AEMO_simulated_trade_sdp](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade_sdp)
- Source repo (generators, env, executor): [mrvictoru/energydecision](https://github.com/mrvictoru/energydecision)
