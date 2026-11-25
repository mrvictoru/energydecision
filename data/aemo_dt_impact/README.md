---
license: mit
tags:
  - energy
  - battery-storage
  - decision-transformer
  - offline-rl
  - aemo
  - nem
  - fcas
  - reinforcement-learning
language:
  - en
---

> **⚠️ ARCHIVED / HISTORICAL.** This is the Phase 4 impact-aware companion
> dataset and its "retrain the modern-v2 DT" plan. Both the modern-v2
> pretrained DT and the Phase 4 impact-retraining plan are historical and are
> superseded by the shipped SDP-teacher corpus (`data/aemo_dt_sdp/`) and the
> physics-v2 `v2cal` planner-distilled policy. For current claims and released
> artifacts see [`docs/reproduction.md`](../../docs/reproduction.md) and
> [`docs/huggingface/energydecision-dt-v2-sdp_README.md`](../../docs/huggingface/energydecision-dt-v2-sdp_README.md).
> Do not treat this dataset or its intended-use plan as current guidance.
>
> **Corrected publication state** (release tag
> `preprint-v2-baseline-corrected`): current DT-vs-PPO claims use three tuned
> 4M-step PPO seeds, with within-cell DT/PPO seed-mean ratios
> **1.70× / 1.20× / 1.69× / 1.60×** (dispatch's paired CI includes zero; one
> PPO seed reverses three surface means); under market impact the seed-mean
> ratio is **1.60× with 8/9 wins**, but seed-specific ratios span
> **0.91×–2.96×**. The older 200k / 204,800-step default-PPO ratios are
> historical.

# AEMO Simulated Trade — Impact-Aware Episodes

Impact-aware battery trading episodes from Australia's National Electricity
Market (AEMO/NEM), generated under an **endogenous market-impact model** for
retraining a Decision Transformer (DT) to avoid self-impact.

This is the Phase 4 companion dataset to
[`mrvictoru/AEMO_simulated_trade`](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade):
where the original is price-taking, every episode here is rolled out with a
piecewise-linear merit-order market-impact model enabled — the battery's own
energy dispatch and FCAS bids move the realized clearing prices.

## Dataset summary

| Property | Value |
|---|---|
| **Episodes** | 1,169 |
| **Rows** | 29,270,943 |
| **Regions** | NSW1, QLD1, SA1, TAS1, VIC1 |
| **Date range** | 2021–2023 (per-region windows) |
| **Battery sizes** | 8 / 50 / 150 / 250 MWh |
| **Horizons** | 12-day, 8-week, 26-week, ~9-month |
| **Degradation** | `real_world` (LFP, 30 °C) |
| **Market impact** | piecewise-linear merit-order (energy + 8 FCAS services) |
| **Action space** | `full_fcas` (1 energy + 8 FCAS bids) |

## Dataset structure

| Column | Type | Description |
|---|---|---|
| `episode_id` | `int32` | Episode index (0..1168) |
| `step` | `int64` | Step index within the episode |
| `norm_observation` | `list<float32>` (18) | Normalized observation (see below) |
| `action` | `list<float64>` (9) | `full_fcas` action: [energy dispatch, 8 FCAS bids] |
| `reward` | `float32` | Per-step reward (revenue − degradation cost) |
| `source_policy` | `string` | Trajectory source policy (see below) |

**Observation (18-dim, normalized):** `[hour_sin, hour_cos, day_sin, day_cos,
is_peak, RRP_norm, DEMAND_norm, 8×FCAS_norm, GEN_solar_pct, GEN_wind_pct, SOC]`
in the `AEMOBatteryTradingEnv` layout.

## Source policies

| Policy | Rows | Description |
|---|---|---|
| `ppo` | 11.0M | Existing SB3 PPO (online RL) under impact |
| `dt_v2` | 6.8M | Modern v2 DT (self-generated) under impact |
| `a2c` | 5.6M | Existing SB3 A2C under impact |
| `fcas_rule` | 3.0M | FCAS rule baseline under impact |
| `oracle_mi` | 2.2M | Impact-aware optimal LP (Oracle_MI) over varied sub-windows |
| `oracle_pt` | 0.7M | Price-taking optimal LP replayed under impact (failure-mode contrast) |

Episode counts by region / battery / horizon:

```
region: NSW1 232 · QLD1 232 · SA1 234 · TAS1 232 · VIC1 239
battery: b08 312 · b50 285 · b150 263 · b250 309
horizon: short 614 · medium 323 · long 163 · xlong 69
```

## Intended use

Retrain the modern v2 Decision Transformer (8×768 GQA) on impact-aware
trajectories so it learns to **moderate dispatch to avoid self-impact** and to
account for degradation over long horizons. Evaluation targets: the Phase 3
market-impact surface (identity vs. impact, batteries 8/50/150/250 MWh).

**Note on deterministic Oracle sources:** `oracle_mi`/`oracle_pt` are
deterministic LP solves, so their diversity comes from solving over distinct
14-day (or 8-week) sub-windows per region — each is a different optimal
trajectory. They are restricted to short/medium horizons (LP size).

## Generation

- Pipeline: `scripts/generate_impact_dataset.py`
  (sources: `oracle_mi`, `oracle_pt`, `ppo`, `a2c`, `dt_v2`, `fcas_rule`)
- Precomputed 2021–2023 supply curves + FCAS depth:
  `scripts/precompute_supply_curves.py`
- Observation reconstruction + assembly:
  `scripts/assemble_impact_dataset.py`
- Research context: `docs/aemo_research_plan.md` (Thread 1, Phase 4), branch
  `feature/market-impact-modeling` in
  [mrvictoru/energydecision](https://github.com/mrvictoru/energydecision)

## Related models

- [`mrvictoru/energydecision-dt-v2`](https://huggingface.co/mrvictoru/energydecision-dt-v2)
  — modern v2 pretrained DT (historical SOTA baseline at the time; superseded — see banner above)
- [`mrvictoru/energydecision-dt-v2-forecast`](https://huggingface.co/mrvictoru/energydecision-dt-v2-forecast)
  — forecast-conditioned DT (negative result)
