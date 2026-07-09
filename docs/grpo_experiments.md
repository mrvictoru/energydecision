# GRPO Post-Training Experiments

## Overview

Group Relative Policy Optimization (GRPO) is applied as an online fine-tuning
stage on top of a pretrained AEMO Decision Transformer. The v2 HF DT (trained
on realistic battery configurations) serves as the baseline.

## Key Finding: Battery Sizing Discrepancy (July 2026)

All early dispatch replay comparisons were **unfair** because `run_dispatch_replay`
used the dispatch station's actual battery specs from the AEMO registry
(e.g. 250 MWh for Torrens Island), while the DT ran on the evaluation config's
template battery (typically 10 MWh). This made dispatch replay look 25× more
profitable — not because the strategy was better, but because the battery was 25×
larger.

**Fix**: `autoresearch_evaluator.py` now passes the `comparison_cfg` to
`run_policy_episodes` and overrides the selection's battery capacity / max flow
with the template battery when `use_dispatch_asset_sizing=false`.
Alternatively, `use_dispatch_asset_sizing=true` matches the DT to the dispatch
station's actual battery — the recommended approach since the v2 dataset
includes comparable battery configurations (e.g. `fast_375c`: 8 MWh / 30 MW
matching Dalrymple North).

## Phase 1 GRPO Features (Implemented July 2026)

| Feature | Purpose | Implementation |
|---------|---------|---------------|
| **Degradation penalty** | Penalise battery wear beyond market cost | `deg_penalty_weight` in `GRPOTrainer` — extra weight multiplies degradation cost in the reward signal seen by GRPO |
| **Reference model sync** | Prevent KL drift over many iterations | `sync_reference_every=N` copies the current policy to the reference model every N iterations |
| **Adaptive RTG** | Keep RTG targets aligned with realised returns | EWMA of mean return from each iteration used to resample RTG prompts for the next iteration |
| **Larger groups** | Lower variance in advantage estimates | `--group-size` increased from 4 to 8 |

### Ablation Study (2 iterations, NSW1-only, 48h episodes)

| Feature | Final loss | vs baseline |
|---------|:----------:|:-----------:|
| Baseline | 0.099 | reference |
| sync_ref (every 2) | 0.078 | **−22%** |
| deg_penalty (1.5×) | **0.075** | **−25%** |
| adaptive_rtg | 0.107 | +8% (worse) |

Sync_ref and deg_penalty are independently beneficial. Adaptive RTG with
negative realised returns is counterproductive — it pulls RTG targets down.

## Dispatch-Matched Evaluation (Q4 2024, SA1 Oct+Nov, 5-min, full_fcas)

All policies evaluated on the **same dispatch-matched battery** derived
from Dalrymple North (8 MWh / 30 MW, 3.75 C).

| Policy | Profit/ep | FCAS/ep | Deg/ep | vs dispatch |
|--------|:---------:|:-------:|:------:|:-----------:|
| v2 HF DT (baseline) | $648 | $4,882 | $5,287 | — |
| Old GRPO (no Phase 1) | $4,108 | $9,209 | $5,875 | — |
| **Phase 1 GRPO** | **$5,451** | $7,962 | $2,769 | **−5.28** |
| PPO reference | $7,757 | $5,523 | $310 | −3.63 |
| Dispatch Dalrymple North | $3,663 | $1,512 | $1,371 | ref |
| Dispatch Hornsdale | $57,435 | $10,102 | $22,706 | +56.13 |
| Dispatch Torrens Island | $114,365 | $24,035 | $20,575 | +113.51 |
| FCAS rule | −$126,124 | $4,818 | $138,159 | −127.82 |

### Key Takeaways

1. **Phase 1 GRPO beats Dalrymple North dispatch** ($5,451 vs $3,663, +49%) —
   the first time a DT model surpasses real dispatch replay on the same battery.
2. **Phase 1 GRPO also beats old GRPO** ($5,451 vs $4,108, +33%) — the
   degradation penalty and reference sync are effective.
3. **Degradation is the remaining frontier** — PPO ($7,757, deg $310) achieves
   higher profit with 9× less degradation than Phase 1 GRPO (deg $2,769).
4. **FCAS learning works** — DT models earn 3–6× more FCAS than Dalrymple North
   dispatch ($4,882–$9,209 vs $1,512).
5. **Large station dispatch replays dominate** — Hornsdale and Torrens Island
   dispatch strategies, designed for 150–250 MW batteries, transfer profitably
   even to an 8 MWh battery because their FCAS-heavy bidding scales down to
   the smaller asset.

## Hyperparameter Sweep (21 experiments)

Conducted on NSW1 Jan 2024, 144h episodes, 5‑min steps.

| Sweep | Best value | Improvement | Notes |
|-------|-----------|:-----------:|-------|
| Iterations | **5** (144h) | +0.49 | 30 iter regresses (−1.54) — KL drift |
| KL coeff | 0.02 (default) | +0.44 | Higher KL hurts |
| Entropy coeff | 0.0 (default) | +0.53 | Any positive entropy is worse |
| Learning rate | 1e‑5 (144h) / 5e‑5 (24h) | +1.32 | 24h proxy does NOT predict 144h |
| RTG count | 4 (144h) / 2 (24h) | +1.32 | More RTG values dilute advantages |
| Multi‑region | NSW1+SA1+QLD1 | +1.60 | Single region +1.32 |

**Critical lesson**: 24h proxy metrics do **not** reliably predict 144h
evaluation performance. Always validate on the target episode length.

## Phase 2 (Future Work — Requires MoLab Retraining)

1. **Multi‑round self‑improvement**: Generate new rollouts from GRPO-improved
   model → append to dataset → retrain DT → repeat GRPO.
2. **Full env reward restructuring**: Change `battery_life_cost` and retrain
   SB3 models + regenerate dataset + retrain DT.
