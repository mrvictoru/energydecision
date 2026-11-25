# Retraining and re-evaluating the PPO baseline — AEMO Decision Transformer paper

**Date:** 2026-11-25
**Repository:** `/mnt/rescue-hdd/energydecision`
**Git SHA at start and end:** `81a929afa24666d8056c8afb3815e4c9822f51bf` (`main`, no commits or pushes made)
**Runtime:** Distrobox `energydecision-gpu` (Fedora 39), Python 3.10.12, PyTorch 2.9.0+cu130,
Stable-Baselines3 2.8.0, polars 1.41.0, numpy 2.2.6, scipy 1.15.3, optuna 4.8.0
**Hardware:** 1× NVIDIA RTX 2080 Ti (22 GB), 24 CPU threads (≈15 were occupied by unrelated
`rustdesk` processes throughout, which is why wall-clock times below are slower than the
isolated throughput measurement of ≈1,500 env steps/s)

## Bottom line

The shipped PPO reference is **not** the tuned 4M-step baseline the manuscript describes: it is a
204,800-step, unseeded, default-configuration `MlpPolicy` PPO. Retraining PPO properly
(seeded, ≥4M environment steps, tuned on a leakage-free pre-2024 validation split) changes the
headline comparison materially:

| Surface | DT A$/ep | old PPO | new PPO (pre-specified primary) | DT / old | DT / new |
|---|---:|---:|---:|---:|---:|
| Standard 2024 (n=30) | 16,501 | 4,303 | **8,836** | 3.84× | **1.87×** |
| Dispatch-matched (n=12) | 33,791 | 18,818 | **28,114** | 1.80× | **1.20×** (CI includes 0) |
| Expanded broad-2024 (n=30) | 29,124 | 4,350 | **13,736** | 6.70× | **2.12×** |
| 2025 OOD (n=6) | 30,791 | 4,817 | **14,922** | 6.39× | **2.06×** |

With the pre-specified primary checkpoint the DT still leads on the mean everywhere, but the
advantage is **1.2–2.1×**, not 1.8–6.7×, and on the dispatch-matched surface the paired 95% CI
includes zero. Across the three final seeds the ordering is **not stable**: seed 1 has a higher
mean profit than the DT on dispatch-matched (0.92×), expanded (0.91×) and 2025 OOD (0.89×), and
under the nine-cell market-impact gate seed 1 wins only 7/9 cells with a ratio of means of 0.913
and a CI that includes zero.

**Verdict: the manuscript's current claim is contradicted as stated ("beats PPO on all four
identity surfaces, 1.8–6.7×") and weakened where it survives.** The claim "the standalone DT is
the preferred policy" can still be defended on the pre-specified primary checkpoint, but not with
the four-surface clean sweep or the impact-gate ratios printed in the preprint. A correction is
needed before arXiv.

---

## 1. What the evaluated baseline actually was (Phase 0)

`models/aemo_sb3/ppo_aemo_fcas_model.zip`, SHA-256
`4281f8fbdeb13b9e28be48d7f02d850a9e979f5f01657830bf43e8fb622fb825` (unchanged, never overwritten):

| field | value |
|---|---|
| `num_timesteps` | 204,800 |
| `_total_timesteps` | 200,000 |
| `n_envs` / `n_steps` / `batch_size` / `n_epochs` | 4 / 2048 / 64 / 10 |
| `gamma` / `learning_rate` / `clip_range` | 0.99 / 3e-4 / `ConstantSchedule(0.2)` |
| `ent_coef` / `vf_coef` / `max_grad_norm` / `gae_lambda` | 0.0 / 0.5 / 0.5 / 0.95 |
| `policy_kwargs` | `{}` (default SB3 `MlpPolicy`, 2×64) |
| `seed` | `null` |
| `_n_updates` | 250 |

`scripts/retrain_sb3_full_fcas.py` confirms how it was produced: `N_TRIALS = 0`,
`default_model=True`, `TOTAL_TIMESTEPS = 200_000`, `sorted(CACHE_DIR.glob(...))[0]` per region,
and an `eval_env_fn` built from the *same* frame as training (no held-out split).

CPU smoke tests: `python3 -m pytest tests/test_market_impact.py tests/test_aemo_fcas_order.py -q`
→ **7 passed** (run after the modelling work; the initial attempt used `python`, which does not
exist inside the container, hence `python3` everywhere). The full `pytest tests/ -q` suite was not
re-run: no simulator or library code was modified, only new scripts, new configs and new output
directories were added.

---

## 2. Leakage-free protocol (Phase 1)

Everything below is recorded machine-readably in
`models/aemo_sb3/ppo_retrain_20261125/data_split_manifest.json` and mirrored in §11.

**Data — explicit files, no globbing** (all strictly pre-2024; the loader aborts if any file
extends past 2023-12-31):

| Region | File | Rows | Span | SHA-256 |
|---|---|---:|---|---|
| NSW1 | `processed_NSW1_2021-01-01_2023-04-01_0.0833h.parquet` | 236,160 | 2021-01-01 → 2023-04-01 | `0fbabbbd…bab9ba2f` |
| QLD1 | `processed_QLD1_2021-01-01_2023-04-01_0.0833h.parquet` | 236,160 | 2021-01-01 → 2023-04-01 | `a995f37b…aa2afb3` |
| TAS1 | `processed_TAS1_2021-01-01_2023-04-01_0.0833h.parquet` | 236,160 | 2021-01-01 → 2023-04-01 | `f28a4d14…526ca5f` |
| SA1 | `processed_SA1_2022-04-01_2023-12-01_0.0833h.parquet` | 171,984 | 2022-04-01 → 2023-12-01 | `cb0750e6…489854f3` |
| VIC1 | `processed_VIC1_2021-04-01_2023-12-01_0.0833h.parquet` | 277,103 | 2021-04-01 → 2023-12-01 | `4b4f2ef1…482a7292` |

TAS1 lacks `GEN_solar` / `GEN_solar_pct`; those columns are added as `0.0` (the same
`diagonal_relaxed` + `fill_null(0.0)` semantics as `retrain_sb3_full_fcas.py`), so all regions
produce the same 18-dim observation.

**Chronological per-region split (80/20), both halves pre-2024:**

| Region | Train window | Train rows | Validation window | Val rows |
|---|---|---:|---|---:|
| NSW1 | 2021-01-01 → 2022-10-19 | 188,928 | 2022-10-19 → 2023-04-01 | 47,232 |
| QLD1 | 2021-01-01 → 2022-10-19 | 188,928 | 2022-10-19 → 2023-04-01 | 47,232 |
| TAS1 | 2021-01-01 → 2022-10-19 | 188,928 | 2022-10-19 → 2023-04-01 | 47,232 |
| SA1 | 2022-04-01 → 2023-08-03 | 137,587 | 2023-08-03 → 2023-12-01 | 34,397 |
| VIC1 | 2021-04-01 → 2023-05-10 | 221,682 | 2023-05-10 → 2023-12-01 | 55,421 |

The loader asserts `train.SETTLEMENTDATE.max() < valid.SETTLEMENTDATE.min()` per region.

**Environment (identical to the original comparison except where noted):** 5-minute steps,
`action_mode=full_fcas` (9-dim), `degradation_mode=real_world`, LFP, 30 °C,
`market_impact=identity`, 1-week (2,016-step) training episodes with random starts inside the
train slice only, batteries `medium_1c`, `large_07c`, `small_05c`, `fast_375c`.

**Validation** (used for *all* tuning and checkpoint selection, never for the reported claims):
40 fixed episodes = 5 regions × 2 pre-specified 144 h windows (offset at 10 % and 55 % of each
validation slice) × 4 battery variants. Deterministic policy; raw profit is accumulated from
per-step `info` (`energy_revenue + fcas_revenue - degradation_cost`), cross-checked against the
env's cumulative `total_revenue`/`total_degradation_cost`.

**Two deliberate deviations from the original script, both documented:**

1. **Per-(region × battery) environments, not a concatenated frame.** `retrain_sb3_full_fcas.py`
   sorts the 5 region frames by `SETTLEMENTDATE` and concatenates them, so consecutive 5-minute
   rows alternate between regions. This driver builds 20 independent environments (5 regions ×
   4 batteries), `n_envs = 20` (the original used `n_envs=4`). The region/battery/step/action/
   degradation distributions are unchanged; only the ordering artefact is removed.
2. **Train/validation split** (see above); the original had none.

**Device:** PPO `MlpPolicy` is trained on CPU (SB3's own guidance for MLPs). The original
checkpoint was trained with `device=None` → CUDA. The algorithm and hyperparameters are
unaffected; the difference is wall-clock only.

---

## 3. Phase 2 — is the default PPO under-trained? Yes, markedly

Three seeds of the *default* configuration (`n_steps=2048`, `batch=64`, `epochs=10`,
γ=0.99, lr=3e-4, clip=0.2, no entropy, default 2×64 MLP) for 4M environment steps, with
validation at every milestone:

| steps | default s0 | default s1 | default s2 | tuned s0 | tuned s1 | tuned s2 |
|---:|---:|---:|---:|---:|---:|---:|
| ~0.25M | 7,887 | 14,497 | 11,574 | 15,098 | 21,222 | 25,053 |
| ~0.5M | 14,377 | 16,250 | 13,145 | 36,319 | 33,458 | 33,799 |
| ~1M | 17,365 | 21,056 | 20,598 | 37,629 | 42,855 | 35,245 |
| ~2M | 24,211 | 25,763 | 28,109 | 39,724 | 45,039 | 37,060 |
| ~4M | 40,271 | 40,012 | 43,452 | 41,405 | 46,668 | 37,195 |

(The shipped 204,800-step checkpoint scores **17,892** on the same 40-episode validation set.)

Learning curves (validation mean profit, A$/episode) are written to
`paper/audit/artifacts/ppo_retrain_learning_curves.csv`; the matching `.png` plot is generated
locally by `scripts/ppo_retrain_learning_curves.py` and is git-ignored.

At 4M steps the **default** configuration is worth ≈2.3× the shipped 200k checkpoint on the
validation arena. The shipped baseline is therefore substantially under-trained — the paper's
"4M-step Optuna-tuned" description does not correspond to any checkpoint evaluated.

Note that the *tuned* configuration's advantage is concentrated at 0.25M–1M steps; by 4M the
default configuration has caught up ($41.2k vs $41.8k mean across seeds), i.e. most of the gain
comes from training longer, not from the hyperparameter search.

---

## 4. Phase 3 — tuning study (validation only, 10 trials)

TPE sampler, fixed seed 42, 400k environment steps per trial, identical 40-episode pre-2024
validation set, `n_envs=20` fixed, `n_steps ∈ {512, 1024, 2048}` searched, batch 64 fixed:

| trial | score (val A$/ep) | lr | gamma | clip | ent_coef | vf_coef | n_steps | net_arch |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 16,972 | 3.97e-4 | 0.958 | 0.131 | 2.2e-8 | 0.880 | 1024 | 256×256 |
| 1 | 19,435 | 1.52e-4 | 0.959 | 0.161 | 1.4e-5 | 0.489 | 1024 | 64×64 |
| 2 | 25,920 | 6.10e-4 | 0.960 | 0.203 | 3.6e-5 | 0.142 | 512 | 400×300 |
| 3 | 15,351 | 2.02e-4 | 0.955 | 0.237 | 4.4e-6 | 0.210 | 2048 | 256×256 |
| 4 | 16,578 | 3.31e-4 | 0.977 | 0.137 | 6.6e-3 | 0.798 | 512 | 256×256 |
| 5 | 17,535 | 8.33e-4 | 0.998 | 0.286 | 1.0e-4 | 0.135 | 512 | 400×300 |
| **6** | **29,126** | **9.20e-4** | **0.976** | **0.218** | **3.3e-7** | **0.468** | **512** | **400×300** |
| 7 | 10,896 | 8.71e-4 | 0.980 | 0.249 | 2.9e-8 | 0.545 | 2048 | 400×300 |
| 8 | 22,834 | 4.96e-4 | 0.990 | 0.189 | 4.5e-7 | 0.382 | 512 | 64×64 |
| 9 | 14,340 | 1.06e-4 | 0.969 | 0.105 | 4.9e-7 | 0.697 | 512 | 400×300 |

Selected configuration (trial 6): `learning_rate=9.198e-4`, `gamma=0.97607`, `clip_range=0.21809`,
`ent_coef=3.271e-7`, `vf_coef=0.46763`, `n_steps=512`, `net_arch=[400,300]`, everything else SB3
default. Full trial records: `models/aemo_sb3/ppo_retrain_20261125/optuna/trials.json`.

No 2024/2025 data was touched: all ten trials, the learning-curve milestones and the final
checkpoint selection read only the pre-2024 validation slices above.

---

## 5. Phase 4 — final seeds and pre-specified selection

Three seeds (0, 1, 2) of the selected configuration, 4M environment steps each
(actual 4,003,840 = 391 rollouts × 10,240 samples):

| seed | checkpoint | SHA-256 | validation A$/ep |
|---|---|---|---:|
| 0 | `tuned_seed0_4003k.zip` | `bf0d9cb2…3094bf` | 41,405 |
| 1 | `tuned_seed1_4003k.zip` | `a76f0150…f043fa` | 46,668 |
| 2 | `tuned_seed2_4003k.zip` | `24565c9a…e45cb` | 37,195 |

Default-configuration 4M checkpoints (kept for the learning-curve diagnosis) are
`default_seed{0,1,2}_4014k.zip` with SHA-256 `88ee5f12…ee79ae`, `0213e9a9…2cc1566a8`,
`a2d16e66…712f2926`.

Embedded PPO metadata of the new checkpoints (read back from the `.zip`):

| checkpoint | `num_timesteps` | `n_envs` | `n_steps` | `batch_size` | `n_epochs` | `gamma` | `learning_rate` | `policy_kwargs` | `seed` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `tuned_seed0_4003k.zip` | 4,003,840 | 20 | 512 | 64 | 10 | 0.97607 | 9.198e-4 | `{"net_arch": [400, 300]}` | 0 |
| `tuned_seed1_4003k.zip` | 4,003,840 | 20 | 512 | 64 | 10 | 0.97607 | 9.198e-4 | `{"net_arch": [400, 300]}` | 1 |
| `tuned_seed2_4003k.zip` | 4,003,840 | 20 | 512 | 64 | 10 | 0.97607 | 9.198e-4 | `{"net_arch": [400, 300]}` | 2 |
| `default_seed0_4014k.zip` | 4,014,080 | 20 | 2048 | 64 | 10 | 0.99 | 3e-4 | `{}` | 0 |

**Selection rule (fixed before any test-surface number was seen):**
1. per seed, take the milestone with the best validation mean profit (all three selected the 4M
   checkpoint);
2. the paper-facing "primary" checkpoint is the seed whose selected validation profit is the
   **median** of the three — here **seed 0** (41.4k, between seed 2 at 37.2k and seed 1 at 46.7k);
3. report all three seeds on every surface; never pick a seed per surface.

---

## 6. Phase 5 — the four identity surfaces (primary checkpoint = seed 0)

All four surfaces use the unchanged shipped DT (`models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt`,
`rtg_mode=j_t_soc`, `rtg_value=0`), the unchanged scenario/battery/physics definitions, and a new
reference-cache directory per (surface, seed).

| Surface | n | DT A$/ep | new PPO A$/ep | ratio | paired diff | 95% CI | win rate | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| Standard 2024 | 30 | 16,501 | 8,836 | **1.87×** | +7,665 | [5,934, 9,599] | 0.97 | 5.6e-9 |
| Dispatch-matched | 12 | 33,791 | 28,114 | **1.20×** | +5,677 | [−4,956, 13,459] | 0.83 | 0.176 |
| Expanded broad-2024 | 30 | 29,124 | 13,736 | **2.12×** | +15,389 | [5,501, 30,751] | 0.97 | 2.8e-6 |
| 2025 OOD | 6 | 30,791 | 14,922 | **2.06×** | +15,869 | [2,491, 39,453] | 0.83 | 0.0625 |

**Seed-level results (same DT cells, three PPO checkpoints):**

| Surface | seed | DT | PPO | ratio | diff | 95% CI | win rate | p |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Standard | 0 | 16,501 | 8,836 | 1.868 | +7,665 | [5,934, 9,599] | 0.97 | 5.6e-9 |
| Standard | 1 | 16,501 | 14,180 | **1.164** | +2,321 | [−1,118, 5,203] | 0.83 | 7.3e-4 |
| Standard | 2 | 16,501 | 6,109 | 2.701 | +10,392 | [8,467, 12,512] | 1.00 | 1.9e-9 |
| Dispatch | 0 | 33,791 | 28,114 | 1.202 | +5,677 | [−4,956, 13,459] | 0.83 | 0.176 |
| Dispatch | 1 | 33,791 | 36,778 | **0.919** | −2,987 | [−19,932, 7,483] | 0.75 | 0.266 |
| Dispatch | 2 | 33,791 | 19,318 | 1.749 | +14,473 | [11,908, 17,416] | 1.00 | 4.9e-4 |
| Expanded | 0 | 29,124 | 13,736 | 2.120 | +15,389 | [5,501, 30,751] | 0.97 | 2.8e-6 |
| Expanded | 1 | 29,124 | 31,850 | **0.914** | −2,726 | [−10,040, 3,081] | 0.67 | 0.124 |
| Expanded | 2 | 29,124 | 6,057 | 4.809 | +23,068 | [12,022, 39,606] | 1.00 | 1.9e-9 |
| 2025 OOD | 0 | 30,791 | 14,922 | 2.063 | +15,869 | [2,491, 39,453] | 0.83 | 0.0625 |
| 2025 OOD | 1 | 30,791 | 34,802 | **0.885** | −4,010 | [−9,626, 644] | 0.33 | 0.563 |
| 2025 OOD | 2 | 30,791 | 7,912 | 3.892 | +22,879 | [12,639, 41,435] | 1.00 | 0.031 |

Seed-mean PPO (equal weight over the three checkpoints) gives DT/PPO ratios of
**1.70× (standard), 1.20× (dispatch), 1.69× (expanded), 1.60× (2025)**.

One extra diagnostic (not used for selection): the *default* configuration at 4M, seed 0, scores
**13,202 A$/ep** on the standard surface — better than the tuned seed-0 checkpoint (8,836)
despite a marginally lower validation score. Validation-arena rank does not transfer reliably to
the identity surfaces at this sample size; this is a further reason to report seed spread rather
than a single checkpoint.

### Scenario/identity verification

The canonical numbers are reproduced exactly, which proves that the new runs share scenarios,
batteries, five-minute steps, degradation and normalisation with the shipped evaluation:

* standard: DT `16,501.136354` and PPO `4,302.678864` — both **bit-identical** to
  `eval_output/physics_v2/standard_year_v2cal` (two independent re-runs: with
  `fit_global_stats=true` and with the cached frames).
* dispatch: DT `33,790.62` (12 cells) and the Dalrymple replay `19,326.27` — identical to
  `dispatch_year_v2cal` when the surface's own global stats are materialised
  (`fit_global_stats=true`).
* expanded / 2025: two-scenario spot checks reproduced the shipped DT cells to six decimals
  (`210,015.658053`, `21,422.968994`; `14,966.877442`, `97,195.685014`). The full 150-episode
  expanded DT re-rollout was skipped for compute reasons (≈6 h) and the shipped DT cells are
  reused; the spot checks are the evidence that they are current.

---

## 7. Phase 6 — nine-cell market-impact gate

Re-run with `scripts/ppo_retrain_impact_eval.py` (same scenarios, batteries, piecewise
merit-order model, `rtg_mode='auto'`, `rtg_value=0`), writing to
`eval_output/ppo_retrain_20261125/phase3_impact/results.json`. The canonical
`eval_output/phase3_impact/results.json` was **not** modified; the DT leg reproduces it exactly
(e.g. `identity … small_sa1_oct_2024 = 72,300` and `piecewise … small_sa1_oct_2024 = 49,277`).

**Primary checkpoint (tuned seed 0):**

| | old PPO | new PPO (s0) |
|---|---:|---:|
| cells won by DT | 9/9 | **9/9** |
| ratio of means | 4.36 / 4.71 / 4.67 (small/Hornsdale/Torrens) | **1.87 / 2.22 / 2.37** |
| paired mean advantage | A$167,099 | **A$118,636** |
| 95% bootstrap CI | [101,213, 227,362] | **[61,166, 183,971]** |
| Wilcoxon p | 0.0039 | **0.0039** |

| battery | scenario | DT | new PPO | ratio |
|---|---|---:|---:|---:|
| Hornsdale | sa1_oct_2024 | 300,768 | 79,371 | 3.79 |
| Hornsdale | sa1_nov_2024 | 273,778 | 192,725 | 1.42 |
| Hornsdale | vic1_oct_2024 | 223,476 | 87,689 | 2.55 |
| small | sa1_oct_2024 | 49,277 | 22,576 | 2.18 |
| small | sa1_nov_2024 | 39,585 | 23,985 | 1.65 |
| small | vic1_oct_2024 | 55,432 | 30,596 | 1.81 |
| Torrens | sa1_oct_2024 | 391,056 | 89,620 | 4.36 |
| Torrens | sa1_nov_2024 | 305,743 | 213,345 | 1.43 |
| Torrens | vic1_oct_2024 | 276,008 | 107,494 | 2.57 |

**Seed sensitivity of the gate (this is the important part):**

| checkpoint | DT wins | ratio of means | paired diff | 95% CI | per-battery ratio (small/Hornsdale/Torrens) |
|---|---:|---:|---:|---|---|
| tuned seed 0 (primary) | 9/9 | 2.260 | +118,636 | [61,166, 183,971] | 1.87 / 2.22 / 2.37 |
| tuned seed 1 | **7/9** | **0.913** | −20,353 | **[−138,710, 79,174]** | 1.62 / 0.86 / 0.90 |
| tuned seed 2 | 9/9 | 2.961 | +140,924 | [81,193, 198,816] | 1.99 / 2.93 / 3.23 |
| default config, seed 0 | 9/9 | 4.379 | +164,200 | [98,858, 226,651] | 4.05 / 4.46 / 4.37 |

With seed 1 the PPO beats the DT on two SA1-November cells by roughly 2× (Hornsdale
A$605k vs A$274k, Torrens A$616k vs A$306k), which flips the ratio below 1 and widens the CI
across zero. Conversely, the *untuned* default configuration at 4M reproduces the paper's
original 4.4× ratio almost exactly (4.379; per battery 4.05 / 4.46 / 4.37). Across four PPO
checkpoints trained to the same 4M budget the gate ratio therefore spans **0.91–4.38** while the
DT cells are fixed, so the gate measures PPO checkpoint variance at least as much as a DT
property. A 3-scenario cluster bootstrap gives the same picture ([63,017, 183,178] for the
primary because all three scenario means are positive; the seed-1 CI includes zero).

The gate is therefore **not** a 9/9-with-huge-margins result under a properly trained PPO; it is
"the DT wins on the primary checkpoint with a roughly 2.3× advantage, but the outcome is
checkpoint-dependent".

---

## 8. Old versus new PPO

| Surface | old PPO (200k, default) | new PPO seed 0 | new / old (seed 0) | new / old (seed mean) | DT / old | DT / new (seed 0) |
|---|---:|---:|---:|---:|---:|---:|
| Standard 2024 | 4,303 | 8,836 | 2.05× | 2.26× | 3.84× | 1.87× |
| Dispatch-matched | 18,818 | 28,114 | 1.49× | 1.49× | 1.80× | 1.20× |
| Expanded broad-2024 | 4,350 | 13,736 | 3.16× | 3.96× | 6.70× | 2.12× |
| 2025 OOD | 4,817 | 14,922 | 3.10× | 3.99× | 6.39× | 2.06× |

On the validation arena the shipped checkpoint scores 17,892 A$/ep against 41,405–46,668 for the
4M checkpoints (2.3–2.6×). The original PPO numbers were therefore not "the DT beating tuned
PPO"; they were the DT beating a short, unseeded, default run.

In-distribution, on the same pre-2024 arena, the shipped DT scores **37,981 A$/ep on the tested
`medium_1c` battery** (10 episodes) against 9,402 for the shipped 200k PPO, 19,579–25,529 for the
three tuned 4M PPOs and 21,529–23,196 for the three default 4M PPOs on the same battery — i.e.
≈1.7× the tuned PPO and ≈4.0× the shipped checkpoint. That ordering is stable on the validation
arena; it is the transfer to the 2024/2025 identity surfaces and to the impact gate that is noisy.

---

## 9. Statistics (Phase 7)

Method: for each surface, paired cells `(scenario, battery)`; ratio of means; paired DT−PPO
difference with a 10,000-resample percentile bootstrap 95% CI (seed 42); paired Wilcoxon
signed-rank p; per-cell win rate. Identical to `paper/audit/reproduce_significance.py` and
`paper/audit/analyse_v2cal_impact.py`.

Two cluster-aware sensitivities were added because the shipped cell-level CI treats correlated
cells as independent:

* **region-cluster bootstrap** — resample the 5 NEM regions with replacement, keeping every cell
  of a drawn region;
* **scenario-block bootstrap** — resample `(region, month)` blocks, keeping all episodes of a
  block together (this is the effective n for the 5-episode expanded/2025 surfaces);
* impact gate: the battery scale is treated as a crossed factor (per-battery ratio of means plus
  a scenario-cluster bootstrap over the 3 SA1/VIC1 windows).

| Surface (primary seed) | cell-level CI | region-cluster CI | scenario-block CI |
|---|---|---|---|
| Standard | [5,934, 9,599] | [5,963, 9,662] | [5,974, 9,564] |
| Dispatch | [−4,956, 13,459] | degenerate (1 SA1 cluster) | [−4,851, 13,485] |
| Expanded | [5,501, 30,751] | [5,401, 32,037] | [5,133, 30,032] |
| 2025 OOD | [2,491, 39,453] | [3,286, 40,864] | [2,325, 39,662] |

The cluster-aware CIs are 1–8 % wider than the cell-level CIs and never move a conclusion.

Machine-readable results: `paper/audit/artifacts/ppo_retrain_stats_20261125.json` and
`paper/audit/artifacts/ppo_retrain_20261125.json`.

---

## 10. Limitations

1. **Seed variance dominates the test-surface ranking.** Validation profit (40 episodes) does not
   reliably predict 2024/2025 identity-surface profit: the median-validation checkpoint is mid-pack
   on the standard surface, and the default configuration at 4M beats the tuned one there. The
   seed spread (e.g. standard 6.1k–14.2k) is larger than most of the measured effects.
2. **Only three final seeds.** Three seeds bound the spread but cannot characterise its tail; a
   5-seed study would be better. The impact gate in particular is a 9-cell, single-checkpoint
   measurement: the ratio of means moves from 0.91 to 4.38 across four PPO checkpoints with the
   same 4M budget, so that gate should be reported with a checkpoint distribution, not a point
   estimate.
3. **The expanded surface reuses shipped DT cells** (150 episodes × 3,456 steps ≈ 5 h at the
   observed 40 ms/step under load). Two independent spot checks reproduced the shipped values
   exactly, but the full 150-episode DT leg was not re-run.
4. **Processed-data normalisation is shared across surfaces.** The 2024 processed parquet files
   are shared between the standard, dispatch and expanded surfaces, and `fit_global_stats=true`
   rewrites them with surface-specific statistics. The dispatch surface was therefore run last
   and with its own statistics (`fit_global_stats=true`); re-running the standard surface now
   would read SA1 files normalised with dispatch-only statistics. This is a pre-existing hazard
   (see `paper/audit/artifacts/cache_data_staleness_finding.md`), not something introduced
   here.
5. **Training/validation normalisation.** The processed frames carry min–max normalisation fitted
   over each file's full pre-2024 span, so validation rows influenced the scaling used in
   training. No 2024/2025 data was involved; the effect is limited to observation scaling.
6. **Random-start surfaces are not seed-paired across policies.** For expanded/2025 the evaluator
   draws 5 random start windows per scenario; the DT uses `seed=42+index` while the SB3 path does
   not seed its envs (pre-existing evaluator behaviour). Both policies therefore see the same
   scenario but not the same five windows.
7. **Compute.** The 4M-step budget was met for all six final/default seeds, but the machine was
   heavily loaded by unrelated processes (~15 `rustdesk` threads at 100 % CPU), so wall-clock
   times are not a throughput benchmark.

---

## 11. Exact commands

```bash
# 0. environment / baseline facts
cd /mnt/rescue-hdd/energydecision
git rev-parse HEAD && git status --short
sha256sum models/aemo_sb3/ppo_aemo_fcas_model.zip
python3 - <<'PY'
import zipfile, json
with zipfile.ZipFile("models/aemo_sb3/ppo_aemo_fcas_model.zip") as z:
    d = json.loads(z.read("data"))
print({k: d.get(k) for k in ["num_timesteps","_total_timesteps","n_envs","n_steps",
      "batch_size","n_epochs","gamma","learning_rate","policy_kwargs","seed","_n_updates"]})
PY
python3 -m pytest tests/test_market_impact.py tests/test_aemo_fcas_order.py -q

# 1. protocol manifest (data hashes, split, validation windows)
python3 scripts/ppo_retrain_aemo.py manifest \
  --out-dir models/aemo_sb3/ppo_retrain_20261125

# 2. default PPO learning curves (3 seeds x 4M, milestone checkpoints + validation)
for s in 0 1 2; do
  python3 scripts/ppo_retrain_aemo.py train \
    --out-dir models/aemo_sb3/ppo_retrain_20261125 \
    --tag default_seed${s} --config default --seed ${s} --timesteps 4000000 \
    --milestones 250000 500000 1000000 2000000 4000000
done

# 3. tuning on the pre-2024 validation set only
python3 scripts/ppo_retrain_aemo.py tune \
  --out-dir models/aemo_sb3/ppo_retrain_20261125 --trials 10 --timesteps 400000 --seed 0

# 4. final seeds with the selected configuration
for s in 0 1 2; do
  python3 scripts/ppo_retrain_aemo.py train \
    --out-dir models/aemo_sb3/ppo_retrain_20261125 --tag tuned_seed${s} \
    --config-json models/aemo_sb3/ppo_retrain_20261125/optuna/best_params.json \
    --seed ${s} --timesteps 4000000 --milestones 250000 500000 1000000 2000000 4000000
done

# 4b. validation references (learning-curve context)
python3 scripts/ppo_retrain_aemo.py validate \
  --out-dir models/aemo_sb3/ppo_retrain_20261125 \
  --model models/aemo_sb3/ppo_aemo_fcas_model.zip \
  --out models/aemo_sb3/ppo_retrain_20261125/validate_old_ppo.json

# 5. read-only data mirror so the evaluator cannot rewrite data/aemo
mkdir -p data/aemo_eval_mirror_20261125
find data/aemo -maxdepth 1 -type f ! -name 'processed_*.parquet' \
  -exec ln -f {} data/aemo_eval_mirror_20261125/ \;
cp data/aemo/processed_*.parquet data/aemo_eval_mirror_20261125/
for d in fetch_logs manual _nemosis_static; do
  ln -sfn "$PWD/data/aemo/$d" "data/aemo_eval_mirror_20261125/$d"
done

# 5b. all four surfaces x three seeds (writes only under eval_output/ppo_retrain_20261125)
python3 scripts/ppo_retrain_run_evaluations.py \
  --run-dir models/aemo_sb3/ppo_retrain_20261125 \
  --eval-root eval_output/ppo_retrain_20261125 \
  --seeds tuned_seed0 tuned_seed1 tuned_seed2

# 6. market-impact gate with the new PPO (canonical results.json untouched)
python3 scripts/ppo_retrain_impact_eval.py \
  --ppo-model models/aemo_sb3/ppo_retrain_20261125/tuned_seed0_4003k.zip \
  --dt-checkpoint models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt \
  --dt-config models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json \
  --dt-label stagec_v2cal_auto \
  --results-path eval_output/ppo_retrain_20261125/phase3_impact/results.json \
  --policies dt,ppo --rtg 0.0 --rtg-mode auto

# 7. statistics + artifacts
python3 scripts/ppo_retrain_statistics.py \
  --manifest paper/audit/artifacts/ppo_retrain_manifest_stats_20261125.json \
  --out paper/audit/artifacts/ppo_retrain_stats_20261125.json \
  --impact-results eval_output/ppo_retrain_20261125/phase3_impact/results.json
python3 scripts/ppo_retrain_learning_curves.py \
  --models-dir models/aemo_sb3/ppo_retrain_20261125 \
  --out-dir paper/audit/artifacts \
  --old-ppo-validation models/aemo_sb3/ppo_retrain_20261125/validate_old_ppo.json
python3 scripts/ppo_retrain_build_artifact.py \
  --run-dir models/aemo_sb3/ppo_retrain_20261125 \
  --stats paper/audit/artifacts/ppo_retrain_stats_20261125.json \
  --out paper/audit/artifacts/ppo_retrain_20261125.json
```

All commands were run inside the Distrobox runtime:
`distrobox enter energydecision-gpu -- <command>`.

---

## 12. Artefacts and hashes

New artefacts (nothing pre-existing was modified):

| Path | Contents |
|---|---|
| `models/aemo_sb3/ppo_retrain_20261125/` | 30 milestone checkpoints (six of them the 4M finals), `*_history.json`, `*_meta.json`, per-milestone validation JSON, `optuna/`, `data_split_manifest.json` |
| `eval_output/ppo_retrain_20261125/` | four surfaces × three seeds, `phase3_impact/results*.json`, spot checks, `checkpoint_selection.json` |
| `paper/audit/artifacts/ppo_retrain_20261125.json` | machine-readable result file |
| `paper/audit/artifacts/ppo_retrain_stats_20261125.json` | per-surface and per-seed paired statistics |
| `paper/audit/artifacts/ppo_retrain_impact_seeds_20261125.json` | impact gate per seed + seed-1 cell detail |
| `paper/audit/artifacts/ppo_retrain_learning_curves.csv` | learning curves (the `.png` plot is local-only, git-ignored) |
| `scripts/ppo_retrain_*.py` | protocol driver, config generator, surface runner, impact runner, statistics, learning curves, artifact builder |
| `configs/aemo_autoresearch_evaluator.ppo_retrain_*.json` | copies of the canonical configs pointing at the new checkpoints (canonical configs untouched) |
| `data/aemo_eval_mirror_20261125/` | hardlink/copy mirror used as `heldout.cache_dir`; `data/aemo/processed_*.parquet` was never rewritten by these runs |

Model hashes are listed in §5 and in `ppo_retrain_20261125.json`; data hashes in §2.

---

## 13. Conclusion

* **Supported:** with a properly trained, seeded, validation-selected PPO, the shipped DT still
  has the higher mean profit on all four identity surfaces for the pre-specified primary
  checkpoint (1.20–2.12×), and still wins 9/9 impact-gate cells (2.26× ratio of means).
* **Weakened:** the magnitude claims. The preprint's 3.84× / 1.80× / 6.70× / 6.39× and the
  4.36/4.70/4.67 impact ratios are artefacts of an under-trained baseline; the corresponding
  numbers against a 4M-step PPO are 1.87× / 1.20× / 2.12× / 2.06× and 1.87/2.22/2.37 for the
  pre-specified checkpoint. The impact ratio is extremely checkpoint-sensitive (0.91–4.38 across
  four 4M-step PPO checkpoints), so the 4.4× gate result is reproducible but not a stable
  property of the comparison.
* **Contradicted as stated:** "beats PPO on all four identity surfaces" and the impact-gate 9/9
  sweep are not robust to the PPO seed. Seed 1 beats the DT on dispatch-matched (0.92×),
  expanded (0.91×) and 2025 OOD (0.89×), and wins 2/9 impact cells (ratio 0.913, CI includes
  zero). A claim that survives must be phrased over the seed distribution, e.g. "the DT matches
  or exceeds a 4M-step PPO on average, with the advantage concentrated in the standard and
  expanded surfaces and disappearing on the dispatch-matched and impact surfaces".
* **Also contradicted:** the manuscript's description of the comparator as an Optuna-tuned
  4M-step PPO. The evaluated checkpoint is a 200k-step, unseeded default configuration.

**Recommendation:** correct the manuscript before arXiv — update the protocol description, report
the retrained PPO baseline with seed spread and cluster-aware CIs, and soften the four-surface
and impact-gate claims accordingly. No manuscript, provenance or shipped-artefact file was
modified here; the changes above are proposals for a separate revision.
