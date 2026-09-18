# Household Workflow Guide

This is the main workflow guide for the household solar-battery track.

Use this document when you need:

- the end-to-end household workflow
- the main notebooks for data generation, RL training, and evaluation
- the canonical CLI entrypoint for household Decision Transformer training
- the expected artifact locations for logs, models, and evaluation output

If you only need environment mechanics, read [environment.md](environment.md). If you want the full household docs map, start with [README.md](README.md).

> **Rebaseline required (2026-09-14):** The numerical household results below
> were produced before the A1-A4 physics corrections: round-trip efficiency,
> calendar aging, rainflow C-rate units, and the reset C-rate cap. They are
> historical diagnostics only and must not be treated as current benchmarks.
> Re-run H4.1 corpus generation and policy training before final comparisons;
> then rerun H4.2-H4.5, H4.7, H4.9-H4.12, including all fair statistics and
> shadow proxies. The interrupted 90-day run under
> `eval_output/household/h4_13_real_90d/` is incomplete and must also be rerun.

## Recommended Entry Points

- Environment and baseline sanity check: `notebooks/testrun.ipynb`
- Household log generation: `notebooks/test_simrun.ipynb`
- Household SB3 training: `notebooks/test_sb3train.ipynb`
- Canonical DT training: `scripts/pretrain_decision_transformer.py`
- Household evaluation: `notebooks/test_eval.ipynb`

### H4.9 Long-Horizon Pilot

The completed 30-day pilot is reproducible with:

```bash
python3 scripts/evaluate_household_ood_baselines.py \
  --normalized-dir data/household/real/normalized \
  --output-dir eval_output/household/h4_9_pilot_30d \
  --dt-path models/household/dt/h4_4_persistence_standard_rtg_8x512_ctx576_best.pt \
  --dt-config models/household/dt/h4_4_persistence_standard_rtg_8x512_ctx576_model_kwargs.json \
  --ppo-path models/household/sb3/h4_4_full/ppo_h4_4_fullcorpus.zip \
  --sac-path models/household/sb3/sac_model.zip \
  --td3-path models/household/sb3/td3_model.zip \
  --tariff realistic --forecast-mode persistence \
  --window-days 30 --windows-per-segment 1 \
  --workers 12 --batch-eval --device cuda
```

It compares DT, PPO, SAC, TD3, rule, oracle, and no-battery policies on four
contiguous real-normalized windows. Results are stored in
`eval_output/household/h4_9_pilot_30d/summary.json`; this is a pilot rather
than a final seasonal claim because the surface has only four windows.

#### H4.9 re-baseline (2026-09-16): RTE-matched oracle + calibrated v2c DT

The evaluator now takes `--oracle-roundtrip-eff` (default **0.80**, matching the
environment) so the perfect-foresight oracle is no longer scored lossless, and
`--reference-cache-dir` to cache rule/oracle/SB3 rollouts across runs. The
re-baseline uses the calibrated `h4_v2c` DT and the full-corpus PPO with
`--batch-eval` (confirmed numerically equivalent to per-window stepping: at 90 d
the two agree to $2/yr and <0.1% on clips).

RTE-matched oracle (savings vs no-battery, A$/yr):

| surface | lossless (old) | RTE=0.80 | Δ |
|---|---:|---:|---:|
| real-OOD 7 d (10 windows) | +738.96 | **+689.65** | −6.7% |
| real 30 d (4 windows) | +699.1 | **+650.9** | −6.9% |
| real 90 d (3 windows) | +799.8 | **+728.6** | −8.9% |

DT v2c net-of-wear (gross / EFC·day⁻¹ / clips·day⁻¹), RTG=−2:

| surface | DT v2c net | gross | EFC·day⁻¹ | clips·day⁻¹ | rule net | PPO net |
|---|---:|---:|---:|---:|---:|---:|
| real-OOD 7 d | **+92.8** | +293.0 | 0.649 | 10.7 | −60.1 | −64.6 |
| real 30 d | **−113.8** | +298.2 | 0.648 | 25.1 | −131.5 | −77.5 |
| real 90 d | **−237.5** | +348.6 | 0.733 | 47.6 | −229.3 | −99.1 |

The calibrated v2c model is net-positive on the 7 d real-OOD surface (RTG=−2
best; RTG=0 nets +28.3 at 37.2 clips/day, RTG=−4 nets +83.2 at 11.1 clips/day)
but the long-horizon over-cycling persists: on 30 d/90 d it earns the highest
gross bill yet nets negative because SOC-rail clipping grows with horizon
(10.7 → 25.1 → 47.6 clips/day), concentrating wear. This is genuine policy
behaviour, not an evaluation artifact. For comparison, the earlier uncalibrated
`h4_4` model was net −$641/yr (30 d) and −$1338/yr (90 d) at ~1.1 EFC/day; v2c
is a large improvement but does not close the ~$600–970/yr gap to the
RTE-matched oracle. Runs of the same checkpoint can differ by a few $/yr because
near-rail SOC clipping makes degradation unusually sensitive to a handful of
non-deterministic GPU steps (e.g. the 7 d v2c net has been observed at +$86 and
+$93). Artifacts: `eval_output/household/h4_v2c_rtgsweep_rte080/`,
`h4_9_pilot_{30d,90d}_rte080/`, `h4_9_pilot_90d_rte080_perwindow/`; reference
cache `eval_output/household/reference_cache/`.

```bash
python3 scripts/evaluate_household_ood_baselines.py \
  --normalized-dir data/household/real/normalized \
  --dt-path models/household/dt/h4_v2c_persistence_8x512_ctx576.pt \
  --dt-config models/household/dt/h4_v2c_persistence_8x512_ctx576_model_kwargs.json \
  --ppo-path models/household/sb3/h4_4_full/ppo_h4_4_fullcorpus.zip \
  --tariff realistic --forecast-mode persistence --dt-rtg-value -2 \
  --window-days 7 --windows-per-segment 2 --limit-windows 10 \
  --soc-min 0.01 --soc-max 0.99 \
  --reference-cache-dir eval_output/household/reference_cache \
  --output-dir eval_output/household/h4_v2c_rtgsweep_rte080/rtg-2 \
  --batch-eval --device cuda --workers 8
```

#### Bucket-A re-baseline (2026-09-18): H4.5 re-price + inference-time & synth surfaces

Eval-only refresh of the remaining H4.x surfaces with the calibrated `h4_v2c`
DT, the corrected environment and the RTE-matched oracle (no retraining; the
pre-fix artifacts are preserved under `*_prefix`). Consolidated net-of-wear
savings vs no battery (A$/yr):

| surface | windows | DT net | DT gross | EFC/day | clips/day | rule | PPO | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real-OOD 7 d (RTG −2) | 10 | **+92.8** | +293.0 | 0.649 | 10.7 | −60.1 | −64.6 | +689.6 |
| real-OOD 7 d (RTG −4) | 10 | +83.2 | +292.7 | 0.658 | 11.1 | −60.1 | −64.6 | +689.6 |
| real-OOD 7 d (RTG 0) | 10 | +28.3 | +247.2 | 0.514 | 37.2 | −60.1 | −64.6 | +689.6 |
| real 30 d unconstrained (RTG −2) | 4 | −113.8 | +298.2 | 0.648 | 25.1 | −131.5 | −77.5 | +650.9 |
| real 90 d unconstrained (RTG −2) | 3 | −237.5 | +348.6 | 0.733 | 47.6 | −229.3 | −99.1 | +728.6 |
| real 30 d directional 0.05+0.05 | 12 | −53.9 | +23.8 | 0.050 | 0.0 | −89.3 | −72.1 | +673.3 |
| real 30 d directional + $0.30/$0.10 gate | 12 | −51.7 | +26.1 | 0.050 | 0.0 | −89.3 | −72.1 | +673.3 |
| real 90 d directional 0.05+0.05 | 3 | −103.2 | +14.0 | 0.045 | 9.6 | −229.3 | −99.1 | +728.6 |
| real 90 d directional + gate | 3 | −107.6 | +22.8 | 0.045 | 10.3 | −229.3 | −99.1 | +728.6 |
| offline shadow 14 d + gate | 5 | −45.2 | +25.4 | 0.050 | 0.0 | −67.2 | −55.3 | +668.7 |
| synth 1 y (RTG −2) | 10 | **+298.8** | +463.8 | 0.240 | 10.1 | −49.3 | −86.1 | +1937.8 |
| synth 180 d 5 kWh combined 0.10 | 10 | −91.6 | +34.5 | 0.082 | 6.4 | −63.6 | −93.2 | +768.9 |
| synth 180 d 10 kWh combined 0.10 | 10 | −51.0 | +73.3 | 0.082 | 6.0 | −39.6 | −60.7 | +1360.4 |
| synth 180 d 15 kWh combined 0.10 | 10 | −14.7 | +111.4 | 0.082 | 11.6 | −40.4 | −42.5 | +1799.1 |
| synth 180 d 20 kWh combined 0.10 | 10 | **+30.9** | +154.0 | 0.082 | 9.8 | −32.7 | −54.5 | +2098.9 |
| synth 180 d 5 kWh directional 0.05+0.05 | 10 | −82.8 | +14.3 | 0.042 | 9.1 | −63.6 | −93.2 | +768.9 |
| synth 180 d 10 kWh directional | 10 | −68.5 | +32.8 | 0.042 | 9.6 | −39.6 | −60.7 | +1360.4 |
| synth 180 d 15 kWh directional | 10 | −56.5 | +50.5 | 0.042 | 18.8 | −40.4 | −42.5 | +1799.1 |
| synth 180 d 20 kWh directional | 10 | −39.7 | +71.7 | 0.042 | 19.9 | −32.7 | −54.5 | +2098.9 |

Findings:

- **The only net-positive real-data surface is the unconstrained 7 d real-OOD
  DT** (+$83–93/yr at RTG −2/−4). Every long-horizon or budget-constrained real
  surface is net-negative once corrected wear is charged, even though the
  RTE-matched oracle still saves $650–730/yr — the 5 kWh arbitrage margin is
  smaller than the battery's own wear.
- **The old "small positive" inference-time results do not survive corrected
  physics**: 30 d directional +$2.7 → **−$53.9**/yr; 30 d gate +$4.44 →
  **−$51.7**; 90 d directional +$1.1 → **−$103.2**; offline shadow +$6.08 →
  **−$45.2**.
- **Long-horizon rail handling remains unsolved.** At 90 d the directional
  projection still clips ~9.6–10.3 steps/day (all lower-bound) and incurs a
  ~$649–698/window safety penalty; the 180 d surfaces clip 6.0–19.9/day. The
  30 d surface is the only one with zero clips (`soc_projected_steps` = 0 there).
- **Directional budgeting is no longer better than combined** on corrected
  surfaces: under the combined 0.10 budget net savings rise monotonically with
  capacity and turn positive at 20 kWh (**+$30.9**), while every directional
  capacity is negative (e.g. 15 kWh: −14.7 combined vs −56.5 directional) and
  clips more. The earlier directional advantage was an artifact of the pre-fix
  wear accounting.
- **Safety penalty is a reward-shaping cost, not a bill cost**, and is *not*
  included in `net_savings_vs_no_battery`. The long-horizon penalties are large
  (e.g. $9,180 total for the 1 y synth, $8,938 for 20 kWh directional), so the
  net figures above are optimistic in deployment terms.
- **H4.5 re-price** (same five teacher regimes, eval-only): all regimes remain
  net-negative, but wear is ~2–2.5× cheaper than the pre-fix accounting (full
  realistic −$469.5 → −$186.0/yr; `report.md` §8.1.1).

Paired statistics (per-window net savings, bootstrap 95% CI + Wilcoxon) are in
`eval_output/household/h4_12_bucket_a/statistics.json`. The DT beats PPO
significantly on the 7 d real-OOD (p=0.002), 30 d directional (p<0.001), 1 y
synth (p=0.002) and 180 d 20 kWh combined (p=0.002), and is not significantly
different on the three-window 90 d surface (underpowered).

> **Superseded (2026-09-18).** The inference-time throughput/price-gate,
> offline-shadow, multi-seed SB3, and fair-comparison numbers in the remainder
> of this H4.9 section were produced under the pre-fix pipeline (pre-calibration
> wear, uncorrected physics, lossless oracle) and are retained only as history.
> The re-baselined values are in the bucket-A table above. The SB3 checkpoints
> themselves are pre-fix, so a fully fair DT-vs-RL comparison still requires
> retraining them under the corrected environment.

The longer-horizon extensions use the same checkpoints and no retraining:

- `eval_output/household/h4_9_pilot_90d/summary.json`: three 90-day
  contiguous real-normalized windows.
- `eval_output/household/h4_9_pilot_1y_synth/summary.json`: ten one-year
  windows from held-out two-year H4.1 test episodes. These episodes preserve
  their configured 20 kWh battery sizes and are therefore a synthetic
  stress-test surface, not a direct 5 kWh real-home comparison.

The first no-retraining diagnostic sweep is in
`eval_output/household/h4_9_rtg_sweep/`. It evaluates DT RTG prompts
`-4, -2, -1, 0, +1` on the same 90-day real windows under 1%–99% SOC
operating bounds. RTG `-4` was the best prompt, but it still over-cycled at
about 1.12 EFC/day and remained negative after degradation costs, so the next
diagnostic is constrained action projection rather than prompt calibration
alone.

The constrained inference prototype is implemented with
`--dt-max-efc-per-day`. On the same 90-day windows, a 0.20 EFC/day budget
reduced DT cycling to 0.145 EFC/day and produced an annualized net bill of
about A$1,620 (about A$23/year below the no-battery baseline). Results are in
`eval_output/household/h4_9_projection/efc_0_20/`; the 0.30 EFC/day comparison
is in `efc_0_30/`. This is an inference-time diagnostic, not a retrained
policy.

The budget sweep (`efc_0_10` through `efc_0_30`) selected 0.10 EFC/day.
Applying that normalized budget to the 90-day real windows produced positive
net savings for the larger configurations: approximately A$5.6/year at
10 kWh/5 kW, A$23.2 at 15 kWh/7 kW, and A$49.3 at 20 kWh/7 kW. The 5 kWh/3.3
kW case remained approximately A$10.5/year below no-battery, so the result
supports versatility but still requires broader seasonal and matched-capacity
validation.

Matched-capacity synthetic validation is now available through the explicit
override options:

```bash
python3 scripts/evaluate_household_ood_baselines.py \
  --synth-dir data/household/synth_h4_1 --synth-split test \
  --output-dir eval_output/household/h4_9_validation/synth_6m_20kwh_7kw \
  --override-capacity-kwh 20 --override-max-flow-kw 7 \
  --dt-path models/household/dt/h4_4_persistence_standard_rtg_8x512_ctx576_best.pt \
  --dt-config models/household/dt/h4_4_persistence_standard_rtg_8x512_ctx576_model_kwargs.json \
  --skip-ppo --skip-reference-policies --dt-rtg-mode standard \
  --dt-rtg-value -4 --forecast-mode persistence --tariff realistic \
  --window-days 180 --windows-per-segment 1 --limit-windows 10 \
  --device cuda --soc-min 0.01 --soc-max 0.99 \
  --dt-max-efc-per-day 0.10
```

The held-out 180-day comparison uses the same ten windows for every matched
configuration. At 0.10 EFC/day, annualized net savings versus no battery were
approximately −A$3.5 (5 kWh/3.3 kW), +A$4.1 (10 kWh/5 kW), +A$14.6
(15 kWh/7 kW), and +A$34.6 (20 kWh/7 kW), with mean throughput near 0.0506
EFC/day. Results are stored in
`eval_output/household/h4_9_validation/synth_6m_{5kwh_3_3kw,10kwh_5kw,15kwh_7kw,20kwh_7kw}/summary.json`.

The throughput projector now also performs an exact SOC-feasibility projection
using the environment's current battery level and configured 1%–99% bounds
before applying the daily budget. This prevents the environment from doing a
second safety clip and records `action_soc_projected_steps` in each summary.

For the next ablation, separate directional budgets can be enabled without
changing the default combined-budget behavior:

```bash
--dt-max-charge-efc-per-day 0.10 \
--dt-max-discharge-efc-per-day 0.10
```

These budgets are tracked independently, so charging cannot consume the
discharge allowance (or vice versa). Use both options together when
`--dt-max-efc-per-day` is omitted.

The fair directional ablation used `0.05` EFC/day for each direction, keeping
the maximum combined budget equal to the earlier `0.10` EFC/day experiment.
Across the same ten 30-day held-out windows, net savings were approximately
`+A$0.9/year` for 5 kWh/3.3 kW and `+A$60.1/year` for 20 kWh/7 kW, with zero
SOC clips and observed discharge EFC near `0.05/day`. The corresponding
combined-budget runs produced approximately `+A$0.2/year` and `+A$41.4/year`.
This supports retaining directional budgeting for the next seasonal
validation, while recognizing that these are still 30-day synthetic-window
results.

The first real-data directional run exposed an accounting artifact: resetting
the budget by row count treated irregular or missing timestamps as if every
day
had the same number of samples. The projector now resets from the environment
timestamp date when available, with the row-count method retained only for
integer-timestamp test fixtures. The corrected 12-window, 30-day real
validation (`eval_output/household/h4_9_validation/real_30d_directional_calendar/summary.json`)
held discharge throughput to `0.04999 EFC/day`, produced zero SOC clips, and
delivered approximately `+A$2.7/year` net savings versus no battery. This is
below the earlier combined-budget real result (`+A$3.8/year`), so directional
budgeting remains safe and viable but is not yet an economic replacement for
the combined projector on this surface.

The longer three-window real validation used the same directional limits over
90 days:
`eval_output/household/h4_9_validation/real_90d_directional_calendar/summary.json`.
It held mean throughput to `0.049974 EFC/day`, produced zero SOC clips and
zero safety penalty, and delivered approximately `+A$1.1/year` net savings
versus no battery. This is a positive but small result: the directional
projector is safe over the longer real horizon, but the three-window sample
does not yet establish a robust economic advantage over the combined
projector or across all household seasons.

The first price-aware gating ablation used
`--dt-min-discharge-price 0.30` and `--dt-max-charge-price 0.10` ($/kWh).
It gated low-price discharge and grid charging above $0.10/kWh while still
allowing solar-surplus charging. On the same three 90-day real windows, net
savings increased to approximately `+A$3.7/year` from `+A$1.1/year` without
gating. Mean discharge throughput remained `0.049974 EFC/day`, SOC clips and
safety penalty remained zero, and the gate suppressed approximately 1,900
actions per window. This is promising but not conclusive: it is one threshold
pair on three windows, so the next validation is a small threshold sweep and
broader seasonal coverage before treating the gate as a deployment default.

A focused threshold sweep on the same three 90-day windows tested discharge
thresholds of `$0.25`, `$0.30`, and `$0.35/kWh` against grid-charge thresholds
of `$0.05`, `$0.10`, and `$0.15/kWh`. All combinations through a `$0.30`
discharge threshold produced the same approximately `+A$3.7/year` result
because this tariff has only free-window and standard-price import levels.
The `$0.35` threshold blocked all discharge, produced no useful throughput,
and was economically negative, so it was rejected.

The selected `$0.30/$0.10` gate was then evaluated on all twelve real
30-day windows:
`eval_output/household/h4_9_validation/real_30d_directional_price_gate_030_010/summary.json`.
It produced approximately `+A$4.44/year` net savings, compared with
`+A$2.69/year` for directional budgeting without gating and approximately
`+A$3.82/year` for the earlier combined-budget real comparator. Mean
discharge throughput was `0.049992 EFC/day`, with zero SOC clips and zero
safety penalty across all windows. The result supports the gate as the
current inference-time candidate, subject to shadow-mode validation.

### Multi-seed SB3 baseline status

The generalized `scripts/train_household_sb3.py` entrypoint now supports PPO,
SAC, and TD3 with the same H4.1 corpus, 5 kWh / 3.3 kW hardware, tariff,
network, and 250k-timestep budget. Three seeds (`42`, `7`, and `20260830`)
were trained for each algorithm under
`models/household/sb3/multiseed/`. Held-out synthetic test mean rewards were:

| Algorithm | Mean test reward | Across-seed SD | Interpretation |
|---|---:|---:|---|
| PPO | −A$1,763.5 | 18.8 | Stable across seeds |
| SAC | −A$1,765.3 | 5.0 | Most stable, similar mean to PPO |
| TD3 | −A$1,876.8 | 247.3 | One seed-collapse outlier |

These are training-corpus validation rewards, not a final DT-versus-RL
deployment comparison. The nine checkpoints still need evaluation through the
same real OOD, SOC, throughput, degradation, and price-gating surfaces before
comparative claims are made.

The first shared long-horizon deployment surface is now complete for all nine
checkpoints. It uses 20 held-out synthetic-test episodes, each bounded to 180
complete days, realistic tariffs, persistence forecasts, 1%-99% SOC limits,
and matched 5 kWh / 3.3 kW hardware. These are raw SB3 policies: no
DT-specific throughput or price wrapper was applied.

| Algorithm | Seed | Net savings vs no battery (A$/yr) | EFC/day | Mean capacity fade | SOC clips | Safety penalty |
|---|---:|---:|---:|---:|---:|---:|
| PPO | 42 | -42.44 | 0.241 | 0.792% | 114,743 | 28,685.75 |
| PPO | 7 | -12.22 | 0.127 | 0.384% | 0 | 0 |
| PPO | 20260830 | -54.40 | 0.210 | 0.607% | 1,192 | 298.00 |
| SAC | 42 | -30.18 | 0.354 | 0.822% | 0 | 0 |
| SAC | 7 | -44.55 | 0.489 | 1.025% | 0 | 0 |
| SAC | 20260830 | -43.72 | 0.335 | 0.675% | 4 | 1.00 |
| TD3 | 42 | -661.79 | 0.580 | 6.690% | 749,217 | 187,304.25 |
| TD3 | 7 | -0.11 | 0.004 | 0.013% | 1,034,081 | 258,520.25 |
| TD3 | 20260830 | -8.88 | 0.012 | 0.089% | 1,032,843 | 258,210.75 |

The main conclusion is safety and deployment viability, not an RL ranking:
all three algorithms show substantial seed sensitivity, and the TD3 policies
turn the hard SOC clamp into a behavioral failure mode (millions of clipped
requests and large safety penalties). PPO and SAC avoid that failure more
often, but still produce negative net-of-wear economics on this raw-policy
surface. This raw-policy result is retained as a diagnostic baseline; the fair
shared-wrapper comparison is documented below.

The evaluator now loads SAC and TD3 independently of `--skip-ppo`, so
algorithm-specific runs cannot silently degrade into a no-battery result.
Artifacts are under `eval_output/household/h4_10_multiseed/*_180d_valid/`.

The offline deployment-proxy shadow run
(`eval_output/household/h4_9_validation/offline_shadow_14d_gate_030_010/summary.json`)
replayed the selected configuration over five contiguous 14-day real-data
windows without controlling a battery. It maintained `0.049997 EFC/day`,
zero SOC clips, and zero safety penalty, with approximately `+A$6.08/year`
annualized net savings versus no battery. This validates the logging and
inference-time safety path on deployment-like telemetry, but it is not a
live sim-to-real shadow test: the repository has no device-control or
telemetry-stream interface. A real household shadow run remains required
before closed-loop control.

The fair directional policy was then evaluated on the same ten held-out
180-day matched-capacity windows used by the combined-budget study. Net
savings versus no battery were approximately:

| Capacity / power | Combined `0.10` EFC/day | Directional `0.05` + `0.05` EFC/day |
|---|---:|---:|
| 5 kWh / 3.3 kW | −A$3.5/year | −A$0.9/year |
| 10 kWh / 5 kW | +A$4.1/year | +A$22.2/year |
| 15 kWh / 7 kW | +A$14.6/year | +A$42.5/year |
| 20 kWh / 7 kW | +A$34.6/year | +A$61.4/year |

The directional runs held mean EFC between `0.049945` and `0.049947/day`
for every capacity, with zero environment SOC clips and zero safety penalty.
This is the first longer-horizon result supporting directional budgeting as
the preferred inference-time configuration for the matched-capacity synthetic
surface. It still does not remove the real-data need for shadow-mode
validation, and it does not justify retraining yet.

### Fair DT versus SB3 deployment comparison

The broader fair comparison applies identical inference-time controls to the
DT and all nine SB3 checkpoints: realistic tariffs, persistence forecasts,
1%–99% SOC bounds, matched 5 kWh / 3.3 kW hardware, independent 0.05
charge/discharge EFC/day budgets, and `$0.30` discharge / `$0.10` grid-charge
price gates. All policies were evaluated on the same ten deterministic
90-day windows spanning five household archetypes and both 6-month and
2-year synthetic horizons.

| Policy | Net savings vs no battery (A$/yr) | EFC/day | Mean capacity fade | SOC clips | Safety penalty |
|---|---:|---:|---:|---:|---:|
| DT (`rtg=-4`) | **+3.55** | 0.04997 | 0.1083% | 0 | 0 |
| PPO seed 42 | -11.69 | 0.04379 | 0.1522% | 172 | 43.0 |
| PPO seed 7 | +2.35 | 0.04568 | 0.1076% | 0 | 0 |
| PPO seed 20260830 | -2.04 | 0.04646 | 0.1032% | 0 | 0 |
| SAC seed 42 | -0.13 | 0.04876 | 0.1112% | 0 | 0 |
| SAC seed 7 | +0.54 | 0.04994 | 0.1105% | 0 | 0 |
| SAC seed 20260830 | -2.33 | 0.04727 | 0.0929% | 0 | 0 |
| TD3 seed 42 | -1.11 | 0.03610 | 0.0714% | 3 | 0.75 |
| TD3 seed 7 | +2.28 | 0.00544 | 0.0000% | 7 | 1.75 |
| TD3 seed 20260830 | +1.94 | 0.00544 | 0.0000% | 10 | 2.5 |

The controls reduced unsafe interventions to near zero for every policy, but
the projection layer intervened frequently as expected under the tight
directional budgets. DT had the highest mean economic result and was the only
policy with both target throughput and zero SOC interventions. PPO and SAC
were mixed across seeds, while two TD3 seeds became markedly under-active and
the remaining seed incurred safety clips. This supports the safety wrapper
and provides a broader fair comparison, but does not establish superiority
across real households or hardware configurations.

### Fair real-OOD transfer comparison

The shared-wrapper comparison was transferred to the available normalized
real-household OOD surface. It contains four complete 30-day windows covering
February 2023, April-May 2023, October 2023, and July 2024. All policies used
the same 5 kWh / 3.3 kW hardware, persistence forecast, realistic tariff,
1%–99% SOC bounds, independent 0.05 charge/discharge EFC/day budgets, and
`$0.30/$0.10` price gates.

| Policy | Net savings vs no battery (A$/yr) | EFC/day | Mean capacity fade | SOC clips | Safety penalty |
|---|---:|---:|---:|---:|---:|
| DT (`rtg=-4`) | **+3.70** | 0.04999 | 0.0323% | 0 | 0 |
| PPO seed 42 | -7.51 | 0.04575 | 0.0420% | 24 | 6.0 |
| PPO seed 7 | +2.32 | 0.04648 | 0.0349% | 0 | 0 |
| PPO seed 20260830 | +2.15 | 0.04059 | 0.0196% | 0 | 0 |
| SAC seed 42 | -3.69 | 0.04924 | 0.0338% | 0 | 0 |
| SAC seed 7 | -7.19 | 0.04999 | 0.0331% | 0 | 0 |
| SAC seed 20260830 | -8.32 | 0.04673 | 0.0278% | 0 | 0 |
| TD3 seed 42 | -3.54 | 0.03316 | 0.0164% | 0 | 0 |
| TD3 seed 7 | +4.52 | 0.01640 | 0.0000% | 3 | 0.75 |
| TD3 seed 20260830 | +4.18 | 0.01633 | 0.0000% | 3 | 0.75 |

This is transfer and safety evidence, not a definitive seasonal claim: the
available real-OOD surface has only four windows. DT preserved its synthetic
behavioral profile—throughput at the intended budget, zero SOC clips, and zero
safety penalty—and achieved positive net savings. PPO/SAC remained
seed-sensitive, while TD3's positive results came from under-active policies
using only about one-third of the intended discharge budget. Broader real
coverage is limited by the available normalized telemetry; live shadow mode
remains unavailable.

### Paired statistical analysis

`scripts/household_fair_statistics.py` computes paired bootstrap 95% CIs and
exact two-sided Wilcoxon tests from the per-window annualized net-savings
values. On the ten-window synthetic surface, DT's paired advantage was
significant against PPO seed 42 (+A$15.24/year, CI +A$7.62 to +A$23.51,
`p=0.00195`), PPO seed 20260830 (+A$5.60, CI +A$3.49 to +A$7.74,
`p=0.00195`), SAC seeds 42 and 20260830 (`p=0.00195` and `p=0.01367`),
and TD3 seed 42 (+A$4.66, CI +A$2.87 to +A$6.52, `p=0.00195`).
Differences against the better or more conservative seeds were positive but
not significant at this sample size.

On the four-window real-OOD surface, no DT-versus-RL comparison reached
`p<0.05`; the sample is too small for a reliable significance claim. The
statistics support the synthetic result as seed-dependent evidence rather
than a universal DT win, and characterize the real result as directional
transfer evidence only. Artifacts are
`eval_output/household/h4_11_fair/statistics_10x90.json` and
`eval_output/household/h4_12_fair_real/statistics_4x30.json`.

## Standard Household Workflow

### 1. Prepare raw household data

Place the Ausgrid household CSV files under:

- `data/household/raw/`

The household preprocessing flow converts raw customer data into the schema expected by `SolarBatteryEnv`.

### 2. Generate baseline or rollout logs

Use `notebooks/test_simrun.ipynb` to:

- inspect transformed household data
- run rule-based or planning baselines
- write parquet logs under `data/household/logs/`

These logs are one of the main inputs for offline DT training.

### 3. Train online RL baselines if needed

Use `notebooks/test_sb3train.ipynb` to train SB3 policies and optionally export rollout logs.

Typical model outputs live under:

- `models/household/sb3/`

### 4. Train a household Decision Transformer

Use the canonical CLI surface:

```bash
python scripts/pretrain_decision_transformer.py \
  --data-dir data/household/logs \
  --patterns train_episode_01 train_episode_02 \
  --epochs 2 \
  --batch-size 6 \
  --lr 2e-5 \
  --save-path models/household/dt/dt_model.pt \
  --checkpoint-path models/household/dt/dt_model_checkpoint.pt \
  --loss-csv-path models/household/dt/dt_model_loss_history.csv
```

This is the shared DT trainer used across the repo. The household track usually differs from the AEMO track in:

- data source
- state and action dimensions
- artifact locations

### 5. Evaluate household policies

Use `notebooks/test_eval.ipynb` for notebook-driven comparison across:

- rule-based baselines
- planning baselines
- SB3 models
- Decision Transformer policies

Typical evaluation outputs live under:

- `eval_output/`

### 6. Build the synthetic diverse-household corpus

H1.5 recomposes complete normalized 5-minute days; it does not add row-wise
noise. The generator clusters real load profiles by season and weekday/weekend,
samples five explicit archetypes, injects optional EV/AC/pool blocks under a
60% daily-energy cap, scales the real solar curve, and rejects candidates that
fail any G1–G6 validation gate.

From the repository root in the GPU Distrobox:

```bash
python3 scripts/build_household_synth_corpus.py \
  --normalized-dir data/household/real/normalized \
  --output-dir data/household/synth \
  --episodes 1200 \
  --days-per-episode 7 \
  --seed 42
```

Each episode is an env-view-compatible parquet under
`data/household/synth/<archetype>/`, with `SolarGen` and `HouseLoad` stored as
kWh per 5-minute step and the day-ahead `FutureSolar`/`FutureLoad` columns.
`manifest.json` records the seed, source dates and clusters, archetype,
lambda scale, appliance parameters, solar and battery configuration, gate
metrics, split (`train`/`val`/`test`), and the real source dates reserved for
OOD evaluation. The real household remains the OOD surface of record and must
not be included in synthetic training data.

The H4.1 extension supports a balanced horizon/scenario matrix. Use
`--horizons 1w 2w 6m 2y` to cycle across one-week, two-week, six-month, and
two-year episodes while retaining the five archetypes, seasonal/day-type
sampling, solar/battery variation, and per-episode provenance. For example:

```bash
python3 scripts/build_household_synth_corpus.py \
  --output-dir data/household/synth_h4_1 \
  --episodes 240 --horizons 1w 2w 6m 2y --seed 20260830
```

The manifest records the horizon and degradation configuration. The generated
data provides held-out synthetic surfaces; real household segments remain the
primary OOD surface.

### 6a. Train a fresh modern SB3 baseline

The legacy PPO checkpoint is not a modern-data baseline. Train a new PPO
directly on the H4.1 corpus with parallel CPU environments:

```bash
python3 scripts/train_household_sb3.py \
  --corpus-dir data/household/synth_h4_1 \
  --output-dir models/household/sb3/h4_3 \
  --timesteps 250000 --n-envs 12 \
  --capacity-kwh 5 --max-flow-kw 3.3
```

The H4.3 pilot used the realistic tariff and matched 5 kWh/3.3 kW settings.
Its fresh PPO saved +$27/year on the five real OOD segments, below the
+$81/year rule baseline; this does not establish PPO as the preferred
household policy.

### 6b. Forecast-feature experiments

`FutureSolar` and `FutureLoad` are part of the 12-dimensional environment
observation. The current values are an honest 24-hour persistence forecast:
the same time slot from the preceding day, with a first-day fallback to the
current value. They are not a learned weather or load forecast. H4.2 compares
the trained policy with these channels preserved, zeroed, and shuffled before
considering causal rolling/seasonal forecast retraining. Real OOD segments
remain fixed across all modes.

The initial inference-only H4.2 ablation uses ten deterministic seven-day
windows (two per real segment) and the corrected H2 DT with J_t(soc). Annualized
savings were +$92.28 for persistence, +$92.88 with forecasts zeroed, and
+$95.78 with forecasts shuffled. These small differences show that the current
DT does not use the persistence channels beneficially. A stronger forecast must
be generated causally and included during matched policy retraining; replacing
inputs only at inference is not sufficient.

The no-forecast control has now been retrained from the same SDP-teacher corpus
with both forecast channels set to zero in every training observation
(`h4_2_no_forecast_8x512_ctx576`). On the same ten windows, evaluated with
zeroed channels, it saved +$232.32/year (95% CI for annualized bill:
$597–$1,172; n=10). This is a retrained-policy comparison, not directly
comparable to the earlier inference-only zeroing result.

Granite TTM-R3 is integrated as an **offline sidecar**, never as a dependency
of the simulator or main training container. `Containerfile.ttm` provides a
separate Python 3.12/Torch/CUDA environment, and the wrapper creates the
`energydecision-ttm` Distrobox with NVIDIA integration:

```bash
# Precompute a mirrored synthetic corpus with causal one-hour-ahead forecasts.
bash scripts/run_household_ttm_forecasts.sh \
  --synth-dir data/household/synth \
  --output data/household/synth_ttm \
  --device cuda --batch-size 512

# Precompute a timestamp-keyed real-OOD sidecar.
bash scripts/run_household_ttm_forecasts.sh \
  --normalized-dir data/household/real/normalized \
  --output data/household/real/household_ttm_forecasts.parquet \
  --device cuda --batch-size 512
```

The pinned `512-48-dec-512-r3` checkpoint uses 512 historical samples and
predicts 48 future samples. The environment columns use the 12th prediction
(one hour ahead at five-minute cadence). Each output records forecast issuance
and target timestamps, invalid warm-up rows, model revision, and forecast
quality. The full synthetic build improved solar/load MAE by 34.6%/12.2%;
the real-OOD sidecar improved them by 39.2%/17.5%.

Matched J_t(soc) DT evaluation on the fixed ten-window surface saved
+$381.76/year with TTM versus +$264.40/year without forecasts. The paired
TTM advantage was +$117.36/year (bootstrap 95% CI +$64.55–$163.82, 9/10
windows, one-sided Wilcoxon p=0.0029). This is supporting evidence only:
J_t(soc) uses future actuals when constructing its inference prompt.

The deployment-style control retrains all policies with
`--rtg-source constant`, identical architecture/optimizer/seed/data labels,
and `--stride 288`. Inference updates RTG only from realized rewards. The
shared fixed prompt is RTG=-2, selected from the training RTG median
(approximately -1.76), not from OOD policy performance:

```bash
python3 scripts/pretrain_decision_transformer.py \
  --surface-preset household_baseline \
  --data-dir data/household/dt \
  --patterns h4_2_ttm_sdp_train \
  --val-data-dir data/household/dt \
  --val-patterns h4_2_ttm_sdp_val \
  --split-policy explicit_validation \
  --context-length 576 --stride 288 \
  --n-block 8 --h-dim 512 --n-heads 8 --drop-p 0.15 \
  --batch-size 16 --epochs 5 --lr 3e-5 --seed 42 \
  --rtg-source constant --return-scale 1.0 \
  --action-loss-weight 0.999 --state-loss-weight 0.002 \
  --return-loss-weight 0.0001 --device cuda --amp-mode auto

python3 scripts/evaluate_household_ood_baselines.py \
  --dt-rtg-mode standard --dt-rtg-value -2 \
  --forecast-sidecar data/household/real/household_ttm_forecasts.parquet \
  --tariff realistic --window-days 7 --windows-per-segment 2 \
  --skip-reference-policies --skip-ppo --device cuda
```

Change the train/validation patterns and evaluation forecast input for the
matched persistence and no-forecast controls. On the same ten real-OOD
windows, annualized savings were:

| Forecast input | Savings vs no battery |
|---|---:|
| TTM-R3 | +$258.50/year |
| 24-hour persistence | +$216.74/year |
| No forecast | +$155.12/year |

TTM beat persistence by +$41.75/year (paired bootstrap 95% CI
+$16.56–$69.43, 9/10 windows, one-sided Wilcoxon p=0.0068) and no forecast
by +$103.37/year (95% CI +$78.23–$127.67, 10/10, p=0.0010). Persistence
also beat no forecast by +$61.62/year (95% CI +$33.96–$90.77, 8/10,
p=0.0049).

Prompt calibration matters: at the optimistic out-of-distribution RTG=0,
TTM saved -$4.21/year and no forecast saved +$38.53/year. Therefore use a
prompt justified from the training distribution and report it with every
result. The matched RTG=-2 experiment supports retaining offline TTM
forecasts in the observation pipeline; broader households and prompt
robustness belong to H4.4.

### 6c. H4.4 full-corpus forecast generalization (reproducible pipeline)

H4.2 was trained on the controlled H2 corpus (1,200 fixed seven-day,
5/10/20 kWh episodes). H4.4 repeats the matched three-way comparison on the
full horizon/scenario-diverse H4.1 corpus: 240 episodes, one per
archetype × season × battery-capacity × horizon cell (horizons `1w`, `2w`,
`6m`, `2y`; 55,860 episode-days), split 165 train / 35 val / 40 test with
the same 158 real source dates held out for OOD. Build it with:

```bash
python3 scripts/build_household_synth_corpus.py \
  --output-dir data/household/synth_h4_1 \
  --episodes 240 --horizons 1w 2w 6m 2y --seed 20260830
```

Precompute the causal TTM mirror (offline, isolated `energydecision-ttm`
Distrobox; 16.1M rows took ~3 h on the 2080 Ti at batch 2048):

```bash
bash scripts/run_household_ttm_forecasts.sh \
  --synth-dir data/household/synth_h4_1 \
  --output data/household/synth_h4_1_ttm \
  --device cuda --batch-size 2048
```

Generate matched SDP-teacher trajectories. The teacher optimizes against
actual solar/load, so action labels, rewards, and `rtg_value` are identical
across variants; only the stored forecast channels in the observations
differ:

```bash
# 24-hour persistence (corpus default channels)
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1 --split train \
  --forecast-mode persistence \
  --out data/household/dt/h4_4_persistence_sdp_train.parquet
# (repeat --split val)

# no forecast (both channels zeroed)
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1 --split train \
  --forecast-mode zero \
  --out data/household/dt/h4_4_no_forecast_sdp_train.parquet
# (repeat --split val)

# TTM (mirror corpus already carries causal one-hour-ahead channels)
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1_ttm --split train \
  --forecast-mode persistence \
  --out data/household/dt/h4_4_ttm_sdp_train.parquet
# (repeat --split val)
```

Train the three deployment-style standard-RTG DTs with the exact H4.2
recipe (state 12, act 1, 8×512, ctx 576, drop 0.15, batch 16, lr 3e-5,
5 epochs, seed 42, `--stride 288`, `--rtg-source constant`,
`--return-scale 1.0`, loss weights 0.999/0.002/0.0001). The shared
inference prompt RTG=-2 is again justified from the training RTG median
(-1.78 on the full corpus, vs -1.76 on H4.2), not from OOD rankings:

```bash
for variant in ttm persistence no_forecast; do
python3 scripts/pretrain_decision_transformer.py \
  --surface-preset household_baseline \
  --data-dir data/household/dt \
  --patterns h4_4_${variant}_sdp_train \
  --val-data-dir data/household/dt \
  --val-patterns h4_4_${variant}_sdp_val \
  --split-policy explicit_validation \
  --context-length 576 --stride 288 \
  --n-block 8 --h-dim 512 --n-heads 8 --drop-p 0.15 \
  --batch-size 16 --epochs 5 --lr 3e-5 --seed 42 \
  --rtg-source constant --return-scale 1.0 \
  --action-loss-weight 0.999 --state-loss-weight 0.002 \
  --return-loss-weight 0.0001 --device cuda --amp-mode auto \
  --save-path models/household/dt/h4_4_${variant}_standard_rtg_8x512_ctx576.pt \
  --checkpoint-path models/household/dt/h4_4_${variant}_standard_rtg_8x512_ctx576_checkpoint.pt \
  --loss-csv-path models/household/dt/h4_4_${variant}_standard_rtg_8x512_ctx576_loss.csv
done
```

Train the matching full-corpus PPO baseline (not the H4.3 pilot):

```bash
python3 scripts/train_household_sb3.py \
  --corpus-dir data/household/synth_h4_1 \
  --output-dir models/household/sb3/h4_4_full \
  --timesteps 500000 --n-envs 12 \
  --capacity-kwh 5 --max-flow-kw 3.3 \
  --battery-life-cost 5000 --seed 20260830 \
  --model-name ppo_h4_4_fullcorpus.zip
```

Evaluate on the fixed ten-window real-OOD surface (TTM from sidecars only;
TTM is never run live inside the simulator) and on the synthetic test
split with `--synth-dir` (per-episode battery configs from the manifest,
`--limit-windows` for a deterministic subsample), then compute paired
window-level bootstrap CIs, win counts, and one-sided Wilcoxon tests with
`scripts/household_forecast_stats.py` over the three `summary.json` files.

**H4.4 outcome (2026-09-02).** On the fixed 10-window real-OOD surface, the
full-corpus matched standard-RTG DTs saved: **TTM +$357.29/yr**, **24-hour
persistence +$309.35/yr**, **no forecast +$310.90/yr**, with fresh full-corpus
PPO +$23.66/yr, rule +$58.03/yr, and oracle +$738.96/yr as references. Paired:
TTM beats persistence by +$47.94/yr (95% CI +$27.99–$67.59, 9/10, Wilcoxon
p=0.0020) and no-forecast by +$46.39/yr (95% CI +$23.78–$68.91, 9/10,
p=0.0020). Every arm improved on the H4.2 seven-day-corpus run (TTM +$98.79),
so the offline-forecast benefit generalizes to the diverse corpus. The
persistence-vs-no-forecast gap collapsed to −$1.55/yr (p=0.46): with broad
training data the policy only exploits the genuinely-better TTM channel. On the
20-window synthetic test surface (mixed 5/10/20 kWh per-episode batteries) the
three DTs were statistically indistinguishable (TTM−no-forecast −$15.49/yr,
p=0.86, per-horizon mixed) — the forecast advantage is proven on the held-out
real household, not yet on the broad multi-battery synthetic surface. A matched
TTM/no-forecast/persistence sidecar layout bug (the forecast sidecar previously
reordered the Future columns) was fixed in `src/household_forecast.py`;
the mirror corpus and TTM trajectories were regenerated so all three arms share
identical action/reward/RTG labels and observation layout, differing only in
dims 6–7.


### 7. Compare observed and optimized real-battery dispatch

Use the H3 harness to replay recorded VPP actions and compare them with a
new cost-minimizing dispatch over each complete real day. It uses normalized
kW telemetry directly (not `build_year_dataset()`, whose values are already
converted to kWh per step), fits one corpus-wide action sign, and never spans
a gap seam:

```bash
python3 scripts/evaluate_household_tariffs.py \
  --normalized-dir data/household/real/normalized \
  --capacity-kwh 5 --max-flow-kw 3.3 --roundtrip-eff 0.80
```

The output is `eval_output/household/tariff_optimization/summary.json` with
per-day bootstrap CIs for observed replay, optimized dispatch, the
optimization gap, and the no-battery baseline under flat and free-window ToU
tariffs. Do not label a spot-pass-through result until a time-aligned retail
spot-price series is supplied.

### 8. Evaluate legacy policies on the real OOD surface

The legacy benchmark script evaluates the existing rule, PPO, and cloning-era
DT checkpoints with a fresh `SolarBatteryEnv` for every contiguous real-data
segment, alongside a daily perfect-foresight oracle. This isolates the
renovation gap and reports bootstrap CIs over segments:

```bash
python3 scripts/evaluate_household_ood_baselines.py \
  --capacity-kwh 5 --max-flow-kw 3.3
```

On the current five-segment OOD surface, the legacy PPO saves only $2/year and
the cloning-era DT loses $6/year against no battery (annualized segment
bootstrap means); the rule saves $70/year and the oracle indicates $729/year
is available. These legacy checkpoints therefore do not transfer to modern
household telemetry. See `eval_output/household/ood_baselines/summary.json`
for exact CI values and do not conflate this with an H2-trained policy.

### 9. Generate H2 SDP-teacher trajectories

The H2 handoff reuses the shared DT data schema. It solves a deterministic
per-day cost-to-go table over each synthetic environment episode, rolls its
actions through `SolarBatteryEnv`, and writes `rtg_value = -J_t(soc)` alongside
the normalized observation, action, and realized reward:

```bash
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth --split train \
  --out data/household/dt/sdp_teacher_train.parquet
```

Build a matching `--split val` corpus before invoking the sanctioned shared
trainer. Do not include real OOD segments in either corpus. The full corpus
build is now complete: 840 training and 180 validation episodes emitted
1,693,440 and 362,880 rows respectively. **Teacher data regenerated with
realistic tariff (31.042c import, free 11:00–14:00, 1c FiT) and RTE=0.80.**

**Standard-RTG baseline:** A 2×128 standalone DT trained for 5 epochs achieves
0.0424 validation total loss; on real OOD it saves **+$254/yr** (beats rule
+$82/yr).

**J_t(soc) at AEMO scale:** An 8×512 ctx576 model trained on corrected-RTE
teacher data achieves **+$300/yr** with `J_t(soc)` inference (3.6× rule, 24%
of oracle gap). This is a **positive transfer** result — planner distillation
transfers across scales when (1) model capacity ≥ AEMO scale, (2) RTE matches
the environment (0.80), (3) teacher data uses realistic tariff, and (4)
configs persist correctly.

The teacher data carries exact `rtg_value = -J_t(soc)`. The H2 inference
evaluation now precomputes an exact deterministic table per segment-local
calendar day and supplies the current-SOC prompt before each DT action; it
does not use the standard realized-reward recurrence for this policy. The
trainer also writes `<checkpoint-stem>_model_kwargs.json` beside every
checkpoint, so evaluation must use that file rather than a legacy model config.

## Main Artifacts

Common household artifact locations:

- `data/household/raw/`
- `data/household/logs/`
- `models/household/sb3/`
- `models/household/dt/`
- `eval_output/household/`

## Related Modules

- `src/helper.py`: household data transformation, evaluation, visualization
- `src/EnergySimEnv.py`: household simulation environment
- `src/decision.py`: agents and rollout helpers
- `src/sdp_algorithm.py`: planning baseline
- `src/mrdp_algorithm.py`: multi-resolution planning baseline
- `src/sb3train.py`: SB3 helper functions
- `src/decision_transformer.py`: DT model implementation
- `src/transformer_training.py`: DT training engine
- `src/household_synthetic.py`: clustered day library, archetypes, appliance/solar synthesis, validation gates, and episode export
- `scripts/build_household_synth_corpus.py`: reproducible H1.5 corpus builder
- `src/household_optimization.py`: deterministic dispatch optimizer and bootstrap CI helper
- `scripts/evaluate_household_tariffs.py`: H3 replay-gap and tariff evaluation
- `scripts/evaluate_household_ood_baselines.py`: H1 rule/oracle/PPO/DT real-OOD evaluation
- `scripts/generate_household_sdp_trajectories.py`: H2 synthetic SDP-teacher trajectory builder
- `scripts/train_household_sb3.py`: fresh modern-data SB3 (PPO) baseline trainer over a synthetic corpus
- `scripts/household_forecast_stats.py`: paired bootstrap/Wilcoxon stats for the H4.2/H4.4 forecast ablation
- `scripts/generate_household_ttm_forecasts.py` / `src/household_forecast.py`: offline causal TTM-R3 forecast sidecar
- `scripts/dump_household_behavior.py`: per-step rollouts (solar/load/price/power/SOC) of the matched H4.4 arms on one real-OOD window, feeding the website household behaviour charts
- `scripts/h4_degradation_study.py`: degradation-aware policy study (H4.5) — trains and evaluates DTs across multiple degradation modes (disabled, cycle-only, full, high/low battery-life-cost)

### 6d. H4.5 degradation-aware policy study

Run the five matched teacher/DT conditions from the repository root in the GPU
Distrobox:

```bash
python3 scripts/h4_degradation_study.py \
  --config all --train --eval \
  --output-dir results/h4_5_degradation
```

The study uses the H4.1 horizon-diverse corpus and holds the DT architecture,
optimizer, seed, and training schedule fixed while changing the teacher's
degradation mode and battery-life cost. Per-configuration models and
trajectories are written under `results/h4_5_degradation/`; real-OOD
evaluation summaries are written under
`eval_output/household/h4_5_degradation/`.

**H4.5 definitive outcome (2026-09-06).** We reran the study across five
conditions and three seeds (42, 20260830, 7) on the same fixed ten-window
real-OOD surface. All models use the 24-hour persistence forecast channels
(not the TTM sidecar), so these figures are not directly comparable with the
H4.4 TTM-forecast savings — this study isolates the degradation regime, not
the forecast input. The evaluator now records both the original
**grid-bill-only** metric and the primary **net-of-wear** metric using each
rollout's actual `info["step_degradation"]` and the default battery-life cost
of A$5,000, plus cycling-mechanism metrics (EFC/day, rainflow cycles/day,
capacity fade/day) accumulated from the same per-step logs.

| Training condition | Grid-bill savings vs no battery | Net-of-wear savings vs no battery | EFC/day | Cycles/day | Capacity fade/day |
|---|---:|---:|---:|---:|---:|
| Degradation disabled | +A$364.9/yr | **−A$401.1/yr** | 1.02 | 5.06 | 0.042% |
| Cycle-only | +A$310.3/yr | **−A$494.9/yr** | 1.10 | 4.72 | 0.044% |
| Full realistic (A$5,000) | +A$369.7/yr | **−A$469.5/yr** | 1.08 | 4.30 | 0.046% |
| Full, high cost (A$10,000) | +A$321.7/yr | **−A$459.8/yr** | 1.09 | 4.26 | 0.043% |
| Full, low cost (A$1,000) | +A$270.8/yr | **−A$647.0/yr** | 1.12 | 4.75 | 0.050% |

The unambiguous headline: **every condition is negative net-of-wear** once
battery degradation is priced into the ledger — on these short seven-day
windows the battery does not pay for its own wear under any training regime.
The sign (all net-negative) is consistent across seeds; the *specific* ranking
is not robust (the A$1,000 vs A$10,000 conditions remain non-monotonic, and
their pairwise intervals overlap). The earlier expectation that the
degradation-disabled policy would be the hardest cycler is **not** supported
by the mechanism data: EFC/day is flat across conditions (1.02–1.12) and
disabled is actually the *lowest* (1.02), while cycle counts overlap within-seed
(2.8–6.2 across all conditions). So the honest reading is about the economics —
short-window arbitrage is not a valid objective once wear is charged — rather
than "one regime cycles more than another."

Pairwise comparisons across the ten windows: grid-bill differences are
statistically clear in several cases (e.g., full_realistic vs low_cost
**+A$98.9/yr**, 95% CI **A$67.6–A$131.9**, p=0.0010; disabled vs cycle_only
**+A$54.6/yr**, 95% CI **A$20.8–A$87.1**, p=0.0137), while net-of-wear
differences remain economically large but noisier because wear is a short-window
quantity (intervals span both signs). The A$1,000 vs A$10,000 inversion does
not survive the multi-seed aggregate — it collapses into non-monotonic noise
rather than a stable dose-response relationship.

In short, the earlier pilot should be treated as a historical grid-bill-only
sanity check. The final, definitive result is the three-seed net-of-wear study:
**wear matters and every regime is net-negative on short windows; the
cost-sweep does not establish a clean ordering across seeds.**

## Validation And Iteration

For code changes that affect the household track, use pytest as the main validation path:

```bash
python -m pytest tests/ -v
```

If you are changing only a narrow area, prefer a single relevant test file first.

## Notes

- Treat `scripts/pretrain_decision_transformer.py` as the canonical household DT entrypoint.
- Treat notebooks as the best surface for exploration, inspection, and demonstration.
- Keep household results separate from AEMO results in both reporting and interpretation.