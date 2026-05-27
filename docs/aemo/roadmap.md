# AEMO Decision Transformer short-term roadmap note

This is a **roadmap note** for near-term AEMO DT research priorities.

Use this document when you need:

- a quick summary of what is still missing in AEMO DT work
- suggested near-term milestones
- guidance on what to prioritize next

This is not the main operating workflow for running AEMO experiments. For the actual notebook and training flow, read [workflow.md](workflow.md). For the full AEMO docs map, start with [README.md](README.md).

## Purpose
This note focuses on the **next short-term steps** needed to evaluate whether Decision Transformers (DTs) are competitive in the **AEMO battery trading environment**, which is currently behind the household benchmark in experimental maturity.

## Current State
- The repository already has a working AEMO environment: `AEMOBatteryTradingEnv`.
- The repository already has a replay/data-generation path through `AEMOAgent` + `run_dispatch_replay()`.
- The repository already has a more realistic utility-scale degradation model (`real_world`) with `LFP`/`NMC` chemistry presets.
- However, the main roadmap still shows AEMO DT work as incomplete:
  - `Conduct data gathering and training on AEMO env`
  - `DT prompt calibration`
  - `Offline dataset studies`

## Short-Term Recommendation
The immediate goal should **not** be to claim DT superiority on AEMO yet. The short-term goal should be to establish a **credible early AEMO benchmark** that can support a fair competitiveness claim later.

## Recommended 4-Step Near-Term Plan

### 1. Build a clean AEMO offline dataset first
Prioritize a small but well-defined offline dataset over a large mixed dataset.

Suggested first dataset scope:
- Region: start with **SA1** only
- Horizon: select **multiple non-overlapping windows** across different market conditions
- Episode length: standardize to **fixed horizons** (for example 1 day, 3 days, 7 days)
- Degradation setting: fix to:
  - `degradation_mode='real_world'`
  - `degradation_chemistry='LFP'`
  - realistic battery life cost
- Behavior policies for dataset generation:
  - rule-based AEMO agent
  - dispatch replay
  - at most one RL baseline initially

Why this matters:
- It keeps the first AEMO study interpretable.
- It reduces confounding from mixing too many behavior policies before the data pipeline is trustworthy.
- Dispatch replay gives a realistic anchor for “how real batteries behaved,” which is especially valuable in AEMO.

### 2. Enforce train/validation/test separation by episode window, not sliding-window sample
The current DT training pipeline splits the combined dataset with `random_split(...)` after sliding-window construction.

That is acceptable for optimization, but it is **not sufficient for a clean benchmark claim**, because highly overlapping windows from the same episode can end up in both train and validation.

For AEMO, the short-term fix should be:
- split at the **episode/date-window level before window extraction**
- keep test windows fully disjoint in time from train windows
- log the exact split in a manifest file

Minimum split suggestion:
- train: earlier windows
- validation: mid-period windows
- test: later windows

This is the most important methodological fix before claiming DT competitiveness.

### 3. Freeze one evaluation protocol and stop tuning on the test set
The household DT comparison currently shows strong RTG sensitivity, which is useful scientifically, but it also creates a risk of overinterpreting the best prompt.

For AEMO, use a stricter protocol:
- choose `rtg_value` on validation only
- choose `return_scale` from training data only
- report exactly one primary DT model on test
- treat prompt-sensitivity plots as secondary analysis

Before inference, calibrate the prompt on the **target evaluation scenario**. The earlier sweeps showed that RTGs recommended on one region/time slice can degrade performance when reused on another slice, so the chosen `rtg_value` should match the held-out scenario you actually plan to deploy against.

Primary comparison should be:
- DT
- rule baseline
- dispatch replay baseline
- one planning baseline if available for AEMO
- one trained RL baseline if available

This will produce a much cleaner “is DT competitive?” result.

### 4. Expand evaluation beyond reward only
For AEMO, reward alone is too weak as a first validation target because multiple flawed policies can look competitive on mean reward.

Short-term AEMO evaluation should always report:
- mean episode reward
- reward distribution / tail risk
- total energy revenue
- total FCAS revenue
- degradation cost
- calendar degradation vs cycle degradation
- battery throughput / cycling intensity
- feasibility / incident counts

In early AEMO work, DT should be considered promising only if it is competitive on reward **without relying on obviously unrealistic battery usage**.

## Suggested Short-Term Milestones

### Milestone A — AEMO early benchmark ready
Deliverables:
- one curated AEMO offline dataset
- fixed train/val/test split
- one notebook or script that reproduces the split and manifest
- baseline evaluation for rule + dispatch replay

### Milestone B — First AEMO DT training run
Deliverables:
- train one DT on the curated AEMO dataset
- evaluate on held-out AEMO test windows
- report one primary prompt setting chosen from validation only

### Milestone C — First competitiveness statement
Deliverables:
- compare DT against rule / dispatch replay / RL (if available)
- include degradation-aware metrics
- explicitly state whether DT is:
  - better in mean reward
  - better in tail risk
  - better or worse in degradation cost

## What to Avoid in the Short Term
- Avoid mixing many AEMO regions before the SA1 pipeline is stable.
- Avoid claiming DT beats real dispatch replay after only prompt sweeps.
- Avoid selecting the best RTG prompt based on the final test set.
- Avoid comparing methods trained/evaluated under different degradation settings.
- Avoid mixing different episode lengths in the same headline table unless normalized.

## Practical Next Repo Tasks
1. Add an **episode-level split manifest** for AEMO trajectory generation.
2. Add an **AEMO DT training config** separate from the household config.
3. Save **training-data provenance** for each AEMO dataset build.
4. Add a **single AEMO evaluation notebook/report** dedicated to DT vs baselines.
5. Add a small **AEMO prompt-calibration study** on validation only.

## Success Criterion for the Next Iteration
A good next iteration is **not** “DT beats every baseline.”

A good next iteration is:
- the AEMO dataset is reproducible,
- the split is methodologically clean,
- the degradation setup is realistic,
- and DT can be compared against baselines under one fixed protocol.

Once that is in place, any competitiveness claim will be much more credible.
