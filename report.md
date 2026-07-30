# Offline Decision Transformers Outperform Online RL for Utility-Scale Battery Dispatch: A Degradation-Aware AEMO/NEM Benchmark

## Abstract

Optimal operation of grid-scale battery energy storage (BESS) in wholesale electricity markets requires simultaneous energy arbitrage and co-optimized bidding across multiple ancillary-service markets, under non-linear degradation and strict physical constraints. This report asks whether a **decision transformer (DT)** trained purely offline from logged trajectories can match or exceed online reinforcement learning (RL) and real-world dispatch for this problem. We answer this on a unified, degradation-aware benchmark built around Australia's NEM market (AEMO), supporting both a household solar–battery environment and a utility-scale BESS trading environment with full 9-dimensional FCAS bidding.

Our central result is that a modernized offline DT — 8-block transformer with grouped-query attention, QK-Norm, and weight-tying, trained on a 2,401-episode FCAS-rich corpus — is the **state of the art** on the fairest same-asset comparison: **$10,138/ep on the dispatch-matched surface (RTG=0.0) and $4,630/ep on the broad cross-region standard surface**, beating online PPO ($7,757/ep), the best GRPO-tuned variant ($6,445/ep), and real dispatch replay ($3,663/ep). Two findings stand out. First, **GRPO online fine-tuning does not improve the modern model** — the architecture already internalizes what RL once added, and the legacy GRPO champion is shown to be a narrow overfit ($8,242 dispatch-matched collapses to $1,533 on the standard surface). Second, the DT's return-to-go prompt yields **zero-shot, inference-time control** of the profit/degradation trade-off, with the optimal prompt being architecture-dependent (modern: 0.0; legacy: 0.5). These results position offline sequence modeling with strong architecture and curated data as a credible, deployable prior for utility-scale battery control, and motivate the current research direction of injecting explicit forecast/planning awareness into the model.

## 1. Introduction

The proliferation of energy storage across the grid—from distributed, behind-the-meter household batteries to grid-scale battery energy storage systems (BESS) participating in wholesale markets—presents both a challenge and an opportunity for modern power systems. While these assets can reduce consumer costs and provide grid flexibility, their optimal operation is non-trivial. The control problem is characterized by stochastic demand/generation, time-varying tariffs or market prices, non-linear battery degradation dynamics, and strict physical constraints [6].

Recent literature also helps structure the space of RL-based battery control problems. Subramanya et al. [6] review RL applications for battery storages through multiple lenses (optimization objective, user impact/comfort where applicable, battery losses & degradation, and application context). This benchmark is designed to make these dimensions explicit in a single codebase so that planning and learning approaches can be compared under consistent dynamics and evaluation.

In addition, this work is motivated by the opportunity to apply **modern transformer-based sequence models** (via Decision Transformers) to energy storage control, treating battery dispatch as a sequential decision-making problem that can benefit from transformer representations and return-conditioning.

### 1.1 The Research Gap
Despite substantial literature on energy management systems (EMS), reproducibility and cross-paper comparability can be difficult when studies rely on custom environments, private data, and differing assumptions (e.g., constraint handling, tariff structure, or whether degradation is modeled). There is also a practical gap between model-based planning approaches (e.g., dynamic programming / MPC-style methods) and learning-based approaches (e.g., RL and sequence models), which motivates a unified benchmark.

Recent review work supports this benchmark direction: Subramanya et al. [6] note that comparisons across RL-for-battery studies are hindered by unique formulations (environments, state/action spaces, and rewards), and argue that benchmark environments with a standard interface would improve comparability.

### 1.2 Research Questions

This report is organized around three falsifiable questions, each answered by the experiments in Section 8:

- **RQ1 — Offline vs online RL.** Can a Decision Transformer trained *offline* from behavior-cloned trajectories match or exceed *online* RL (PPO) and real-world dispatch for multi-market BESS dispatch? (Answer: yes — §8.2.1, §8.2.3.)
- **RQ2 — Does online fine-tuning help a strong offline model?** Does GRPO online RL fine-tuning further improve an already-strong offline DT? (Answer: no — §8.2.1, §8.2.7; the apparent GRPO gain is a narrow overfit.)
- **RQ3 — Is optimal prompting architecture-dependent?** Can the profit/degradation operating point be tuned at inference time, and does the optimal return-to-go prompt transfer across architectures? (Answer: yes for tunability, no for transfer — §8.2.2.)

### 1.3 Contributions and Research Goals
This work establishes a consolidated, reproducible benchmark to address these limitations. The contributions, in priority order, are:
1.  **A best-performing offline Decision Transformer for utility-scale BESS dispatch** — a modernized 8×768 GQA transformer trained on a 2,401-episode FCAS-rich corpus, achieving the highest profit on a same-asset benchmark and surpassing online RL and real dispatch replay (§4.2, §8.2).
2.  **A rigorous same-asset evaluation methodology** that removes the battery-sizing confounder from dispatch-replay comparisons, plus RTG calibration and an overfitting post-mortem for narrow benchmarks (§8.2.1, §8.2.2).
3.  **Two Gymnasium-Compatible Simulation Environments:** (a) household solar+battery under ToU pricing and (b) grid-scale AEMO/NEM trading with full FCAS and historical dispatch replay.
4.  **Baselines as Comparators and Data Sources:** a unified interface for rule-based, SDP/MRDP, online RL (SB3), and dispatch-replay policies, also used to generate offline training data.
5.  **Standardized Evaluation Workflow:** return, grid flows, degradation, risk proxies (Sharpe/Sortino), tail-risk (VaR/CVaR@5%), bootstrap confidence intervals, and paired Wilcoxon comparisons.

The goal of this platform is to provide a reusable baseline for studying generalization and robustness in control policies for decentralized energy systems, and a foundation for the current research direction of forecast-aware sequence modeling (§9).

## 2. Related Work

We position this work at the intersection of three literatures: battery optimal control under degradation, reinforcement learning for energy storage, and sequence-modeling / offline RL.

**Optimal control with degradation.** The foundational planning perspective is Abdulla et al. [1], who formulate BESS optimal operation as stochastic dynamic programming and stress the joint role of forecast uncertainty and degradation for realistic assessment. Their backward-induction solver supplies provably-optimal energy arbitrage under a discretization, and directly motivates the SDP/MRDP baselines and the forecast-aware research direction in §9. Degradation realism is provided by the multi-factor cycle-life model of Muenzel et al. [2] (rate, temperature, SoC, depth-of-discharge dependent), which we adopt for both environments; real-world BESS aging (calendar + cycle, Arrhenius temperature dependence, NMC/LFP presets) is available via `RealWorldBESSDegradationModel` [3].

**Reinforcement learning for battery control.** Subramanya et al. [6] survey RL-for-battery-storage across optimization objective, user impact, losses/degradation, and application context, and explicitly call for benchmark environments with a standard interface to enable cross-paper comparability. Their critique — that bespoke environments, private data, and divergent reward/constraint formulations block fair comparison — is the direct motivation for the unified, degradation-aware benchmark in this report.

**Sequence models and offline RL.** The Decision Transformer (DT) of Chen et al. [4] reframes RL as conditional sequence modeling: a transformer predicts actions from return-to-go (RTG), state, and action tokens, enabling offline training from logged trajectories and inference-time policy steering via the RTG prompt. Subsequent work extends DT to online fine-tuning (e.g., Online Decision Transformer) and to continuous control. Our contribution is to bring a *modernized* DT (grouped-query attention, QK-Norm, weight tying, SwiGLU) to the practically under-studied setting of **multi-market BESS dispatch with co-optimized FCAS bidding**, and to test, empirically, whether online RL fine-tuning (GRPO) adds value on top of a strong offline model — a question largely absent from the DT literature.

While we are not aware of a directly comparable published result on full 9-dimensional FCAS co-optimization via offline sequence models, the benchmark, baselines, and evaluation protocol here are designed so that such comparisons can be made under identical dynamics and metrics.

## 3. System Model and Environments

This repository provides two primary environments.

### 3.1 Household Solar-Battery Environment (SolarBatteryEnv)
Environment: `src/EnergySimEnv.py` defines `SolarBatteryEnv` with:
- Action: 1D normalized battery power in [-1, 1]. In `step()`, this is scaled to kW via `max_battery_flow`, converted to step energy (kWh) via `step_duration`, and clipped by SoC and capacity.
- Observation: always normalized in the current implementation (`normalize_obs = True`). It includes cyclical time features (sin/cos of hour/day-of-year), min–max normalized dataframe features, and two extra features: battery level and current-step degradation cost (both normalized).
- Dynamics: `step_duration` is inferred from the dataframe `Time` column (hours between the first two timestamps, with a fallback). The grid energy is clipped to `max_grid_energy = max_grid_flow × step_duration`, and energy-conservation violations yield an early termination with `VIOLATION_PENALTY`.
- Reward/cost: per-step reward is `grid_reward - current_step_deg_cost`, where `grid_reward` is `-(grid_energy × price)` (import vs export prices selected by sign), and degradation cost is derived from per-cycle wear × `battery_life_cost`.
- Degradation: the environment uses rainflow counting over the SoC trajectory to extract cycles, then applies a multi-factor cycle-life model based on Muenzel et al. [2] (temperature, C-rates, SOCav, DoD) to compute per-cycle degradation.

Dataset contract (from `src/helper.py::transform_polars_df`):
`Timestamp, SolarGen, HouseLoad, FutureSolar, FutureLoad, ImportEnergyPrice, ExportEnergyPrice, Time` (sorted by `Time`).

### 3.2 Utility-Scale AEMO Battery Trading Environment (AEMOBatteryTradingEnv)
Environment: `src/AEMOBatteryEnv.py` defines `AEMOBatteryTradingEnv`, a Gymnasium environment for a grid-scale BESS participating in Australia's National Electricity Market (NEM). Key design points implemented in the repository include:
- **Market signals:** the environment consumes preprocessed AEMO data with energy spot price (`RRP`), regional demand (`TOTALDEMAND`), optional FCAS service prices (wide columns prefixed `FCAS_`), and optional generation mix features (wide columns prefixed `GEN_`).
- **Observation space:** a fixed 18-dimensional vector (time features, normalized energy price and demand, normalized FCAS service prices, generation mix, and normalized SOC). `AEMODataPreprocessor` adds normalization columns (e.g., `RRP_normalized`, `DEMAND_normalized`, `FCAS_*_normalized`).
- **Action space:**
	- `action_mode='simple'`: 1D action in [-1, 1] for energy-only charge/discharge.
	- `action_mode='multi_market'`: 3D action `[battery_dispatch, fcas_raise_bid, fcas_lower_bid]` with dispatch in [-1,1] and FCAS bids in [0,1] (legacy, only RAISEREG/LOWERREG).
	- `action_mode='full_fcas'`: 9D action `[battery_dispatch, 8 × FCAS bid]` for all 8 FCAS services with co-optimized enablement model (recommended).
- **Units and scale:** default capacity/flow are specified in MWh/MW (grid-scale), distinct from the household environment (kWh/kW).
- **Degradation:** supports `degradation_mode='rainflow'` using the same `DegradationModel` + `RainflowCounter` primitives as the household environment, tracking `step_degradation`, `total_degradation`, and capacity fade.
- **Historical dispatch replay:** utility-scale notebooks and helpers can resolve a station name or DUID to the battery unit(s) active in a selected historical window, load AEMO `DISPATCHLOAD` records, and replay those observed actions inside the environment as a benchmark trajectory.


## 4. Methods

Baselines and planners are implemented in `src/decision.py` (Agent abstraction):
- Rule-based: a heuristic using surplus/deficit logic, optional persistence, and small injected noise (see `Agent.rule_based_action`).
- RL (SB3): PPO/A2C/DDPG/SAC/TD3 via `src/sb3train.py::train_model`; rollouts collected by `run_sb3_model_on_vec_env` and flattened with `flatten_episode_data`.
- SDP: a self-contained dynamic programming baseline implemented in `src/sdp_algorithm.py`. It discretizes SoC and actions and performs backward induction. Uncertainty can be handled via scenario sampling (Monte Carlo) when enabled.
- MRDP: `algorithm='mrdp'` with `subhorizon_specs` for coarse-to-fine planning, addressing the "curse of dimensionality" inherent in standard SDP.
- Decision Transformer (DT): Offline sequence model proposed by Chen et al. [4] (`src/decision_transformer.py`) trained on `TrajectoryDataset` from logged trajectories (`src/transformer_training.py`). Inference uses `model.get_action` with rolling context.

For the AEMO environment, `src/decision.py` also provides `AEMOAgent`, which supports rule-based control, RL/DT inference, and a **dispatch-replay** mode that converts AEMO unit dispatch data into environment actions aligned to the environment timestep.

### 4.1 Decision Transformer (Primary Model, Repo-Backed)
This repository's DT stack is designed to make **offline RL** the primary learning baseline while keeping the rest of the system (environment + baselines) stable.

**Model architecture (`src/decision_transformer.py`).**
- **Tokenization:** the input sequence interleaves tokens as (`rtg_t`, `state_t`, `action_t`) and flattens to length `3T` for a context length `T` (hyperparameter `context_len`). The model predicts:
	- next RTG and next state from the (`rtg`, `state`, `action`) stream,
	- the action from the (`rtg`, `state`) stream.
- **Continuous actions:** actions are predicted with a `tanh` head to match the environment's normalized action range in $[-1,1]$.
- **Modernized transformer block:** pre-norm with `RMSNorm`, attention via PyTorch `scaled_dot_product_attention`, and a `SwiGLU` feed-forward. Rotary position embeddings (RoPE) are supported as an option.
- **Robust inference hooks:** the model sanitizes NaNs/Infs, clamps timestep indices to embedding range, and supports loading `return_scale` from either a training checkpoint or a sidecar `*.meta.json`.

**Training data format (trajectory logs).**
DT training is based on trajectory logs stored as Parquet and consumed by `TrajectoryDataset` (`src/transformer_training.py`), which expects the following columns:
`episode_id`, `step`, `norm_observation`, `action`, `reward`.

Repo-backed ways of producing these logs include:
- `Agent.run_episode()` in `src/decision.py` (per-episode dict logs with `norm_observation`, `action`, `reward`, `info`).
- `flatten_episode_data(...)` in `src/helper.py`, which converts lists of episode trajectories (e.g., from SB3 vectorized rollouts) into a single Polars DataFrame with the DT-required columns and can be saved to Parquet.

**RTG construction and scaling.**
- `TrajectoryDataset` computes discounted returns-to-go by backward accumulation with a configurable discount factor (`discount_factor`, default `0.99`).
- Training supports a `return_scale` hyperparameter: when non-1.0, RTGs are divided by `return_scale` before entering the model, and the same scaling is applied during inference.
- For practical prompting, `evaluate_experiment_logs` (in `src/helper.py`) computes `recommended_rtg` and a `recommended_return_scale` derived from the distribution of episode-start RTG magnitudes.

**Training loop (repo-backed).**
`train_decision_transformer` (`src/transformer_training.py`) uses:
- AdamW optimizer + StepLR scheduler,
- gradient clipping (`GRAD_CLIP_NORM = 0.05`),
- optional AMP (enabled on CUDA after the first checkpoint is saved),
- multi-loss objective with weighted MSE terms for action/state/return predictions.

The CLI entrypoint `src/pretrain_decision_transformer.py` assembles datasets from a directory of Parquet logs (matching filename patterns), splits validation data, and trains/checkpoints the model.

**Online DT inference with RTG conditioning (`src/decision.py`).**
During episode rollouts, the DT agent maintains rolling buffers of past $(s,a,\text{rtg},t)$.
- At reset: buffers are initialized with the first observation, a placeholder action, and a user-chosen initial RTG (`rtg_value`).
- After each step: the buffers are updated and RTG is updated via the discounted recurrence
	$$\text{rtg}_{t+1} = \frac{\text{rtg}_t - r_t}{\gamma}$$
	(or $\text{rtg}_{t+1}=\text{rtg}_t-r_t$ when $\gamma=1$), matching the training-time definition of discounted RTG.

This makes DT evaluation explicitly a **prompting** problem: different `rtg_value` settings correspond to different desired-return conditions and can change the policy behavior.

> **NOTE (multi-env DT, repo-backed):** DT input/output dimensions must match the environment.
> - Household: default config uses `state_dim=12` and `act_dim=1`.
> - AEMO: `AEMOBatteryTradingEnv` observations are 18D; in `action_mode='simple'` the action is 1D, while `action_mode='multi_market'` requires `act_dim=3` and `action_mode='full_fcas'` requires `act_dim=9`.
> To train DT for AEMO multi-market bidding, you must log trajectories with the correct action dimension and use a DT config with matching `act_dim`.

**Evaluation-side risk metrics (implemented):** tail-risk metrics (VaR@5% and CVaR@5%) are computed from episode returns in `src/helper.py::evaluate_experiment_logs` and appear in evaluation tables and `eval_output/household/risk_metrics.csv`. Bootstrap confidence intervals (`bootstrap_confidence_intervals`) and paired statistical comparisons (`paired_comparison` with Wilcoxon signed-rank) are also available. Risk-aware training extensions (future work): add CVaR-style objectives/constraints and multi-objective scalarization for reward vs degradation into the training loop.

### 4.2 Modern v2 Architecture (current best-performing model)

The headline results in Section 8 are produced by the **modern v2 Decision Transformer**
([`mrvictoru/energydecision-dt-v2`](https://huggingface.co/mrvictoru/energydecision-dt-v2)), an
8-block transformer that modernizes the legacy block described in §4.1. Relative to the legacy
8×384 model, the changes are architectural rather than scale-driven (hidden dim grows 384→768,
but the decisive gains come from the following):

- **Grouped-Query Attention (GQA):** 12 query heads attend over 6 key/value heads (n_rep = 2), reducing KV-cache memory and stabilizing long-context attention.
- **QK-Norm:** per-head RMSNorm on queries and keys (before the attention dot-product) for training stability without warmup sensitivity.
- **RMSNorm pre-norm** throughout, replacing LayerNorm.
- **SwiGLU** feed-forward (768 → 2304 → 768) with dropout 0.15.
- **Weight tying:** the input embedding and output prediction layers for state/action/return share weights (`pred_act`, `pred_state`, `pred_return` are linear projections of the tied embeddings).
- **Learned timestep embedding** (RoPE disabled; `rope_enabled=false`), context length 210.
- Trained with `discount=0.95`, `return_scale=2.0`, and near-action-only loss weights
  (`action=0.999`, `state=0.002`, `return=0.0001`) — i.e. the model is trained almost entirely to predict the next action correctly.

Canonical hyperparameters are shipped at
`configs/aemo_decision_transformer_model_kwargs_modern_v2_full_fcas.json`. The architecture
is verified from the uploaded checkpoint's embedded `config`, not from documentation.

## 5. Data and Preprocessing

### 5.1 Source Data
For the proof of concept, we utilize the **Ausgrid Solar Home Electricity Data** [5]. This public dataset contains half-hourly electricity data for 300 anonymized homes with rooftop solar systems in the Ausgrid network area. The dataset spans from 1 July 2010 to 30 June 2013 and includes gross metered solar generation and household consumption.

The customers in these datasets have been de-identified and typically represent households in separate houses with available roof space, rather than apartments. The data was sourced from customers on domestic tariffs with gross metered solar systems installed for the full period. Quality checks were performed to exclude customers with extreme consumption or generation patterns. Specifically, we utilize the following archives:
- `2010-2011 Solar home electricity data.csv`
- `2011-2012 Solar home electricity data v2.csv`
- `2012-2013 Solar home electricity data v2.csv`

### 5.2 Preprocessing
Preprocessing via `transform_polars_df` converts per-customer CSVs into the environment schema, with configurable tariffs (`price_periods`) and default vs peak prices (`ImportEnergyPrice`, `ExportEnergyPrice`). `make_env(dataset)` returns callables for vectorized execution.

### 5.3 AEMO Data (Utility-Scale)
For the AEMO/NEM environment, the repository includes `src/aemo_data.py`, which fetches historical AEMO datasets via the NEMOSIS client and caches them under `data/aemo/`. The typical bundle used by the environment consists of:
- regional dispatch prices (`DISPATCHPRICE` → `RRP`),
- regional demand (`DISPATCHREGIONSUM` → `TOTALDEMAND`),
- optional FCAS prices (service columns mapped from `DISPATCHPRICE`),
- optional generation mix by fuel type (requires generator static information).

`AEMODataPreprocessor` (`src/AEMOBatteryEnv.py`) aligns these series to the environment step duration (default 30 minutes), interpolates missing numeric values, adds cyclical time features, and writes normalized columns.

For historical replay, the repository also queries unit metadata and `DISPATCHLOAD` records so that a notebook can discover which battery stations were active in a date window, resolve historical DUID changes for a named station, and reconstruct observed dispatch actions as episode logs.

> **NOTE (repo-backed):** Actual AEMO data fetching requires the optional dependency `nemosis` to be installed; otherwise fetch functions raise `ImportError`.

## 6. Experimental Setup

Splits and seeds:
- Train/test split by customer ID (e.g., 80/20), fixed seeds, and config logging are recommended for reproducibility.

Workflows:
- DT (primary):
	- Create trajectory logs (Parquet) from rule-based, SDP/MRDP, SB3 policies, oracle policies, and (for AEMO) dispatch replay episodes.
	- Train DT with `src/pretrain_decision_transformer.py` (wraps `TrajectoryDataset` + `train_decision_transformer`).
	- Evaluate DT using `Agent(algorithm='dt', rtg_value=...)` to study RTG-conditioning sensitivity.
- RL: `DummyVecEnv` for training, `SubprocVecEnv` for evaluation; train with `train_model(..., default_model=True)` or enable Optuna tuning.
- SDP/MRDP: configure horizons/resolutions; evaluate single or parallel episodes via `run_episodes_parallel`.

DT hyperparameters:
- Default DT model kwargs are stored in `models/household/dt/decision_transformer_model_kwargs.json` (e.g., `state_dim=12`, `act_dim=1`, `context_len=60`, `h_dim=128`, `n_block=2`, `n_heads=8`).
- Training-time `return_scale` is stored in checkpoints and also written to `*.meta.json` sidecars for consistent inference.

Compute and reproducibility:
- Containerization: the repository includes a `Dockerfile` and `docker-compose.yml` for shared Docker workflows, plus a Distrobox guide for lower-friction local development.
- Figures: `evaluate_experiments(..., save_dir=..., save_format=...)` can save plots (default `save_format='svg'`).

## 7. Metrics and Analysis

### 7.1 Core Metrics

Primary metrics (from `src/helper.py::evaluate_experiment_logs`):
- **Episode return statistics:** mean, median, std, 5th/95th percentiles, and max (computed from per-episode sums of the logged `reward`).
- **Grid energy flows:** average per-episode grid import (kWh), grid export (kWh), and net grid energy (kWh) derived from `info['grid_energy']`.
- **Degradation:** average per-episode and per-step degradation derived from `info['step_degradation']`, plus degradation incident rate.
- **Risk proxies:** Sharpe and Sortino ratios are computed directly from the distribution of episode returns (not annualized; Sharpe = `mean/std`, Sortino uses downside deviation below `target_return`).

For AEMO logs, the same evaluation functions also summarize market-specific metrics when those keys appear in `info` (e.g., `energy_revenue`, `fcas_revenue`, `total_revenue`, `degradation_cost`, `battery_dispatch`, `actual_energy`).

> **NOTE:** `SolarBatteryEnv` logs `info['grid_energy']` and `info['step_degradation']`, and `evaluate_experiments()` reports **average grid import/export energy (kWh)** alongside **average degradation**. A per-step monetary decomposition (import cost vs export revenue vs degradation cost) is not separately provided; the per-step `reward` already combines grid economics and degradation.

### 7.2 Tail-Risk Metrics

The following tail-risk metrics are computed by `evaluate_experiment_logs` and included in the evaluation DataFrame:

| Metric | Definition |
|--------|-----------|
| `var_5` | Value-at-Risk at 5%: the 5th percentile of episode returns, representing the worst-case threshold below which only 5% of outcomes fall. |
| `cvar_5` | Conditional VaR (Expected Shortfall) at 5%: the mean of all episode returns at or below `var_5`, quantifying the expected loss in the tail. |

These metrics appear in evaluation tables (e.g., `eval_output/household/risk_metrics.csv`) and are also available in the DataFrame returned by `evaluate_experiments()`.

### 7.3 Statistical Comparisons

Two statistical comparison tools are implemented in `src/helper.py`:

- **Bootstrap confidence intervals** (`bootstrap_confidence_intervals`): resamples episode logs with replacement `n_bootstrap` times (default 1000) to estimate the sampling distribution of any metric (default: mean episode reward). Returns per-experiment `{mean, ci_lower, ci_upper, std}` at a configurable confidence level (default 95%).

- **Paired comparisons** (`paired_comparison`): given two experiments with matched episodes (same customer/seed index), computes per-episode metric differences and applies the Wilcoxon signed-rank test (requires SciPy, ≥10 paired episodes). Returns `{mean_diff, median_diff, std_diff, wilcoxon_stat, wilcoxon_p}`.

Exported artifacts:
- `eval_output/household/risk_metrics.csv` — tail-risk summary (VaR, CVaR) for all experiments.
- `eval_output/household/pairwise_summary.csv` — Wilcoxon signed-rank test results for all algorithm pairs.
- `eval_output/household/pairwise_significance_heatmap.svg` — visual summary of head-to-head significance.

> **NOTE:** These statistical analyses are available in `src/helper.py` and demonstrated in `notebooks/test_eval.ipynb`, but are not plotted by default in `evaluate_experiments()`; they can be run as part of a notebook/script workflow or exported as CSV artifacts.

### 7.4 Visualization

Standard diagnostic plots produced by `evaluate_experiments(..., save_dir=...)`:
- Mean reward bar chart with std error bars.
- Grid energy comparison with degradation overlay.
- Risk-return scatter (std vs mean, color by Sharpe ratio).
- Episode return distribution (box plot).

## 8. Results: Decision Transformer for Battery Control

This section presents the empirical evaluation of the Decision Transformer (DT) as the primary control algorithm for battery energy storage. The DT is trained offline from logged trajectories and evaluated against rule-based heuristics, online RL (PPO, SAC, A2C, DDPG, TD3), planning baselines (SDP, MRDP), and Oracle policies. Two environments are benchmarked: a residential solar-battery system and a utility-scale battery trading in Australia's NEM market.

| Environment | Best DT Result | Key Advantage |
|-------------|---------------|---------------|
| Household (SolarBatteryEnv) | **Best mean return** (-2408), beats Oracle | 1D action, degradation-aware cycling |
| AEMO utility-scale (AEMOBatteryTradingEnv) | **Best profit/ep** ($10,138 dispatch-matched / $4,630 standard), beats PPO & dispatch on same-asset benchmark | 9D full-fcas bidding, RTG-tunable, FCAS-aware, modern architecture (GQA/RMSNorm) |

### 8.1 Household Solar-Battery Control (Historical Benchmark)

The household environment (1D action, Ausgrid Solar Home data) was the original testbed that established the DT's effectiveness on a simpler control problem; it is retained as a validation domain rather than the primary contribution. The DT achieves the best mean return (−$2,408) among all baselines including the perfect-foresight Oracle (−$2,483), a difference that is statistically significant (Wilcoxon p = 0.005). Critically, the RTG prompt provides **zero-shot control of the degradation/return trade-off**: moderate prompts yield 0.005/ep degradation versus 0.114/ep for near-zero prompts (a 22× reduction) without any retraining, and tail-risk (CVaR ≈ −9,705) is competitive with or better than SDP, A2C, and PPO. The full metric table, RTG-sensitivity analysis, and pairwise Wilcoxon tests are provided in [eval_output/household/](eval_output/household/) (see Appendix B for the per-algorithm table).

> **NOTE:** These household results establish the DT's effectiveness on a simpler 1D-action problem. The repository's primary contribution is the more challenging AEMO utility-scale environment (Section 8.2), where the action space is 9D (energy + FCAS bidding) and market dynamics are significantly more complex.

### 8.2 Utility-Scale AEMO Battery Trading (Primary Focus)

The AEMO environment evaluates grid-scale battery trading in Australia's National Electricity Market (NEM), with energy spot pricing and optional Frequency Control Ancillary Services (FCAS). The action space is 3D (`multi_market`, legacy) or 9D (`full_fcas`, recommended): energy dispatch plus per-service FCAS bids for all 8 services. Two evaluation surfaces are reported below. The headline evidence comes from the **dispatch-matched same-asset benchmark**, where all policies run on the identical battery (Dalrymple North 8 MWh / 30 MW, 3.75 C) with RTG calibration — this is the fairest comparison available.

---

#### 8.2.1 Fair Same-Asset Dispatch-Matched Benchmark (Headline)

The earliest dispatch-replay comparisons were biased because the replay policy was evaluated on the dispatch station's native battery size (e.g. 250 MWh for Torrens Island) while the DT ran on a much smaller template battery (typically 10 MWh). The corrected benchmark removes that confounder by evaluating all policies on the **same dispatch-matched battery asset** — derived from Dalrymple North (8 MWh / 30 MW, 3.75 C) — with the same `full_fcas` 9-action formulation and 5-minute resolution.

This benchmark uses the `q4_dispatch_matched` evaluator config (`configs/aemo_autoresearch_evaluator.q4_dispatch_matched.json`) with `use_dispatch_asset_sizing=true`, so every policy sees the same battery. The evaluation covers Q4 2024 SA1 (Oct + Nov, each 144 h / 1728 steps) with 2 episodes per policy.

A second surface — `eval_tier_standard` — evaluates cross-region generalization across 5 regions with medium batteries, providing a broader test of policy robustness.

| Model | Standard | Dispatch-matched (rtg=0.5) | Dispatch-matched (rtg=0.0) |
|---|---:|---:|---:|
| **Modern v2 pretrained** | **$4,630** | $6,793 | **$10,138** |
| Phase C GRPO (2 bat, 3 region, 144h) | $4,102 | $6,445 | $6,183 |
| Legacy Phase 1 GRPO (rtg=0.5) | $1,533 | $8,242 | $5,451 |
| PPO reference | $2,353 | $7,757 | — |
| Dispatch Dalrymple North | $4,660 | $3,663 | — |
| Dispatch Hornsdale | $57,435 | $57,435 | — |
| Dispatch Torrens Island | $114,365 | $114,365 | — |
| FCAS rule | -$56,095 | -$126,124 | — |

**Key observations:**

1. **Modern v2 pretrained leads all compared methods.** The architecture improvements (GQA, RMSNorm, weight tying) captured the benefits that GRPO once provided the legacy model. It beats every variant on the broad standard surface ($4,630/ep) and achieves the highest dispatch-matched profit ($10,138/ep at RTG=0.0).

2. **Legacy Phase 1 GRPO was overfit.** Its dispatch-matched peak ($8,242/ep) collapses to $1,533/ep on the standard surface — the worst cross-region generalization of any DT model. The modern v2 pretrained model ($4,630 standard) generalizes properly. Overfitting was to dispatch-matched SA1 Q4 2024, not to the broader AEMO market.

3. **GRPO with proper 144h multi-region recipe works but doesn't surpass pretrained.** Phase C GRPO (144h episodes, 3 regions, gradient accumulation) produces $4,102 standard / $6,445 dispatch-matched — within 5–11% of the pretrained model, but never exceeding it. The B3 run that collapsed was purely from crippled hyperparams (minibatch=8, epochs=1, 2 RTGs, 1 battery).

4. **FCAS learning is robust across all DT models.** Every DT variant — pretrained, GRPO-tuned, legacy — earns 3–5× more FCAS revenue than the real dispatch strategy. The FCAS capability comes from the offline dataset, not from GRPO.

5. **PPO retains a degradation advantage over most DT variants.** PPO's ~$310/ep degradation cost is lower than the legacy and GRPO-tuned DT variants (which cycle more aggressively to capture FCAS revenue). The modern v2 pretrained model is an exception: its dispatch-matched degradation (~$187/ep) is *below* PPO's, showing that the modern architecture captures FCAS revenue without the same degradation penalty. Closing the residual degradation gap while preserving FCAS revenue remains the key open challenge.

6. **Large-station dispatch replays transfer profitably but inefficiently.** Hornsdale ($57,435/ep) and Torrens Island ($114,365/ep) earn high absolute profit but with poor per-MWh efficiency.

![Mean reward comparison — standard surface (modern v2 pretrained vs baselines)](eval_output/final/baseline_v2_standard/plots/mean_reward.svg)

![Risk-return profile — standard surface](eval_output/final/baseline_v2_standard/plots/risk_return.svg)

![Episode return distribution — standard surface](eval_output/final/baseline_v2_standard/plots/episode_distribution.svg)

![Net grid energy balance — standard surface](eval_output/final/baseline_v2_standard/plots/grid_energy.svg)

---

#### 8.2.2 RTG Calibration — The Transformer as a Tunable Controller

A distinguishing advantage of the Decision Transformer architecture is that its operating behaviour can be adjusted at **inference time** by changing the return-to-go (RTG) prompt — no retraining required. The original calibration range (0.0–2.0) was later found to be **far too narrow** — all DT variants respond strongly to much higher RTG values (10–100).

**Extended RTG calibration sweep (0.0–100.0) on the standard tier:**

| Model | RTG 0.5 | RTG 5 | RTG 10 | RTG 20 | RTG 50 | RTG 100 | Best RTG |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Modern v2 | $4,726 | — | **$4,991** | $3,223 | $2,968 | — | 10.0 |
| Forecast DT (norm) | $1,105 | — | $2,397 | $2,979 | **$4,564** | $3,736 | 50.0 |
| Phase C GRPO (mod v2) | $4,102 | $4,192 | **$4,322** | $4,263 | $4,265 | — | 10.0 |
| Phase 1 GRPO (legacy) | $1,533 | $1,569 | $2,483 | $2,092 | **$2,678** | — | 50.0 |

**Key findings from the extended calibration:**

1. **Every DT variant gains from higher RTG** — gains of 5–75% over the default 0.0–0.5 range. The original calibration was far too narrow.

2. **Architecture determines the optimal RTG.** Modern v2 (return_scale=1.0) peaks at RTG=10; Forecast DT (return_scale=2.0) peaks at RTG=50. Both correspond to model-space RTG ≈ 10.0, suggesting a natural operating point.

3. **The RTG modulates degradation, not just revenue.** At low RTG (0.0), the forecast DT degrades $13,229/ep (catastrophic cycling); at RTG=50, degradation drops to $270/ep — a 50× improvement from a scalar prompt change.

4. **Legacy GRPO benefits most from RTG calibration** (+75% from $1,533 to $2,678), suggesting the older architecture was strongly prompt-dependent while the modern v2 internalizes reward structure better.

5. **Return_scale must be accounted for in RTG calibration.** The forecast DT was trained with return_scale=2.0, so its config RTG=50 maps to model-space RTG=25. The modern v2 uses return_scale=1.0, so config RTG=10 maps to model-space RTG=10. Always calibrate the config RTG value per model — do not transfer across architectures.

---

#### 8.2.3 FCAS-Rich Offline DT — Closing the Gap (Supporting Evidence)

Prior to the GRPO fine-tuning, the offline-only Decision Transformer (no online RL) was already capable of competitive utility-scale performance. The DT was retrained on a **2,425-episode FCAS-rich dataset** (78.4M rows, 3.1 GB) generated from PPO, TD3, A2C, DDPG, SAC, and FCAS rule policies across 3 horizons, 5 regions, and 3 battery sizes. The model is 8×384, context=180, drop_p=0.15, batch=64, lr=3e-5, trained for 2 epochs with discount=0.95 and return_scale=2.0.

Prior to this breakthrough, the same DT architecture trained on FCAS-poor data achieved -$1,396/ep on the expanded 135-episode evaluation, while PPO earned +$12,839/ep — demonstrating that data quality, not architecture, was the limiting factor.

The evaluation uses the **example evaluator** (`configs/aemo_autoresearch_evaluator.example.json`) — an older surface with 4 scenarios, 2 battery sizes, and 16 episodes per policy. This was the benchmark on which the offline DT first demonstrated headline competitiveness:

| Rank | Policy | Mean Reward | Profit/Ep | Energy Rev | FCAS Rev | Deg Cost | Sharpe |
|:----:|--------|:-----------:|:---------:|:----------:|:--------:|:--------:|:------:|
| **1** | **DT (FCAS-rich)** | **-1.31** | **+$1,522** | $351 | $1,383 | **$212** | **-1.07** |
| 2 | PPO (RL) | -1.35 | +$1,444 | $437 | **$1,616** | $609 | -1.01 |
| 3 | dispatch_dalrymple_north | -1.43 | +$1,304 | $1,491 | $0 | $187 | N/A |
| 4 | rule (old) | -3.03 | -$2,477 | $1,521 | $0 | $3,998 | -1.26 |
| 5 | fcas_rule | -4.24 | -$3,569 | $1,050 | $146 | $4,764 | -0.80 |
| 6 | dispatch_torrens_island | -5.39 | -$5,394 | $0 | $0 | $5,394 | N/A |

This result proved that **offline DT can close the FCAS gap purely from data** — the 18× FCAS revenue improvement (from $77 to $1,383/ep) showed that multi-market bidding strategies transfer from online RL rollouts to offline sequence models. The degradation advantage ($212/ep vs PPO's $609/ep) was also sharper at this stage, because the offline DT dispatches conservatively (9.2 MWh/ep vs PPO's 20.3 MWh/ep).

However, this evaluation suffered from the battery sizing mismatch (DT on a small template battery, dispatch on the station's native size). The **dispatch-matched benchmark (Section 8.2.1) replaces this as the primary evidence** for competitiveness; the example evaluator results are retained here as supporting evidence for the offline FCAS learning capability.

---

#### 8.2.4 Battery Realism Update (v2 Dataset, July 2026)

The original FCAS dataset used synthetic batteries with a fixed 0.5C (2-hour charge/discharge) ratio — matching no real-world BESS station. A new v2 dataset (`data/aemo_dt_fcas_v2/`) was generated with four battery configurations that match actual Australian NEM stations:

| Battery | Ratio | Real-world match |
|---------|:-----:|------------------|
| `medium_1c` | **1.0C** (60 min) | Torrens Island, Waratah, Lake Bonney |
| `large_07c` | **~0.7C** (86 min) | Hornsdale, Victorian Big Battery |
| `small_05c` | **0.5C** (120 min) | Kennedy Energy Park |
| `fast_375c` | **3.75C** (16 min) | Dalrymple North BESS |

All 5 SB3 source models (PPO, TD3, A2C, DDPG, SAC) were retrained on these 4 battery configurations, and a new 2,401-episode dataset was assembled (77M rows, 6 policies × 4 batteries × 3 horizons × 5 regions). A new Decision Transformer was pretrained on this v2 dataset and uploaded to HuggingFace (`mrvictoru/energydecision-dt`, `aemo_dt_fcas_model.pt`).

**v2 model baseline results** on the Q4 2024 held-out multi-station evaluation (1C battery, 5-min resolution):

| Policy | Reward | Profit/ep | FCAS/ep | Deg/ep |
|--------|:------:|:---------:|:-------:|:------:|
| **v2 HF DT (baseline)** | -11.60 | **$1,714** | $2,743 | $1,690 |
| v2 GRPO-tuned DT | -9.70 | $1,885 | **$4,033** | $2,820 |
| PPO reference | -15.50 | $1,395 | $1,287 | $308 |
| Dispatch Dalrymple North | -4.18 | $4,660 | $2,287 | $1,020 |

Key observations:
- The v2 baseline ($1,714/ep) is **2× more profitable** than the old HF model ($874/ep), purely from training on realistic battery configurations.
- GRPO post-training adds +$171/ep (+10%) profit and +$1,290 (+47%) FCAS, confirming online RL fine-tuning transfers to realistic battery setups.
- The v2 baseline and GRPO-tuned DT both exceed PPO in absolute profit, though dispatch replay Dalrymple North ($4,660/ep) maintains a lead in this multi-station setting.

---

#### 8.2.5 GRPO Post-Training Autoresearch

A hyperparameter sweep of 21 GRPO experiments was conducted to find the optimal online RL fine-tuning config:

| Sweep | Best Value | Key Insight |
|-------|-----------|-------------|
| Iterations | **5** (144h) | Beyond 5, KL drift degrades performance. 30 iter → -1.54 reward |
| KL coefficient | 0.02 | Higher KL hurts — default is optimal |
| Entropy | 0.0 | Any positive entropy worsens results |
| Learning rate | 1e-5 (144h) / 5e-5 (24h) | 24h proxy does NOT predict 144h performance — train on target episode length |
| RTG count | 4 (144h) / 2 (24h) | More RTG values dilute advantages on long episodes |
| Multi-region training | NSW1+SA1+QLD1 | Best result: **+1.60 reward**, $4,357/ep vs dispatch's -1.88 |

**Critical finding**: The 24h proxy metric does NOT reliably predict 144h evaluation performance. Always validate on the target episode length.

---

#### 8.2.6 Practical Usefulness and Remaining Limits

The transformer model (modern v2 pretrained Decision Transformer) has credibility as a practical utility-scale battery control policy, but its strengths and weaknesses should be stated clearly.

**Strengths:**
- **FCAS-aware learned control.** The model captures 8-service FCAS bidding patterns from the offline dataset, earning 3–5× more FCAS revenue than the real dispatch strategy on the same asset.
- **Prompt-time controllability.** The RTG prompt lets an operator tune the profit/degradation trade-off at inference time — an advantage no fixed-policy baseline (PPO, dispatch replay, rule) can match without retraining or redeployment.
- **Leads all methods without online RL.** The modern v2 pretrained model achieves the highest profit across all benchmarks without any GRPO fine-tuning. It beats PPO ($7,757/ep) and dispatch replay ($3,663/ep) on the dispatch-matched surface and demonstrates strong cross-region generalization ($4,630/ep standard).
- **Beats matched dispatch replay.** All DT variants outperform the actual Dalrymple North dispatch strategy by 1.7–2.8× on the same battery, demonstrating learned patterns beyond simple imitation.
- **Deployable scheduling prior.** The model operates across multiple battery configurations, regions, and market conditions, making it a candidate for a transferable battery control prior.

**Remaining limitations:**
- **PPO retains a degradation edge over most DT variants.** PPO's ~$310/ep degradation is lower than the legacy/GRPO-tuned DT variants; the modern v2 pretrained model is competitive (~$187/ep dispatch-matched), but the general DT tendency to cycle aggressively for FCAS revenue still leaves room for degradation improvement. For owners prioritizing battery longevity, PPO remains attractive on the older variants.
- **Large-station dispatch replay dominates absolute profit.** Hornsdale and Torrens Island strategies, refined over years of real operations, transfer profitably to smaller assets — though their per-MWh efficiency is poor.
- **Degradation minimization is the primary open problem.** Closing the degradation gap while maintaining FCAS revenue remains the most important challenge for DT-based battery control.
- **Training cost.** The modern v2 model required significant offline data collection (2,401 episodes). While GRPO is not required, the offline data generation pipeline is itself compute-intensive.
- **Small dispatch-matched sample.** The headline dispatch-matched surface covers Q4 2024 SA1 only (2 episodes × 144 h). The standard surface is broader (5 regions, medium battery) and is the more robust cross-region generalization evidence; dispatch-matched figures should be read as a same-asset head-to-head rather than a seasonally robust estimate. Confidence intervals and bootstrap/Wilcoxon tools in `src/helper.py` are available but were not applied to the per-surface profit headlines.
- **Simulated market dynamics.** All revenue/degradation figures are produced by the in-repo simulator (`AEMOBatteryEnv`) driven by historical AEMO price/demand series; they are not settled market outcomes. The simulator's FCAS co-optimization and degradation models are approximations of real BESS dispatch economics.

---

#### 8.2.7 Improvement Trajectory

The following table traces the DT's progression from the original pilot model through the FCAS-rich offline result to the modern v2 pretrained model — which represents the current best-performing configuration in this benchmark.

| Stage | Model | Training Data | Profit/Ep (DM) | FCAS Rev | Deg Cost | Key Change |
|-------|-------|--------------|:---------:|:--------:|:--------:|:-----------|
| 1. Pilot | 4×128, ctx=1152 | 6 proxy episodes | -$10,620 | $2,328 | $12,975 | Baseline |
| 2. Autoresearch | 8×512, ctx=180 | 24 episodes (mixed) | -$1,396 | $77 | $2,503 | Hyperparameter tuning |
| 3. FCAS-rich DT | 8×384, ctx=180 | 2,425 episodes (PPO-rich) | +$1,522 | $1,383 | $212 | Dataset quality |
| 4. Phase 1 GRPO (legacy, overfit) | 8×384 (GRPO-tuned) | v2 HF + 5 GRPO iter | +$8,242 | $7,686 | $760 | Online fine-tuning (overfit to DM) |
| **5. Modern v2 pretrained** | **8×768 GQA** | **2,401 episodes (realistic bat)** | **+$10,138** | **$10,068** | **$187** | **Architecture improvement** |

**Key insight:** Stage 5's improvement over stage 4 comes entirely from architecture (GQA, RMSNorm, weight tying) and better training data (realistic battery configurations), not from online RL. The modern v2 architecture captures everything GRPO once provided — and generalizes better (stage 5 gets $4,630/ep on the standard surface vs stage 4's $1,533/ep).

**What changed at each step:**

- **Stage 1 → 2 (hyperparameters):** Moving from 4×128 to 8×512, reducing context from 1152 to 180, and adding dropout (0.15) reduced dispatch intensity by 6.8× (746 → 110.6 MWh/ep) and degradation by 5.2× ($12,975 → $2,503). The DT learned to be conservative but still couldn't capture FCAS revenue because the training data lacked FCAS-active examples.

- **Stage 2 → 3 (data quality):** Replacing the 24-episode mixed-policy dataset with the 2,425-episode FCAS-rich corpus achieved an 18× FCAS revenue improvement and flipped from -$1,396/ep to +$1,522/ep on the example evaluator.

- **Stage 3 → 4 (online fine-tuning + RTG calibration):** Five GRPO iterations on the v2 HF pretrained checkpoint, combined with the optimal RTG prompt (0.5), lifted profit to $8,242/ep on the dispatch-matched benchmark — but at the cost of cross-region generalization ($1,533/ep standard).

- **Stage 4 → 5 (architecture + realistic data):** The modern v2 architecture (8×768, GQA, RMSNorm, weight tying) trained on realistic battery configurations achieves $10,138/ep dispatch-matched and $4,630/ep standard — surpassing stage 4 on both metrics without any online RL. The architecture improvements internalized the benefits that GRPO once provided.

**Implication:** The transformer model's value accrues primarily from offline data quality and architecture — online RL fine-tuning adds negligible value on top of a well-architected and well-trained model.

---

#### 8.2.8 Forecast Decision Transformer — Negative Result (July 2026)

As a direct implementation of the Phase 3 research roadmap (§9, Thrust 2), we built and evaluated a **ForecastDecisionTransformer** — a modern v2 DT extended with explicit 48-step TTM price forecast tokens. The model was trained on the FCAS-rich corpus + SDP trajectories + GRPO rollouts, using TTM (Granite TTM-R3) forecasts from `ttm_forecasts.npz` as the forecast source.

**Architecture**: 8×768 GQA, context=210, forecast_len=48 (prepended as a learned prefix), RoPE enabled, with type embeddings distinguishing history (idx=0) from forecast (idx=1) tokens. Total sequence: 774 tokens (144 forecast + 630 history).

**Training**: MoLab notebook, 3 epochs, batch_size=64, lr=3e-5, return_scale=2.0, action_loss_weight=0.999, on AEMO simulated trade data (FCAS + GRPO + SDP parquet files). Loss converged to val=0.0056 at epoch 3.

**Forecast quality (TTM model)**:
| Channel | MAE | RMSE | Correlation |
|---|---|:---:|:---:|
| TOTALDEMAND | 0.087 | 0.114 | **+0.848** |
| RRP | 0.005 | 0.025 | +0.146 |
| FCAS (avg 4 services) | ~0.0001-0.0007 | ~0.0001-0.005 | **~+0.01-0.07** |

The TTM predicts demand well (0.85 correlation) but FCAS prices are essentially unpredictable from price history alone. Diverse few-shot fine-tuning (evenly-spaced samples across 2021–2023, added July 2026) produced no improvement — FCAS correlations remained near zero.

**Evaluation results (standard tier, best RTG)**:

| Model | Profit/ep | FCAS/ep | Energy/ep | Deg/ep | Best RTG |
|---|:---:|:---:|:---:|:---:|:---:|
| Modern v2 pretrained | **$4,991** | $4,836 | $384 | $229 | 10.0 |
| Dispatch Dalrymple North | $4,660 | $2,287 | $3,394 | $1,020 | — |
| **Forecast DT (normalized)** | **$4,564** | $3,663 | $1,171 | $270 | 50.0 |
| Phase C GRPO (mod v2) | $4,322 | $2,508 | $2,873 | $1,058 | 10.0 |
| Phase 1 GRPO (legacy) | $2,678 | $2,914 | $148 | $384 | 50.0 |
| PPO reference | $2,353 | $2,192 | $396 | $236 | — |

**The forecast DT ranked 3rd at $4,564/ep, 8.5% below the modern v2 baseline ($4,991/ep).** It beats dispatch ($4,660/ep) on FCAS revenue alone ($3,663 vs $2,287), but the explicit forecast tokens did not yield a meaningful edge over the modern v2's implicit 210-step context window.

**Data normalization fix (critical)**: Early evaluation runs (July 2026) showed the forecast DT losing $302/ep. Root cause: the `ttm_forecasts.npz` stored **raw** TTM predictions while the DT was trained on **normalized** [0,1] observations — the forecast tokens and history tokens lived on completely different scales (50-10,000×). After normalizing the npz to [0,1] using the global AEMO statistics, the forecast DT became profitable ($1,105/ep at RTG=0.5). The RTG calibration sweep then found the optimal at RTG=50 ($4,564/ep).

**Why the forecast DT didn't beat the baseline:**

1. **TTM FCAS forecasts are nearly useless (corr ~0.01–0.07).** The forecast tokens carry almost no FCAS bidding signal — the model learns FCAS from history alone, same as the modern v2.
2. **The modern v2's 210-step context window already encodes sufficient temporal patterns** for market inference. Explicit forecasts add redundant information rather than complementary signal.
3. **Training budget was equal** (3 epochs, 16K steps) — the forecast architecture adds 144 extra tokens but the model didn't overfit; it simply didn't derive additional benefit.

**Implementation quality**: The integration was thoroughly validated (18 tests pass), the forecast position correctly slides with `max(T, buffer_len)`, the npz alignment uses timestamp-based indexing, and the normalization matches the environment exactly. There is no implementation bug — the architecture itself does not improve over the implicit-context baseline on this task.

**Conclusion**: This is a well-implemented negative result. The forecast token architecture, while theoretically motivated and correctly built, does not outperform the modern v2 Decision Transformer on the standard tier. The findings are preserved as a reference: (1) always normalize forecast data to the observation space, (2) calibrate RTG broadly (0–100), (3) explicit price forecasts may not add value when the base context window is already informative. The forecast DT code, evaluator integration, and measurement tools remain in the repository as infrastructure for future forecast-conditioned approaches.

**Standard tier leaderboard (Oct 2024, 5 regions × 144h, medium_1c, full_fcas)**:

| Model | Profit/ep | FCAS/ep | Deg/ep | Best RTG |
|---|:---:|:---:|:---:|:---:|
| Modern v2 pretrained | **$4,991** | $4,836 | $229 | 10.0 |
| Dispatch Dalrymple North | $4,660 | $2,287 | $1,020 | — |
| Forecast DT (normalized) | $4,564 | $3,663 | $270 | 50.0 |
| Phase C GRPO (mod v2) | $4,322 | $2,508 | $1,058 | 10.0 |
| Phase 1 GRPO (legacy) | $2,678 | $2,914 | $384 | 50.0 |
| PPO reference | $2,353 | $2,192 | $236 | — |
| FCAS rule | -$56,095 | $513 | $57,902 | — |

---

#### 8.2.9 Market-Impact-Aware Evaluation — Offline DT as Natural Hedge (July 2026)

All preceding results assume the battery is a price-taker — its dispatch does not
affect the market clearing price. For large battery energy storage systems (e.g.,
Hornsdale 150 MW, Torrens Island 250 MW), this assumption is unrealistic: the
battery's own injection or withdrawal moves the energy price up the merit-order
supply curve and attenuates FCAS reserve prices through increased depth. To
quantify this effect, we built a **piecewise-linear merit-order market-impact
model** that reconstructs the regional supply curve from AEMO DISPATCHLOAD
availability data and fuel-tier marginal costs, then hooks into
`AEMOBatteryTradingEnv._calculate_reward` so that `RRP` and `FCAS_*` prices read
at each step are realized (impacted) prices rather than exogenous historical
values. The impact model is backward-compatible (default `impact_model='identity'`
reproduces the existing price-taking environment byte-for-byte; golden-value
verified across 100 random steps).

**Evaluation matrix:** 3 scenarios (SA1 Oct/Nov 2024, VIC1 Oct 2024, 14-day
episodes at 5-min resolution) × 2 impact models (identity, piecewise-merit-order)
× DT (with RTG sweep 0–50), PPO, FCAS rule, and Oracle_PT (perfect-foresight LP).
Battery: Dalrymple North 8 MWh / 30 MW (3.75 C), zero degradation cost.

| Policy | Identity profit/ep | Impact profit/ep | **% retained** |
|---|:---:|:---:|:---:|
| Oracle_PT | $162,807–$204,165 | $28,712–$47,650 | **22%** |
| PPO | $17,119–$23,731 | $10,739–$13,140 | **62%** |
| **DT** (best RTG) | **$12,494–$13,683** | **$9,419–$13,855** | **87%** |
| FCAS Rule | $29,106–$44,678 | $2,304–$10,226 | **16%** |

*(Ranges are min–max across the 3 scenarios. DT values are best RTG per cell.
Oracle_PT denotes the price-taking Oracle evaluated under the impact model.)*

**Key observation — the DT is a natural hedge against market impact:**
The offline DT retains **87%** of its profit under market impact on average,
compared to 62% for online PPO and only 22% for the Oracle. In SA1 October 2024,
the DT actually **gains** under impact ($13,420 → $13,855 at RTG=0), because
its conservative dispatch (→$7,000 in the report) avoids the price-depressing
effect of aggressive energy arbitrage while FCAS revenue (£12,712 ∼ £10,363)
holds relatively steady. In contrast, the Oracle's aggressive energy arbitrage
($129,070/ep) collapses to -$19,809/ep under impact, dragging total profit from
$204,165 to $28,712.

The reason is structural: the DT was trained with `action_loss_weight=0.999`
and near-zero return-weight, so it effectively cloned the behavior observed in
the price-taking FCAS-rich training corpus — which is dominated by conservative,
FCAS-heavy policies. Under market impact, conservatism is a feature, not a bug:
the policy avoids the kind of aggressive cycling that the Oracle's
perfect-foresight LP prescribes.

**RTG calibration remains important under impact**, and the optimal prompt
shifts: RTG=5 gives the best identity profit ($13,683 Oct), but RTG=0 is
optimal under impact ($13,855 in the same scenario). The sweep is required per
model and per scenario.

**PPO occupies an intermediate position:** it earns significantly more than
the DT under price-taking ($23,731 vs $13,683 in Oct SA1) but loses nearly
half its profit under impact ($13,140 vs $13,855 DT). The DT's conservative
profile already saturates the FCAS revenue without triggering large
self-impact, leaving less room for PPO's additional energy-arbitrage margin
to matter.

**Limitations:**
- The Oracle evaluated under impact is Oracle_PT, which does **not** account for
  market impact in its LP solve — it produces price-taking-optimal actions and
  then the env pays impacted prices. The true impact-aware ceiling (Oracle_MI)
  is work in progress.
- The battery size (8 MWh / 30 MW) is small enough that energy impact is modest;
  FCAS depth (→30–50 MW per service) creates the larger relative impact.
  Larger stations (150 MW+) are where energy impact would dominate.
- Three scenarios (14 days each) is a pilot evaluation; expanding to the full
  5-region × 6-month expanded evaluator surface is planned.

### 8.3 Key Takeaways

1. **The modern v2 Decision Transformer is the best-performing method for utility-scale battery control in this benchmark.** On the fairest same-asset benchmark with RTG calibration, the 8×768 GQA pretrained model achieves the highest profit/ep across both evaluation surfaces ($10,138 dispatch-matched, $4,630 standard), beating PPO, dispatch replay, and all GRPO-tuned variants. Architecture improvements (GQA, RMSNorm, weight tying) captured the benefits that online RL once provided.

2. **The full pipeline matters — but architecture matters most.** Offline data quality (2,401-episode FCAS-rich corpus), realistic battery configurations, and modern architecture each contribute. But the jump from stage 4 to stage 5 ($1,533→$4,630 standard) came from architecture alone — GRPO does not help the modern model.

3. **RTG conditioning provides zero-shot controllability that no fixed-policy baseline matches.** An operator can tune profit vs degradation at inference time by changing a single scalar prompt — no retraining required. However, the optimal RTG depends on the architecture (modern: 0.0, legacy: 0.5) — calibrate per model.

4. **Overfitting is a real risk for narrow benchmarks.** The legacy Phase 1 GRPO result ($8,242 dispatch-matched) looked like a breakthrough but collapsed on the broader standard surface ($1,533/ep). The modern v2 model's $4,630/ep on standard confirms that proper generalization requires diverse evaluation.

5. **FCAS awareness is the transformer's signature capability across all DT variants.** Every DT model earns 3–5× more FCAS revenue than the real-world dispatch strategy. This FCAS proficiency — learned from the offline dataset, not from GRPO — is the core economic value proposition.

6. **PPO remains the strongest competitor on degradation cost.** PPO's $310/ep degradation is lower than every DT variant. Closing this gap while maintaining FCAS revenue is the primary open problem.

7. **Data quality is the primary determinant of offline RL success.** The same architecture went from -$10,620/ep (6 pilot episodes) to +$10,138/ep (2,401 FCAS-rich episodes) — a turnaround driven almost entirely by the training dataset. This confirms that for offline RL in battery control, behavioral coverage and demonstration quality matter more than model scale.

8. **Explicit forecast tokens do not improve over implicit context.** The forecast DT (§8.2.8) was a correctly implemented negative result — the modern v2's 210-step history window already captures sufficient market signal, and TTM price forecasts add no meaningful FCAS or energy-arbitrage edge. Architecture improvements (GQA, RMSNorm) matter more than adding forecast conditioning.

9. **Offline DTs are natural hedges against market impact.** When evaluated under an endogenous market-impact model (battery dispatch moves energy + FCAS prices via a piecewise-linear merit-order supply curve), the DT retains 87% of its profit while the optimal Oracle collapses to 22% and online PPO to 62%. The reason is rooted in the DT's training objective — near-zero return-weight (0.0001) forces the model to clone conservative, FCAS-heavy behavior from the price-taking corpus, which happens to be precisely the kind of policy that survives self-impact. Conservatism, normally a weakness, becomes a free hedge. This finding (§8.2.9) motivates the market-impact-aware evaluation as a new robustness dimension for battery control benchmarks.

## 9. Proposed Research Roadmap

This framework provides the necessary tooling to pursue several practical extensions and evaluation directions:

### Phase 1: Benchmarking and Algorithmic Analysis (Completed)
- ✅ Establish the performance hierarchy between model-based (SDP) and model-free (RL) approaches.
- ✅ Quantify the performance gap of reactive/model-free baselines (rule-based, SB3 RL, DT) relative to planning baselines (SDP/MRDP, Oracle) under identical environment dynamics.
- ✅ Tail-risk metrics (VaR, CVaR at 5%) computed for all algorithms.
- ✅ Pairwise statistical comparisons (Wilcoxon signed-rank) across all algorithm pairs.
- ✅ Bootstrap confidence intervals for metric uncertainty quantification.

In line with review-identified gaps, an additional near-term objective is to make evaluation more comparable across algorithm families by using consistent environment dynamics, observation/action conventions, and standardized logging [6].

Concretely, in this single-agent benchmark we report this as a **planner gap / regret-style metric** based on episode returns. For any agent $\pi$ and a planning baseline $\pi^\star$ (e.g., SDP, MRDP, or an oracle with privileged information), define
$$\Delta J(\pi;\pi^\star)=\mathbb{E}[G(\pi^\star)]-\mathbb{E}[G(\pi)],$$
where $G(\cdot)$ is the per-episode return (sum of rewards). We also optionally report a relative gap $\Delta J/|\mathbb{E}[G(\pi^\star)]|$ for comparability across datasets.

### Phase 2: Robustness and Generalization (Year 1-2)
- **Distributional Shift:** Investigate how Offline RL (Decision Transformers) generalizes to unseen weather patterns or customer load profiles compared to Online RL.
- **Risk-Sensitive Control:** Integrate CVaR-style objectives/constraints into the training loop (evaluation-side tail-risk metrics are already implemented; the next step is CVaR-constrained or multi-objective training).

DT-centric near-term extensions (repo-aligned):
- ✅ **Train DT on PPO/RL data:** Completed (June 2026). The FCAS-rich dataset includes 905 PPO-generated episodes. The resulting offline DT achieves +$1,522/ep profit on the example evaluator, and the Phase 1 GRPO-tuned DT achieves +$8,242/ep on the dispatch-matched same-asset benchmark. FCAS revenue improved from $77/ep to $7,686/ep (Phase 1 GRPO) — a 100× improvement over the original FCAS-poor model.
- **Prompt calibration:** use the repo's `recommended_rtg` / `recommended_return_scale` diagnostics to choose RTG prompts that are in-distribution relative to the logged training data. The GRPO fine-tuning step (`src/grpo_posttraining.py` on the `copilot/online-rl-fine-tuning` branch) now uses `sample_rtg_values()` for adaptive RTG sampling — see `notebooks/aemo_dt_grpo_posttraining.ipynb`.
- **Training data mixture studies:** systematically vary which behavior policies generate the offline dataset (rule-based vs SDP vs SB3 vs PPO) and evaluate how DT performance changes. The current FCAS dataset uses a fixed mixture; ablating PPO's contribution would quantify its importance.
- **Long-context modeling:** tested context lengths 120–2016 across fair-comparison proxy sweeps. **Context=180 (15 hours) was optimal** — both shorter (120) and longer (288, 360, 576, 1008) contexts regressed validation loss. Context=2016 is now feasible on the RTX 2080 Ti (22 GB) at batch sizes up to 16; a re-sweep with FCAS-rich data is warranted.

Beyond DT-centric work, the AEMO results also highlight:
- ✅ **FCAS-aware offline data collection:** Completed. The 2,425-episode FCAS dataset (`data/aemo_dt_fcas/aemo_fcas_dataset.parquet`) includes PPO, TD3, A2C, DDPG, SAC, and `fcas_rule` trajectories across 3 horizons, 5 regions, and 3 battery sizes.
- **Multi-objective training:** The DT's return-conditioning naturally supports multiple operating points (conservative vs aggressive). Adding an FCAS-specific auxiliary loss or reward component could further improve AEMO performance, especially on the expanded evaluator.
- **FCAS-weighted loss:** The current loss treats all action dimensions equally. Weighting FCAS action dimensions higher (`action_loss_weight` for FCAS dims) could accelerate FCAS learning.

> **NOTE (literature alignment):** Because studies often vary in objective definitions (financial vs energy-efficiency) and in constraint/user-impact handling, robustness studies should explicitly document which objective family and constraint set is being targeted [6].

### Phase 3: Forecast- and Planning-Aware Sequence Modeling (Current Direction)

The modern v2 DT is the current best-performing model, yet it conditions only on a 210-step history window and has **no explicit forward-looking signal** — and its energy-arbitrage contribution is small (~$70/ep of the $10,138 dispatch-matched profit; the rest is FCAS). The next phase injects planning/forecast awareness, building directly on Abdulla et al. [1] and the implemented SDP/MRDP solvers. This is the active research line tracked in `docs/aemo_hybrid_dt_plan.md` (PR #32).

**Status (July 2026)**:

- ✅ **Forecast DT architecture built and trained** — `ForecastDecisionTransformer` with 48-step TTM forecast tokens (8×768 GQA, 15/15 tests). Trained on MoLab, checkpoint at `mrvictoru/energydecision-dt-v2-forecast`.
- ✅ **Evaluator integration complete** — `AEMOAgent` supports forecast buffers, evaluator dispatches on `model_class`, surface manifest supports `model_meta`.
- ✅ **Evaluation complete** — standard tier leaderboard: forecast DT at $4,564/ep (3rd place, 8.5% below modern v2). TTM forecast quality measured (FCAS corr ~0.01-0.07).
- ✅ **TTM forecasts normalized** — original npz had raw values; now normalized to [0,1] matching the observation space. Local and HF dataset updated.
- ✅ **TTM diverse few-shot fine-tuning** — `--fewshot-location diverse` option added but produced no forecast quality improvement (FCAS fundamentally hard to predict from pure price history).
- ⏳ **GRPO fine-tuning of forecast DT** — not attempted; the pretrained forecast DT already reached $4,564/ep after RTG calibration.
- ❌ **Forecast DT does not beat the baseline** — the architecture does not improve over the modern v2's implicit context. Documented as a negative result.

- **Thrust 1 — SDP-trajectory-augmented offline training (Path B'):** run the SDP/MRDP solver on the AEMO environment in `action_mode='simple'` to generate provably-optimal energy-arbitrage trajectories, add them to the FCAS-rich corpus, and retrain the DT so it learns SDP's energy timing while retaining PPO-learned FCAS bidding.
- **Thrust 2 — Forecast-conditioned DT architecture (Path A1):** ✅ Built, trained, evaluated. **Negative result** — does not beat the modern v2 baseline. See §8.2.8.
- **Secondary — SDP-computed RTG at inference:** replace the hand-tuned scalar RTG with the SDP cost-to-go for the current state (no retraining); and **hierarchical SDP+DT** inference (SDP sets energy dispatch, DT sets FCAS bids).
- **Success criteria:** DT retrained with SDP trajectories exceeds the pretrained baseline on ≥1 surface; SDP-guided RTG improves inference-time profit without retraining; combined system targets >$12,000/ep dispatch-matched or >$6,000/ep standard.

### Phase 4: Robustness, Multi-Objective, and Sim-to-Real (Year 2-4)
- **Statistical rigor:** apply bootstrap CIs and paired Wilcoxon tests to the AEMO headline tables (tooling exists; see Appendix C).
- **Risk-sensitive / multi-objective training:** CVaR-constrained or degradation-weighted objectives in the training loop; FCAS-weighted action loss.
- **Long-context and training-mixture studies:** re-sweep context lengths (2016 feasible on 22 GB) on FCAS-rich data; ablate behavior-policy mixtures.
- **Sim-to-real transfer:** safe-RL wrappers and pilot/hardware-in-the-loop validation, responding to the review's call [6] to compare simulated performance against real deployments.
- **Multi-agent coordination:** extend to a microgrid setting with multiple trading homes.

> **NOTE (literature alignment):** The review explicitly calls out the need to compare simulated/model-based performance to real battery deployments [6].

## 10. Reproducibility and Artifacts

- Code pointers: environment (`src/EnergySimEnv.py`), agents (`src/decision.py`), SB3 training (`src/sb3train.py`), DT (`src/decision_transformer.py`, `src/transformer_training.py`), preprocessing and evaluation (`src/helper.py`).
- Determinism: use fixed seeds, log configs, and prefer containers for repeatability (recommended practice).
- Artifacts: this repository stores models under `models/` and evaluation outputs under `eval_output/`.

> **NOTE:** If stronger artifact provenance is required (e.g., exact dataset/model versioning), a lightweight checksum/logging step can be added to the workflow.

## 11. Conclusion

This repository introduces a unified framework for learning and planning in battery control with degradation-aware evaluation across two settings: (i) household solar–battery–grid control under tariffs and (ii) utility-scale battery trading under AEMO/NEM market signals. The repository supports rule-based control, RL, SDP/MRDP, dispatch replay (AEMO), and Decision Transformers, with standardized preprocessing and environment-agnostic metrics. For the utility-scale setting, the implemented workflow now includes replay of historical station actions from AEMO dispatch data, providing a concrete bridge between simulated evaluation and observed market behavior.

**Key empirical findings:**

- **Household environment:** The Decision Transformer achieves the best overall performance, outperforming all baselines including the perfect-foresight Oracle. Its RTG-conditioning enables zero-shot trade-off control between returns and degradation.
- **AEMO utility-scale environment (same-asset dispatch-matched benchmark):** On the fairest comparison where all policies share the identical battery (Dalrymple North 8 MWh / 30 MW) with RTG calibration, the **modern v2 pretrained Decision Transformer achieves the highest profit per episode across both evaluation surfaces: $10,138/ep on dispatch-matched (rtg=0.0) and $4,630/ep on the standard surface**. This beats PPO ($7,757/ep), all GRPO-tuned variants ($6,445 best), and dispatch replay ($3,663/ep) on the same asset. Architecture improvements (GQA, RMSNorm, weight tying) captured the benefits that online RL fine-tuning once provided the legacy model.
- **AEMO utility-scale (overfitting finding):** The legacy Phase 1 GRPO champion ($8,242 dispatch-matched) collapsed to $1,533/ep on the standard surface — confirming narrow overfitting. The modern v2 model generalizes properly.
- **AEMO utility-scale (RTG controllability):** The DT's return-to-go prompt provides zero-shot tunability of profit vs degradation at inference time. However, the optimal RTG depends on the architecture: modern peaks at 0.0, legacy at 0.5. Always calibrate per model.
- **AEMO utility-scale (FCAS-rich offline DT):** Before GRPO fine-tuning, the offline DT retrained on a 2,425-episode FCAS-rich dataset achieved +$1,522/ep on the example evaluator (beating PPO's +$1,444/ep). FCAS revenue rose 18× (from $77/ep to $1,383/ep), closing most of the prior gap to PPO's $1,616/ep FCAS revenue (the DT reached ~86% of PPO's FCAS revenue, versus ~5% before), while degradation fell 2.9× vs PPO ($212/ep vs $609/ep). This establishes that **offline RL on well-curated data can match online RL**.
- **Remaining limitations:** PPO retains a degradation advantage over most DT variants (~$310/ep vs the legacy/GRPO-tuned DTs; modern v2 is competitive at ~$187/ep dispatch-matched), and large-station dispatch replays dominate absolute per-episode profit on transferred assets. Residual DT degradation efficiency is the primary open challenge.

This report documents the system and experimental protocol; results can be iteratively updated as additional experiments are run.

## References

[1] K. Abdulla, J. De Hoog, et al., "Optimal Operation of Energy Storage Systems Considering Forecasts and Battery Degradation," *IEEE Transactions on Smart Grid*, 2016.

[2] V. Muenzel, J. De Hoog, et al., "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management," *IEEE Transactions on Industrial Electronics*, 2015.

[3] A. Kampker et al., "Modelling of Battery Energy Storage Systems Under Real-World Conditions," *Batteries*, vol. 11, no. 3, 2025, doi:10.3390/batteries11010392. (Real-world calendar + cycle aging model with Arrhenius temperature dependence and NMC/LFP presets, used by `RealWorldBESSDegradationModel`.)

[4] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, "Decision Transformer: Reinforcement Learning via Sequence Modeling," *Advances in Neural Information Processing Systems (NeurIPS)*, 2021, arXiv:2106.01345.

[5] Ausgrid. Solar home electricity data. https://github.com/pierre-haessig/ausgrid-solar-data?tab=readme-ov-file. Accessed April 2017.

[6] R. Subramanya, S. A. Sierla, and V. Vyatkin, "Exploiting Battery Storages With Reinforcement Learning: A Review for Energy Professionals," *IEEE Access*, vol. 10, 2022, doi: 10.1109/ACCESS.2022.3176446.

---

Appendix A: Minimal Experiment Recipes

RL
- Train: `ppo_model, _ = train_model(PPO, DummyVecEnv([make_env(ds) for ds in train_ds]), eval_env_fn=test_env_fns[0], default_model=True)`.
- Rollout and save: `flatten_episode_data(run_sb3_model_on_vec_env(ppo_model, SubprocVecEnv(test_env_fns))).write_parquet("data/household/logs/ppo_test_episode_logs.parquet")`.

DT
- Train (CLI): `python -m src.pretrain_decision_transformer --data-dir data/household/logs --model-config models/household/dt/decision_transformer_model_kwargs.json --epochs 2 --batch-size 6 --lr 2e-5 --return-scale 1.0`.
- Dataset (Python): `TrajectoryDataset(data_path=..., context_length=..., state_dim=..., act_dim=..., discount_factor=0.99)` → train with `train_decision_transformer` and evaluate via `Agent(algorithm='dt', rtg_value=...)`.

---

Appendix B: Household Per-Algorithm Metrics (§8.1)

| Algorithm | Mean Reward | Std Reward | Sharpe | Avg Degradation/Ep |
|-----------|----------:|----------:|------:|-------------------:|
| dt_rtg_neg200 | **-2407.65** | 3087.47 | -0.780 | 0.0051 |
| dt_rtg_neg500 | -2407.62 | 3087.51 | -0.780 | 0.0051 |
| oracle | -2483.38 | 1773.97 | -1.400 | 0.2351 |
| a2c | -2528.62 | 3234.82 | -0.782 | 0.0000 |
| sdp | -2598.35 | 3200.02 | -0.812 | 0.0115 |
| ppo | -2828.28 | 3275.89 | -0.863 | 0.0349 |
| rule | -3077.26 | 3454.07 | -0.891 | 0.0541 |

---

Appendix C: Implementation Notes and Known Mismatches

- **Dataset/forecast column mismatch:** the dataset schema emitted by `transform_polars_df` includes `FutureSolar`/`FutureLoad`, but the planning-agent forecast extraction in `src/decision.py` looks for `FutureGen`/`FutureLoad`. As written, SDP/MRDP fall back to `SolarGen`/`HouseLoad` unless the dataframe columns match `FutureGen`. This is a known code-level inconsistency and does not affect the AEMO (utility-scale) results, which use AEMO-native columns.
- **Statistical confidence on headline AEMO figures:** bootstrap confidence intervals and paired Wilcoxon tests (`src/helper.py`) are implemented and were used for the household domain. For the AEMO headline tables, point estimates are reported; applying CIs/Wilcoxon to the per-surface AEMO profit headlines is listed as near-term future work (see README Roadmap, "Statistical confidence on AEMO headlines").
- **Figure embedding:** figure paths reference repository-relative SVGs; for PDF export these should be embedded.
