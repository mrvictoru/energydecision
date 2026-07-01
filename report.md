# Benchmarking and Advancing Control Strategies for Energy Storage: A Unified Framework Across Household Solar-Battery Control and Utility-Scale AEMO Battery Operation

## Abstract

The effective integration of battery energy storage is critical for a reliable, renewable-dominant grid, spanning both behind-the-meter residential operation and utility-scale market participation. Developing and comparing control strategies can be challenging when environments omit key factors such as stochastic demand/generation, time-varying tariffs or market prices, and battery degradation. This report documents a research codebase that provides two Gymnasium-compatible environments—(i) a household solar+battery controller and (ii) a utility battery trading environment for AEMO/NEM with an implemented historical dispatch-replay workflow—and a shared evaluation workflow.

The **primary learning model** in this codebase is an **offline Decision Transformer (DT)** trained from logged trajectories to produce continuous battery-charge/discharge actions conditioned on a desired return-to-go (RTG). A core motivation of this repository is to bring **modern transformer-based sequence modeling** to the practical challenge of battery operation, and to evaluate these models against established planning and RL baselines under consistent dynamics and metrics. Rule-based heuristics, dynamic-programming planners (SDP/MRDP), online RL baselines (Stable-Baselines3), and dispatch-replay baselines for the AEMO environment are included primarily as comparators and data-generators for DT training.

> **Key empirical finding:** On the household environment, the DT achieves state-of-the-art results (best mean return, beating Oracle). **On the utility-scale AEMO environment, the picture is nuanced:** on the large-scale expanded evaluation (135 episodes), PPO dominates with mean_reward = +12.82 vs the DT's -3.11 because the DT's offline training data lacked FCAS bidding patterns. However, when the DT is retrained on an FCAS-rich dataset (2,425 episodes including 905 PPO-generated trajectories), the **DT achieves the highest profit per episode (+$1,522/ep), beating PPO (+$1,444/ep)** on the example evaluator. FCAS revenue improves 18× (from $77 to $1,383/ep) and degradation is 2.9× lower than PPO. These results demonstrate that training DT on RL-generated trajectories successfully closes the FCAS gap, and that the choice between online and offline learning depends critically on the quality and coverage of the offline dataset.

## 1. Introduction

The proliferation of energy storage across the grid—from distributed, behind-the-meter household batteries to grid-scale battery energy storage systems (BESS) participating in wholesale markets—presents both a challenge and an opportunity for modern power systems. While these assets can reduce consumer costs and provide grid flexibility, their optimal operation is non-trivial. The control problem is characterized by stochastic demand/generation, time-varying tariffs or market prices, non-linear battery degradation dynamics, and strict physical constraints [6].

Recent literature also helps structure the space of RL-based battery control problems. Subramanya et al. [6] review RL applications for battery storages through multiple lenses (optimization objective, user impact/comfort where applicable, battery losses & degradation, and application context). This benchmark is designed to make these dimensions explicit in a single codebase so that planning and learning approaches can be compared under consistent dynamics and evaluation.

In addition, this work is motivated by the opportunity to apply **modern transformer-based sequence models** (via Decision Transformers) to energy storage control, treating battery dispatch as a sequential decision-making problem that can benefit from transformer representations and return-conditioning.

### 1.1 The Research Gap
Despite substantial literature on energy management systems (EMS), reproducibility and cross-paper comparability can be difficult when studies rely on custom environments, private data, and differing assumptions (e.g., constraint handling, tariff structure, or whether degradation is modeled). There is also a practical gap between model-based planning approaches (e.g., dynamic programming / MPC-style methods) and learning-based approaches (e.g., RL and sequence models), which motivates a unified benchmark.

Recent review work supports this benchmark direction: Subramanya et al. [6] note that comparisons across RL-for-battery studies are hindered by unique formulations (environments, state/action spaces, and rewards), and argue that benchmark environments with a standard interface would improve comparability.

### 1.2 Contributions and Research Goals
This work establishes a consolidated, reproducible benchmark to address these limitations. We provide:
1.  **Two Gymnasium-Compatible Simulation Environments:** Environments for (a) household solar+battery operation under ToU import/export pricing and (b) grid-scale battery trading under AEMO/NEM market signals (energy + optional FCAS), including replay of historical utility-scale station actions.
2.  **Decision Transformer as the Primary Model:** A modernized Decision Transformer implementation plus an offline training pipeline built around trajectory logging, RTG construction, return scaling, and robust checkpoint loading.
3.  **Baselines as Comparators and Data Sources:** A unified interface for comparing rule-based heuristics, SDP/MRDP planners, online RL (PPO, SAC, etc.), and (for AEMO) dispatch replay against DT, and for generating trajectory data for offline learning.
4.  **Standardized Evaluation Workflow:** Metrics for return, grid energy flows, degradation, risk proxies (Sharpe/Sortino ratios), tail-risk analysis (VaR/CVaR at 5%), bootstrap confidence intervals, and paired statistical comparisons (Wilcoxon signed-rank), plus plotting utilities.

The goal of this platform is to provide a reusable baseline for studying generalization and robustness in control policies for decentralized energy systems.

## 2. Related Work

This work is inspired by Abdulla et al. [1], which formulates optimal operation of energy storage using Stochastic Dynamic Programming (SDP) and emphasizes the importance of uncertainty and degradation for realistic assessment.

We adopt an SDP-style planning baseline (implemented in this repository) and a multi-factor degradation model based on Muenzel et al. [2] (implemented in `src/batterydeg.py`).  Alongside these planning components, the Decision Transformer framework [4] motivated our implementation of a transformer‑based sequence model trained with offline RL; this becomes the primary learning baseline in the codebase.  We extend the planning baseline with additional learning‑based baselines and a Gymnasium environment wrapper.

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
	- `action_mode='multi_market'`: 3D action `[battery_dispatch, fcas_raise_bid, fcas_lower_bid]` with dispatch in [-1,1] and FCAS bids in [0,1].
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
> - AEMO: `AEMOBatteryTradingEnv` observations are 18D; in `action_mode='simple'` the action is 1D, while `action_mode='multi_market'` requires `act_dim=3`.
> To train DT for AEMO multi-market bidding, you must log trajectories with the 3D action and use a DT config with `act_dim=3`.

**Evaluation-side risk metrics (implemented):** tail-risk metrics (VaR@5% and CVaR@5%) are computed from episode returns in `src/helper.py::evaluate_experiment_logs` and appear in evaluation tables and `eval_output/household/risk_metrics.csv`. Bootstrap confidence intervals (`bootstrap_confidence_intervals`) and paired statistical comparisons (`paired_comparison` with Wilcoxon signed-rank) are also available. Risk-aware training extensions (future work): add CVaR-style objectives/constraints and multi-objective scalarization for reward vs degradation into the training loop.

> **NOTE (important repo mismatch):** The dataset schema includes `FutureSolar`/`FutureLoad` (see `transform_polars_df`), but the current planning-agent forecast extraction in `src/decision.py` looks for `FutureGen`/`FutureLoad`. As written, SDP/MRDP will fall back to using `SolarGen`/`HouseLoad` unless the dataframe columns match `FutureGen`.

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
| AEMO utility-scale (AEMOBatteryTradingEnv) | **Best profit/ep** (+$1,522), beats PPO | 3D multi-market bidding, 2.9× lower degradation |

### 8.1 Household Solar-Battery Control (Historical Benchmark)

The household environment was the original benchmark for this repository. The DT was evaluated against rule-based, SDP/MRDP, online RL (PPO, SAC, A2C, DDPG, TD3), and Oracle agents on the Ausgrid Solar Home dataset. Full metrics are in [eval_output/household/baseline/evaluation_metrics.csv](eval_output/household/baseline/evaluation_metrics.csv).

| Algorithm | Mean Reward | Std Reward | Sharpe | Avg Degradation/Ep |
|-----------|----------:|----------:|------:|-------------------:|
| dt_rtg_neg200 | **-2407.65** | 3087.47 | -0.780 | 0.0051 |
| dt_rtg_neg500 | -2407.62 | 3087.51 | -0.780 | 0.0051 |
| oracle | -2483.38 | 1773.97 | -1.400 | 0.2351 |
| a2c | -2528.62 | 3234.82 | -0.782 | 0.0000 |
| sdp | -2598.35 | 3200.02 | -0.812 | 0.0115 |
| ppo | -2828.28 | 3275.89 | -0.863 | 0.0349 |
| rule | -3077.26 | 3454.07 | -0.891 | 0.0541 |

![Household mean episode return by agent](eval_output/household/baseline/mean_reward.svg)

**Key takeaways:**
- **DT achieves best mean return:** `dt_rtg_neg200` (-2408) and `dt_rtg_neg500` (-2408) outperform all baselines including perfect-foresight Oracle (-2483). Difference vs Oracle is statistically significant (p = 0.005).
- **RTG prompt controls degradation:** Moderate prompts achieve 0.005/ep degradation vs 0.114/ep for near-zero prompt — a 22× difference. The DT enables zero-shot trade-off control without retraining.
- **Tail risk is competitive:** DT moderate-prompt CVaR (≈ -9705) is better than A2C (-9966), SDP (-9965), and PPO (-10089).
- **Full analysis:** RTG sensitivity, tail-risk metrics, and pairwise Wilcoxon tests are documented in [eval_output/household/](eval_output/household/).

> **NOTE:** These household results establish the DT's effectiveness on a simpler 1D-action problem. The repository's current focus is on the more challenging AEMO utility-scale environment (Section 8.2), where the action space is 3D (energy + FCAS bidding) and market dynamics are significantly more complex.

### 8.2 Utility-Scale AEMO Battery Trading (Primary Focus)

The AEMO environment evaluates grid-scale battery trading in Australia's National Electricity Market (NEM), with energy spot pricing and optional Frequency Control Ancillary Services (FCAS). The action space is 3D (`multi_market`): energy dispatch, FCAS raise bid, and FCAS lower bid.

#### 8.2.1 Closing the Gap — FCAS-Rich Dataset Training

This is the primary result. The DT was retrained on a **2,425-episode FCAS-rich dataset** (78.4M rows, 3.1 GB) generated from PPO, TD3, A2C, DDPG, SAC, and FCAS rule policies across 3 horizons, 5 regions, and 3 battery sizes. The model is 8×384, context=180, drop_p=0.15, batch=64, lr=3e-5, trained for 2 epochs with discount=0.95 and return_scale=2.0.

Prior to this breakthrough, the same DT architecture trained on FCAS-poor data achieved -$1,396/ep on the expanded 135-episode evaluation, while PPO earned +$12,839/ep — demonstrating that data quality, not architecture, was the limiting factor.

The evaluation uses the **example evaluator** (`configs/aemo_autoresearch_evaluator.example.json`) with 4 scenarios (NSW1 Jan 2024, SA1 Winter 2024, QLD1 Jan 2024, VIC1 Jan 2024), 2 battery sizes (medium, small), 2 episodes per variant, 144-hour episodes, and 8 parallel workers. This gives **16 episodes per policy** for DT/RL/rule/fcas_rule.

| Rank | Policy | Mean Reward | Profit/Ep | Energy Rev | FCAS Rev | Deg Cost | Sharpe |
|:----:|--------|:-----------:|:---------:|:----------:|:--------:|:--------:|:------:|
| **1** | **DT (FCAS-rich)** | **-1.31** | **+$1,522** | $351 | $1,383 | **$212** | **-1.07** |
| 2 | PPO (RL) | -1.35 | +$1,444 | $437 | **$1,616** | $609 | -1.01 |
| 3 | dispatch_dalrymple_north | -1.43 | +$1,304 | $1,491 | $0 | $187 | N/A |
| 4 | rule (old) | -3.03 | -$2,477 | $1,521 | $0 | $3,998 | -1.26 |
| 5 | fcas_rule | -4.24 | -$3,569 | $1,050 | $146 | $4,764 | -0.80 |
| 6 | dispatch_torrens_island | -5.39 | -$5,394 | $0 | $0 | $5,394 | N/A |

![Mean reward comparison — FCAS-rich dataset evaluation](eval_output/autoresearch/example_baseline_full_b64/plots/mean_reward.svg)

![Risk-return profile — FCAS-rich dataset evaluation](eval_output/autoresearch/example_baseline_full_b64/plots/risk_return.svg)

![Episode return distribution across policies](eval_output/autoresearch/example_baseline_full_b64/plots/episode_distribution.svg)

![Net grid energy balance by policy](eval_output/autoresearch/example_baseline_full_b64/plots/grid_energy.svg)

**Key observations:**

1. **DT achieves #1 profit per episode:** The DT earns **+$1,522/ep**, beating PPO (+$1,444/ep) by $78/ep (5% margin). This demonstrates that offline RL can match and exceed online RL when the training dataset captures the right behaviors.

2. **FCAS gap closed:** DT FCAS revenue is $1,383/ep vs PPO's $1,616/ep — only 14% behind. This is an **18× improvement** from the old DT's $77/ep. Training on PPO-generated trajectories (905 PPO episodes in the FCAS dataset) successfully transferred FCAS bidding behavior to the offline model.

3. **Degradation is DT's advantage:** DT degradation cost is $212/ep vs PPO's $609/ep — **2.9× lower**. This gentler battery operation is the primary reason DT beats PPO on total profit despite lower FCAS revenue. The DT dispatches only 9.2 MWh/ep vs PPO's 20.3 MWh/ep.

4. **Rule-based baselines are unprofitable:** Both the old rule (-$2,477/ep) and fcas_rule (-$3,569/ep) lose money, with the fcas_rule suffering from excessive cycling ($4,764/ep degradation). Neither approaches DT or PPO profitability.

5. **Dispatch replay shows real-world energy-only profit:** Dalrymple North achieves +$1,304/ep but only from energy arbitrage (zero FCAS revenue, as DISPATCHLOAD does not capture FCAS enablement). Torrens Island shows zero actions because it historically provided only contingency FCAS, which the `multi_market` action space cannot represent.

6. **Context=180 remains optimal:** This model uses the same context length confirmed by prior proxy sweeps. Longer contexts (288, 576, 1008) regressed validation loss in earlier experiments. Context=2016 is now feasible on 22 GB VRAM and warrants re-testing with FCAS data.

#### 8.2.2 From Gap to Leadership — The Improvement Trajectory

The following table traces the DT's progression from the original pilot model to the current FCAS-rich result. Each step represents a targeted intervention:

| Stage | Model | Training Data | Profit/Ep | FCAS Rev | Deg Cost | Key Change |
|-------|-------|--------------|:---------:|:--------:|:--------:|:-----------|
| 1. Pilot | 4×128, ctx=1152 | 6 proxy episodes | -$10,620 | $2,328 | $12,975 | Baseline |
| 2. Autoresearch | 8×512, ctx=180 | 24 episodes (mixed) | -$1,396 | $77 | $2,503 | Hyperparameter tuning |
| 3. **FCAS-rich** | **8×384, ctx=180** | **2,425 episodes (PPO-rich)** | **+$1,522** | **$1,383** | **$212** | **Dataset quality** |

**What changed at each step:**

- **Stage 1 → 2 (hyperparameters):** Moving from 4×128 to 8×512, reducing context from 1152 to 180, and adding dropout (0.15) reduced dispatch intensity by 6.8× (746 → 110.6 MWh/ep) and degradation by 5.2× ($12,975 → $2,503). The DT learned to be conservative but still couldn't capture FCAS revenue because the training data lacked FCAS-active examples.

- **Stage 2 → 3 (data quality):** The critical leap came from replacing the 24-episode mixed-policy dataset with the 2,425-episode FCAS-rich corpus. This dataset includes 905 PPO-generated episodes, 300 FCAS rule episodes, and episodes from 4 other RL algorithms. The same transformer architecture (8 layers, adjusted width from 512→384) with the same context length (180) achieved an 18× FCAS revenue improvement and flipped from -$1,396/ep to +$1,522/ep.

**Implication:** For offline RL in battery control, **dataset quality and behavioral coverage matter more than model scale**. A well-curated offline dataset with diverse, high-performing demonstrations can enable an offline model to match or exceed the online policy that generated the data.

### 8.3 Key Takeaways

1. **DT is an effective battery control algorithm across environments.** On the household benchmark, it beats Oracle. On the utility-scale AEMO benchmark (with FCAS-rich training), it achieves the highest profit per episode, beating PPO by 5%.

2. **Data quality is the primary determinant of offline RL success.** The same DT architecture went from -$1,396/ep to +$1,522/ep solely by improving the training dataset. Model scale and hyperparameters were secondary to behavioral coverage.

3. **Degradation-aware operation is a natural strength of offline RL.** The DT's conservative dispatch (9.2 MWh/ep vs PPO's 20.3 MWh/ep) yields 2.9× lower degradation cost while maintaining competitive revenue. This aligns with the practical need to preserve battery asset life.

4. **FCAS market participation can be learned from demonstrations.** The 18× FCAS revenue improvement ($77 → $1,383/ep) proves that multi-market bidding strategies transfer from online RL rollouts to offline sequence models. The remaining 14% gap to PPO ($1,383 vs $1,616) is a target for future work (FCAS-weighted loss, RTG calibration, expanded evaluation validation).

5. **RTG conditioning provides zero-shot controllability.** Unlike online RL, the DT can adjust operating aggressiveness at inference time by changing the RTG prompt — no retraining required. This was demonstrated on the household environment and is applicable to AEMO.

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
- ✅ **Train DT on PPO/RL data:** Completed (June 2026). The FCAS-rich dataset includes 905 PPO-generated episodes. The resulting DT achieves +$1,522/ep profit, beating PPO (+$1,444/ep) on the example evaluator. FCAS revenue improved 18× (from $77 to $1,383/ep). Remaining: validate on the expanded evaluator (135 episodes) to confirm the ranking holds at scale.
- **Prompt calibration:** use the repo's `recommended_rtg` / `recommended_return_scale` diagnostics to choose RTG prompts that are in-distribution relative to the logged training data. The GRPO fine-tuning step (`src/grpo_posttraining.py` on the `copilot/online-rl-fine-tuning` branch) now uses `sample_rtg_values()` for adaptive RTG sampling — see `notebooks/aemo_dt_grpo_posttraining.ipynb`.
- **Training data mixture studies:** systematically vary which behavior policies generate the offline dataset (rule-based vs SDP vs SB3 vs PPO) and evaluate how DT performance changes. The current FCAS dataset uses a fixed mixture; ablating PPO's contribution would quantify its importance.
- **Long-context modeling:** tested context lengths 120–2016 across fair-comparison proxy sweeps. **Context=180 (15 hours) was optimal** — both shorter (120) and longer (288, 360, 576, 1008) contexts regressed validation loss. Context=2016 is now feasible on the RTX 2080 Ti (22 GB) at batch sizes up to 16; a re-sweep with FCAS-rich data is warranted.

Beyond DT-centric work, the AEMO results also highlight:
- ✅ **FCAS-aware offline data collection:** Completed. The 2,425-episode FCAS dataset (`data/aemo_dt_fcas/aemo_fcas_dataset.parquet`) includes PPO, TD3, A2C, DDPG, SAC, and `fcas_rule` trajectories across 3 horizons, 5 regions, and 3 battery sizes.
- **Multi-objective training:** The DT's return-conditioning naturally supports multiple operating points (conservative vs aggressive). Adding an FCAS-specific auxiliary loss or reward component could further improve AEMO performance, especially on the expanded evaluator.
- **FCAS-weighted loss:** The current loss treats all action dimensions equally. Weighting FCAS action dimensions higher (`action_loss_weight` for FCAS dims) could accelerate FCAS learning.

> **NOTE (literature alignment):** Because studies often vary in objective definitions (financial vs energy-efficiency) and in constraint/user-impact handling, robustness studies should explicitly document which objective family and constraint set is being targeted [6].

### Phase 3: Advanced Architectures and Multi-Agent Systems (Year 2-3)
- **Transformer Architectures:** Explore modifications to the Decision Transformer architecture (e.g., long-context attention) to better capture seasonal periodicities in energy data.
- **Multi-Agent Coordination:** Extend the environment to a microgrid setting where multiple homes trade energy, studying the emergence of cooperative behaviors.

### Phase 4: Sim-to-Real Transfer (Year 3-4)
- Develop "safe RL" wrappers to ensure constraints are met during deployment.
- Validate policies on hardware-in-the-loop setups or pilot deployments.

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
- **AEMO utility-scale environment (expanded evaluation):** PPO dominates on the large-scale benchmark (135 episodes, 5 regions, 6 months) with mean_reward = +12.82 and $12,839/ep profit vs the full-pretrained DT's -$1,396/ep. The key gap was FCAS: PPO earned $10,628/ep vs DT's $77/ep because the DT's offline training data lacked FCAS bidding patterns.
- **AEMO utility-scale environment (FCAS-rich dataset):** When retrained on a 2,425-episode FCAS-rich dataset (including 905 PPO-generated episodes), the **DT achieves the highest profit per episode (+$1,522/ep)**, beating PPO (+$1,444/ep) by 5% on the example evaluator. FCAS revenue improves 18× to $1,383/ep, and degradation is 2.9× lower than PPO ($212 vs $609). This confirms that **training DT on RL-generated trajectories successfully closes the FCAS gap**.
- **Autoresearch optimization improved the DT substantially:** The full-pretrained DT (8×512, ctx=180) beats the rule baseline (-3.11 vs -4.82) and is dramatically better than the old pretrain model (-13.55). Degradation was reduced 5.2× and dispatch intensity 6.8× through frontier hyperparameter optimization. Context=180 remains optimal across all evaluations.

This report documents the system and experimental protocol; results can be iteratively updated as additional experiments are run.

## References

[1] K. Abdulla, J. De Hoog, et al., "Optimal Operation of Energy Storage Systems Considering Forecasts and Battery Degradation," *IEEE Transactions on Smart Grid*, 2016.

[2] V. Muenzel, J. De Hoog, et al., "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management," *IEEE Transactions on Industrial Electronics*, 2015.

[3] Sutton & Barto. Reinforcement Learning: An Introduction.

[4] Chen et al. Decision Transformer: Reinforcement Learning via Sequence Modeling.

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
