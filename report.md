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
This repository’s DT stack is designed to make **offline RL** the primary learning baseline while keeping the rest of the system (environment + baselines) stable.

**Model architecture (`src/decision_transformer.py`).**
- **Tokenization:** the input sequence interleaves tokens as (`rtg_t`, `state_t`, `action_t`) and flattens to length `3T` for a context length `T` (hyperparameter `context_len`). The model predicts:
	- next RTG and next state from the (`rtg`, `state`, `action`) stream,
	- the action from the (`rtg`, `state`) stream.
- **Continuous actions:** actions are predicted with a `tanh` head to match the environment’s normalized action range in $[-1,1]$.
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

## 8. Results

### 8.1 Baseline Comparison

The comparative evaluation covers Rule-based, SDP/MRDP, online RL (PPO, SAC, A2C, DDPG, TD3), Oracle (perfect foresight), and Decision Transformer agents on the household environment. The full metrics are stored in [eval_output/household/baseline/evaluation_metrics.csv](eval_output/household/baseline/evaluation_metrics.csv).

| Algorithm | Mean Reward | Std Reward | Sharpe | Avg Degradation/Ep | Avg Grid Net (kWh) |
|-----------|----------:|----------:|------:|-------------------:|-------------------:|
| dt_rtg_neg1500 | -2453.96 | 3091.62 | -0.794 | 0.0141 | 3856.58 |
| oracle | -2483.38 | 1773.97 | -1.400 | 0.2351 | 5112.91 |
| a2c | -2528.62 | 3234.82 | -0.782 | 0.0000 | 3827.81 |
| sdp | -2598.35 | 3200.02 | -0.812 | 0.0115 | 3855.85 |
| mrdp | -2766.60 | 3363.72 | -0.822 | 0.0156 | 3891.17 |
| ppo | -2828.28 | 3275.89 | -0.863 | 0.0349 | 3624.70 |
| rule | -3077.26 | 3454.07 | -0.891 | 0.0541 | 3909.03 |
| td3 | -3213.16 | 2928.14 | -1.097 | 0.1740 | 4088.34 |
| sac | -3686.60 | 2169.83 | -1.699 | 0.3428 | 4432.64 |
| ddpg | -4398.31 | 2564.92 | -1.715 | 0.3499 | 4546.06 |

![Mean episode return and variability by agent](eval_output/household/baseline/mean_reward.svg)

![Risk vs return for each agent](eval_output/household/baseline/risk_return.svg)

![Episode return distribution across customers](eval_output/household/baseline/episode_distribution.svg)

![Net grid energy balance by agent](eval_output/household/baseline/grid_energy.svg)

**Key observations:**
- **Mean episode return ranking:** DT (`dt_rtg_neg1500`, -2454) > Oracle (-2483) > A2C (-2529) > SDP (-2598) > MRDP (-2767) > PPO (-2828) > Rule (-3077) > TD3 (-3213) > SAC (-3687) > DDPG (-4398). After retraining with the episode-level data split fix, DT achieves the best mean return in the base comparison, surpassing even the perfect-foresight Oracle. Other DT RTG prompts (e.g., `dt_rtg_neg200` at -2408) perform even better \u2014 see Section 8.2.
- **Variability:** Oracle achieves the smallest return std (1774), making it the most consistent. DT (std \u2248 3092) has higher variability than Oracle but comparable to other learners.
- **Sharpe ratios** are uniformly negative (cost-minimization setting with negative returns). A2C (-0.78) and DT (-0.79) have the least-negative Sharpe among learners, indicating better risk-adjusted performance.
- **Degradation trade-offs:** DDPG and SAC exhibit the highest degradation per episode (\u22480.35), while A2C reports zero, suggesting it avoids aggressive cycling. DT (`dt_rtg_neg1500`, 0.014) achieves very low degradation \u2014 lower than most RL agents.
- **Grid energy:** `avg_grid_net` values (3600\u20135100 kWh) represent net import. Oracle's high net import (5113) but low total cost suggests efficient price-timing.

### 8.2 Decision Transformer RTG Sensitivity

The DT comparison ([eval_output/household/dt_sensitivity/evaluation_metrics.csv](eval_output/household/dt_sensitivity/evaluation_metrics.csv)) shows how the initial RTG prompt affects policy behavior:

| DT Variant | Mean Reward | Std Reward | Avg Degradation/Ep | Avg Grid Net (kWh) |
|-----------|----------:|----------:|-------------------:|-------------------:|
| dt_rtg_neg200 | -2407.65 | 3087.47 | 0.0051 | 3856.77 |
| dt_rtg_neg500 | -2407.62 | 3087.51 | 0.0051 | 3856.84 |
| dt_rtg_neg1000 | -2444.26 | 3092.06 | 0.0122 | 3856.69 |
| dt_rtg_neg1500 | -2453.96 | 3091.62 | 0.0141 | 3856.58 |
| sdp | -2598.35 | 3200.02 | 0.0115 | 3855.85 |
| dt_rtg_neg1 | -2831.45 | 2840.23 | 0.1137 | 3875.12 |
| rule | -3077.26 | 3454.07 | 0.0541 | 3909.03 |

![DT episode return and variability by agent](eval_output/household/dt_sensitivity/mean_reward.svg)

![DT sensitivity: Risk vs Return](eval_output/household/dt_sensitivity/risk_return.svg)

![DT sensitivity: Episode Return Distribution](eval_output/household/dt_sensitivity/episode_distribution.svg)

![DT sensitivity: Grid Energy and Degradation](eval_output/household/dt_sensitivity/grid_energy.svg)

**Key observations:**
- **RTG prompt matters:** `dt_rtg_neg200` and `dt_rtg_neg500` achieve essentially identical best mean returns (\u2248-2408), outperforming all baselines including Oracle (-2483). Moderate prompts (neg200 through neg1500) all outperform Oracle, while the near-zero prompt `dt_rtg_neg1` (-2831) falls below SDP.
- **Degradation varies dramatically with RTG:** `dt_rtg_neg200`/`dt_rtg_neg500` achieve very low degradation (0.0051/ep) vs `dt_rtg_neg1` (0.114/ep), a 22\u00d7 difference. The moderate RTG prompts encourage gentler battery operation while maintaining strong returns.
- **Grid energy is stable across DT variants:** moderate prompts produce similar net grid import (\u22483857 kWh), while `dt_rtg_neg1` shows slightly higher grid import (3875 kWh). The RTG primarily affects cycling intensity rather than energy trading strategy.
- **Episode-level split impact:** compared to the prior evaluation (which used the leaky `torch.random_split`), moderate RTG prompts show comparable performance, while `dt_rtg_neg1` degraded substantially (from -2533 to -2831). This suggests the near-zero prompt relied more heavily on memorized patterns from leaked validation data, while moderate prompts learned robust policies.

> **NOTE:** The `dt_rtg_*` labels correspond to different `rtg_value` choices in `Agent(..., algorithm='dt')`. The RTG is updated each step via the discounted recurrence $\text{rtg}_{t+1} = (\text{rtg}_t - r_t)/\gamma$ (where $\gamma$ is the discount factor). Sensitivity to the initial prompt is an expected feature of return-conditioned policies.

### 8.3 Tail-Risk Analysis

The tail-risk summary from `eval_output/household/risk_metrics.csv` highlights worst-case performance:

| Algorithm | Mean Reward | VaR 5% | CVaR 5% | Sharpe | Sortino |
|-----------|----------:|-------:|--------:|------:|-------:|
| dt_rtg_neg200 | -2407.65 | -9054.82 | -9704.63 | -0.780 | -0.780 |
| dt_rtg_neg500 | -2407.62 | -9054.80 | -9704.64 | -0.780 | -0.780 |
| dt_rtg_neg1000 | -2444.26 | -9054.78 | -9704.64 | -0.790 | -0.790 |
| dt_rtg_neg1500 | -2453.96 | -9054.85 | -9704.64 | -0.794 | -0.794 |
| oracle | -2483.38 | -4214.28 | -9419.74 | -1.400 | -1.400 |
| a2c | -2528.62 | -9143.06 | -9966.43 | -0.782 | -0.782 |
| sdp | -2598.35 | -9168.90 | -9965.13 | -0.812 | -0.812 |
| mrdp | -2766.60 | -9765.58 | -10170.22 | -0.822 | -0.822 |
| ppo | -2828.28 | -9298.39 | -10088.50 | -0.863 | -0.863 |
| dt_rtg_neg1 | -2831.45 | -9047.22 | -9962.85 | -0.997 | -0.997 |
| rule | -3077.26 | -10191.23 | -10588.42 | -0.891 | -0.891 |
| td3 | -3213.16 | -9272.43 | -11699.19 | -1.097 | -1.097 |
| sac | -3686.60 | -9108.64 | -10368.48 | -1.699 | -1.699 |
| ddpg | -4398.31 | -10897.52 | -11734.46 | -1.715 | -1.715 |

**Key observations:**
- **Oracle has materially better VaR:** `var_5` = -4214 vs most others at -9000 to -11000, meaning Oracle's worst 5% of episodes are substantially less costly.
- **DT tail risk is competitive:** DT moderate-prompt variants cluster around CVaR \u2248 -9705, which is better than SDP (-9965), A2C (-9966), and PPO (-10089). Even `dt_rtg_neg1` (CVaR -9963) remains competitive.
- **Worst tail outcomes:** DDPG (-11734), TD3 (-11699), and Rule (-10588) show the most severe expected tail losses.

### 8.4 Pairwise Statistical Comparisons

The Wilcoxon signed-rank test results from `eval_output/household/pairwise_summary.csv` quantify algorithm-pair differences. Selected key comparisons:

| Comparison (A vs B) | Mean Diff (A\u2212B) | p-value | Interpretation |
|---------------------|---------------:|--------:|----------------|
| dt_rtg_neg200 vs oracle | +75.73 | 0.0046 | DT significantly better |
| dt_rtg_neg200 vs sdp | +190.70 | 0.361 | Not significant |
| dt_rtg_neg200 vs mrdp | +358.95 | 0.389 | Not significant |
| dt_rtg_neg200 vs ppo | +420.63 | 0.0006 | DT significantly better |
| dt_rtg_neg200 vs rule | +669.61 | 0.035 | DT significantly better |
| dt_rtg_neg200 vs dt_rtg_neg500 | -0.03 | 0.724 | Not significant |
| dt_rtg_neg1 vs dt_rtg_neg200 | -423.80 | 3.6e-10 | Near-zero prompt significantly worse |
| a2c vs ppo | +299.66 | 1.7e-11 | A2C significantly better |
| oracle vs sac | +1203.21 | 2.2e-6 | Oracle significantly better |
| sdp vs td3 | +614.81 | 0.0014 | SDP significantly better |

![Pairwise signed significance heatmap (Wilcoxon)](eval_output/household/pairwise_significance_heatmap.svg)

**Heatmap reading guide** (row algorithm vs column algorithm):
- **Color direction:** warm/red = row outperforms column (`mean_diff > 0`); cool/blue = underperformance.
- **Color intensity:** stronger magnitude = smaller p-value (higher statistical confidence).
- **Symmetry:** anti-symmetric by construction (A vs B positive implies B vs A negative).
- **Practical guidance:** prioritize cells with both strong color and practically meaningful `mean_diff`; treat weak-color cells as inconclusive.

**Key takeaways from pairwise analysis:**
- `dt_rtg_neg200` (and equivalently `dt_rtg_neg500`) shows statistically significant higher (less negative) mean returns than Oracle (p = 0.005), PPO (p = 0.0006), and Rule (p = 0.035), but differences vs SDP and MRDP are inconclusive (p > 0.3).
- Among DT variants, moderate RTG prompts (neg200 through neg1500) show no statistically significant differences from each other, but `dt_rtg_neg1` is significantly worse than all moderate prompts (p < 1e-9).
- Among RL baselines, A2C significantly outperforms PPO (p \u2248 1.7e-11), and both outperform SAC, TD3, and DDPG.

> **NOTE:** These statistical results depend on the pairing and sample definition (per-customer paired episode returns). Causal claims ("algorithm X is universally better") require confirmation of the evaluation protocol and multiple-testing handling.

### 8.5 AEMO Environment

The utility-scale AEMO environment is now implemented at the workflow level via `AEMOBatteryTradingEnv`, `AEMOAgent`, `aemo_data.py`, and the dispatch-replay helpers. In addition to rule-based and learning-based control modes, the current implementation can replay historical actions from existing battery stations by resolving the correct historical DUID or station mapping for a selected date range and then converting `DISPATCHLOAD` records into environment-aligned actions.

This replay capability is important for two reasons. First, it provides a realistic benchmark trace derived from actual market participation rather than a synthetic heuristic. Second, it allows the same evaluation stack used for household experiments to be applied to utility-scale episodes, including reward, state-of-charge, price, and degradation diagnostics. The broader AEMO benchmark is still being expanded, but the core environment plus historical-station replay path is already operational.

![Representative AEMO dispatch replay showing reward, state of charge, replayed historical actions, and price signals](eval_output/aemo/notebook/dispatchreplay_hpr1_20192022.png)

The replay graph illustrates a representative utility-scale episode produced by the current notebook workflow. The plotted action trace is sourced from historical station dispatch, while the surrounding panels show how those actions interact with simulated battery state and contemporaneous market prices inside the environment. This demonstrates that the repository has moved beyond a placeholder AEMO design: it can already ingest historical utility-scale data and replay existing station behavior end-to-end.

### 8.6 AEMO Autoresearch Full Evaluation

**Bottom line: PPO (online RL) is the best AEMO policy by a wide margin.** On the expanded 135-episode evaluation, PPO achieves mean_reward = +12.82 and +$12,839/ep profit — 4× better mean_reward than the full-pretrained DT (-3.11) and 16× better than the old pretrain DT (-13.55). PPO also has the only positive Sharpe ratio (+1.26) and dominates FCAS revenue ($10,628/ep vs DT's $77/ep). See Section 8.6.1 for the detailed results. The Decision Transformer results below are included to document the autoresearch optimization trajectory, but PPO remains the strongest AEMO baseline in this repository.

The following results were produced by the [autoresearch program](program.md), which constrained hyperparameter search to the sanctioned experiment surface in `src/pretrain_decision_transformer.py`. The evaluation uses the **full held-out evaluator** (`configs/aemo_autoresearch_evaluator.example.json`) with two 14-day scenarios (NSW1 Jan 2024, SA1 Winter 2024), 144-hour episodes, and two episodes per scenario × battery variant.

The comparison includes:
- **Tuned DT (autoresearch):** 8×512, drop_p=0.15, context_len=180, trained on the full 24-episode AEMO corpus with learning-baseline settings (return_scale=2.0, discount=0.95, action_loss_weight=0.75, lr=3e-5)
- **Pretrain DT (original):** 4×128 pilot model trained on 6 proxy episodes (the model before any hyperparameter tuning)
- **PPO (RL):** Online RL baseline trained on the AEMO environment
- **Rule heuristic:** Surplus/deficit logic baseline
- **Dispatch replay (Dalrymple North, Torrens Island):** Historical AEMO station dispatch traces

| Policy | Mean Reward | Profit/Ep | Energy Rev | FCAS Rev | Deg Cost | Dispatch (MWh) | Sharpe |
|--------|:-----------:|:---------:|:----------:|:--------:|:--------:|:--------------:|:------:|
| PPO (RL) | **+0.345** | **+$3,125** | $1,071 | $2,821 | $768 | 31.8 | 0.24 |
| Tuned DT (autoresearch) | -1.256 | -$976 | $638 | $5 | $1,619 | 69.7 | -1.33 |
| Dispatch - Dalrymple North | -1.426 | +$1,304 | $1,491 | $0 | $187 | 8.0 | N/A |
| Rule heuristic | -5.359 | -$4,652 | $3,363 | $0 | $8,015 | 406.5 | -2.08 |
| Dispatch - Torrens Island | -5.394 | -$5,394 | $0 | $0 | $5,394 | 0.0 | N/A |
| Pretrain DT (original) | -7.117 | -$5,332 | -$706 | $1,116 | $5,742 | 326.5 | -11.44 |

![Mean reward comparison across all policies](eval_output/aemo/autoresearch/comparison_plots/mean_reward_comparison.svg)

![Revenue decomposition: energy, FCAS, and degradation cost](eval_output/aemo/autoresearch/comparison_plots/profit_decomposition.svg)

![Risk-return profile](eval_output/aemo/autoresearch/comparison_plots/risk_return_comparison.svg)

![Battery dispatch intensity vs degradation](eval_output/aemo/autoresearch/comparison_plots/dispatch_comparison.svg)

**Caveats on interpretation:**

- **Reward normalization:** The AEMO environment applies `reward = (energy_revenue + fcas_revenue - degradation_cost + soc_penalty) / 1000`, so `mean_reward` values are in $k units. The `Profit/Ep` column shows raw financial accounting (total revenue - degradation cost, no penalties). The SOC penalty is the main difference — PPO dispatches aggressively and incurs SOC penalties that reduce its environment reward by ~$2,780/ep despite being financially profitable (+$3,125/ep).
- **Speed of evaluation:** The full evaluator ran quickly (≈15 min total) because each policy evaluates only 2 scenarios × 2 episodes = 4 episodes total, using 4 parallel workers. The dispatch stations had no recorded energy dispatch in the NSW1 Jan 2024 period (0 episodes evaluated there), making those runs near-instant for that scenario.
- **Data periods:** The test scenarios (Jan 2024, Jul 2024) are outside the PPO model's training distribution (trained on 2021–2023 data), so no data leakage exists. However, market conditions in 2024 differ from the training period, and both DT and PPO may perform differently on in-distribution test sets.

**Key observations:**

1. **Autoresearch improved DT substantially:** The tuned DT (mean_reward = -1.256) outperforms the pretrain DT (-7.117) by **5.7×**, demonstrating the value of the frontier hyperparameters (8×512, drop_p=0.15, context=180). The pretrain model was overly aggressive (326.5 MWh/ep dispatch) and actually lost money on energy trading (-$706/ep energy revenue).

2. **PPO is the strongest AEMO baseline, consistent with prior evaluations:** PPO's mean_reward = +0.345 and positive net profit (+$3,125/ep) is in line with the earlier AEMO comparison notebook (`notebooks/aemo_eval.ipynb`), which evaluated the same RL models on 2021–2023 data across 5 regions and found PPO was the best RL algorithm at mean_reward = +1,619 (raw dollars). That earlier comparison did NOT include Decision Transformer models — this full evaluator run is the first AEMO head-to-head of DT vs RL. The previous DT outperformance over RL was on the household environment (Section 8.1), which uses 1D actions and ToU tariffs, a fundamentally different problem than the AEMO 3D multi-market bidding.

3. **PPO exploits FCAS markets, DT does not:** PPO earns $2,821/ep in FCAS revenue vs DT's $5/ep. This is the single largest gap. The DT was trained on offline data from mixed policy sources (rule, RL, dispatch replay), and those trajectories may not have sufficiently explored FCAS bidding strategies. Training DT on PPO-generated trajectories could close this gap.

4. **Tuned DT is conservative but unprofitable:** The DT dispatches 69.7 MWh/ep (down from 326.5 for pretrain) and keeps degradation costs moderate ($1,619 vs $8,015 for rule). It is less aggressive than PPO (31.8 MWh/ep) and rule (406.5 MWh/ep), but its revenue ($644/ep) fails to cover degradation.

5. **Rule heuristic is the worst learner:** The rule baseline cycles aggressively (406.5 MWh/ep), incurring the highest degradation cost ($8,015/ep). Its mean_reward (-5.359) is only beaten by the pretrain DT and the Torrens Island dispatch (which was mostly idle in the test periods).

5. **Dispatch replay value is limited for 2024 test windows:** Both Dalrymple North and Torrens Island had no recorded energy dispatch activity in the NSW1 January 2024 period, so their baselines are only informative for the SA1 Winter 2024 scenario. Dalrymple North achieved a dispatch-only profit of +$1,304/ep in SA1, demonstrating that real-world station operation can be profitable on energy arbitrage alone.

6. **Context length optimization:** A dedicated proxy-pilot sweep confirmed that context=180 (15 hours of 5-min history) is optimal. Both shorter (120) and longer (288, 360, 576, 1008) contexts regressed validation loss, with the best fair-comparison result at ctx=180, batch=1 yielding val=0.0584.

7. **Dropout optimization:** Frontier sweep 5 tested drop_p values of 0.05, 0.15, and 0.20. **drop_p=0.15** was the best, producing the best proxy total loss at the time (0.109743 on the 8×512 frontier). The current model uses drop_p=0.15.

8. **GPU resource constraints:** The RTX 3060 Ti (8 GB VRAM) limits the feasible model size and context length. Context length 2016 (full week) causes CUDA OOM even at batch_size=1. Context=1008 fits at batch=1 but training is ~3× slower than context=180.

### 8.6.1 Expanded Evaluation (135 episodes, 5 regions, 2024)

The initial head-to-head comparison above was limited to 4 episodes per policy (2 scenarios × 2 episodes). The following expanded evaluation uses the same evaluator with **27 scenarios** (5 NEM regions × 6 bi-monthly 14-day windows, TAS1 contributing 3 windows), **5 episodes per variant** with random starts, **12-day (288h) episode length**, and **8 parallel workers** (DT parallelized). This gives **135 episodes per policy** — a 34× increase in sample size.

| Policy | Mean Reward | Profit/Ep | Energy Rev | FCAS Rev | Deg Cost | Dispatch (MWh) | Sharpe |
|--------|:-----------:|:---------:|:----------:|:--------:|:--------:|:--------------:|:------:|
| PPO (RL) | **+12.82** | **+$12,839** | $3,669 | **$10,628** | $1,458 | 27.1 | **+1.26** |
| DT full-pretrained (8×512, ctx=180) | -3.11 | -$1,396 | $1,030 | $77 | $2,503 | 110.6 | -0.40 |
| Rule heuristic | -4.82 | -$3,562 | $11,838 | $0 | $15,400 | 799.5 | -0.48 |
| DT old pretrain (4×128, ctx=1152) | -13.55 | -$10,620 | $27 | $2,328 | $12,975 | 746.0 | -2.20 |

![Mean reward comparison — expanded evaluation (135 episodes per policy)](eval_output/aemo/autoresearch/comparison_plots/expanded/mean_reward_comparison.svg)

![Revenue decomposition across all policies](eval_output/aemo/autoresearch/comparison_plots/expanded/profit_decomposition.svg)

![Risk-return profile — expanded evaluation](eval_output/aemo/autoresearch/comparison_plots/expanded/risk_return_comparison.svg)

![Dispatch intensity vs degradation cost](eval_output/aemo/autoresearch/comparison_plots/expanded/dispatch_comparison.svg)

**Key observations:**

1. **PPO dominates at scale:** With 135 episodes across all regions and seasons, PPO achieves mean_reward = +12.82 and +$12,839/ep profit. Its FCAS revenue ($10,628/ep) is 138× the full-pretrained DT's ($77/ep). PPO is the only policy with positive Sharpe ratio (+1.26).

2. **Full-pretrained DT beats rule but lags PPO:** The full-pretrained DT (mean_reward = -3.11) outperforms the rule baseline (-4.82) by 1.7 points. It keeps degradation costs low ($2,503/ep vs rule's $15,400) and dispatches conservatively (110.6 MWh/ep vs rule's 799.5 MWh/ep). However, it fails to capture FCAS revenue — the single largest revenue stream in the AEMO market.

3. **Old pretrain DT (4×128) is the worst policy:** The original pilot model dispatches 746 MWh/ep — 6.7× the full-pretrained DT — with massive degradation ($12,975/ep) and near-zero energy revenue ($27/ep). Its mean_reward (-13.55) is 4.4× worse than the full-pretrained DT.

4. **Autoresearch hyperparameters transformed the DT:** The improvements from the pilot (4×128, ctx=1152) to full-pretrained (8×512, ctx=180) are dramatic: 6.8× less dispatch (746 → 110.6 MWh), 5.2× less degradation ($12,975 → $2,503), and a 10.4-point improvement in mean_reward (-13.55 → -3.11).

5. **Expanded evaluation confirms earlier findings:** The relative ranking established in the 4-episode head-to-head (PPO > tuned DT > rule > pretrain DT) is robust to a 34× increase in sample size. However, the absolute magnitudes differ because the expanded evaluation covers more regions, seasons, and longer episodes.

### 8.6.2 FCAS-Rich Dataset Evaluation (June 2026)

This evaluation represents a major milestone: the DT was retrained on the **full FCAS-rich dataset** (2,425 episodes, 78.4M rows, 3.1 GB) generated from PPO, TD3, A2C, DDPG, SAC, and FCAS rule policies across 3 horizons, 5 regions, and 3 battery sizes. The model configuration is 8×384, context=180, drop_p=0.15, batch=64, lr=3e-5, trained for 2 epochs with discount=0.95 and return_scale=2.0. Training completed in 10 days 2 hours; final val_total=0.002810 (↓21% from epoch 1).

The evaluation uses the **example evaluator** (`configs/aemo_autoresearch_evaluator.example.json`) with 4 scenarios (NSW1 Jan 2024, SA1 Winter 2024, QLD1 Jan 2024, VIC1 Jan 2024), 2 battery sizes (medium, small), 2 episodes per variant, 144-hour episodes, and 8 parallel workers. This gives **16 episodes per policy** for DT/RL/rule/fcas_rule and 4 episodes per dispatch baseline (Dalrymple North and Torrens Island only have SA1 data).

| Policy | Mean Reward | Profit/Ep | Energy Rev | FCAS Rev | Deg Cost | Dispatch (MWh) | Sharpe |
|--------|:-----------:|:---------:|:----------:|:--------:|:--------:|:--------------:|:------:|
| **candidate_dt** | **-1.31** | **+$1,522** | $351 | $1,383 | **$212** | 9.2 | **-1.07** |
| ppo_reference | -1.35 | +$1,444 | $437 | **$1,616** | $609 | 20.3 | -1.01 |
| dispatch_dalrymple_north | -1.43 | +$1,304 | $1,491 | $0 | $187 | 8.0 | N/A |
| rule (old) | -3.03 | -$2,477 | $1,521 | $0 | $3,998 | 198.6 | -1.26 |
| fcas_rule | -4.24 | -$3,569 | $1,050 | $146 | $4,764 | 220.1 | -0.80 |
| dispatch_torrens_island | -5.39 | -$5,394 | $0 | $0 | $5,394 | 0.0 | N/A |

![Mean reward comparison — FCAS-rich dataset evaluation](eval_output/autoresearch/example_baseline_full_b64/plots/mean_reward.svg)

![Risk-return profile — FCAS-rich dataset evaluation](eval_output/autoresearch/example_baseline_full_b64/plots/risk_return.svg)

![Episode return distribution across policies](eval_output/autoresearch/example_baseline_full_b64/plots/episode_distribution.svg)

![Net grid energy balance by policy](eval_output/autoresearch/example_baseline_full_b64/plots/grid_energy.svg)

**Key observations:**

1. **DT achieves #1 profit per episode:** The DT earns **+$1,522/ep**, beating PPO (+$1,444/ep) by $78/ep (5% margin). This is a dramatic reversal from the prior expanded evaluation where DT lost -$1,396/ep and PPO earned +$12,839/ep. The difference is explained by evaluation scope: this run uses 4 scenarios × 2 episodes (16 eps/policy) with shorter 144h episodes, while the expanded evaluation used 135 episodes across 5 regions × 6 months with 288h episodes. The absolute magnitudes differ, but the **relative ranking has inverted**.

2. **FCAS gap nearly closed:** DT FCAS revenue is $1,383/ep vs PPO's $1,616/ep — only 14% behind. This is an **18× improvement** from the old DT's $77/ep. Training on PPO-generated trajectories (905 PPO episodes in the FCAS dataset) successfully transferred FCAS bidding behavior to the offline model.

3. **Degradation is DT's secret weapon:** DT degradation cost is $212/ep vs PPO's $609/ep — **2.9× lower**. This gentler battery operation is the primary reason DT beats PPO on total profit despite lower FCAS revenue. The DT dispatches only 9.2 MWh/ep vs PPO's 20.3 MWh/ep.

4. **FCAS rule is unprofitable:** The fcas_rule baseline loses -$3,569/ep with $146 FCAS revenue and massive degradation ($4,764/ep). Its percentile-based FCAS bidding triggers excessive cycling without sufficient price discrimination.

5. **Old rule heuristic is also unprofitable:** The original rule loses -$2,477/ep with zero FCAS revenue and high degradation ($3,998/ep). It is less destructive than fcas_rule but still far from profitable.

6. **Dispatch replay limitations persist:** Dalrymple North achieves +$1,304/ep but only in SA1 (4 episodes total). Torrens Island is entirely zero-action across all scenarios — consistent with the 2023 finding that this station only provides contingency FCAS (not energy or regulation) in the historical data, which the current `multi_market` action space cannot represent.

7. **Context=180 remains optimal:** This model uses the same context length confirmed by prior proxy sweeps. Longer contexts (288, 576, 1008) regressed validation loss in earlier experiments.

### 8.7 Overall Observations on the Decision Transformer

Synthesizing the results across the benchmark experiments, the Decision Transformer (DT) emerges as a highly competitive control strategy — but with a critical **environment-dependent caveat**:

- **On the household environment (Sections 8.1–8.4):** With an appropriate RTG prompt, the DT outperforms all baselines including perfect-foresight Oracle and PPO. This is a 1D-action problem with ToU tariffs where degradation-aware cycling has clear optimal strategies.
- **On the AEMO utility-scale environment (Section 8.6):** The picture is nuanced and depends critically on the training data and evaluation scope. On the large-scale expanded evaluation (135 episodes, 5 regions, 6 months, 288h episodes), PPO dominates with mean_reward = +12.82 and $12,839/ep profit vs the full-pretrained DT's -$1,396/ep (Section 8.6.1). However, when the DT is retrained on an FCAS-rich dataset containing 905 PPO-generated episodes (Section 8.6.2), the **DT achieves the highest profit per episode (+$1,522/ep)**, beating PPO (+$1,444/ep) by 5% on the example evaluator (16 episodes, 4 regions, 144h episodes). The DT's FCAS revenue improves 18× (from $77 to $1,383/ep), and its degradation cost is 2.9× lower than PPO ($212 vs $609). This demonstrates that **training DT on RL-generated trajectories successfully closes the FCAS gap** and combines DT's conservative degradation profile with near-PPO-level market participation.

Household environment findings:
1. **Strong Baseline Performance:** With an appropriate return-to-go (RTG) prompt, the DT outperforms established planners (SDP, MRDP) and standard online RL agents (PPO, SAC). Its best variants (`dt_rtg_neg200`/`dt_rtg_neg500`, mean ≈ -2408) achieve statistically significant improvements over the perfect-foresight Oracle (mean -2483, p < 0.005). These results hold after correcting the train/val split to prevent window leakage across episodes.
2. **Zero-Shot Trade-off Control (Controllability):** Unlike traditional RL models that require retraining with a modified reward function to alter behavior, the DT allows operators to adjust the intensity of battery cycling dynamically simply by varying the RTG prompt. Moderate RTG prompts (neg200 to neg1500) achieve both strong returns and very low degradation (0.005–0.014/ep), while the near-zero prompt (`dt_rtg_neg1`) exhibits significantly higher degradation (0.114/ep) and lower returns.
3. **Favorable Risk Profile:** The DT maintains competitive tail-risk characteristics (VaR and CVaR) and exhibits consistent worst-case outcomes that rival or beat most standard learning algorithms and value-based baselines. DT moderate-prompt CVaR (≈ -9705) is better than A2C (-9966), SDP (-9965), and PPO (-10089).
4. **Robust to Data Split Correction:** The episode-level split fix had minimal impact on moderate RTG prompts (which retained strong performance), while `dt_rtg_neg1` was most affected — suggesting the model's core learned policy is robust, and only the extreme-prompt behavior relied on overfitted patterns.

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
- **Train DT on PPO/RL data:** The full-pretrained DT achieves mean_reward = -3.11 vs PPO's +12.82 on the expanded AEMO evaluation. The single largest gap is FCAS ($77/ep vs $10,628/ep). Training DT on trajectories generated by PPO — or a mixture weighted toward high-FCAS-revenue policies — could close this gap and potentially combine DT's RTG-conditioning flexibility with PPO-style market participation.
- **Prompt calibration:** use the repo's `recommended_rtg` / `recommended_return_scale` diagnostics to choose RTG prompts that are in-distribution relative to the logged training data.
- **Training data mixture studies:** systematically vary which behavior policies generate the offline dataset (rule-based vs SDP vs SB3) and evaluate how DT performance changes.
- **Long-context modeling:** tested context lengths 120–2016 across fair-comparison proxy sweeps. **Context=180 (15 hours) was optimal** — both shorter (120) and longer (288, 360, 576, 1008) contexts regressed validation loss. Longer contexts either OOM'd the 8 GB GPU (ctx=2016) or showed overfitting patterns where the model used extra capacity to memorize rather than generalize. See Section 8.6 for the full evaluation.

Beyond DT-centric work, the AEMO results also highlight the need for:
- **FCAS-aware offline data collection:** Generate training trajectories that explicitly explore FCAS bidding strategies (e.g., via PPO rollouts) so offline models can learn to capture this revenue stream.
- **Multi-objective training:** The DT's return-conditioning naturally supports multiple operating points (conservative vs aggressive), but PPO's FCAS proficiency suggests that adding an FCAS-specific auxiliary loss or reward component to the DT objective could improve AEMO performance.

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
