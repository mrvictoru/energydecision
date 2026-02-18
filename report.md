# Benchmarking and Advancing Control Strategies for Energy Storage: A Unified Framework Across Household Solar-Battery Control and Utility-Scale AEMO Battery Operation

## Abstract

The effective integration of battery energy storage is critical for a reliable, renewable-dominant grid, spanning both behind-the-meter residential operation and utility-scale market participation. Developing and comparing control strategies can be challenging when environments omit key factors such as stochastic demand/generation, time-varying tariffs or market prices, and battery degradation. This report documents a research codebase that provides two Gymnasium-compatible environments—(i) a household solar+battery controller and (ii) a utility battery trading environment for AEMO/NEM—and a shared evaluation workflow.

The **primary learning model** in this codebase is an **offline Decision Transformer (DT)** trained from logged trajectories to produce continuous battery-charge/discharge actions conditioned on a desired return-to-go (RTG). Rule-based heuristics, dynamic-programming planners (SDP/MRDP), online RL baselines (Stable-Baselines3), and dispatch-replay baselines for the AEMO environment are included primarily as comparators and data-generators for DT training.

## 1. Introduction

The proliferation of energy storage across the grid—from distributed, behind-the-meter household batteries to grid-scale battery energy storage systems (BESS) participating in wholesale markets—presents both a challenge and an opportunity for modern power systems. While these assets can reduce consumer costs and provide grid flexibility, their optimal operation is non-trivial. The control problem is characterized by stochastic demand/generation, time-varying tariffs or market prices, non-linear battery degradation dynamics, and strict physical constraints.

> **NOTE (literature support):** Electricity market/tariff structures considered in RL-for-battery work include time-of-use pricing, real-time pricing, and day-ahead markets, among others [6].

Recent literature also helps structure the space of RL-based battery control problems. Subramanya et al. [6] review RL applications for battery storages through multiple lenses (optimization objective, user impact/comfort where applicable, battery losses & degradation, and application context). This benchmark is designed to make these dimensions explicit in a single codebase so that planning and learning approaches can be compared under consistent dynamics and evaluation.

### 1.1 The Research Gap
Despite substantial literature on energy management systems (EMS), reproducibility and cross-paper comparability can be difficult when studies rely on custom environments, private data, and differing assumptions (e.g., constraint handling, tariff structure, or whether degradation is modeled). There is also a practical gap between model-based planning approaches (e.g., dynamic programming / MPC-style methods) and learning-based approaches (e.g., RL and sequence models), which motivates a unified benchmark.

Recent review work supports this motivation: Subramanya et al. [6] note that comparisons across RL-for-battery studies are hindered by unique formulations (environments, state/action spaces, and rewards), and argue that benchmark environments with a standard interface would improve comparability.

### 1.2 Contributions and Research Goals
This work establishes a consolidated, reproducible benchmark to address these limitations. We provide:
1.  **Two High-Fidelity Simulation Environments:** Gymnasium-compatible environments for (a) household solar+battery operation under ToU import/export pricing and (b) grid-scale battery trading under AEMO/NEM market signals (energy + optional FCAS).
2.  **Decision Transformer as the Primary Model:** A modernized Decision Transformer implementation plus an offline training pipeline built around trajectory logging, RTG construction, return scaling, and robust checkpoint loading.
3.  **Baselines as Comparators and Data Sources:** A unified interface for comparing rule-based heuristics, SDP/MRDP planners, online RL (PPO, SAC, etc.), and (for AEMO) dispatch replay against DT, and for generating trajectory data for offline learning.
4.  **Comprehensive Evaluation Suite:** Standardized metrics for return, grid energy flows, degradation, and simple risk proxies (Sharpe/Sortino ratios).

The goal of this platform is to serve as the foundational infrastructure for a PhD thesis investigating **robust, generalization-capable control policies for decentralized energy systems**.

## 2. Related Work

This work is inspired by Abdulla et al. [1], which formulates optimal operation of energy storage using Stochastic Dynamic Programming (SDP) and emphasizes the importance of uncertainty and degradation for realistic assessment.

We adopt an SDP-style planning baseline (implemented in this repository) and a multi-factor degradation model based on Muenzel et al. [2] (implemented in `src/batterydeg.py`). We extend the planning baseline with additional learning-based baselines and a Gymnasium environment wrapper.

> **NOTE (literature support):** Modeling or limiting battery degradation is a recurring theme in RL-for-battery applications, and the diversity of degradation modeling approaches complicates direct comparisons across studies [6].

### 2.1 RL-for-Battery Benchmarks (Review Context)
Subramanya et al. [6] highlight several recurring themes that directly motivate a unified, Gym-style benchmark:
- **Heterogeneous formulations hinder comparison:** across studies, environments, state/action spaces, and reward definitions vary substantially, making it hard to compare reported performance and to transfer insights between applications.
- **Benchmark environments are valuable:** the review argues that standardized benchmark environments with a common agent–environment interface would improve comparative assessment; it notes that Gym-style benchmarks are common in other domains but comparatively scarce in energy/battery domains.
- **Degradation modeling is inconsistent:** long-term degradation is captured in only a minority of works, and when included, degradation is incorporated in diverse ways (constraints, reward penalties, or auxiliary objectives), further complicating comparisons.
- **Sim-to-real validation is underexplored:** the review identifies the need to deploy trained agents and compare performance on physical systems versus model-based/simulated performance.

This repository’s design choices (Gymnasium API, explicit constraint handling, and logging of grid energy and degradation signals) align with these needs.
1.  **Modernizing the Interface:** Wrapping the simulation in a standard Gymnasium API to bridge the gap between the optimization and Deep RL communities.
2.  **Expanding the Algorithmic Suite:** Introducing Online Deep RL (PPO, SAC) and Offline RL (Decision Transformers) to compare learning-based approaches against the theoretical optimality of SDP.
3.  **Open Reproducibility:** Providing a fully open-source, containerized benchmark, addressing the lack of public code in prior studies.

## 3. System Model and Environments

This repository provides two primary environments.

### 3.1 Household Solar-Battery Environment (SolarBatteryEnv)
Environment: `src/EnergySimEnv.py` defines `SolarBatteryEnv` with:
- Action: 1D normalized battery power in [-1, 1]. In `step()`, this is scaled to kW via `max_battery_flow`, converted to step energy (kWh) via `step_duration`, and clipped by SoC and capacity.
- Observation: always normalized in the current implementation (`normalize_obs = True`). It includes cyclical time features (sin/cos of hour/day-of-year), min–max normalized dataframe features, and two extra features: battery level and current-step degradation cost (both normalized).
- Dynamics: `step_duration` is inferred from the dataframe `Time` column (hours between the first two timestamps, with a fallback). The grid energy is clipped to `max_grid_energy = max_grid_flow × step_duration`, and energy-conservation violations yield an early termination with `VIOLATION_PENALTY`.
- Reward/cost: per-step reward is `grid_reward - current_step_deg_cost`, where `grid_reward` is `-(grid_energy × price)` (import vs export prices selected by sign), and degradation cost is derived from per-cycle wear × `battery_life_cost`.
- Degradation: the environment uses rainflow counting over the SoC trajectory to extract cycles, then applies a multi-factor cycle-life model based on Muenzel et al. [2] (temperature, C-rates, SOCav, DoD) to compute per-cycle degradation.

> **NOTE (repo-backed):** These details are implemented in `src/EnergySimEnv.py` and `src/batterydeg.py`.

Dataset contract (from `src/helper.py::transform_polars_df`):
`Timestamp, SolarGen, HouseLoad, FutureSolar, FutureLoad, ImportEnergyPrice, ExportEnergyPrice, Time` (sorted by `Time`).

> **NOTE (repo-backed):** `transform_polars_df` also drops the final row after shifting `FutureSolar/FutureLoad` to avoid null future values.

### 3.2 Utility-Scale AEMO Battery Trading Environment (AEMOBatteryTradingEnv)
Environment: `src/AEMOBatteryEnv.py` defines `AEMOBatteryTradingEnv`, a Gymnasium environment for a grid-scale BESS participating in Australia's National Electricity Market (NEM). Key design points implemented in the repository include:
- **Market signals:** the environment consumes preprocessed AEMO data with energy spot price (`RRP`), regional demand (`TOTALDEMAND`), optional FCAS service prices (wide columns prefixed `FCAS_`), and optional generation mix features (wide columns prefixed `GEN_`).
- **Observation space:** a fixed 18-dimensional vector (time features, normalized energy price and demand, normalized FCAS service prices, generation mix, and normalized SOC). `AEMODataPreprocessor` adds normalization columns (e.g., `RRP_normalized`, `DEMAND_normalized`, `FCAS_*_normalized`).
- **Action space:**
	- `action_mode='simple'`: 1D action in [-1, 1] for energy-only charge/discharge.
	- `action_mode='multi_market'`: 3D action `[battery_dispatch, fcas_raise_bid, fcas_lower_bid]` with dispatch in [-1,1] and FCAS bids in [0,1].
- **Units and scale:** default capacity/flow are specified in MWh/MW (grid-scale), distinct from the household environment (kWh/kW).
- **Degradation:** supports `degradation_mode='rainflow'` using the same `DegradationModel` + `RainflowCounter` primitives as the household environment, tracking `step_degradation`, `total_degradation`, and capacity fade.

> **NOTE (repo-backed):** The AEMO environment interface and feature definitions are documented in docs/AEMO_ENV_README.md and implemented in src/AEMOBatteryEnv.py.

## 4. Methods

Baselines and planners are implemented in `src/decision.py` (Agent abstraction):
- Rule-based: a heuristic using surplus/deficit logic, optional persistence, and small injected noise (see `Agent.rule_based_action`).
- RL (SB3): PPO/A2C/DDPG/SAC/TD3 via `src/sb3train.py::train_model`; rollouts collected by `run_sb3_model_on_vec_env` and flattened with `flatten_episode_data`.
- SDP: a self-contained dynamic programming baseline implemented in `src/sdp_algorithm.py`. It discretizes SoC and actions and performs backward induction. Uncertainty can be handled via scenario sampling (Monte Carlo) when enabled.
- MRDP: `algorithm='mrdp'` with `subhorizon_specs` for coarse-to-fine planning, addressing the "curse of dimensionality" inherent in standard SDP.
- Decision Transformer (DT): Offline sequence model proposed by Chen et al. [4] (`src/decision_transformer.py`) trained on `TrajectoryDataset` from logged trajectories (`src/transformer_training.py`). Inference uses `model.get_action` with rolling context.

For the AEMO environment, `src/decision.py` also provides `AEMOAgent`, which supports rule-based control, RL/DT inference, and a **dispatch-replay** mode that converts AEMO unit dispatch data into environment actions aligned to the environment timestep.

### 4.1 Decision Transformer (Primary Model, Repo-Backed)
This repository’s DT stack is designed to make **offline RL** the main research vehicle while keeping the rest of the system (environment + baselines) stable.

**Model architecture (`src/decision_transformer.py`).**
- **Tokenization:** the input sequence interleaves tokens as $(\text{rtg}_t, \text{state}_t, \text{action}_t)$ and flattens to length $3T$ for a context length $T=\texttt{context\_len}$. The model predicts:
	- next RTG and next state from the $(\text{rtg},\text{state},\text{action})$ stream,
	- the action from the $(\text{rtg},\text{state})$ stream.
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

Risk-aware extensions (proposed): add CVaR-style evaluation and multi-objective scalarization for reward vs degradation.

> **TODO (repo check):** CVaR is not currently implemented in `src/helper.py` evaluation. If you keep CVaR in the report, label it explicitly as future work (as above) or add an implementation + citation.

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

> **NOTE (repo-backed):** Actual AEMO data fetching requires the optional dependency `nemosis` to be installed; otherwise fetch functions raise `ImportError`.

## 6. Experimental Setup

Splits and seeds:
- Train/test split by customer ID (e.g., 80/20), fixed seeds, and config logging are recommended for reproducibility.

> **TODO (needs source or repo pointer):** If you want to claim a specific split protocol is used in current experiments, cite the exact notebook/script where the split and seeding are performed.

Workflows:
- DT (primary):
	- Create trajectory logs (Parquet) from rule-based, SDP/MRDP, SB3 policies, oracle policies, and (for AEMO) dispatch replay episodes.
	- Train DT with `src/pretrain_decision_transformer.py` (wraps `TrajectoryDataset` + `train_decision_transformer`).
	- Evaluate DT using `Agent(algorithm='dt', rtg_value=...)` to study RTG-conditioning sensitivity.
- RL: `DummyVecEnv` for training, `SubprocVecEnv` for evaluation; train with `train_model(..., default_model=True)` or enable Optuna tuning.
- SDP/MRDP: configure horizons/resolutions; evaluate single or parallel episodes via `run_episodes_parallel`.

DT hyperparameters:
- Default DT model kwargs are stored in `models/decision_transformer_model_kwargs.json` (e.g., `state_dim=12`, `act_dim=1`, `context_len=60`, `h_dim=128`, `n_block=2`, `n_heads=8`).
- Training-time `return_scale` is stored in checkpoints and also written to `*.meta.json` sidecars for consistent inference.

> **TODO (multi-env configs):** Add a second DT config JSON for AEMO (e.g., `state_dim=18`, `act_dim=1` or `3`) if you want the report to claim DT is trained on AEMO logs in the current experiments.

Compute and reproducibility:
- Containerization: the repository includes a `Dockerfile` and `docker-compose.yml` for running a consistent environment.
- Figures: `evaluate_experiments(..., save_dir=..., save_format=...)` can save plots (default `save_format='svg'`).

## 7. Metrics and Analysis

Primary metrics (from `src/helper.py`):
- Episode return statistics: mean/median/std and 5th/95th percentiles (computed from per-episode sums of the logged `reward`).
- Grid energy flows: average per-episode grid import (kWh), grid export (kWh), and net grid energy (kWh) derived from `info['grid_energy']`.
- Degradation: average per-episode and per-step degradation derived from `info['step_degradation']`.
- Risk proxies: Sharpe and Sortino are computed directly from the distribution of episode returns (not annualized; Sharpe is `mean/std`).

For AEMO logs, the same evaluation functions also summarize market-specific metrics when those keys appear in `info` (e.g., `energy_revenue`, `fcas_revenue`, `total_revenue`, `degradation_cost`, `battery_dispatch`, `actual_energy`).

> **NOTE:** A separate “cost decomposition” into grid import cost vs export revenue is not currently reported as explicit metrics for `SolarBatteryEnv` runs; the environment’s `reward` already combines grid economics and degradation cost.

Visualization:
- Mean reward bar with std; stacked costs with percent annotations; risk–return scatter; episode return distribution (box plot). All figures can be saved via `save_dir`.

Statistical testing (proposed):
- Bootstrap confidence intervals for mean differences; paired tests on per-customer aggregates.

> **TODO (needs source or implementation):** These statistical tests are not implemented in `src/helper.py` as of this report; either implement them, cite where they are done (e.g., a notebook), or keep this section labeled as proposed.

## 8. Preliminary Results and Evaluation Plan

We are currently conducting the initial comparative evaluation across Rule-based, SDP/MRDP, PPO, and Decision Transformer agents, with **DT as the primary research model** and the other approaches serving as (i) competitive baselines and (ii) data generators for offline learning.

In parallel, the repository now supports utility-scale evaluation in the AEMO/NEM setting via `AEMOBatteryTradingEnv` and `AEMOAgent`. Results for AEMO experiments will be added once a consistent set of AEMO episode logs and evaluation outputs are generated.

The first version of the comparative metrics is already stored in [eval_output/base/evaluation_metrics.csv](eval_output/base/evaluation_metrics.csv), and the accompanying return graph highlights the mean ± std for each agent.

![Mean episode return and variability by agent](eval_output/base/mean_reward.svg)

![Risk vs return for each agent](eval_output/base/risk_return.svg)

![Episode return distribution across customers](eval_output/base/episode_distribution.svg)

![Net grid energy balance by agent](eval_output/base/grid_energy.svg)

Preliminary observations from the current runs (from `eval_output/base/evaluation_metrics.csv`) are:

- Mean episode return ranking in this run: Oracle (-2483.38) > DT (`dt_rtg0`, -2534.05) > SDP (-2598.35) > MRDP (-2766.60) > PPO (-2828.28) > Rule (-3077.26).
- Variability: in this run, Oracle has the smallest return standard deviation (std_reward ≈ 1774). Among the listed planners/learners, the Sharpe proxy values are similar (e.g., SDP ≈ -0.812, MRDP ≈ -0.822, DT ≈ -0.871), and all are negative because mean returns are negative.
- Grid energy: the metrics report average per-episode grid import/export (kWh) and net grid energy (kWh). Here, `avg_grid_net` is net import (import − export), so values around 3800–5100 indicate net import, not net export.
- **Decision Transformer sensitivity (repo-backed):** In `eval_output/dt_compare/evaluation_metrics.csv`, conditioning RTG affects outcomes. In this run, `dt_rtg_neg1500` has the best mean return among the shown DT variants (-2390.79).
- **Degradation dynamics (repo-backed):** The same DT comparison shows large differences in average degradation per episode: `dt_rtg_neg1500` ≈ 0.00166 vs `dt_rtg_neg1` ≈ 0.05666.

> **NOTE (DT-specific, repo-backed):** These `dt_rtg_*` experiment names correspond to different choices of the DT agent’s initial RTG prompt (`rtg_value` in `Agent(..., algorithm='dt')`). The agent then updates RTG online each step using the discounted recurrence described in Section 4.1.

> **NOTE (interpretation):** Explanations such as “out-of-distribution RTG prompts” are plausible hypotheses for DT sensitivity, but they are not directly established by these metrics alone. Keep such statements labeled as hypotheses unless you add an analysis of the training RTG distribution and prompt distances.

![DT sensitivity: Risk vs Return](eval_output/dt_compare/risk_return.svg)

![DT sensitivity: Episode Return Distribution](eval_output/dt_compare/episode_distribution.svg)

![DT sensitivity: Grid Energy and Degradation](eval_output/dt_compare/grid_energy.svg)

## 9. Proposed Research Roadmap

This framework provides the necessary tooling to pursue several high-impact research directions suitable for a doctoral thesis:

### Phase 1: Benchmarking and Algorithmic Analysis (Current Status)
- Establish the performance hierarchy between model-based (SDP) and model-free (RL) approaches.
- Quantify gaps between decentralized baselines and planning-based baselines.

In line with review-identified gaps, an additional near-term objective is to make evaluation more comparable across algorithm families by using consistent environment dynamics, observation/action conventions, and standardized logging [6].

> **TODO (needs source/definition):** “Price of Anarchy” has a specific game-theoretic meaning. If you intend to use it formally, define the game + equilibrium concept and add citations; otherwise keep it as a general “performance gap” statement.

### Phase 2: Robustness and Generalization (Year 1-2)
- **Distributional Shift:** Investigate how Offline RL (Decision Transformers) generalizes to unseen weather patterns or customer load profiles compared to Online RL.
- **Risk-Sensitive Control:** Integrate CVaR constraints into the RL objective to develop agents that avoid catastrophic costs during extreme weather events.

DT-centric near-term extensions (repo-aligned):
- **Prompt calibration:** use the repo’s `recommended_rtg` / `recommended_return_scale` diagnostics to choose RTG prompts that are in-distribution relative to the logged training data.
- **Training data mixture studies:** systematically vary which behavior policies generate the offline dataset (rule-based vs SDP vs SB3) and evaluate how DT performance changes.
- **Long-context modeling:** increase `context_len` and/or enable RoPE to better represent weekly/seasonal structure, and evaluate sensitivity to context truncation.

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

> **TODO (needs source or commit):** If you want to claim checksums are recorded, add the mechanism (e.g., a script that hashes datasets/models) or cite the exact output file where checksums are stored.

## 11. Conclusion

We introduce a unified, open framework for learning and planning in battery control with degradation- and risk-aware evaluation across two settings: (i) household solar–battery–grid control under tariffs and (ii) utility-scale battery trading under AEMO/NEM market signals. The repository supports rule-based control, RL, SDP/MRDP, dispatch replay (AEMO), and Decision Transformers, with standardized preprocessing and environment-agnostic metrics. This report documents the system and experimental protocol; results will follow in an updated version and accompanying repository tags.

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
- Rollout and save: `flatten_episode_data(run_sb3_model_on_vec_env(ppo_model, SubprocVecEnv(test_env_fns))).write_parquet("data/ppo_test_episode_logs.parquet")`.

DT
- Train (CLI): `python -m src.pretrain_decision_transformer --data-dir data --model-config models/decision_transformer_model_kwargs.json --epochs 2 --batch-size 6 --lr 2e-5 --return-scale 1.0`.
- Dataset (Python): `TrajectoryDataset(data_path=..., context_length=..., state_dim=..., act_dim=..., discount_factor=0.99)` → train with `train_decision_transformer` and evaluate via `Agent(algorithm='dt', rtg_value=...)`.

