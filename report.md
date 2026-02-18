# Benchmarking and Advancing Control Strategies for Residential Energy Storage: A Unified Framework for Reinforcement Learning and Optimization

## Abstract

The effective integration of residential solar and battery storage is critical for the transition to a decentralized, renewable energy grid. Developing and comparing control strategies can be challenging when environments omit key factors such as stochastic load/generation, time-varying tariffs, and battery degradation. This report documents a research codebase that provides a Gymnasium-compatible environment, a suite of baselines (rule-based heuristics, dynamic programming planners, online RL via Stable-Baselines3, and offline sequence modeling via a Decision Transformer), and an evaluation workflow focused on return, grid energy flows, and degradation.

## 1. Introduction

The proliferation of distributed energy resources (DERs), specifically residential solar PV and battery storage, presents both a challenge and an opportunity for modern power grids. While these assets can significantly reduce consumer costs and provide grid flexibility, their optimal operation is non-trivial. The control problem is characterized by high stochasticity in demand and generation, complex time-of-use (ToU) tariffs, non-linear battery degradation dynamics, and strict physical constraints.

### 1.1 The Research Gap
Despite substantial literature on energy management systems (EMS), reproducibility and cross-paper comparability can be difficult when studies rely on custom environments, private data, and differing assumptions (e.g., constraint handling, tariff structure, or whether degradation is modeled). There is also a practical gap between model-based planning approaches (e.g., dynamic programming / MPC-style methods) and learning-based approaches (e.g., RL and sequence models), which motivates a unified benchmark.

> **TODO (needs source):** If you want to keep stronger statements like “the field suffers from a lack of reproducibility” or “there is a disconnect between communities”, add citations to survey/benchmarking papers; otherwise keep the phrasing general as above.

### 1.2 Contributions and Research Goals
This work establishes a consolidated, reproducible benchmark to address these limitations. We provide:
1.  **A High-Fidelity Simulation Environment:** A Gymnasium-compatible environment incorporating explicit constraints and degradation-aware reward shaping.
2.  **Diverse Algorithmic Baselines:** A unified interface for comparing Rule-based heuristics, Stochastic Dynamic Programming (SDP), Online RL (PPO, SAC, etc.), and Offline RL (Decision Transformers).
3.  **Comprehensive Evaluation Suite:** Standardized metrics for economic performance, battery health, and financial risk (Sharpe/Sortino ratios).

The goal of this platform is to serve as the foundational infrastructure for a PhD thesis investigating **robust, generalization-capable control policies for decentralized energy systems**.

## 2. Related Work

This work is inspired by Abdulla et al. [1], which formulates optimal operation of energy storage using Stochastic Dynamic Programming (SDP) and emphasizes the importance of uncertainty and degradation for realistic assessment.

We adopt an SDP-style planning baseline (implemented in this repository) and a multi-factor degradation model based on Muenzel et al. [2] (implemented in `src/batterydeg.py`). We extend the planning baseline with additional learning-based baselines and a Gymnasium environment wrapper.
1.  **Modernizing the Interface:** Wrapping the simulation in a standard Gymnasium API to bridge the gap between the optimization and Deep RL communities.
2.  **Expanding the Algorithmic Suite:** Introducing Online Deep RL (PPO, SAC) and Offline RL (Decision Transformers) to compare learning-based approaches against the theoretical optimality of SDP.
3.  **Open Reproducibility:** Providing a fully open-source, containerized benchmark, addressing the lack of public code in prior studies.

## 3. System Model and Environment

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

## 4. Methods

Baselines and planners are implemented in `src/decision.py` (Agent abstraction):
- Rule-based: a heuristic using surplus/deficit logic, optional persistence, and small injected noise (see `Agent.rule_based_action`).
- RL (SB3): PPO/A2C/DDPG/SAC/TD3 via `src/sb3train.py::train_model`; rollouts collected by `run_sb3_model_on_vec_env` and flattened with `flatten_episode_data`.
- SDP: a self-contained dynamic programming baseline implemented in `src/sdp_algorithm.py`. It discretizes SoC and actions and performs backward induction. Uncertainty can be handled via scenario sampling (Monte Carlo) when enabled.
- MRDP: `algorithm='mrdp'` with `subhorizon_specs` for coarse-to-fine planning, addressing the "curse of dimensionality" inherent in standard SDP.
- Decision Transformer (DT): Offline sequence model proposed by Chen et al. [4] (`src/decision_transformer.py`) trained on `TrajectoryDataset` from logged trajectories (`src/transformer_training.py`). Inference uses `model.get_action` with rolling context.

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

## 6. Experimental Setup

Splits and seeds:
- Train/test split by customer ID (e.g., 80/20), fixed seeds, and config logging are recommended for reproducibility.

> **TODO (needs source or repo pointer):** If you want to claim a specific split protocol is used in current experiments, cite the exact notebook/script where the split and seeding are performed.

Workflows:
- RL: `DummyVecEnv` for training, `SubprocVecEnv` for evaluation; train with `train_model(..., default_model=True)` or enable Optuna tuning.
- SDP/MRDP: configure horizons/resolutions; evaluate single or parallel episodes via `run_episodes_parallel`.
- DT: build `TrajectoryDataset` from logs; train with `train_decision_transformer` (AMP, gradient clipping, scheduler, checkpoints).

Compute and reproducibility:
- Containerization: the repository includes a `Dockerfile` and `docker-compose.yml` for running a consistent environment.
- Figures: `evaluate_experiments(..., save_dir=..., save_format=...)` can save plots (default `save_format='svg'`).

## 7. Metrics and Analysis

Primary metrics (from `src/helper.py`):
- Episode return statistics: mean/median/std and 5th/95th percentiles (computed from per-episode sums of the logged `reward`).
- Grid energy flows: average per-episode grid import (kWh), grid export (kWh), and net grid energy (kWh) derived from `info['grid_energy']`.
- Degradation: average per-episode and per-step degradation derived from `info['step_degradation']`.
- Risk proxies: Sharpe and Sortino are computed directly from the distribution of episode returns (not annualized; Sharpe is `mean/std`).

> **NOTE:** A separate “cost decomposition” into grid import cost vs export revenue is not currently reported as explicit metrics for `SolarBatteryEnv` runs; the environment’s `reward` already combines grid economics and degradation cost.

Visualization:
- Mean reward bar with std; stacked costs with percent annotations; risk–return scatter; episode return distribution (box plot). All figures can be saved via `save_dir`.

Statistical testing (proposed):
- Bootstrap confidence intervals for mean differences; paired tests on per-customer aggregates.

> **TODO (needs source or implementation):** These statistical tests are not implemented in `src/helper.py` as of this report; either implement them, cite where they are done (e.g., a notebook), or keep this section labeled as proposed.

## 8. Preliminary Results and Evaluation Plan

We are currently conducting the initial comparative evaluation across Rule-based, SDP/MRDP, PPO, and Decision Transformer agents. 

The first version of the comparative metrics is already stored in [eval_output/base/evaluation_metrics.csv](eval_output/base/evaluation_metrics.csv), and the accompanying return graph highlights the mean ± std for each agent.

![Mean episode return and variability by agent](eval_output/base/mean_reward.svg)

![Risk vs return for each agent](eval_output/base/risk_return.svg)

![Episode return distribution across customers](eval_output/base/episode_distribution.svg)

![Net grid energy balance by agent](eval_output/base/grid_energy.svg)

Preliminary observations from the current runs (from `eval_output/base/evaluation_metrics.csv`) are:

- Mean episode return ranking in this run: Oracle (-2483.38) > DT (`dt_rtg0`, -2534.05) > SDP (-2598.35) > MRDP (-2766.60) > PPO (-2828.28) > Rule (-3077.26).
- Variability: in this run, Oracle has the smallest return standard deviation (std_reward ≈ 1774). Among the listed planners/learners, the Sharpe proxy values are similar (e.g., SDP ≈ -0.812, MRDP ≈ -0.822, DT ≈ -0.871), and all are negative because mean returns are negative.
- Grid energy: the metrics report average per-episode grid import/export (kWh) and net grid energy (kWh). Here, `avg_grid_net` is net import (import − export), so values around 3800–5100 indicate net import, not net export.
- **Decision Transformer sensitivity (repo-backed):** In `eval_output/dt_compare/evaluation_metrics.csv`, conditioning RTG affects outcomes: `dt_rtg_neg1500` has the best mean return among the shown DT variants (-2390.79), while more aggressive conditioning (`dt_rtg_neg400`, -2448.29) is worse.
- **Degradation dynamics (repo-backed):** The same DT comparison shows large differences in average degradation per episode: `dt_rtg_neg1500` ≈ 0.00166 vs `dt_rtg_neg1` ≈ 0.05666.

> **NOTE (interpretation):** Explanations such as “out-of-distribution RTG prompts” are plausible hypotheses for DT sensitivity, but they are not directly established by these metrics alone. Keep such statements labeled as hypotheses unless you add an analysis of the training RTG distribution and prompt distances.

![DT sensitivity: Risk vs Return](eval_output/dt_compare/risk_return.svg)

![DT sensitivity: Episode Return Distribution](eval_output/dt_compare/episode_distribution.svg)

![DT sensitivity: Grid Energy and Degradation](eval_output/dt_compare/grid_energy.svg)

## 9. Proposed Research Roadmap

This framework provides the necessary tooling to pursue several high-impact research directions suitable for a doctoral thesis:

### Phase 1: Benchmarking and Algorithmic Analysis (Current Status)
- Establish the performance hierarchy between model-based (SDP) and model-free (RL) approaches.
- Quantify gaps between decentralized baselines and planning-based baselines.

> **TODO (needs source/definition):** “Price of Anarchy” has a specific game-theoretic meaning. If you intend to use it formally, define the game + equilibrium concept and add citations; otherwise keep it as a general “performance gap” statement.

### Phase 2: Robustness and Generalization (Year 1-2)
- **Distributional Shift:** Investigate how Offline RL (Decision Transformers) generalizes to unseen weather patterns or customer load profiles compared to Online RL.
- **Risk-Sensitive Control:** Integrate CVaR constraints into the RL objective to develop agents that avoid catastrophic costs during extreme weather events.

### Phase 3: Advanced Architectures and Multi-Agent Systems (Year 2-3)
- **Transformer Architectures:** Explore modifications to the Decision Transformer architecture (e.g., long-context attention) to better capture seasonal periodicities in energy data.
- **Multi-Agent Coordination:** Extend the environment to a microgrid setting where multiple homes trade energy, studying the emergence of cooperative behaviors.

### Phase 4: Sim-to-Real Transfer (Year 3-4)
- Develop "safe RL" wrappers to ensure constraints are met during deployment.
- Validate policies on hardware-in-the-loop setups or pilot deployments.

## 10. Reproducibility and Artifacts

- Code pointers: environment (`src/EnergySimEnv.py`), agents (`src/decision.py`), SB3 training (`src/sb3train.py`), DT (`src/decision_transformer.py`, `src/transformer_training.py`), preprocessing and evaluation (`src/helper.py`).
- Determinism: use fixed seeds, log configs, and prefer containers for repeatability (recommended practice).
- Artifacts: this repository stores models under `models/` and evaluation outputs under `eval_output/`.

> **TODO (needs source or commit):** If you want to claim checksums are recorded, add the mechanism (e.g., a script that hashes datasets/models) or cite the exact output file where checksums are stored.

## 11. Conclusion

We introduce a unified, open framework for learning and planning in solar–battery–grid control with degradation- and risk-aware evaluation. It supports rule-based control, RL, SDP/MRDP, and Decision Transformers, with standardized preprocessing and metrics. This report documents the system and experimental protocol; results will follow in an updated version and accompanying repository tags.

## References

[1] K. Abdulla, J. De Hoog, et al., "Optimal Operation of Energy Storage Systems Considering Forecasts and Battery Degradation," *IEEE Transactions on Smart Grid*, 2016.
[2] V. Muenzel, J. De Hoog, et al., "A Multi-Factor Battery Cycle Life Prediction Methodology for Optimal Battery Management," *IEEE Transactions on Industrial Electronics*, 2015.
[3] Sutton & Barto. Reinforcement Learning: An Introduction.
[4] Chen et al. Decision Transformer: Reinforcement Learning via Sequence Modeling.
[5] Ausgrid. Solar home electricity data. https://github.com/pierre-haessig/ausgrid-solar-data?tab=readme-ov-file. Accessed April 2017.

---

Appendix A: Minimal Experiment Recipes

RL
- Train: `ppo_model, _ = train_model(PPO, DummyVecEnv([make_env(ds) for ds in train_ds]), eval_env_fn=test_env_fns[0], default_model=True)`.
- Rollout and save: `flatten_episode_data(run_sb3_model_on_vec_env(ppo_model, SubprocVecEnv(test_env_fns))).write_parquet("data/ppo_test_episode_logs.parquet")`.

DT
- Dataset: `TrajectoryDataset(data_path=..., context_length=36, state_dim, act_dim)` → train with `train_decision_transformer` and evaluate via `Agent(algorithm='dt')`.

