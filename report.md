# Benchmarking and Advancing Control Strategies for Residential Energy Storage: A Unified Framework for Reinforcement Learning and Optimization

## Abstract

The effective integration of residential solar and battery storage is critical for the transition to a decentralized, renewable energy grid. However, the development of optimal control strategies is hindered by the lack of standardized benchmarks that account for stochastic load/generation, complex tariffs, and battery degradation. This report presents a comprehensive, open-source research framework designed to bridge this gap. We introduce a Gymnasium-compatible environment, a suite of diverse baselines—ranging from heuristic rules and stochastic dynamic programming (SDP) to online Reinforcement Learning (PPO/SAC) and offline Decision Transformers—and a unified evaluation protocol focusing on cost, risk, and degradation. This platform serves as a foundation for doctoral research into robust, data-driven energy management, enabling rigorous comparison of model-based and model-free approaches under realistic uncertainty.

## 1. Introduction

The proliferation of distributed energy resources (DERs), specifically residential solar PV and battery storage, presents both a challenge and an opportunity for modern power grids. While these assets can significantly reduce consumer costs and provide grid flexibility, their optimal operation is non-trivial. The control problem is characterized by high stochasticity in demand and generation, complex time-of-use (ToU) tariffs, non-linear battery degradation dynamics, and strict physical constraints.

### 1.1 The Research Gap
Despite the abundance of literature on energy management systems (EMS), the field suffers from a lack of reproducibility and standardization. Studies often employ custom, simplified environments that neglect critical factors like battery health or realistic tariff structures. Furthermore, there is a disconnect between the optimization community (focusing on MPC/SDP) and the learning community (focusing on RL/Transformers), with few benchmarks allowing for a fair, rigorous comparison of these distinct paradigms.

### 1.2 Contributions and Research Goals
This work establishes a consolidated, reproducible benchmark to address these limitations. We provide:
1.  **A High-Fidelity Simulation Environment:** A Gymnasium-compatible environment incorporating explicit constraints and degradation-aware reward shaping.
2.  **Diverse Algorithmic Baselines:** A unified interface for comparing Rule-based heuristics, Stochastic Dynamic Programming (SDP), Online RL (PPO, SAC, etc.), and Offline RL (Decision Transformers).
3.  **Comprehensive Evaluation Suite:** Standardized metrics for economic performance, battery health, and financial risk (Sharpe/Sortino ratios).

The goal of this platform is to serve as the foundational infrastructure for a PhD thesis investigating **robust, generalization-capable control policies for decentralized energy systems**.

## 2. Related Work (brief)

Energy storage dispatch has been explored with rule/control heuristics, stochastic dynamic programming, model predictive control, and reinforcement learning. Recent work applies offline RL and sequence models (Decision Transformers) to control from logged data. Our contribution is to bridge these lines in a single, open benchmark with reproducible pipelines and risk/degradation-aware evaluation.

## 3. System Model and Environment

Environment: `src/EnergySimEnv.py` defines `SolarBatteryEnv` with:
- Action: normalized battery power in [-1, 1]; mapped to kW and bounded by `max_battery_flow`, SoC, and grid limits.
- Observation: cyclical time features (sin/cos of hour/day) plus normalized data frame features and [battery_level, degr_cost].
- Dynamics: `step_duration` inferred from timestamps; grid energy per step capped by `max_grid_flow × step_duration`.
- Reward/cost: grid import cost/export revenue and degradation cost (see `src/batterydeg.py`). Violation penalties prevent infeasible behavior.
- Forecast features: one-step-ahead `FutureSolar`/`FutureLoad` enable planning-based agents.

Dataset contract (from `src/helper.py::transform_polars_df`):
`Timestamp, SolarGen, HouseLoad, FutureSolar, FutureLoad, ImportEnergyPrice, ExportEnergyPrice, Time` (sorted by `Time`).

## 4. Methods

Baselines and planners are implemented in `src/decision.py` (Agent abstraction):
- Rule-based: a heuristic with persistence and noise damping for stability.
- RL (SB3): PPO/A2C/DDPG/SAC/TD3 via `src/sb3train.py::train_model`; rollouts collected by `run_sb3_model_on_vec_env` and flattened with `flatten_episode_data`.
- SDP: `algorithm='sdp'` with horizon, `soc_resolution`, `action_resolution`. Vectorized backward induction with optional Monte Carlo over quantile scenarios (`src/quantile_scenarios.py`).
- MRDP: `algorithm='mrdp'` with `subhorizon_specs=[{start,length,soc_resolution,action_resolution,step_duration}, …]` for coarse-to-fine planning.
- Decision Transformer (DT): Offline sequence model (`src/decision_transformer.py`) trained on `TrajectoryDataset` from logged trajectories (`src/transformer_training.py`). Inference uses `model.get_action` with rolling context.

Risk-aware extensions (proposed): add CVaR-style evaluation and multi-objective scalarization for cost vs degradation; robustify SDP/MRDP using uncertainty bands.

## 5. Data and Preprocessing

Source data: public Solar home electricity datasets in `data/`, plus precomputed episode logs for various algorithms (Parquet). Preprocessing via `transform_polars_df` converts per-customer CSVs into the environment schema, with configurable tariffs (`price_periods`) and default vs peak prices (`ImportEnergyPrice`, `ExportEnergyPrice`). `make_env(dataset)` returns callables for vectorized execution.

## 6. Experimental Setup

Splits and seeds:
- Train/test split by customer ID (e.g., 80/20). Fix NumPy/Torch seeds and record configs.

Workflows:
- RL: `DummyVecEnv` for training, `SubprocVecEnv` for evaluation; train with `train_model(..., default_model=True)` or enable Optuna tuning.
- SDP/MRDP: configure horizons/resolutions; evaluate single or parallel episodes via `run_episodes_parallel`.
- DT: build `TrajectoryDataset` from logs; train with `train_decision_transformer` (AMP, gradient clipping, scheduler, checkpoints).

Compute and reproducibility:
- Dockerized environment (`docker-compose.yml`). Figures saved with `evaluate_experiments(..., save_dir=..., save_format='png')` for direct paper inclusion.

## 7. Metrics and Analysis

Primary metrics (from `src/helper.py`):
- Reward statistics: mean, median, std, 5th/95th percentiles per episode.
- Cost decomposition: `avg_grid_cost`, `avg_grid_revenue`, `avg_deg_cost`, and `avg_operational_cost = grid_cost − revenue + deg_cost`.
- Risk: Sharpe and Sortino. Optional CVaR@α extension.

Visualization:
- Mean reward bar with std; stacked costs with percent annotations; risk–return scatter; episode return distribution (box plot). All figures can be saved via `save_dir`.

Statistical testing:
- Bootstrap confidence intervals for mean differences; paired t-test or Wilcoxon on per-customer aggregates when appropriate.

## 8. Preliminary Results and Evaluation Plan

We are currently conducting the initial comparative evaluation across Rule-based, SDP/MRDP, PPO, and Decision Transformer agents. This section will be populated with:

- **Table 1: Comparative Performance:** Mean ± std episode reward and operational cost by algorithm, highlighting the trade-off between optimality (SDP) and computational tractability (RL).
- **Table 2: Cost Decomposition:** A detailed breakdown of grid cost, export revenue, and degradation cost to understand *how* agents achieve their results (e.g., does RL sacrifice battery health for short-term gain?).
- **Figure 1: Risk–Return Analysis:** Scatter plots (Return vs. Sharpe Ratio) to visualize the stability of learned policies.
- **Figure 2: Distributional Robustness:** Box plots of episode returns across diverse customer profiles to assess generalization.

We will include per-customer breakdowns in the appendix and release all plots in `eval_output/figures/`.

## 9. Proposed Research Roadmap

This framework provides the necessary tooling to pursue several high-impact research directions suitable for a doctoral thesis:

### Phase 1: Benchmarking and Algorithmic Analysis (Current Status)
- Establish the performance hierarchy between model-based (SDP) and model-free (RL) approaches.
- Quantify the "Price of Anarchy" in decentralized control vs. optimal centralized planning.

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
- Determinism: set seeds; log configs; prefer Docker. Store models in `models/`, logs in `data/`, and results in `eval_output/`.
- Public datasets and generated logs will be referenced in the final version with precise checksums.

## 11. Conclusion

We introduce a unified, open framework for learning and planning in solar–battery–grid control with degradation- and risk-aware evaluation. It supports rule-based control, RL, SDP/MRDP, and Decision Transformers, with standardized preprocessing and metrics. This report documents the system and experimental protocol; results will follow in an updated version and accompanying repository tags.

## References (selected)

[1] Sutton & Barto. Reinforcement Learning: An Introduction.
[2] Silver et al. Deterministic Policy Gradient Algorithms.
[3] Kostrikov et al. Offline Reinforcement Learning.
[4] Chen et al. Decision Transformer.
[5] Tariff and energy storage operation references (utility docs and related literature).

---

Appendix A: Minimal Experiment Recipes

RL
- Train: `ppo_model, _ = train_model(PPO, DummyVecEnv([make_env(ds) for ds in train_ds]]), eval_env_fn=test_env_fns[0], default_model=True)`.
- Rollout and save: `flatten_episode_data(run_sb3_model_on_vec_env(ppo_model, SubprocVecEnv(test_env_fns))).write_parquet("data/ppo_test_episode_logs.parquet")`.

DT
- Dataset: `TrajectoryDataset(data_path=..., context_length=36, state_dim, act_dim)` → train with `train_decision_transformer` and evaluate via `Agent(algorithm='dt')`.

