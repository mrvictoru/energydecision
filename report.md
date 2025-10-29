## Title

Learning and Planning for Residential Solar–Battery–Grid Control: An Open, Reproducible Benchmark with RL, SDP/MRDP, and Decision Transformers

## Abstract

We present an open codebase for evaluating control algorithms that operate a residential solar–battery–grid system under uncertain load, generation, and time-varying tariffs. The framework provides a Gymnasium environment with degradation-aware rewards, a suite of baseline agents (rule-based, Stable Baselines3 RL), planning methods (stochastic dynamic programming, multi-resolution dynamic programming), and an offline learning pipeline using a Decision Transformer. We standardize preprocessing and metrics (cost, revenue, degradation, operational cost, and risk metrics such as Sharpe/Sortino). This report documents the system design, methods, experimental setup, and evaluation protocol. Initial comparative results across algorithms will be reported in an updated version; we outline figure and table placeholders and ensure reproducibility via Docker and deterministic settings.

## 1. Introduction

Residential batteries with rooftop solar can reduce electricity cost and provide grid services. Operating such assets is challenging due to stochastic demand and generation, time-of-use tariffs, physical constraints, and battery degradation. Research often evaluates methods in disparate setups, hindering comparability. This work consolidates a reproducible benchmark: a standardized simulation environment, algorithm baselines spanning control, planning, and learning, and a unified evaluation suite. The goal is to accelerate research on robust, cost- and degradation-aware controllers.

Contributions:
- A Gymnasium-compatible environment with explicit grid and battery constraints, and degradation-aware reward shaping.
- Baseline agents and planners: a robust rule policy, SB3-based RL, SDP, MRDP, and a Decision Transformer pipeline for offline RL.
- Unified preprocessing, metrics, and plotting (including risk–return visualization), with support for saving figures programmatically.

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

## 8. Results (TBD)

We are currently running the first evaluation across rule, SDP/MRDP, PPO, and DT. This section will report:
- Table 1: Mean ± std episode reward and operational cost by algorithm.
- Table 2: Cost decomposition (grid cost, export revenue, degradation cost).
- Figure 1: Risk–return scatter (colour=Sharpe).
- Figure 2: Episode return distribution (box plot) and stacked cost bars.

We will include per-customer breakdowns in the appendix and release all plots in `eval_output/figures/`.

## 9. Discussion and Limitations

The environment abstracts network and device details (efficiencies and dynamics are simplified), and degradation models are approximate (static or linear). However, explicit constraints and degradation-aware rewards improve realism over purely economic formulations. We plan to add richer degradation models and scenario generation. Sim-to-real transfer remains future work.

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

