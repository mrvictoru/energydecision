# Energy Decision - Source Components Reference

This document provides comprehensive documentation for all key components in the `src/` directory of the energydecision project.

---

## Table of Contents

1. [Environment (`EnergySimEnv.py`)](#1-environment-energysimenvpy)
2. [AEMO Market Environment (`AEMOBatteryEnv.py`)](#2-aemo-market-environment-aemobatteryenvpy)
3. [AEMO Data Pipeline (`aemo_data.py`)](#3-aemo-data-pipeline-aemo_datapy)
4. [Decision Agent (`decision.py`)](#4-decision-agent-decisionpy)
5. [Multi-Resolution Dynamic Programming (`mrdp_algorithm.py`)](#5-multi-resolution-dynamic-programming-mrdp_algorithmpy)
6. [Scenario Generation (`quantile_scenarios.py`)](#6-scenario-generation-quantile_scenariospy)
7. [Battery Degradation (`batterydeg.py`)](#7-battery-degradation-batterydegpy)
8. [Data Transformation (`helper.py`)](#8-data-transformation-helperpy)
9. [Decision Transformer (`decision_transformer.py`)](#9-decision-transformer-decision_transformerpy)
10. [Transformer Training (`transformer_training.py`)](#10-transformer-training-transformer_trainingpy)
11. [Stable-Baselines3 Training (`sb3train.py`)](#11-stable-baselines3-training-sb3trainpy)
12. [Dispatch Replay Utilities (`dispatch_utils.py`)](#12-dispatch-replay-utilities-dispatch_utilspy)
13. [Performance Optimizations](#13-performance-optimizations)

---

## 1. Environment (`EnergySimEnv.py`)

### Overview

`SolarBatteryEnv` is a Gymnasium-compatible environment simulating a household with solar PV, battery storage, and grid connection. For a detailed description of the physics, logic, and reward function, see **[docs/HOUSEHOLD_ENV_README.md](docs/HOUSEHOLD_ENV_README.md)**. It features realistic constraints, time-of-use tariffs, and degradation-aware rewards.

### Key Features

- **Gymnasium Interface**: Standard `reset()`, `step()`, `render()` methods
- **Normalized Observations**: Suitable for reinforcement learning methods
- **Degradation Modeling**: Integrates with `batterydeg.py` for battery aging costs
- **Dynamic Correction**: Adaptive correction factors for improved reward shaping

### Basic Usage

```python
import polars as pl
from src.EnergySimEnv import SolarBatteryEnv
from src.helper import transform_polars_df

# Load and transform data
df = pl.read_csv("data/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
customer_df = df.filter(pl.col("Customer") == df["Customer"][0])
dataset = transform_polars_df(
    customer_df,
    import_energy_price=0.23,
    export_energy_price=0.015,
    price_periods="7am-10am | 4pm-9pm",
    default_import_energy_price=0.15,
    default_export_energy_price=0.01,
)

# Create environment
env = SolarBatteryEnv(
    dataset,
    battery_capacity=10.0,      # kWh
    max_battery_flow=5.0,       # kW
    init_battery_level=5.0,     # kWh
    max_step=1000
)

# Run a simple episode
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | Required | Energy data with Time, SolarGen, HouseLoad, prices |
| `battery_capacity` | `float` | 10.0 | Battery capacity in kWh |
| `max_battery_flow` | `float` | 5.0 | Max charge/discharge rate in kW |
| `max_grid_flow` | `float` | 10.0 | Max grid import/export in kW |
| `init_battery_level` | `float` | 5.0 | Initial battery state in kWh |
| `max_step` | `int` | 1000 | Max steps per episode |
| `battery_life_cost` | `float` | 1000.0 | Battery replacement cost |
| `step_duration` | `float` | 0.5 | Time step duration in hours |

---

## 2. AEMO Market Environment (`AEMOBatteryEnv.py`)

### Overview

`AEMOBatteryTradingEnv` is a Gymnasium-compatible environment that simulates a grid-scale battery participating in AEMO energy and FCAS markets. For a detailed description of the market mechanics and observation space, see **[docs/AEMO_ENV_README.md](docs/AEMO_ENV_README.md)**. It consumes preprocessed AEMO data (Polars) and supports single-market (energy-only) or multi-market (energy + FCAS) action modes.

### Key Components

- **`AEMODataPreprocessor`**: Resamples AEMO datasets to the environment step duration, adds cyclical time features, handles missing data, and normalizes inputs.
- **`AEMOBatteryTradingEnv`**: Uses normalized market features and battery SOC to compute rewards based on energy arbitrage and optional FCAS revenue.

### Basic Usage

```python
from datetime import datetime
from src.AEMOBatteryEnv import create_aemo_env_from_data

env = create_aemo_env_from_data(
    start_date=datetime(2024, 6, 1),
    end_date=datetime(2024, 6, 7),
    region="NSW1",
    battery_capacity=10.0,
    max_battery_flow=5.0,
    init_battery_level=5.0,
    max_step=288,
    step_duration=0.5,
    action_mode="multi_market",
)

obs, info = env.reset()
```

---

## 3. AEMO Data Pipeline (`aemo_data.py`)

### Overview

Provides Polars-based data ingestion from NEMOSIS, with caching and fallbacks for AEMO static table downloads. The main entry points are `fetch_aemo_dispatch_price`, `fetch_aemo_fcas_price`, `fetch_aemo_generation_by_fuel`, and the convenience wrappers `fetch_aemo_data_bundle` / `fetch_aemo_data_bundle_with_dispatch`.

### Basic Usage

```python
from datetime import datetime
from src.aemo_data import fetch_aemo_data_bundle

data = fetch_aemo_data_bundle(
    start_date=datetime(2023, 6, 1),
    end_date=datetime(2023, 6, 2),
    region="NSW1",
    fcas_services=["RAISEREG", "LOWERREG"],
    fuel_types=["solar", "wind"],
)

prices = data["prices"]
fcas = data["fcas"]
generation = data["generation"]
```

### Notes

- Data is cached under `data/aemo/` by default.
- If AEMO static tables are blocked, set `AEMO_GENERATORS_FILE` to a local XLS/XLSX/CSV file.

---

## 4. Decision Agent (`decision.py`)

### Overview

The `Agent` class implements multiple control strategies for the solar-battery system:
- **Rule-based**: Heuristic controller with safety constraints
- **SDP**: Stochastic Dynamic Programming with receding horizon
- **Oracle**: Exhaustive single-step search (for validation)
- **RL Models**: Load pre-trained Stable-Baselines3 models
- **Decision Transformer**: Offline RL using transformer architecture

### Basic Usage

```python
from src.decision import Agent
from src.EnergySimEnv import SolarBatteryEnv

# Create environment and agent
env = SolarBatteryEnv(dataset)
agent = Agent(env, algorithm="sdp", horizon=48, soc_resolution=20, action_resolution=41)

# Run an episode
episode_log = agent.run_episode()
print(episode_log.head())
```

### Algorithm Options

```python
# Rule-based agent
agent_rule = Agent(env, algorithm="rule")

# SDP agent with Monte Carlo
agent_sdp = Agent(
    env, 
    algorithm="sdp",
    horizon=48,
    soc_resolution=20,
    action_resolution=41,
    use_monte_carlo=True,
    mc_samples=100,
    mc_seed=42
)

# Oracle agent (exhaustive search)
agent_oracle = Agent(env, algorithm="oracle", horizon=1, action_resolution=5)

# Decision Transformer agent
agent_dt = Agent(
    env,
    algorithm="dt",
    dt_model=trained_model,  # Pre-loaded DecisionTransformer
    dt_context_len=36
)

# AEMO Agent (grid-scale market agent)
# Supports: 'rule', 'dispatch' (replay), 'rl', 'dt'
from src.decision import AEMOAgent

# Create AEMO agent to replay a unit's dispatch as actions
# (use fetch_aemo_unit_dispatch to obtain unit dispatch data)
# Example: dispatch_df = fetch_aemo_unit_dispatch(start, end, duid='LBBG1')
aemo_agent = AEMOAgent(env_aemo, algorithm='dispatch', dispatch_data=dispatch_df, dispatch_duid='LBBG1')

# Rule-based AEMO agent (price arbitrage heuristic)
aemo_rule = AEMOAgent(env_aemo, algorithm='rule')

# Decision Transformer / RL usage is the same: provide a trained model to the agent
# aemo_dt = AEMOAgent(env_aemo, algorithm='dt', model=trained_dt_model)
```

### Running Multiple Environments in Parallel

```python
import numpy as np
import polars as pl
from src.helper import make_env, transform_polars_df
from src.decision import Agent, run_episodes_parallel

# Load data
df = pl.read_csv("data/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
customers = df["Customer"].unique()
rng = np.random.default_rng(seed=42)
sample_ids = rng.choice(customers, size=4, replace=False)

# Create datasets for each customer
datasets = []
for cid in sample_ids:
    customer_df = df.filter(pl.col("Customer") == cid)
    datasets.append(transform_polars_df(
        customer_df,
        import_energy_price=0.23,
        export_energy_price=0.015,
        price_periods="7am-10am | 4pm-9pm",
        default_import_energy_price=0.15,
        default_export_energy_price=0.01,
    ))

# Create environments
envs = [make_env(ds)() for ds in datasets]

# Run parallel episodes with SDP agent
agent_kwargs = {
    'algorithm': 'sdp',
    'soc_resolution': 21,
    'action_resolution': 41,
    # SDP uses rainflow-based degradation internally; no need to pass degradation_model
}

episode_logs = run_episodes_parallel(
    Agent,
    envs,
    agent_kwargs=agent_kwargs,
    max_workers=4,
)
print(f"Completed {len(episode_logs)} episodes")
```

---

## 5. Algorithm Implementations (SDP, MRDP, Oracle)

The project implements three primary optimization solvers as self-contained classes to make the algorithm flow easy to read and test:

- **Stochastic Dynamic Programming (SDP)** — `src/sdp_algorithm.py` implements the `SDPSolver` class. Use this for standard receding-horizon stochastic DP with Monte Carlo or deterministic stage costs. Example:

```python
from src.sdp_algorithm import SDPSolver
solver = SDPSolver(env, horizon=48, soc_resolution=20, action_resolution=41)
policy = solver.solve(forecasts, start_index=0)
```

- **Multi-Resolution Dynamic Programming (MRDP)** — `src/mrdp_algorithm.py` implements the `MRDPSolver` class and is recommended for long horizons where a coarse-far/ fine-near decomposition improves speed. Run the Agent with `algorithm='mrdp'` to use it.

```python
from src.mrdp_algorithm import MRDPSolver
mrdp = MRDPSolver(env, subhorizon_specs=subhorizon_specs)
policy = mrdp.solve(forecasts, start_index=0)
```

- **Oracle (perfect information)** — `src/oracle_algorithm.py` implements `OracleSolver` for benchmarking using actual future values.

Agent integration: The `Agent` class creates and calls these solvers based on the `algorithm` parameter (e.g., `'sdp'`, `'mrdp'`, `'oracle'`). This makes it easy to switch solvers without changing calling code.

> For a deep dive into the mathematical logic, inner workings, and a guide on how to read/debug these files, see **[docs/DP_ALGORITHM_README.md](docs/DP_ALGORITHM_README.md)**.

---

## 6. Scenario Generation (`quantile_scenarios.py`)

### Overview

The `QuantileScenarioGenerator` creates multiple scenarios from historical energy data based on quantiles. This is essential for uncertainty modeling in stochastic optimization.

### Key Features

- **Quantile-based scenarios**: Statistical quantiles of historical data
- **Automatic column detection**: Identifies suitable numeric columns
- **Grouped generation**: Different scenarios per customer/location
- **Monte Carlo integration**: Expected cost computation methods

### Basic Usage

```python
import polars as pl
from src.quantile_scenarios import QuantileScenarioGenerator

# Load energy data
df = pl.read_csv("energy_data.csv")

# Create scenario generator (defaults to 5 scenarios)
generator = QuantileScenarioGenerator()

# Generate scenarios for specific columns
scenarios_df = generator.generate_scenarios(
    df, 
    columns=['SolarGen', 'HouseLoad', 'ImportEnergyPrice']
)

print(scenarios_df.columns)
# Output: [...original columns..., 'scenario_1_SolarGen', 'scenario_2_SolarGen', ...]
```

### Configuration Options

```python
# Custom number of scenarios
generator = QuantileScenarioGenerator(n_scenarios=3)
# Uses quantiles: [0.25, 0.5, 0.75]

# Custom quantiles
generator = QuantileScenarioGenerator(
    n_scenarios=5,
    quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]
)

# Custom prefix
generator = QuantileScenarioGenerator(scenario_prefix="forecast")
# Creates: 'forecast_1_SolarGen', 'forecast_2_SolarGen', etc.
```

### Generate Time-Step Scenario Arrays for SDP

```python
generator = QuantileScenarioGenerator(n_scenarios=5)

# Generate per-timestep scenario arrays
scenario_cache = generator.generate_time_step_scenarios(df)

# Returns dict with arrays for each variable:
# scenario_cache['solar'] -> (values_array, probabilities_array)
# scenario_cache['load'] -> (values_array, probabilities_array)
# scenario_cache['import_price'] -> (values_array, probabilities_array)
# scenario_cache['export_price'] -> (values_array, probabilities_array)
```

### Expected Cost Computation

```python
# Define a stage cost function
def stage_cost_fn(solar, load, imp_price, exp_price):
    grid_energy = load - solar
    if grid_energy > 0:
        return grid_energy * imp_price
    else:
        return grid_energy * exp_price

# Monte Carlo expected cost (for large scenario counts)
mc_cost = generator.expected_cost_monte_carlo(
    values_solar, probs_solar,
    values_load, probs_load,
    values_imp, probs_imp,
    values_exp, probs_exp,
    stage_cost_fn,
    n_samples=1000,
    rng_seed=42
)

# Exact Cartesian expected cost (for small scenario counts)
exact_cost = generator.expected_cost_cartesian(
    values_solar, probs_solar,
    values_load, probs_load,
    values_imp, probs_imp,
    values_exp, probs_exp,
    stage_cost_fn
)
```

### Grouped Scenario Generation

```python
# Generate scenarios by customer
scenarios_df = generator.generate_scenarios(
    df, 
    columns=['SolarGen', 'HouseLoad'],
    group_by='Customer'
)
# Each customer gets their own scenario values based on their historical data
```

---

## 7. Battery Degradation (`batterydeg.py`)

### Overview

Implements semi-empirical battery degradation models. The module provides three model classes:

1. **`DegradationModel`** — Muenzel et al. (2015) multi-factor cycle-life model with rainflow cycle counting. Best for cell-level research and the household environment.
2. **`RealWorldBESSDegradationModel`** — Combined calendar + cycle aging model for utility-scale BESS, adapted from the framework in Kampker et al. (2025, doi:10.3390/batteries11110392). Uses Arrhenius temperature dependency and NMC/LFP chemistry presets. **Recommended for AEMO grid-scale simulations.**
3. **`RainflowCounter`** — ASTM E1049-85 rainflow cycle counting, used by both degradation models.

For a detailed explanation of the math, the factors, the Rainflow implementation, and the real-world BESS model, see **[docs/BATTERY_DEGRADATION_DETAILS.md](docs/BATTERY_DEGRADATION_DETAILS.md)**.

### Available Models

1. **DegradationModel (class)** — Muenzel et al. (2015) API. Provides:
   - `nCL_T(T)`, `nCL_Id(Id)`, `nCL_Ich(Ich)`, `nCL_SOCav_DOD(SOCav, DOD)` — normalized cycle-life factors
   - `cycle_life(T, Id, Ich, SOCav, DOD)` — combined cycle life (CL)
   - `degradation_per_cycle(T, Id, Ich, SOCav, DOD)` — fractional degradation per cycle (1 / CL)
2. **RealWorldBESSDegradationModel (class)** — Kampker et al. (2025) adaptation. Provides:
   - `calendar_aging_per_step(T_celsius, soc_frac, dt_hours)` — fractional calendar capacity loss per timestep
   - `cycle_aging_per_cycle(T_celsius, dod_pct, c_rate)` — fractional cycle capacity loss per rainflow cycle
   - `describe()` — parameter introspection dict
   - Chemistry presets: `BESS_CHEMISTRY_PRESETS['NMC']` and `BESS_CHEMISTRY_PRESETS['LFP']`
3. **Static helper** — `static_degradation(Id, Ich, SoC_avg, DoD)` (convenience wrapper using a default `DegradationModel`)
4. **Dynamic / Rainflow** — `RainflowCounter`, `rainflow_counting(...)`, `dynamic_degradation(...)` for extracting closed cycles and computing cumulative aging from SoC time series

### Basic Usage

```python
from src.batterydeg import (
    DegradationModel,
    RealWorldBESSDegradationModel,
    BESS_CHEMISTRY_PRESETS,
    static_degradation,
    rainflow_counting,
    dynamic_degradation,
)

# --- Muenzel et al. (2015) model ---
model = DegradationModel(CL_nom=3650.0, T_nom=25.0, Id_nom=0.25, Ich_nom=0.125, SOCav_nom=50.0, DOD_nom=90.0)

# Normalized factors
discharge_factor = model.nCL_Id(0.3)
charge_factor = model.nCL_Ich(0.15)
soc_dod_factor = model.nCL_SOCav_DOD(50.0, 80.0)

# Combined cycle life and per-cycle degradation
CL = model.cycle_life(T=25.0, Id=0.3, Ich=0.1, SOCav=50.0, DOD=80.0)
deg_per_cycle = model.degradation_per_cycle(T=25.0, Id=0.3, Ich=0.1, SOCav=50.0, DOD=80.0)

# Static convenience wrapper (keeps backward compatibility)
deg_cost = static_degradation(Id=0.3, Ich=0.15, SoC_avg=50.0, DoD=80.0)

# Dynamic / rainflow-based total degradation
soc_profile = [20, 40, 60, 40, 20]  # example SoC time series
cycles = rainflow_counting(soc_profile, step_duration=0.5)
total_deg, n_cycles = dynamic_degradation(soc_profile, step_duration=0.5)

# --- Real-World BESS model (Kampker et al. 2025) ---
rw_model = RealWorldBESSDegradationModel(chemistry='LFP')

# Calendar aging: 30-minute idle step at 35°C, 80% SOC
cal_loss = rw_model.calendar_aging_per_step(T_celsius=35.0, soc_frac=0.8, dt_hours=0.5)

# Cycle aging: one cycle at 80% DoD, 0.5C, 25°C
cyc_loss = rw_model.cycle_aging_per_cycle(T_celsius=25.0, dod_pct=80.0, c_rate=0.5)

# Inspect model parameters
print(rw_model.describe())
```

> **Note:** Former top-level helper names (e.g. `nCL_Id`) are now instance methods on `DegradationModel` (e.g. `model.nCL_Id(...)`). Update tests or call sites accordingly.

### Integration with Environment

**Household environment** (`SolarBatteryEnv`): Creates and uses a `DegradationModel` by default (per-step degradation costs via `info['deg_cost']`).

**AEMO environment** (`AEMOBatteryTradingEnv`): Supports three degradation modes:

```python
# Real-world BESS (recommended for grid-scale — includes calendar aging)
env = AEMOBatteryTradingEnv(
    aemo_data=data,
    degradation_mode='real_world',
    degradation_chemistry='LFP',
    degradation_temperature=30.0,
)

# Muenzel et al. rainflow (cycle aging only)
env = AEMOBatteryTradingEnv(aemo_data=data, degradation_mode='rainflow')

# Simple linear (backward compatible)
env = AEMOBatteryTradingEnv(aemo_data=data, degradation_mode='simple')
```


---

## 8. Data Transformation (`helper.py`)

### Overview

Provides utilities for transforming raw energy data into the format required by `SolarBatteryEnv`, plus evaluation and plotting functions. The evaluation helpers are environment-agnostic and also support AEMO trading logs by auto-detecting info keys such as revenue, degradation cost, dispatch energy, and SOC. See [docs/HELPER_README.md](docs/HELPER_README.md) for full details.

### Key Functions

```python
from src.helper import transform_polars_df, make_env, flatten_episode_data

# Transform raw Ausgrid data to environment format
dataset = transform_polars_df(
    customer_df,
    import_energy_price=0.23,
    export_energy_price=0.015,
    price_periods="7am-10am | 4pm-9pm",
    default_import_energy_price=0.15,
    default_export_energy_price=0.01,
)

# Create environment factory function
env_fn = make_env(dataset)
env = env_fn()  # Creates SolarBatteryEnv instance

# Flatten episode data for analysis
flattened = flatten_episode_data(episode_logs)
flattened.write_parquet("data/episode_logs.parquet")
```

### Evaluation Functions

```python
from src.helper import evaluate_experiment_logs, evaluate_experiments, evaluate_by_conditions

# Evaluate single experiment
ppo_logs = [
    pl.read_parquet("data/ppo_test_episode_01_logs.parquet"),
    pl.read_parquet("data/ppo_test_episode_02_logs.parquet"),
]
metrics = evaluate_experiment_logs(ppo_logs, target_return=0.0)
print(metrics)
# Includes: mean_reward, std_reward, sharpe_ratio, sortino_ratio,
#           var_5, cvar_5, recommended_rtg, avg_grid_import, etc.

# Compare multiple experiments
comparison = evaluate_experiments(
    {
        "rule": rule_logs,
        "ppo": ppo_logs,
        "sdp": sdp_logs,
    },
    target_return=0.0,
    save_dir="eval_output/figures",
    save_format="png",
)

# Optional: evaluate reward under specific conditions
conditions = {
    "high_price": lambda obs, info: info.get("energy_price", 0.0) > 200.0,
    "low_soc": lambda obs: obs[-2] < 0.2,
}
conditional = evaluate_by_conditions(ppo_logs, conditions)
print(conditional)
```

### Risk Metrics and Statistical Comparisons

```python
from src.helper import bootstrap_confidence_intervals, paired_comparison

# Bootstrap 95% confidence intervals on mean reward
cis = bootstrap_confidence_intervals(
    {"rule": rule_logs, "ppo": ppo_logs},
    n_bootstrap=1000, confidence_level=0.95, seed=42,
)
for algo, ci in cis.items():
    print(f"{algo}: mean={ci['mean']:.2f}  95% CI=[{ci['ci_lower']:.2f}, {ci['ci_upper']:.2f}]")

# Paired Wilcoxon signed-rank test between two algorithms
result = paired_comparison(ppo_logs, rule_logs)
print(f"mean diff = {result['mean_diff']:.4f}, p = {result['wilcoxon_p']:.4f}")
```

---

## 9. Decision Transformer (`decision_transformer.py`)

### Overview

Implements the Decision Transformer architecture for offline reinforcement learning. Uses transformer self-attention to model sequences of (return, state, action) tuples.

### Model Architecture

```python
from src.decision_transformer import DecisionTransformer

model = DecisionTransformer(
    state_dim=10,       # Observation space dimension
    act_dim=1,          # Action space dimension
    n_block=4,          # Number of transformer blocks
    h_dim=128,          # Hidden dimension
    context_len=60,     # Context length (history window)
    n_heads=8,          # Attention heads
    drop_p=0.1,         # Dropout probability
    max_timestep=10000, # Maximum timestep for embeddings
)
```

### Usage in Agent

```python
from src.decision import Agent

# Load pre-trained model
model = DecisionTransformer(...)
model.load_state_dict(torch.load("models/dt_model.pt"))

# Create agent with Decision Transformer
agent = Agent(
    env,
    algorithm="dt",
    dt_model=model,
    dt_context_len=60,
)

# Run episode
episode_log = agent.run_episode()
```

---

## 10. Transformer Training (`transformer_training.py`)

### Overview

Provides training utilities for the Decision Transformer, including dataset handling, training loops with mixed precision, and checkpoint management.

### TrajectoryDataset

```python
from src.transformer_training import TrajectoryDataset

# Create dataset from logged trajectories
train_ds = TrajectoryDataset(
    data_path="data/rule_train_episode_01_logs.parquet",
    context_length=36,
    state_dim=10,
    act_dim=1,
    discount_factor=0.99,
)
```

### Training Function

```python
from src.transformer_training import train_decision_transformer
from src.decision_transformer import DecisionTransformer

# Create model
model = DecisionTransformer(
    state_dim=10,
    act_dim=1,
    n_block=2,
    h_dim=128,
    context_len=36,
    n_heads=8,
    drop_p=0.1,
)

# Train
trained_model, train_losses, val_losses = train_decision_transformer(
    ds=train_ds,
    model=model,
    batch_size=32,
    lr=1e-4,
    epochs=5,
    device="cuda",
    save_path="models/dt_model.pt",
    checkpoint_path="models/dt_checkpoint.pt",
)
```

### Command-Line Training

```bash
python src/train_decision_transformer.py \
    --data-dir ./data \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 8 \
    --context-length 60 \
    --lr 5e-6 \
    --weight-decay 1e-4 \
    --checkpoint-path ./models/dt_model_checkpoint.pt \
    --save-path ./models/dt_model.pt \
    --loss-csv-path ./models/dt_model_loss_history.csv \
    --rope-enabled \
    --amp-mode "auto" \
    --num-workers 6 \
    --prefetch-factor 2
```

Notes:

- `dt_model_loss_history.csv` stores epoch-level train/val totals and component losses.
- `dt_model_loss_history_checkpoints.csv` stores per-checkpoint/segment snapshots for within-epoch progress.
- Persistent DataLoader workers are enabled by default; pass `--no-persistent-workers` to disable.

---

## 11. Stable-Baselines3 Training (`sb3train.py`)

### Overview

Utilities for training online RL agents using Stable-Baselines3, with optional Optuna hyperparameter tuning.

### Basic Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.sb3train import train_model
from src.helper import make_env

# Create vectorized environment
env_fns = [make_env(ds) for ds in training_datasets]
vec_env = DummyVecEnv(env_fns)

# Train PPO model
model, eval_summary = train_model(
    model_class=PPO,
    vec_env=vec_env,
    eval_env_fn=testing_env_fns[0],
    total_timesteps=400_000,
    default_model=True,
)

# Save model
model.save("models/ppo_model")
```

### Generate Trajectories for Offline RL

```python
from src.decision import run_sb3_model_on_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create test environment
test_vec_env = SubprocVecEnv(testing_env_fns)

# Run model and collect trajectories
episode_data = run_sb3_model_on_vec_env(model, test_vec_env, deterministic=True)

# Save for offline training
trajectories = flatten_episode_data(episode_data)
trajectories.write_parquet("data/ppo_test_episode_logs.parquet")
```

---

## 12. Dispatch Replay Utilities (`dispatch_utils.py`)

### Overview

`dispatch_utils` provides high-level helpers for **dispatch replay** simulations
using real AEMO DISPATCHLOAD data.  It bridges the raw data-fetching functions in
`aemo_data` with `AEMOBatteryTradingEnv` and `AEMOAgent` so notebooks stay clean
and readable.

For a full description of the module, see
**[docs/AEMO_DISPATCH_UTILS.md](docs/AEMO_DISPATCH_UTILS.md)**.

### Key Functions

| Function | Description |
|----------|-------------|
| `show_dispatch_table` | Pretty-print a column-filtered DataFrame view |
| `list_dispatch_candidates` | List all battery DUIDs in a region and the dispatch-active subset |
| `resolve_dispatch_selection` | Select a DUID and resolve env-sizing / paired DUID metadata |
| `run_dispatch_replay` | Run N dispatch replay episodes and save logs to parquet |
| `scan_duid_availability` | Cross-region scan of battery DUID activity for a date window |

### Basic Usage

```python
from dispatch_utils import list_dispatch_candidates, resolve_dispatch_selection, run_dispatch_replay

# Step 1 – discover which batteries were dispatched
battery_units, active_units = list_dispatch_candidates(
    region="SA1",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 7),
)

# Step 2 – select a DUID
selection = resolve_dispatch_selection(
    battery_units=battery_units,
    active_battery_units=active_units,
    selected_duid="HPRG1",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 7),
)

# Step 3 – run replay episodes
ep_logs, inc_logs, all_logs = run_dispatch_replay(
    processed_data=processed_data,
    selection=selection,
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 1, 7),
    region="SA1",
    num_episodes=3,
    output_dir="data/aemo_sim_output",
    run_tag="aemo_mm",
)
```

### Notebooks

- `notebooks/aemo_simrun.ipynb` — full dispatch replay workflow inside the notebook-based AEMO offline RL pipeline
- `notebooks/test_aemo_data.ipynb` section 5 — DUID availability exploration

---

## 13. Performance Optimizations

### Summary of Implemented Optimizations

| Component | Optimization | Impact |
|-----------|-------------|--------|
| `EnergySimEnv.py` | Batch min/max queries (2N→1 scan) | Medium |
| `EnergySimEnv.py` | Vectorized observation normalization | High (hot path) |
| `quantile_scenarios.py` | Batch quantile computation (4×Q→1 query) | Medium |
| `decision.py` | Remove redundant cache checks | Low |
| `helper.py` | Batch `with_columns` calls | Low |
| `batterydeg.py` | Pre-compute constant denominators | Low |

### Running Performance Tests

```bash
# Run all performance tests
pytest tests/test_performance.py -v -s

# Run specific benchmark
pytest tests/test_performance.py::TestEnvironmentPerformance -v -s
```

### Example Benchmark Output

```
Environment observation benchmark (1000 calls):
  Total time: 0.0380s
  Per-call time: 0.0380ms

Degradation calculation benchmark (10000 calls):
  Total time: 0.0316s
  Per-call time: 0.0032ms

Quantile scenario generation benchmark (1000 rows, 5 scenarios):
  Total time: 0.0006s
```

---

## Testing

All components are tested in the `tests/` directory:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v
pytest tests/test_decision_agent.py -v
pytest tests/test_quantile_scenarios.py -v
pytest tests/test_performance.py -v -s

# Run with timing info
pytest tests/ -v --durations=10
```

### Test Categories

| Test File | Purpose | Test Count |
|-----------|---------|------------|
| `test_environment.py` | SolarBatteryEnv functionality | 8 |
| `test_decision_agent.py` | Agent/SDP/Oracle tests | 8 |
| `test_performance.py` | Performance benchmarks | 6 |
| `test_quantile_scenarios.py` | Scenario generation | 22 |
| `test_aemo_degradation.py` | Rainflow counter, capacity fade | 12 |
| `test_real_world_degradation.py` | RealWorldBESSDegradationModel, AEMO env integration | 28 |
| `test_episode_visualizer.py` | Env detection, plotting, edge cases | 16 |
| `test_algorithm_classes.py` | SDP/MRDP/Oracle class init | 5 |
| `test_aemo_env_compatibility.py` | Gymnasium/SB3 compatibility | 5 |
| `test_risk_statistics.py` | CVaR/VaR, bootstrap CIs, paired tests | 22 |

**Total: 132 tests**

---

## Quick Reference

### File Summary

| File | Description |
|------|-------------|
| `EnergySimEnv.py` | Gymnasium environment for solar-battery-grid simulation |
| `decision.py` | Agent class with rule-based, SDP, Oracle, RL, and DT algorithms |
| `mrdp_algorithm.py` | Multi-resolution dynamic programming solver |
| `quantile_scenarios.py` | Scenario generation for uncertainty modeling |
| `batterydeg.py` | Battery degradation models |
| `helper.py` | Data transformation and evaluation utilities |
| `decision_transformer.py` | Decision Transformer model architecture |
| `transformer_training.py` | Training utilities for Decision Transformer |
| `sb3train.py` | Stable-Baselines3 training utilities |
| `train_decision_transformer.py` | CLI for Decision Transformer training |

### Import Patterns

```python
# Environment
from src.EnergySimEnv import SolarBatteryEnv

# Agent
from src.decision import Agent, run_episodes_parallel

# MRDP
from src.mrdp_algorithm import MRDPSolver

# Scenarios
from src.quantile_scenarios import QuantileScenarioGenerator

# Data
from src.helper import transform_polars_df, make_env

# Models
from src.decision_transformer import DecisionTransformer
from src.transformer_training import TrajectoryDataset, train_decision_transformer
```
