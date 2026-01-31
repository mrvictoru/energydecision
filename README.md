# Energy Decision: Solar-Battery Control Benchmark

This project provides a comprehensive framework for benchmarking control algorithms in residential solar-battery systems. It integrates a high-fidelity Gymnasium environment, diverse baselines (Rule-based, SDP etc), Online Reinforcement Learning (SB3), and Offline Reinforcement Learning (Decision Transformer). The goal is to minimize energy costs while considering battery degradation.

## Status

[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)]()

## ToDo
*   ~~**Improve SDP algo:** Improve computation speed and run algo in different envs in parallel~~
*   ~~**Online learning loop:** Training loop using stablebaselines3~~
*   ~~**Examine the effectiveness of sb3 trained RL model:** Check and find out if the RL model actually output valid actions~~
*   ~~**Offline learning loop:** Collecting interaction dataset with various algorithms and use it to train a Decision Transformer based control algorithm~~
*   ~~**Plot the simulation:** modify render function from env to plot key metrics~~
*   ~~**Optimize training loop:** Added mixed precision training, gradient clipping and LR scheduler to the training loop for Decision Transformer~~
*   ~~**Performance optimizations:** Batch queries, vectorize hot paths, precompute constants~~
*   ~~**Test suite reorganization:** Consolidated tests into organized `tests/` directory~~
*   ~~**Documentation consolidation:** Unified component documentation in COMPONENTS.md~~
*   ~~**Refactor Agent class:** Refactor Agent class to be less spaghetti~~
*   ~~**Conduct evaluation:** To build framework that can evaluate the effectiveness of different algorithm/parameter~~
*   **Implement Grid like environment:** To build data pipeline and environment that can also simulate Grid market operation
*   **Hyperparameter tuning for DT:** Use Optuna or other hyperparameter tuning library to find the best hyperparameters for Decision Transformer model
*   **RL fine-tuning with DT:** Use the Decision Transformer model to initialize an online RL agent and fine-tune it in the environment, with GRPO or other RL algorithms

## Features

*   **Gymnasium Environment:** [`src/EnergySimEnv.py`](src/EnergySimEnv.py) simulates a household with solar PV, battery storage, and grid connection. It features realistic constraints, time-of-use tariffs, and degradation-aware rewards. The return observation is normalized against the dataset so it is suitable for reinforcement learning methods.

*   **AEMO Market Environment:** [`src/AEMOBatteryEnv.py`](src/AEMOBatteryEnv.py) provides a grid-scale battery trading environment using real AEMO market data (energy + FCAS). It includes a Polars-based preprocessor for resampling, feature engineering, and normalization.

*   **AEMO Data Pipeline:** [`src/aemo_data.py`](src/aemo_data.py) fetches actual AEMO data via NEMOSIS (dispatch prices, FCAS prices, generation mix, and unit dispatch) and returns Polars DataFrames with caching support.

*   **Algorithmic Baselines:** Implements and compares several control strategies within the [`Agent`](src/decision.py) class:
    *   **Rule-Based:** Heuristic controller with safety constraints.
    *   **Optimization:** Stochastic Dynamic Programming (SDP) and Multi-Resolution Dynamic Programming (MRDP) for theoretical optimality.
    *   **Online RL:** PPO, SAC, A2C, DDPG, TD3 via Stable-Baselines3.
    *   **Offline RL:** Decision Transformer (DT) trained on mixed behavioral logs.

    Brief implementation notes: the optimization solvers are implemented as self-contained classes to make the algorithm flow easy to follow—`SDPSolver` (`src/sdp_algorithm.py`), `MRDPSolver` (`src/mrdp_algorithm.py`), and `OracleSolver` (`src/oracle_algorithm.py`). Use `Agent(env, algorithm='sdp'|'mrdp'|'oracle')` to run the solver of your choice. See `COMPONENTS.md` for short examples and `ALGORITHM_GUIDE.md` for a step-by-step reading guide (kept as internal reference).

*   **Battery Degradation:** Detailed semi-empirical models (Rainflow counting, throughput, C-rate) in [`src/batterydeg.py`](src/batterydeg.py).

*   **Quantile Scenarios:** Scenario generation for uncertainty modeling in [`src/quantile_scenarios.py`](src/quantile_scenarios.py).

*   **Evaluation Suite:** Unified metrics for cost, revenue, degradation, and risk (Sharpe/Sortino ratios).

*   **Decision Transformer Training:** Implements a training loop for the Decision Transformer model using offline interaction data ([`src/transformer_training.py`](src/transformer_training.py)).

## Installation

### Option 1: Docker (Recommended)*

The easiest way to run the project is via Docker, which sets up a JupyterLab environment with all dependencies.*

```bash
sudo docker compose up
```

Access JupyterLab at `http://localhost:8888`.*


### Option 2: Local Installation

Requires Python 3.10+.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd energydecision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r torch_req.txt
```


## Project Structure

```
energydecision/
├── ALGORITHM_GUIDE.md       # Algorithm guide and internal references
├── COMPONENTS.md            # Comprehensive component documentation
├── report.md                # Project report / notes
├── data/                    # Datasets and generated parquet logs
│   ├── *.csv                # Solar home electricity data, household data, customer splits
│   ├── *.parquet            # Episode logs for different algorithms
│   └── *.pdf                # Reference papers
├── eval_output/             # Evaluation results and figures
├── models/                  # Trained models and checkpoints
│   ├── *.zip                # RL agent models
│   ├── *.pt                 # Decision Transformer checkpoints
│   └── *.json               # Model configs
├── src/                     # Source code
│   ├── EnergySimEnv.py              # Gymnasium environment for solar-battery-grid simulation
│   ├── AEMOBatteryEnv.py            # AEMO market trading environment (energy + FCAS)
│   ├── aemo_data.py                 # AEMO data ingestion (NEMOSIS) + caching
│   ├── decision.py                  # Agent class: rule-based, RL, DT, and SDP controllers
│   ├── batterydeg.py                # Battery degradation models (static and dynamic)
│   ├── helper.py                    # Data transformation, preparation, and evaluation utilities
│   ├── decision_transformer.py      # Core Decision Transformer model class
│   ├── transformer_training.py      # TrajectoryDataset and training helpers for DT
│   ├── pretrain_decision_transformer.py  # CLI to pretrain Decision Transformer
│   ├── sb3train.py                  # RL training utilities (Stable-Baselines3)
│   ├── quantile_scenarios.py        # Quantile scenario generation for uncertainty modeling
│   ├── mrdp_algorithm.py            # Multi-resolution dynamic programming (MRDP) implementation
│   ├── sdp_algorithm.py             # Stochastic Dynamic Programming implementation
│   ├── oracle_algorithm.py          # Oracle solver for benchmarking
│   └── algorithm_helpers.py         # Helper utilities for algorithm implementations
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_environment.py      # SolarBatteryEnv tests
│   ├── test_decision_agent.py   # Agent/SDP/Oracle tests
│   ├── test_performance.py      # Performance benchmarks
│   ├── test_quantile_scenarios.py   # Quantile scenario tests
│   ├── test_algorithm_classes.py
│   └── test_refactoring.py
├── notebooks/               # Useful example notebooks
│   ├── DemoEnv.ipynb
│   ├── Demosb3.ipynb
│   ├── test_aemo_data.ipynb         # AEMO data ingestion and exploration
│   ├── test_aemo_env.ipynb          # AEMO trading environment smoke tests
│   ├── test_simrun.ipynb
│   ├── test_sb3train.ipynb
│   ├── test_eval.ipynb
│   ├── test_grid_sim.ipynb
│   └── testrun.ipynb
├── requirements.txt         # Python package requirements
├── torch_req.txt            # PyTorch-specific requirements
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Dockerfile for building the environment
└── README.md                # Project documentation (this file)
```

---

## Data Setup

1. Download the **Ausgrid Solar Home Electricity Data** (July 2010 - June 2013).
2. Place the CSV files in the `data/` directory:
    - `data/2010-2011 Solar home electricity data.csv`
    - `data/2011-2012 Solar home electricity data v2.csv`
    - `data/2012-2013 Solar home electricity data v2.csv`

### AEMO Data (Optional)

The AEMO pipeline downloads data on demand via NEMOSIS and caches files under `data/aemo/` by default. In some environments AEMO blocks static table downloads; if that happens, provide a local generator info file (XLS/XLSX/CSV) via:

- Environment variable: `AEMO_GENERATORS_FILE=/path/to/your/file.xlsx`
- Or place the file in `src/data/aemo/` or `data/aemo/` (auto-detected)

---

## Usage Workflow

The project workflow is divided into four main stages: Simulation/Baselines, Training, Offline RL, and Evaluation.

### 1. Simulation & Baselines (`test_simrun.ipynb`)

- Load and preprocess customer data.
- Run **Rule-based**, **SDP**, and **MRDP** agents.
- Generate interaction logs (`.parquet` files) for offline training.
- Test trained Decision Transformer models.

### 2. Online RL Training (`test_sb3train.ipynb`)

- Train Online RL agents (PPO, SAC, A2C, DDPG, TD3) using Stable-Baselines3.
- Save trained models to `models/`.
- Generate interaction logs from these agents to diversify the offline training dataset.

### 3. Offline RL: Decision Transformer

**Training:**  
Train the Decision Transformer using the logs generated in steps 1 & 2.

#### 3.1 Train from scratch

This starts a new model with a context length of 60 and saves checkpoints in `models/`:

```bash
docker exec -it test_energy_container /bin/bash

python3 pretrain_decision_transformer.py \
    --data-dir ../data \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 8 \
    --checkpoints-per-epoch 4 \
    --context-length 60 \
    --lr 1e-5 \
    --weight-decay 1e-4 \
    --checkpoint-path ../models/dt_model_checkpoint.pt \
    --save-path ../models/dt_model_new.pt \
    --loss-csv-path ../models/dt_model_loss_history.csv \
    --rope-enabled \
    --amp-mode "auto" \
    --num-workers 2 \
    --prefetch-factor 1 \
    --no-persistent-workers
```

Notes:

- **DataLoader throughput tuning**:
    - `--num-workers` controls how many worker processes load/pad batches.
    - `--prefetch-factor` controls how many batches each worker preloads (only applies when `--num-workers > 0`).
    - Persistent workers are **enabled by default**; pass `--no-persistent-workers` if you want to disable them.
- **Loss logging files** (values are consistent between what prints and what’s written):
    - `--loss-csv-path .../dt_model_loss_history.csv` stores **epoch-level** totals + components (train/val).
    - A second file is also written next to it: `dt_model_loss_history_checkpoints.csv`, which stores **per-checkpoint/segment** snapshots (useful for plotting progress during an epoch).
- **Best model weights** are saved alongside your `--save-path` as `*_best.pt` when validation improves without obvious divergence.

#### 3.2 Resume from an existing checkpoint

If you already have a compatible checkpoint (same model config, especially `context_len`), you can resume:

```bash
python3 pretrain_decision_transformer.py \
    --data-dir ../data \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 8 \
    --checkpoints-per-epoch 10 \
    --context-length 60 \
    --checkpoint-path ../models/dt_model_checkpoint.pt \
    --save-path ../models/dt_model.pt \
    --resume \
    --num-workers 2 \
    --prefetch-factor 1
```

Notes:

- `--epochs` is the **total** target epoch count; resuming will continue from the last saved epoch.
- `--context-length` must match the value used to create the checkpoint, otherwise `load_state_dict` will fail (e.g. attention mask shape mismatch).

#### 3.3 Start fresh if the checkpoint is incompatible

If your previous run used a different `context_length` (or other model config) and you just want to restart training:

```bash
rm -f models/dt_model_checkpoint.pt

python3 pretrain_decision_transformer.py \
    --data-dir ./data \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 6 \
    --context-length 60 \
    --checkpoint-path ../models/dt_model_checkpoint.pt \
    --save-path ./models/dt_model.pt
```

This removes the stale checkpoint so automatic recovery and `--resume` logic do not try to load an incompatible state.

#### 3.4 Stabilizing training when encountering non‑finite weights

If you see `NonFiniteParameterError` in the logs:

- Reduce the learning rate, e.g.:

```bash
python3 pretrain_decision_transformer.py \
    --data-dir ./data \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 6 \
    --context-length 60 \
    --lr 1e-6 \
    --checkpoint-path ../models/dt_model_checkpoint.pt \
    --save-path ../models/dt_model.pt
```

- Ensure your `return_scale` matches the typical magnitude of returns; very large returns can cause instability.

**Inference:**  
Load the trained model in `test_simrun.ipynb` to evaluate its performance.

### 4. Evaluation (`test_eval.ipynb`)

- Load logs from all algorithms.
- Compute aggregate metrics (Profit, ROI, Degradation).
- Generate comparative plots (Risk-Return, Cost Breakdown).
- Perform temporal analysis of agent behavior.

### AEMO Market Workflow (AEMO notebooks)

- [`test_aemo_data.ipynb`](test_aemo_data.ipynb): fetch and explore AEMO dispatch prices, FCAS prices, generation mix, and unit dispatch.
- [`test_aemo_env.ipynb`](test_aemo_env.ipynb): create and run `AEMOBatteryTradingEnv` episodes using the Polars-based preprocessor.

---

## Usage

- Explore the simulation and agent interactions in the [`testrun.ipynb`](testrun.ipynb) notebook.
- See demo notebooks [`DemoEnv.ipynb`](DemoEnv.ipynb) and [`Demosb3.ipynb`](Demosb3.ipynb) for example usage of the gym and stable-baselines3 library.
- **Using the environment class from code:** Instantiate [`SolarBatteryEnv`](src/EnergySimEnv.py) and [`Agent`](src/decision.py) directly to run a single episode and capture step-level logs.

```python
import polars as pl
from src.helper import transform_polars_df
from src.EnergySimEnv import SolarBatteryEnv
from src.decision import Agent

# Example usage
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
env = SolarBatteryEnv(dataset)
agent = Agent(env, algorithm="rule")
episode_log = agent.run_episode()
print(episode_log.head())
```

*   **Running multiple environments in parallel:** [`run_episodes_parallel`](src/decision.py) can execute one episode per environment for rule/SDP/MRDP/DT agents.

    ```python
    import numpy as np
    import polars as pl
    from src.helper import make_env, transform_polars_df
    from src.decision import Agent, run_episodes_parallel

    df = pl.read_csv("data/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
    customers = df["Customer"].unique()
    rng = np.random.default_rng(seed=42)
    sample_ids = rng.choice(customers, size=4, replace=False)

    datasets = []
    for cid in sample_ids:
        customer_df = df.filter(pl.col("Customer") == cid)
        datasets.append(
            transform_polars_df(
                customer_df,
                import_energy_price=0.23,
                export_energy_price=0.015,
                price_periods="7am-10am | 4pm-9pm",
                default_import_energy_price=0.15,
                default_export_energy_price=0.01,
            )
        )

    envs = [creator() for creator in (make_env(ds) for ds in datasets)]
    agent_kwargs = {
        "algorithm": "sdp",
        "soc_resolution": 21,
        "action_resolution": 41,
        # SDP uses rainflow-based degradation internally; no need to pass degradation_model
    }

    episode_logs = run_episodes_parallel(
        Agent,
        envs,
        agent_kwargs=agent_kwargs,
        max_workers=4,
    )
    print(len(episode_logs))  # number of completed episodes
    ```

*   **Training Stable-Baselines3 policies and logging rollouts:** [`train_model`](src/sb3train.py) wraps Optuna tuning (optional) and SB3 training; [`run_sb3_model_on_vec_env`](src/decision.py) records trajectories for evaluation.

```python
    import numpy as np
    import polars as pl
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from src.helper import make_env, transform_polars_df, flatten_episode_data
    from src.sb3train import train_model
    from src.decision import run_sb3_model_on_vec_env

    df = pl.read_csv("data/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
    customers = df["Customer"].unique()
    rng = np.random.default_rng(seed=0)
    train_ids = rng.choice(customers, size=int(0.8 * len(customers)), replace=False)
    test_ids = np.setdiff1d(customers, train_ids)

    def build_datasets(ids):
        out = []
        for cid in ids:
            customer_df = df.filter(pl.col("Customer") == cid)
            out.append(
                transform_polars_df(
                    customer_df,
                    import_energy_price=0.23,
                    export_energy_price=0.015,
                    price_periods="7am-10am | 4pm-9pm",
                    default_import_energy_price=0.15,
                    default_export_energy_price=0.01,
                )
            )
        return out

    training_datasets = build_datasets(train_ids)
    testing_datasets = build_datasets(test_ids)

    training_env_fns = [make_env(ds) for ds in training_datasets]
    training_vec_env = DummyVecEnv(training_env_fns)
    testing_env_fns = [make_env(ds) for ds in testing_datasets]

    ppo_model, eval_summary = train_model(
        model_class=PPO,
        vec_env=training_vec_env,
        eval_env_fn=testing_env_fns[0],
        total_timesteps=400_000,
        default_model=True,
    )

    test_vec_env = SubprocVecEnv(testing_env_fns)
    ppo_episode_data = run_sb3_model_on_vec_env(ppo_model, test_vec_env, deterministic=True)

    trajectories = flatten_episode_data(ppo_episode_data)
    trajectories.write_parquet("data/ppo_test_episode_logs.parquet")
```

*   **Training the Decision Transformer with offline interaction data:** [`train_decision_transformer`](src/transformer_training.py) consumes a [`TrajectoryDataset`](src/transformer_training.py) built from logged trajectories.

    ```python
    import torch
    import polars as pl
    from src.helper import transform_polars_df
    from src.EnergySimEnv import SolarBatteryEnv
    from src.transformer_training import TrajectoryDataset, train_decision_transformer
    from src.decision_transformer import DecisionTransformer

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

    env = SolarBatteryEnv(dataset)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    train_ds = TrajectoryDataset(
        data_path="data/rule_train_episode_01_logs.parquet",
        context_length=36,
        state_dim=state_dim,
        act_dim=action_dim,
        discount_factor=0.99,
    )

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        n_block=2,
        h_dim=128,
        context_len=36,
        n_heads=8,
        drop_p=0.1,
        max_timestep=len(env.df),
    )

    trained_model, train_losses, val_losses = train_decision_transformer(
        ds=train_ds,
        model=model,
        batch_size=32,
        lr=1e-4,
        epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="models/dt_model.pt",
        checkpoint_path="models/dt_checkpoint.pt",
    )
    ```

*   **Evaluating model performance from interaction data:** Use [`evaluate_experiment_logs`](src/helper.py) for single experiments or [`evaluate_experiments`](src/helper.py) for comparisons. Figures can be saved by passing `save_dir`.

    ```python
    import polars as pl
    from src.helper import evaluate_experiment_logs, evaluate_experiments

    ppo_logs = [
        pl.read_parquet("data/ppo_test_episode_01_logs.parquet"),
        pl.read_parquet("data/ppo_test_episode_02_logs.parquet"),
    ]
    rule_logs = [
        pl.read_parquet("data/rule_test_episode_01_logs.parquet"),
        pl.read_parquet("data/rule_test_episode_02_logs.parquet"),
    ]

    baseline_metrics = evaluate_experiment_logs(rule_logs, target_return=0.0)
    print(baseline_metrics)

    comparison = evaluate_experiments(
        {
            "rule": rule_logs,
            "ppo": ppo_logs,
        },
        target_return=0.0,
        save_dir="eval_output/figures",
        save_format="png",
    )
    print(comparison)
    ```

---

## Testing

The project includes a comprehensive test suite organized in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_environment.py -v

# Run performance benchmarks with output
pytest tests/test_performance.py -v -s

# Run with timing info
pytest tests/ -v --durations=10
```

### Test Categories

| Test File | Purpose | Test Count |
|-----------|---------|------------|
| `test_environment.py` | SolarBatteryEnv functionality, observation handling | 9 |
| `test_decision_agent.py` | SDP solver, Oracle agent, policy computation | 8 |
| `test_performance.py` | Performance benchmarks and optimization validation | 8 |
| `test_quantile_scenarios.py` | Quantile scenario generation | 21 |

**Total: 46 tests**

---

## Dependencies

*   Python 3.10+
*   Gymnasium
*   NumPy
*   Polars
*   PyTorch
*   Stable-Baselines3 (for RL agents)
*   pytest (for testing)

See `requirements.txt` and `torch_req.txt` for complete dependency lists.

---

## Documentation

For detailed documentation on all source components, see **[COMPONENTS.md](COMPONENTS.md)**, which includes:

- Environment setup and usage (`EnergySimEnv.py`)
- Decision agent algorithms (`decision.py`)
- Multi-Resolution Dynamic Programming (`mrdp_algorithm.py`) (legacy `sdp_multires.py` removed) 
- Scenario generation (`quantile_scenarios.py`)
- Battery degradation models (`batterydeg.py`)
- Data transformation utilities (`helper.py`)
- Decision Transformer training (`transformer_training.py`)
- Stable-Baselines3 training (`sb3train.py`)
- Performance optimization details
