# Energy Decision: Solar-Battery Control Benchmark

## Overview
This project establishes a comprehensive, reproducible benchmark for residential and grid-scale energy storage control. It integrates high-fidelity Gymnasium environments, diverse baselines (Rule-based, SDP, Online RL, Offline RL), and a standardized evaluation suite.

**Goal:** To minimize energy costs and maximize revenue while rigorously accounting for battery degradation under realistic uncertainty.

## Key Components

1.  **Simulation Environments:**
    *   **Household:** [SolarBatteryEnv](docs/HOUSEHOLD_ENV_README.md) - Residential PV + Battery with ToU tariffs.
    *   **Grid:** [AEMOBatteryTradingEnv](docs/AEMO_ENV_README.md) - Arbitrage & FCAS in the Australian National Electricity Market.

2.  **Algorithms ([COMPONENTS.md](COMPONENTS.md)):**
    *   **Optimization:** Stochastic Dynamic Programming (SDP) & Multi-Resolution DP (MRDP).
    *   **Online RL:** PPO, SAC, A2C, DDPG, TD3 (via Stable-Baselines3).
    *   **Offline RL:** Decision Transformers (DT).
    *   **Baselines:** Rule-based heuristics & Oracle (perfect foresight).

## Status
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)]()

### Roadmap
*   [x] **Core:** Gymnasium environment & Rule-based agents.
*   [x] **Optimization:** SDP & MRDP solvers.
*   [x] **Online RL:** Training loop with SB3.
*   [x] **Offline RL:** Decision Transformer training loop.
*   [x] **Evaluation:** Metrics for cost, risk, and degradation.
*   [x] **Grid Market:** AEMO Environment Implementation.
*   [ ] **Hyperparameter Tuning:** Optuna for DT.
*   [ ] **RL Fine-tuning:** Initialize Online RL with DT weights.

## Installation

### Option 1: Docker (Recommended)
Sets up a JupyterLab environment with all dependencies.

```bash
docker compose up
```
Access JupyterLab at `http://localhost:8888`.

### Option 2: Local Installation

```bash
git clone <repository-url>
cd energydecision
pip install -r requirements.txt
pip install -r torch_req.txt
```

## Data Setup

1.  **Household Data:** Download **Ausgrid Solar Home Electricity Data** (July 2010 - June 2013) and place in `data/`.
2.  **AEMO Data:** Automatically fetched via `src/aemo_data.py` (cached in `data/aemo/`).

## Usage Workflow

The typical workflow moves from simulation to training and finally evaluation.

### 1. Simulation & Data Collection
Run [test_simrun.ipynb](notebooks/test_simrun.ipynb) to:
- Execute Rule-based and SDP agents.
- Generate interaction logs (`.parquet`) for offline training.

### 2. Online RL Training
Run [test_sb3train.ipynb](notebooks/test_sb3train.ipynb) to:
- Train PPO/SAC agents.
- Save models and log additional trajectories.

### 3. Offline RL Training
Train a Decision Transformer using the collected logs.

#### 3.1 Train from scratch

This starts a new model with a context length of 60 and saves checkpoints in `models/`:

```bash
docker exec -it test_energy_container /bin/bash

python3 pretrain_decision_transformer.py \
    --data-dir ../data \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 6 \
    --checkpoints-per-epoch 4 \
    --context-length 60 \
    --lr 5e-6 \
    --weight-decay 1e-4 \
    --return-scale 1000.0 \
    --return-loss-weight 0.0005 \
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
- **Ensure your `return_scale` matches the typical magnitude of returns; very large returns can cause instability.

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

### 4. Evaluation
Run [test_eval.ipynb](notebooks/test_eval.ipynb) to:
- Compare all agents (Cost, ROI, Degradation).
- Generate Risk-Return plots.

## Project Structure

```
energydecision/
├── COMPONENTS.md            # Usage guide for scripts
├── docs/                    # Deep dive documentation
│   ├── HOUSEHOLD_ENV_README.md
│   ├── AEMO_ENV_README.md
│   └── ALGORITHM_GUIDE.md
├── notebooks/               # Example notebooks
├── src/                     # Source code
│   ├── EnergySimEnv.py      # Household Gym Environment
│   ├── AEMOBatteryEnv.py    # AEMO Market Environment
│   ├── decision.py          # Agent Classes (Rule, RL, SDP)
│   ├── batterydeg.py        # Degradation Models
│   └── ...
└── tests/                   # Pytest suite
```

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

>>>>>>> main
## Documentation
*   **[COMPONENTS.md](COMPONENTS.md)**: Detailed usage guide for key scripts (`decision.py`, `batterydeg.py`, etc.).
*   **[Household Environment](docs/HOUSEHOLD_ENV_README.md)**: Physics, Reward Function, and Observation Space.
*   **[AEMO Environment](docs/AEMO_ENV_README.md)**: Market dynamics, FCAS, and data pipeline.
