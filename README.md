# Energy Decision: Solar-Battery Control Benchmark


This project provides a comprehensive framework for benchmarking control algorithms in residential solar-battery systems. It integrates a high-fidelity Gymnasium environment, diverse baselines (Rule-based, SDP, MRDP), Online Reinforcement Learning (SB3), and Offline Reinforcement Learning (Decision Transformer).This project explores different algorithms for optimizing energy management in a solar-battery-grid system using a gymnasium environment to simulate system interaction and reward calculation. The goal is to minimize energy costs while considering battery degradation.

## ToDo
*   ~~**Improve SDP algo:** Improve computation speed and run algo in different envs in parallel~~
*   ~~**Online learning loop:** Training loop using stablebaselines3~~
*   ~~**Examine the effectiveness of sb3 trained RL model:** Check and find out if the RL model actually output valid actions~~
*   ~~**Offline learning loop:** Collecting interaction dataset with various algorithms and use it to train a Decision Transformer based control algorithm~~
*   ~~**Plot the simulation:** modify render function from env to plot key metrics~~
*   **Refactor Agent class:** Refactor Agent class to be less spaghetti
*   ~~**Optimize training loop:** Added mixed precision training, gradient clipping and LR scheduler to the training loop for Decision Transformer~~
*   **Conduct evaluation:** To build framework that can evaluate the effectiveness of different algorithm/parameter

## Features## Features



*   **Gymnasium Environment:** [`src/EnergySimEnv.py`](src/EnergySimEnv.py) simulates a household with solar PV, battery storage, and grid connection. It features realistic constraints, time-of-use tariffs, and degradation-aware rewards.*   **Simulation Environment:** A custom Gym environment ([`src/EnergySimEnv.py`](src/EnergySimEnv.py)) simulating a household with solar panels, a battery, and grid connection. The return observation is normalized against the dataset so it is sutiable with reinforcement learning method. Seperate method can be used to return raw value for observation.

*   **Algorithmic Baselines:***   **Control Algorithms:** Implements and compares several control strategies within the [`Agent`](src/decision.py) class in [`src/decision.py`](src/decision.py):

    *   **Rule-Based:** Heuristic controller with safety constraints.    *   Rule-Based Controller

    *   **Optimization:** Stochastic Dynamic Programming (SDP) and Multi-Resolution Dynamic Programming (MRDP) for theoretical optimality.    *   Reinforcement Learning (RL) agents (using pre-trained models like A2C, DDPG, PPO)

    *   **Online RL:** PPO, SAC, A2C, DDPG, TD3 via Stable-Baselines3.    *   Decision Transformer (DT)

    *   **Offline RL:** Decision Transformer (DT) trained on mixed behavioral logs.    *   Stochastic Dynamic Programming (SDP) with receding horizon optimization

*   **Battery Degradation:** Detailed semi-empirical models (Rainflow counting, throughput, C-rate) in [`src/batterydeg.py`](src/batterydeg.py).*   **Battery Degradation Modeling:** Includes models for battery degradation based on usage patterns ([`src/batterydeg.py`](src/batterydeg.py)), incorporating both static and dynamic (rainflow counting) approaches.

*   **Evaluation Suite:** Unified metrics for cost, revenue, degradation, and risk (Sharpe/Sortino ratios).*   **Sampling Environment with different agent to use for Offline RL training:** The [`run_episodes_parallel`](src/decision.py) function allows running multiple environments in parallel with different agents, collecting interaction data for offline training.

*   **Decision Transformer Training:** Implements a training loop for the Decision Transformer model using offline interaction data ([`src/transformer_training.py`](src/transformer_training.py)).

## Installation

### Option 1: Docker (Recommended)*

The easiest way to run the project is via Docker, which sets up a JupyterLab environment with all dependencies.*

```bash*   ~~**Offline learning loop:** Collecting interaction dataset with various algorithms and use it to train a Decision Transformer based control algorithm~~

docker compose up --build*   ~~**Plot the simulation:** modify render function from env to plot key metrics~~

```

Access JupyterLab at `http://localhost:8888`.*


### Option 2: Local Installation

Requires Python 3.10+.

## Project Structure

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

### 3. (Optional) Using Docker

Build and run the container, which will spin up a JupyterLab server with all dependencies installed:

```bash
sudo docker compose up
```

---

## Project Structure

```
energydecision/
├── data/                # Datasets and generated parquet logs
│   ├── *.csv            # Solar home electricity data, household data, customer splits
│   ├── *.parquet        # Episode logs for different algorithms
│   ├── *.pdf            # Reference papers
│   └── ...
├── eval_output/         # Evaluation results and figures
├── models/              # Trained models and checkpoints
│   ├── *.zip            # RL agent models
│   ├── *.pt             # Decision Transformer checkpoints
│   ├── *.json           # Model configs
│   └── ...
├── src/                 # Source code
│   ├── EnergySimEnv.py          # Gymnasium environment for solar-battery-grid simulation
│   ├── decision.py              # Agent class: rule-based, RL, DT, and SDP controllers
│   ├── batterydeg.py            # Battery degradation models (static and dynamic)
│   ├── helper.py                # Data transformation, preparation, and evaluation utilities
│   ├── decision_transformer.py  # Core Decision Transformer model class
│   ├── transformer_training.py  # TrajectoryDataset class and train_decision_transformer function
│   ├── sb3train.py              # RL training utilities (Stable-Baselines3)
│   ├── quantile_scenarios.py    # Quantile scenario generation
│   ├── run_sdp_parallel.py      # Parallel SDP simulation
│   ├── sdp_multires.py          # Multi-resolution SDP
│   ├── mrdp_integration_example.py # MRDP integration example
│   ├── test_mrdp_validation.py      # MRDP validation tests
│   ├── test_quantile_scenarios.py   # Quantile scenario tests
│   ├── test_reward_logic.py         # Reward logic tests
│   ├── test_sdp_perf.py             # SDP performance tests
│   ├── test_sdp_timing.py           # SDP timing tests
│   └── ...                     # Other modules/utilities
├── test_simrun.ipynb        # Main simulation notebook
├── test_sb3train.ipynb      # Online RL training notebook
├── test_eval.ipynb          # Evaluation notebook
├── requirements.txt         # Python package requirements
├── torch_req.txt            # PyTorch-specific requirements
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Dockerfile for building the environment
├── MRDP_README.md           # MRDP integration documentation
├── README.md                # Project documentation (this file)
├── README.scenario-support.md # Scenario support documentation
├── testrun.ipynb            # Example Jupyter notebook for running simulations
├── DemoEnv.ipynb            # Demo notebook for environment usage
├── Demosb3.ipynb            # Demo notebook for Stable-Baselines3 usage
└── ...
```

---

## Data Setup

1. Download the **Ausgrid Solar Home Electricity Data** (July 2010 - June 2013).
2. Place the CSV files in the `data/` directory:
    - `data/2010-2011 Solar home electricity data.csv`
    - `data/2011-2012 Solar home electricity data v2.csv`
    - `data/2012-2013 Solar home electricity data v2.csv`

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

```bash
python src/train_decision_transformer.py \
    --data-dir data \
    --patterns rule_train sdp_train ppo_train \
    --epochs 10 \
    --batch-size 64 \
    --context-length 48
```

**Inference:**  
Load the trained model in `test_simrun.ipynb` to evaluate its performance.

### 4. Evaluation (`test_eval.ipynb`)

- Load logs from all algorithms.
- Compute aggregate metrics (Profit, ROI, Degradation).
- Generate comparative plots (Risk-Return, Cost Breakdown).
- Perform temporal analysis of agent behavior.

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
```python    
# Load a customer trace and convert it to the environment format

    df = pl.read_csv("data/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
    customer_id = df["Customer"][0]
    customer_df = df.filter(pl.col("Customer") == customer_id)
    dataset = transform_polars_df(
        customer_df,
        import_energy_price=0.23,
        export_energy_price=0.015,
        price_periods="7am-10am | 4pm-9pm",
        default_import_energy_price=0.15,
        default_export_energy_price=0.01,
    )

    env = SolarBatteryEnv(dataset, max_step=len(dataset) - 1)
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
        "degradation_model": "linear",
        "linear_deg_cost_p_kwh": 0.2,
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

## Dependencies

*   Python 3.x
*   Gymnasium
*   NumPy
*   Polars
*   PyTorch
*   Stable-Baselines3 (for RL agents)
*   (Potentially others listed in `requirements.txt`)
