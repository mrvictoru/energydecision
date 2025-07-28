# Energy Decision

This project explores different algorithms for optimizing energy management in a solar-battery-grid system using a gymnasium environment to simulate system interaction and reward calculation. The goal is to minimize energy costs while considering battery degradation.

## Features

*   **Simulation Environment:** A custom Gym environment ([`src/EnergySimEnv.py`](src/EnergySimEnv.py)) simulating a household with solar panels, a battery, and grid connection. The return observation is normalized against the dataset so it is sutiable with reinforcement learning method. Seperate method can be used to return raw value for observation.
*   **Control Algorithms:** Implements and compares several control strategies within the [`Agent`](src/decision.py) class in [`src/decision.py`](src/decision.py):
    *   Rule-Based Controller
    *   Reinforcement Learning (RL) agents (using pre-trained models like A2C, DDPG, PPO)
    *   Decision Transformer (DT)
    *   Stochastic Dynamic Programming (SDP) with receding horizon optimization
*   **Battery Degradation Modeling:** Includes models for battery degradation based on usage patterns ([`src/batterydeg.py`](src/batterydeg.py)), incorporating both static and dynamic (rainflow counting) approaches.
*   **Scenario Generation:** Supports scenario-based optimization using forecast data ([`src/helper.py`](src/helper.py)).

## ToDo
*   ~~**Improve SDP algo:** Improve computation speed and run algo in different envs in parallel~~
*   ~~**Online learning loop:** Training loop using stablebaselines3~~
*   ~~**Examine the effectiveness of sb3 trained RL model:** Check and find out if the RL model actually output valid actions~~
*   ~~**Offline learning loop:** Collecting interaction dataset with various algorithms and use it to train a Decision Transformer based control algorithm~~
*   ~~**Plot the simulation:** modify render function from env to plot key metrics~~
*   **Refactor Agent class:** Refactor Agent class to be less spaghetti
*   **Optimize training loop:** TBD
*   **Conduct evaluation:** To build framework that can evaluate the effectiveness of different algorithm/parameter


## Project Structure

```
energydecision/
├── data/                       # Data files (CSV, Parquet, etc.)
├── src/                        # Source code
│   ├── EnergySimEnv.py         # Gymnasium environment for solar-battery-grid simulation
│   ├── decision.py             # Agent class: rule-based, RL, DT, and SDP controllers
│   ├── batterydeg.py           # Battery degradation models (static and dynamic)
│   ├── helper.py               # Data transformation and scenario generation utilities
│   ├── decision_transformer.py # Core Decision Transformer model class
│   ├── transformer_training.py # TrajectoryDataset class and train_decision_transformer function
│   ├── sb3train.py             # RL training utilities (Stable-Baselines3)
│   └── ...                     # Other modules/utilities
├── .gitignore
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Dockerfile for building the environment
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Python package requirements
├── testrun.ipynb               # Example Jupyter notebook for running simulations
└── torch_req.txt               # PyTorch-specific requirements
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd energydecision
    ```
2.  **Using Docker:**
*   Build and run the container:
    ```bash
    sudo docker compose up
    ```

## Usage

*   Explore the simulation and agent interactions in the [testrun.ipynb](testrun.ipynb) notebook.
*   Instantiate the [`SolarBatteryEnv`](src/EnergySimEnv.py) and [`Agent`](src/decision.py) classes programmatically to run simulations with different algorithms and parameters.

    ```python
    # Example (Conceptual)
    import polars as pl
    from src.EnergySimEnv import SolarBatteryEnv
    from src.decision import Agent
    from src.helper import transform_polars_df # Or your custom provider

    # Load data
    df = pl.read_csv("data/your_data.csv") # Replace with your data file
    dataset = transform_polars_df(df, import_energy_price=0.15, export_energy_price=0.1, price_periods="7am – 10am | 4pm – 9pm", default_import_energy_price=0.1, default_export_energy_price=0.08) # transform the data into dataset which can be used to build the simulation environment

    # Initialize environment
    env = SolarBatteryEnv(dataset, max_step=len(dataset)-1)

    # Initialize agent (e.g., SDP)
    agent = Agent(
        env,
        algorithm='rule'
    )

    # Run a simulation episode
    results_df = agent.run_episode()
    print(results_df)
    ```

*   if there are mulitple environments, simulation can be run in parallel using [`run_episodes_parallel`](src/decision.py). (only suitable for 'Rule' or 'SDP' based agent)

    ```python
    # Example (Conceptual)
    import polars as pl
    from src.EnergySimEnv import SolarBatteryEnv
    from src.decision import Agent, run_episodes_parallel
    from src.helper import transform_polars_df, make_env

    # Load data
    datapath = '../data/2011-2012 Solar home electricity data v2.csv'
    # skip the first line in csv and read the next line as column
    # then read the rest of the file and store as dataframe
    df = pl.read_csv(datapath, skip_rows=1)
    # get all the unique customers as their own dataframes
    customers = df['Customer'].unique()
    # pick 10% of the random customers as testing data
    testing_customers = np.random.choice(customers, int(0.1*len(customers)), replace=False)
    # transform the data into dataset which can be used build simulation environments
    testing_dataset = []
    for customer in testing_customers:
        customer_df = df.filter(pl.col('Customer') == customer)
        try:
            newcustomerdf = transform_polars_df(customer_df, import_energy_price=0.23, export_energy_price=0.015, price_periods="7am – 10am | 4pm – 9pm", default_import_energy_price=0.15, default_export_energy_price=0.01)
        except Exception as e:
            print(f"Error with customer as testing dataset: {customer}")
            print(e)
            break
        testing_dataset.append(newcustomerdf)

    testing_env_fns = [make_env(ds) for ds in testing_dataset]
    # Initialize environments and SDP agent parameters
    sdp_agent_kwargs = {
        'algorithm': 'sdp',
        'soc_resolution': 20,
        'action_resolution': 41,  # best to be 2*soc_resolution + 1
        'degradation_model': 'linear', # the other option being static degradation 'static'
        'linear_deg_cost_p_kwh': 0.2 # only needed if using linear
    }
    num_step = None # pick the number of step for the simulation
    test_envs = [env_fn(num_step) for env_fn in testing_env_fns]

    # Run all episodes in parallel
    sdp_episode_logs = run_episodes_parallel(Agent, test_envs, agent_kwargs=sdp_agent_kwargs, max_workers=8)

    print(sdp_episode_logs)
    ```

*   Utilise [`train_model`](src/sb3train.py) to train policy using reinforcement learning library stable_baselines3 against the environment

    ```python
    import polars as pl
    from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.helper import transform_polars_df, make_env
    from sb3train import train_model
    from EnergySimEnv import SolarBatteryEnv

    # Load data
    datapath = '../data/2011-2012 Solar home electricity data v2.csv'
    # skip the first line in csv and read the next line as column
    # then read the rest of the file and store as dataframe
    df = pl.read_csv(datapath, skip_rows=1)
    # get all the unique customers as their own dataframes
    customers = df['Customer'].unique()
    # get all the unique customers as their own dataframes
    customers = df['Customer'].unique()
    # pick 80% of the random customers as training data
    training_customers = np.random.choice(customers, int(0.8*len(customers)), replace=False)
    # the rest of the customers are testing data
    testing_customers = np.setdiff1d(customers, training_customers)

    # loop through each customer and use transform_polars_df to get the dataframe and store it in a list call dataset
    training_dataset = []
    for customer in training_customers:
        customer_df = df.filter(pl.col('Customer') == customer)
        try:
            newcustomerdf = transform_polars_df(customer_df, import_energy_price=0.23, export_energy_price=0.015, price_periods="7am – 10am | 4pm – 9pm", default_import_energy_price=0.15, default_export_energy_price=0.01)
        except Exception as e:
            print(f"Error with customer as training dataset: {customer}")
            print(e)
            break
        training_dataset.append(newcustomerdf)

    testing_dataset = []
    for customer in testing_customers:
        customer_df = df.filter(pl.col('Customer') == customer)
        try:
            newcustomerdf = transform_polars_df(customer_df, import_energy_price=0.23, export_energy_price=0.015, price_periods="7am – 10am | 4pm – 9pm", default_import_energy_price=0.15, default_export_energy_price=0.01)
        except Exception as e:
            print(f"Error with customer as testing dataset: {customer}")
            print(e)
            break
        testing_dataset.append(newcustomerdf)
    
    # Create a list of environment creation functions to build a vectorized environment.
    training_env_fns = [make_env(ds) for ds in training_dataset]
    training_vec_env = DummyVecEnv(training_env_fns)

    num_total_steps = len(training_dataset[0])
    print(f"Total number of steps possible in training dataset: {num_total_steps}")

    testing_env_fns = [make_env(ds) for ds in testing_dataset]

    # Create and train a PPO model
    ppo_model, _ = train_model(model_class=PPO, vec_env=training_vec_env, total_timesteps=num_total_steps, eval_env_fn=testing_env_fns[0])

    # simulate the model interaction and record the log
    from decision import run_sb3_model_on_vec_env
    import multiprocessing

    # (only needed if you ever switch to SubprocVecEnv on Linux/notebooks)
    multiprocessing.set_start_method("forkserver", force=True)
    # use SubprocVecEnv
    from stable_baselines3.common.vec_env import SubprocVecEnv
    test_vec_env = SubprocVecEnv(testing_env_fns)

    ppo_episode_logs = run_sb3_model_on_vec_env(ppo_model, test_vec_env)

    # store it as parquet
    from helper import flatten_episode_data
    ppo_logs = flatten_episode_data(ppo_episode_logs)
    ppo_logs.write_parquet("../data/ppo_test_episode_logs.parquet")
    ```

*   Utilise [`train_decision_transformer`](src/transformer_training.py) to train [`DecisionTransformer`](src/decision_transformer.py) using offline interaction data collected through [`run_episodes_parallel`](src/decision.py) or [`run_sb3_model_on_vec_env`](src/decision.py) and load it onto [`TrajectoryDataset`](src/transformer_training.py)

    ```python
    from torch.utils.data import DataLoader
    from src.decision_dataset import TrajectoryDataset, train_decision_transformer
    import polars as pl

    # 1) Prepare environment to get dims

    # Load data
    datapath = '../data/2011-2012 Solar home electricity data v2.csv'
    # skip the first line in csv and read the next line as column
    df = pl.read_csv(datapath, skip_rows=1)
    # then get the data of customer 1
    customer_df = df.filter(pl.col('Customer') == 1)
    newcustomerdf = transform_polars_df(customer_df, import_energy_price=0.23, export_energy_price=0.015, price_periods="7am – 10am | 4pm – 9pm", default_import_energy_price=0.15, default_export_energy_price=0.01)
    env = SolarBatteryEnv(newcustomerdf)
    state_dim = env.observation_space.shape[0]
    act_dim   = env.action_space.shape[0]

    # 2) Load dataset and sample a batch
    context_length = 16
    ds = TrajectoryDataset(
        data_path='../data/rule_all_episode_logs.parquet',
        context_length=context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        discount_factor=0.99,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)

    # 3) Instantiate your DecisionTransformer and train it
    # Get the maximum possible timestep from the environment's data length
    max_steps_in_episode = len(env.df)

    model = DecisionTransformer(
        state_dim   = state_dim,
        act_dim     = act_dim,
        n_block     = 2,
        h_dim       = 128,
        context_len = context_length,
        n_heads     = 8,
        drop_p      = 0.1,
        max_timestep= max_steps_in_episode,
    )

    # Train the Decision Transformer
    trained_model, train_losses = train_decision_transformer(
        ds=ds,
        context_length=context_length,
        state_dim=state_dim,
        act_dim=act_dim,
        max_timestep=max_steps_in_episode,
        model=model,
        batch_size=32,
        lr=1e-4,
        epochs=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_path="../models/dt_model.pt"
    )
    ```

## Dependencies

*   Python 3.x
*   Gymnasium
*   NumPy
*   Polars
*   PyTorch
*   Stable-Baselines3 (for RL agents)
*   (Potentially others listed in `requirements.txt`)
