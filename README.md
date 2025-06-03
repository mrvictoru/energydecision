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
*   **Refactor Agent class** Refactor Agent class to be less spaghetti
*   **Offline learning loop:** Collecting interaction dataset with various algorithms and use it to train a Decision Transformer based control algorithm
*   **Plot the simulation:** modify render function from env to plot key metrics


## Project Structure

```
energydecision/
├── data/                  # Data files (CSV, PDF)
├── src/                   # Source code
│   ├── EnergySimEnv.py    # Gym environment for the simulation
│   ├── decision.py        # Agent class implementing control algorithms
│   ├── batterydeg.py      # Battery degradation models
│   ├── helper.py          # Utility functions (e.g., scenario provider)
│   ├── transformer.py     # Decision Transformer model definition
│   ├── *.zip              # Pre-trained RL models
│   └── ...
├── .gitignore
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Dockerfile for building the environment
├── README.md              # This file
├── requirements.txt       # Python package requirements
├── testrun.ipynb          # Jupyter notebook for testing/running simulations
└── torch_req.txt          # PyTorch specific requirements
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

*   if there are mulitple environments, simulation can be run in parallel using [`run_episodes_parallel`](src/decision.py)

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

    num_total_steps = len(training_dataset[0])*len(training_dataset)
    print(f"Total number of steps possible in training dataset: {num_total_steps}")

    testing_env_fns = [make_env(ds) for ds in testing_dataset]

    # Create and train a PPO model
    ppo_model, _ = train_model(model_class=PPO, vec_env=training_vec_env, total_timesteps=num_total_steps, eval_env_fn=testing_env_fns[0])

## Dependencies

*   Python 3.x
*   Gymnasium
*   NumPy
*   Polars
*   PyTorch
*   Stable-Baselines3 (for RL agents)
*   (Potentially others listed in `requirements.txt`)
