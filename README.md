# Energy Decision

This project explores different algorithms for optimizing energy management in a solar-battery-grid system using a gymnasium environment to simulate system interaction and reward calculation. The goal is to minimize energy costs while considering battery degradation.

## Features

*   **Simulation Environment:** A custom Gym environment ([`src/EnergySimEnv.py`](src/EnergySimEnv.py)) simulating a household with solar panels, a battery, and grid connection.
*   **Control Algorithms:** Implements and compares several control strategies within the [`Agent`](src/decision.py) class in [`src/decision.py`](src/decision.py):
    *   Rule-Based Controller
    *   Reinforcement Learning (RL) agents (using pre-trained models like A2C, DDPG, PPO)
    *   Decision Transformer (DT)
    *   Stochastic Dynamic Programming (SDP) with receding horizon optimization
*   **Battery Degradation Modeling:** Includes models for battery degradation based on usage patterns ([`src/batterydeg.py`](src/batterydeg.py)), incorporating both static and dynamic (rainflow counting) approaches.
*   **Scenario Generation:** Supports scenario-based optimization using forecast data ([`src/helper.py`](src/helper.py)).

## ToDo
*   **Improve SDP algo:** Improve computation speed
*   **Online learning loop:** Training loop using stablebaselines3
*   **Offline learning loop:** Collecting interaction dataset with various algorithms and use it to train a Decision Transformer based control algorithm


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
        sudo docker-compose up --build
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
    dataset = transform_polars_df(df, import_energy_price=0.15, export_energy_price=0.1, price_periods="7am – 10am | 4pm – 9pm", default_import_energy_price=0.1, default_export_energy_price=0.08)

    # Initialize environment
    env = SolarBatteryEnv(dataset, max_step=len(dataset)-1)
    
    resolution  = 20
    # Initialize agent (e.g., SDP)
    SDPagent = Agent(
        env,
        algorithm='sdp',
        soc_resolution=resolution,
        action_resolution=int(resolution+1),
        degradation_model='linear',
        linear_deg_cost_p_kwh=0.2 # based on 5000 cycles life and capcacity of 13.5kWh and replacement cost of $15,300
    )

    # Run a simulation episode
    results_df = agent.run_episode()
    print(results_df)
    ```

## Dependencies

*   Python 3.x
*   Gymnasium
*   NumPy
*   Polars
*   PyTorch
*   Stable-Baselines3 (for RL agents)
*   (Potentially others listed in `requirements.txt`)