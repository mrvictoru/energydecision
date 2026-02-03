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

```bash
python src/pretrain_decision_transformer.py \
    --data-dir ./data \
    --patterns train_episodes \
    --epochs 10 \
    --context-length 60
```

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

## Documentation
*   **[COMPONENTS.md](COMPONENTS.md)**: Detailed usage guide for key scripts (`decision.py`, `batterydeg.py`, etc.).
*   **[Household Environment](docs/HOUSEHOLD_ENV_README.md)**: Physics, Reward Function, and Observation Space.
*   **[AEMO Environment](docs/AEMO_ENV_README.md)**: Market dynamics, FCAS, and data pipeline.
