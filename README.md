# Energy Decision: Solar-Battery Control Benchmark

## Overview
This project establishes a comprehensive, reproducible benchmark for residential and grid-scale energy storage control. It integrates high-fidelity Gymnasium environments, diverse baselines (Rule-based, SDP, Online RL, Offline RL), and a standardized evaluation suite.

**Goal:** To minimize energy costs and maximize revenue while rigorously accounting for battery degradation under realistic uncertainty.

## Key Components

1.  **Simulation Environments:**
    *   **Household:** [Household docs guide](docs/household/README.md) - Start here to find the household environment or degradation deep dive.
    *   **Grid:** [AEMO docs guide](docs/aemo/README.md) - Start here to find the right AEMO environment, workflow, replay, degradation, or roadmap document.

2.  **Algorithms ([COMPONENTS.md](COMPONENTS.md)):**
    *   **Optimization:** Stochastic Dynamic Programming (SDP) & Multi-Resolution DP (MRDP).
    *   **Online RL:** PPO, SAC, A2C, DDPG, TD3 (via Stable-Baselines3).
    *   **Offline RL:** Decision Transformers (DT).
    *   **Baselines:** Rule-based heuristics & Oracle (perfect foresight).
    *   **Grid-Agent:** `AEMOAgent` — specialized agent to interact with `AEMOBatteryTradingEnv`, supports rule-based, dispatch-replay, RL, and Decision Transformer inference modes.

## Status

### Roadmap
*   [x] **Core:** Gymnasium environment & Rule-based agents.
*   [x] **Optimization:** SDP & MRDP solvers.
*   [x] **Online RL:** Training loop with SB3.
*   [x] **Offline RL:** Decision Transformer training loop.
*   [x] **Evaluation:** Metrics for return, risk proxies (Sharpe/Sortino), and degradation.
*   [x] **Grid Market:** AEMO Environment Implementation.
*   [x] **Grid battery degradation modelling:** Rainflow-based degradation and capacity fade in `AEMOBatteryTradingEnv`.
*   [x] **Real-world BESS degradation model:** Combined calendar + cycle aging model (`RealWorldBESSDegradationModel`) for utility-scale BESS, with Arrhenius temperature dependency and NMC/LFP chemistry presets, adapted from the framework in Kampker et al. (2025, doi:10.3390/batteries11110392).
*   [x] **Risk-sensitive evaluation:** CVaR/VaR tail-risk metrics (`var_5`, `cvar_5`) computed by `evaluate_experiment_logs`.
*   [x] **Statistical comparisons:** Bootstrap confidence intervals (`bootstrap_confidence_intervals`) and paired Wilcoxon signed-rank tests (`paired_comparison`) across customers/seeds.
*   [x] **Conduct data gathering and training on AEMO env:** Run dispatch replay and RL-agent in `AEMOBatteryTradingEnv` to collect trajectories for offline training for DT and evaluation.
*   [ ] **DT prompt calibration:** Use `recommended_rtg` / `recommended_return_scale` diagnostics to choose in-distribution prompts.
*   [ ] **RL Fine-tuning:** Initialize Online RL with DT weights.
*   [ ] **Hyperparameter Tuning:** Optuna for DT.
*   [ ] **Offline dataset studies:** Evaluate DT sensitivity to behavior-policy mixtures (rule vs SDP vs SB3) and dataset curation.
*   [ ] **Long-context DT experiments:** Study larger `context_len` and RoPE for seasonal/weekly structure.
*   [ ] **Multi-agent extension:** Microgrid setting with multiple households and coordination.
*   [ ] **Sim-to-real readiness:** Add safety wrappers and evaluate policies with hardware-in-the-loop (where available).
*   [ ] **Artifact provenance:** Add lightweight checksums/config logging for datasets, models, and evaluation outputs.


## Installation

### Option 1: Distrobox (Recommended for Linux development)
Sets up a low-friction local-dev shell that works from the repo root.

```bash
podman build -t energydecision:latest .
distrobox create --name energydecision --image energydecision:latest
distrobox enter energydecision
```

If you want CUDA access for DT training, create a second box with NVIDIA passthrough enabled:

```bash
distrobox create --name energydecision-gpu --image energydecision:latest --nvidia
distrobox enter energydecision-gpu
python3 -c "import torch; print(torch.cuda.is_available())"
```

From inside the box, work from the repository root:

```bash
cd /path/to/energydecision
python3 src/pretrain_decision_transformer.py ...
```

Use normal `data/...` and `models/...` paths from the repo root.

For AEMO DT training, prefer the launcher below instead of calling the wrapper directly. It derives tier
defaults, writes a launch plan JSON, and re-enters the preferred Distrobox automatically when invoked from
the host:

```bash
python3 src/launch_aemo_training.py --run-tier proxy-baseline
```

### Option 2: Docker (shared / CI workflow)
Sets up a JupyterLab environment with all dependencies.

```bash
docker compose up --build
```
Access JupyterLab at `http://localhost:8888`.

For shell-based training commands, open another terminal and enter the running container:

```bash
docker exec -it test_energy_container /bin/bash
```

Inside the container the working directory is `/code/src`, so repository-relative paths usually start with `../`.

### Option 3: Local Installation

```bash
git clone <repository-url>
cd energydecision
pip install -r requirements.txt
pip install -r torch_req.txt
```

See `toolbx_guide.md` for the full command sequence and the repo-root path rule.

## Data Setup

1.  **Household Data:** Download **Ausgrid Solar Home Electricity Data** (July 2010 - June 2013) and place it under `data/household/raw/`.
2.  **AEMO Data:** Automatically fetched via `src/aemo_data.py` (cached in `data/aemo/`).

## Reproducing the experiments

The repository now has one canonical notebook location: `notebooks/`. If you are cloning the repo from scratch, the easiest path is:

1. start Distrobox with `distrobox enter energydecision`
2. open Jupyter at `http://localhost:8888`
3. run the notebooks from `notebooks/` in the order below
4. use the CLI training scripts from inside the Distrobox shell when you want long-running DT training outside the notebook UI

### Residential workflow

Use this path to recreate the residential PV + battery experiments.

1. `notebooks/testrun.ipynb`
   - Quick smoke test for the household environment and agents.
   - Good first notebook after cloning to confirm the environment works.

2. `notebooks/test_simrun.ipynb`
   - Main residential data-generation notebook.
   - Runs rule-based and SDP policies.
   - Writes household trajectory logs under `data/household/logs/`.

3. `notebooks/test_sb3train.ipynb`
   - Main residential online-RL notebook.
   - Trains SB3 agents and saves models under `models/household/sb3/`.
   - Can also generate rollouts you may want to compare against the rule/SDP baselines.

4. `python3 pretrain_decision_transformer.py ...`
    - Main residential offline-RL training entrypoint.
    - Run this from the repo root as `python3 src/pretrain_decision_transformer.py ...`.
    - Reads logs from `data/household/logs/`.
    - Saves DT checkpoints and models under `models/household/dt/`.

5. `notebooks/test_eval.ipynb`
   - Main residential evaluation notebook.
   - Compares baselines, SB3 agents, and DT policies.
   - Writes plots/reports to `eval_output/` as configured in the notebook.

If you only want to recreate the core residential benchmark, the minimum useful sequence is:

`testrun.ipynb` -> `test_simrun.ipynb` -> `test_sb3train.ipynb` -> `pretrain_decision_transformer.py` -> `test_eval.ipynb`

### AEMO workflow

Use this path to recreate the grid-scale AEMO experiments.

1. `notebooks/test_aemo_env.ipynb`
   - Quick environment sanity check for `AEMOBatteryTradingEnv`.
   - Useful right after cloning if you want to verify the AEMO environment before long runs.

2. `notebooks/test_aemo_data.ipynb`
   - Interactive data exploration notebook.
   - Useful for inspecting regions, DUID coverage, and available dispatched batteries before deciding experiment settings.

3. `notebooks/aemo_sb3train.ipynb`
   - Main AEMO online-RL notebook.
   - Trains AEMO SB3 agents (`PPO`, `A2C`, `DDPG`, `SAC`, `TD3`).
   - Saves trained models and rollout logs under `models/aemo_sb3/`.

4. `notebooks/aemo_simrun.ipynb`
   - Main AEMO offline-data notebook.
   - Fetches/caches market data, runs rule-based, dispatch-replay, and optional SB3 behavior policies.
   - Builds the DT dataset at `data/aemo_dt/aemo_dt_dataset.parquet`.
   - Writes raw logs to `data/aemo_dt/raw_logs/` and the config/manifest needed for DT training.

5. `python3 src/launch_aemo_training.py --run-tier ...`
    - Canonical AEMO offline-RL launcher for CLI runs.
    - Derives safe defaults from `proxy-smoke`, `proxy-baseline`, or `learning-baseline`.
    - Re-enters `energydecision-gpu` automatically when available, writes `aemo_training_launch_plan.json`,
      and launches the live dashboard through `src/dt_progress_runner.py`.
    - Uses `src/pretrain_aemo_decision_transformer.py` underneath for the actual training job.

6. `notebooks/aemo_eval.ipynb`
   - Main AEMO evaluation notebook.
   - Use it to compare AEMO baselines, SB3 models, and DT policies after training.

If you only want the main AEMO reproduction path, use:

`test_aemo_env.ipynb` -> `test_aemo_data.ipynb` -> `aemo_sb3train.ipynb` -> `aemo_simrun.ipynb` -> `pretrain_aemo_decision_transformer.py` -> `aemo_eval.ipynb`

### Notebook-to-artifact map

- `notebooks/test_simrun.ipynb` -> `data/household/logs/`
- `notebooks/test_sb3train.ipynb` -> `models/household/sb3/`
- `python3 pretrain_decision_transformer.py` -> `models/household/dt/`
- `notebooks/aemo_sb3train.ipynb` -> `models/aemo_sb3/`
- `notebooks/aemo_simrun.ipynb` -> `data/aemo_dt/`
- `python3 src/launch_aemo_training.py` -> `models/aemo/dt/`
- evaluation notebooks -> `eval_output/` (depending on notebook settings)

### Recreating experiments from code instead of notebooks

If you prefer scripting over notebooks:

- Household environment and agents start from `src/EnergySimEnv.py` and `src/decision.py`
- Residential DT training starts from `src/pretrain_decision_transformer.py`
- AEMO data access starts from `src/aemo_data.py`
- AEMO environment starts from `src/AEMOBatteryEnv.py`
- AEMO notebook helpers live in `src/aemo_notebook_utils.py`
- AEMO DT training starts from `src/pretrain_aemo_decision_transformer.py`
- The robust AEMO training harness starts from `src/launch_aemo_training.py`

The `COMPONENTS.md` file is the best code-oriented reference once you want to move beyond the notebook-first workflow.

## Training Decision Transformers from the CLI

Unless noted otherwise, the command blocks in this section assume you are already inside `/code/src`
after `docker exec -it test_energy_container /bin/bash`. If you run them from the repository root
instead, prefix the script path with `src/` and drop the leading `../` from data/model paths.

### Residential DT training

#### Train from scratch

This starts a new model with a context length of 60 and saves checkpoints in `models/household/dt/`:

```bash
docker exec -it test_energy_container /bin/bash

python3 pretrain_decision_transformer.py \
    --data-dir ../data/household/logs \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 6 \
    --checkpoints-per-epoch 4 \
    --context-length 60 \
    --lr 5e-6 \
    --weight-decay 1e-4 \
    --return-scale 1000.0 \
    --return-loss-weight 0.0005 \
    --checkpoint-path ../models/household/dt/dt_model_checkpoint.pt \
    --save-path ../models/household/dt/dt_model_new.pt \
    --loss-csv-path ../models/household/dt/dt_model_loss_history.csv \
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
- **Ensure your** `return_scale` matches the typical magnitude of returns; very large returns can cause instability.

#### Resume from an existing checkpoint

If you already have a compatible checkpoint (same model config, especially `context_len`), you can resume:

```bash
python3 pretrain_decision_transformer.py \
    --data-dir ../data/household/logs \
    --patterns train test_episodes_01 \
    --epochs 2 \
    --batch-size 8 \
    --checkpoints-per-epoch 10 \
    --context-length 60 \
    --checkpoint-path ../models/household/dt/dt_model_checkpoint.pt \
    --save-path ../models/household/dt/dt_model.pt \
    --resume \
    --num-workers 2 \
    --prefetch-factor 1
```

Notes:

- `--epochs` is the **total** target epoch count; resuming will continue from the last saved epoch.
- `--context-length` must match the value used to create the checkpoint, otherwise `load_state_dict` will fail (e.g. attention mask shape mismatch).

### AEMO DT training

#### Train from AEMO trajectories

After running `notebooks/aemo_simrun.ipynb`, use the AEMO-specific wrapper to train from the exported AEMO parquet dataset without keeping the notebook kernel busy:

```bash
docker exec -it test_energy_container /bin/bash

python3 pretrain_aemo_decision_transformer.py \
    --dataset-path ../data/aemo_dt/aemo_dt_dataset.parquet \
    --model-config ../configs/aemo_decision_transformer_model_kwargs.json \
    --epochs 2 \
    --batch-size 6 \
    --lr 2e-5 \
    --val-split 0.1 \
    --seed 8964 \
    --save-path ../models/aemo/dt/aemo_dt_model.pt \
    --checkpoint-path ../models/aemo/dt/aemo_dt_checkpoint.pt \
    --loss-csv-path ../models/aemo/dt/aemo_dt_loss_history.csv \
    --amp-mode "auto" \
    --num-workers 2 \
    --prefetch-factor 2
```

This wrapper forwards to `pretrain_decision_transformer.py` with the AEMO dataset stem and AEMO DT model config, so the underlying training loop stays the same while the inputs and defaults are AEMO-specific.

For large AEMO datasets, enable episode-based subset training so the combined parquet is broken into smaller subset files and trained sequentially with checkpoint resume:

```bash
python3 pretrain_aemo_decision_transformer.py \
    --dataset-path ../data/aemo_dt/aemo_dt_dataset.parquet \
    --model-config ../configs/aemo_decision_transformer_model_kwargs.json \
    --train-in-subsets \
    --subset-episodes 24 \
    --epochs-per-subset 1 \
    --batch-size 24 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --amp-mode "auto" \
    --save-path ../models/aemo/dt/aemo_dt_model.pt \
    --checkpoint-path ../models/aemo/dt/aemo_dt_checkpoint.pt \
    --loss-csv-path ../models/aemo/dt/aemo_dt_loss_history.csv
```

### Editable DT training surface

`src/pretrain_decision_transformer.py` is the single sanctioned Decision Transformer experiment surface for constrained DT training changes in the current codebase.

- **Editable surface**: `src/pretrain_decision_transformer.py`
- **Stable implementation layers**: `src/decision_transformer.py` and `src/transformer_training.py`
- **Adapters that should stay compatible**: `src/pretrain_aemo_decision_transformer.py` and `src/aemo_notebook_utils.py`
- **Read-only areas for this workflow**: evaluation logic, environment dynamics, dataset schema, and notebooks

The editable surface exposes only approved, validated knobs:

- Searchable knobs include presets, model variants, DT dimensions, dropout, RoPE settings, batch size, epochs, learning rate, loss weights, AMP mode, and DataLoader worker settings.
- Frozen invariants include the parquet trajectory schema, the shared DT training engine, the shared model implementation, adapter invocation contracts, and the existing artifact layout (`*.pt`, checkpoint, loss CSVs, metadata sidecars).

Safety and compatibility rules enforced by the shared entrypoint:

- AEMO-shaped DT runs must keep `act_dim` aligned with the action mode (`simple -> 1`, `multi_market -> 3`).
- Transformer width settings must remain internally consistent (`h_dim` divisible by `n_heads`).
- Unknown model-config keys and unsupported preset/variant names are rejected early.
- The editable surface logs a resolved training-surface manifest next to the loss CSV so each run is explicit and reproducible.
- Output artifact paths remain inside the repository root so the harness cannot redirect writes to arbitrary filesystem locations.

Canonical command for the editable surface:

```bash
python3 src/pretrain_decision_transformer.py \
    --surface-preset autoresearch_safe \
    --data-dir data/household/logs \
    --patterns train_episode_01 train_episode_02 \
    --epochs 2 \
    --batch-size 6 \
    --lr 2e-5 \
    --save-path models/household/dt/dt_model.pt \
    --checkpoint-path models/household/dt/dt_model_checkpoint.pt \
    --loss-csv-path models/household/dt/dt_model_loss_history.csv
```

Notes:

- Interactive DT runs now show a built-in live terminal monitor with epoch/batch progress, loss, LR, skipped batches, CPU usage, RAM usage, and GPU/VRAM stats when available.
- The same live monitor works from the repo root and from the Docker shell under `/code/src` because it is built into the shared DT trainer.
- `--subset-episodes` controls how many whole episodes are written into each temporary subset parquet.
- The wrapper now computes one global episode-level train/validation split before writing subset files, so validation stays consistent across all subset stages.
- The first subset starts fresh; later subsets automatically add `--resume` so optimizer and checkpoint state carry forward.
- `--epochs-per-subset` is cumulative across subset stages. For example, `--epochs-per-subset 1` means subset 1 trains to epoch 1, subset 2 resumes and trains to epoch 2, subset 3 resumes and trains to epoch 3, and so on.
- Use conservative settings for the first real run on the large AEMO corpus, especially `--num-workers 0` and a smaller batch size.

Helper evaluations are environment-agnostic and also compute AEMO trading metrics
(revenue, degradation cost, dispatch energy) when those keys exist in `info`.
See [docs/HELPER_README.md](docs/HELPER_README.md) for details.

## Project Structure

```
energydecision/
├── COMPONENTS.md            # Usage guide for scripts
├── configs/                 # JSON model/training configs
├── docs/                    # Deep dive documentation, assets, and references
│   ├── assets/
│   ├── references/
│   ├── aemo/
│   ├── household/
│   ├── DP_ALGORITHM_README.md
│   └── HELPER_README.md
├── notebooks/               # Canonical workflow/demo notebooks
├── src/                     # Source code
│   ├── EnergySimEnv.py      # Household Gym Environment
│   ├── AEMOBatteryEnv.py    # AEMO Market Environment
│   ├── decision.py          # Agent Classes (Rule, RL, SDP)
│   ├── batterydeg.py        # Degradation Models
│   └── ...
├── data/                    # Local datasets, caches, and generated logs (gitignored)
│   ├── household/raw/
│   ├── household/splits/
│   ├── household/logs/
│   ├── aemo/
│   └── aemo_dt/
├── models/                  # Local checkpoints and trained models (gitignored)
│   ├── household/sb3/
│   ├── household/dt/
│   ├── aemo_sb3/
│   └── aemo/dt/
├── eval_output/             # Saved evaluation reports/plots
└── tests/                   # Pytest suite
```

---

## Usage

- Explore the simulation and agent interactions in the [`testrun.ipynb`](notebooks/testrun.ipynb) notebook.
- See demo notebooks [`DemoEnv.ipynb`](notebooks/DemoEnv.ipynb) and [`Demosb3.ipynb`](notebooks/Demosb3.ipynb) for example usage of the gym and stable-baselines3 library.
- **Using the environment class from code:** Instantiate [`SolarBatteryEnv`](src/EnergySimEnv.py) and [`Agent`](src/decision.py) directly to run a single episode and capture step-level logs.

```python
import polars as pl
from src.helper import transform_polars_df
from src.EnergySimEnv import SolarBatteryEnv
from src.decision import Agent

# Example usage
df = pl.read_csv("data/household/raw/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
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

    df = pl.read_csv("data/household/raw/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
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

    df = pl.read_csv("data/household/raw/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
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
    trajectories.write_parquet("data/household/logs/ppo_test_episode_logs.parquet")
```

*   **Training the Decision Transformer with offline interaction data:** [`train_decision_transformer`](src/transformer_training.py) consumes a [`TrajectoryDataset`](src/transformer_training.py) built from logged trajectories.

    ```python
    import torch
    import polars as pl
    from src.helper import transform_polars_df
    from src.EnergySimEnv import SolarBatteryEnv
    from src.transformer_training import TrajectoryDataset, train_decision_transformer
    from src.decision_transformer import DecisionTransformer

    df = pl.read_csv("data/household/raw/2011-2012 Solar home electricity data v2.csv", skip_rows=1)
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
        data_path="data/household/logs/rule_train_episode_01_logs.parquet",
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
        save_path="models/household/dt/dt_model.pt",
        checkpoint_path="models/household/dt/dt_model_checkpoint.pt",
    )
    ```

*   **Evaluating model performance from interaction data:** Use [`evaluate_experiment_logs`](src/helper.py) for single experiments or [`evaluate_experiments`](src/helper.py) for comparisons. Figures can be saved by passing `save_dir`.

    ```python
    import polars as pl
    from src.helper import evaluate_experiment_logs, evaluate_experiments

    ppo_logs = [
        pl.read_parquet("data/household/logs/ppo_test_episode_01_logs.parquet"),
        pl.read_parquet("data/household/logs/ppo_test_episode_02_logs.parquet"),
    ]
    rule_logs = [
        pl.read_parquet("data/household/logs/rule_test_episode_01_logs.parquet"),
        pl.read_parquet("data/household/logs/rule_test_episode_02_logs.parquet"),
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

| Test File | Purpose |
|-----------|---------|
| `test_environment.py` | SolarBatteryEnv functionality, observation handling |
| `test_decision_agent.py` | SDP solver, Oracle agent, policy computation |
| `test_performance.py` | Performance benchmarks and optimization validation |
| `test_quantile_scenarios.py` | Quantile scenario generation |
| `test_aemo_degradation.py` | Rainflow counter, capacity fade, SOC tracking |
| `test_real_world_degradation.py` | RealWorldBESSDegradationModel unit tests, AEMO env integration, mode switching |
| `test_episode_visualizer.py` | Env type detection, plotting, saving, edge cases |
| `test_algorithm_classes.py` | SDP/MRDP/Oracle class imports & init |
| `test_aemo_env_compatibility.py` | Gymnasium API, SB3 compat, observation space |
| `test_risk_statistics.py` | CVaR/VaR, bootstrap CIs, paired comparisons |

Run `pytest tests/ -v` to see the current test total for your checkout and environment.

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
*   **[COMPONENTS.md](COMPONENTS.md)**: Detailed usage guide for key scripts (`decision.py`, `batterydeg.py`, etc.).
*   **[program.md](program.md)**: Repository-specific instructions for an autonomous autoresearch harness operating on the constrained DT training surface.
*   **[Household docs guide](docs/household/README.md)**: Entry point for the household documentation set.
*   **[AEMO docs guide](docs/aemo/README.md)**: Entry point for the AEMO documentation set.
*   **[Household Environment](docs/household/environment.md)**: Physics, reward function, and observation space.
*   **[AEMO Environment](docs/aemo/environment.md)**: Market dynamics, FCAS, and data pipeline.
*   **[Dispatch Replay Utilities](docs/aemo/dispatch-replay.md)**: `dispatch_utils` API — selecting DUIDs, resolving sizing, and running replay episodes.
*   **[AEMO DT Workflow](docs/aemo/workflow.md)**: Notebook-first AEMO offline-data collection, SB3 training, and Decision Transformer workflow.
