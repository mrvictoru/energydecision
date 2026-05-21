# Copilot Instructions for `energydecision`

## Commands

### Build / runtime setup
- **Recommended Linux dev shell:** build the image, then work from the repo root inside Distrobox:
  ```bash
  podman build -t energydecision:latest .
  distrobox create --name energydecision --image energydecision:latest
  distrobox enter energydecision
  ```
- **GPU Distrobox for DT training:**
  ```bash
  distrobox create --name energydecision-gpu --image energydecision:latest --nvidia
  distrobox enter energydecision-gpu
  python3 -c "import torch; print(torch.cuda.is_available())"
  ```
- **Docker / shared workflow:**
  ```bash
  docker compose up --build
  docker exec -it test_energy_container /bin/bash
  ```

### Tests
- **Full suite:**
  ```bash
  python -m pytest tests/ -v
  ```
- **Single test file:**
  ```bash
  python -m pytest tests/test_environment.py -v
  ```
- **Single test:**
  ```bash
  python -m pytest tests/test_environment.py::test_<name> -v
  ```
- **Performance-focused test run:**
  ```bash
  python -m pytest tests/test_performance.py -v -s
  ```

## High-level architecture

- The repo has **two main simulation tracks**:
  - **Household**: `src/helper.py` transforms Ausgrid-style household data into the schema expected by `src/EnergySimEnv.py` (`SolarBatteryEnv`).
  - **AEMO / grid-scale**: `src/aemo_data.py` fetches and caches market data, `src/AEMOBatteryEnv.py` preprocesses it and exposes `AEMOBatteryTradingEnv`.

- `src/decision.py` is the main **agent orchestration layer**. It wraps multiple policy families over those environments: rule-based, SDP, MRDP, oracle, SB3 RL, and Decision Transformer inference. `AEMOAgent` is the grid-scale counterpart for AEMO-specific action modes and dispatch replay flows.

- The repo is **not just a library**; it is a **notebook-first experiment pipeline** with CLI training entrypoints:
  - household logs are generated into `data/household/logs/`
  - AEMO offline datasets are generated into `data/aemo_dt/`
  - SB3 models land under `models/household/sb3/` or `models/aemo_sb3/`
  - DT checkpoints land under `models/household/dt/` or `models/aemo/dt/`
  - evaluation outputs land under `eval_output/`

- Decision Transformer training is intentionally layered:
  - `src/pretrain_decision_transformer.py` is the **canonical shared DT entrypoint**
  - `src/decision_transformer.py` holds the model implementation
  - `src/transformer_training.py` holds the dataset/training engine and writes live progress snapshots
  - `src/pretrain_aemo_decision_transformer.py` is an AEMO-specific wrapper that forwards into the shared entrypoint
  - `src/launch_aemo_training.py` is the higher-level AEMO launcher that picks run tiers, prepares artifact paths, and handles Distrobox/runtime re-entry
  - `src/autoresearch_evaluator.py` is the fixed evaluator for DT checkpoints on held-out AEMO scenarios

- For AEMO workflows, notebooks and helpers in `src/aemo_notebook_utils.py` bridge data collection, dispatch replay, DT dataset generation, subset partitioning, and training/evaluation handoff.

## Key conventions

- **Path convention depends on runtime**:
  - In the recommended repo-root / Distrobox workflow, run scripts as `python3 src/...` and use `data/...` and `models/...` paths directly.
  - In the Docker Compose shell, the working directory is `/code/src`, so commands use bare script names and `../data/...` / `../models/...`.

- **Prefer `src/launch_aemo_training.py` for AEMO CLI runs** instead of calling the wrapper directly. It encodes the repository’s canonical AEMO run tiers (`proxy-smoke`, `proxy-baseline`, `learning-baseline`), writes a launch plan, and handles the preferred container/runtime behavior.

- **Treat `src/pretrain_decision_transformer.py` as the sanctioned DT experiment surface.** The surrounding DT model/training layers and adapter contracts are intentionally stable. Unless the task is explicitly about those layers, keep `src/decision_transformer.py`, `src/transformer_training.py`, `src/pretrain_aemo_decision_transformer.py`, `src/aemo_notebook_utils.py`, environment logic, notebooks, and the evaluator unchanged.

- **AEMO DT dimensions are constrained by action mode**:
  - `simple` -> `act_dim=1`
  - `multi_market` -> `act_dim=3`
  - also keep `h_dim` divisible by `n_heads`

- **Keep artifact locations inside the repo-standard layout.** Household logs go under `data/household/logs/`; AEMO DT datasets under `data/aemo_dt/`; DT outputs under `models/household/dt/` or `models/aemo/dt/`; evaluation output under `eval_output/`.

- **AEMO subset training is episode-based, not row-based.** The wrapper/helper code performs an episode-level train/validation split before writing subset parquet files, then resumes checkpoints across subset stages.

- **The live DT progress UI is driven by snapshot files written by the trainer.** `src/dt_progress_runner.py` is the companion monitor and supports `--attach` for already-running jobs.
