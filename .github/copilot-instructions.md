# Copilot Instructions for `energydecision`

This repository is a benchmark and research codebase for residential and grid-scale battery control. The two main tracks are household solar-battery control and AEMO market trading with FCAS; understanding both tracks usually requires reading multiple modules rather than a single file.

## Build, test, and runtime commands

### Runtime setup
- Preferred Linux workflow: build the image and work from the repo root inside Distrobox.
  ```bash
  podman build -t energydecision:latest .
  distrobox create --name energydecision --image energydecision:latest
  distrobox enter energydecision
  ```
- GPU-capable box for DT training:
  ```bash
  distrobox create --name energydecision-gpu --image energydecision:latest --nvidia
  distrobox enter energydecision-gpu
  python3 -c "import torch; print(torch.cuda.is_available())"
  ```
- Shared Docker workflow:
  ```bash
  docker compose up --build
  docker exec -it test_energy_container /bin/bash
  ```
- Local install:
  ```bash
  pip install -r requirements.txt
  pip install -r torch_req.txt
  ```

### Tests
- No dedicated linter or formatter is defined in this repo; the main validation path is pytest.
- Full suite:
  ```bash
  python -m pytest tests/ -v
  ```
- Single test file:
  ```bash
  python -m pytest tests/test_environment.py -v
  ```
- Single test:
  ```bash
  python -m pytest tests/test_environment.py::test_<name> -v
  ```
- Performance-focused run:
  ```bash
  python -m pytest tests/test_performance.py -v -s
  ```

## High-level architecture

- The repo has two main tracks:
  - Household: `src/helper.py` prepares Ausgrid-style data for `src/EnergySimEnv.py` (`SolarBatteryEnv`).
  - AEMO / grid-scale: `src/aemo_data.py` fetches and caches market data, and `src/AEMOBatteryEnv.py` exposes `AEMOBatteryTradingEnv`.
- `src/decision.py` is the main orchestration layer for rule-based control, SDP, MRDP, oracle, SB3 RL, and Decision Transformer inference. `AEMOAgent` is the grid-scale counterpart for AEMO action modes and dispatch replay.
- The codebase is notebook-first, but CLI entrypoints exist for repeatable experiments. The repo-standard artifact directories are:
  - `data/household/logs/`
  - `data/aemo_dt/` and `data/aemo_dt_fcas/`
  - `models/household/sb3/`, `models/aemo_sb3/`, `models/household/dt/`, `models/aemo/dt/`
  - `eval_output/`
- Decision Transformer training is layered:
  - `src/pretrain_decision_transformer.py` is the shared residential DT entrypoint.
  - `src/decision_transformer.py` contains the model.
  - `src/transformer_training.py` contains the dataset/training engine and snapshot writing.
  - `src/pretrain_aemo_decision_transformer.py` wraps the shared entrypoint for AEMO.
  - `src/launch_aemo_training.py` selects AEMO tiers, prepares paths, and handles runtime re-entry.
  - `src/autoresearch_evaluator.py` evaluates AEMO DT checkpoints on held-out scenarios.
- `src/aemo_notebook_utils.py` bridges AEMO data collection, dispatch replay, DT dataset assembly, subset splitting, and training/evaluation handoff.

## Key conventions

- Use the runtime-specific path style that matches the shell you are in:
  - Distrobox / repo-root: run `python3 src/...` and use `data/...` and `models/...`.
  - Docker Compose shell: the working directory is `/code/src`, so scripts are bare names and repo paths usually use `../data/...` and `../models/...`.
- The repo uses a flat namespace: there is no `setup.py`, no `pyproject.toml`, and `src/` has no `__init__.py`. Files import each other directly (for example, `from EnergySimEnv import SolarBatteryEnv`).
- Tests add `src/` to `sys.path` via `tests/conftest.py`, so you do not need a separate `PYTHONPATH` setup.
- Prefer the sanctioned DT entrypoints: `src/pretrain_decision_transformer.py` for residential DT work and `src/launch_aemo_training.py` for AEMO CLI runs. Avoid changing the surrounding DT stack unless the task explicitly requires it.
- AEMO action dimensions are mode-dependent: `simple` uses `act_dim=1`, `multi_market` uses `act_dim=3`, and `full_fcas` uses `act_dim=9`. Keep `h_dim` divisible by `n_heads` when modifying model dimensions.
- Keep artifacts in the repo-standard layout. AEMO subset training is episode-based: split by episode before writing subset parquet files, then resume checkpoints across subset stages.
- The live DT progress UI reads trainer snapshots; use `src/dt_progress_runner.py --attach` to monitor an existing job.
- For AEMO data issues, prefer local cache/manual data over the network. Use `data/aemo/manual/` or `AEMO_GENERATORS_FILE` for static table fallback, and `AEMO_CACHE_ONLY=1` to force cached monthly MMS files.

## GPU crash telemetry (Xid79 / GPU lost)

For long GPU runs (especially DT pretraining or optimizer sweeps), collect host-side telemetry so an NVIDIA Xid79 / “GPU is lost” event leaves a timeline that survives reboots.

Recommended runner from the repo root:
```bash
bash scripts/run_full_learning_baseline.sh <TAG>
```

This captures `nvidia_smi_query.csv`, `nvidia_dmon.txt`, `vmstat.txt`, `iostat.txt`, `timeline.txt`, `journalctl_k_follow.txt`, and `train_stdout_stderr.log` under `system_logs/<TAG>/`.

Only shut down or reboot once the run directory contains `SAFE_TO_SHUTDOWN.txt`. If `CRASH_DETECTED.txt` exists, treat the host as unstable and reboot after saving logs.
