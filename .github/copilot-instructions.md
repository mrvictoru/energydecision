# Copilot Instructions for `energydecision`

## Commands

### Build / runtime setup
- **Preferred Linux workflow:** build the image, then work from the repo root inside Distrobox.
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
- **Shared Docker workflow:**
  ```bash
  docker compose up --build
  docker exec -it test_energy_container /bin/bash
  ```
- **Local install:**
  ```bash
  pip install -r requirements.txt
  pip install -r torch_req.txt
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
- **Performance-focused run:**
  ```bash
  python -m pytest tests/test_performance.py -v -s
  ```

## High-level architecture

- The project has two main tracks:
  - **Household:** `src/helper.py` prepares Ausgrid-style data for `src/EnergySimEnv.py` (`SolarBatteryEnv`).
  - **AEMO / grid-scale:** `src/aemo_data.py` fetches and caches market data, and `src/AEMOBatteryEnv.py` exposes `AEMOBatteryTradingEnv`.
- `src/decision.py` is the main policy/orchestration layer for rule-based control, SDP, MRDP, oracle, SB3 RL, and Decision Transformer inference. `AEMOAgent` is the grid-scale counterpart for AEMO action modes and dispatch replay.
- The repo is notebook-first, with CLI entrypoints for repeatable runs. Typical artifacts are:
  - `data/household/logs/`
  - `data/aemo_dt/`
  - `models/household/sb3/`
  - `models/aemo_sb3/`
  - `models/household/dt/`
  - `models/aemo/dt/`
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

- **Use the right path style for the runtime.**
  - Distrobox / repo-root: run `python3 src/...` and use `data/...` and `models/...`.
  - Docker Compose shell: the working directory is `/code/src`, so scripts are bare names and repo paths usually use `../data/...` and `../models/...`.
- **Prefer `src/launch_aemo_training.py` for AEMO CLI runs.** It encodes the canonical tiers (`proxy-smoke`, `proxy-baseline`, `learning-baseline`) and writes a launch plan.
- **Treat `src/pretrain_decision_transformer.py` as the sanctioned DT surface.** Avoid changing the surrounding DT/model/training stack unless the task explicitly requires it.
- **AEMO action dimensions are mode-dependent.** `simple` uses `act_dim=1`; `multi_market` uses `act_dim=3`; keep `h_dim` divisible by `n_heads`.
- **Keep artifacts in the repo-standard layout.** Household logs, AEMO datasets, DT outputs, and evaluation output should stay in their documented directories.
- **AEMO subset training is episode-based.** Split by episode before writing subset parquet files, then resume checkpoints across subset stages.
- **The live DT progress UI reads trainer snapshots.** Use `src/dt_progress_runner.py --attach` to monitor an existing job.
- **For AEMO cache issues, prefer local data over the network.** Use manual `data/aemo/manual/` / `AEMO_GENERATORS_FILE` for static table fallback, and `AEMO_CACHE_ONLY=1` to force cached monthly MMS files.

## GPU crash telemetry (Xid79 / GPU lost)

When running long GPU training (DT pretrain / optimizer experiments), always collect host-side telemetry so an NVIDIA Xid79 / “GPU is lost” event leaves a timeline that survives reboots.

### Recommended runner (host-side)

From repo root (host), launch the telemetry-wrapped training script (logs are written under `eval_output/training/monitor/<TAG>/`):

```bash
bash eval_output/training/monitor/run_full_learning_baseline.sh <TAG>
```

This captures:
- `nvidia_smi_query.csv` (util/temp/power/PCIe link gen+width)
- `nvidia_dmon.txt` (dmon streaming incl PCIe throughput/error counters)
- `vmstat.txt` / `iostat.txt` + `timeline.txt`
- `journalctl_k_follow.txt` (kernel follow; Xid timeline)
- `train_stdout_stderr.log`

### Shutdown safety contract

Only shut down / reboot once the run directory contains `SAFE_TO_SHUTDOWN.txt`.
- The runner is designed to write `SAFE_TO_SHUTDOWN.txt` even when the training process exits non-zero (e.g. SIGABRT) or the shell receives SIGHUP/TERM.
- If `CRASH_DETECTED.txt` exists, assume the system is unstable and a reboot is **required** after logs are saved.
- If `SAFE_TO_SHUTDOWN.txt` exists without `CRASH_DETECTED.txt`, the run finished without detecting Xid79 signatures and it is safe to exit Distrobox and shut down.
