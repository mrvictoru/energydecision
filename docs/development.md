# Development Guide

This guide is the operational entrypoint for contributors and future research work.

## Runtime Modes

The repo supports three common ways of working:

### Distrobox

Preferred for Linux development and longer training runs.

```bash
podman build -t energydecision:latest .
distrobox create --name energydecision --image energydecision:latest
distrobox enter energydecision
```

For CUDA-enabled training:

```bash
distrobox create --name energydecision-gpu --image energydecision:latest --nvidia
distrobox enter energydecision-gpu
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Docker Compose

```bash
docker compose up --build
docker exec -it test_energy_container /bin/bash
```

### Local Python install

```bash
pip install -r requirements.txt
pip install -r torch_req.txt
```

## Path Conventions

The repo uses different path conventions depending on where commands are run.

### Repo root or Distrobox repo root

- Run `python scripts/...` or `python src/...` depending on the entrypoint.
- Use `data/...`, `models/...`, and `eval_output/...` paths.

### Docker Compose shell

- The shell usually starts in `/code/src`.
- Script names are often bare file names.
- Repo-relative artifact paths often need `../data/...` and `../models/...`.

## Canonical Command Surfaces

Prefer the following CLI entrypoints for repeatable work:

- Household DT training: `scripts/pretrain_decision_transformer.py`
- AEMO DT wrapper: `scripts/pretrain_aemo_decision_transformer.py`
- AEMO tier launcher: `scripts/launch_aemo_training.py`
- AEMO held-out evaluation: `scripts/autoresearch_evaluator.py`

Notebook-first workflows still exist, but `scripts/` should be treated as the canonical automation surface.

## Common Commands

### Run tests

```bash
python -m pytest tests/ -v
```

### Run a single test file

```bash
python -m pytest tests/test_environment.py -v
```

### Launch AEMO training

```bash
python scripts/launch_aemo_training.py --run-tier proxy-baseline
python scripts/launch_aemo_training.py --run-tier learning-baseline
```

### Run the AEMO evaluator

```bash
python scripts/autoresearch_evaluator.py \
  --surface-manifest-path <surface-manifest.json> \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json \
  --output-dir eval_output/autoresearch/<run-tag>
```

## Test Suite Coverage (332 tests)

| File | Area | Key Coverage |
|------|------|--------------|
| `test_aemo_data_local_file.py` | Data ingestion | Local AEMO cache/zip handling, fallback to NEMOSIS |
| `test_aemo_degradation.py` | Degradation (rainflow) | Rainflow cycle counting, Muenzel model, capacity fade |
| `test_aemo_dispatch_replay.py` | Dispatch replay | Historical DISPATCHLOAD replay as env actions |
| `test_aemo_dt_hf.py` | HuggingFace models | Model download/verify, surface manifest creation |
| `test_aemo_env_compatibility.py` | Env API | Gymnasium API compliance (step, reset, spaces) |
| `test_aemo_fcas_units.py` | FCAS units | FCAS enablement model, power/energy calculations |
| `test_aemo_full_fcas.py` | FCAS env | 9-dim full_fcas action space, co-optimized bidding |
| `test_aemo_notebook_utils.py` | Notebook utilities | DT dataset building, cache management, eval helpers |
| `test_aemo_oracle_invariant.py` | Oracle baseline | Perfect-foresight LP revenue dominance invariant |
| `test_algorithm_classes.py` | Planning algorithms | SDP/MRDP/Oracle solver initialization & basic ops |
| `test_autoresearch_evaluator.py` | Held-out evaluation | Parallel rollouts, reference caching, metric summarization |
| `test_build_aemo_autoresearch_pilot.py` | Pilot dataset | Curated train/val split from FCAS-rich corpus |
| `test_decision_agent.py` | Agent abstraction | Rule/RL/DT/ORACLE dispatch, episode runners |
| `test_decision_transformer.py` | DT model | Architecture, forward pass, action head, RTG handling |
| `test_dispatch_utils.py` | Dispatch utilities | DISPATCHLOAD parsing, station resolution, replay |
| `test_environment.py` | Household env | SolarBatteryEnv dynamics, reward, degradation |
| `test_episode_visualizer.py` | Visualization | Episode plotting, grid energy, SOC trajectories |
| `test_grpo_posttraining.py` | GRPO fine-tuning | Online RL fine-tuning, mixed action distribution |
| `test_launch_aemo_training.py` | Training launcher | Tier defaults, command building, dry-run plan |
| `test_pretrain_aemo_decision_transformer.py` | AEMO DT CLI | Command building, checkpoint epoch parsing |
| `test_pretrain_decision_transformer.py` | DT training CLI | Legacy CLI contract, surface presets, artifact manifest |
| `test_prewarm_aemo_cache.py` | Cache warming | Eval config-driven AEMO cache precomputation |
| `test_quantile_scenarios.py` | Scenario generation | QuantileScenarioGenerator for SDP/MRDP |
| `test_real_world_degradation.py` | RealWorld BESS deg | Calendar+cycle aging, NMC/LFP chemistry, Arrhenius |
| `test_risk_statistics.py` | Risk metrics | VaR/CVaR@5%, Sharpe/Sortino, bootstrap CIs, Wilcoxon |
| `test_transformer_training.py` | Training internals | Resource monitor, loss aggregation, AMP/grad clip |

## Contributor Rules Of Thumb

### Documentation

- Keep high-level navigation in `README.md` and `docs/README.md`.
- Keep stable operational guidance in this file.
- Keep volatile experiment status in research-note docs, not in onboarding docs.

### Code placement

- Put reusable logic in `src/`.
- Put runnable workflows and entrypoints in `scripts/`.
- Put exploratory or teaching material in `notebooks/`.

### Research hygiene

- Keep household and AEMO results separate.
- Prefer explicit config files and saved manifests for reproducibility.
- Avoid changing dataset schema casually; many workflows assume the existing parquet shape.

## Where To Learn More

- Repo structure: [architecture.md](architecture.md)
- AEMO workflow: [aemo/workflow.md](aemo/workflow.md)
- Household docs: [household/README.md](household/README.md)
- Research-note index: [research/README.md](research/README.md)