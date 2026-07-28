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