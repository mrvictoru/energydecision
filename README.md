# energydecision

energydecision is a research codebase and benchmark for battery control across two related but distinct tracks:

- Household solar-battery control with `SolarBatteryEnv`
- Grid-scale AEMO/NEM battery trading with `AEMOBatteryTradingEnv`

The repository combines simulation environments, classical planning baselines, online RL, offline Decision Transformer training, and evaluation tooling. It is intended to support reproducible experiments and to serve as a base for future research work.

## Start Here

- [docs/README.md](docs/README.md): human documentation hub
- [docs/architecture.md](docs/architecture.md): codebase map and system overview
- [docs/development.md](docs/development.md): setup, runtime modes, testing, and contributor workflow
- [report.md](report.md): research report and current benchmark narrative

## Choose Your Track

### Household

Use the household track if you want residential PV + battery control under household load and tariff dynamics.

- Overview: [docs/household/README.md](docs/household/README.md)
- Workflow: [docs/household/workflow.md](docs/household/workflow.md)
- Environment reference: [docs/household/environment.md](docs/household/environment.md)
- Degradation reference: [docs/household/degradation.md](docs/household/degradation.md)

### AEMO / Grid-Scale

Use the AEMO track if you want grid-scale battery trading in energy and FCAS markets.

- Overview: [docs/aemo/README.md](docs/aemo/README.md)
- Workflow: [docs/aemo/workflow.md](docs/aemo/workflow.md)
- Environment reference: [docs/aemo/environment.md](docs/aemo/environment.md)
- Evaluation: [docs/evaluation_guide.md](docs/evaluation_guide.md)

## Quick Setup

### Preferred runtime

The repo is primarily documented around two runtime modes:

- Distrobox from the repo root on Linux
- Docker Compose with a shell inside the running container

The detailed setup and path conventions are in [docs/development.md](docs/development.md).

### Install dependencies locally

```bash
pip install -r requirements.txt
pip install -r torch_req.txt
```

## Fastest Common Workflows

### Household workflow

1. Generate or inspect household logs under `data/household/logs/`.
2. Train a Decision Transformer with:

```bash
python scripts/pretrain_decision_transformer.py \
  --data-dir data/household/logs \
  --patterns train_episode_01 train_episode_02
```

3. Validate with:

```bash
python -m pytest tests/ -v
```

### AEMO workflow

1. Read [docs/aemo/workflow.md](docs/aemo/workflow.md) to choose notebook-first or CLI-first execution.
2. For canonical CLI training, use:

```bash
python scripts/launch_aemo_training.py --run-tier proxy-baseline
python scripts/launch_aemo_training.py --run-tier learning-baseline
```

3. Evaluate checkpoints with:

```bash
python scripts/autoresearch_evaluator.py \
  --surface-manifest-path <surface-manifest.json> \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json \
  --output-dir eval_output/autoresearch/<run-tag>
```

## Repository Layout

```text
configs/      Model configs and evaluator configs
data/         Cached data, generated datasets, and logs
docs/         Human documentation
eval_output/  Evaluation outputs and reports
notebooks/    Exploratory and notebook-first workflows
scripts/      Canonical runnable entrypoints
src/          Reusable implementation modules
tests/        Pytest suite
```

## Research Artifacts

- Report: [report.md](report.md)
- Research notes index: [docs/research/README.md](docs/research/README.md)
- Hugging Face models and data:
  - [Pretrained Decision Transformer v1](https://huggingface.co/mrvictoru/energydecision-dt)
  - [GRPO finetuned Decision Transformer v1](https://huggingface.co/mrvictoru/energydecision-dt-grpo)
  - [Dataset](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade)

## Notes For Contributors

- Treat `scripts/` as the canonical CLI surface.
- Treat `src/` as reusable implementation modules.
- Keep household and AEMO results conceptually separate.
- Prefer adding stable operational guidance to [docs/development.md](docs/development.md) and [docs/architecture.md](docs/architecture.md) rather than expanding this README.