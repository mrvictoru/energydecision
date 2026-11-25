# Source modules

Simulator, planner, policy, Decision Transformer, evaluation and GRPO implementations. This is a flat namespace: there is no `src/__init__.py`, so modules import each other directly, e.g. `from EnergySimEnv import SolarBatteryEnv`.

## Start here

- [Documentation hub](../docs/README.md)
- [Architecture](../docs/architecture.md)
- [Components](../COMPONENTS.md)
- [Control policies reference](../docs/control_policies.md)
- [Repository guide](../REPOSITORY_GUIDE.md)

## Conventions

- Tests add this directory via `sys.path.insert(0, ...)` in `tests/conftest.py`; no `PYTHONPATH` is needed.
- `action_mode` determines AEMO action dimension: `simple -> 1`, `multi_market -> 3`, `full_fcas -> 9`.
- Transformer width `h_dim` must be divisible by `n_heads`.
- Checkpoint architecture must be verified from weights/sidecars, not docs.

See [known issues](../docs/known_issues.md) for modelling caveats.
