# Tests

Pytest suite for simulator physics, FCAS ordering, market impact, Decision Transformer components and evaluation helpers.

## Start here

- [Development guide](../docs/development.md)
- [Repository guide](../REPOSITORY_GUIDE.md)
- [Project overview](../README.md)

## Run

```bash
python -m pytest tests/ -v
```

`tests/conftest.py` inserts `src/` into `sys.path`; no `PYTHONPATH` is required. The full suite collected 387 tests at the original preprint release state. There is no CI, coverage gate or linter.
