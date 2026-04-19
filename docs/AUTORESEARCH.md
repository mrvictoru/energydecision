# Autoresearch

Autoresearch in this repository is a constrained optimization loop over Decision Transformer training configurations.

## What it does

- Runs reproducible training/evaluation cycles for household and AEMO DT setups.
- Applies Stage A screening before expensive full evaluation.
- Applies Stage B keep/discard decisions from held-out metrics.
- Stores every run in an append-only JSONL ledger.

## Architecture

- Benchmarks: `configs/benchmark_household.json`, `configs/benchmark_aemo.json`
- Eval harness: `src/eval_common.py`, `src/eval_household.py`, `src/eval_aemo.py`
- Core loop: `src/autoresearch/runner.py`, `stage_a.py`, `stage_b.py`, `ledger.py`, `config_utils.py`
- LLM loop: `llm_backend.py`, `prompts.py`, `agent.py`

## Manual mode

```bash
python -m src.autoresearch.cli \
  --mode manual \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --candidate-config configs/candidate_household.json
```

## Agent mode

```bash
python -m src.autoresearch.cli \
  --mode agent \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --llm-backend llamacpp \
  --llm-endpoint http://localhost:8080/v1 \
  --iterations 5
```

## Mutable surface

The mutable key surface is defined in `ALLOWED_MUTABLE_KEYS_V1` in `src/autoresearch/config_utils.py`.

## Ledger

Ledger path defaults to `eval_output/autoresearch/ledger.jsonl`.
Each line is one run with decision and artifact pointers.
