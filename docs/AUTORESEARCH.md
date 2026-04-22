# Autoresearch for Decision Transformer Tuning

Autoresearch in this repository is a **constrained experiment loop** for improving DT configs while keeping benchmark/evaluation fixed.

It is designed to answer: _"Did this config change improve held-out performance under guardrails?"_

## What Autoresearch Does

- Trains a DT candidate config.
- Applies **Stage A** screening (cheap artifact sanity checks).
- If Stage A passes, runs full training + **Stage B** held-out evaluation.
- Compares candidate against current baseline/best.
- Records full evidence in a JSONL ledger.

## What It Does **Not** Do

- It does not mutate notebooks.
- It does not regenerate datasets/splits automatically.
- It does not change environment reward logic.
- It should only mutate approved config keys.

---

## Key Files

- Frozen benchmarks:
  - `configs/benchmark_household.json`
  - `configs/benchmark_aemo.json`
- Evaluation CLIs:
  - `src/eval_household.py`
  - `src/eval_aemo.py`
  - shared helpers in `src/eval_common.py`
- Autoresearch core:
  - `src/autoresearch/config_utils.py`
  - `src/autoresearch/stage_a.py`
  - `src/autoresearch/stage_b.py`
  - `src/autoresearch/runner.py`
  - `src/autoresearch/ledger.py`
- LLM loop:
  - `src/autoresearch/llm_backend.py`
  - `src/autoresearch/prompts.py`
  - `src/autoresearch/agent.py`
  - `src/autoresearch/cli.py`

---

## Prerequisites

1. Data and benchmark artifacts are prepared (household logs or AEMO dataset).
2. Docker service is available if using `--docker` (`autoresearch-train` in `docker-compose.yml`).
3. You have baseline/candidate config JSON files.
4. For agent mode, your LLM backend is reachable.

---

## Mutable Surface (v1)

Autoresearch is intentionally narrow. Mutations should stay in config surface such as:

- Architecture: `n_block`, `h_dim`, `n_heads`, `drop_p`, `context_len`, `rope_*`
- Training: `batch_size`, `lr`, `epochs`, `return_scale`, `*_loss_weight`, `weight_decay`
- Prompting: `rtg_value`, `recommended_rtg_percentile`
- AEMO-only knobs: `action_mode`, `degradation_mode`, `degradation_chemistry`, `step_duration_hours`

Frozen benchmark keys (dataset paths, eval setup, guardrails, state/action dims) should not be mutated.

---

## End-to-End Walkthrough (Manual Mode)

Manual mode is the safest way to start.

### 1) Prepare a baseline config

Create a JSON config (example path: `configs/baseline_household.json`) containing your current DT settings.

### 2) Prepare a candidate config

Create `configs/candidate_household.json` by copying baseline and changing only intended mutable keys.

### Example config templates

Use these as starting points and adjust to your environment.

`configs/baseline_household.json`

```json
{
  "n_block": 2,
  "h_dim": 128,
  "n_heads": 8,
  "drop_p": 0.1,
  "context_len": 60,
  "rope_enabled": false,
  "rope_base": 10000.0,
  "rope_max_position": 180,
  "batch_size": 6,
  "lr": 2e-5,
  "epochs": 2,
  "return_scale": 1000.0,
  "action_loss_weight": 1.0,
  "state_loss_weight": 0.01,
  "return_loss_weight": 0.002,
  "weight_decay": 0.0001,
  "rtg_value": 0.0,
  "recommended_rtg_percentile": 95
}
```

`configs/candidate_household.json` (single-change example)

```json
{
  "n_block": 2,
  "h_dim": 128,
  "n_heads": 8,
  "drop_p": 0.1,
  "context_len": 80,
  "rope_enabled": false,
  "rope_base": 10000.0,
  "rope_max_position": 240,
  "batch_size": 6,
  "lr": 2e-5,
  "epochs": 2,
  "return_scale": 1000.0,
  "action_loss_weight": 1.0,
  "state_loss_weight": 0.01,
  "return_loss_weight": 0.002,
  "weight_decay": 0.0001,
  "rtg_value": 0.0,
  "recommended_rtg_percentile": 95
}
```

Tip: start with one or two mutations per candidate so keep/discard outcomes are easier to interpret.

### 3) Run one candidate cycle

```bash
python -m src.autoresearch \
  --mode manual \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --candidate-config configs/candidate_household.json \
  --output-dir eval_output/autoresearch \
  --ledger-path eval_output/autoresearch/ledger.jsonl
```

Use `--docker` to execute training through `docker compose run autoresearch-train ...`.

### 3b) Evaluate an existing checkpoint without retraining

If you already trained a checkpoint and only want to run Stage B evaluation, use
`--skip-training` together with `--model-path`.

```bash
python -m src.autoresearch \
  --mode manual \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --candidate-config configs/candidate_household.json \
  --skip-training \
  --model-path models/household/dt/dt_model.pt \
  --output-dir eval_output/autoresearch
```

This is useful when:

- you want to validate an existing DT checkpoint against the frozen benchmark,
- Stage A training is not needed,
- or you want to iterate on eval-side changes without re-running training.

When `--skip-training` is enabled:

- `--model-path` is required,
- Stage A is treated as passed with `stage_a_reason = "skipped training"`,
- the run still writes a ledger entry and evaluation outputs,
- and the runner compares the candidate against the current kept baseline as usual.

### 4) Inspect artifacts

For each run, artifacts are saved under:

- `eval_output/autoresearch/<run_id>/stage_a/`
- `eval_output/autoresearch/<run_id>/stage_b/`
- `eval_output/autoresearch/<run_id>/run_summary.json`

### 5) Inspect ledger

```bash
python -m src.autoresearch.ledger --summary eval_output/autoresearch/ledger.jsonl
```

The ledger is append-only and keeps decision provenance (`keep`, `discard`, `crash`, `stage_a_reject`).

---

## End-to-End Walkthrough (Agent Mode)

Agent mode proposes config diffs with an LLM and runs repeated cycles.

### Option A: llama.cpp (default local)

```bash
python -m src.autoresearch \
  --mode agent \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --llm-backend llamacpp \
  --llm-endpoint http://localhost:8080/v1 \
  --iterations 5
```

### Option B: Ollama

```bash
python -m src.autoresearch \
  --mode agent \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --llm-backend ollama \
  --llm-endpoint http://localhost:11434/v1 \
  --llm-model qwen2.5:32b \
  --iterations 5
```

### Option C: OpenAI-compatible cloud

```bash
export OPENAI_API_KEY=...

python -m src.autoresearch \
  --mode agent \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --llm-backend openai \
  --llm-model gpt-4o \
  --iterations 5
```

---

## Using Eval CLIs Directly

Useful when validating checkpoints independently of autoresearch:

```bash
python -m src.eval_household \
  --benchmark configs/benchmark_household.json \
  --model-path models/household/dt/dt_model.pt \
  --output-dir eval_output/household_eval

python -m src.eval_aemo \
  --benchmark configs/benchmark_aemo.json \
  --model-path models/aemo/dt/aemo_dt_model.pt \
  --output-dir eval_output/aemo_eval
```

Both write:

- `eval_metrics.json`
- `eval_summary.json`

## Using the Runner Directly

The runner is the lowest-level entrypoint if you want to run a single candidate
cycle without the LLM wrapper.

### Full train + evaluate

```bash
python -m src.autoresearch.runner \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --candidate-config configs/candidate_household.json \
  --output-dir eval_output/autoresearch
```

### Evaluate an existing checkpoint

If you already have a trained checkpoint, skip retraining and run Stage B only:

```bash
python -m src.autoresearch.runner \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --candidate-config configs/candidate_household.json \
  --skip-training \
  --model-path models/household/dt/dt_model.pt
```

This path is useful for:

- quick checkpoint validation,
- comparing eval-side changes without re-running training,
- and reproducing the upstream `skip-training` runner tests.

## Docker + Local LLM Access

If you later connect autoresearch to a local LLM server running on the host
machine, remember that the container needs a host mapping on Linux.

In `docker-compose.yml`, add:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

The current repository Docker compose file already includes this mapping for
both `app` and `autoresearch-train`.

Then point the LLM backend at the host from inside the container:

- llama.cpp: `http://host.docker.internal:8080/v1`
- Ollama: `http://host.docker.internal:11434/v1`

This is not required for manual or skip-training runs, but it is needed for
agent mode when the backend runs outside the container.

---

## Keep/Discard Logic Summary

- **Stage A reject**: crashed training, missing artifacts, divergence, invalid losses.
- **Stage B discard**: guardrail failures or no primary metric improvement.
- **Keep**: candidate improves primary metric while passing guardrails.

---

## Practical Tuning Strategy

1. Start with manual mode for 3-5 controlled candidates.
2. Confirm Stage A/Stage B and ledger behavior are sensible.
3. Run short agent loops (`iterations=3..10`) before long loops.
4. Keep mutation surface narrow; avoid changing too many knobs at once.
5. Promote only stable improvements to baseline config.

---

## Troubleshooting

- LLM connection failures: check endpoint and model server process.
- Empty/invalid LLM JSON: prompts parser retries, then raises explicit error.
- Import path issues: use `python -m src.autoresearch ...` from repo root.
- Missing training artifacts: inspect `stage_a/loss.csv` and run summary.

---

## Recommended Validation Checklist

- Manual household run completes and appends ledger entry.
- Manual AEMO run completes and appends ledger entry.
- Agent loop runs for multiple iterations with your chosen backend.
- `python -m src.autoresearch.ledger --summary ...` shows expected rows.
