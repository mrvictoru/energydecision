# Autoresearch for Battery DT — File-by-File Implementation Handoff

> **Purpose:** This document is the complete implementation spec for an LLM-driven,
> constrained keep/discard optimization loop over Decision Transformer training
> configs with frozen benchmarks, scriptable evaluation, and pluggable local LLM
> backends. It is written so that a coding agent (or human) can implement each
> file top-to-bottom without further clarification.

---

## 0. Context & Motivation

The codebase already has:

| What | Where |
|------|-------|
| DT model | `src/decision_transformer.py` |
| DT training CLI (household) | `src/pretrain_decision_transformer.py` |
| DT training CLI (AEMO) | `src/pretrain_aemo_decision_transformer.py` |
| Episode rollout agents | `src/decision.py` (`Agent`, `AEMOAgent`) |
| Evaluation metrics | `src/helper.py` -> `evaluate_experiment_logs()` |
| Trajectory dataset | `src/transformer_training.py` -> `TrajectoryDataset` |
| AEMO data utilities | `src/aemo_notebook_utils.py` |
| Battery env (household) | `src/EnergySimEnv.py` -> `SolarBatteryEnv` |
| Battery env (AEMO) | `src/AEMOBatteryEnv.py` -> `AEMOBatteryTradingEnv` |
| Model kwargs configs | `configs/aemo_decision_transformer_model_kwargs.json` |

**The gap:** A human must manually write configs, run training, roll out episodes,
read metrics, decide what to try next, and repeat. This plan adds:

1. **Frozen benchmark definitions** -- canonical JSON files that never change.
2. **Scriptable evaluation CLIs** -- deterministic, subprocess-safe eval scripts.
3. **A two-stage screening gate** -- Stage A (training artifact check) and Stage B
   (full held-out evaluation + keep/discard comparison).
4. **An experiment ledger** -- append-only JSONL tracking every run.
5. **An LLM-driven mutation proposer** -- the agent reads the ledger and asks an LLM
   to propose the next config to try. The LLM outputs structured JSON configs only,
   never code. Local <=30B models (llama.cpp, Ollama) are the default.

---

## 1. Architecture

```
+--------------------------------------------------------------------------+
|                        AutoresearchAgent (agent.py)                      |
|                                                                          |
|   +----------+   +------------+   +----------------------------------+  |
|   |  Ledger  |-->|   Prompt   |-->|         LLM Backend              |  |
|   |  (JSONL) |   |   Builder  |   |       (llm_backend.py)           |  |
|   +----------+   |(prompts.py)|   |                                  |  |
|        ^         +------------+   |  +----------------------------+  |  |
|        |                          |  | LlamaCppBackend (default)  |  |  |
|        |         +------------+   |  |  -> http://localhost:8080  |  |  |
|        |         | JSON Parse |<--|  +----------------------------+  |  |
|        |         | + Validate |   |  | OllamaBackend              |  |  |
|        |         +------+-----+   |  |  -> http://localhost:11434 |  |  |
|        |                |         |  +----------------------------+  |  |
|        |                v         |  | OpenAIBackend (cloud)      |  |  |
|        |     +------------------+ |  |  -> api.openai.com         |  |  |
|        +-----|  Runner           | |  +----------------------------+  |  |
|   (keep/     |  (runner.py)      | +----------------------------------+  |
|   discard)   +--------+----------+                                        |
|                       |                                                  |
+-------------------------|--------------------------------------------------+
                        |
         +--------------+--------------+
         v              v              v
   +---------+  +------------+  +----------+
   | Stage A |  |  DT Train  |  | Stage B  |
   | Screen  |  |  Pipeline  |  |  Eval    |
   |(stage_a)|  | pretrain_* |  |(stage_b) |
   +---------+  +------------+  +----------+
                                      |
                            +---------+-----------+
                            v                    v
                   eval_household.py       eval_aemo.py
                            +--------+------------+
                                  eval_common.py
                             (model load, guardrails,
                              output writing)
```

### Key principle: LLM is the mutation proposer only

The LLM reads the ledger and outputs a JSON config diff. It never sees training
code, never calls APIs directly, and never writes to disk. All actual work
(training, evaluation, ledger writing) is done by deterministic Python code.

---

## 2. File Manifest

| # | File | Action | Purpose |
|---|------|--------|---------|
| 1 | `configs/benchmark_household.json` | CREATE | Frozen Household benchmark definition |
| 2 | `configs/benchmark_aemo.json` | CREATE | Frozen AEMO benchmark definition |
| 3 | `src/eval_common.py` | CREATE | Shared eval utilities (model loading, guardrails, output writing) |
| 4 | `src/eval_household.py` | CREATE | Household DT evaluation CLI |
| 5 | `src/eval_aemo.py` | CREATE | AEMO DT evaluation CLI |
| 6 | `src/autoresearch/__init__.py` | CREATE | Package init; re-exports public API |
| 7 | `src/autoresearch/config_utils.py` | CREATE | Config loading, diffing, mutable surface validation |
| 8 | `src/autoresearch/ledger.py` | CREATE | JSONL experiment ledger |
| 9 | `src/autoresearch/stage_a.py` | CREATE | Training artifact screening |
| 10 | `src/autoresearch/stage_b.py` | CREATE | Held-out eval + keep/discard comparison |
| 11 | `src/autoresearch/runner.py` | CREATE | Main autoresearch runner CLI |
| 12 | `src/autoresearch/llm_backend.py` | CREATE | LLM backend abstraction + implementations |
| 13 | `src/autoresearch/prompts.py` | CREATE | System/user prompt templates + JSON parser |
| 14 | `src/autoresearch/agent.py` | CREATE | `AutoresearchAgent` -- LLM-driven outer loop |
| 15 | `src/autoresearch/cli.py` | CREATE | CLI entrypoint (`python -m src.autoresearch`) |
| 16 | `src/autoresearch/__main__.py` | CREATE | Delegates to `cli.main()` |
| 17 | `docker-compose.yml` | MODIFY | Add `autoresearch-train` service |
| 18 | `docs/AUTORESEARCH.md` | CREATE | Full user-facing documentation |
| 19 | `README.md` | MODIFY | Add autoresearch section with link |
| 20 | `tests/test_autoresearch_ledger.py` | CREATE | Ledger unit tests |
| 21 | `tests/test_autoresearch_stage_a.py` | CREATE | Stage A unit tests |
| 22 | `tests/test_autoresearch_stage_b.py` | CREATE | Stage B unit tests |
| 23 | `tests/test_autoresearch_config_utils.py` | CREATE | Config utils unit tests |
| 24 | `tests/test_eval_common.py` | CREATE | Eval common unit tests |
| 25 | `tests/test_autoresearch_llm.py` | CREATE | LLM backend + prompt + agent tests (mocked) |

**Total: 23 new files, 2 modified files. No existing src files are touched.**

---

## 3. Dependency Policy

**No new pip dependencies.** All LLM backends use `requests` (already in
`requirements.txt`). The runner invokes existing training CLIs via `subprocess`.
Polars is already in `requirements.txt` for ledger DataFrames.

---

## 4. Dependency Graph

```
configs/benchmark_household.json ----------------------------------------+
configs/benchmark_aemo.json ---------------------------------------------+|
                                                                        ||
src/eval_common.py  <-- src/decision_transformer.py                    ||
    |                                                                   ||
    +-- src/eval_household.py  <-- src/decision.py,                    ||
    |                               src/helper.py,                     ||
    |                               src/EnergySimEnv.py                ||
    |                                                                   ||
    +-- src/eval_aemo.py  <-- src/decision.py,                         ||
                               src/helper.py,                          ||
                               src/aemo_notebook_utils.py,             ||
                               src/AEMOBatteryEnv.py                   ||
                                                                       ||
src/autoresearch/config_utils.py   (no autoresearch deps)              ||
src/autoresearch/ledger.py         (stdlib + polars only)              ||
src/autoresearch/stage_a.py        (stdlib only)                       ||
src/autoresearch/stage_b.py  <--  eval_household.py, eval_aemo.py     ||
src/autoresearch/runner.py   <--  all of the above  <------------------++
                                   + training CLIs (subprocess)

src/autoresearch/llm_backend.py    (requests only)
src/autoresearch/prompts.py  <--  config_utils.py
src/autoresearch/agent.py    <--  llm_backend, prompts, runner, ledger
src/autoresearch/cli.py      <--  agent + all autoresearch modules
```

---

## 5. File-by-File Specification

---

### File 1: `configs/benchmark_household.json` (CREATE)

**Purpose:** Single canonical Household benchmark definition. All autoresearch
runs reference this file. It must never be mutated by the agent.

```json
{
  "environment": "household",
  "data_dir": "data/household/logs",
  "train_patterns": ["train_ep_*.parquet"],
  "val_patterns": ["val_ep_*.parquet"],
  "test_patterns": ["test_ep_*.parquet"],
  "state_dim": 12,
  "act_dim": 1,
  "discount": 0.99,
  "max_timestep": 17567,
  "eval_episodes": 10,
  "eval_seed": 42,
  "primary_metric": "mean_reward",
  "higher_is_better": true,
  "guardrails": {
    "max_avg_degradation_per_episode": 0.05,
    "max_var_5": -9500.0,
    "max_deg_incident_rate": 0.1
  },
  "env_kwargs": {
    "battery_capacity": 13.5,
    "solar_panel_size": 5.0,
    "degradation_mode": "rainflow"
  },
  "stage_a_timeout": 600,
  "stage_b_timeout": 3600
}
```

**Depends on:** Existing data in `data/household/logs/`, `SolarBatteryEnv` in
`src/EnergySimEnv.py`, `Agent` in `src/decision.py`.

---

### File 2: `configs/benchmark_aemo.json` (CREATE)

**Purpose:** Single canonical AEMO benchmark definition. Frozen for autoresearch.

```json
{
  "environment": "aemo",
  "dataset_path": "data/aemo_dt/aemo_dt_dataset.parquet",
  "manifest_path": "data/aemo_dt/manifest.json",
  "region": "SA1",
  "train_window": {"start": "2021-01-01", "end": "2022-12-31"},
  "val_window":   {"start": "2023-01-01", "end": "2023-06-30"},
  "test_window":  {"start": "2023-07-01", "end": "2023-12-31"},
  "state_dim": 18,
  "act_dim": 3,
  "action_mode": "multi_market",
  "discount": 0.99,
  "max_timestep": 157680,
  "step_duration": 5,
  "episode_hours": 24,
  "eval_episodes": 5,
  "eval_seed": 42,
  "primary_metric": "avg_profit_per_episode",
  "higher_is_better": true,
  "guardrails": {
    "max_avg_degradation_cost_per_episode": 50.0,
    "max_deg_incident_rate": 0.05,
    "max_var_5": -5000.0,
    "max_cvar_5": -6000.0
  },
  "battery_variants": [
    {"capacity": 100.0, "power": 50.0, "initial_soc": 0.5}
  ],
  "degradation_mode": "real_world",
  "degradation_chemistry": "LFP",
  "degradation_temperature": 25.0,
  "scenario_kwargs": {
    "cache_dir": "data/aemo_cache"
  },
  "stage_a_timeout": 600,
  "stage_b_timeout": 7200
}
```

**Depends on:** `data/aemo_dt/`, `AEMOBatteryTradingEnv` in `src/AEMOBatteryEnv.py`,
`AEMOAgent` in `src/decision.py`.

---

### File 3: `src/eval_common.py` (CREATE)

**Purpose:** Shared utilities for both evaluation scripts. Avoids code duplication
between `eval_household.py` and `eval_aemo.py`.

#### `EvalSummary` dataclass

```python
@dataclass
class EvalSummary:
    primary_metric_name: str
    primary_metric_value: float | None
    guardrails_passed: bool
    guardrail_details: dict   # {metric: {"value": v, "threshold": t, "passed": bool}}
    model_path: str
    benchmark_path: str
    timestamp: str            # ISO 8601
```

#### Functions

```python
def load_benchmark(path: str) -> dict:
    """Load and validate benchmark JSON; resolve relative data paths to absolute."""

def load_dt_model(model_path: str, model_config: dict, device: str):
    """
    Instantiate DecisionTransformer with model_config kwargs,
    load state_dict from model_path, move to device, set eval mode.
    """

def read_return_scale(model_path: str, cli_override: float | None) -> float:
    """
    1. If cli_override is not None, return cli_override.
    2. Try {model_path}.meta.json -> read "return_scale" key.
    3. Fall back to 1.0.
    """

def check_guardrails(metrics: dict, guardrails: dict) -> dict:
    """
    Compare each guardrail key against the corresponding metrics value.
    'max_*' keys: metrics[key] must be <= threshold.
    Returns {"passed": bool, "details": {key: {"value": v, "threshold": t, "passed": bool}}}
    Keys missing from metrics are treated as a violation.
    """

def write_eval_outputs(output_dir: str, metrics: dict, summary: EvalSummary) -> None:
    """
    Write {output_dir}/eval_metrics.json (full metrics dict).
    Write {output_dir}/eval_summary.json (EvalSummary as dict).
    Creates output_dir if it does not exist.
    """
```

**Depends on:** `src/decision_transformer.py` for the model class.

---

### File 4: `src/eval_household.py` (CREATE)

**Purpose:** CLI evaluation script for Household DT. Loads a trained DT, runs
held-out episodes, computes metrics, writes machine-readable JSON output.

#### CLI arguments (argparse)

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmark` | required | Path to `configs/benchmark_household.json` |
| `--model-path` | required | Path to trained DT `.pt` state_dict file |
| `--model-config` | `None` | Path to model kwargs JSON |
| `--rtg-value` | from benchmark | Return-to-go prompt value |
| `--return-scale` | from sidecar/1.0 | Return scale factor |
| `--output-dir` | required | Directory to write evaluation results |
| `--device` | `"cpu"` | Compute device |
| `--num-workers` | `1` | Parallel episode workers |
| `--save-episodes` | `False` | Save per-episode parquet files |

#### Main flow

1. `load_benchmark(args.benchmark)` -> extract `env_kwargs`, `test_patterns`, `eval_seed`.
2. `load_dt_model(args.model_path, model_config, args.device)`.
3. `read_return_scale(args.model_path, args.return_scale)`.
4. Create test environments using `SolarBatteryEnv(**benchmark["env_kwargs"])`.
5. Instantiate `Agent(algorithm='dt', model=model, rtg_value=rtg_value, dt_gamma=benchmark["discount"])`.
6. Run held-out episodes via `run_episodes_parallel()` from `src/decision.py`.
7. Compute metrics via `evaluate_experiment_logs()` from `src/helper.py`.
8. `check_guardrails(metrics, benchmark["guardrails"])`.
9. Build `EvalSummary` and call `write_eval_outputs(args.output_dir, metrics, summary)`.
10. Optionally write per-episode parquet files to `{output_dir}/episode_logs/`.
11. **Exit code:** 0 on success, 1 on crash/model load failure.
12. Expose `main(argv: list[str] | None = None)` so `stage_b.py` can import and call it.

**Reuses:** `Agent`, `run_episodes_parallel` from `src/decision.py`;
`evaluate_experiment_logs` from `src/helper.py`;
`SolarBatteryEnv` from `src/EnergySimEnv.py`;
all utilities from `src/eval_common.py`.

---

### File 5: `src/eval_aemo.py` (CREATE)

**Purpose:** CLI evaluation script for AEMO DT. Same contract as `eval_household.py`
but for AEMO environments.

#### CLI arguments (argparse)

Same as `eval_household.py` plus:

| Argument | Default | Description |
|----------|---------|-------------|
| `--cache-dir` | from benchmark | AEMO data cache directory |

#### Main flow

1. `load_benchmark(args.benchmark)` -> extract `test_window`, `battery_variants`,
   `degradation_*`, `scenario_kwargs`, `eval_seed`.
2. `load_dt_model(...)` and `read_return_scale(...)`.
3. Fetch/load AEMO test-window data via `fetch_and_preprocess_aemo_data()` from
   `src/aemo_notebook_utils.py` (with caching).
4. Create test environments using `make_aemo_env_fns()` from
   `src/aemo_notebook_utils.py` with `battery_variants` and degradation settings.
5. Instantiate `AEMOAgent(algorithm='dt', model=model, rtg_value=rtg_value, ...)`.
6. Run episodes via `run_episodes_parallel()` from `src/decision.py`.
7. Compute metrics via `evaluate_experiment_logs()` from `src/helper.py`.
8. `check_guardrails(metrics, benchmark["guardrails"])`.
9. `write_eval_outputs(args.output_dir, metrics, summary)`.
10. **Exit code:** 0 on success, 1 on crash.
11. Expose `main(argv: list[str] | None = None)` for import by `stage_b.py`.

**Reuses:** `AEMOAgent`, `run_episodes_parallel` from `src/decision.py`;
`evaluate_experiment_logs` from `src/helper.py`;
`fetch_and_preprocess_aemo_data`, `make_aemo_env_fns` from `src/aemo_notebook_utils.py`;
`AEMOBatteryTradingEnv` from `src/AEMOBatteryEnv.py`;
all utilities from `src/eval_common.py`.

---

### File 6: `src/autoresearch/__init__.py` (CREATE)

```python
"""LLM-driven autoresearch loop for Decision Transformer hyperparameter search."""

from .agent import AutoresearchAgent
from .ledger import ExperimentLedger, LedgerEntry
from .runner import AutoresearchRunner
from .stage_a import StageAScreen
from .stage_b import StageBEvaluator
from .llm_backend import LlamaCppBackend, OllamaBackend, OpenAIBackend

__all__ = [
    "AutoresearchAgent",
    "ExperimentLedger",
    "LedgerEntry",
    "AutoresearchRunner",
    "StageAScreen",
    "StageBEvaluator",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAIBackend",
]
```

---

### File 7: `src/autoresearch/config_utils.py` (CREATE)

**Purpose:** Load, validate, and diff training configs. Enforces the mutable
surface -- only whitelisted keys can be changed by the agent.

#### Contents

```python
# Keys the agent may change in v1
ALLOWED_MUTABLE_KEYS_V1: frozenset[str] = frozenset({
    # Model architecture
    "n_block", "h_dim", "n_heads", "drop_p",
    "context_len", "rope_enabled", "rope_base", "rope_max_position",
    # Training
    "batch_size", "lr", "epochs", "return_scale",
    "action_loss_weight", "state_loss_weight", "return_loss_weight",
    "weight_decay",
    # RTG prompting
    "rtg_value", "recommended_rtg_percentile",
    # AEMO-specific (ignored for household benchmark)
    "action_mode", "degradation_mode", "degradation_chemistry",
    "step_duration_hours",
})

# Keys that are NEVER mutable -- frozen per benchmark
FROZEN_KEYS: frozenset[str] = frozenset({
    "state_dim", "act_dim", "max_timestep", "discount",
    "data_dir", "dataset_path", "env_kwargs", "eval_episodes", "eval_seed",
    "primary_metric", "guardrails", "stage_a_timeout", "stage_b_timeout",
})


def load_config(path: str) -> dict:
    """Load JSON config, raise ValueError if required keys are missing."""


def diff_configs(baseline: dict, candidate: dict) -> dict:
    """
    Return {key: {"old": baseline[key], "new": candidate[key]}}
    for every key that differs. Includes added/removed keys.
    """


def validate_mutable_surface(
    candidate: dict,
    allowed_keys: frozenset = ALLOWED_MUTABLE_KEYS_V1,
) -> None:
    """
    Raise ValueError listing any key in candidate not in allowed_keys.
    Hard guardrail -- enforced before any training is started.
    """


def build_training_cli_args(
    config: dict,
    benchmark: dict,
    output_dir: str,
    script: str,
    epochs_override: int | None = None,
) -> list[str]:
    """
    Convert config dict + benchmark into a CLI arg list for subprocess.run().
    Maps config keys to flag names (e.g. n_block -> --n-block).
    epochs_override lets Stage A force 1 epoch without mutating the config.
    """
```

---

### File 8: `src/autoresearch/ledger.py` (CREATE)

**Purpose:** Append-only JSONL experiment ledger. Records every autoresearch run.

#### `LedgerEntry` dataclass (all fields)

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | UUID string |
| `timestamp` | `str` | ISO 8601 |
| `environment` | `str` | `"household"` or `"aemo"` |
| `benchmark_path` | `str` | Path to the frozen benchmark JSON |
| `benchmark_sha256` | `str` | sha256 hex of benchmark file (integrity check) |
| `candidate_config` | `dict` | Full config for this run |
| `baseline_config` | `dict` | Config being compared against |
| `diff_from_baseline` | `dict` | `{key: {"old": v1, "new": v2}}` |
| `training_summary` | `dict` | `{final_train_loss, final_val_loss, epochs_completed, divergence_ratio, crashed, checkpoint_path, model_path, loss_csv_path}` |
| `stage_a_passed` | `bool` | Result of Stage A screening |
| `stage_a_reason` | `str` | Why it passed/failed |
| `evaluation_summary` | `dict` or `None` | Full `eval_metrics.json` content |
| `eval_summary` | `dict` or `None` | `eval_summary.json` content |
| `stage_b_passed` | `bool` or `None` | Result of Stage B evaluation |
| `stage_b_reason` | `str` | Keep/discard reason |
| `decision` | `str` | `"keep"`, `"discard"`, `"crash"`, or `"stage_a_reject"` |
| `artifact_dir` | `str` | Path to all run artifacts |

#### `ExperimentLedger` class

```python
class ExperimentLedger:
    def __init__(self, path: str | Path): ...

    def _load(self) -> None: ...
    def append(self, entry: LedgerEntry) -> None: ...
    def last_n(self, n: int) -> list[LedgerEntry]: ...
    def current_best(self, environment: str) -> LedgerEntry | None:
        """Return the most recent 'keep' entry for the given environment."""
    def format_history(self, last_n: int = 10) -> str:
        """
        Human-readable lines for LLM prompt. Example:
        '  - Run 7: context_len=60->80, h_dim=128->256 -> mean_reward=-2380.5, KEPT'
        """
    def next_run_id(self) -> str: ...
    def summary_dataframe(self) -> "pl.DataFrame": ...
    def to_tsv(self, path: str) -> None: ...
```

**Format:** JSONL (one JSON object per line), append-only, git-friendly.

**CLI summary mode:**
`python -m src.autoresearch.ledger --summary <path>` prints a formatted table.

**Depends on:** stdlib + polars.

---

### File 9: `src/autoresearch/stage_a.py` (CREATE)

**Purpose:** Cheap screening of training artifacts before expensive evaluation.

#### `StageAScreen` class

```python
class StageAScreen:
    def __init__(
        self,
        max_divergence_ratio: float = 4.0,
        max_final_val_loss: float | None = None,
        require_checkpoint: bool = True,
    ): ...

    def screen(self, training_summary: dict) -> tuple[bool, str]:
        """
        Returns (passed, reason).

        Check 1: training_summary["crashed"] is True -> reject.
        Check 2: model file at training_summary["model_path"] missing -> reject.
        Check 3: checkpoint missing (if require_checkpoint) -> reject.
        Check 4: divergence_ratio > max_divergence_ratio -> reject.
        Check 5: final_val_loss > max_final_val_loss (if set) -> reject.
        Check 6: final_val_loss is NaN or Inf -> reject.
        All pass -> return (True, "ok").
        """
```

**How `training_summary` is built:** The runner parses the loss CSV written by
`train_decision_transformer()`, checks for model/checkpoint files on disk, and
computes `divergence_ratio = final_train_loss / initial_train_loss`.

**Depends on:** `os.path.exists`, `math.isnan`, `math.isinf` only.

---

### File 10: `src/autoresearch/stage_b.py` (CREATE)

**Purpose:** Run the full held-out evaluation and make the keep/discard decision.

#### `StageBEvaluator` class

```python
class StageBEvaluator:
    def __init__(self, benchmark_path: str, environment: str): ...

    def evaluate(
        self,
        model_path: str,
        model_config: dict,
        rtg_value: float,
        return_scale: float,
        output_dir: str,
        device: str = "cpu",
    ) -> dict:
        """
        Imports eval_household.main() or eval_aemo.main() and calls it
        in-process with the given parameters.
        Returns the parsed eval_summary.json content as a dict.
        """

    def compare(
        self,
        candidate_summary: dict,
        baseline_summary: dict | None,
    ) -> tuple[str, str]:
        """
        Returns (decision, reason).

        No baseline       -> ("keep",    "first run, no baseline to compare")
        Guardrail fail    -> ("discard", "guardrail violation: {details}")
        Metric improved   -> ("keep",    "improved {metric} from {old} to {new}")
        No improvement    -> ("discard", "no improvement: {metric} {new} <= {old}")
        """
```

**Design note:** `evaluate()` calls eval scripts' `main(argv)` in-process
(not subprocess) to avoid Python interpreter startup overhead, while the scripts
remain independently runnable from the command line.

**Depends on:** `src/eval_household.py`, `src/eval_aemo.py`, `src/eval_common.py`.

---

### File 11: `src/autoresearch/runner.py` (CREATE)

**Purpose:** Orchestrates a single train -> Stage A -> Stage B -> keep/discard ->
ledger cycle. Callable programmatically by the agent or directly via CLI.

#### `AutoresearchRunner` class

```python
class AutoresearchRunner:
    def __init__(
        self,
        environment: str,
        benchmark_path: str,
        output_dir: str = "eval_output/autoresearch",
        ledger_path: str = "eval_output/autoresearch/ledger.jsonl",
        device: str = "cpu",
        use_docker: bool = False,
        stage_a_screen: StageAScreen | None = None,
    ): ...

    def run_candidate(
        self,
        candidate_config: dict,
        baseline_config: dict,
    ) -> LedgerEntry:
        """
        Full single-candidate cycle:

        1. validate_mutable_surface(candidate_config).
        2. Create artifact_dir = {output_dir}/{run_id}/.
        3. Stage A training (1 epoch):
           a. Write temp model kwargs JSON.
           b. build_training_cli_args(..., epochs_override=1).
           c. subprocess.run() (or docker-compose run if use_docker).
           d. Parse loss CSV -> build training_summary.
           e. StageAScreen.screen(training_summary).
           f. If rejected -> append ledger decision="stage_a_reject" -> return.
        4. Stage B training (full epochs):
           a. build_training_cli_args(...) with full epoch count.
           b. subprocess.run().
           c. Parse training artifacts -> update training_summary.
        5. Stage B evaluation:
           a. StageBEvaluator.evaluate(model_path, ...).
           b. StageBEvaluator.compare(candidate_summary, baseline_summary).
        6. Append LedgerEntry to ledger.
        7. Write {artifact_dir}/run_summary.json.
        8. Return LedgerEntry.
        """
```

#### Output directory layout

```
eval_output/autoresearch/
+-- ledger.jsonl
+-- {run_id_1}/
|   +-- candidate_config.json
|   +-- model_kwargs.json
|   +-- stage_a/
|   |   +-- loss.csv
|   |   +-- checkpoint_epoch1.pt
|   +-- stage_b/
|   |   +-- loss.csv
|   |   +-- model_final.pt
|   |   +-- eval_metrics.json
|   |   +-- eval_summary.json
|   +-- run_summary.json
+-- {run_id_2}/
    +-- ...
```

#### Standalone CLI

```
python -m src.autoresearch.runner \
  --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline.json \
  --candidate-config configs/candidate.json \
  --output-dir eval_output/autoresearch \
  [--device cuda] [--docker] [--skip-training]
```

**Depends on:** `config_utils`, `ledger`, `stage_a`, `stage_b`, `eval_common`,
existing training CLIs.

---

### File 12: `src/autoresearch/llm_backend.py` (CREATE)

**Purpose:** LLM backend abstraction. All implementations POST to an
OpenAI-compatible `/v1/chat/completions` endpoint, so llama.cpp, Ollama,
vLLM, LM Studio, and OpenAI all share one HTTP client.

#### Abstract base class

```python
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text completion. Raise LLMBackendError on failure."""
```

#### Concrete classes

| Class | Default endpoint | Notes |
|-------|-----------------|-------|
| `LlamaCppBackend` | `http://localhost:8080/v1` | **Default. Zero-config for local.** |
| `OllamaBackend` | `http://localhost:11434/v1` | Same protocol. |
| `OpenAIBackend` | `https://api.openai.com/v1` | Requires `api_key`. |

#### Constructor signature (all classes)

```python
def __init__(
    self,
    endpoint: str = <class_default>,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 512,
    api_key: str | None = None,
    timeout: int = 120,
):
```

#### `complete()` implementation pattern

POST body:

```json
{
  "messages": [
    {"role": "system", "content": "<system_prompt>"},
    {"role": "user",   "content": "<user_prompt>"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "model": "<model_name_if_set>"
}
```

Headers: `Content-Type: application/json`, `Authorization: Bearer <api_key>` (if set).

URL: `{endpoint}/chat/completions`

Return: `response.json()["choices"][0]["message"]["content"]`

#### Error handling

- `ConnectionError` -> `LLMBackendError`:
  `"Cannot reach LLM server at {url}. Is llama-server / ollama running?"`
- HTTP 4xx/5xx -> `LLMBackendError(status_code, body)`.
- JSON decode failure -> `LLMBackendError`.

---

### File 13: `src/autoresearch/prompts.py` (CREATE)

**Purpose:** System/user prompt templates and JSON response parser.

#### Templates

**System prompt:** Enumerates all keys in `ALLOWED_MUTABLE_KEYS_V1` with their
type, valid range, and description. Instructs the LLM to respond with ONLY a
JSON object containing the keys to change.

**User prompt:** Shows the current best config, the best metric value, and the
last N ledger entries formatted as one-line history (from `ledger.format_history()`).
Ends with: `"Propose a new configuration that improves {metric_name}{constraint_clause}."`

#### Functions

```python
def build_system_prompt(allowed_keys: frozenset, mutable_params: dict) -> str:
    """Fill system prompt template with the mutable keys description table."""

def build_user_prompt(
    best_config: dict,
    best_metric: float | str,
    metric_name: str,
    history_lines: str,
    constraint_clause: str = "",
) -> str:
    """Fill user prompt template."""

def parse_llm_response(raw: str, allowed_keys: frozenset) -> dict:
    """
    1. Find the first {...} block using brace-matching (not regex).
    2. json.loads() the extracted block.
    3. Drop unknown keys silently.
    4. Attempt type cast for wrong-type values; drop on failure.
    5. Return validated dict of only the changed keys.
    6. Raise ParseError on total parse failure.
    """
```

---

### File 14: `src/autoresearch/agent.py` (CREATE)

**Purpose:** LLM-driven outer loop. Calls the runner with LLM-proposed configs.

#### `AutoresearchAgent` class

```python
class AutoresearchAgent:
    def __init__(
        self,
        backend: LLMBackend,
        runner: AutoresearchRunner,
        ledger: ExperimentLedger,
        environment: str,
        primary_metric: str = "mean_reward",
        higher_is_better: bool = True,
        max_llm_retries: int = 3,
        history_window: int = 10,
        constraint_clause: str = "",
    ): ...

    def propose(self) -> dict:
        """
        Ask LLM to propose a config diff.
        Reads current best from ledger, formats history, builds prompts,
        calls backend.complete(), parses response.
        Retries up to max_llm_retries on ParseError.
        Raises AutoresearchError after exhausting retries.
        """

    def step(self) -> LedgerEntry:
        """
        One full propose -> run_candidate cycle.
        1. propose() -> candidate_diff (only changed keys)
        2. Merge diff with current best config to form full candidate_config.
        3. runner.run_candidate(candidate_config, baseline_config) -> LedgerEntry
        4. Return LedgerEntry.
        """

    def run(self, iterations: int) -> None:
        """Run the loop for N iterations, printing progress to stdout."""
```

---

### File 15: `src/autoresearch/cli.py` (CREATE)

**Purpose:** Unified CLI entrypoint for both manual and agent modes.

```
python -m src.autoresearch [OPTIONS]
```

#### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `agent` | `agent` (LLM loop) or `manual` (single run, no LLM) |
| `--environment` | required | `household` or `aemo` |
| `--benchmark` | required | Path to frozen benchmark JSON |
| `--baseline-config` | required | Path to baseline training config JSON |
| `--candidate-config` | *(manual only)* | Path to candidate config JSON |
| `--iterations` | `20` | Loop iterations (agent mode) |
| `--llm-backend` | `llamacpp` | `llamacpp`, `ollama`, or `openai` |
| `--llm-endpoint` | *(class default)* | Override LLM server URL |
| `--llm-model` | `""` | Model name (required for Ollama/OpenAI) |
| `--llm-api-key` | `$OPENAI_API_KEY` | API key (OpenAI only) |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--max-tokens` | `512` | LLM max response tokens |
| `--primary-metric` | from benchmark | Metric to optimise |
| `--constraint` | `""` | Optional constraint clause for LLM prompt |
| `--ledger-path` | `eval_output/autoresearch/ledger.jsonl` | Ledger path |
| `--output-dir` | `eval_output/autoresearch` | Base artifact directory |
| `--device` | `cpu` | `cpu`, `cuda`, or `auto` |
| `--docker` | `False` | Run training inside Docker |

**Manual mode:** Runs one candidate config through the full pipeline without
any LLM interaction. Useful for testing the pipeline or running hand-crafted
configs.

**Agent mode:** Runs the LLM-driven loop for `--iterations` cycles.

---

### File 16: `src/autoresearch/__main__.py` (CREATE)

```python
from .cli import main
main()
```

---

### File 17: `docker-compose.yml` (MODIFY)

Add the `autoresearch-train` service alongside the existing `app` service:

```yaml
  autoresearch-train:
    build: .
    working_dir: /code/src
    volumes:
      - ".:/code"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    # No default command -- called with:
    # docker-compose run autoresearch-train pretrain_decision_transformer.py --arg val ...
    entrypoint: ["python"]
```

The runner uses this service when `--docker` is passed.

---

### File 18: `docs/AUTORESEARCH.md` (CREATE)

Full user-facing documentation. Contents:

1. What autoresearch is and is not.
2. Architecture diagram.
3. How to run a single manual candidate cycle.
4. How to run the LLM-driven agent loop (Local LLM Quick Start from section 6).
5. How to freeze a new benchmark.
6. Mutable surface specification (`ALLOWED_MUTABLE_KEYS_V1` table).
7. Ledger format specification (JSONL schema, all field definitions).
8. Guardrail definitions for Household and AEMO.
9. Extending to population search (future, v2).

---

### File 19: `README.md` (MODIFY)

Add after existing content:

- Brief description (constrained optimization loop over DT training configs).
- Link to `docs/AUTORESEARCH.md`.
- Quick-start: run one manual candidate cycle (two commands).

---

### File 20: `tests/test_autoresearch_ledger.py` (CREATE)

- `test_ledger_entry_serialization` -- `LedgerEntry` round-trips through JSON.
- `test_ledger_append_writes_jsonl` -- `append()` produces valid JSONL.
- `test_ledger_load_roundtrips` -- entries survive write->load cycle.
- `test_ledger_current_best_returns_kept_entry`.
- `test_ledger_current_best_none_when_empty`.
- `test_ledger_summary_dataframe_schema` -- expected column names present.

---

### File 21: `tests/test_autoresearch_stage_a.py` (CREATE)

- `test_stage_a_rejects_crashed_run`.
- `test_stage_a_rejects_missing_model_file`.
- `test_stage_a_rejects_missing_checkpoint` (`require_checkpoint=True`).
- `test_stage_a_rejects_high_divergence`.
- `test_stage_a_rejects_nan_val_loss`.
- `test_stage_a_accepts_valid_summary`.

---

### File 22: `tests/test_autoresearch_stage_b.py` (CREATE)

- `test_compare_no_baseline_returns_keep`.
- `test_compare_improvement_returns_keep`.
- `test_compare_no_improvement_returns_discard`.
- `test_compare_guardrail_violation_discards_even_if_metric_improved`.

---

### File 23: `tests/test_autoresearch_config_utils.py` (CREATE)

- `test_diff_configs_detects_changed_keys`.
- `test_diff_configs_detects_added_keys`.
- `test_validate_mutable_surface_rejects_frozen_key`.
- `test_validate_mutable_surface_accepts_allowed_keys`.
- `test_build_training_cli_args_produces_correct_flags`.
- `test_build_training_cli_args_respects_epochs_override`.

---

### File 24: `tests/test_eval_common.py` (CREATE)

- `test_check_guardrails_passes_when_all_ok`.
- `test_check_guardrails_fails_on_violation`.
- `test_check_guardrails_missing_metric_is_violation`.
- `test_load_dt_model_with_synthetic_state_dict`.
- `test_write_eval_outputs_writes_correct_json_schema`.
- `test_read_return_scale_from_sidecar`.
- `test_read_return_scale_falls_back_to_cli`.

---

### File 25: `tests/test_autoresearch_llm.py` (CREATE)

**LLM backend tests (all mocked -- no real LLM required):**

- `test_llamacpp_backend_posts_to_correct_url`.
- `test_ollama_backend_posts_to_correct_url`.
- `test_openai_backend_includes_auth_header`.
- `test_backend_connection_error_raises_clear_message`.
- `test_backend_http_500_raises_llm_backend_error`.
- `test_backend_json_decode_failure_raises_llm_backend_error`.

**Prompt tests:**

- `test_build_system_prompt_contains_all_mutable_keys`.
- `test_build_user_prompt_contains_history_lines`.
- `test_build_user_prompt_contains_constraint_clause`.
- `test_parse_clean_json` -- `'{"lr": 5e-5}'` -> `{"lr": 5e-5}`.
- `test_parse_json_with_markdown_fences`.
- `test_parse_json_with_preamble_text`.
- `test_parse_nested_braces_extracts_correctly`.
- `test_parse_invalid_raises_parse_error`.
- `test_parse_drops_unknown_keys`.

**Agent tests (mocked LLM + mocked runner):**

- `test_agent_step_keep` -- mock LLM diff + mock runner good metrics -> `decision="keep"`.
- `test_agent_step_discard` -- mock runner bad metrics -> `decision="discard"`.
- `test_agent_retries_on_parse_error` -- LLM fails twice then succeeds -> entry recorded.
- `test_agent_raises_after_max_retries` -- LLM always returns garbage -> `AutoresearchError`.

---

## 6. Local LLM Quick Start

### Option A: llama.cpp (recommended, zero-config)

```bash
# Download a GGUF model (example: Qwen2.5-32B quantized)
wget https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/qwen2.5-32b-instruct-q4_k_m.gguf

# Start llama-server
./llama-server -m qwen2.5-32b-instruct-q4_k_m.gguf --port 8080 --ctx-size 2048 --n-gpu-layers 99

# Run autoresearch (default backend connects to localhost:8080)
python -m src.autoresearch \
  --mode agent --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --iterations 20
```

### Option B: Ollama

```bash
ollama serve && ollama pull qwen2.5:32b

python -m src.autoresearch \
  --mode agent --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --llm-backend ollama --llm-model qwen2.5:32b --iterations 20
```

### Option C: OpenAI (cloud, opt-in)

```bash
export OPENAI_API_KEY=sk-...

python -m src.autoresearch \
  --mode agent --environment aemo \
  --benchmark configs/benchmark_aemo.json \
  --baseline-config configs/baseline_aemo.json \
  --llm-backend openai --llm-model gpt-4o --iterations 10
```

### Manual mode (no LLM, single candidate)

```bash
python -m src.autoresearch \
  --mode manual --environment household \
  --benchmark configs/benchmark_household.json \
  --baseline-config configs/baseline_household.json \
  --candidate-config configs/my_candidate.json
```

---

## 7. Recommended Implementation Order

1. `src/eval_common.py` -- no autoresearch deps.
2. `configs/benchmark_household.json`.
3. `src/eval_household.py`.
4. `tests/test_eval_common.py`.
5. `configs/benchmark_aemo.json`.
6. `src/eval_aemo.py`.
7. `src/autoresearch/config_utils.py`.
8. `src/autoresearch/ledger.py`.
9. `tests/test_autoresearch_ledger.py`.
10. `src/autoresearch/stage_a.py`.
11. `tests/test_autoresearch_stage_a.py`.
12. `src/autoresearch/stage_b.py`.
13. `tests/test_autoresearch_stage_b.py`.
14. `src/autoresearch/runner.py`.
15. `tests/test_autoresearch_config_utils.py`.
16. `src/autoresearch/llm_backend.py`.
17. `src/autoresearch/prompts.py`.
18. `src/autoresearch/agent.py`.
19. `tests/test_autoresearch_llm.py`.
20. `src/autoresearch/cli.py` + `__main__.py` + `__init__.py`.
21. `docker-compose.yml` modification.
22. `docs/AUTORESEARCH.md` + `README.md` modification.

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No new pip deps** | `requests` already present. Keeps install simple. |
| **OpenAI-compatible API** | llama.cpp, Ollama, vLLM, LM Studio all speak this. One HTTP client handles all. |
| **JSON-only LLM output** | Keeps task trivial for <=30B models. No code generation. |
| **Frozen benchmark JSONs** | Evaluation target never drifts during a search run. |
| **Two-stage gate** | Stage A (1-epoch screen) rejects diverging configs before expensive full training. |
| **JSONL ledger with full schema** | Append-only, human-readable, git-friendly, no DB dependency. |
| **Subprocess for training** | Existing CLIs handle device/checkpointing; no need to refactor. |
| **Import (not subprocess) for eval** | Avoids Python startup overhead; eval scripts expose `main(argv)`. |
| **Mutable surface enforced in code** | `validate_mutable_surface()` is a hard error, not advisory. |
| **Purely additive** | Zero changes to existing `src/` files. Safe to merge independently. |
| **Local LLM as default** | `LlamaCppBackend` at `localhost:8080` is the zero-config default. Cloud is opt-in. |

---

## 9. Integration Points (no changes to existing files)

| Existing Component | How Autoresearch Uses It |
|--------------------|--------------------------|
| `pretrain_decision_transformer.py` | Invoked via subprocess with candidate CLI args |
| `pretrain_aemo_decision_transformer.py` | Same, for AEMO benchmarks |
| `Agent` / `AEMOAgent` (`decision.py`) | Used by eval scripts for episode rollouts |
| `evaluate_experiment_logs` (`helper.py`) | Used by eval scripts to compute metrics |
| `SolarBatteryEnv` (`EnergySimEnv.py`) | Used by `eval_household.py` for environments |
| `AEMOBatteryTradingEnv` (`AEMOBatteryEnv.py`) | Used by `eval_aemo.py` for environments |
| `fetch_and_preprocess_aemo_data` (`aemo_notebook_utils.py`) | Used by `eval_aemo.py` |
| `make_aemo_env_fns` (`aemo_notebook_utils.py`) | Used by `eval_aemo.py` |
| `TrajectoryDataset` (`transformer_training.py`) | Used by training CLIs (unchanged) |
| `configs/aemo_decision_transformer_model_kwargs.json` | Base model kwargs for runner |

---

## 10. Success Criteria

The implementation is complete when:

- [ ] `python -m src.autoresearch --mode manual --environment household --benchmark configs/benchmark_household.json --baseline-config B --candidate-config C` trains and evaluates, printing a `LedgerEntry` JSON.
- [ ] `python -m src.autoresearch --mode agent --environment household --benchmark configs/benchmark_household.json --baseline-config B --llm-backend llamacpp --iterations 3` runs 3 cycles with a local LLM, producing `ledger.jsonl` with 3 entries.
- [ ] `--llm-backend ollama --llm-model qwen2.5:32b` and `--llm-backend openai --llm-model gpt-4o` both work.
- [ ] `python -m src.autoresearch.runner --environment household ... --candidate-config C` works standalone (no LLM).
- [ ] `python -m src.eval_household --benchmark configs/benchmark_household.json --model-path M --output-dir O` writes `eval_metrics.json` and `eval_summary.json`.
- [ ] `python -m src.eval_aemo --benchmark configs/benchmark_aemo.json --model-path M --output-dir O` works equivalently.
- [ ] `python -m src.autoresearch.ledger --summary eval_output/autoresearch/ledger.jsonl` prints a formatted summary table.
- [ ] All test functions across `tests/test_autoresearch_*.py` and `tests/test_eval_common.py` pass.
- [ ] No existing tests are broken (`pytest` from repo root).
