# Autoresearch for Battery DT — File-by-File Implementation Handoff

> **Purpose:** This document is the complete implementation spec for an LLM-driven
> autoresearch loop that automatically searches for better Decision Transformer
> hyperparameters. It is written so that a coding agent (or human) can implement
> each file top-to-bottom without further clarification.

---

## 0. Context & Motivation

The codebase already has:

| What | Where |
|------|-------|
| DT model | `src/decision_transformer.py` |
| DT training CLI (household) | `src/pretrain_decision_transformer.py` |
| DT training CLI (AEMO) | `src/pretrain_aemo_decision_transformer.py` |
| Episode rollout agents | `src/decision.py` (`Agent`, `AEMOAgent`) |
| Evaluation metrics | `src/helper.py` → `evaluate_experiment_logs()` |
| Trajectory dataset | `src/transformer_training.py` → `TrajectoryDataset` |
| Model kwargs configs | `configs/aemo_decision_transformer_model_kwargs.json` |

**The gap:** A human must manually write configs, run training, roll out episodes,
read metrics, decide what to try next, and repeat. The autoresearch agent
automates this loop: **propose config → train → evaluate → record → repeat.**

The LLM is the *mutation proposer* — it outputs structured JSON configs, never
code. This keeps the task within the capability of local ≤ 30B models.

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   AutoresearchAgent (agent.py)                   │
│                                                                  │
│  ┌──────────┐   ┌────────────┐   ┌─────────────────────────┐   │
│  │  Ledger  │──▶│   Prompt   │──▶│     LLM Backend         │   │
│  │ (JSONL)  │   │   Builder  │   │   (llm_backend.py)      │   │
│  └──────────┘   │(prompts.py)│   │                         │   │
│       ▲         └────────────┘   │  ┌───────────────────┐  │   │
│       │                          │  │ LlamaCppBackend   │  │   │
│       │         ┌────────────┐   │  │ (default, local)  │  │   │
│       │         │ JSON Parse │◀──│  ├───────────────────┤  │   │
│       │         │ + Validate │   │  │ OllamaBackend     │  │   │
│       │         └─────┬──────┘   │  ├───────────────────┤  │   │
│       │               │          │  │ OpenAIBackend     │  │   │
│       │               ▼          │  │ (cloud, optional) │  │   │
│       │         ┌────────────┐   │  └───────────────────┘  │   │
│       └─────────│   Runner   │   └─────────────────────────┘   │
│     (keep/      │(runner.py) │                                  │
│      discard)   └─────┬──────┘                                  │
│                       │                                          │
└───────────────────────┼──────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  ┌───────────┐   ┌───────────┐   ┌───────────┐
  │ DT Train  │   │  Episode  │   │  Eval +   │
  │ Pipeline  │   │  Rollout  │   │  Metrics  │
  └───────────┘   └───────────┘   └───────────┘
  pretrain_*.py    decision.py     helper.py
```

---

## 2. File Manifest

| # | File | Purpose | New? |
|---|------|---------|------|
| 1 | `src/autoresearch/__init__.py` | Package init; re-exports public API | ✅ |
| 2 | `src/autoresearch/llm_backend.py` | LLM backend abstraction + implementations | ✅ |
| 3 | `src/autoresearch/prompts.py` | System/user prompt templates + JSON parser | ✅ |
| 4 | `src/autoresearch/config_schema.py` | Mutable config surface, validation, defaults | ✅ |
| 5 | `src/autoresearch/ledger.py` | JSONL experiment ledger read/write/format | ✅ |
| 6 | `src/autoresearch/runner.py` | Subprocess runner for train → eval pipeline | ✅ |
| 7 | `src/autoresearch/agent.py` | `AutoresearchAgent` — outer LLM loop | ✅ |
| 8 | `src/autoresearch/cli.py` | CLI entrypoint (`python -m src.autoresearch`) | ✅ |
| 9 | `src/autoresearch/__main__.py` | Delegates to `cli.main()` | ✅ |
| 10 | `configs/autoresearch_household.json` | Default benchmark for household DT | ✅ |
| 11 | `configs/autoresearch_aemo.json` | Default benchmark for AEMO DT | ✅ |
| 12 | `tests/test_autoresearch.py` | Unit tests (prompt, parse, ledger, config, backend) | ✅ |

**No existing files are modified.** The package is purely additive.

---

## 3. Dependency Policy

**No new pip dependencies.** The LLM backends use only `requests` (already in
`requirements.txt`). The runner invokes existing training CLIs via `subprocess`.

---

## 4. File-by-File Specification

---

### File 1: `src/autoresearch/__init__.py`

```python
"""LLM-driven autoresearch loop for Decision Transformer hyperparameter search."""

from .agent import AutoresearchAgent
from .llm_backend import LlamaCppBackend, OllamaBackend, OpenAIBackend

__all__ = [
    "AutoresearchAgent",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAIBackend",
]
```

---

### File 2: `src/autoresearch/llm_backend.py`

#### Design

All backends implement a single interface:

```python
class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text completion from the LLM."""
```

Each backend POSTs to an **OpenAI-compatible** `/v1/chat/completions` endpoint.
This means llama.cpp, Ollama, vLLM, LM Studio, and OpenAI all work with the
same HTTP call — only the URL and optional API key differ.

#### Classes to implement

| Class | Default URL | Notes |
|-------|-------------|-------|
| `LlamaCppBackend` | `http://localhost:8080/v1` | Default. No API key needed. |
| `OllamaBackend` | `http://localhost:11434/v1` | Same protocol. |
| `OpenAIBackend` | `https://api.openai.com/v1` | Requires `api_key` constructor arg. |

#### Constructor signature (all backends)

```python
def __init__(
    self,
    endpoint: str = <default>,
    model: str = "",         # llama.cpp ignores this; Ollama/OpenAI need it
    temperature: float = 0.7,
    max_tokens: int = 512,
    api_key: str | None = None,   # only OpenAI; ignored by local backends
    timeout: int = 120,           # HTTP timeout seconds
):
```

#### `complete()` implementation sketch

```python
def complete(self, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": self.temperature,
        "max_tokens": self.max_tokens,
    }
    if self.model:
        payload["model"] = self.model

    headers = {"Content-Type": "application/json"}
    if self.api_key:
        headers["Authorization"] = f"Bearer {self.api_key}"

    url = f"{self.endpoint.rstrip('/')}/chat/completions"
    resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
```

#### Error handling

- Wrap `requests.exceptions.ConnectionError` with a clear message:
  `"Cannot reach LLM server at {url}. Is llama-server / ollama running?"`
- On HTTP 4xx/5xx, raise `LLMBackendError(status_code, body)`.
- On JSON decode failure of the response, raise `LLMBackendError`.

#### Why this works for local ≤ 30B models

Prompts are ~200 token system + ~800 token user (ledger history). Expected output
is ~100 tokens of JSON. Even 7B models handle structured JSON output at this scale.

---

### File 3: `src/autoresearch/prompts.py`

#### System prompt template

```python
SYSTEM_PROMPT = """You are a hyperparameter optimization agent for a Decision Transformer
applied to battery energy storage control.

You can ONLY modify the following parameters:
{mutable_keys_json}

Each parameter has a type, valid range, and current default shown above.

Respond with ONLY a JSON object containing your proposed configuration.
Include only the keys you want to change from the current best config.
Do not include explanations, markdown, or any text outside the JSON object."""
```

`{mutable_keys_json}` is populated at runtime from `config_schema.MUTABLE_KEYS`.

#### User prompt template

```python
USER_PROMPT = """Current best config:
{best_config_json}

Current best metric ({metric_name}): {best_metric_value}

Experiment history (last {n} attempts):
{history_lines}

Propose a new configuration that improves {metric_name}{constraint_clause}."""
```

Where:
- `{history_lines}` is formatted by `ledger.format_history(last_n=N)`.
  Each line looks like:
  `  - Attempt 5: context_len=60→80, h_dim=128→256 → mean_reward=-2380.5, degradation=0.008, KEPT`
- `{constraint_clause}` is e.g. ` while keeping degradation below 0.05` or empty.

#### `parse_llm_response(raw: str, schema: ConfigSchema) -> dict`

1. Find the first `{...}` block in `raw` using a simple brace-matching scanner
   (not regex — handles nested braces).
2. `json.loads()` the extracted block.
3. Validate every key against `schema.mutable_keys`:
   - Unknown keys → drop silently.
   - Wrong type → attempt cast; drop if cast fails.
   - Out of range → clamp to valid range.
4. Return the validated dict of **only changed keys**.
5. On total parse failure, raise `ParseError`.

#### `build_system_prompt(schema: ConfigSchema) -> str`

Fills `SYSTEM_PROMPT` with `schema.mutable_keys_description()`.

#### `build_user_prompt(best_config, best_metric, metric_name, history_lines, constraint_clause="") -> str`

Fills `USER_PROMPT`.

---

### File 4: `src/autoresearch/config_schema.py`

#### `ConfigSchema` dataclass

```python
@dataclass
class ConfigSchema:
    mutable_keys: dict[str, MutableParam]  # key → MutableParam
    immutable_keys: dict[str, Any]         # key → fixed value
    defaults: dict[str, Any]               # key → default value for all mutable

    def validate(self, candidate: dict) -> dict: ...
    def clamp(self, key: str, value: Any) -> Any: ...
    def mutable_keys_description(self) -> str: ...
    def merge_with_defaults(self, partial: dict) -> dict: ...
```

#### `MutableParam` dataclass

```python
@dataclass
class MutableParam:
    type: str          # "int", "float", "bool", "str"
    range: Any         # [min, max] for numeric, set of allowed values for str/int-set
    default: Any
    description: str   # one-line human-readable description
```

#### DT Mutable Keys

| Key | Type | Range | Default | Description |
|-----|------|-------|---------|-------------|
| `context_len` | int | [10, 1200] | 60 | Transformer context window length |
| `h_dim` | int | {64, 128, 256, 512} | 128 | Hidden dimension |
| `n_blocks` | int | [1, 6] | 2 | Number of transformer blocks |
| `n_heads` | int | {2, 4, 8} | 8 | Number of attention heads |
| `drop_p` | float | [0.0, 0.5] | 0.1 | Dropout probability |
| `lr` | float | [1e-6, 1e-3] | 2e-5 | Learning rate |
| `batch_size` | int | {4, 6, 8, 16, 32} | 6 | Training batch size |
| `return_scale` | float | [0.1, 100.0] | 1.0 | RTG normalization divisor |
| `discount_factor` | float | [0.9, 1.0] | 0.99 | RTG discount factor |
| `epochs` | int | [1, 10] | 2 | Training epochs |
| `rtg_value` | float | [-5000, -1] | -1500 | Evaluation RTG prompt |
| `use_rope` | bool | {true, false} | false | Enable RoPE positional encoding |
| `action_loss_weight` | float | [0.1, 10.0] | 1.0 | Action prediction loss weight |
| `state_loss_weight` | float | [0.0, 1.0] | 0.01 | State prediction loss weight |
| `return_loss_weight` | float | [0.0, 1.0] | 0.002 | Return prediction loss weight |
| `weight_decay` | float | [0.0, 0.1] | 1e-4 | AdamW weight decay |

#### AEMO-specific Mutable Keys (added to above for AEMO benchmarks)

| Key | Type | Range | Default | Description |
|-----|------|-------|---------|-------------|
| `action_mode` | str | {simple, multi_market} | simple | Env action dimensionality |
| `degradation_mode` | str | {none, rainflow, real_world} | none | Battery degradation model |
| `degradation_chemistry` | str | {NMC, LFP} | LFP | Battery chemistry for degradation |
| `step_duration_hours` | float | {0.25, 0.5, 1.0} | 0.5 | Environment timestep |

#### Immutable Keys (fixed per benchmark, never sent to LLM)

`state_dim`, `act_dim`, data paths, evaluation episode count, random seeds,
`max_timestep`, checkpoint directory root.

#### Helper functions

- `load_benchmark_config(path: str | Path) -> ConfigSchema`:
  Reads a JSON benchmark file (e.g., `configs/autoresearch_household.json`) and
  returns a `ConfigSchema`.
- `HOUSEHOLD_SCHEMA` / `AEMO_SCHEMA`: pre-built schema constants.

---

### File 5: `src/autoresearch/ledger.py`

#### JSONL format — one line per experiment attempt

```json
{
  "attempt_id": 7,
  "timestamp": "2026-04-19T12:34:56Z",
  "config": {"context_len": 80, "h_dim": 256, "lr": 5e-5, "epochs": 3},
  "config_diff": {"context_len": "60→80", "h_dim": "128→256"},
  "baseline_config": {"context_len": 60, "h_dim": 128, "lr": 2e-5, "epochs": 2},
  "stage_a": {"passed": true, "train_loss_final": 0.042, "duration_s": 180},
  "stage_b": {
    "mean_reward": -2380.5,
    "std_reward": 3050.1,
    "sharpe": -0.78,
    "degradation": 0.008,
    "var_5": -9000.0,
    "cvar_5": -9600.0,
    "duration_s": 1200
  },
  "decision": "keep",
  "primary_metric": -2380.5,
  "best_so_far": true
}
```

#### Class `ExperimentLedger`

```python
class ExperimentLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: list[dict] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None: ...          # read JSONL, populate self._entries
    def append(self, entry: dict) -> None: ...   # append + flush to disk
    def last_n(self, n: int) -> list[dict]: ...  # last n entries
    def best_entry(self, metric: str = "mean_reward", higher_is_better: bool = True) -> dict | None: ...
    def format_history(self, last_n: int = 10) -> str: ...  # human-readable lines for prompt
    def next_attempt_id(self) -> int: ...
    def summary(self) -> str: ...          # table of all attempts + best
```

#### CLI summary mode

When run directly (`python -m src.autoresearch.ledger --summary <path>`), prints
a formatted table of all attempts with the best highlighted.

---

### File 6: `src/autoresearch/runner.py`

#### Purpose

Runs the **train → evaluate** pipeline for a single candidate config. This is the
"lab worker" that the agent delegates to.

#### Class `ExperimentRunner`

```python
class ExperimentRunner:
    def __init__(
        self,
        benchmark: str,             # "household" | "aemo"
        schema: ConfigSchema,
        base_output_dir: Path = Path("eval_output/autoresearch"),
        device: str = "auto",       # "cpu", "cuda", "auto"
    ): ...

    def run(self, candidate_config: dict, attempt_id: int) -> dict:
        """
        Execute full train + eval pipeline for candidate_config.
        Returns a result dict with stage_a / stage_b metrics or failure info.
        """
```

#### `run()` implementation

**Stage A — Quick screen (1 epoch, small data subset):**

1. Merge `candidate_config` with `schema.defaults` to get full config.
2. Write a temp model kwargs JSON to `{base_output_dir}/attempt_{id}/model_kwargs.json`.
3. Build the training command:
   ```
   python -m src.pretrain_decision_transformer \
     --data-dir <data_dir> \
     --model-config <temp_kwargs_json> \
     --epochs 1 \
     --batch-size <candidate_batch_size> \
     --lr <candidate_lr> \
     --return-scale <candidate_return_scale> \
     --discount-factor <candidate_discount_factor> \
     ... (other CLI args from candidate) \
     --output-dir <attempt_dir>/stage_a/
   ```
   For AEMO, use `pretrain_aemo_decision_transformer` instead.
4. Run via `subprocess.run(cmd, capture_output=True, timeout=stage_a_timeout)`.
5. Parse stdout/stderr for final training loss. Check for:
   - NaN loss → **fail Stage A**.
   - Loss > 10× baseline → **fail Stage A**.
   - Process exit code ≠ 0 → **fail Stage A**.
6. Return `{"passed": True/False, "train_loss_final": ..., "duration_s": ...}`.

**Stage B — Full evaluation (if Stage A passed):**

1. Run training with full epoch count:
   ```
   python -m src.pretrain_decision_transformer \
     --epochs <candidate_epochs> \
     ... (all candidate args) \
     --output-dir <attempt_dir>/stage_b/
   ```
2. After training completes, run evaluation episodes.
   - For household: use `Agent(algorithm='dt', ...)` rollout.
   - For AEMO: use `AEMOAgent(algorithm='dt', ...)` rollout.
   - The runner imports and calls the rollout functions in-process
     (not subprocess) for simplicity, OR runs a small eval script.
3. Collect metrics via `evaluate_experiment_logs()` from `src/helper.py`.
4. Return the full metrics dict.

#### Timeouts

- Stage A: `stage_a_timeout = 600` seconds (10 min) by default.
- Stage B: `stage_b_timeout = 3600` seconds (1 hour) by default.
- Both configurable via benchmark config.

#### Output directory layout

```
eval_output/autoresearch/
├── ledger.jsonl
├── attempt_001/
│   ├── model_kwargs.json
│   ├── candidate_config.json
│   ├── stage_a/
│   │   └── ... (1-epoch checkpoint + logs)
│   └── stage_b/
│       ├── ... (full training checkpoint)
│       └── eval_episodes.parquet
├── attempt_002/
│   └── ...
└── summary.csv
```

---

### File 7: `src/autoresearch/agent.py`

#### Class `AutoresearchAgent`

```python
class AutoresearchAgent:
    def __init__(
        self,
        backend: LLMBackend,
        schema: ConfigSchema,
        ledger: ExperimentLedger,
        runner: ExperimentRunner,
        primary_metric: str = "mean_reward",
        higher_is_better: bool = True,
        tolerance: float = 50.0,        # metric tolerance for keep/discard
        max_retries: int = 3,           # LLM parse retries
        history_window: int = 10,       # ledger entries shown to LLM
        constraint_clause: str = "",    # e.g. "while keeping degradation below 0.05"
    ): ...

    def propose(self) -> dict:
        """Ask the LLM to propose a new candidate config."""
        system = build_system_prompt(self.schema)
        best = self.ledger.best_entry(self.primary_metric, self.higher_is_better)
        best_config = best["config"] if best else self.schema.defaults
        best_metric = best.get("primary_metric", "N/A") if best else "N/A"
        history = self.ledger.format_history(self.history_window)

        user = build_user_prompt(
            best_config=best_config,
            best_metric=best_metric,
            metric_name=self.primary_metric,
            history_lines=history,
            constraint_clause=self.constraint_clause,
        )

        for attempt in range(self.max_retries):
            raw = self.backend.complete(system, user)
            try:
                return parse_llm_response(raw, self.schema)
            except ParseError:
                continue
        raise AutoresearchError("LLM failed to produce valid JSON after retries")

    def evaluate(self, candidate: dict) -> dict:
        """Run train+eval pipeline and return result dict."""
        attempt_id = self.ledger.next_attempt_id()
        return self.runner.run(candidate, attempt_id)

    def decide(self, result: dict) -> str:
        """Return 'keep' or 'discard' based on result metrics."""
        best = self.ledger.best_entry(self.primary_metric, self.higher_is_better)
        if best is None:
            return "keep"
        best_val = best["primary_metric"]
        new_val = result["stage_b"].get(self.primary_metric)
        if new_val is None:
            return "discard"
        if self.higher_is_better:
            return "keep" if new_val > best_val - self.tolerance else "discard"
        else:
            return "keep" if new_val < best_val + self.tolerance else "discard"

    def step(self) -> dict:
        """One full propose → train → eval → record cycle."""
        candidate = self.propose()
        result = self.evaluate(candidate)
        decision = self.decide(result) if result.get("stage_a", {}).get("passed") else "discard"

        entry = {
            "attempt_id": self.ledger.next_attempt_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "config": candidate,
            "config_diff": self._compute_diff(candidate),
            "stage_a": result.get("stage_a", {}),
            "stage_b": result.get("stage_b", {}),
            "decision": decision,
            "primary_metric": result.get("stage_b", {}).get(self.primary_metric),
            "best_so_far": decision == "keep",
        }
        self.ledger.append(entry)
        return entry

    def run(self, iterations: int) -> None:
        """Run the full autoresearch loop for N iterations."""
        for i in range(iterations):
            print(f"\n{'='*60}")
            print(f"  Autoresearch iteration {i+1}/{iterations}")
            print(f"{'='*60}")
            entry = self.step()
            print(f"  Decision: {entry['decision']} | "
                  f"Metric: {entry.get('primary_metric', 'N/A')}")
            if entry["best_so_far"]:
                print(f"  ★ New best config found!")

    def _compute_diff(self, candidate: dict) -> dict: ...
```

---

### File 8: `src/autoresearch/cli.py`

#### CLI interface

```
python -m src.autoresearch [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `agent` | `agent` (LLM loop) or `manual` (run one config) |
| `--benchmark` | required | `household` or `aemo`, or path to a benchmark JSON |
| `--iterations` | 20 | Number of propose→train→eval cycles (agent mode) |
| `--llm-backend` | `llamacpp` | `llamacpp`, `ollama`, or `openai` |
| `--llm-endpoint` | *(backend default)* | Override LLM server URL |
| `--llm-model` | `""` | Model name (needed for Ollama/OpenAI) |
| `--llm-api-key` | `None` | API key (OpenAI only; or `$OPENAI_API_KEY` env var) |
| `--temperature` | 0.7 | LLM sampling temperature |
| `--max-tokens` | 512 | LLM max response tokens |
| `--primary-metric` | `mean_reward` | Metric to optimize |
| `--tolerance` | 50.0 | Keep/discard tolerance |
| `--constraint` | `""` | Constraint clause for LLM prompt |
| `--ledger-path` | `eval_output/autoresearch/ledger.jsonl` | Ledger file path |
| `--output-dir` | `eval_output/autoresearch` | Base output directory |
| `--candidate-config` | *(none)* | Path to JSON config (manual mode only) |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

#### `main()` implementation

```python
def main():
    args = parse_args()

    # 1. Load benchmark schema
    schema = load_benchmark_config(args.benchmark)

    # 2. Create LLM backend
    backend = create_backend(args)  # factory from --llm-backend + options

    # 3. Create ledger
    ledger = ExperimentLedger(args.ledger_path)

    # 4. Create runner
    runner = ExperimentRunner(
        benchmark=args.benchmark,
        schema=schema,
        base_output_dir=Path(args.output_dir),
        device=args.device,
    )

    if args.mode == "agent":
        agent = AutoresearchAgent(
            backend=backend,
            schema=schema,
            ledger=ledger,
            runner=runner,
            primary_metric=args.primary_metric,
            tolerance=args.tolerance,
            constraint_clause=args.constraint,
        )
        agent.run(args.iterations)

    elif args.mode == "manual":
        config = json.loads(Path(args.candidate_config).read_text())
        validated = schema.validate(config)
        result = runner.run(validated, attempt_id=ledger.next_attempt_id())
        print(json.dumps(result, indent=2))
```

---

### File 9: `src/autoresearch/__main__.py`

```python
from .cli import main

main()
```

---

### File 10: `configs/autoresearch_household.json`

```json
{
  "benchmark": "household",
  "training_script": "src.pretrain_decision_transformer",
  "data_dir": "data/household/logs",
  "eval_episodes": 10,
  "model_config_base": null,
  "immutable": {
    "state_dim": 12,
    "act_dim": 1,
    "seed": 42,
    "max_timestep": 4096
  },
  "defaults": {
    "context_len": 60,
    "h_dim": 128,
    "n_blocks": 2,
    "n_heads": 8,
    "drop_p": 0.1,
    "lr": 2e-5,
    "batch_size": 6,
    "return_scale": 1.0,
    "discount_factor": 0.99,
    "epochs": 2,
    "rtg_value": -1500,
    "use_rope": false,
    "action_loss_weight": 1.0,
    "state_loss_weight": 0.01,
    "return_loss_weight": 0.002,
    "weight_decay": 1e-4
  },
  "stage_a_timeout": 600,
  "stage_b_timeout": 3600,
  "primary_metric": "mean_reward",
  "higher_is_better": true,
  "tolerance": 50.0,
  "constraint_clause": ""
}
```

---

### File 11: `configs/autoresearch_aemo.json`

```json
{
  "benchmark": "aemo",
  "training_script": "src.pretrain_aemo_decision_transformer",
  "data_dir": "data/aemo/logs",
  "eval_episodes": 5,
  "model_config_base": "configs/aemo_decision_transformer_model_kwargs.json",
  "immutable": {
    "state_dim": 18,
    "act_dim": 3,
    "seed": 42,
    "max_timestep": 157680
  },
  "defaults": {
    "context_len": 1152,
    "h_dim": 128,
    "n_blocks": 4,
    "n_heads": 8,
    "drop_p": 0.1,
    "lr": 2e-5,
    "batch_size": 8,
    "return_scale": 1.0,
    "discount_factor": 0.99,
    "epochs": 2,
    "rtg_value": -1500,
    "use_rope": true,
    "action_loss_weight": 1.0,
    "state_loss_weight": 0.01,
    "return_loss_weight": 0.002,
    "weight_decay": 1e-4,
    "action_mode": "multi_market",
    "degradation_mode": "none",
    "degradation_chemistry": "LFP",
    "step_duration_hours": 0.5
  },
  "stage_a_timeout": 600,
  "stage_b_timeout": 7200,
  "primary_metric": "mean_reward",
  "higher_is_better": true,
  "tolerance": 100.0,
  "constraint_clause": ""
}
```

---

### File 12: `tests/test_autoresearch.py`

#### Test categories

**1. Config schema tests:**
- `test_validate_accepts_valid_config` — all keys in range → passes.
- `test_validate_clamps_out_of_range` — lr=999 → clamped to 1e-3.
- `test_validate_drops_unknown_keys` — extra key `foo` → dropped.
- `test_validate_casts_types` — `"128"` for h_dim → cast to int 128.
- `test_merge_with_defaults` — partial config fills from defaults.

**2. Prompt tests:**
- `test_build_system_prompt_contains_mutable_keys` — all key names appear.
- `test_build_user_prompt_contains_history` — history lines appear.
- `test_build_user_prompt_contains_constraint` — constraint clause included.

**3. JSON parsing tests:**
- `test_parse_clean_json` — `'{"lr": 5e-5}'` → `{"lr": 5e-5}`.
- `test_parse_json_with_markdown` — `` ```json\n{...}\n``` `` → extracts inner JSON.
- `test_parse_json_with_explanation` — text before/after JSON → extracts JSON.
- `test_parse_nested_braces` — JSON with nested dicts → correct extraction.
- `test_parse_invalid_raises` — `"I don't know"` → `ParseError`.
- `test_parse_clamps_values` — lr=999 in JSON → clamped.

**4. Ledger tests:**
- `test_append_and_read` — write entry, read back, matches.
- `test_best_entry` — 3 entries → returns the one with best metric.
- `test_format_history` — formatted lines contain attempt IDs and arrows.
- `test_empty_ledger` — fresh ledger, `best_entry()` returns None.

**5. LLM backend tests (mock):**
- `test_llamacpp_backend_calls_correct_url` — mock requests.post, check URL.
- `test_backend_connection_error` — mock ConnectionError → clear message.
- `test_backend_timeout` — mock Timeout → raises with message.
- `test_backend_bad_status` — mock 500 → raises LLMBackendError.

**6. Agent tests (integration, mocked LLM):**
- `test_agent_step_keep` — mock LLM returns valid JSON, mock runner returns
  good metrics → decision = "keep", ledger has 1 entry.
- `test_agent_step_discard` — mock runner returns bad metrics → "discard".
- `test_agent_step_stage_a_fail` — mock runner returns stage_a.passed=False → "discard".

---

## 5. Local LLM Quick Start

### Option A: llama.cpp (recommended)

```bash
# 1. Download a GGUF model (example: Qwen2.5-32B quantized)
wget https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/qwen2.5-32b-instruct-q4_k_m.gguf

# 2. Start llama.cpp server
./llama-server \
  -m qwen2.5-32b-instruct-q4_k_m.gguf \
  --port 8080 \
  --ctx-size 2048 \
  --n-gpu-layers 99

# 3. Run autoresearch (default connects to localhost:8080)
python -m src.autoresearch \
  --benchmark household \
  --iterations 20
```

### Option B: Ollama

```bash
# 1. Start Ollama
ollama serve

# 2. Pull a model
ollama pull qwen2.5:32b

# 3. Run autoresearch
python -m src.autoresearch \
  --benchmark household \
  --llm-backend ollama \
  --llm-model qwen2.5:32b \
  --iterations 20
```

### Option C: OpenAI (cloud, optional)

```bash
export OPENAI_API_KEY=sk-...

python -m src.autoresearch \
  --benchmark aemo \
  --llm-backend openai \
  --llm-model gpt-4o \
  --iterations 10
```

---

## 6. Implementation Order

The implementing agent should create files in this order:

1. **`config_schema.py`** — no deps on other autoresearch files.
2. **`llm_backend.py`** — no deps on other autoresearch files.
3. **`prompts.py`** — depends on `config_schema`.
4. **`ledger.py`** — no deps on other autoresearch files.
5. **`runner.py`** — depends on `config_schema`; calls existing training CLIs.
6. **`agent.py`** — depends on all above.
7. **`cli.py`** + **`__main__.py`** — depends on `agent` + factory for backends.
8. **`__init__.py`** — trivial re-exports.
9. **Benchmark configs** — JSON files in `configs/`.
10. **Tests** — `tests/test_autoresearch.py`.

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No new pip deps** | `requests` already exists. Keeps install simple. |
| **OpenAI-compatible API only** | llama.cpp, Ollama, vLLM, LM Studio all speak this. One HTTP client handles all. |
| **JSON-only LLM output** | Keeps task trivial for ≤ 30B models. No code generation needed. |
| **Two-stage gate** | Stage A (1 epoch) catches divergence early; saves GPU hours. |
| **JSONL ledger** | Append-only, human-readable, easy to parse. No database dep. |
| **Subprocess for training** | Training CLIs already exist and handle device/checkpointing. No need to refactor. |
| **Purely additive** | Zero changes to existing src files. Safe to merge independently. |
| **Local LLM as default** | `LlamaCppBackend` with `localhost:8080` is the zero-config default. Cloud is opt-in. |

---

## 8. Integration Points (no changes needed)

| Existing Component | How Autoresearch Uses It |
|--------------------|--------------------------|
| `pretrain_decision_transformer.py` | Invoked via subprocess with candidate config CLI args |
| `pretrain_aemo_decision_transformer.py` | Same, for AEMO benchmarks |
| `Agent` / `AEMOAgent` (`decision.py`) | Used by runner for evaluation rollouts |
| `evaluate_experiment_logs` (`helper.py`) | Used by runner to compute metrics |
| `TrajectoryDataset` (`transformer_training.py`) | Used by training pipeline (unchanged) |
| `configs/*.json` | Read by runner as base model kwargs |

---

## 9. Success Criteria

The implementation is complete when:

- [ ] `python -m src.autoresearch --benchmark household --mode manual --candidate-config <file>` trains and evaluates a single config, printing metrics.
- [ ] `python -m src.autoresearch --benchmark household --llm-backend llamacpp --iterations 3` runs 3 propose→train→eval cycles with a local LLM, producing a ledger file with 3 entries.
- [ ] `python -m src.autoresearch.ledger --summary eval_output/autoresearch/ledger.jsonl` prints a formatted summary table.
- [ ] All tests in `tests/test_autoresearch.py` pass.
- [ ] No existing tests are broken (run `pytest` from repo root).
