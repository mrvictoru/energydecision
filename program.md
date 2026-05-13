# energydecision autoresearch program

This document tells an autonomous agent harness how to run constrained autoresearch on this repository.

The goal is **not** to freely mutate the whole project. The goal is to run repeatable Decision Transformer training experiments while keeping the environment, dataset schema, notebook workflow, and adapter layers stable.

## Setup

Before starting an autoresearch run, work with the human to:

1. **Agree on a run tag**
   - Use a fresh branch such as `autoresearch/<tag>`.
   - Do not reuse an old experiment branch unless the human explicitly asks for continuation.

2. **Choose one fixed research track**
   - **Household DT track**: run `src/pretrain_decision_transformer.py` directly on household parquet logs.
   - **AEMO DT track**: launch through `src/pretrain_aemo_decision_transformer.py`, but still edit only the shared DT training surface in `src/pretrain_decision_transformer.py`.
   - Pick one track per run and keep it fixed. Do not compare metrics across different datasets, tracks, or command shapes.

3. **Read the in-scope files**
   - `README.md`
   - `program.md`
   - `src/pretrain_decision_transformer.py`
   - `src/pretrain_aemo_decision_transformer.py`
   - `docs/AEMO_DT_WORKFLOW.md` if you are using the AEMO track

4. **Verify data and paths exist**
   - Household track requires parquet logs under `data/household/logs/`.
   - AEMO track requires `data/aemo_dt/aemo_dt_dataset.parquet` and `configs/aemo_decision_transformer_model_kwargs.json`.
   - Model outputs must stay under the repository root, typically under `models/household/dt/` or `models/aemo/dt/`.

5. **Create an untracked results file**
   - Create `results.tsv` in the repo root if it does not exist.
   - Keep it **untracked** by git.
   - Header:

```tsv
commit	track	metric	status	description
```

6. **Establish a baseline**
   - The first run must be the current code with no experiment changes.
   - Record that result as `baseline`.

## In-scope files

### Editable surface

- `src/pretrain_decision_transformer.py`

This is the single sanctioned experiment surface for autoresearch in this repository.

### Stable / read-only during autoresearch

- `src/decision_transformer.py`
- `src/transformer_training.py`
- `src/pretrain_aemo_decision_transformer.py`
- `src/aemo_notebook_utils.py`
- notebooks
- environment dynamics
- evaluation helpers
- dataset schema

## What you can change

Only change knobs already exposed by `src/pretrain_decision_transformer.py`, including:

- surface preset / approved model variant selection
- DT dimensions (`state_dim`, `act_dim`, `n_block`, `h_dim`, `n_heads`, `context_len`, `max_timestep`)
- dropout and RoPE settings
- training hyperparameters (`batch_size`, `epochs`, `lr`, `discount`, `return_scale`, loss weights, weight decay)
- DataLoader settings (`num_workers`, `persistent_workers`, `prefetch_factor`)
- split-policy handling already supported by the file

You may also improve the code inside the editable surface if the change still preserves the existing adapter contract and artifact contract.

## What you must not change

- Do not modify dataset columns or parquet schema.
- Do not modify notebook workflows just to make the harness easier.
- Do not move output artifacts outside the repository.
- Do not change the meaning of the shared adapter interfaces.
- Do not modify evaluation/environment code as part of DT training-surface autoresearch.
- Do not add new dependencies.

## Hard constraints from the codebase

- `src/pretrain_decision_transformer.py` is the canonical entrypoint.
- Approved optimizers and schedulers are currently restricted by the code surface.
- AEMO-shaped DT runs must keep `act_dim` aligned with `action_mode`:
  - `simple -> 1`
  - `multi_market -> 3`
- Transformer width must remain internally consistent:
  - `h_dim` must be divisible by `n_heads`
- Unknown model-config keys, unsupported presets, and unsupported variants should be rejected early.

## Running experiments

Always keep the training command fixed for the whole run except for intentional experiment changes that are part of the research idea.

### Recommended baseline command: household track

Run from the repository root:

```bash
python3 src/pretrain_decision_transformer.py \
  --surface-preset autoresearch_safe \
  --data-dir data/household/logs \
  --patterns train_episode_01 train_episode_02 \
  --epochs 2 \
  --batch-size 6 \
  --lr 2e-5 \
  --save-path models/household/dt/dt_model.pt \
  --checkpoint-path models/household/dt/dt_model_checkpoint.pt \
  --loss-csv-path models/household/dt/dt_model_loss_history.csv
```

### Recommended baseline command: AEMO track

Run from the repository root:

```bash
python3 src/pretrain_aemo_decision_transformer.py \
  --dataset-path data/aemo_dt/aemo_dt_dataset.parquet \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --epochs 2 \
  --batch-size 6 \
  --lr 2e-5 \
  --val-split 0.1 \
  --save-path models/aemo/dt/aemo_dt_model.pt \
  --checkpoint-path models/aemo/dt/aemo_dt_checkpoint.pt \
  --loss-csv-path models/aemo/dt/aemo_dt_loss_history.csv
```

### Distrobox note

Run the autoresearch agent inside the `energydecision` Distrobox container rather than the Docker Compose shell. From the repo root, the agent should keep using `src/...` script paths and normal `data/...` and `models/...` paths.

The DT trainer includes a live terminal monitor for epoch/batch progress plus CPU, RAM, GPU, and VRAM stats. It works from the repo root and from interactive Distrobox shells opened with `distrobox enter energydecision`.

For a separate live dashboard while training runs, use `src/dt_progress_runner.py` with the training command and the matching `--progress-snapshot-path`. It watches the JSON snapshot and shows the latest training, validation, best-metric, and resource signals in a dedicated terminal.

## Primary metric

Use one metric consistently for the whole run:

- **Primary metric**: final validation total loss from the DT run (`lower is better`)

Read it from the loss CSV written by `--loss-csv-path`, or from the final console summary if needed.

If a run has no validation set, record that explicitly and treat comparisons as weak evidence. Prefer runs with validation enabled.

## Logging results

Append one line per experiment to `results.tsv`:

```tsv
commit	track	metric	status	description
abc1234	household	1.234567	keep	baseline
def5678	household	1.210000	keep	increase context length to 120
987fedc	household	1.240000	discard	switch to wide variant
```

Status must be one of:

- `keep`
- `discard`
- `crash`

## Experiment loop

Once setup is complete, loop autonomously:

1. Check git state and confirm you are on the intended autoresearch branch.
2. Make one focused change in `src/pretrain_decision_transformer.py`.
3. Commit the change.
4. Run the fixed training command directly in the terminal so the live DT monitor stays visible. If you also need a log file, mirror the output with `tee` instead of fully redirecting stdout/stderr away from the terminal.
5. Read the final validation metric from the loss CSV or log.
6. Record the result in `results.tsv`.
7. If the metric improved, keep the commit and continue from there.
8. If the metric was worse or the idea added complexity without clear benefit, revert to the previous kept commit.

## Simplicity rule

Prefer the smallest change that improves validation loss.

If two changes perform about the same:

- keep the simpler one
- keep the one that preserves clearer adapter compatibility
- keep the one that is easier for a human to understand and maintain

## Crash handling

If a run crashes:

1. Inspect the log.
2. If the problem is a trivial bug in the current experiment, fix it and rerun once.
3. If the idea is fundamentally bad or unstable, log `crash`, revert, and move on.

Do not repeatedly dig deeper into broken ideas when the failure mode is obvious.

## Validation

Before and after autoresearch edits, use the repository test command when the environment supports it:

```bash
python -m pytest tests/ -v
```

If the environment is missing test dependencies, note that clearly in the run log and continue with the constrained training experiment only if the human still wants autoresearch.

## Artifacts to watch

During runs, expect outputs such as:

- model weights (`*.pt`)
- checkpoints
- loss CSVs
- checkpoint-segment loss CSVs
- training-surface manifest JSON written next to the loss CSV

These artifacts are part of the expected contract and are useful for auditing what changed.

## Stop conditions

Stop when:

- the human interrupts you
- the data required for the chosen track is missing
- the environment cannot run the chosen training command
- repeated experiments stop producing useful ideas
- a human-defined experiment budget or time budget is reached

The harness should be persistent, but not reckless: stay within the constrained DT training surface and keep experiments reproducible.
