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
   - `src/autoresearch_evaluator.py`
   - the evaluator config chosen for the run
   - `docs/aemo/workflow.md` if you are using the AEMO track

4. **Verify data and paths exist**
   - Household track requires parquet logs under `data/household/logs/`.
    - AEMO track requires `data/aemo_dt_fcas/aemo_fcas_dataset.parquet` (current recommended corpus) or `data/aemo_dt/aemo_dt_dataset.parquet` (legacy 162-episode dataset) and `configs/aemo_decision_transformer_model_kwargs.json`.
   - Model outputs must stay under the repository root, typically under `models/household/dt/` or `models/aemo/dt/`.
   - Before running experiments, define which episodes/date ranges/regions are training data, which are validation-only, and which are held out for simulator evaluation.

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

7. **Define the evaluation ladder**
   - Use validation loss as the fast inner-loop ranking metric during search.
   - Also reserve held-out simulator scenarios that differ from the training distribution, such as different date ranges, regions, or operating regimes.
   - Decide up front which policy baseline or existing controller the held-out simulator rollouts should be compared against.

## In-scope files

### Editable surface

- `src/pretrain_decision_transformer.py`

This is the single sanctioned experiment surface for autoresearch in this repository.

### Stable / read-only during autoresearch

- `src/decision_transformer.py`
- `src/transformer_training.py`
- `src/pretrain_aemo_decision_transformer.py`
- `src/aemo_notebook_utils.py`
- `src/autoresearch_evaluator.py`
- the evaluator config fixed for the run
- notebooks
- environment dynamics
- evaluation helpers
- dataset schema

## What the harness may change

Only change knobs already exposed by `src/pretrain_decision_transformer.py`, including:

- surface preset / approved model variant selection
- DT dimensions (`state_dim`, `act_dim`, `n_block`, `h_dim`, `n_heads`, `context_len`, `max_timestep`)
- dropout and RoPE settings
- training hyperparameters (`batch_size`, `epochs`, `lr`, `discount`, `return_scale`, loss weights, weight decay)
- approved optimizer / scheduler selection, including optional custom class-path hooks already supported by the surface
- DataLoader settings (`num_workers`, `persistent_workers`, `prefetch_factor`)
- split-policy handling already supported by the file

The AEMO wrapper may forward a curated subset of those same knobs, but manual mixed-corpus runs that need
`--patterns` or other direct-trainer-only flags should call `src/pretrain_decision_transformer.py` directly.

The harness may also improve the code inside the editable surface if the change still preserves the existing adapter contract and artifact contract.

## What the harness must not change

- Do not modify dataset columns or parquet schema.
- Do not modify notebook workflows just to make the harness easier.
- Do not move output artifacts outside the repository.
- Do not change the meaning of the shared adapter interfaces.
- Do not modify evaluation/environment code as part of DT training-surface autoresearch.
- Do not add new dependencies.

## Hard constraints from the codebase

- `src/pretrain_decision_transformer.py` is the canonical entrypoint.
- Approved optimizers and schedulers are restricted by the code surface allowlist.
- Custom optimizer / scheduler paths are only valid when they are already importable in the repo/runtime and do not require new dependencies.
- AEMO-shaped DT runs must keep `act_dim` aligned with `action_mode`:
  - `simple -> 1`
  - `multi_market -> 3`
  - `full_fcas -> 9`
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
  --dataset-path data/aemo_dt_fcas/aemo_fcas_dataset.parquet \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --epochs 2 \
  --batch-size 16 \
  --lr 3e-5 \
  --val-split 0.1 \
  --save-path models/aemo/dt/aemo_dt_model.pt \
  --checkpoint-path models/aemo/dt/aemo_dt_checkpoint.pt \
  --loss-csv-path models/aemo/dt/aemo_dt_loss_history.csv
```

### Distrobox note

Run the autoresearch agent inside the `energydecision` Distrobox container rather than the Docker Compose shell. From the repo root, the agent should keep using `src/...` script paths and normal `data/...` and `models/...` paths.

The DT trainer includes a live terminal monitor for epoch/batch progress plus CPU, RAM, GPU, and VRAM stats. It works from the repo root and from interactive Distrobox shells opened with `distrobox enter energydecision`.

If the host has an NVIDIA GPU, create and use a GPU-enabled box for training:

```bash
distrobox create --name energydecision-gpu --image energydecision:latest --nvidia
distrobox enter energydecision-gpu
python3 -c "import torch; print(torch.cuda.is_available())"
```

Use the GPU box for DT training commands so the trainer can see CUDA and use `device=cuda` automatically when available.

For AEMO CLI runs, prefer `src/launch_aemo_training.py` over calling the wrapper by hand. The launcher
derives tier defaults (`proxy-smoke`, `proxy-baseline`, `learning-baseline`), writes a launch plan under
the run directory, and re-enters the preferred Distrobox automatically when invoked from the host.

```bash
python3 src/launch_aemo_training.py --run-tier proxy-baseline
python3 src/launch_aemo_training.py --run-tier learning-baseline
```

The baked-in `proxy-baseline` defaults now follow the current frontier pilot setting: fixed pilot
train/validation split, `context_length=180`, `batch_size=16`, `epochs=2`, and a deeper+wider
transformer (`n_block=8`, `h_dim=384`, `n_heads=8`).

For a separate live dashboard while training runs, use `src/dt_progress_runner.py` with the training command and the matching `--progress-snapshot-path`. It watches the JSON snapshot and shows the latest training, validation, best-metric, and resource signals in a dedicated terminal.

If the training process is already running, use `--attach` with the same `--progress-snapshot-path` and no child command:

```bash
python3 src/dt_progress_runner.py \
  --attach \
  --progress-snapshot-path models/aemo/dt/<run-tag>/aemo_dt_loss_history_progress.json
```

The tracker now prefers a **Rich** full-screen terminal dashboard when stdout is a TTY and `rich` is installed. Keep `rich` available in the runtime environment for the nicer color dashboard. If Rich is unavailable or stdout is not a TTY, the tracker falls back to the plain text monitor automatically. Use `--ui plain` if you explicitly want the old plain-text mode.

For long AEMO subset runs, the dashboard only refreshes when the trainer writes a checkpoint snapshot. If you
need mid-epoch visibility during long subset stages, prefer `--checkpoints-per-epoch` values above `1`
instead of waiting until the end of a single huge epoch.

When the harness is launched from an SSH session, it must start the autoresearch run inside the GPU box/container and keep the progress tracker visible to the human in a separate tmux pane or terminal in that same box/container. If the run was started elsewhere, the human can still attach with `--attach` as long as the snapshot file is reachable. The harness should not hide the tracker behind a detached-only process.

Example layout:

```bash
tmux new -s autoresearch
# pane 1: enter the GPU box/container and run the trainer with --progress-snapshot-path models/aemo/dt/<run-tag>/aemo_dt_loss_history_progress.json

# split pane
Ctrl-b %

# pane 2: enter the same GPU box/container and run
python3 src/dt_progress_runner.py --progress-snapshot-path models/aemo/dt/<run-tag>/aemo_dt_loss_history_progress.json
```

Replace `<run-tag>` with the actual run directory (for example `restart_20260514_pilot/baseline`). Keep the tracker pointed at the same progress snapshot file that the trainer writes so the human can watch the live dashboard while autoresearch is running.

## Immutable evaluator

Use `src/autoresearch_evaluator.py` as the fixed evaluation entrypoint for autoresearch checkpoints. The autoresearch agent must treat the evaluator script and the chosen evaluator config as read-only for the duration of a run.

For AEMO runs, start from `configs/aemo_autoresearch_evaluator.example.json`, copy it to a run-specific config outside the editable DT training surface, and fix the held-out scenarios, battery variants, dispatch replay stations, baseline policies, and DT `rtg_value` before the search loop begins.

For faster autoresearch loops, enable evaluator rollout parallelism in the held-out config and keep it on during the search:

```json
"heldout": {
  "...": "...",
  "parallel_workers": 4,
  "parallelize_candidate_dt": false
}
```

Use `parallel_workers > 1` for parallel scenario × battery × policy rollout execution. Keep `parallelize_candidate_dt=false` unless you explicitly want DT candidate rollouts to run in parallel with reference policies.

Before the search loop begins, ensure the evaluator `cache_dir` is writable by the same user who will run
autoresearch. Prewarming the fixed held-out scenario windows in that cache directory is recommended so
evaluator reruns do not spend the inner loop fetching and preprocessing the same AEMO data repeatedly.

The repo now includes a dedicated cache-prewarm helper:

```bash
python3 src/prewarm_aemo_cache.py \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json
```

It validates processed-cache permissions up front and writes a manifest describing which held-out windows
were warmed.

When the evaluator includes dispatch replay baselines, choose at least two stations from the same held-out time window and confirm they both have `DISPATCHLOAD` coverage before freezing the config. If one station is missing data in that window, replace it with another station from the same region/window rather than leaving a brittle default in place.

Typical command:

```bash
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/aemo_dt_loss_history_surface_manifest.json \
  --evaluation-config <path-to-fixed-evaluator-config.json> \
  --output-dir eval_output/autoresearch/<run-tag>/<commit>
```

The evaluator always reports training-loss summaries from the DT loss CSV and, for AEMO, also runs held-out simulator rollouts plus baseline-relative comparisons, safety counts, stability/risk metrics, and condition-sliced summaries.

## AEMO baseline tiers

For AEMO autoresearch, treat the training loop as two separate tiers:

1. **Fast proxy loop**
   - use this to rank cheap ideas quickly
   - compact models and narrow slices are acceptable here
   - do not treat the result as the project baseline without evaluator confirmation
   - for kickstart and interactive loops, use the lengthened fixed pilot train parquet plus a fixed explicit validation parquet (for example under `data/aemo_dt/autoresearch_pilot/`) so the full corpus is not needed until later
   - keep every proxy comparison on the same pilot split

2. **Learning baseline**
   - use this when establishing the actual baseline checkpoint that future experiments should branch from
   - prefer broader train subsets, explicit held-out validation parquet files, and a materially longer context window
   - this is the baseline that should be compared with the immutable evaluator before broad sweeps continue

For the current AEMO dataset layout:

- treat `aemo_dt_dataset_train_subset_007` as a **proxy-only** slice, not the main learning baseline
- the fixed autoresearch pilot uses contiguous week-long slices per episode, so each example carries about one week of 5-minute history
- prefer one of the normal 24-episode train subsets plus explicit validation subsets/files for learning baselines
- prefer `context_len=288` for learning baselines; `120` is an acceptable runtime fallback, but `60` is primarily a proxy-loop setting
- wrapper launches can now forward `context_len` and other approved DT shape knobs, but the direct trainer is still the right entrypoint for manual mixed-corpus command lines
- prefer `lr=3e-5` over `2e-5` as the starting learning-baseline LR unless new evaluator evidence contradicts it

## Primary metric

Use one fast proxy metric consistently for the inner loop:

- **AEMO proxy tier (`aemo_proxy`)**: best validation action loss (`best_val_action_loss`, lower is better)
- **Guardrail for AEMO proxy tier**: best validation total loss (`best_val_total_loss`)
- **Broader baselines / non-proxy tiers**: best validation total loss (`best_val_total_loss`)

Read it from the DT loss CSV / surface manifest summary, not only from the final console line.

If a run has no validation set, record that explicitly and treat comparisons as weak evidence. Prefer runs with validation enabled.

Validation loss is only a search proxy. It is useful for ranking many training-surface ideas quickly, but it is not the final success criterion for the project.

## Required evaluation beyond validation loss

For any promising checkpoint, also evaluate policy behavior in held-out simulator rollouts that were not used for training or validation.

At minimum, report:

1. **Held-out return / objective**
   - Run the policy on unseen simulator scenarios and compare total return or the project-relevant objective against the baseline controller.

2. **Generalization split**
   - Use holdouts that differ from the training distribution, such as different date ranges, regions, weather patterns, or market regimes.
   - Prefer multiple holdout slices over a single lucky test set.

3. **Safety / constraint metrics**
   - Record invalid actions, dispatch-limit violations, infeasible transitions, clipped actions, or any simulator-defined rule breaches.

4. **Stability metrics**
   - Check variance across random seeds or repeated rollouts.
   - Track catastrophic failures, very short episodes, or episodes with extremely poor return.

5. **Baseline-relative regret**
   - Compare against the current heuristic/controller, not just against other DT checkpoints.
   - A lower validation loss is not enough if held-out simulator performance regresses.

If validation loss improves but held-out simulator behavior gets worse, prefer the simpler/safer checkpoint or reject the change entirely.

## Data coverage expectations

Small fixed pilot slices are acceptable for fast iteration, but they are not enough to establish robust simulator performance.

When planning or extending autoresearch datasets:

- collect more full episodes if the current run uses only a narrow slice of behavior
- add episodes from different date ranges and regions instead of only sampling more from the same regime
- preserve a strict held-out evaluation set that is never used for training or validation tuning
- include rare and difficult operating conditions so the agent does not only learn common easy cases

More simulated episodes usually help when they increase coverage of the target deployment distribution. More of the same narrow regime is less useful than broader regime coverage.

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
2. Re-establish whether the current branch point is a **proxy loop** checkpoint or the **learning baseline** checkpoint.
3. Make one focused change in `src/pretrain_decision_transformer.py`.
4. Commit the change.
5. Run the fixed training command directly in the terminal so the live DT monitor stays visible. If you also need a log file, mirror the output with `tee` instead of fully redirecting stdout/stderr away from the terminal.
6. Read the proxy ranking metric from the loss CSV / surface manifest summary.
7. Record the result in `results.tsv`.
8. For AEMO proxy loops, prefer `configs/aemo_autoresearch_evaluator.mini.json` as the first simulator screen.
9. Run `src/autoresearch_evaluator.py` on the baseline checkpoint before interpreting later experiments.
10. Periodically rerun the evaluator on the strongest checkpoints rather than waiting until the very end.
11. If the proxy metric improved, but the evaluator shows worse held-out simulator objective, safety metrics, or stability metrics, do not automatically keep the change.
12. If both the proxy metric and the evaluator evidence improved, keep the commit and continue from there.
13. If the metric was worse, or the idea added complexity without clear evaluator benefit, revert to the previous kept commit.

For AEMO learning-baseline refreshes, prefer this low-hanging-fruit order:

1. broaden the data slice / explicit validation setup first
2. move baseline LR to `3e-5`
3. compare longer contexts (`120` vs `288`)
4. only then test extra epochs or larger models

## Simplicity rule

Prefer the smallest change that improves validation loss without hurting held-out simulator behavior.

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
