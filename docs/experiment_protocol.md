# Constrained Decision Transformer experiment protocol

Audience: contributors, maintainers, and automated experiment runners. This protocol applies only to explicitly requested autoresearch runs, not general maintenance or paper reproduction. For the shipped preprint, see the [reproduction guide](reproduction.md).

## Setup

Before starting a constrained run:

1. **Agree on a run tag with the project maintainer or run supervisor.**
   - Use a fresh branch such as `autoresearch/<tag>`.
   - Do not reuse an old experiment branch unless continuation is explicitly requested.

2. **Choose one fixed research track.**
   - Household DT track: run `scripts/pretrain_decision_transformer.py` directly on household parquet logs.
   - AEMO DT track: launch through `scripts/pretrain_aemo_decision_transformer.py`, while still editing only the shared DT training surface in `scripts/pretrain_decision_transformer.py`.
   - Pick one track per run and keep it fixed. Do not compare metrics across datasets, tracks, or command shapes.

3. **Read the in-scope files.**
   - `README.md`
   - `docs/experiment_protocol.md`
   - `scripts/pretrain_decision_transformer.py`
   - `scripts/pretrain_aemo_decision_transformer.py`
   - `scripts/autoresearch_evaluator.py`
   - the evaluator config chosen for the run
   - `docs/aemo/workflow.md` for the AEMO track

4. **Verify data and paths exist.**
   - Household track requires parquet logs under `data/household/logs/`.
   - For AEMO, select a corpus and matching model config before running. The pilot launcher uses behaviour-policy experiment data, not the paper's `dt_trajectories_jtsoc_v2cal_conservative.parquet`. The combined SDP corpus and FCAS-rich dataset are historical experiment inputs.
   - Model outputs stay under the repository root, typically `models/household/dt/` or `models/aemo/dt/`.
   - Before running, define which episodes/date ranges/regions are training data, which are validation-only, and which are held out for simulator evaluation.

5. **Create a run ledger** under ignored `results/`, for example `results/<tag>.tsv`. Do not overwrite tracked historical `results.tsv`. Header:

```tsv
commit	track	metric	status	description
```

6. **Establish a baseline.** The first run must be the current code with no experiment changes. Record it as `baseline`.

7. **Define the evaluation ladder.** Use validation loss as the fast inner-loop ranking metric, and reserve held-out simulator scenarios that differ from the training distribution. Decide in advance which policy baseline or existing controller the held-out rollouts should be compared against.

## In-scope files

### Editable surface

- `scripts/pretrain_decision_transformer.py`

This is the defined experiment surface for autoresearch in this repository.

### Stable during autoresearch

- `src/decision_transformer.py`
- `src/transformer_training.py`
- `scripts/pretrain_aemo_decision_transformer.py`
- `src/aemo_notebook_utils.py`
- `scripts/autoresearch_evaluator.py`
- `src/decision.py` (`AEMOAgent`)
- `src/aemo_sdp_executor.py`
- `src/aemo_oracle_algo.py`
- `src/market_impact.py`
- `src/grpo_posttraining.py`
- `scripts/generate_sdp_dt_trajectories.py`
- the evaluator config fixed for the run
- notebooks
- environment dynamics
- evaluation helpers
- dataset schema

## Allowed changes

Only change knobs already exposed by `scripts/pretrain_decision_transformer.py`, including:

- surface preset / approved model variant selection
- DT dimensions (`state_dim`, `act_dim`, `n_block`, `h_dim`, `n_heads`, `context_len`, `max_timestep`, `n_kv_heads`, `qk_norm`, `tie_weights`, `action_head_mode`)
- dropout and RoPE settings
- training hyperparameters (`batch_size`, `epochs`, `lr`, `discount`, `return_scale`, loss weights, weight decay)
- RTG conditioning (`--rtg-source` `auto`/`constant`/`j_t_soc`, `--auto-return-scale`)
- approved optimizer / scheduler selection, including optional custom class-path hooks already supported by the surface
- DataLoader settings (`num_workers`, `persistent_workers`, `prefetch_factor`)
- split-policy handling already supported by the file

The AEMO wrapper may forward a curated subset of those knobs. Manual mixed-corpus runs that need `--patterns` or other direct-trainer-only flags should call `scripts/pretrain_decision_transformer.py` directly. The run may also improve code inside the editable surface if the adapter and artifact contracts are preserved.

## Constraints

- Do not modify dataset columns or parquet schema.
- Do not modify notebook workflows to make the run easier.
- Do not move output artifacts outside the repository.
- Do not change the meaning of shared adapter interfaces.
- Do not modify evaluation or environment code as part of DT training-surface autoresearch.
- Do not add new dependencies.

Additional hard constraints:

- `scripts/pretrain_decision_transformer.py` is the canonical entrypoint.
- Approved optimizers and schedulers are restricted by the code-surface allowlist.
- Custom optimizer / scheduler paths are valid only when already importable in the runtime and requiring no new dependencies.
- AEMO-shaped DT runs must keep `act_dim` aligned with `action_mode`: `simple -> 1`, `multi_market -> 3`, `full_fcas -> 9`.
- `h_dim` must be divisible by `n_heads`.
- Unknown model-config keys, unsupported presets, and unsupported variants should be rejected early.

## Running experiments

Keep the training command fixed for the whole run except for intentional experiment changes.

### Household baseline example

```bash
python3 scripts/pretrain_decision_transformer.py \
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

### AEMO baseline example (historical FCAS-rich corpus)

```bash
python3 scripts/pretrain_aemo_decision_transformer.py \
  --dataset-path data/aemo_dt_fcas/aemo_fcas_dataset.parquet \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --epochs 2 \
  --batch-size 64 \
  --lr 3e-5 \
  --val-split 0.1 \
  --save-path models/aemo/dt/aemo_dt_model.pt \
  --checkpoint-path models/aemo/dt/aemo_dt_checkpoint.pt \
  --loss-csv-path models/aemo/dt/aemo_dt_loss_history.csv
```

### Runtime notes

Run from the repository root in the `energydecision` Distrobox, using `scripts/...` entry points and repo-relative `data/...` and `models/...` paths. See the [development guide](development.md) for runtime conventions.

The DT trainer includes a terminal monitor for epoch/batch progress plus CPU, RAM, GPU and VRAM stats. If the host has an NVIDIA GPU, create and use a GPU-enabled box:

```bash
distrobox create --name energydecision-gpu --image energydecision:latest --nvidia
distrobox enter energydecision-gpu
python3 -c "import torch; print(torch.cuda.is_available())"
```

For AEMO CLI runs, prefer `scripts/launch_aemo_training.py`:

```bash
python3 scripts/launch_aemo_training.py --run-tier proxy-baseline
python3 scripts/launch_aemo_training.py --run-tier learning-baseline
```

The launcher derives tier defaults, writes a launch plan under the run directory, and re-enters the preferred Distrobox when invoked from the host. Verify exact tier defaults in `scripts/launch_aemo_training.py`; the launcher is the source of truth.

For a separate live dashboard, use `scripts/dt_progress_runner.py` with the training command and matching `--progress-snapshot-path`. To attach to an already running process:

```bash
python3 scripts/dt_progress_runner.py \
  --attach \
  --progress-snapshot-path models/aemo/dt/<run-tag>/aemo_dt_loss_history_progress.json
```

The tracker prefers a Rich full-screen dashboard when stdout is a TTY and `rich` is installed; it falls back to plain text otherwise. Use `--ui plain` to force the old mode. For long AEMO subset runs, prefer `--checkpoints-per-epoch > 1` if mid-epoch visibility is needed.

When a run is launched from an SSH session, start it inside the GPU box/container and keep the progress tracker visible to the operator in a separate tmux pane or terminal in that same environment. If the run was started elsewhere, the operator can still attach with `--attach` as long as the snapshot file is reachable.

## Immutable evaluator

Use `scripts/autoresearch_evaluator.py` as the fixed evaluation entrypoint. Treat the evaluator script and chosen config as read-only for the duration of a run.

For AEMO runs, start from `configs/aemo_autoresearch_evaluator.example.json`, copy it to a run-specific config outside the editable DT training surface, and fix held-out scenarios, battery variants, dispatch replay stations, baseline policies, and DT `rtg_value` before the search loop begins.

For faster loops, enable evaluator rollout parallelism:

```json
"heldout": {
  "...": "...",
  "parallel_workers": 4,
  "parallelize_candidate_dt": false
}
```

Use `parallel_workers > 1` for parallel scenario × battery × policy rollouts. Keep `parallelize_candidate_dt=false` unless DT candidate rollouts should run in parallel with reference policies.

Ensure the evaluator `cache_dir` is writable by the same user who will run the experiment. Prewarm fixed held-out windows:

```bash
python3 scripts/prewarm_aemo_cache.py \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json
```

When dispatch replay baselines are included, choose at least two stations from the same held-out window and confirm `DISPATCHLOAD` coverage before freezing the config.

Typical command:

```bash
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/aemo_dt_loss_history_surface_manifest.json \
  --evaluation-config <path-to-fixed-evaluator-config.json> \
  --output-dir eval_output/autoresearch/<run-tag>/<commit>
```

The evaluator reports training-loss summaries from the DT loss CSV and, for AEMO, held-out simulator rollouts, baseline-relative comparisons, safety counts, stability/risk metrics, and condition-sliced summaries.

## AEMO baseline tiers

1. **Fast proxy loop**: rank cheap ideas quickly; compact models and narrow slices are acceptable. Do not treat the result as the project baseline without evaluator confirmation. Keep every proxy comparison on the same pilot split.
2. **Learning baseline**: establish the checkpoint that future experiments branch from. Prefer broader train subsets, explicit held-out validation parquet files, and a materially longer context window. Compare with the immutable evaluator before broad sweeps.

For the current AEMO dataset layout:

- `aemo_dt_dataset_train_subset_007` is proxy-only.
- The fixed autoresearch pilot uses contiguous week-long slices per episode.
- Prefer one of the normal 24-episode train subsets plus explicit validation subsets/files for learning baselines.
- Prefer `context_len=180` for learning baselines (the `aemo_learning_baseline` preset default); see the `REPOSITORY_GUIDE.md` context-sweep note. `120` is an acceptable runtime fallback; `60` is primarily a proxy-loop setting.
- Wrapper launches can forward `context_len` and other approved DT shape knobs, but the direct trainer is still the right entrypoint for manual mixed-corpus command lines.
- Prefer `lr=3e-5` over `2e-5` as the starting learning-baseline LR unless new evaluator evidence contradicts it.

## Primary metric

Use one fast proxy metric consistently for the inner loop:

- AEMO proxy tier (`aemo_proxy`): `best_val_action_loss`, lower is better.
- Guardrail for AEMO proxy tier: `best_val_total_loss`.
- Broader baselines / non-proxy tiers: `best_val_total_loss`.

Read it from the DT loss CSV / surface manifest summary, not only from the final console line. If a run has no validation set, record that explicitly and treat comparisons as weak evidence. Validation loss is only a search proxy, not the final success criterion.

## Required evaluation beyond validation loss

For any promising checkpoint, also evaluate policy behaviour in held-out simulator rollouts that were not used for training or validation. At minimum report:

1. Held-out return / objective against the baseline controller.
2. Generalisation split with holdouts that differ by date range, region, weather pattern, or market regime.
3. Safety / constraint metrics: invalid actions, dispatch-limit violations, infeasible transitions, clipped actions, or simulator-defined rule breaches.
4. Stability metrics: variance across seeds or repeated rollouts, catastrophic failures, very short episodes, or extremely poor returns.
5. Baseline-relative regret against the current heuristic/controller, not just other DT checkpoints.

If validation loss improves but held-out simulator behaviour worsens, prefer the simpler/safer checkpoint or reject the change.

## Data coverage expectations

Small fixed pilot slices are acceptable for fast iteration but not for robust simulator performance. When planning or extending datasets:

- collect more full episodes if the current run uses only a narrow slice of behaviour
- add different date ranges and regions instead of sampling more from the same regime
- preserve a strict held-out evaluation set that is never used for training or validation tuning
- include rare and difficult operating conditions so the model does not only learn common easy cases

More simulated episodes help when they increase coverage of the target deployment distribution. More of the same narrow regime is less useful than broader coverage.

## Logging results

Append one line per experiment to the run ledger under `results/`:

```tsv
commit	track	metric	status	description
abc1234	household	1.234567	keep	baseline
def5678	household	1.210000	keep	increase context length to 120
987fedc	household	1.240000	discard	switch to wide variant
```

Status must be `keep`, `discard`, or `crash`.

## Experiment loop

After the run is authorised and setup is complete:

1. Check git state and confirm the intended autoresearch branch.
2. Re-establish whether the current branch point is a proxy-loop or learning-baseline checkpoint.
3. Make one focused change in `scripts/pretrain_decision_transformer.py`.
4. Record or commit the change if explicitly authorised for the run.
5. Run the fixed training command directly in the terminal so the live DT monitor stays visible. Mirror output with `tee` if a log file is also needed.
6. Read the proxy ranking metric from the loss CSV / surface manifest summary.
7. Record the result in the run ledger.
8. For AEMO proxy loops, prefer `configs/aemo_autoresearch_evaluator.mini.json` as the first simulator screen.
9. Run `scripts/autoresearch_evaluator.py` on the baseline checkpoint before interpreting later experiments.
10. Periodically rerun the evaluator on the strongest checkpoints.
11. If the proxy metric improves but the evaluator shows worse held-out objective, safety, or stability, do not automatically keep the change.
12. If both proxy and evaluator evidence improve, keep the commit and continue.
13. If the metric worsens, or the idea adds complexity without clear evaluator benefit, discard the experiment's own changes and preserve unrelated work.

For AEMO learning-baseline refreshes, prefer this order: broaden the data slice / explicit validation setup, move baseline LR to `3e-5`, compare longer contexts (`120` vs `288`), then test extra epochs or larger models.

## Simplicity rule

Prefer the smallest change that improves validation loss without hurting held-out simulator behaviour. If two changes perform about the same, keep the simpler one, the one preserving clearer adapter compatibility, and the one easier for a maintainer to understand.

## Crash handling

1. Inspect the log.
2. If the problem is a trivial bug in the current experiment, fix it and rerun once.
3. If the experiment is unstable, log `crash` and set that experiment aside.

Do not repeatedly dig deeper into broken ideas when the failure mode is obvious.

## Validation

Before and after autoresearch edits, use the repository test command when the environment supports it:

```bash
python -m pytest tests/ -v
```

If test dependencies are missing, note that clearly in the run log and continue only if the run remains authorised.

## Artifacts to watch

Expected outputs include model weights (`*.pt`), checkpoints, loss CSVs, checkpoint-segment loss CSVs, and the training-surface manifest JSON written next to the loss CSV. These artifacts are part of the expected contract and are useful for auditing changes.

## Stop conditions

Stop when:

- the operator stops the run
- required data is missing
- the environment cannot run the chosen training command
- repeated experiments stop producing useful ideas
- a predefined experiment budget or time budget is reached
