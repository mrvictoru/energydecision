# Repository guide

Audience: human contributors, maintainers, and coding assistants. This guide replaces the former `AGENTS.md`. It is not auto-loaded by tooling; tool-specific instruction files remain under `.github/` and `.opencode/`.

## Start here

- [Project overview](README.md): current preprint and separate household benchmark
- [Documentation hub](docs/README.md): navigation for all guides and reference docs
- [Preprint reproduction guide](docs/reproduction.md): current claims, audit artifacts and simulator prerequisites
- [Development guide](docs/development.md): environment setup, runtime modes and tests
- [Architecture](docs/architecture.md): repository structure
- [Known issues](docs/known_issues.md): modelling caveats and stale-command history
- [Paper directory](paper/README.md): manuscript, checklist and audit evidence
- [Audit evidence](paper/audit/README.md): current versus historical provenance

## Two tracks, never compare across them

For current claims and artifact availability, see [the reproduction guide](docs/reproduction.md). Older experiment settings below are retained for engineering context.

- **Household / Ausgrid**: `SolarBatteryEnv`. On the legacy benchmark the DT beats the Oracle; on the modern 2019+ rebuild the efficiency-matched perfect-foresight oracle dominates (best DT gross +$293–357/yr vs oracle +$690/yr). Only the unconstrained 7-day real-OOD DT is net-positive (+$83–93/yr); constrained/long-horizon surfaces are net-negative and clip at SOC rails. See `docs/household/workflow.md` §H4.9 and `docs/known_issues.md` A6/A9.
- **AEMO / grid-scale**: `AEMOBatteryTradingEnv`, NEM energy plus eight FCAS markets. The shipped physics-v2 Stage C Decision Transformer (`models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt`) is a planner-distilled, solver-free-per-decision candidate with a precomputable `J_t(soc)` prompt. A 2026-11-25 retrain corrected the PPO reference. Against three tuned 4M PPO seeds, the within-cell DT/PPO ratios are **1.70× / 1.20× / 1.69× / 1.60×** (n=30/12/30/6); dispatch's paired CI includes zero and one PPO seed reverses three surface means. Under market impact the seed-mean ratio is **1.60× with 8/9 wins**, but seed-specific ratios span **0.91–2.96×**. Do not describe the DT as universally preferred or claim a stable 9/9 gate pass. Current evidence: `paper/audit/ppo_retrain_20261125.md`, `paper/audit/artifacts/ppo_retrain_seed_aggregate_20261125.json`, `paper/audit/provenance.tsv` rows P4/P5, and `docs/reproduction.md`. Earlier v2cal and v1 ratios are historical comparisons to the old 204,800-step default PPO.

Household caveat: buckets A+B are re-baselined; bucket C (live shadow, spot pass-through, broader telemetry) remains open.

## Repository conventions

- Single package, flat namespace. No `setup.py` or `pyproject.toml`; `src/` has no `__init__.py`, so modules import each other directly.
- Tests add `src/` via `sys.path.insert(0, ...)` in `tests/conftest.py`; no `PYTHONPATH` is needed.
- No linter, formatter, CI, coverage, or Makefile. Only `pytest.ini`. Run `python -m pytest tests/ -v` for the full suite.
- Distrobox is preferred: run CLI entry points from the repo root with `python3 scripts/...`; use `data/...` and `models/...` paths.
- Docker Compose workdir is `/code/src`: scripts are bare names and paths use `../data/...`.
- If paths look wrong, check which runtime is active.

## Constrained experiment runs

The [experiment protocol](docs/experiment_protocol.md) applies only to explicitly requested autoresearch runs. It is not general maintenance or paper reproduction.

- The editable training surface is `scripts/pretrain_decision_transformer.py`.
- For AEMO, call it through `scripts/pretrain_aemo_decision_transformer.py`, or prefer the tier launcher:

```bash
python3 scripts/launch_aemo_training.py --run-tier proxy-baseline
python3 scripts/launch_aemo_training.py --run-tier learning-baseline
```

The launcher derives tier defaults, writes a launch plan, and re-enters Distrobox from the host. Everything outside the editable surface is read-only during a constrained run.

## Historical engineering context

These settings are retained for context, not as current paper claims.

| Topic | Historical setting |
|---|---|
| Expanded-eval DT | 8×512, ctx=180, dropout=0.15, batch=16, lr=3e-5, epochs=2, discount=0.95, return_scale=2.0 |
| FCAS-rich DT | 8×384, ctx=180, dropout=0.15, batch=64, lr=3e-5, epochs=2, discount=0.95, return_scale=2.0 |
| Context sweep | ctx=180 best on pilot data; longer contexts regressed; ctx=2016 is feasible at batch ≤16 on 22 GB |
| Proxy metric | AEMO proxy tier: `best_val_action_loss`; broader tiers: `best_val_total_loss` |
| RTG prompts | Evaluate multiple RTG values; the DT responds to the prompt |
| Batch size | 64 default for proxy/learning tiers; 16 for 8×512 expanded eval |
| Checkpoint resume | `--context-length` must match the checkpoint |

## GPU and crash telemetry

- 22 GB VRAM (RTX 2080 Ti), single GPU.
- Xid79 / GPU-lost crashes have occurred; no recurrence after the PCIe clock cap.
- For long GPU runs use:

```bash
bash scripts/run_full_learning_baseline.sh <TAG>
```

It captures `nvidia_smi`, `dmon`, `vmstat`, `iostat` and kernel logs, writes `SAFE_TO_SHUTDOWN.txt` on completion and `CRASH_DETECTED.txt` on Xid79. Shut down only after `SAFE_TO_SHUTDOWN.txt` exists.

## Data and caching

- AEMO data is fetched via NEMOSIS (slow, ~30 min). Use `AEMO_CACHE_ONLY=1` to force cache.
- Manual generator files go in `data/aemo/manual/`.
- Cache prewarmer: `python3 scripts/prewarm_aemo_cache.py --evaluation-config <config>`
- Original DT corpus: `data/aemo_dt/aemo_dt_dataset.parquet` (162 episodes across 5 regions).
- FCAS-rich historical corpus: `data/aemo_dt_fcas/aemo_fcas_dataset.parquet` (2,425 episodes, 78.4M rows, 3.1 GB), generated by `scripts/generate_fcas_dataset.py`. It contains PPO (905 eps), TD3/A2C/DDPG/SAC (300 eps each), 300 FCAS-rule eps and 20 old-rule eps. Fetch regional data first with `scripts/fetch_aemo_region.py` inside Distrobox. This corpus is historical; the shipped Stage C DT uses SDP-teacher trajectories.
- SDP-teacher trajectories: `data/aemo_dt_sdp/`. `dt_trajectories_jtsoc_v2cal_conservative.parquet` is the shipped 320-episode corpus; `dt_trajectories_jtsoc_combined.parquet` is the historical v1 corpus. Generated by `scripts/generate_sdp_dt_trajectories.py` using `src/aemo_sdp_executor.py`.
- Autoresearch pilot: `data/aemo_dt_fcas/autoresearch_pilot/` (8 train + 4 val episodes), rebuilt with `python3 scripts/build_aemo_fcas_pilot_spec.py --build-pilot`.

## Evaluator tiers

The autoresearch evaluator validates DT checkpoints in held-out simulator environments, not only by validation loss.

| Config | Use | Step | Hours | Regions | Batteries | Baselines | Workers | Wall clock |
|---|---|---|---:|---|---|---|---:|---|
| `mini` | Smoke / quick screening | 5 min | 24 h | NSW1, SA1 | medium | DT, FCAS rule | 4 | ~2 min |
| `example` | Promotion check | 5 min | 144 h | NSW1, SA1, QLD1, VIC1 | medium, small | DT, rule, FCAS rule, 2× dispatch, PPO | 8 | ~15 min |
| `expanded` | Full season sweep | 5 min | 288 h | All 5 regions × 6 months | medium | DT, rule, PPO | 8 | ~60 min |
| `dispatch_matched` | GRPO vs dispatch replay | 5 min | 144 h | SA1 (4 seasons) | dispatch-asset-sized | DT, dispatch, FCAS rule | 8 | ~15 min |

All AEMO DT evaluation configs use five-minute steps to match the training data. The `mini` evaluator uses `fcas_rule` as the reference baseline. Non-DT policy rollouts are cached in `eval_output/autoresearch/reference_cache/`; prewarm with `scripts/prewarm_aemo_cache.py`.

## Experiment tracking

- `results.tsv` is the tracked historical ledger: `commit\ttrack\tmetric\tstatus\tdescription`.
- Statuses: `keep`, `discard`, `crash`.
- New scratch runs belong under ignored `results/`.
- Autoresearch uses fresh `autoresearch/<tag>` branches. Establish a baseline first.

## Action dimensions

| `action_mode` | `act_dim` | Description |
|---|---:|---|
| `simple` | 1 | Energy-only charge/discharge |
| `multi_market` | 3 | Legacy energy + RAISEREG/LOWERREG |
| `full_fcas` | 9 | Energy + all eight FCAS services |

`h_dim` must be divisible by `n_heads`.

## FCAS rule

`AEMOAgent(algorithm='fcas_rule')` in `src/decision.py` uses the original rule for energy dispatch and adds percentile-based FCAS bidding.

- Charge ≤ $30, discharge ≥ $120, with SOC guardrails.
- Bid 1.0 on RAISEREG/LOWERREG when the current price exceeds a data-driven percentile threshold (default p80) and SOC headroom permits.
- Thresholds come from the environment's own `aemo_data` at agent initialisation; no trained model is required.

```python
agent = AEMOAgent(env, algorithm='fcas_rule', fcas_pctile=0.80)
```

Explicit thresholds can be set with `fcas_raise_threshold` and `fcas_lower_threshold`. Historical NSW1 12-day medium result: $2,941 FCAS revenue per episode versus $0 for the original rule.

## Historical models and GRPO

The shipped preprint checkpoint is `aemo_dt_sdp_jtsoc_v2cal.pt` plus its `.pt.meta.json` sidecar. See [the reproduction guide](docs/reproduction.md) for downloads and dependencies. The legacy and modern-v2 Hugging Face checkpoints, and GRPO post-training, are historical. GRPO did not improve the modern v2 model; the legacy Phase 1 GRPO result was a narrow overfit. See `report.md` §8.2.7 and `docs/grpo_experiments.md` for history.

## Historical improvement priorities

The Stage C SDP-teacher + `J_t(soc)` work superseded the earlier priorities: PPO-generated trajectories, RTG calibration sweeps, corpus scale-up, FCAS-weighted loss, and context re-sweeps. See `docs/aemo_dt_preferred_policy_plan.md` and `docs/FUTURE_PLAN.md` for the archived plan. No universal policy preference is claimed.
