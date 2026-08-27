# energydecision

energydecision is a research codebase and benchmark for battery control across two related but distinct tracks:

- Household solar-battery control with `SolarBatteryEnv`
- Grid-scale AEMO/NEM battery trading with `AEMOBatteryTradingEnv`

The repository combines simulation environments, classical planning baselines, online RL, offline Decision Transformer training, and evaluation tooling. It is intended to support reproducible experiments and to serve as a base for future research work.

## Start Here

- [docs/README.md](docs/README.md): human documentation hub
- [docs/architecture.md](docs/architecture.md): codebase map and system overview
- [docs/development.md](docs/development.md): setup, runtime modes, testing, and contributor workflow
- [report.md](report.md): research report and current benchmark narrative

## Choose Your Track

### Household

Use the household track if you want residential PV + battery control under household load and tariff dynamics.

- Overview: [docs/household/README.md](docs/household/README.md)
- Workflow: [docs/household/workflow.md](docs/household/workflow.md)
- Environment reference: [docs/household/environment.md](docs/household/environment.md)
- Degradation reference: [docs/household/degradation.md](docs/household/degradation.md)
- Modern-data rebuild plan: [docs/FUTURE_PLAN.md](docs/FUTURE_PLAN.md) §6b — real 2019+ telemetry, planner distillation, replay-gap study

### AEMO / Grid-Scale

Use the AEMO track if you want grid-scale battery trading in energy and FCAS markets.

- Overview: [docs/aemo/README.md](docs/aemo/README.md)
- Workflow: [docs/aemo/workflow.md](docs/aemo/workflow.md)
- Environment reference: [docs/aemo/environment.md](docs/aemo/environment.md)
- Evaluation: [docs/evaluation_guide.md](docs/evaluation_guide.md)

### Roadmap
*   [x] **Core:** Gymnasium environment & Rule-based agents.
*   [x] **Optimization:** SDP & MRDP solvers.
*   [x] **Online RL:** Training loop with SB3.
*   [x] **Offline RL:** Decision Transformer training loop.
*   [x] **Evaluation:** Metrics for return, risk proxies (Sharpe/Sortino), and degradation.
*   [x] **Grid Market:** AEMO Environment Implementation.
*   [x] **Grid battery degradation modelling:** Rainflow-based degradation and capacity fade in `AEMOBatteryTradingEnv`.
*   [x] **Real-world BESS degradation model:** Combined calendar + cycle aging model (`RealWorldBESSDegradationModel`) for utility-scale BESS, with Arrhenius temperature dependency and NMC/LFP chemistry presets, adapted from the framework in Kampker et al. (2025, doi:10.3390/batteries11110392).
*   [x] **Risk-sensitive evaluation:** CVaR/VaR tail-risk metrics (`var_5`, `cvar_5`) computed by `evaluate_experiment_logs`.
*   [x] **Statistical comparisons:** Bootstrap confidence intervals (`bootstrap_confidence_intervals`) and paired Wilcoxon signed-rank tests (`paired_comparison`) across customers/seeds.
*   [x] **Conduct data gathering and training on AEMO env:** Run dispatch replay and RL-agent in `AEMOBatteryTradingEnv` to collect trajectories for offline training for DT and evaluation.
*   [x] **Test out autoresearch:** Run the autoresearch loop for DT training.
*   [x] **DT prompt calibration:** Use `recommended_rtg` / `recommended_return_scale` diagnostics to choose in-distribution prompts; calibrate against the target held-out scenario before inference.
*   [x] **FCAS-aware offline data collection:** Generate a 2,425-episode FCAS-rich dataset (`data/aemo_dt_fcas/aemo_fcas_dataset.parquet`) from PPO, TD3, A2C, DDPG, SAC, and `fcas_rule` policies.
*   [x] **FCAS-rich DT training:** Retrain DT on the FCAS-rich dataset — DT now achieves **+$1,522/ep profit**, beating PPO (+$1,444/ep) on the example evaluator (Section 8.6.2 of `report.md`).
*   [x] **RL Fine-tuning:** GRPO Phase 1 support for pretrained DT weights is now available through the current CLI workflow, including mixed-bound action distribution for `full_fcas`, adaptive RTG sampling, periodic reference syncing, and degradation-weighted reward shaping.
*   [x] **Hyperparameter Tuning:** using Autoresearch for DT.
*   [x] **Forecast-conditioned DT (negative result):** Built and evaluated a ForecastDecisionTransformer with 48-step TTM forecast tokens. Result: $4,564/ep vs modern v2's $4,991/ep — explicit forecasts do not beat implicit context. See report.md §8.2.8.
*   [x] **Market-impact BESS evaluation:** Piecewise-linear merit-order impact model (`src/market_impact.py`) hooking into `AEMOBatteryTradingEnv` (backward-compatible `identity` default). Phase 3: v2 DT under impact retains 62/83/49% of identity profit at 8/150/250 MW vs PPO's 62/40/32%. Phase 4: impact-aware DT retrain (`mrvictoru/energydecision-dt-v2-impact`) beats PPO significantly (+$115K/cell, p=0.004) and edges the naive v2 6/9 cells (+$96K/cell). See report.md §8.2.9.
*   [x] **AEMO Oracle upper bound:** Perfect-foresight LP co-optimizer (`src/aemo_oracle_algo.py`) as evaluator baseline; Oracle_MI (impact-aware LP) is the impact ceiling, but its fixed-point solve at 150 MW+ exceeds 100% of PT (unreliable at large scale). Invariant-validated as a *revenue* ceiling (dominates every policy on 9/9 Phase-3 cells and 6/6 dispatch-matched episodes); under real-world degradation its net profit is beaten by degradation-aware DTs on small batteries (LP is degradation-blind) — see report §8.2.9.3.
*   [x] **AEMO preferred-policy recommendation (verified 2026-08):** The standalone DT ships with **surface-aware `rtg_mode="auto"`** (SDP-teacher distillation + J_t(soc) prompting). On all 4 identity surfaces: standard **$11.6k/ep vs PPO $2.35k** (4.9×), dispatch-matched **$35.3k vs $22.5k** (1.57×), expanded broad-2024 **$34.8k vs $19.5k** (1.78×), 2025 OOD **$25.9k vs $6.5k** (3.98×). Passes the impact gate (2.5–3.1× under merit-order impact). Weights + training corpus on Hugging Face: [mrvictoru/energydecision-dt-v2-sdp](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp). See `docs/aemo_dt_preferred_policy_plan.md` and `report.md §8.2.10`.
*   [ ] **Sim-to-real readiness (highest priority):** Add safety wrappers and evaluate policies with hardware-in-the-loop (where available). The real path to operation; the DT is positioned for its winning surfaces (impact, dispatch-matched, mild markets, FCAS/degradation efficiency).
*   [ ] **Artifact provenance:** Add lightweight checksums/config logging for datasets, models, and evaluation outputs.
*   [ ] **Offline dataset studies (de-prioritized — data-ceiling):** Evaluate DT sensitivity to behavior-policy mixtures. The 2026 session showed re-composition only partially helps (+23% FCAS from FCAS-heavy policies); the DT-vs-PPO gap is a data ceiling, not a mixture-tuning artifact.
*   [ ] **Long-context DT experiments (de-prioritized):** Study larger `context_len` and RoPE for seasonal/weekly structure.
*   [ ] **Multi-agent extension:** Microgrid setting with multiple households and coordination.
*   [x] **Statistical confidence on AEMO headlines:** Bootstrap confidence intervals and paired Wilcoxon signed-rank tests applied to the market-impact headline tables (n=9 cells, §8.2.9.1) and to the per-surface profit headlines — dispatch-matched ($10,138; n=6 episodes, Jul–Dec 2024) and standard ($4,630; n=15, Sep–Nov 2024) — via `scripts/phase3_bootstrap_over_scenarios.py`, `phase3_paired_wilcoxon.py`, and the evaluator's built-in bootstrap.
*   [ ] **Full_fcas broad-surface evaluation:** Run expanded surface with 9-dim `full_fcas` actions (currently uses 3-dim `multi_market`); closes disclosed protocol asymmetry.
*   [ ] **Sim-to-real settlement backtest:** Validate simulator revenue against AEMO settlement data for Dalrymple North; produce calibration curve.
*   [ ] **Safety-constrained policies:** Constrained policy with degradation budget, SOC guardrails, FCAS enablement limits (CMDP formulation).
*   [ ] **Offline-Q baseline (IQL/CQL):** Implement and compare to planner-distillation on the same FCAS-rich corpus + impact surface.
*   [ ] **Impact-aware J_t(soc):** Recompute cost-to-go with post-impact realized prices; validate on grid-scale batteries under merit-order.
*   [ ] **Multi-agent NEM extension:** Multi-BESS coordination under merit-order impact (PettingZoo integration).
*   [ ] **Paper pipeline:** Benchmark paper (Datasets & Benchmarks track) + Method paper (planner distillation + impact gate).
*   [ ] **Household modern-data rebuild (H0–H3):** Ingest real 2019+ household telemetry (solar + home battery, privacy-gated); re-establish benchmark under modern tariff economics; port the AEMO playbook (SDP distillation + cost-to-go prompting); real-battery replay-gap analysis. See `docs/FUTURE_PLAN.md` §6b.
*   [x] **Synthetic diverse-household corpus (H1.5):** Whole-day clustered recomposition with five household archetypes, capped EV/AC/pool appliance injection, solar/battery scaling, automated G1–G6 validation, and reproducible train/val/test manifests. See `scripts/build_household_synth_corpus.py` and `docs/household/workflow.md`.

## Quick Setup

### Preferred runtime

The repo is primarily documented around two runtime modes:

- Distrobox from the repo root on Linux
- Docker Compose with a shell inside the running container

The detailed setup and path conventions are in [docs/development.md](docs/development.md).

### Install dependencies locally

```bash
pip install -r requirements.txt
pip install -r torch_req.txt
```

## Fastest Common Workflows

### Household workflow

1. Generate or inspect household logs under `data/household/logs/`.
2. Train a Decision Transformer with:

```bash
python scripts/pretrain_decision_transformer.py \
  --data-dir data/household/logs \
  --patterns train_episode_01 train_episode_02
```

3. Validate with:

```bash
python -m pytest tests/ -v
```

### AEMO workflow

1. Read [docs/aemo/workflow.md](docs/aemo/workflow.md) to choose notebook-first or CLI-first execution.
2. For canonical CLI training, use:

```bash
python scripts/launch_aemo_training.py --run-tier proxy-baseline
python scripts/launch_aemo_training.py --run-tier learning-baseline
```

3. Evaluate checkpoints with:

```bash
python scripts/autoresearch_evaluator.py \
  --surface-manifest-path <surface-manifest.json> \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json \
  --output-dir eval_output/autoresearch/<run-tag>
```

## Repository Layout

```text
configs/      Model configs and evaluator configs
data/         Cached data, generated datasets, and logs
docs/         Human documentation
eval_output/  Evaluation outputs and reports
notebooks/    Exploratory and notebook-first workflows
scripts/      Canonical runnable entrypoints
src/          Reusable implementation modules
tests/        Pytest suite
```

## Research Artifacts

- Report: [report.md](report.md)
- Research notes index: [docs/research/README.md](docs/research/README.md)
- Hugging Face models:
  - [Stage C Decision Transformer (shipped, SDP-teacher distilled)](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp) — `aemo_dt_sdp_jtsoc_fullcorpus.pt`; also mirrored in-repo at `models/aemo/dt/`
  - [Modern v2 Decision Transformer (8×768 GQA, historical)](https://huggingface.co/mrvictoru/energydecision-dt-v2)
  - [Forecast-conditioned Decision Transformer (negative result)](https://huggingface.co/mrvictoru/energydecision-dt-v2-forecast)
  - [Pretrained Decision Transformer v1 (legacy)](https://huggingface.co/mrvictoru/energydecision-dt)
  - [GRPO finetuned Decision Transformer v1 (legacy)](https://huggingface.co/mrvictoru/energydecision-dt-grpo)
- Hugging Face datasets:
  - [AEMO SDP-teacher trajectories (Stage B corpora + shipped J_t(soc) combined corpus)](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade_sdp)
  - [AEMO simulated trade episodes (FCAS + SDP + GRPO + TTM forecasts, behaviour-cloning era)](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade)

## Notes For Contributors

- Treat `scripts/` as the canonical CLI surface.
- Treat `src/` as reusable implementation modules.
- Keep household and AEMO results conceptually separate.
- Prefer adding stable operational guidance to [docs/development.md](docs/development.md) and [docs/architecture.md](docs/architecture.md) rather than expanding this README.