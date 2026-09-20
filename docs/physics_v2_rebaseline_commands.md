# Physics-v2 Re-baseline Commands (P5)

> **Superseded by the final calibration (2026-09-15/20).** The commands below
> were the *first* P5 pass and produced the `*_v2` / `h4_v2` artifacts. The
> shipped results use the **calibrated** variants: AEMO `--deg-calibration 0.12`
> → `aemo_dt_sdp_jtsoc_v2cal.pt` (corpus
> `dt_trajectories_jtsoc_v2cal_conservative.parquet`); household
> `deg_mode="step"` + `deg_calibration=0.29` → `h4_v2c`
> (`data/household/dt/sdp_teacher_h4_v2c_*`). See `known_issues.md` A6/B4,
> `report.md` §8.2.11, and `docs/household/workflow.md` §H4.9 (buckets A/B).

Exact commands for the A1–A8 "degradation-physics v2" re-baseline. All commands
run from the repo root inside Distrobox `energydecision-gpu`. Outputs are
written to **new** `*_v2` paths so the shipped artifacts are preserved for
comparison.

See `docs/known_issues.md` §A and §D for the underlying issues and scope.

---

## 1. AEMO — regenerate SDP-teacher trajectories

A5 (state-dependent planner degradation) changes the honest SDP teacher's
planned energy; regenerating is required before retraining.

Conservative teacher (λ_deg = 50 $/MWh), Stage C J_t(soc) RTG:

```bash
python3 scripts/generate_sdp_dt_trajectories.py \
  --regions NSW1 QLD1 SA1 TAS1 VIC1 \
  --horizons short medium \
  --batteries medium_1c fast_375c large_07c small_05c \
  --episodes-per-slot 8 \
  --deg-cost-per-mwh 50.0 \
  --deg-calibration 0.12 \
  --rtg-mode j_t_soc \
  --out data/aemo_dt_sdp/dt_trajectories_jtsoc_v2_conservative.parquet
```

Aggressive teacher (λ_deg = 20 $/MWh):

```bash
python3 scripts/generate_sdp_dt_trajectories.py \
  --regions NSW1 QLD1 SA1 TAS1 VIC1 \
  --horizons short medium \
  --batteries medium_1c fast_375c large_07c small_05c \
  --episodes-per-slot 8 \
  --deg-cost-per-mwh 20.0 \
  --deg-calibration 0.12 \
  --rtg-mode j_t_soc \
  --out data/aemo_dt_sdp/dt_trajectories_jtsoc_v2_aggressive.parquet
```

Combine (640 eps) for the Stage C corpus:

```bash
python3 - <<'PY'
import polars as pl, glob
files = sorted(glob.glob("data/aemo_dt_sdp/dt_trajectories_jtsoc_v2_*.parquet"))
df = pl.concat([pl.read_parquet(f) for f in files])
df.write_parquet("data/aemo_dt_sdp/dt_trajectories_jtsoc_v2_combined.parquet")
print(df.height, df["episode_id"].n_unique())
PY
```

Observed rate (2026-09-13): short slots ~10 s, medium slots ~45 s; ~320 eps/run
≈ 2–3 h.

## 2. AEMO — retrain Stage C

Use the shipped Stage C architecture config and `--rtg-source j_t_soc` with
auto return-scale, writing to a `_v2` checkpoint (mirrors the original recipe:
3 epochs, batch 16, stride 105, lr 3e-5, `aemo_decision_transformer_model_kwargs_sdp_jtsoc_fullcorpus.json`):

```bash
python3 scripts/pretrain_aemo_decision_transformer.py \
  --dataset-path data/aemo_dt_sdp/dt_trajectories_jtsoc_v2_combined.parquet \
  --model-config configs/aemo_decision_transformer_model_kwargs_sdp_jtsoc_fullcorpus.json \
  --rtg-source j_t_soc --auto-return-scale \
  --context-length 210 --stride 105 --batch-size 16 --epochs 3 --lr 3e-5 \
  --save-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2.pt \
  --checkpoint-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2_ckpt.pt \
  --loss-csv-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2_loss.csv
```

(Confirm flags against `scripts/pretrain_decision_transformer.py --help`; the
surface manifest written beside the run records the exact knobs.)

## 3. AEMO — re-run evaluation surfaces

Identity surfaces + 2025 OOD via `autoresearch_evaluator.py` using the
`sdp_teacher_*` configs (standard, dispatch, expanded, 2025):

```bash
python3 scripts/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2_loss_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.sdp_teacher_standard.json \
  --output-dir eval_output/physics_v2/standard
# ...repeat for sdp_teacher_dispatch / sdp_teacher_expanded / sdp_teacher_2025
```

Impact gate (with the B8 fix now applying the sidecar return_scale):

```bash
python3 scripts/phase3_impact_eval.py \
  --impact-config configs/impact_benchmark.json \
  --checkpoint models/aemo/dt/aemo_dt_sdp_jtsoc_v2.pt \
  --config configs/aemo_decision_transformer_model_kwargs_sdp_jtsoc_fullcorpus.json \
  --label stagec_v2 --rtg-mode auto
```

Quick j_t_soc-vs-constant check (cached scenarios, ~20 min):

```bash
python3 scripts/verify_jtsoc_impact_sign.py --device cuda \
  --checkpoint models/aemo/dt/aemo_dt_sdp_jtsoc_v2.pt \
  --out eval_output/phase3_impact/jtsoc_sign_verification_v2.json
```

## 4. Household — regenerate SDP-teacher corpora

A6 (λ_deg=50), A2 (RTE=0.80), A1 (calendar in `full`) now apply.

```bash
python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1 --split train \
  --roundtrip-eff 0.80 --deg-cost-per-mwh 50.0 \
  --degradation-mode full \
  --out data/household/dt/sdp_teacher_h4_v2_train.parquet

python3 scripts/generate_household_sdp_trajectories.py \
  --synth-dir data/household/synth_h4_1 --split val \
  --roundtrip-eff 0.80 --deg-cost-per-mwh 50.0 \
  --degradation-mode full \
  --out data/household/dt/sdp_teacher_h4_v2_val.parquet
```

## 5. Household — retrain + re-evaluate

Retrain with the H4 recipe (8×512, ctx 576, stride 288, weights
0.999/0.002/0.0001), then re-run the real-OOD surface:

```bash
python3 scripts/pretrain_decision_transformer.py \
  --data-dir data/household/dt --patterns sdp_teacher_h4_v2_train \
  --val-data-dir data/household/dt --val-patterns sdp_teacher_h4_v2_val \
  --split-policy explicit_validation \
  --context-length 576 --stride 288 --n-block 8 --h-dim 512 --n-heads 8 \
  --drop-p 0.15 --batch-size 16 --epochs 5 --lr 3e-5 --seed 42 \
  --rtg-source constant --return-scale 1.0 \
  --action-loss-weight 0.999 --state-loss-weight 0.002 --return-loss-weight 0.0001 \
  --save-path models/household/dt/h4_v2_persistence_8x512_ctx576.pt \
  --checkpoint-path models/household/dt/h4_v2_persistence_8x512_ctx576_ckpt.pt \
  --loss-csv-path models/household/dt/h4_v2_persistence_8x512_ctx576_loss.csv

python3 scripts/evaluate_household_ood_baselines.py \
  --normalized-dir data/household/real/normalized \
  --output-dir eval_output/household/h4_v2_real_ood \
  --dt-path models/household/dt/h4_v2_persistence_8x512_ctx576.pt \
  --dt-config models/household/dt/h4_v2_persistence_8x512_ctx576_model_kwargs.json \
  --tariff realistic --forecast-mode persistence --window-days 7 --workers 8
```

## 6. Wrap-up

- Update `report.md` §8.1.x/§8.2.10, `docs/FUTURE_PLAN.md`, and
  `docs/known_issues.md` with the re-baselined numbers; mark the pre-v2 results
  superseded.
- Record new `results.tsv` rows for each surface.
