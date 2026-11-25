# Reproducing the AEMO preprint

This guide covers **Distilling a Planner into a Decision Transformer for Multi-Market Battery Dispatch**. It does not cover the separate [household benchmark](household/README.md).

## Find the evidence

- The [manuscript](../paper/manuscript/main.tex) explains the corrected method, evaluation protocol and limitations. The [2026-11-25 PPO retraining report](../paper/audit/ppo_retrain_20261125.md) is the source of truth for the corrected baseline comparison; the older E3/C3–C5 evidence remains historical provenance for the under-trained 204,800-step PPO comparator.
- Use [the seed aggregate artifact](../paper/audit/artifacts/ppo_retrain_seed_aggregate_20261125.json) for the current paper table. Its PPO result is the mean of `tuned_seed0/1/2` within each matched scenario cell. The [per-seed statistics](../paper/audit/artifacts/ppo_retrain_stats_20261125.json) preserve every individual checkpoint and the [impact seed artifact](../paper/audit/artifacts/ppo_retrain_impact_seeds_20261125.json) preserves seed-specific impact results.
- Raw simulator rollouts, market caches and most evaluator outputs live under `data/`, `models/` and `eval_output/`; they are not staged by a fresh checkout. The retraining scripts and compact summary artifacts make the aggregate auditable, but exact headline rollout reproduction still requires the external PPO checkpoints and processed AEMO caches.

## Verify the corrected three-seed headline

This CPU-only check requires only Python's standard library. It prints the canonical three-seed aggregate saved in the audit artifact; it does not rerun the simulator.

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("paper/audit/artifacts/ppo_retrain_seed_aggregate_20261125.json")
x = json.loads(p.read_text())
for name, row in x["surfaces"].items():
    print(name, f"n={row['n']}", f"DT/PPO={row['ratio_of_means']:.2f}x",
          f"PPO={row['ppo_mean']:,.0f}", f"diff CI={row['diff_ci']}")
i = x["impact"]
print("impact", f"DT/PPO={i['ratio_of_means']:.2f}x", f"wins={i['win_rate']:.3f}",
      f"diff={i['diff_mean']:,.0f}", f"CI={i['diff_ci']}",
      f"seed ratios={i['ppo_seed_ratios']}")
PY
```

Expected identity ratios are **1.70×, 1.20×, 1.69× and 1.60×**. Only dispatch-matched has a paired interval containing zero. The impact seed-mean ratio is **1.60×**, with 8/9 cell wins and Δ A$79,736 [A$29,842, A$132,933]; individual PPO-seed ratios are 2.26×, 0.91× and 2.96×. This verifies the released aggregate arithmetic, not raw market rollouts.

## Rerun the simulator evaluations

This is the optional, heavier route. The release state and dependencies are recorded in [Appendix C](../paper/manuscript/main.tex) and [`paper/environment.txt`](../paper/environment.txt). The evaluated code is the `preprint-v1` tag plus the same-day changes through `ea558bf`; the tag alone predates the final cache and surface changes.

1. Get `aemo_dt_sdp_jtsoc_v2cal.pt`, its `.pt.meta.json` sidecar, the `*_loss_surface_manifest.json` and the seasonal price profiles from the [model repository](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp). Keep the sidecar beside the weights: its `return_scale` is authoritative for inference and does not reproduce exactly from the available corpus. The published manifest has absolute paths from the original machine. Set `--model-path` and `--loss-csv-path` explicitly rather than relying on these paths.
2. Obtain the run inputs: the original loss CSV, the three tuned PPO checkpoints referenced in the retraining artifact (`tuned_seed{0,1,2}_4003k.zip`), and the AEMO market caches for the chosen surface. These are not included in a fresh Git checkout. The scripts under `scripts/ppo_retrain_*.py` retrain/evaluate them from the explicit data manifest. Dispatch replay needs the matched station's dispatch data; impact needs the corresponding market-impact configuration.
3. Use the extended configs below. The identity configs explicitly request `j_t_soc`, the branch selected by `rtg_mode="auto"` on price-taking surfaces. For example, with the original inputs staged under `models/aemo/dt/`, this runs the standard surface on CPU and writes to a new directory:

   ```bash
   CUDA_VISIBLE_DEVICES="" AEMO_CACHE_ONLY=1 python3 scripts/autoresearch_evaluator.py \
     --surface-manifest-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json \
     --model-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt \
     --loss-csv-path models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss.csv \
     --evaluation-config configs/aemo_autoresearch_evaluator.sdp_teacher_standard_year.json \
     --output-dir eval_output/reproduction/standard_2024 --device cpu
   ```

   | Surface | Config |
   | --- | --- |
   | Standard 2024 | [standard_year](../configs/aemo_autoresearch_evaluator.sdp_teacher_standard_year.json) |
   | Dispatch-matched | [dispatch_year](../configs/aemo_autoresearch_evaluator.sdp_teacher_dispatch_year.json) |
   | Expanded broad-2024 | [expanded_full](../configs/aemo_autoresearch_evaluator.sdp_teacher_expanded_full.json) |
   | 2025 out of distribution | [2025_fresh](../configs/aemo_autoresearch_evaluator.sdp_teacher_2025_fresh.json) |

   For dispatch-matched, `use_dispatch_asset_sizing` resolves the same 8 MWh / 30 MW Dalrymple North asset for all policies, as recorded in `eval_output/physics_v2/dispatch_year_v2cal/evaluation_summary.json`. The manuscript protocol and table caption now quote this effective matched asset.

The audit records the original [extended runs](../paper/audit/run_extended_surfaces.sh), [fresh 2025 run](../paper/audit/probe_fresh_2025.sh), [impact gate](../paper/audit/run_v2cal_impact.sh) and [teacher runs](../paper/audit/run_teacher_eval.sh). These scripts assume the original paths and inputs and can write to existing run directories. Use a separate working copy or new output and reference-cache paths when rerunning. The [cache finding](../paper/audit/artifacts/cache_data_staleness_finding.md) explains why earlier cached comparisons must not be used; small variation in PPO rollouts means exact replay of all profits is not expected.

**Corpus regeneration is separate from evaluation.** In the [Hugging Face dataset repository](https://huggingface.co/datasets/mrvictoru/AEMO_simulated_trade_sdp), the paper’s file is `dt_trajectories_jtsoc_v2cal_conservative.parquet` (SHA-256 `14c94ab9c7aa7db3e9a55cf8e245cbb72ab10c9f2666403e4917801eee3cf883`). The superseded `dt_trajectories_jtsoc_v2_conservative.parquet` has been removed publicly. Keep it only in local historical/reproducibility records if needed. The [SOC-waypoint model](https://huggingface.co/mrvictoru/energydecision-waypoint-dt) and [trajectory generator](../scripts/generate_sdp_dt_trajectories.py) are relevant to corpus generation only, not to the shipped policy's action loop.
