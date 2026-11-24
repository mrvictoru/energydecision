# Reproducing the AEMO preprint

This guide covers **Distilling a Planner into a Decision Transformer for Multi-Market Battery Dispatch**. It does not cover the separate [household benchmark](household/README.md).

## Find the evidence

- The [manuscript](../paper/manuscript/main.tex) explains the method, evaluation protocol and limitations. The [provenance table](../paper/audit/provenance.tsv) maps claims to evidence: E3 for the extended identity surfaces, C3–C5 for the impact gate, D2 for dispatch replay, and T1 for teacher retention.
- The [tracked audit artifacts](../paper/audit/artifacts/) contain the published aggregates and impact-cell data. Use the `v2cal_extended` set in the [identity statistics](../paper/audit/artifacts/stagec_v2cal_extended_significance.json), not the `v2cal` canonical small-sample set or `v1_fullcorpus`. Use `v2cal_shipped_fixed_rtg0` in the [impact analysis](../paper/audit/artifacts/stagec_v2cal_impact_analysis.json), not the best-prompt-per-cell sweep.
- Raw simulator rollouts, market caches and most evaluator outputs live under `data/`, `models/` and `eval_output/`; they are not staged by a fresh checkout. Some historical outputs are tracked. The [dated audit](../paper/audit/audit_findings.md) records how earlier claims were corrected; see §7 for which surfaces became the preprint headline and §8 for the scale discrepancy.

## Verify the market-impact headline

From the repository root, recompute the paired result and the three ratios of means from the nine published cell pairs. Only NumPy and SciPy are needed (`python3 -m pip install numpy scipy` in a virtual environment); no simulator, weights, credentials or market data are required. The command reads the artifact and prints results without rewriting any files.

```bash
python3 - <<'PY'
import json
from pathlib import Path
from paper.audit.analyse_v2cal_impact import paired, ratio_of_means

audit = Path("paper/audit/artifacts/stagec_v2cal_impact_analysis.json")
cells = json.loads(audit.read_text())["v2cal_shipped_fixed_rtg0"]["paired"]["cells"]
dt = {tuple(key.split("|")): cell["dt"] for key, cell in cells.items()}
ppo = {tuple(key.split("|")): cell["ppo"] for key, cell in cells.items()}
result = paired(dt, ppo)
lo, hi = result["diff_ci"]
wins = sum(dt[key] > ppo[key] for key in dt)
print(f"{wins}/{result['n']} wins; mean advantage A${result['diff_mean']:,.0f}/episode;")
print(f"95% CI [A${lo:,.0f}, A${hi:,.0f}]; Wilcoxon p={result['wilcoxon_p']:.4f}")
for battery, ratio in ratio_of_means(dt, ppo).items():
    print(f"{battery}: {ratio:.2f}×")
PY
```

Expected: **9/9 wins**, **A$167,099/episode**, **95% CI [A$101,213, A$227,362]**, **p=0.0039**. The fixed-fallback ratios are **4.36× small**, **4.70× Hornsdale** and **4.67× Torrens**. This verifies the statistics from saved profits; it does not recreate those profits from market data.

## Rerun the simulator evaluations

This is the optional, heavier route. The release state and dependencies are recorded in [Appendix C](../paper/manuscript/main.tex) and [`paper/environment.txt`](../paper/environment.txt). The evaluated code is the `preprint-v1` tag plus the same-day changes through `ea558bf`; the tag alone predates the final cache and surface changes.

1. Get `aemo_dt_sdp_jtsoc_v2cal.pt`, its `.pt.meta.json` sidecar, the `*_loss_surface_manifest.json` and the seasonal price profiles from the [model repository](https://huggingface.co/mrvictoru/energydecision-dt-v2-sdp). Keep the sidecar beside the weights: its `return_scale` is authoritative for inference and does not reproduce exactly from the available corpus. The published manifest has absolute paths from the original machine. Set `--model-path` and `--loss-csv-path` explicitly rather than relying on these paths.
2. Obtain the run inputs: the original loss CSV (the evaluator's training-summary input), the PPO reference `models/aemo_sb3/ppo_aemo_fcas_model.zip`, and the AEMO market caches for the chosen surface. These are not included in the GitHub checkout or the shipped DT model repository. The [AEMO workflow](aemo/workflow.md) explains cache preparation. Dispatch replay also needs the matched station's dispatch data; the impact gate needs the corresponding market-impact data and configuration.
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
