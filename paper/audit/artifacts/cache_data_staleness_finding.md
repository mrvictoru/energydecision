# Reference-cache and processed-data staleness — Phase 1 finding

**Date:** 2026-09-22 · **Frozen release:** `preprint-v1`

## Summary

The evaluator's reference-rollout cache (`reference_cache_dir`) **persisted across
data, preprocessing, and physics changes**, because the cache key hashes the policy
*config* (including the model *path*), scenario, battery variant, and heldout params —
but **not a content hash of the PPO checkpoint or of the processed price/FCAS data**.
Canonical surfaces therefore reused stale PPO/fcas-rule baselines while the DT
(never cached) was recomputed fresh. Separately, the evaluator **rewrites
`data/aemo/processed_*.parquet`** on each run, and some canonical surfaces used
earlier versions of those files.

## Evidence

### 1. Canonical runs used pre-fix caches

| Canonical surface | cache dir | cache file dates | reported cache result |
|---|---|---|---|
| standard_v2cal | `reference_cache/tier_standard` | **2026-07-15** | 10 hits / 0 misses |
| expanded_v2cal | `autoresearch/reference_cache/soc_oracle_expanded` | **2026-08-15** | hits |
| 2025_v2cal | `autoresearch/reference_cache/soc_oracle_2025` | **2026-08-14** | hits |

The physics-v2 fixes (A1–A8, B1/B8) landed 2026-09-13/14, i.e. **after** these
cache files were written.

### 2. Impact on baselines (cached vs fresh re-run, current code/data)

| Surface | policy | canonical (stale) | fresh | Δ |
|---|---|---:|---:|---:|
| standard Oct | PPO mean | 2,352.70 | 2,354.82 | +0.09% |
| standard Oct | fcas_rule mean | −56,095 | −48,460 | −14% |
| expanded | PPO mean | 19,503.53 | 4,349.79 | **−78%** |
| expanded (nsw1_may) | PPO | 361,011.98 | 1,760.75 | **−99.5%** |
| 2025 OOD | PPO mean | 6,497.51 | 4,817.02 | **−26%** |

The DT was **identical** on standard (16,208.586) and 2025 (30,791.31); it differed on
expanded `nsw1_may` (canonical 245,720 vs fresh 210,016) because that processed data
file was regenerated between runs. So the headline gap is affected mainly through the
**baseline**, and worst on the spike-heavy expanded scenarios.

### 3. Determinism of each policy (two fresh runs, identical config/data)

One-scenario probe (`nsw1_may_2024`, 288 h, fresh cache dirs A/B):

| policy | run A | run B | verdict |
|---|---:|---:|---|
| candidate_dt | 210,015.658053 | 210,015.658053 | **deterministic (identical)** |
| ppo_reference | 1,760.75 | 1,772.48 | differs ~0.7% |
| fcas_rule | −71,253.98 | −64,376.12 | differs ~10% |

The cache files for PPO/fcas-rule under the **same key** differ byte-for-byte between
runs → PPO (SB3 on GPU) and fcas-rule are **not bit-reproducible** on long episodes.
On the 144 h standard surface they matched to <0.1%.

## Implications for the preprint

1. **Use only self-consistent fresh runs.** The extended surfaces
   (`standard_year_v2cal`, `dispatch_year_v2cal`, `expanded_full_v2cal`,
   `2025_v2cal_fresh`) were all run at the frozen release with empty caches, so their
   DT and PPO sides share the same data/physics. These are the trustworthy set.
2. **Canonical `expanded_v2cal` is superseded** — its PPO baseline is stale and
   dominated by an implausible spike outlier. Do not cite A3's $19,504 PPO / 1.65×.
3. **Canonical standard Oct is unaffected** (PPO Δ +0.09%, DT identical) — A1 stands.
4. **Report PPO baselines with a reproducibility caveat** (~1% run-to-run on long
   episodes; the paired CI is wide enough that this does not change conclusions).
5. **Fix the cache key** before any further runs: include a hash of the PPO
   checkpoint and of the processed data (or a physics-code version) in `payload`,
   and record the processed-file hash in the eval summary.

Artifacts: `probe_fresh_standard.log`, `probe_fresh_2025.log`, `det_test.log`,
`artifacts/spotcheck_standard.txt`; configs `sdp_teacher_standard_freshprobe.json`,
`sdp_teacher_2025_fresh.json`, `det_nsw1may_{a,b}.json`.
