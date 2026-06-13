"""Validate FCAS dataset format without loading all into RAM."""
import sys
sys.path.insert(0, "src")

import polars as pl

DATASET_PATH = "data/aemo_dt_fcas/aemo_fcas_dataset.parquet"

# Use lazy/streaming
ds = pl.scan_parquet(DATASET_PATH)

# 1. Schema
schema = ds.collect_schema()
print("=== Schema ===")
for c, t in schema.items():
    print(f"  {c}: {t}")

# 2. Row/episode counts (streaming)
stats = ds.select([
    pl.col("episode_id").n_unique().alias("episodes"),
    pl.len().alias("rows"),
    pl.col("step").min().alias("min_step"),
    pl.col("step").max().alias("max_step"),
    pl.col("norm_observation").list.len().min().alias("min_state_dim"),
    pl.col("norm_observation").list.len().max().alias("max_state_dim"),
    pl.col("action").list.len().min().alias("min_act_dim"),
    pl.col("action").list.len().max().alias("max_act_dim"),
]).collect(engine="streaming")
print("\n=== Dataset stats ===")
for c, v in stats.row(0, named=True).items():
    print(f"  {c}: {v}")

# 3. Check episode length consistency (40 random episodes)
sample_eps = (
    ds.select("episode_id").unique().collect(engine="streaming")
    .sample(40, seed=42)["episode_id"].to_list()
)
ep_check = (
    ds.filter(pl.col("episode_id").is_in(sample_eps))
    .group_by("episode_id")
    .agg([
        pl.len().alias("step_count"),
        pl.col("step").min().alias("s0"),
        pl.col("step").max().alias("sN"),
        pl.col("norm_observation").list.len().first().alias("obs_dim"),
        pl.col("action").list.len().first().alias("act_dim"),
        pl.col("reward").is_not_null().all().alias("reward_valid"),
    ])
    .collect(engine="streaming")
)
print("\n=== Random episode validation (n=40) ===")
print(f"  step range per ep: {ep_check['step_count'].min()} - {ep_check['step_count'].max()} steps")
print(f"  obs_dim all match: {ep_check['obs_dim'].n_unique() == 1} (value: {ep_check['obs_dim'][0] if ep_check['obs_dim'].n_unique() == 1 else 'mixed'})")
print(f"  act_dim all match: {ep_check['act_dim'].n_unique() == 1} (value: {ep_check['act_dim'][0] if ep_check['act_dim'].n_unique() == 1 else 'mixed'})")
print(f"  reward valid (no nulls): {float(ep_check['reward_valid'].sum())}")
print(f"  step continuity (sN == step_count - 1): "
      f"{float(ep_check.filter(pl.col('sN') == pl.col('step_count') - 1).height)}/{float(len(ep_check))}")

# 4. Check episode_id uniqueness and order
eid_stats = ds.select([
    pl.col("episode_id").min().alias("min_eid"),
    pl.col("episode_id").max().alias("max_eid"),
    pl.col("episode_id").n_unique().alias("unique_eids"),
]).collect(engine="streaming")
print("\n=== Episode ID validity ===")
for c, v in eid_stats.row(0, named=True).items():
    print(f"  {c}: {v}")

# 5. Check specific episode for DT compatibility
print("\n=== DT trainer compatibility check ===")
ep0 = ds.filter(pl.col("episode_id") == 0).collect(engine="streaming")
obs = ep0["norm_observation"][0]
act = ep0["action"][0]
print(f"  Episode 0 steps: {len(ep0)}")
print(f"  norm_observation len: {len(obs)} (should be 18 for AEMO)")
print(f"  action len: {len(act)} (should be 3 for multi_market)")
print(f"  reward count: {ep0['reward'].len()}")
print(f"  step count: {ep0['step'].len()}")
print(f"  Columns: {sorted(ep0.columns)}")

# Check for NaN in the sample
import math
nan_obs = sum(1 for v in obs if math.isnan(v))
nan_act = sum(1 for a in act if math.isnan(a))
nan_rew = sum(1 for r in ep0["reward"] if math.isnan(r))
print(f"  NaN in norm_observation: {nan_obs} / {len(obs)}")
print(f"  NaN in action: {nan_act} / {len(act)}")
print(f"  NaN in reward: {nan_rew} / {len(ep0['reward'])}")

print("\n✅ Validation complete!")
