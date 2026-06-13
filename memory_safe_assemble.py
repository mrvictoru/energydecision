"""Assemble dataset in batches to avoid RAM spikes.
Usage: python3 memory_safe_assemble.py
"""
import sys, json, random, gc
sys.path.insert(0, "src")

from pathlib import Path
import polars as pl

OUTPUT_DIR = Path("data/aemo_dt_fcas")
RAW_LOGS_DIR = OUTPUT_DIR / "raw_logs"
BATCH_SIZE = 300

all_ep_files = sorted(RAW_LOGS_DIR.rglob("*.parquet"))
print(f"Total episode files: {len(all_ep_files)}")

batch_dirs = []
global_ep_offset = 0

for batch_idx, start in enumerate(range(0, len(all_ep_files), BATCH_SIZE)):
    batch_files = all_ep_files[start:start + BATCH_SIZE]
    batch_dir = OUTPUT_DIR / f"batch_{batch_idx:03d}"
    batch_dir.mkdir(exist_ok=True)
    frames = []
    for local_idx, path in enumerate(batch_files):
        stem = path.stem
        parts = stem.split("__")
        policy = parts[1] if len(parts) >= 4 else "unknown"
        df = pl.read_parquet(str(path)).select(["step", "norm_observation", "action", "reward"])
        df = df.with_columns([
            pl.lit(policy).alias("source_policy"),
            pl.lit(global_ep_offset + local_idx).alias("episode_id"),
        ])
        frames.append(df)
    batch_df = pl.concat(frames, how="diagonal_relaxed")
    batch_path = batch_dir / "batch.parquet"
    batch_df.write_parquet(str(batch_path))
    batch_dirs.append(batch_path)
    print(f"  Batch {batch_idx}: {len(batch_files)} eps (IDs {global_ep_offset}-{global_ep_offset+len(batch_files)-1}), rows {len(batch_df):,}")
    global_ep_offset += len(batch_files)
    del frames, batch_df; gc.collect()

# Add 20 old rule episodes
old_path = Path("data/aemo_dt/aemo_dt_dataset.parquet")
if old_path.exists():
    old_df = pl.read_parquet(str(old_path))
    old_rule = old_df.filter(pl.col("source_policy").str.contains("rule"))
    rule_ep_ids = old_rule["episode_id"].unique().to_list()
    random.seed(42)
    selected = random.sample(rule_ep_ids, min(20, len(rule_ep_ids)))
    old_rule_sample = old_rule.filter(pl.col("episode_id").is_in(selected))
    old_rows = []
    for i, (_, grp) in enumerate(old_rule_sample.group_by("episode_id")):
        grp = grp.with_columns(pl.lit(global_ep_offset + i).alias("episode_id"))
        grp = grp.select(["step", "norm_observation", "action", "reward", "source_policy", "episode_id"])
        old_rows.append(grp)
    if old_rows:
        old_batch_dir = OUTPUT_DIR / "batch_old_rule"; old_batch_dir.mkdir(exist_ok=True)
        pl.concat(old_rows).write_parquet(old_batch_dir / "batch.parquet")
        batch_dirs.append(old_batch_dir / "batch.parquet")
        print(f"  Old rule: {len(selected)} eps (IDs {global_ep_offset}-{global_ep_offset+len(selected)-1})")
        global_ep_offset += len(selected)

# Concat all batch parquets via streaming
print(f"\nConcatenating {len(batch_dirs)} batch files...")
combined = pl.concat([pl.scan_parquet(str(p)) for p in batch_dirs], how="diagonal_relaxed")
dataset = combined.collect(engine="streaming")

out_path = OUTPUT_DIR / "aemo_fcas_dataset.parquet"
dataset.write_parquet(str(out_path))
actual_eps = int(dataset["episode_id"].n_unique())
print(f"\nSaved: {out_path}")
print(f"Total episodes: {actual_eps}")
print(f"Total rows: {len(dataset):,}")

source_counts = dataset.group_by("source_policy").agg(pl.len()).sort("source_policy")
manifest = {
    "episode_count": actual_eps, "row_count": len(dataset),
    "sources": {str(r[0]): int(r[1]) for r in source_counts.iter_rows()},
    "state_dims": [len(dataset["norm_observation"][0])],
    "act_dims": [len(dataset["action"][0])],
}
OUTPUT_DIR.joinpath("aemo_fcas_manifest.json").write_text(json.dumps(manifest, indent=2))
for row in source_counts.iter_rows():
    print(f"  {row[0]}: {row[1]:,} rows")

# Cleanup
import shutil
for p in OUTPUT_DIR.glob("batch_*"):
    shutil.rmtree(p)
print("Cleaned up batch directories.")
