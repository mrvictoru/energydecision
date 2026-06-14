# AEMO FCAS Dataset Training Commands

## Training corpus

The primary training dataset is the FCAS-rich assembly:

- **Path**: `data/aemo_dt_fcas/aemo_fcas_dataset.parquet`
- **Size**: 2,425 episodes, 78.4M rows, 3.1 GB
- **Sources**: PPO (905), A2C (300), DDPG (300), SAC (300), TD3 (300), FCAS rule (300), old rule (20)
- **State**: 18-dim normalized observation vector
- **Action**: 3-dim (energy dispatch, FCAS raise, FCAS lower)

## Pilot (fast iteration)

```bash
# Generate stratified pilot spec and build parquet files
python3 src/build_aemo_fcas_pilot_spec.py --build-pilot

# Train on pilot
python3 src/launch_aemo_training.py --run-tier proxy-baseline
```

## Learning baseline (full dataset)

```bash
# Prewarm evaluator caches first (optional, ~30min first time)
python3 src/prewarm_aemo_cache.py \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json

# Train on full dataset via subset training
python3 src/launch_aemo_training.py --run-tier learning-baseline

# Run full held-out evaluation
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/<run-tag>/surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json \
  --output-dir eval_output/autoresearch/<run-tag>
```

## Standalone training

```bash
python3 src/pretrain_aemo_decision_transformer.py \
  --dataset-path data/aemo_dt_fcas/aemo_fcas_dataset.parquet \
  --val-dataset-path data/aemo_dt_fcas/autoresearch_pilot/aemo_dt_val_pilot.parquet \
  --surface-preset aemo_learning_baseline \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --batch-size 16 \
  --epochs 2 \
  --lr 3e-5 \
  --train-in-subsets \
  --subset-episodes 24 \
  --epochs-per-subset 2
```

## Evaluator configs

| Config | Use | Speed | Details |
|--------|-----|-------|---------|
| `mini` | Screening | ~2 min | 2 regions, 1 battery, 24h, 5-min steps, DT vs FCAS rule |
| `example` | Promotion | ~15 min | 4 regions, 2 batteries, 144h, DT vs rule+FCAS+dispatch+PPO |
| `expanded` | Full sweep | ~60 min | 5 regions × 6 months, medium battery, 3 baselines |
