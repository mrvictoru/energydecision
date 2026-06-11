# AEMO full-corpus manual DT commands

## What is actually slow?

- **The main bottleneck is training on the mixed corpus**, especially once the long-horizon 2021–2023 data is combined with the new 8-week augmentation.
- The **longer evaluator** (`configs/aemo_autoresearch_evaluator.example.json`) is slower than the mini evaluator, but it is **not** the main blocker.
- The full evaluator already includes the non-DT references you asked for:
  - `rule`
  - `dispatch_dalrymple_north`
  - `dispatch_torrens_island`
  - `ppo_reference`
- The evaluator also uses `reference_cache_dir`, so after the first full run the reference policies do not need to be recomputed every time.

## Assumptions

- The commands below are already written so they can be run **directly from the host shell**.
- Each command explicitly enters the recommended GPU runtime:

```bash
distrobox enter energydecision-gpu -- bash -lc 'cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision && <command>'
```

- You now have two training corpora:

  - long-horizon base corpus: `data/aemo_dt/aemo_dt_dataset.parquet`
  - short-horizon augmentation: `data/aemo_dt/aemo_dt_8week_dataset.parquet`

- The training commands below stage both files into a small mixed-corpus directory and train against that directory.
- The AEMO wrapper now forwards a few more trainer knobs, but these mixed-corpus commands still need the
  direct trainer because they rely on `--data-dir` and `--patterns`.

```bash
data/aemo_dt/aemo_dt_8week_dataset.parquet
```

- These commands use the **best downstream config so far** from the pilot experiments:
  - baseline: `aemo_proxy` defaults (`batch_size=128`, `lr=3e-5`)
  - promoted best: `exp12-batch32` (`batch_size=32`, `lr=3e-5`)

- For a **multi-day** history window, this guide uses:
  - `--context-length 576` (2 days at 5-minute steps)
  - `--rope-max-position 1728` (3x context length for R, S, A token stacks)

- Keep a small held-out slice for progress tracking:
  - `--val-split 0.1`

If you want the **best proxy-metric winner** instead, swap the second training command to `--batch-size 128 --lr 1e-4`.

## Optional: prewarm evaluator caches first

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
python3 src/prewarm_aemo_cache.py \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json
'
```

## 1. Train the baseline config on the available full training corpus

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
mkdir -p data/aemo_dt/manual_full_corpus_mix models/aemo/dt/manual_mixed_proxy_baseline &&
ln -sf ../aemo_dt_dataset.parquet data/aemo_dt/manual_full_corpus_mix/aemo_dt_long_2021_2023.parquet &&
ln -sf ../aemo_dt_8week_dataset.parquet data/aemo_dt/manual_full_corpus_mix/aemo_dt_short_8week.parquet &&
python3 src/pretrain_decision_transformer.py \
  --data-dir data/aemo_dt/manual_full_corpus_mix \
  --patterns aemo_dt_long_2021_2023 aemo_dt_short_8week \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --surface-preset aemo_proxy \
  --save-path models/aemo/dt/manual_mixed_proxy_baseline/aemo_dt_model.pt \
  --checkpoint-path models/aemo/dt/manual_mixed_proxy_baseline/aemo_dt_checkpoint.pt \
  --loss-csv-path models/aemo/dt/manual_mixed_proxy_baseline/aemo_dt_loss_history.csv \
  --context-length 576 \
  --rope-max-position 1728 \
  --train-in-subsets \
  --subset-episodes 192 \
  --epochs 1 \
  --epochs-per-subset 1 \
  --batch-size 128 \
  --lr 3e-5 \
  --val-split 0.1 \
  --num-workers 0 \
  --amp-mode auto \
  --checkpoint-interval 1 \
  --checkpoints-per-epoch 4
'
```

## 2. Train the best-so-far config on the available full training corpus

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
mkdir -p data/aemo_dt/manual_full_corpus_mix models/aemo/dt/manual_mixed_proxy_best_batch32 &&
ln -sf ../aemo_dt_dataset.parquet data/aemo_dt/manual_full_corpus_mix/aemo_dt_long_2021_2023.parquet &&
ln -sf ../aemo_dt_8week_dataset.parquet data/aemo_dt/manual_full_corpus_mix/aemo_dt_short_8week.parquet &&
python3 src/pretrain_decision_transformer.py \
  --data-dir data/aemo_dt/manual_full_corpus_mix \
  --patterns aemo_dt_long_2021_2023 aemo_dt_short_8week \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --surface-preset aemo_proxy \
  --save-path models/aemo/dt/manual_mixed_proxy_best_batch32/aemo_dt_model.pt \
  --checkpoint-path models/aemo/dt/manual_mixed_proxy_best_batch32/aemo_dt_checkpoint.pt \
  --loss-csv-path models/aemo/dt/manual_mixed_proxy_best_batch32/aemo_dt_loss_history.csv \
  --context-length 576 \
  --rope-max-position 1728 \
  --train-in-subsets \
  --subset-episodes 192 \
  --epochs 1 \
  --epochs-per-subset 1 \
  --batch-size 32 \
  --lr 3e-5 \
  --val-split 0.1 \
  --num-workers 0 \
  --amp-mode auto \
  --checkpoint-interval 1 \
  --checkpoints-per-epoch 4
'
```

## 3. Run the longer held-out evaluator on the baseline model

This command evaluates:

- the trained DT candidate
- `rule`
- dispatch replay references
- PPO reference

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
mkdir -p eval_output/aemo/autoresearch/manual_mixed_proxy_baseline_full &&
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/manual_mixed_proxy_baseline/aemo_dt_loss_history_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json \
  --output-dir eval_output/aemo/autoresearch/manual_mixed_proxy_baseline_full
'
```

## 4. Run the longer held-out evaluator on the best-so-far model

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
mkdir -p eval_output/aemo/autoresearch/manual_mixed_proxy_best_batch32_full &&
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path models/aemo/dt/manual_mixed_proxy_best_batch32/aemo_dt_loss_history_surface_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.example.json \
  --output-dir eval_output/aemo/autoresearch/manual_mixed_proxy_best_batch32_full
'
```

## Optional: attach to a running training job

Use `dt_progress_runner --attach` with the progress snapshot and surface manifest that the launcher writes next to the run artifacts. That gives you a live tracker without launching a second training job.

### Baseline

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
python3 src/dt_progress_runner.py --attach \
  --progress-snapshot-path models/aemo/dt/manual_mixed_proxy_baseline/aemo_dt_loss_history_progress.json \
  --surface-manifest-path models/aemo/dt/manual_mixed_proxy_baseline/aemo_dt_loss_history_surface_manifest.json
'
```

### Best-so-far

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
python3 src/dt_progress_runner.py --attach \
  --progress-snapshot-path models/aemo/dt/manual_mixed_proxy_best_batch32/aemo_dt_loss_history_progress.json \
  --surface-manifest-path models/aemo/dt/manual_mixed_proxy_best_batch32/aemo_dt_loss_history_surface_manifest.json
'
```

## Optional alternate best command: proxy-metric winner instead of downstream winner

If you want to rerun the **best proxy-metric** config instead of `exp12-batch32`, use this in place of section 2:

```bash
distrobox enter energydecision-gpu -- bash -lc '
cd /run/host/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision &&
mkdir -p data/aemo_dt/manual_full_corpus_mix models/aemo/dt/manual_mixed_proxy_best_lr1e4 &&
ln -sf ../aemo_dt_dataset.parquet data/aemo_dt/manual_full_corpus_mix/aemo_dt_long_2021_2023.parquet &&
ln -sf ../aemo_dt_8week_dataset.parquet data/aemo_dt/manual_full_corpus_mix/aemo_dt_short_8week.parquet &&
python3 src/pretrain_decision_transformer.py \
  --data-dir data/aemo_dt/manual_full_corpus_mix \
  --patterns aemo_dt_long_2021_2023 aemo_dt_short_8week \
  --model-config configs/aemo_decision_transformer_model_kwargs.json \
  --surface-preset aemo_proxy \
  --save-path models/aemo/dt/manual_mixed_proxy_best_lr1e4/aemo_dt_model.pt \
  --checkpoint-path models/aemo/dt/manual_mixed_proxy_best_lr1e4/aemo_dt_checkpoint.pt \
  --loss-csv-path models/aemo/dt/manual_mixed_proxy_best_lr1e4/aemo_dt_loss_history.csv \
  --context-length 576 \
  --rope-max-position 1728 \
  --train-in-subsets \
  --subset-episodes 192 \
  --epochs 1 \
  --epochs-per-subset 1 \
  --batch-size 128 \
  --lr 1e-4 \
  --val-split 0.1 \
  --num-workers 0 \
  --amp-mode auto \
  --checkpoint-interval 1 \
  --checkpoints-per-epoch 4
'
```
