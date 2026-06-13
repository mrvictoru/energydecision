# Data Generation Progress

## AEMO Data Fetch — ✅ All 5 regions complete

| Region | Date Range | Rows |
|--------|-----------|:----:|
| **NSW1** | 2021-01-01 → 2023-04-01 | 236,160 |
| **QLD1** | 2021-01-01 → 2023-04-01 | 236,160 |
| **SA1** | 2022-04-01 → 2023-12-01 | 171,984 |
| **TAS1** | 2021-01-01 → 2023-04-01 | 236,160 |
| **VIC1** | 2021-04-01 → 2023-12-01 | 277,103 |

## FCAS Dataset Generation — ✅ Complete

**Final dataset:** `data/aemo_dt_fcas/aemo_fcas_dataset.parquet` (439 MB, 2,125 episodes, 68.9M rows)

| Source | Episodes | Notes |
|--------|:--------:|-------|
| PPO | 905 | Best FCAS ($8.98/step) |
| TD3 | 300 | Medium FCAS ($4.24/step) |
| A2C | 300 | Medium FCAS ($3.82/step) |
| DDPG | 300 | Medium FCAS ($3.84/step) |
| SAC | 300 | Medium FCAS ($2.72/step) |
| Old rule | 20 | Zero FCAS (kept as baseline) |

3 horizons (12-day, 8-week, 26-week), 3 battery sizes (medium bias), 5 regions.

## How to fetch a region (if needed for new data)

```bash
distrobox enter energydecision
python3 src/fetch_aemo_region.py --region NSW1 --start 2021-01-01 --end 2023-04-01 --cache-dir data/aemo
```

Fetch logs: `data/aemo/fetch_logs/`
