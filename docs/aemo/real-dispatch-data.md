# Real Dispatch Data: Generation Plan (2023–2025)

## Background

The FCAS dataset (2,425 episodes, 78.4M rows) is **99.5% synthetic**. Real AEMO
DISPATCHLOAD data (actual battery dispatch) was dropped during FCAS dataset
assembly because the `assemble_dataset` filter only matched `"rule"` source
policies, not `"dispatch"` policies.

**Goal**: Generate dispatch replay episodes for all 14 registered battery
stations across 2023, 2024, and 2025 windows to serve as a held-out validation
set for the DT, measuring the sim-to-real gap.

## Execution Script

Run from repo root (inside distrobox):

```bash
# Full generation (all stations, all windows)
python3 src/generate_dispatch_replays.py

# Single window
python3 src/generate_dispatch_replays.py --window 2024

# Single station
python3 src/generate_dispatch_replays.py --station hornsdale
```

Output goes to `data/aemo_dispatch_replays/`.

## Battery Registry Sizing (from AEMO)

Each station's real-life capacity is baked into `BATTERY_REGISTRY` in
`src/aemo_data.py`. The script uses `apply_unit_sizing=True` to match the sim
env to real life:

| Station | Region | Capacity (MWh) | Max Power (MW) | Commissioned |
|---------|--------|---------------|----------------|--------------|
| hornsdale | SA1 | 194 | 150 | 2017 (gen/load → 2022 bidi) |
| lake_bonney | SA1 | 25 | 25 | 2019 (gen/load → 2022 bidi) |
| dalrymple_north | SA1 | 8 | 30 | 2018 (2023 bidi) |
| blyth | SA1 | 14 | 10 | 2023 |
| bungama | SA1 | 50 | 50 | 2023 |
| torrens_island | SA1 | 250 | 250 | 2024 |
| ballarat | VIC1 | 30 | 30 | 2019 (→ 2023 bidi) |
| gannawarra | VIC1 | 25 | 25 | 2018 (→ 2023 bidi) |
| victorian_big_battery | VIC1 | 450 | 300 | 2021 |
| bulgana | VIC1 | 20 | 20 | 2019 |
| kennedy_energy_park | QLD1 | 4 | 2 | 2019 |
| wandoan | QLD1 | 150 | 100 | 2021 |
| wallgrove | NSW1 | 50 | 50 | 2021 |
| waratah | NSW1 | 850 | 850 | 2024 |

## Target Windows

| Window | Start | End | Notes |
|--------|-------|-----|-------|
| 2023 | 2023-01-01 | 2023-12-01 | Covers SA1, VIC1 fully; NSW1/QLD1/TAS1 have limited processed data |
| 2024 | 2024-01-01 | 2025-01-01 | Full coverage for all regions (two 6-month processed blocks) |
| 2025 | 2025-01-01 | 2025-06-01 | H1 2025 — limited processed data (need `fetch_aemo_region.py` for full coverage) |

## Expected Active Stations by Window

*Determined by `list_dispatch_candidates(station_name=...)` which queries the
cached DISPATCHLOAD data. Batteries with zero dispatch intervals in a window
are skipped.*

### Likely active in 2023
- **SA1**: hornsdale, lake_bonney, dalrymple_north, blyth (from 2023+), bungama (from 2023+)
- **VIC1**: ballarat (confirmed: 92,594 non-zero intervals), gannawarra, victorian_big_battery, bulgana
- **QLD1**: kennedy_energy_park, wandoan
- **NSW1**: wallgrove
- **TAS1**: none registered

### Likely active in 2024
- All of the above, plus torrens_island (SA1, from 2024+)
- Possibly waratah (NSW1, from 2024+)

### Likely active in 2025 H1
- Same as 2024 plus any newly commissioned stations

## Output Structure

```
data/aemo_dispatch_replays/
├── dispatch_replay_manifest.json    # overall results summary
├── 2023_hornsdale_dispatch_logs.parquet
├── 2023_hornsdale_dispatch_incident_logs.parquet
├── 2023_lake_bonney_dispatch_logs.parquet
├── 2024_hornsdale_dispatch_logs.parquet
├── 2024_lake_bonney_dispatch_logs.parquet
└── ...
```

Each parquet file contains step-by-step logs with columns: `observations`,
`actions`, `reward`, `timestep`, `episode_id`.

## Validation Use

After generation, the dispatch replay logs serve as a **held-out real-data
validation set** for DT evaluation:

```bash
python3 src/autoresearch_evaluator.py \
  --surface-manifest-path data/aemo_dispatch_replays/dispatch_replay_manifest.json \
  --evaluation-config configs/aemo_autoresearch_evaluator.dispatch.json \
  --output-dir eval_output/autoresearch/dispatch_validation
```

Key metrics: action MSE (energy + FCAS), reward gap to real dispatch.

## Caching

- DISPATCHLOAD raw data is cached in `data/aemo/` as feather files (one per month)
- First run of any station in a window triggers NEMOSIS download (~30 min cumulative)
- Subsequent runs reuse the cache
- Processed market data is already cached in `data/aemo/processed_*.parquet`
