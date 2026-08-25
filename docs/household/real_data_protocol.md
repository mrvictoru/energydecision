# Real Household Data — Privacy & Ingestion Protocol

> Governs the use of real household telemetry (solar + home battery, 2019+,
> portal CSV exports) in the household track. See `docs/FUTURE_PLAN.md` §6b.

## Consent & scope

- Data belongs to a private individual who has granted access for this research.
- Consent covers research use inside this repository's workflows; it does **not**
  cover publishing raw metering data anywhere (git, Hugging Face, papers, websites).
- Before any external release (dataset upload, paper figure with raw traces),
  re-confirm consent explicitly and anonymize.

## What may be committed

| Artifact | Commit? | Notes |
|---|---|---|
| Raw portal CSVs (`data/household/real/raw/`) | **NEVER** | Gitignored; local-only |
| Normalized parquets (`data/household/real/normalized/`) | **NEVER** | Gitignored; still raw telemetry |
| `household_ingest_manifest.json` (repo root) | Yes | Checksums + row/validation counts only; no metering values. Kept outside `data/` so it can be tracked despite the blanket `data/` ignore |
| Aggregated statistics, plots with binned/anonymized data, synthetic households derived from the schema | Case-by-case | Prefer derived/synthetic for anything public |

The manifest is designed to be share-safe: `tests/test_household_ingest.py::test_manifest_tracks_checksums_and_stats_only`
asserts no metering values leak into it.

## Workflow

1. Download weekly CSVs from the portal into `data/household/real/raw/` using the
   naming convention `<portal>_<start-date>.csv` (e.g. `portal_2023-06-01.csv`).
2. Normalize each batch:
   ```bash
   python3 scripts/ingest_household_portal_csv.py \
     --input "data/household/real/raw/*.csv" \
     --resolution-minutes 5
   ```
   Auto-detection maps common portal column names; pass `--column-map '{"canonical": "Portal Name", ...}'`
   if detection fails. Review every warning — gaps/DST anomalies/negative values are
   reported, never silently fixed or fabricated.
3. The manifest updates incrementally; commit it so ingested batches are traceable
   by checksum without exposing any telemetry.
4. Downstream experiments read only from `data/household/real/normalized/`.

## Known format unknowns (resolve against first real sample)

- Exact column names and units (kW vs kWh) of the portal export → finalize
  `DEFAULT_COLUMN_HINTS` / provide a repo-default column map.
- Whether battery channels (SOC, charge/discharge power) are exported at all —
  determines whether true replay-gap analysis is possible (H3) vs simulated placement.
- Tariff history: record which import/export plans were active when, since H1
  surface splits may need tariff-regime boundaries.
