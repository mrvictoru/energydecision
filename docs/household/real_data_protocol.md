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
| Aggregated statistics, plots with binned/anonymized data, synthetic households derived from the schema | Case-by-case | Prefer derived/synthetic for anything public; synthetic outputs remain traceable through their manifest |

The manifest is designed to be share-safe: `tests/test_household_ingest.py::test_manifest_tracks_checksums_and_stats_only`
asserts no metering values leak into it.

## Workflow

1. Download weekly/daily CSVs from the portal per the acquisition section above.
2. Normalize each batch (command shown above), reviewing every warning — gaps,
   DST anomalies and negative values are reported, never silently fixed or fabricated.
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

## Data acquisition: manual download (current approach)

The portal's energy-balance view exposes an internal API
(`https://uiapi.sunnyportal.com/api/v1/measurements/{plantId}/energybalance?dateBeginLocal=...&interval=...`),
but its auth is a Keycloak OIDC flow with short-lived tokens (~15 min) and a
session cookie set after the SSO callback — automating it requires browser
automation (Playwright), which was explored and **deferred** (see git history on
`feature/household-modern-data` if ever revisited). Manual download it is.

**Manual workflow:**

1. Download from the portal UI. Note the two granularities:
   - **Daily batch** → 5-minute resolution (preferred; fetch these when possible)
   - **Weekly batch** → 15-minute resolution (fallback)
2. Save into `data/household/real/raw/` as `<portal>_<start-date>_<res>.csv`
   (e.g. `ennexos_2023-06-01_5min.csv`, `ennexos_2023-06-01_15min.csv`) — the
   resolution suffix matters because gap detection needs the expected step size.
3. Normalize each batch, passing the matching resolution:
   ```bash
   python3 scripts/ingest_household_portal_csv.py \
     --input "data/household/real/raw/*_5min.csv" --resolution-minutes 5 \
     --decimal-comma --watts-to-kilo   # SMA/ennexos conventions; drop if not applicable
   python3 scripts/ingest_household_portal_csv.py \
     --input "data/household/real/raw/*_15min.csv" --resolution-minutes 15 \
     --decimal-comma --watts-to-kilo
   ```
4. Commit the updated `household_ingest_manifest.json` (checksums + stats only).

## Synthetic-data handling

The H1.5 builder reads local normalized telemetry but writes only derived
synthetic episodes and a parameter manifest. Run it with a held-out real-source
date fraction (the default is 15%); those dates are recorded as the OOD surface
and must not be used to train the synthetic corpus. Do not commit raw or
normalized telemetry, and review the generated manifest before any external
release. The manifest contains source dates and generation parameters, not
metering values.
