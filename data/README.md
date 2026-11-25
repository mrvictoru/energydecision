# Data directory

Runtime datasets, caches, processed AEMO parquets and generated corpora. Most contents are ignored by Git; exact public corpus availability is documented in [the reproduction guide](../docs/reproduction.md).

## Start here

- [Preprint reproduction guide](../docs/reproduction.md)
- [Documentation hub](../docs/README.md)
- [Repository guide](../REPOSITORY_GUIDE.md)
- [Audit artifact index](../paper/audit/artifacts/README.md)

## Key subdirectories

- [aemo_dt_sdp/](aemo_dt_sdp/README.md): shipped SDP-teacher v2cal corpus and historical variants
- [aemo_dt_fcas/](aemo_dt_fcas/README.md): historical FCAS-rich corpus and autoresearch pilot
- `aemo/`: processed AEMO market data and seasonal RRP profiles
- `aemo_dt_fcas_v2/`, `aemo_dt_impact/`, `aemo_dt_fcas_v3_combined/`: historical behaviour-cloning/GRPO corpora with their own READMEs

Do not assume a local cache exists in a fresh clone. Use `AEMO_CACHE_ONLY=1` only after staging the caches.
