#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HOUSEHOLD_TTM_IMAGE:-localhost/energydecision-ttm:r3}"
CONTAINER="${HOUSEHOLD_TTM_CONTAINER:-energydecision-ttm}"

if ! command -v podman >/dev/null 2>&1; then
  echo "podman is required to run the isolated TTM-R3 sidecar." >&2
  exit 2
fi

if ! podman image exists "${IMAGE}"; then
  podman build --file "${ROOT}/Containerfile.ttm" --tag "${IMAGE}" "${ROOT}"
fi

if ! podman container exists "${CONTAINER}"; then
  distrobox create \
    --name "${CONTAINER}" \
    --image "${IMAGE}" \
    --nvidia \
    --yes
fi

# Distrobox shares $HOME; ignore host user-site packages so only image-pinned
# TTM/PyTorch dependencies are imported.
exec distrobox enter "${CONTAINER}" -- \
  env PYTHONNOUSERSITE=1 \
  python3 "${ROOT}/scripts/generate_household_ttm_forecasts.py" "$@"
