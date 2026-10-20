#!/usr/bin/env bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "## Runtime (distrobox energydecision-gpu)"
python3 -V
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "cuda", torch.version.cuda)
PY
echo
echo "## pip freeze"
pip freeze
