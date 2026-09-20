# Hugging Face upload handoff (physics-v2)

The current host can't push these (large files), so copy the files below to a
machine with network access and run the commands there. Both repos are public.

## 1. Model repo: `mrvictoru/energydecision-dt-v2-sdp`

Copy from the repo root:

| Local path | Path in repo | Size | sha256 |
|---|---|---:|---|
| `models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt` | `aemo_dt_sdp_jtsoc_v2cal.pt` | 303,502,589 B | `73fe14bd111b1620b4784c7f05919c499261cbda0365ede8a815f7672981e09f` |
| `models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal.pt.meta.json` | `aemo_dt_sdp_jtsoc_v2cal.pt.meta.json` | 229 B | `47a4a98865d19f295f0941bff4c0124022f42355363e0d5d932f3c99300d1c6a` |
| `models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json` | `aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json` | 319 B | (arch kwargs) |
| `models/aemo/dt/aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json` | `aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json` | 5.4 KB | (full arch + training kwargs) |
| `docs/huggingface/energydecision-dt-v2-sdp_README.md` | `README.md` | — | (model card) |

```bash
# from the machine that has the copied files
hf auth login   # or: huggingface-cli login

REPO=mrvictoru/energydecision-dt-v2-sdp
hf upload "$REPO" aemo_dt_sdp_jtsoc_v2cal.pt aemo_dt_sdp_jtsoc_v2cal.pt --repo-type model
hf upload "$REPO" aemo_dt_sdp_jtsoc_v2cal.pt.meta.json aemo_dt_sdp_jtsoc_v2cal.pt.meta.json --repo-type model
hf upload "$REPO" aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json aemo_dt_sdp_jtsoc_v2cal_model_kwargs.json --repo-type model
hf upload "$REPO" aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json aemo_dt_sdp_jtsoc_v2cal_loss_surface_manifest.json --repo-type model
hf upload "$REPO" README.md README.md --repo-type model
```

(Equivalent legacy syntax: `huggingface-cli upload "$REPO" <local> <remote>`.)

Notes:
- The `.pt` is a **pure `state_dict`**; architecture is in `*.meta.json`
  (`return_scale=23284.83`) and the manifest. Loading it with
  `load_from_checkpoint` without the sidecar would leave `return_scale=1.0`.
- We keep the previous `aemo_dt_sdp_jtsoc_fullcorpus.pt` in the repo and mark it
  historical in the card. If you prefer a stable shipped filename instead,
  overwrite `aemo_dt_sdp_jtsoc_fullcorpus.pt` with the new weights **and** update
  its `.meta.json` — but the new-name approach is cleaner for provenance.

## 2. Dataset repo: `mrvictoru/AEMO_simulated_trade_sdp`

| Local path | Path in repo | Size | sha256 |
|---|---|---:|---|
| `data/aemo_dt_sdp/dt_trajectories_jtsoc_v2cal_conservative.parquet` | `dt_trajectories_jtsoc_v2cal_conservative.parquet` | 401,816,493 B | `14c94ab9c7aa7db3e9a55cf8e245cbb72ab10c9f2666403e4917801eee3cf883` |
| `docs/huggingface/AEMO_simulated_trade_sdp_README.md` | `README.md` | — | (dataset card) |

```bash
REPO=mrvictoru/AEMO_simulated_trade_sdp
hf upload "$REPO" dt_trajectories_jtsoc_v2cal_conservative.parquet dt_trajectories_jtsoc_v2cal_conservative.parquet --repo-type dataset
hf upload "$REPO" README.md README.md --repo-type dataset
```

Keep the existing `dt_trajectories_jtsoc_combined.parquet` / `_full` /
`_aggressive` files; the card marks them historical.

## 3. Verify after upload

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
import hashlib, json, torch

p = hf_hub_download("mrvictoru/energydecision-dt-v2-sdp", "aemo_dt_sdp_jtsoc_v2cal.pt", repo_type="model")
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
print("sha256:", h)  # expect 73fe14bd...
print("dtype keys:", len(torch.load(p, map_location="cpu").keys()))

m = json.load(open(hf_hub_download("mrvictoru/energydecision-dt-v2-sdp", "aemo_dt_sdp_jtsoc_v2cal.pt.meta.json", repo_type="model")))
print("return_scale:", m["return_scale"])  # 23284.83
PY
```
