import json

import torch
from huggingface_hub import hf_hub_download

print("Downloading physics-v2 SDP-teacher DT checkpoint from HF...")
repo = "mrvictoru/energydecision-dt-v2-sdp"
filename = "aemo_dt_sdp_jtsoc_v2cal.pt"
path = hf_hub_download(
    repo_id=repo,
    filename=filename,
    local_dir="models/aemo/dt/",
    repo_type="model",
    local_dir_use_symlinks=False,
)
meta = json.loads(hf_hub_download(repo, f"{filename}.meta.json", repo_type="model"))
print(f"Downloaded to: {path}")
print(f"return_scale={meta.get('return_scale')} model={meta.get('model')}")

# The main .pt is a pure state_dict; load it with DecisionTransformer.load_from_checkpoint
state = torch.load(path, map_location="cpu")
print(f"state_dict entries: {len(state)}")
