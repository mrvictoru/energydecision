import torch
from huggingface_hub import hf_hub_download

print("Downloading SDP-teacher DT checkpoint from HF...")
path = hf_hub_download(
    repo_id="mrvictoru/energydecision-dt-v2-sdp",
    filename="aemo_dt_fcas_best_checkpoint.pt",
    local_dir="models/aemo/dt/",
    repo_type="model",
    local_dir_use_symlinks=False,
)
print(f"Downloaded to: {path}")

# Verify checkpoint structure
ckpt = torch.load(path, map_location="cpu")
print(f"Checkpoint keys: {list(ckpt.keys())[:10]}")
if "meta" in ckpt:
    print(f"Meta: {ckpt['meta']}")
if "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
    print(f"Model state dict has {len(sd)} keys")
    # Check for action_head_mode by looking for sigmoid-related params
    act_keys = [k for k in sd.keys() if 'act' in k.lower()]
    print(f"Action-related keys: {act_keys[:5]}...")
