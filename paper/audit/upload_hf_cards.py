"""Re-upload the updated HF model-card READMEs (token from HF_TOKEN env)."""
import os
from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=os.environ["HF_TOKEN"])
print("whoami:", api.whoami()["name"], flush=True)

uploads = [
    ("docs/huggingface/energydecision-dt-v2-sdp_README.md",
     "mrvictoru/energydecision-dt-v2-sdp", "model"),
    ("docs/huggingface/energydecision-waypoint-dt_README.md",
     "mrvictoru/energydecision-waypoint-dt", "model"),
]
for src, repo, kind in uploads:
    api.upload_file(path_or_fileobj=src, path_in_repo="README.md", repo_id=repo, repo_type=kind)
    print("uploaded README ->", repo, flush=True)

# verify the control-loop wording is live
for repo in ["mrvictoru/energydecision-dt-v2-sdp", "mrvictoru/energydecision-waypoint-dt"]:
    p = hf_hub_download(repo, "README.md", repo_type="model", force_download=True)
    t = open(p).read()
    print(repo, "| 'control loop':", "control loop" in t, "| 'solver at inference':", "solver at inference" in t)
print("CARDS_UPLOADED")
