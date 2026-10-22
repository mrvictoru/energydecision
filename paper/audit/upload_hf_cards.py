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
    ("docs/huggingface/AEMO_simulated_trade_sdp_README.md",
     "mrvictoru/AEMO_simulated_trade_sdp", "dataset"),
]
for src, repo, kind in uploads:
    api.upload_file(path_or_fileobj=src, path_in_repo="README.md", repo_id=repo, repo_type=kind)
    print("uploaded README ->", repo, flush=True)

# verify the control-loop wording is live
checks = [
    ("mrvictoru/energydecision-dt-v2-sdp", "model", "4.70"),
    ("mrvictoru/energydecision-waypoint-dt", "model", "control loop"),
    ("mrvictoru/AEMO_simulated_trade_sdp", "dataset", "12-day"),
]
for repo, kind, needle in checks:
    p = hf_hub_download(repo, "README.md", repo_type=kind, force_download=True)
    t = open(p).read()
    print(repo, f"| '{needle}' present:", needle in t)
print("CARDS_UPLOADED")
