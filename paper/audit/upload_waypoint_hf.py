"""One-shot HF upload (minimal scope) for the SOC-waypoint teacher helper.

Creates mrvictoru/energydecision-waypoint-dt and uploads the waypoint checkpoint
+ sidecars + card, then refreshes the dataset and shipped-model README cards to
document the waypoint dependency. Token is read from the HF_TOKEN env var; it is
never written to disk.
"""
import os
import sys

from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)

WAYPOINT_REPO = "mrvictoru/energydecision-waypoint-dt"
DATASET_REPO = "mrvictoru/AEMO_simulated_trade_sdp"
MODEL_REPO = "mrvictoru/energydecision-dt-v2-sdp"

print("whoami:", api.whoami()["name"], flush=True)
api.create_repo(repo_id=WAYPOINT_REPO, repo_type="model", private=False, exist_ok=True)
print("repo ready:", WAYPOINT_REPO, flush=True)

uploads = [
    ("models/aemo/dt/soc_waypoint_dt_best.pt", "soc_waypoint_dt_best.pt"),
    ("models/aemo/dt/soc_waypoint_dt_best.pt.meta.json", "soc_waypoint_dt_best.pt.meta.json"),
    ("models/aemo/dt/soc_waypoint_dt_best.pt.metrics.json", "soc_waypoint_dt_best.pt.metrics.json"),
    ("models/aemo/dt/soc_waypoint_dt_loss_surface_manifest.json", "soc_waypoint_dt_loss_surface_manifest.json"),
    ("docs/huggingface/energydecision-waypoint-dt_README.md", "README.md"),
]
for src, dst in uploads:
    print(f"uploading {dst} ...", flush=True)
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=WAYPOINT_REPO, repo_type="model")

print("refreshing dataset card ...", flush=True)
api.upload_file(
    path_or_fileobj="docs/huggingface/AEMO_simulated_trade_sdp_README.md",
    path_in_repo="README.md", repo_id=DATASET_REPO, repo_type="dataset",
)
print("refreshing model card ...", flush=True)
api.upload_file(
    path_or_fileobj="docs/huggingface/energydecision-dt-v2-sdp_README.md",
    path_in_repo="README.md", repo_id=MODEL_REPO, repo_type="model",
)
print("UPLOAD_DONE", flush=True)
sys.exit(0)
