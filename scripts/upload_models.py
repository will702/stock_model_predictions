"""
Upload trained model artifacts to HF Hub.
Run: python -m scripts.upload_models
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi
from app.config import HF_TOKEN, MODEL_REPO, MODEL_DIR

ARTIFACTS = [
    ("lstm_stock.pt", "lstm_stock.pt"),
    ("embeddings.json", "embeddings.json"),
]


def upload():
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set. Add it to ml/.env or export it.")
        sys.exit(1)
    if not MODEL_REPO:
        print("Error: HF_MODEL_REPO not set. Add it to ml/.env (e.g. yourname/stockpro-lstm).")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Verify token is valid before doing anything
    try:
        whoami = api.whoami()
        print(f"Authenticated as: {whoami['name']}")
    except Exception as e:
        print(f"Error: HF_TOKEN is invalid or expired — {e}")
        sys.exit(1)

    # Create repo if it doesn't exist
    try:
        repo_url = api.create_repo(
            repo_id=MODEL_REPO,
            repo_type="model",
            exist_ok=True,
            private=True,
        )
        print(f"Repo ready: {repo_url}")
    except Exception as e:
        print(f"Error creating repo '{MODEL_REPO}': {e}")
        print("Make sure HF_MODEL_REPO is in the format 'username/repo-name'.")
        sys.exit(1)

    print(f"Uploading to {MODEL_REPO}...")
    uploaded = 0
    for local_name, repo_name in ARTIFACTS:
        local_path = os.path.join(MODEL_DIR, local_name)
        if not os.path.exists(local_path):
            print(f"  Skipping {local_name} — file not found at {local_path}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_name,
                repo_id=MODEL_REPO,
                repo_type="model",
                commit_message=f"Upload {repo_name}",
            )
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            print(f"  ✓ {local_name} ({size_mb:.1f} MB) → {MODEL_REPO}/{repo_name}")
            uploaded += 1
        except Exception as e:
            print(f"  ✗ Failed to upload {local_name}: {e}")

    if uploaded == 0:
        print("No files uploaded.")
        sys.exit(1)
    print(f"Done. {uploaded} file(s) uploaded.")


if __name__ == "__main__":
    upload()
