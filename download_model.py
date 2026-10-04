"""Helper script to manually download IndicConformer-600M weights from Hugging Face."""
import os
import sys
from huggingface_hub import snapshot_download

REPO_ID = "sunilmahendrakar/indic-conformer-600m-multilingual"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "models", "indic-conformer-600m")

def main():
    print(f"[*] Downloading IndicConformer-600M weights from Hugging Face Hub...")
    print(f"[*] Repo ID: {REPO_ID}")
    print(f"[*] Destination: {TARGET_DIR}")
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False
    )
    print(f"\n[*] Model successfully downloaded to: {TARGET_DIR}")
    print(f"[*] Total files downloaded. The pipeline can now run 100% offline.")

if __name__ == "__main__":
    main()
