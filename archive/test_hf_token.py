from pathlib import Path
from huggingface_hub import HfApi

env_path = Path(r"D:\ASR\.env")
vals = {}
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip().strip("\"'")

hf_token = vals.get("HF_TOKEN")
if hf_token:
    api = HfApi(token=hf_token)
    try:
        user_info = api.whoami()
        print("Hugging Face Auth SUCCESS! Username:", user_info.get("name"))
    except Exception as e:
        print("Hugging Face Auth Note:", e)
else:
    print("HF_TOKEN not found in .env")
