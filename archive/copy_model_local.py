import os
import shutil
import glob

src_cache = r"D:\hf_cache\hub\models--sunilmahendrakar--indic-conformer-600m-multilingual"
target_dir = r"D:\ASR\models\indic-conformer-600m"

os.makedirs(target_dir, exist_ok=True)

# Find snapshot directory inside Hugging Face cache
snapshots = glob.glob(os.path.join(src_cache, "snapshots", "*"))
if snapshots:
    src_folder = snapshots[0]
else:
    src_folder = src_cache

print(f"[*] Copying model files from: {src_folder} -> {target_dir}")

# Copy files
for item in os.listdir(src_folder):
    s = os.path.join(src_folder, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        if os.path.exists(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"[*] Model successfully copied into local project folder: {target_dir}")
print(f"[*] Folder contents: {os.listdir(target_dir)}")
