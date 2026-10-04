import os
import shutil
import glob

src_cache = r"D:\hf_cache\hub\models--sunilmahendrakar--indic-conformer-600m-multilingual"
target_dir = r"D:\ASR\models\indic-conformer-600m"

os.makedirs(target_dir, exist_ok=True)

snapshot_dir = os.path.join(src_cache, "snapshots", "ff6bbf95f1e30b772248f0b2ddb1dd1f0a20b422")

print(f"[*] Copying resolved model files from: {snapshot_dir} -> {target_dir}")

for root, dirs, files in os.walk(snapshot_dir):
    rel_path = os.path.relpath(root, snapshot_dir)
    dest_dir = os.path.join(target_dir, rel_path) if rel_path != "." else target_dir
    os.makedirs(dest_dir, exist_ok=True)
    
    for f in files:
        src_file = os.path.join(root, f)
        real_file = os.path.realpath(src_file) # Resolves symlink to blob
        dest_file = os.path.join(dest_dir, f)
        
        if os.path.exists(real_file):
            shutil.copy2(real_file, dest_file)

print(f"[*] Resolved copy complete into: {target_dir}")
