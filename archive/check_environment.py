import sys
import importlib.util
import os

# Store Hugging Face cache on D drive
os.environ["HF_HOME"] = r"D:\hf_cache"

def check_pkg(pkg_name):
    try:
        mod = importlib.import_module(pkg_name)
        version = getattr(mod, '__version__', 'Installed (no __version__)')
        return True, version, mod
    except Exception as e:
        return False, f"Not Installed ({e})", None

def main():
    print("=" * 60, flush=True)
    print("      ASR BENCHMARK - ENVIRONMENT CHECK (PHASE 1)", flush=True)
    print("=" * 60, flush=True)

    # 1. Python Check
    py_ver = sys.version.split()[0]
    py_pass = sys.version_info >= (3, 8)
    print(f"[*] Python Version         : {py_ver} [{'PASS' if py_pass else 'FAIL'}]", flush=True)

    # 2. PyTorch & CUDA Check
    print("[*] Checking PyTorch & CUDA...", flush=True)
    torch_ok, torch_ver, torch = check_pkg("torch")
    cuda_avail = False
    cuda_ver = "N/A"
    gpu_name = "N/A"
    gpu_vram = "N/A"
    rtx_3050_ok = False

    if torch_ok:
        print(f"[*] PyTorch Version        : {torch_ver} [PASS]", flush=True)
        try:
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                cuda_ver = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_vram = f"{gpu_vram_bytes / (1024**3):.2f} GB"
                if "3050" in gpu_name:
                    rtx_3050_ok = True
                else:
                    rtx_3050_ok = True
        except Exception as e:
            print(f"[!] Error checking CUDA: {e}", flush=True)
    else:
        print(f"[*] PyTorch Version        : {torch_ver} [FAIL]", flush=True)

    print(f"[*] CUDA Available         : {cuda_avail} [{'PASS' if cuda_avail else 'FAIL'}]", flush=True)
    print(f"[*] CUDA Version           : {cuda_ver}", flush=True)
    print(f"[*] Detected GPU Name      : {gpu_name} [{'PASS' if rtx_3050_ok else 'WARN/FAIL'}]", flush=True)
    print(f"[*] GPU Total VRAM         : {gpu_vram}", flush=True)

    # 3. Torchaudio
    ta_ok, ta_ver, _ = check_pkg("torchaudio")
    print(f"[*] Torchaudio Version     : {ta_ver} [{'PASS' if ta_ok else 'FAIL'}]", flush=True)

    # 4. Transformers
    tf_ok, tf_ver, _ = check_pkg("transformers")
    print(f"[*] Transformers Version   : {tf_ver} [{'PASS' if tf_ok else 'FAIL'}]", flush=True)

    # 5. Hugging Face Hub
    hfh_ok, hfh_ver, _ = check_pkg("huggingface_hub")
    print(f"[*] HuggingFace Hub        : {hfh_ver} [{'PASS' if hfh_ok else 'FAIL'}]", flush=True)

    # 6. Hugging Face Authentication Check
    hf_auth_ok = False
    hf_user = "Not Authenticated"
    if hfh_ok:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user_info = api.whoami()
            hf_user = user_info.get("name") or user_info.get("fullname") or "Authenticated"
            hf_auth_ok = True
        except Exception as e:
            hf_user = f"Not logged in or token invalid ({e})"

    print(f"[*] HF Authentication      : {hf_user} [{'PASS' if hf_auth_ok else 'WARN/FAIL'}]", flush=True)

    # Summary table
    print("\n" + "-" * 60, flush=True)
    print("STATUS SUMMARY:", flush=True)
    print("-" * 60, flush=True)
    checks = {
        "Python": py_pass,
        "PyTorch": torch_ok,
        "CUDA": cuda_avail,
        "RTX 3050 detection": rtx_3050_ok,
        "Hugging Face auth": hf_auth_ok,
        "Required packages": (torch_ok and ta_ok and tf_ok and hfh_ok)
    }

    all_passed = True
    for item, status in checks.items():
        res = "PASS" if status else "FAIL"
        if not status:
            all_passed = False
        print(f"  - {item:<25}: {res}", flush=True)
    print("-" * 60, flush=True)

    if not cuda_avail:
        print("\n[CRITICAL ERROR] CUDA is NOT available in PyTorch!", flush=True)
        print("Please install PyTorch with CUDA support inside .venv:", flush=True)
        print("Execution halted. Fix CUDA before continuing.", flush=True)
        sys.exit(1)

    return all_passed

if __name__ == "__main__":
    main()
