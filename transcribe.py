import os
import sys
import torch
import soundfile as sf
import torchaudio
from transformers import AutoModel

def load_indic_conformer(device):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.environ.get("INDIC_MODEL_DIR", os.path.join(base_dir, "models", "indic-conformer-600m"))
    
    # Check if local model directory exists
    if os.path.isdir(local_model_dir):
        abs_path = os.path.abspath(local_model_dir)
        print(f"[*] Loading offline local IndicConformer model from: {abs_path}", flush=True)
        try:
            model = AutoModel.from_pretrained(abs_path, trust_remote_code=True, local_files_only=True)
            model = model.to(device)
            model.eval()
            return model, f"local:{abs_path}"
        except Exception as e:
            print(f"[!] Local model load note: {e}. Trying Hub repository...", flush=True)

    # Fallback to HuggingFace Hub repository
    fallback_id = os.environ.get("INDIC_HUB_REPO", "sunilmahendrakar/indic-conformer-600m-multilingual")
    print(f"[*] Loading model from Hugging Face Hub: {fallback_id} ...", flush=True)
    model = AutoModel.from_pretrained(fallback_id, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, fallback_id

def load_audio(audio_path, target_sr=16000):
    try:
        data, sr = sf.read(audio_path, dtype="float32")
        w = torch.tensor(data)
        if w.ndim > 1:
            w = w.mean(dim=1)
        if sr != target_sr:
            w = torchaudio.functional.resample(w, sr, target_sr)
        return w, target_sr
    except Exception:
        import librosa
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return torch.tensor(waveform, dtype=torch.float32), target_sr
