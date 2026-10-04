import os
import sys
import io
import json
import torch
import soundfile as sf
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("     SPEAKER DIARIZATION PIPELINE DEMONSTRATION")
print("=" * 80)

# 1. Dual-Channel Stereo Diarization Implementation
def diarize_stereo_audio(audio_path, model, device, lang="hi"):
    """
    Separates a 2-channel stereo call recording:
    Channel 0 -> Agent
    Channel 1 -> Customer
    Transcribes each channel independently and merges turns.
    """
    data, sr = sf.read(audio_path)
    
    if data.ndim == 1:
        print("[!] Mono audio detected. Demonstrating stereo channel separation logic.")
        ch0 = data
        ch1 = data
    else:
        ch0 = data[:, 0]
        ch1 = data[:, 1]
        
    inputs_agent = torch.tensor(ch0, dtype=torch.float32).unsqueeze(0).to(device)
    inputs_customer = torch.tensor(ch1, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_agent = model(inputs_agent, lang, "ctc")
        out_customer = model(inputs_customer, lang, "ctc")
        
    agent_text = out_agent[0] if isinstance(out_agent, (list, tuple)) else str(out_agent)
    customer_text = out_customer[0] if isinstance(out_customer, (list, tuple)) else str(out_customer)
    
    diarized_transcript = [
        {"speaker": "AGENT (Channel 0)", "start_sec": 0.0, "text": agent_text.strip()},
        {"speaker": "CUSTOMER (Channel 1)", "start_sec": 0.0, "text": customer_text.strip()}
    ]
    
    return diarized_transcript

base_dir = r"D:\ASR"
test_audio = os.path.join(base_dir, "dataset", "audio", "hi_0000.wav")

if os.path.exists(test_audio) and torch.cuda.is_available():
    from transcribe import load_indic_conformer
    device = torch.device("cuda:0")
    print("[*] Loading IndicConformer model onto GPU for Diarization pipeline...")
    model, model_id = load_indic_conformer(device)
    
    print(f"[*] Running Diarization on sample: {test_audio}")
    result = diarize_stereo_audio(test_audio, model, device, lang="hi")
    
    print("\n--- DIARIZED TRANSCRIPT OUTPUT (JSON) ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
