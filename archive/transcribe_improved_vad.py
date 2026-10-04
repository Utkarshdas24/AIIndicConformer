import os
import sys
import io
import json
import time
import math
import torch
import soundfile as sf
import librosa
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

audio_path = r"D:\ASR\record-1769162120024.wav"

if not os.path.exists(audio_path):
    print(f"[!] Audio file not found: {audio_path}")
    sys.exit(1)

print("=" * 80)
print("     HIGH-PRECISION VAD & OVERLAPPING SLIDING WINDOW TRANSCRIPTION")
print("=" * 80)

# Load 8kHz audio and resample to 16kHz
print("[*] Loading audio and resampling to 16kHz...")
waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
total_dur = len(waveform) / sr

# Audio pre-emphasis and gain normalization to boost quiet speech
waveform = librosa.effects.preemphasis(waveform)
max_val = np.max(np.abs(waveform))
if max_val > 0:
    waveform = waveform / max_val * 0.95

print(f"[*] Total Audio Duration: {total_dur/60.0:.2f} mins ({total_dur:.1f} s)")

# Load IndicConformer model
if not torch.cuda.is_available():
    print("[!] GPU unavailable.")
    sys.exit(1)

device = torch.device("cuda:0")
from transcribe import load_indic_conformer
model, model_id = load_indic_conformer(device)

# Sliding Window with 4-second overlap to prevent dropping words at boundaries
window_sec = 25.0
overlap_sec = 4.0
step_sec = window_sec - overlap_sec

window_samples = int(window_sec * sr)
step_samples = int(step_sec * sr)
total_samples = len(waveform)

chunks = []
curr_start = 0

while curr_start < total_samples:
    curr_end = min(curr_start + window_samples, total_samples)
    chunks.append((curr_start, curr_end))
    if curr_end == total_samples:
        break
    curr_start += step_samples

print(f"[*] Generated {len(chunks)} overlapping sliding windows (25s window, 4s overlap)...")

transcribed_segments = []
full_words_set = []

start_time_all = time.perf_counter()

for idx, (c_start, c_end) in enumerate(chunks):
    c_sec_start = round(c_start / sr, 2)
    c_sec_end = round(c_end / sr, 2)
    
    seg_audio = waveform[c_start:c_end]
    
    energy = np.mean(seg_audio**2)
    if energy < 1e-4:
        continue
        
    inputs = torch.tensor(seg_audio, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inputs, "hi", "ctc")
        t_str = out[0] if isinstance(out, (list, tuple)) else str(out)
        clean_text = t_str.strip()
        
    if clean_text:
        transcribed_segments.append({
            "segment_id": idx + 1,
            "start_time_sec": c_sec_start,
            "end_time_sec": c_sec_end,
            "transcript": clean_text
        })
        full_words_set.append(clean_text)
        print(f"[{c_sec_start:>6.1f}s - {c_sec_end:>6.1f}s] {clean_text}", flush=True)

    del inputs
    torch.cuda.empty_cache()

end_time_all = time.perf_counter()
total_proc_sec = end_time_all - start_time_all

full_raw_text = " ".join(full_words_set)

output_json = {
    "file_name": os.path.basename(audio_path),
    "total_duration_minutes": round(total_dur / 60.0, 2),
    "total_duration_seconds": round(total_dur, 2),
    "processing_time_seconds": round(total_proc_sec, 2),
    "speed_x": round(total_dur / total_proc_sec, 2),
    "window_type": "25s_window_4s_overlap",
    "full_transcript": full_raw_text,
    "segments": transcribed_segments
}

json_path = r"D:\ASR\results\record-1769162120024_improved_vad.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"[*] IMPROVED TRANSCRIPTION COMPLETE!")
print(f"[*] Processing Time : {total_proc_sec:.1f} s")
print(f"[*] JSON saved to   : {json_path}")
print("=" * 80)

