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
    print(json.dumps({"error": f"Audio file not found: {audio_path}"}))
    sys.exit(1)

print("=" * 80)
print("     PURE PYTORCH SPEAKER DIARIZATION & TRANSCRIPTION PIPELINE")
print("=" * 80)
print(f"[*] Processing Audio: {audio_path}")

# Load audio at 16kHz
waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
total_dur = len(waveform) / sr

print(f"[*] Total Audio Duration: {total_dur/60.0:.2f} mins ({total_dur:.1f} s)")

# Step 1: Detect Speech Intervals using Librosa VAD
print("[*] Performing Voice Activity Detection (VAD)...")
non_silent_intervals = librosa.effects.split(waveform, top_db=25, frame_length=2048, hop_length=512)

print(f"[*] Detected {len(non_silent_intervals)} active speech segments.")

# Load IndicConformer model
if not torch.cuda.is_available():
    print("[!] GPU unavailable.")
    sys.exit(1)

device = torch.device("cuda:0")
from transcribe import load_indic_conformer
model, model_id = load_indic_conformer(device)

# Process speech segments and extract acoustic MFCC features
features_list = []
valid_intervals = []

for interval in non_silent_intervals:
    start_sec = interval[0] / sr
    end_sec = interval[1] / sr
    dur = end_sec - start_sec
    if dur < 0.8: # Skip micro segments < 0.8s
        continue
        
    seg_audio = waveform[interval[0]:interval[1]]
    mfcc = librosa.feature.mfcc(y=seg_audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    features_list.append(mfcc_mean)
    valid_intervals.append((start_sec, end_sec, interval[0], interval[1]))

features_np = np.array(features_list)
print(f"[*] Extracted acoustic features for {len(valid_intervals)} valid speech intervals.")

# Pure PyTorch 2-Cluster K-Means on GPU
feat_tensor = torch.tensor(features_np, dtype=torch.float32).to(device)

idx_min = torch.argmin(feat_tensor[:, 0])
idx_max = torch.argmax(feat_tensor[:, 0])
c0 = feat_tensor[idx_min]
c1 = feat_tensor[idx_max]
centroids = torch.stack([c0, c1])

for _ in range(25):
    dists = torch.cdist(feat_tensor, centroids)
    labels_tensor = torch.argmin(dists, dim=1)
    mask0 = (labels_tensor == 0)
    mask1 = (labels_tensor == 1)
    if mask0.any():
        centroids[0] = feat_tensor[mask0].mean(dim=0)
    if mask1.any():
        centroids[1] = feat_tensor[mask1].mean(dim=0)

labels = labels_tensor.cpu().numpy()

# Merge adjacent segments belonging to the same speaker
merged_turns = []
current_turn = None

for idx, (start_sec, end_sec, start_sample, end_sample) in enumerate(valid_intervals):
    spk_id = f"SPEAKER_0{labels[idx]}"
    
    if current_turn is None:
        current_turn = {
            "speaker": spk_id,
            "start": start_sec,
            "end": end_sec,
            "start_sample": start_sample,
            "end_sample": end_sample
        }
    elif current_turn["speaker"] == spk_id and (start_sec - current_turn["end"]) < 1.5:
        current_turn["end"] = end_sec
        current_turn["end_sample"] = end_sample
    else:
        merged_turns.append(current_turn)
        current_turn = {
            "speaker": spk_id,
            "start": start_sec,
            "end": end_sec,
            "start_sample": start_sample,
            "end_sample": end_sample
        }

if current_turn:
    merged_turns.append(current_turn)

print(f"[*] Merged into {len(merged_turns)} speaker dialogue turns.")

# Step 2: Transcribe each speaker turn with IndicConformer
diarized_json_list = []

for turn_idx, turn in enumerate(merged_turns):
    seg_audio = waveform[turn["start_sample"]:turn["end_sample"]]
    
    max_chunk = 30 * sr
    sub_texts = []
    
    for c_start in range(0, len(seg_audio), max_chunk):
        c_end = min(c_start + max_chunk, len(seg_audio))
        sub_wave = seg_audio[c_start:c_end]
        if len(sub_wave) == 0:
            continue
            
        inputs = torch.tensor(sub_wave, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inputs, "hi", "ctc")
            t_str = out[0] if isinstance(out, (list, tuple)) else str(out)
            if t_str.strip():
                sub_texts.append(t_str.strip())
        del inputs
        torch.cuda.empty_cache()
        
    turn_transcript = " ".join(sub_texts)
    
    if turn_transcript.strip():
        diarized_json_list.append({
            "turn_number": turn_idx + 1,
            "speaker": turn["speaker"],
            "start_time": f"{turn['start']:.2f}s",
            "end_time": f"{turn['end']:.2f}s",
            "start_seconds": round(turn["start"], 2),
            "end_seconds": round(turn["end"], 2),
            "duration_seconds": round(turn["end"] - turn["start"], 2),
            "transcript": turn_transcript.strip()
        })
        print(f"[{turn['start']:>6.1f}s - {turn['end']:>6.1f}s] {turn['speaker']}: {turn_transcript.strip()}", flush=True)

# Step 3: Save Output JSON
output_json = {
    "file_name": os.path.basename(audio_path),
    "file_path": audio_path,
    "total_duration_minutes": round(total_dur / 60.0, 2),
    "total_duration_seconds": round(total_dur, 2),
    "total_speaker_turns": len(diarized_json_list),
    "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
    "model_used": model_id,
    "language": "hi",
    "diarized_transcript": diarized_json_list
}

json_path = r"D:\ASR\results\record-1769162120024_diarized.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"[*] DIARIZATION & TRANSCRIPTION COMPLETE!")
print(f"[*] Output saved to: {json_path}")
print("=" * 80)

