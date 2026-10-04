import os
import sys
import io
import json
import time
import math
import torch
import soundfile as sf
import librosa
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

audio_path = r"D:\ASR\record-1769162120024.wav"

if not os.path.exists(audio_path):
    print(f"[!] Error: Audio file not found at {audio_path}")
    sys.exit(1)

print("=" * 80)
print("     LONG AUDIO TRANSCRIPTION PIPELINE")
print("=" * 80)
print(f"[*] Processing audio file: {audio_path}")

info = sf.info(audio_path)
total_duration = info.duration
print(f"[*] Original Sample Rate: {info.samplerate} Hz | Channels: {info.channels} | Duration: {total_duration/60.0:.2f} mins ({total_duration:.2f} s)")

# Load audio resampled to 16000 Hz
print("[*] Loading and resampling audio to 16kHz...")
waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

if not torch.cuda.is_available():
    print("[!] Error: CUDA GPU required.")
    sys.exit(1)

device = torch.device("cuda:0")

from transcribe import load_indic_conformer
print("[*] Loading IndicConformer model onto GPU...")
model, model_id = load_indic_conformer(device)

# Chunking settings (30-second chunks = 480,000 samples at 16kHz)
chunk_sec = 30.0
chunk_samples = int(chunk_sec * sr)
total_samples = len(waveform)
num_chunks = math.ceil(total_samples / chunk_samples)

print(f"[*] Splitting {total_duration:.1f}s audio into {num_chunks} chunks ({chunk_sec}s per chunk)...")

segments = []
full_transcript_words = []

start_time_all = time.perf_counter()

for i in range(num_chunks):
    start_idx = i * chunk_samples
    end_idx = min((i + 1) * chunk_samples, total_samples)
    chunk_wave = waveform[start_idx:end_idx]
    
    start_sec = round(start_idx / sr, 2)
    end_sec = round(end_idx / sr, 2)
    chunk_dur = end_sec - start_sec
    
    if len(chunk_wave) == 0:
        continue

    inputs = torch.tensor(chunk_wave, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inputs, "hi", "ctc")
        if isinstance(out, (list, tuple)):
            text = out[0]
        else:
            text = str(out)
            
    clean_text = text.strip()
    if clean_text:
        segments.append({
            "segment_index": i + 1,
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "duration_sec": round(chunk_dur, 2),
            "transcript": clean_text
        })
        full_transcript_words.append(clean_text)
        print(f"[{start_sec:>6.1f}s - {end_sec:>6.1f}s] {clean_text}", flush=True)
    else:
        print(f"[{start_sec:>6.1f}s - {end_sec:>6.1f}s] <SILENCE>", flush=True)

    del inputs
    torch.cuda.empty_cache()

end_time_all = time.perf_counter()
total_proc_sec = end_time_all - start_time_all
speed_x = total_duration / total_proc_sec if total_proc_sec > 0 else 0

full_text = " ".join(full_transcript_words)

final_json = {
    "file_name": os.path.basename(audio_path),
    "file_path": audio_path,
    "duration_minutes": round(total_duration / 60.0, 2),
    "duration_seconds": round(total_duration, 2),
    "processing_time_seconds": round(total_proc_sec, 2),
    "speed_x": round(speed_x, 2),
    "model": model_id,
    "language": "hi",
    "full_transcript": full_text,
    "segments": segments
}

json_out_path = r"D:\ASR\results\record-1769162120024_transcript.json"
txt_out_path = r"D:\ASR\results\record-1769162120024_transcript.txt"

with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(final_json, f, indent=2, ensure_ascii=False)

with open(txt_out_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print("\n" + "=" * 80)
print(f"[*] TRANSCRIPTION COMPLETE!")
print(f"[*] Total Audio Duration : {total_duration/60.0:.2f} mins ({total_duration:.1f} s)")
print(f"[*] Processing Time      : {total_proc_sec/60.0:.2f} mins ({total_proc_sec:.1f} s)")
print(f"[*] Speed Multiplier     : {speed_x:.2f}x real-time")
print(f"[*] JSON saved to        : {json_out_path}")
print(f"[*] Text saved to        : {txt_out_path}")
print("=" * 80)

