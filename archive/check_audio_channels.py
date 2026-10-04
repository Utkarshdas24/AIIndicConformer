import os
import sys
import io
import json
import torch
import soundfile as sf
import librosa
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

audio_path = r"D:\ASR\record-1769162120024.wav"

if not os.path.exists(audio_path):
    print(json.dumps({"error": "Audio file not found"}))
    sys.exit(1)

info = sf.info(audio_path)
print(f"[*] Audio channels: {info.channels} | Duration: {info.duration:.2f}s | Sample Rate: {info.samplerate}Hz")

# Load raw audio data
data, orig_sr = sf.read(audio_path)

# Let's inspect channel data
if data.ndim == 2 and data.shape[1] == 2:
    print("[*] Dual-channel Stereo Audio Detected!")
    is_stereo = True
else:
    print("[*] Single-channel Mono Audio Detected.")
    is_stereo = False
