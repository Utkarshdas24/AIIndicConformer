import os
import sys
import io
import json
import torch
import soundfile as sf
import librosa

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

audio_path = r"D:\ASR\record-1769162120024.wav"

if not os.path.exists(audio_path):
    print(json.dumps({"error": f"Audio file not found: {audio_path}"}))
    sys.exit(1)

# Check audio file details using soundfile
info = sf.info(audio_path)
print(f"[*] Audio Info: {info.samplerate} Hz, {info.channels} channels, {info.duration:.2f} seconds")

# Load audio waveform at 16000 Hz
waveform, sr = librosa.load(audio_path, sr=16000, mono=False)

if not torch.cuda.is_available():
    print(json.dumps({"error": "CUDA GPU unavailable"}))
    sys.exit(1)

device = torch.device("cuda:0")

from transcribe import load_indic_conformer
model, loaded_model_id = load_indic_conformer(device)

# Function to run transcription
def transcribe_channel(mono_signal, lang):
    tensor_input = torch.tensor(mono_signal, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor_input, lang, "ctc")
        if isinstance(out, (list, tuple)):
            text = out[0]
        else:
            text = str(out)
    return text.strip()

# Test both Hindi ('hi') and Marathi ('mr')
results = {
    "file": os.path.basename(audio_path),
    "file_path": audio_path,
    "duration_seconds": round(info.duration, 2),
    "sample_rate": info.samplerate,
    "channels": info.channels,
    "transcriptions": {}
}

if waveform.ndim == 1:
    # Mono audio
    for lang in ["hi", "mr"]:
        results["transcriptions"][lang] = transcribe_channel(waveform, lang)
else:
    # Multi-channel / Stereo audio
    results["stereo_channels"] = {}
    ch0 = waveform[0]
    ch1 = waveform[1]
    for lang in ["hi", "mr"]:
        results["stereo_channels"][f"channel_0_{lang}"] = transcribe_channel(ch0, lang)
        results["stereo_channels"][f"channel_1_{lang}"] = transcribe_channel(ch1, lang)

# Output final clean JSON
json_output_path = r"D:\ASR\results\record-1769162120024_transcript.json"
os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n--- TRANSCRIPTION JSON OUTPUT ---")
print(json.dumps(results, indent=2, ensure_ascii=False))
