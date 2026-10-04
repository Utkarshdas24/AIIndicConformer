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

def process_call(audio_path, language="hi"):
    if not os.path.exists(audio_path):
        print(f"[!] Error: File not found: {audio_path}")
        return None

    base_name = os.path.basename(audio_path)
    file_stem = os.path.splitext(base_name)[0]
    results_dir = r"D:\ASR\results"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"     PRODUCTION CALL TRANSCRIPTION & DIARIZATION PIPELINE")
    print("=" * 80)
    print(f"[*] Processing Call Recording: {base_name}")

    # Load audio at 16kHz
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_dur = len(waveform) / sr

    # Pre-emphasis filter to enhance high frequency consonant speech
    waveform = librosa.effects.preemphasis(waveform)
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.95

    print(f"[*] Duration: {total_dur/60.0:.2f} mins ({total_dur:.1f} s)")

    if not torch.cuda.is_available():
        print("[!] GPU unavailable.")
        return None

    device = torch.device("cuda:0")
    from transcribe import load_indic_conformer
    model, model_id = load_indic_conformer(device)

    # 1. Voice Activity Detection & Speaker Clustering
    print("[*] Running Voice Activity Detection (VAD) & Speaker Clustering...")
    intervals = librosa.effects.split(waveform, top_db=25, frame_length=2048, hop_length=512)

    valid_intervals = []
    features_list = []

    for interval in intervals:
        start_sec = interval[0] / sr
        end_sec = interval[1] / sr
        dur = end_sec - start_sec
        if dur < 0.8:
            continue
        seg_audio = waveform[interval[0]:interval[1]]
        mfcc = librosa.feature.mfcc(y=seg_audio, sr=sr, n_mfcc=13)
        features_list.append(np.mean(mfcc.T, axis=0))
        valid_intervals.append((start_sec, end_sec, interval[0], interval[1]))

    if len(features_list) == 0:
        print("[!] No speech detected.")
        return None

    feat_tensor = torch.tensor(np.array(features_list), dtype=torch.float32).to(device)

    # Pure PyTorch GPU K-Means clustering (K=2)
    idx_min = torch.argmin(feat_tensor[:, 0])
    idx_max = torch.argmax(feat_tensor[:, 0])
    centroids = torch.stack([feat_tensor[idx_min], feat_tensor[idx_max]])

    for _ in range(25):
        dists = torch.cdist(feat_tensor, centroids)
        labels_tensor = torch.argmin(dists, dim=1)
        m0 = (labels_tensor == 0)
        m1 = (labels_tensor == 1)
        if m0.any(): centroids[0] = feat_tensor[m0].mean(dim=0)
        if m1.any(): centroids[1] = feat_tensor[m1].mean(dim=0)

    labels = labels_tensor.cpu().numpy()

    # Merge consecutive intervals belonging to same speaker
    merged_turns = []
    curr = None

    for idx, (s_sec, e_sec, s_samp, e_samp) in enumerate(valid_intervals):
        spk = f"SPEAKER_0{labels[idx]}"
        if curr is None:
            curr = {"speaker": spk, "start": s_sec, "end": e_sec, "s_samp": s_samp, "e_samp": e_samp}
        elif curr["speaker"] == spk and (s_sec - curr["end"]) < 1.5:
            curr["end"] = e_sec
            curr["e_samp"] = e_samp
        else:
            merged_turns.append(curr)
            curr = {"speaker": spk, "start": s_sec, "end": e_sec, "s_samp": s_samp, "e_samp": e_samp}
    if curr:
        merged_turns.append(curr)

    print(f"[*] Identified {len(merged_turns)} speaker dialogue turns.")

    # 2. GPU Transcription per Dialogue Turn
    print("[*] Transcribing dialogue turns on GPU...")
    start_gpu = time.perf_counter()

    diarized_turns = []
    full_transcript_list = []

    for turn_idx, turn in enumerate(merged_turns):
        seg_audio = waveform[turn["s_samp"]:turn["e_samp"]]
        
        # Slicing long turns with 25s window & 4s overlap to prevent word drops
        max_chunk = int(25 * sr)
        overlap = int(4 * sr)
        step = max_chunk - overlap
        
        turn_len = len(seg_audio)
        sub_texts = []
        
        c_start = 0
        while c_start < turn_len:
            c_end = min(c_start + max_chunk, turn_len)
            chunk_wave = seg_audio[c_start:c_end]
            
            if len(chunk_wave) > 0 and np.mean(chunk_wave**2) > 1e-5:
                inputs = torch.tensor(chunk_wave, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inputs, language, "ctc")
                    txt = out[0] if isinstance(out, (list, tuple)) else str(out)
                    if txt.strip():
                        sub_texts.append(txt.strip())
                del inputs
                torch.cuda.empty_cache()
                
            if c_end == turn_len:
                break
            c_start += step

        clean_turn_text = " ".join(sub_texts).strip()
        if clean_turn_text:
            diarized_turns.append({
                "turn_number": len(diarized_turns) + 1,
                "speaker": turn["speaker"],
                "start_sec": round(turn["start"], 2),
                "end_sec": round(turn["end"], 2),
                "duration_sec": round(turn["end"] - turn["start"], 2),
                "transcript": clean_turn_text
            })
            full_transcript_list.append(f"[{turn['speaker']}]: {clean_turn_text}")

    end_gpu = time.perf_counter()
    proc_sec = end_gpu - start_gpu
    speed_x = total_dur / proc_sec if proc_sec > 0 else 0

    full_transcript = "\n".join(full_transcript_list)

    final_payload = {
        "file_name": base_name,
        "file_path": audio_path,
        "duration_minutes": round(total_dur / 60.0, 2),
        "duration_seconds": round(total_dur, 2),
        "processing_time_seconds": round(proc_sec, 2),
        "speed_x": round(speed_x, 2),
        "model_used": model_id,
        "language": language,
        "total_dialogue_turns": len(diarized_turns),
        "diarized_turns": diarized_turns
    }

    # Save JSON and Markdown report
    json_path = os.path.join(results_dir, f"{file_stem}_production_call_report.json")
    md_path = os.path.join(results_dir, f"{file_stem}_production_call_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    md_content = f"""# Production Call Transcription & Diarization Report

## Call Overview
- **File**: `{base_name}`
- **Call Duration**: {total_dur/60.0:.2f} minutes ({total_dur:.1f} s)
- **Processing Time**: {proc_sec:.1f} seconds ({proc_sec/60.0:.2f} mins)
- **GPU Throughput Speed**: **{speed_x:.2f}x Real-Time**
- **Dialogue Turns Detected**: {len(diarized_turns)} turns

---

## Diarized Call Transcript

"""
    for t in diarized_turns[:30]: # top 30 turns in md
        md_content += f"**[{t['speaker']}]** ({t['start_sec']}s - {t['end_sec']}s):\n> {t['transcript']}\n\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print(f"[*] PRODUCTION PIPELINE COMPLETE!")
    print(f"[*] Speed Multiplier     : {speed_x:.2f}x Real-Time")
    print(f"[*] Production JSON      : {json_path}")
    print(f"[*] Call Report Markdown : {md_path}")
    print("=" * 80)

if __name__ == "__main__":
    audio_file = r"D:\ASR\record-1769162120024.wav"
    process_call(audio_file, language="hi")

