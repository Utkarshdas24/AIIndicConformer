"""Stage 1: exact, timestamped ASR segments (no LLM, no text editing)."""
import os, sys, io, json, time
import numpy as np, torch, torchaudio, soundfile as sf
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from transcribe import load_indic_conformer

AUDIO = sys.argv[1] if len(sys.argv) > 1 else r"D:\ASR\record-1769162120024.wav"
LANG = sys.argv[2] if len(sys.argv) > 2 else "hi"
DECODER = sys.argv[3] if len(sys.argv) > 3 else "rnnt"   # single best decoder (RNNT: WER 8.73% vs CTC 9.32%)
SR = 16000
FRAME = int(0.02 * SR)          # 20 ms frames
MIN_GAP, MIN_SEG, PAD, MAX_SEG = 0.6, 0.3, 0.25, 8.0   # RNNT drops text on segments > ~10 s

stem = os.path.splitext(os.path.basename(AUDIO))[0]
OUT = rf"D:\ASR\results\{stem}_asr_segments.json"

data, osr = sf.read(AUDIO, dtype="float32")
w = torch.tensor(data)
if w.ndim > 1: w = w.mean(dim=1)
wav = torchaudio.functional.resample(w, osr, SR).numpy()
dur = len(wav) / SR
print(f"[*] {stem}: {dur/60:.2f} min, original {osr} Hz")

# ---- energy VAD (no librosa) ----
n = len(wav) // FRAME
rms = np.sqrt(np.mean(wav[: n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-10)
db = 20 * np.log10(rms)
floor = np.percentile(db, 10)
speech = db > floor + 10
segs, start = [], None
for i, s in enumerate(speech):
    if s and start is None: start = i
    if not s and start is not None: segs.append([start, i]); start = None
if start is not None: segs.append([start, n])
f2s = FRAME / SR
merged = []
for s, e in segs:
    if merged and (s - merged[-1][1]) * f2s < MIN_GAP: merged[-1][1] = e
    else: merged.append([s, e])
merged = [m for m in merged if (m[1] - m[0]) * f2s >= MIN_SEG]
# split long segments at the quietest frame
final = []
def split(s, e):
    if (e - s) * f2s <= MAX_SEG: final.append((s, e)); return
    lo, hi = s + int(2 / f2s), e - int(2 / f2s)
    cut = lo + int(np.argmin(db[lo:hi]))
    split(s, cut); split(cut, e)
for s, e in merged: split(s, e)

times = [(max(0.0, s * f2s - PAD), min(dur, e * f2s + PAD)) for s, e in final]
speech_sec = sum(e - s for s, e in times)
print(f"[*] VAD: {len(times)} segments, speech {speech_sec/60:.2f} min ({100*speech_sec/dur:.1f}%)")

# ---- ASR: one decoder, verbatim ----
dev = torch.device("cuda:0")
model, model_id = load_indic_conformer(dev)
def run(x, dec):
    with torch.no_grad():
        o = model(x, LANG, dec)
    return (o[0] if isinstance(o, (list, tuple)) else str(o)).strip()

def decode(a, b, depth=0):
    """Decode [a,b) seconds. If output is empty / too sparse, split at the quietest point and retry."""
    x = torch.tensor(wav[int(a * SR):int(b * SR)]).unsqueeze(0).to(dev)
    t = run(x, DECODER); del x
    if (b - a) > 3.0 and depth < 3 and len(t.split()) < 0.8 * (b - a):
        fa, fb = int(a / f2s) + int(1 / f2s), int(b / f2s) - int(1 / f2s)
        cut = (fa + int(np.argmin(db[fa:fb]))) * f2s if fb > fa else (a + b) / 2
        t2 = (decode(a, cut, depth + 1) + " " + decode(cut, b, depth + 1)).strip()
        if len(t2.split()) > len(t.split()): return t2
    return t

out, t0 = [], time.perf_counter()
for i, (s, e) in enumerate(times):
    txt = decode(s, e)
    mm = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
    out.append({"segment_id": i + 1, "start_sec": round(s, 2), "end_sec": round(e, 2),
                "timestamp": f"{mm(s)}-{mm(e)}", "transcript": txt})
    print(f"[{mm(s)}-{mm(e)}] {txt}", flush=True)
proc = time.perf_counter() - t0

json.dump({"file_name": os.path.basename(AUDIO), "duration_sec": round(dur, 2),
           "original_sample_rate": osr, "model": model_id, "language": LANG, "decoder": DECODER,
           "processing_sec": round(proc, 1), "speed_x": round(dur / proc, 2),
           "vad": {"segments": len(out), "speech_sec": round(speech_sec, 1)},
           "note": "Transcripts are raw model output, unedited.",
           "segments": out}, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"[*] Done in {proc/60:.1f} min ({dur/proc:.2f}x). Saved {OUT}")
