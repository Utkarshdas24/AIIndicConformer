"""Pick the better decoder: CTC vs RNNT WER/CER on Kathbath Hindi (53 files with references)."""
import sys, io, os, re, time, unicodedata, json
import pandas as pd, torch, torchaudio, soundfile as sf, jiwer
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from transcribe import load_indic_conformer

def norm(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = re.sub(r"[\u0964\u0965\.\?\!\,\-\:\;\"\'\(\)\[\]]", " ", t).replace("\u093c", "").lower()
    return " ".join(t.split())

df = pd.read_csv(r"D:\ASR\dataset\manifest.csv")
df = df[df["language"] == "hi"]
dev = torch.device("cuda:0")
model, _ = load_indic_conformer(dev)

res = {"ctc": {"refs": [], "hyps": [], "sec": 0.0}, "rnnt": {"refs": [], "hyps": [], "sec": 0.0}}
audio_sec = 0.0
for _, r in df.iterrows():
    data, osr = sf.read(os.path.join(r"D:\ASR\dataset", r["file"]), dtype="float32")
    w = torch.tensor(data)
    if w.ndim > 1: w = w.mean(dim=1)
    if osr != 16000: w = torchaudio.functional.resample(w, osr, 16000)
    audio_sec += len(w) / 16000
    x = w.unsqueeze(0).to(dev)
    for dec in ["ctc", "rnnt"]:
        t = time.perf_counter()
        with torch.no_grad():
            o = model(x, "hi", dec)
        res[dec]["sec"] += time.perf_counter() - t
        res[dec]["refs"].append(norm(r["reference_transcript"]))
        res[dec]["hyps"].append(norm(o[0] if isinstance(o, (list, tuple)) else o))

summary = {}
for dec, v in res.items():
    summary[dec] = {"WER_%": round(100 * jiwer.wer(v["refs"], v["hyps"]), 2),
                    "CER_%": round(100 * jiwer.cer(v["refs"], v["hyps"]), 2),
                    "speed_x": round(audio_sec / v["sec"], 2)}
    print(f"{dec.upper():5s} WER {summary[dec]['WER_%']}%  CER {summary[dec]['CER_%']}%  speed {summary[dec]['speed_x']}x")
json.dump({"files": len(df), "audio_min": round(audio_sec / 60, 2), "results": summary},
          open(r"D:\ASR\results\decoder_comparison_hi.json", "w"), indent=2)
