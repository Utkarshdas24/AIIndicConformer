"""Master pipeline runner for IndicConformer ASR & LLM Call Intelligence.
Works cross-platform on Windows, Linux, and inside Docker containers.
"""
import os
import sys
import io
import time
import json
import argparse
import subprocess
from pathlib import Path

# Force UTF-8 standard output
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End IndicConformer GPU ASR & LLM Diarization Pipeline"
    )
    parser.add_argument("audio", nargs="?", default="record-1769162120024.wav",
                        help="Path to audio file (.wav, .mp3, .flac)")
    parser.add_argument("--lang", default="hi",
                        help="Indic language code (default: hi)")
    parser.add_argument("--decoder", default="rnnt", choices=["rnnt", "ctc"],
                        help="ASR decoder to use (default: rnnt - most accurate)")
    parser.add_argument("--out-dir", default="results",
                        help="Directory to save output JSON and MD reports (default: results)")
    parser.add_argument("--skip-asr", action="store_true",
                        help="Skip ASR stage if segments JSON already exists")

    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(audio_path) and not args.skip_asr:
        print(f"[!] Error: Audio file not found at: {audio_path}")
        print("    Please provide a valid audio path or place audio in recordings/ directory.")
        sys.exit(1)

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    segments_json = os.path.join(out_dir, f"{stem}_asr_segments.json")
    final_json = os.path.join(out_dir, f"{stem}_final.json")
    final_md = os.path.join(out_dir, f"{stem}_final.md")

    python_exe = sys.executable
    base_dir = os.path.dirname(os.path.abspath(__file__))

    total_t0 = time.perf_counter()

    # ================= Stage 1: GPU ASR Segmentation =================
    if not args.skip_asr:
        print("\n" + "=" * 65)
        print(f"[*] STAGE 1: IndicConformer GPU ASR ({args.decoder.upper()})")
        print(f"[*] Input Audio : {audio_path}")
        print(f"[*] Output JSON : {segments_json}")
        print("=" * 65)

        t1_start = time.perf_counter()
        asr_script = os.path.join(base_dir, "asr_segments.py")
        cmd_asr = [python_exe, asr_script, audio_path, args.lang, args.decoder]
        
        proc_asr = subprocess.run(cmd_asr, check=False)
        if proc_asr.returncode != 0:
            print(f"[!] Error: ASR stage failed with exit code {proc_asr.returncode}")
            sys.exit(proc_asr.returncode)

        t1_dur = time.perf_counter() - t1_start
        print(f"[*] Stage 1 completed in {t1_dur / 60:.2f} minutes.")
    else:
        print(f"[*] Skipping Stage 1. Reusing existing segments: {segments_json}")

    # ================= Stage 2: LLM Diarization & Findings =================
    print("\n" + "=" * 65)
    print(f"[*] STAGE 2: LLM Speaker Alignment & Fact Extraction")
    print(f"[*] Processing  : {stem}")
    print("=" * 65)

    t2_start = time.perf_counter()
    diarize_script = os.path.join(base_dir, "final_call_json.py")
    cmd_diarize = [python_exe, diarize_script, stem, "--decoder", args.decoder]

    proc_diarize = subprocess.run(cmd_diarize, check=False)
    if proc_diarize.returncode != 0:
        print(f"[!] Error: Stage 2 failed with exit code {proc_diarize.returncode}")
        sys.exit(proc_diarize.returncode)

    t2_dur = time.perf_counter() - t2_start
    print(f"[*] Stage 2 completed in {t2_dur / 60:.2f} minutes.")

    # ================= Stage 3: Turn-Level Summary & Findings =================
    print("\n" + "=" * 65)
    print(f"[*] STAGE 3: Enriching Turns with English Summaries & Findings")
    print(f"[*] Target JSON : {final_json}")
    print("=" * 65)

    t3_start = time.perf_counter()
    enrich_script = os.path.join(base_dir, "enrich_final_json.py")
    cmd_enrich = [python_exe, enrich_script]

    proc_enrich = subprocess.run(cmd_enrich, check=False)
    if proc_enrich.returncode != 0:
        print(f"[!] Error: Stage 3 failed with exit code {proc_enrich.returncode}")
        sys.exit(proc_enrich.returncode)

    t3_dur = time.perf_counter() - t3_start
    total_dur = time.perf_counter() - total_t0

    print("\n" + "=" * 65)
    print(f"[*] PIPELINE COMPLETE! Total wall clock: {total_dur / 60:.2f} minutes")
    print(f"[*] Final Structured JSON : {final_json}")
    print(f"[*] Human-Readable Report : {final_md}")
    print("=" * 65)

if __name__ == "__main__":
    main()
