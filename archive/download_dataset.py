import os
import sys
import io
import argparse
import pandas as pd
import soundfile as sf

# Redirect HF Cache to D drive
os.environ["HF_HOME"] = r"D:\hf_cache"

from datasets import load_dataset, Audio

LANGUAGES = {
    'hi': 'Hindi',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'ta': 'Tamil',
    'te': 'Telugu'
}

DATASET_NAME = "ekacare/vistaar_small_asr_eval"

def download_and_create_manifest(mode='sample'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    audio_dir = os.path.join(dataset_dir, "audio")
    manifest_path = os.path.join(dataset_dir, "manifest.csv")

    os.makedirs(audio_dir, exist_ok=True)

    # Duration targets per language (in seconds)
    # sample mode: ~30s per lang (6 langs * 30s = 3 minutes total)
    # full mode: ~600s (10 mins) per lang (6 langs * 10m = 60 minutes total)
    target_duration_per_lang = 30.0 if mode == 'sample' else 600.0

    print(f"[*] Downloading Vistaar Small ASR dataset ({mode.upper()} mode)...", flush=True)
    print(f"[*] Target duration per language: {target_duration_per_lang / 60:.1f} minutes", flush=True)

    manifest_rows = []

    for lang_code, lang_name in LANGUAGES.items():
        print(f"\n[+] Fetching dataset split for language: {lang_name} ({lang_code})...", flush=True)
        try:
            ds = load_dataset(DATASET_NAME, lang_code, split='test')
            ds = ds.cast_column('audio', Audio(decode=False))
        except Exception as e:
            print(f"[!] Could not load dataset for {lang_code}: {e}", flush=True)
            continue

        accumulated_duration = 0.0
        saved_count = 0

        for idx, item in enumerate(ds):
            if accumulated_duration >= target_duration_per_lang:
                break

            audio_data = item['audio']
            if isinstance(audio_data, dict) and 'bytes' in audio_data and audio_data['bytes']:
                audio_bytes = audio_data['bytes']
                array, sr = sf.read(io.BytesIO(audio_bytes))
            elif isinstance(audio_data, dict) and 'path' in audio_data and audio_data['path']:
                array, sr = sf.read(audio_data['path'])
            else:
                continue

            transcript = item.get('transcript') or item.get('text') or item.get('normalized_text') or ""
            duration_sec = len(array) / float(sr)

            filename = f"{lang_code}_{idx:04d}.wav"
            rel_path = os.path.join("audio", filename)
            abs_path = os.path.join(audio_dir, filename)

            sf.write(abs_path, array, sr)

            manifest_rows.append({
                'file': rel_path,
                'language': lang_code,
                'reference_transcript': str(transcript).strip(),
                'duration_seconds': round(duration_sec, 2),
                'source': f"vistaar_{lang_code}"
            })

            accumulated_duration += duration_sec
            saved_count += 1

        print(f"    -> Saved {saved_count} files for {lang_name} ({accumulated_duration/60:.2f} mins / {accumulated_duration:.1f}s)", flush=True)

    if not manifest_rows:
        print("[!] No audio files saved.", flush=True)
        sys.exit(1)

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(manifest_path, index=False, encoding='utf-8')

    print("\n" + "=" * 60, flush=True)
    print("      DATASET DOWNLOAD & MANIFEST SUMMARY", flush=True)
    print("=" * 60, flush=True)
    num_files = len(df_manifest)
    total_dur_sec = df_manifest['duration_seconds'].sum()
    total_dur_min = total_dur_sec / 60.0
    avg_dur_sec = df_manifest['duration_seconds'].mean()

    print(f"Number of files        : {num_files}", flush=True)
    print(f"Total duration         : {total_dur_min:.2f} minutes ({total_dur_sec:.1f} seconds)", flush=True)
    print(f"Average file duration  : {avg_dur_sec:.2f} seconds", flush=True)
    print("\nLanguage distribution:", flush=True)
    lang_dist = df_manifest.groupby('language')['duration_seconds'].agg(['count', 'sum'])
    lang_dist['sum_minutes'] = lang_dist['sum'] / 60.0
    for l_code, row in lang_dist.iterrows():
        l_name = LANGUAGES.get(l_code, l_code)
        print(f"  - {l_name:<12} ({l_code}): {int(row['count'])} files | {row['sum_minutes']:.2f} mins ({row['sum']:.1f}s)", flush=True)

    print(f"\n[*] Manifest saved to: {manifest_path}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Vistaar ASR Dataset Sample / Full benchmark set")
    parser.add_argument("--mode", choices=['sample', 'full'], default='sample', help="'sample' (1-5 min) or 'full' (~60 min)")
    args = parser.parse_args()
    download_and_create_manifest(args.mode)
