import sys
import io
import os
import re
import unicodedata
import pandas as pd
import numpy as np
import jiwer
import soundfile as sf

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

base_dir = r"D:\ASR"
results_dir = os.path.join(base_dir, "results")
dataset_dir = os.path.join(base_dir, "dataset")

csv_path = os.path.join(results_dir, "indicconformer_rtx3050_results.csv")
manifest_path = os.path.join(dataset_dir, "manifest.csv")

if not os.path.exists(csv_path):
    print(f"[!] Results file not found: {csv_path}")
    sys.exit(1)

df_results = pd.read_csv(csv_path)
df_filtered = df_results[df_results['language'].isin(['hi', 'mr'])].copy()

def normalize_indic(text):
    if not isinstance(text, str):
        return ""
    # Unicode NFKC
    text = unicodedata.normalize('NFKC', text)
    # Remove Hindi Purna Viram and standard punctuation
    text = re.sub(r'[\u0964\u0965\.\?\!\,\-\:\;\–\—\"\'\(\)\[\]\{\}\/\\]', ' ', text)
    # Remove nukta sign \u093c for fair phonetic comparison
    text = text.replace('\u093c', '')
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    words = text.split()
    return " ".join(words)

file_level_rows = []

for idx, row in df_filtered.iterrows():
    rel_file = row['file']
    abs_audio = os.path.join(dataset_dir, rel_file)
    lang = row['language']
    
    raw_ref = str(row.get('reference_transcript', '')).strip()
    raw_hyp = str(row.get('transcript', '')).strip()
    
    norm_ref = normalize_indic(raw_ref)
    norm_hyp = normalize_indic(raw_hyp)
    
    if len(norm_ref) == 0:
        continue

    # Jiwer word & character level evaluation
    w_out = jiwer.process_words(norm_ref, norm_hyp)
    c_out = jiwer.process_characters(norm_ref, norm_hyp)
    
    ref_words_list = norm_ref.split()
    hyp_words_list = norm_hyp.split()
    
    ref_words_count = len(ref_words_list)
    ref_chars_count = len(norm_ref)
    
    file_wer = w_out.wer
    file_cer = c_out.cer
    
    subs = w_out.substitutions
    dels = w_out.deletions
    ins = w_out.insertions
    hits = w_out.hits
    
    exact_match = (norm_ref == norm_hyp)
    
    # Check if audio file exists and check basic audio properties
    audio_exists = os.path.exists(abs_audio)
    duration_actual = 0.0
    if audio_exists:
        try:
            info = sf.info(abs_audio)
            duration_actual = info.duration
        except Exception:
            duration_actual = row.get('audio_duration_seconds', 0.0)
            
    file_level_rows.append({
        'file': rel_file,
        'language': lang,
        'audio_duration_seconds': round(duration_actual, 2),
        'processing_time_seconds': round(row.get('processing_time_seconds', 0.0), 3),
        'speed_x': round(row.get('speed_x', 0.0), 2),
        'raw_reference': raw_ref,
        'raw_prediction': raw_hyp,
        'normalized_reference': norm_ref,
        'normalized_prediction': norm_hyp,
        'wer': round(file_wer, 4),
        'cer': round(file_cer, 4),
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'hits': hits,
        'ref_words_count': ref_words_count,
        'ref_chars_count': ref_chars_count,
        'exact_match': exact_match
    })

df_file_level = pd.DataFrame(file_level_rows)

# Save hindi_marathi_file_level_results.csv
file_level_csv_path = os.path.join(results_dir, "hindi_marathi_file_level_results.csv")
df_file_level.to_csv(file_level_csv_path, index=False, encoding='utf-8')
print(f"[*] File level results saved to: {file_level_csv_path}")

# Calculate summary metrics per language & combined
def compute_lang_metrics(df_sub):
    tot_files = len(df_sub)
    tot_words = df_sub['ref_words_count'].sum()
    tot_chars = df_sub['ref_chars_count'].sum()
    tot_subs = df_sub['substitutions'].sum()
    tot_dels = df_sub['deletions'].sum()
    tot_ins = df_sub['insertions'].sum()
    tot_hits = df_sub['hits'].sum()
    
    corpus_wer = (tot_subs + tot_dels + tot_ins) / tot_words if tot_words > 0 else 0.0
    corpus_cer = df_sub['cer'].mean() # average CER
    
    exact_matches = df_sub['exact_match'].sum()
    exact_match_pct = (exact_matches / tot_files * 100.0) if tot_files > 0 else 0.0
    
    avg_wer = df_sub['wer'].mean()
    median_wer = df_sub['wer'].median()
    p90_wer = np.percentile(df_sub['wer'], 90) if tot_files > 0 else 0.0
    
    return {
        'num_files': tot_files,
        'total_ref_words': tot_words,
        'total_ref_chars': tot_chars,
        'substitutions': tot_subs,
        'deletions': tot_dels,
        'insertions': tot_ins,
        'corpus_wer_pct': round(corpus_wer * 100, 2),
        'corpus_cer_pct': round(corpus_cer * 100, 2),
        'exact_match_pct': round(exact_match_pct, 2),
        'avg_wer_pct': round(avg_wer * 100, 2),
        'median_wer_pct': round(median_wer * 100, 2),
        'p90_wer_pct': round(p90_wer * 100, 2)
    }

hi_metrics = compute_lang_metrics(df_file_level[df_file_level['language'] == 'hi'])
mr_metrics = compute_lang_metrics(df_file_level[df_file_level['language'] == 'mr'])
comb_metrics = compute_lang_metrics(df_file_level)

print("\n--- HINDI METRICS ---", hi_metrics)
print("\n--- MARATHI METRICS ---", mr_metrics)
print("\n--- COMBINED METRICS ---", comb_metrics)

