import sys
import io
import os
import re
import unicodedata
import pandas as pd
import numpy as np
import jiwer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def normalize_indic(text):
    if not isinstance(text, str):
        return ""
    # Unicode NFKC
    text = unicodedata.normalize('NFKC', text)
    # Remove Hindi Purna Viram and standard punctuation
    text = re.sub(r'[\u0964\u0965\.\?\!\,\-\:\;\–\—\"\'\(\)\[\]\{\}\/\\]', ' ', text)
    # Optional nukta normalization for Indic ASR fairness: remove nukta sign \u093c
    text = text.replace('\u093c', '')
    # Lowercase Latin if present
    text = text.lower()
    # Normalize extra whitespace
    words = text.split()
    return " ".join(words)

df_results = pd.read_csv(r"D:\ASR\results\indicconformer_rtx3050_results.csv")
df_filtered = df_results[df_results['language'].isin(['hi', 'mr'])].copy()

print(f"Total filtered files: {len(df_filtered)} (Hindi: {len(df_filtered[df_filtered['language']=='hi'])}, Marathi: {len(df_filtered[df_filtered['language']=='mr'])})")

rows = []
for idx, row in df_filtered.iterrows():
    raw_ref = str(row.get('reference_transcript', ''))
    raw_hyp = str(row.get('transcript', ''))
    
    norm_ref = normalize_indic(raw_ref)
    norm_hyp = normalize_indic(raw_hyp)
    
    if len(norm_ref) == 0:
        continue
        
    res = jiwer.process_words(norm_ref, norm_hyp)
    cer_res = jiwer.process_characters(norm_ref, norm_hyp)
    
    ref_words = len(norm_ref.split())
    ref_chars = len(norm_ref)
    
    rows.append({
        'file': row['file'],
        'language': row['language'],
        'raw_reference': raw_ref,
        'raw_prediction': raw_hyp,
        'normalized_reference': norm_ref,
        'normalized_prediction': norm_hyp,
        'audio_duration_seconds': row.get('audio_duration_seconds', 0.0),
        'processing_time_seconds': row.get('processing_time_seconds', 0.0),
        'speed_x': row.get('speed_x', 0.0),
        'wer': res.wer,
        'cer': cer_res.cer,
        'substitutions': res.substitutions,
        'deletions': res.deletions,
        'insertions': res.insertions,
        'hits': res.hits,
        'ref_words': ref_words,
        'ref_chars': ref_chars,
        'exact_match': (norm_ref == norm_hyp)
    })

df_eval = pd.DataFrame(rows)

for lang in ['hi', 'mr', 'COMBINED']:
    if lang == 'COMBINED':
        sub = df_eval
    else:
        sub = df_eval[df_eval['language'] == lang]
        
    tot_words = sub['ref_words'].sum()
    tot_chars = sub['ref_chars'].sum()
    tot_subs = sub['substitutions'].sum()
    tot_dels = sub['deletions'].sum()
    tot_ins = sub['insertions'].sum()
    
    corpus_wer = (tot_subs + tot_dels + tot_ins) / tot_words if tot_words > 0 else 0
    exact_pct = (sub['exact_match'].sum() / len(sub)) * 100 if len(sub) > 0 else 0
    
    mean_wer = sub['wer'].mean()
    median_wer = sub['wer'].median()
    p90_wer = np.percentile(sub['wer'], 90)
    
    print(f"\n=== {lang} METRICS ===")
    print(f"Files Evaluated       : {len(sub)}")
    print(f"Total Ref Words       : {tot_words}")
    print(f"Total Ref Chars       : {tot_chars}")
    print(f"Word Substitutions    : {tot_subs}")
    print(f"Word Deletions        : {tot_dels}")
    print(f"Word Insertions       : {tot_ins}")
    print(f"Corpus WER            : {corpus_wer*100:.2f}%")
    print(f"Average File WER      : {mean_wer*100:.2f}%")
    print(f"Median File WER       : {median_wer*100:.2f}%")
    print(f"P90 File WER          : {p90_wer*100:.2f}%")
    print(f"Exact Match %         : {exact_pct:.2f}%")
