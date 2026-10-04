import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv(r"D:\ASR\results\hindi_marathi_file_level_results.csv")
hi_df = df[df['language'] == 'hi'].sort_values(by='wer')

print("=== TOP 10 BEST HINDI SAMPLES ===")
for idx, r in hi_df.head(10).iterrows():
    print(f"File: {r['file']} | WER: {r['wer']*100:.1f}% | CER: {r['cer']*100:.1f}%")
    print(f"  REF: {r['raw_reference']}")
    print(f"  ASR: {r['raw_prediction']}\n")

print("=== TOP 10 WORST HINDI SAMPLES ===")
for idx, r in hi_df.tail(10).iloc[::-1].iterrows():
    print(f"File: {r['file']} | WER: {r['wer']*100:.1f}% | CER: {r['cer']*100:.1f}%")
    print(f"  REF: {r['raw_reference']}")
    print(f"  ASR: {r['raw_prediction']}\n")
