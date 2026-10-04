import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df_m = pd.read_csv(r"D:\ASR\dataset\manifest.csv")
print("--- Manifest Head ---")
for idx, row in df_m.head(5).iterrows():
    print(f"{row['file']} [{row['language']}]: {row['reference_transcript']}")

df_r = pd.read_csv(r"D:\ASR\results\indicconformer_rtx3050_results.csv")
print("\n--- Results Head ---")
for idx, row in df_r.head(5).iterrows():
    print(f"FILE: {row['file']} [{row['language']}]")
    print(f"  REF: {row['reference_transcript']}")
    print(f"  HYP: {row['transcript']}\n")
