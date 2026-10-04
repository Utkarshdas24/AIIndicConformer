import os
import json
import pandas as pd

base_dir = r"D:\ASR"
csv_path = os.path.join(base_dir, "results", "indicconformer_rtx3050_results.csv")
json_out_path = os.path.join(base_dir, "results", "transcriptions.json")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient='records')
    with open(json_out_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[*] Exported {len(records)} transcripts to: {json_out_path}")
