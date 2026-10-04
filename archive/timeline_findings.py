"""Stage 2: speaker roles + timestamped findings via LLM.
The LLM NEVER returns transcript text. Exact ASR text is copied from stage 1 by segment_id."""
import os, sys, io, json
from pathlib import Path
from openai import AzureOpenAI
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

stem = sys.argv[1] if len(sys.argv) > 1 else "record-1769162120024"
IN = rf"D:\ASR\results\{stem}_asr_segments.json"
OUT_JSON = rf"D:\ASR\results\{stem}_timeline.json"
OUT_MD = rf"D:\ASR\results\{stem}_timeline.md"
BATCH = 40

env = {}
for l in Path(r"D:\ASR\.env").read_text(encoding="utf-8").splitlines():
    l = l.strip()
    if l and not l.startswith("#") and "=" in l:
        k, v = l.split("=", 1); env[k.strip()] = v.strip().strip("\"'")
client = AzureOpenAI(azure_endpoint=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                     api_version=env.get("LLM_AZURE_API_VERSION", "2024-02-01"))
MODEL = env["LLM_MODEL"]

def ask(system, user):
    r = client.chat.completions.create(model=MODEL, max_completion_tokens=8000,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)

data = json.load(open(IN, encoding="utf-8"))
segs = data["segments"]
by_id = {s["segment_id"]: s for s in segs}
print(f"[*] {len(segs)} segments, model {MODEL}")

RULES = """Rules:
- Use only what is in the transcript. Never invent names, numbers, companies or facts.
- Company as spoken is "बजाज लाइफ इंश्योरेंस" (Bajaj Life Insurance). Never write "Allianz".
- Do NOT rewrite or return transcript text. Refer to segments only by segment_id.
- Speaker: AGENT (explains products, asks underwriting questions, quotes premium), CUSTOMER (states needs, gives personal/family details, asks questions), or MIXED (both speak in one segment), or UNCLEAR.
- The ASR is noisy 8 kHz call audio. If a segment is too garbled to judge, use UNCLEAR."""

speaker, findings = {}, []
for i in range(0, len(segs), BATCH):
    b = segs[i:i + BATCH]
    ctx = "\n".join(f"{s['segment_id']} [{s['timestamp']}] CTC: {s['transcript_ctc']} || RNNT: {s['transcript_rnnt']}" for s in b)
    res = ask("You label speakers and extract findings from Hindi insurance call ASR, with strict grounding.",
f"""{RULES}

Return JSON:
{{"speakers": [{{"segment_id": int, "speaker": "AGENT|CUSTOMER|MIXED|UNCLEAR"}}],
  "findings": [{{"segment_ids": [int], "category": "product|benefit|exclusion|premium_amount|sum_assured|personal_info|health_disclosure|objection|compliance_disclosure|follow_up|pii",
                 "finding_en": "one short English sentence, grounded", "confidence": "high|medium|low"}}]}}
Only create findings that matter for a business reviewer. Low confidence if the ASR is garbled.

Segments:
{ctx}""")
    for x in res.get("speakers", []): speaker[x["segment_id"]] = x["speaker"]
    findings += res.get("findings", [])
    print(f"[*] segments {b[0]['segment_id']}-{b[-1]['segment_id']}: findings so far {len(findings)}")

# Attach exact transcript quotes + timestamps to findings (copied, not LLM-generated)
clean_findings = []
for f in findings:
    ids = [i for i in f.get("segment_ids", []) if i in by_id]
    if not ids: continue
    s0, s1 = by_id[min(ids)], by_id[max(ids)]
    clean_findings.append({
        "timestamp": f"{s0['timestamp'].split('-')[0]}-{s1['timestamp'].split('-')[1]}",
        "start_sec": s0["start_sec"], "category": f.get("category"), "finding": f.get("finding_en"),
        "confidence": f.get("confidence"), "segment_ids": ids,
        "exact_transcript": [by_id[i]["transcript_ctc"] for i in sorted(ids)]})
clean_findings.sort(key=lambda x: x["start_sec"])

timeline = [{**s, "speaker": speaker.get(s["segment_id"], "UNCLEAR")} for s in segs]
json.dump({"file_name": data["file_name"], "duration_sec": data["duration_sec"], "asr_model": data["model"],
           "llm_model": MODEL, "note": "transcript_ctc/transcript_rnnt are exact ASR output. Speaker labels and findings are LLM-inferred.",
           "findings": clean_findings, "timeline": timeline},
          open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

d = data["duration_sec"]
md = [f"# Call Timeline: `{data['file_name']}`", "",
      f"- Duration: {int(d)//60:02d}:{int(d)%60:02d} | ASR: `{data['model']}` | LLM: `{MODEL}`",
      "- **Transcript text is exact ASR output (unedited).** Speaker labels and findings are LLM-inferred.", "",
      "## Key Findings", "", "| Time | Category | Finding | Exact transcript | Conf. |", "|---|---|---|---|---|"]
for f in clean_findings:
    q = " / ".join(f["exact_transcript"]).replace("|", "/")
    md.append(f"| {f['timestamp']} | {f['category']} | {f['finding']} | {q} | {f['confidence']} |")
md += ["", "## Full Transcript", "", "| # | Time | Speaker | Transcript (CTC) | Alt (RNNT) |", "|---|---|---|---|---|"]
for t in timeline:
    md.append(f"| {t['segment_id']} | {t['timestamp']} | **{t['speaker']}** | {t['transcript_ctc'].replace('|','/')} | {t['transcript_rnnt'].replace('|','/')} |")
open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))

from collections import Counter
print(f"[*] Speakers: {dict(Counter(t['speaker'] for t in timeline))}")
print(f"[*] Findings: {len(clean_findings)}")
print(f"[*] Saved {OUT_JSON}\n[*] Saved {OUT_MD}")
