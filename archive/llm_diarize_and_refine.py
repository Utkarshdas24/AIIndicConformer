import os
import sys
import io
import json
from pathlib import Path
from openai import AzureOpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------------- Config ----------------
INPUT_JSON = r"D:\ASR\results\record-1769162120024_diarized.json"
OUT_JSON = r"D:\ASR\results\record-1769162120024_llm_refined_diarized.json"
OUT_MD = r"D:\ASR\results\record-1769162120024_llm_refined_call_report.md"
BATCH_TURNS = 50  # turns per LLM call so the WHOLE call is processed

env = {}
for line in Path(r"D:\ASR\.env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("\"'")

client = AzureOpenAI(
    azure_endpoint=env["LLM_BASE_URL"],
    api_key=env["LLM_API_KEY"],
    api_version=env.get("LLM_AZURE_API_VERSION", "2024-02-01"),
)
MODEL = env["LLM_MODEL"]

GROUNDING_RULES = """STRICT GROUNDING RULES:
- Use ONLY information present in the transcript. Never invent names, numbers, IDs, premiums, ages, or companies.
- The company name must be written exactly as heard. "बजाज लाइफ इंश्योरेंस" = "Bajaj Life Insurance". Do NOT write "Bajaj Allianz".
- Do not add brand names that are not clearly spoken.
- If something is unclear or not stated, use the string "not_stated".
- If ASR text is garbled beyond recovery, keep it and set "uncertain": true. Do not guess."""

def chat_json(system, user, max_tokens=8000):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)

def fmt_ts(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"

# ---------------- Load ASR turns ----------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)
turns = raw["diarized_transcript"]
print(f"[*] Loaded {len(turns)} ASR turns. Model: {MODEL}")

lines = [f"[{t['turn_number']}] {t['speaker']} @{fmt_ts(t['start_seconds'])}: {t['transcript']}" for t in turns]
full_text = "\n".join(lines)

# ---------------- Pass 1: identify roles from the call opening ----------------
role_prompt = f"""{GROUNDING_RULES}

This is a Hindi/Hinglish insurance tele-sales call. The acoustic diarizer produced labels SPEAKER_00 / SPEAKER_01, but these labels are UNRELIABLE (clustering errors are common).
Based on the content, describe how to tell the agent apart from the customer, and give the agent name and company only if spoken.

Return JSON:
{{"agent_name": "...or not_stated", "company_as_spoken": "...or not_stated", "role_cues": "short description"}}

Transcript (first 80 turns):
{chr(10).join(lines[:80])}"""
roles = chat_json("You analyze call transcripts with strict factual grounding.", role_prompt, 2000)
print(f"[*] Roles: {roles}")

# ---------------- Pass 2: refine dialogue in batches (whole call) ----------------
curated = []
for i in range(0, len(lines), BATCH_TURNS):
    batch = lines[i:i + BATCH_TURNS]
    p = f"""{GROUNDING_RULES}

Context: agent = {roles.get('agent_name')}, company = {roles.get('company_as_spoken')}.
Role cues: {roles.get('role_cues')}

For EACH input turn below, decide the true speaker from MEANING (agent explains products / asks underwriting questions; customer asks about their needs / gives personal details). Ignore the SPEAKER_00/01 label when it contradicts the content.
Clean up obvious ASR errors in Hindi, write spoken numbers as digits/₹ only where they are clearly spoken, and translate to English.

Return JSON: {{"turns": [{{"source_turn": <int>, "speaker": "AGENT|CUSTOMER", "timestamp": "MM:SS", "clean_hindi": "...", "english_translation": "...", "uncertain": true|false}}]}}
Keep one output item per input turn (you may merge consecutive turns from the same speaker; then list the first source_turn).

Turns:
{chr(10).join(batch)}"""
    out = chat_json("You refine Hindi ASR transcripts with strict factual grounding.", p)
    curated.extend(out.get("turns", []))
    print(f"[*] Refined turns {i + 1}-{min(i + BATCH_TURNS, len(lines))} -> total {len(curated)}")

# ---------------- Pass 3: extract call facts from the FULL transcript ----------------
facts_prompt = f"""{GROUNDING_RULES}

From the complete call transcript below, extract facts. Every value must be traceable to the text; include the source turn numbers.

Return JSON:
{{
  "call_topic": "...",
  "agent": {{"name": "...", "company": "...", "evidence_turns": [..]}},
  "customer_needs": ["..."],
  "people_to_be_insured": [{{"relation": "...", "age": "...", "income": "...", "health": "...", "evidence_turns": [..]}}],
  "products_discussed": [{{"plan_name_as_spoken": "...", "sum_assured": "...", "riders_or_features": ["..."], "premium_mentioned": "...", "evidence_turns": [..]}}],
  "objections_or_concerns": ["..."],
  "outcome": {{"status": "...", "follow_up": "...", "evidence_turns": [..]}},
  "data_quality_notes": ["parts that were too garbled to interpret"]
}}

Full transcript:
{full_text}"""
facts = chat_json("You extract call facts with strict factual grounding and citations.", facts_prompt, 8000)

result = {
    "file_name": raw.get("file_name"),
    "duration_minutes": raw.get("total_duration_minutes"),
    "llm_model": MODEL,
    "asr_turns_in": len(turns),
    "roles": roles,
    "call_facts": facts,
    "curated_dialogue": curated,
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# ---------------- Markdown report ----------------
md = [f"# LLM-Refined Call Report: `{raw.get('file_name')}`", "",
      f"- Duration: {raw.get('total_duration_minutes')} min",
      f"- LLM: `{MODEL}` (grounded; values not in transcript are marked `not_stated`)",
      f"- Agent: {facts.get('agent', {}).get('name')} | Company: {facts.get('agent', {}).get('company')}",
      f"- Topic: {facts.get('call_topic')}", "",
      "## Call Facts", "```json", json.dumps(facts, indent=2, ensure_ascii=False), "```", "",
      "## Dialogue", "", "| Time | Speaker | Hindi | English | ? |", "|---|---|---|---|---|"]
for t in curated:
    h = str(t.get("clean_hindi", "")).replace("|", "/")
    e = str(t.get("english_translation", "")).replace("|", "/")
    md.append(f"| {t.get('timestamp')} | **{t.get('speaker')}** | {h} | {e} | {'⚠️' if t.get('uncertain') else ''} |")
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"[*] Saved: {OUT_JSON}")
print(f"[*] Saved: {OUT_MD}")
print(f"[*] Agent: {facts.get('agent')}")
