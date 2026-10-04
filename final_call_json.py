"""Final call JSON: exact ASR text split into AGENT / CUSTOMER turns, plus timestamped findings.
The LLM only returns speaker-change word indices and finding references; all transcript text is copied
verbatim from the ASR output by this script."""
import os, sys, io, json, argparse
from pathlib import Path
from collections import Counter
from openai import AzureOpenAI
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("stem", nargs="?", default="record-1769162120024")
ap.add_argument("--decoder", default="rnnt", choices=["ctc", "rnnt"])
args = ap.parse_args()
IN = rf"D:\ASR\results\{args.stem}_asr_segments.json"
OUT_JSON = rf"D:\ASR\results\{args.stem}_final.json"
OUT_MD = rf"D:\ASR\results\{args.stem}_final.md"
BATCH = 8          # small batches -> LLM splits speakers inside segments
LONG_TURN = 45     # pieces longer than this (words) get a focused re-split pass

env = dict(os.environ)
env_path = Path(".env") if Path(".env").is_file() else Path(r"D:\ASR\.env")
if env_path.is_file():
    for l in env_path.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l and not l.startswith("#") and "=" in l:
            k, v = l.split("=", 1); env.setdefault(k.strip(), v.strip().strip("\"'"))
client = AzureOpenAI(azure_endpoint=env["LLM_BASE_URL"], api_key=env["LLM_API_KEY"],
                     api_version=env.get("LLM_AZURE_API_VERSION", "2024-02-01"))
MODEL = env["LLM_MODEL"]

def ask(system, user):
    extra = {"reasoning_effort": env["LLM_REASONING_EFFORT"]} if env.get("LLM_REASONING_EFFORT") else {}
    r = client.chat.completions.create(model=MODEL, max_completion_tokens=16000, **extra,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content)

mm = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"

data = json.load(open(IN, encoding="utf-8"))
segs = []
for s in data["segments"]:
    text = s.get("transcript") or s.get(f"transcript_{args.decoder}") or ""
    if text.strip():
        segs.append({**s, "text": text.strip(), "words": text.split()})
print(f"[*] {len(segs)} non-empty segments | decoder={args.decoder} | llm={MODEL}")

ROLE_GUIDE = """Roles in an insurance sales call:
- AGENT: insurance company representative. Introduces self/company, explains plans/benefits/exclusions,
  asks mandatory questions (income, DOB, tobacco, medical history), quotes premiums, confirms phone number, schedules follow-up.
- CUSTOMER: person enquiring. Describes own/family needs, answers personal questions, asks about benefits, raises doubts.
Hindi verb gender (e.g. "रही हूँ / लेती हूँ" vs "रहा हूँ / लेता हूँ") is a strong cue: once you know each speaker's gender from context, use it consistently.
Speakers often alternate inside one segment (question then short answer like "हाँ जी", "ओके", numbers)."""

# ---------- Pass 0: speaker profile (gender / name cues) ----------
full_txt = "\n".join(f"[{s['timestamp']}] {s['text']}" for s in segs)
prof = ask("You analyse Hindi call-center ASR text. Answer only from the text.",
f"""{ROLE_GUIDE}
From this whole call, infer: the agent's gender (from verb forms she/he uses about herself/himself),
the customer's gender, the agent's name if stated, and typical phrases each side uses.
Return JSON: {{"agent_gender": "female|male|unknown", "customer_gender": "female|male|unknown",
 "agent_name": "string or null", "agent_cues": ["..."], "customer_cues": ["..."]}}

Call:
{full_txt}""")
PROFILE = (f"Speaker profile for THIS call: agent gender={prof.get('agent_gender')}, customer gender={prof.get('customer_gender')}, "
           f"agent name={prof.get('agent_name')}. Agent cues: {prof.get('agent_cues')}. Customer cues: {prof.get('customer_cues')}.")
print("[*] " + PROFILE[:300])

SPLIT_RULES = """Rules:
- Each word is prefixed with its index. Return where each speaker's run starts.
- A segment is NOT one speaker by default. Phone calls alternate fast: split at EVERY change, including
  short replies ("haan", "ji", "ok", "nahi", a number, a name) between agent sentences.
- Typical pattern: agent asks a question -> customer answers -> agent acknowledges/continues. Each is a new run.
- Questions about the customer's income, age, health, tobacco, family, phone number are asked by AGENT; the answers are CUSTOMER.
- Plan explanation, premium quotes, benefits, disclaimers are AGENT. Doubts/"what if" questions about the plan are usually CUSTOMER.
- First run starts at word 0; start indices strictly increasing. speaker is AGENT, CUSTOMER or UNCLEAR (only if impossible).
- Never output transcript text."""

def label(batch, ctx):
    body = "\n".join(f"SEG {s['segment_id']} [{s['timestamp']}]: " +
                     " ".join(f"{j}:{w}" for j, w in enumerate(s["words"])) for s in batch)
    res = ask("You perform content-based speaker diarization of Hindi call-center ASR text. You never rewrite text.",
f"""{ROLE_GUIDE}
{PROFILE}

{SPLIT_RULES}
{('Preceding conversation (already labelled, for context only):' + chr(10) + ctx) if ctx else ''}

Return JSON: {{"segments": [{{"segment_id": int, "runs": [{{"start_word": int, "speaker": "AGENT|CUSTOMER|UNCLEAR"}}]}}]}}

Segments to label:
{body}""")
    return {x["segment_id"]: x.get("runs", []) for x in res.get("segments", []) if "segment_id" in x}

def runs_text(s, runs):
    rs = sorted(runs, key=lambda r: int(r.get("start_word", 0)))
    out = []
    for i, r in enumerate(rs):
        k = int(r.get("start_word", 0)); e = int(rs[i + 1].get("start_word", 0)) if i + 1 < len(rs) else len(s["words"])
        out.append(f"{r.get('speaker')}: {' '.join(s['words'][k:e])}")
    return out

# ---------- Pass 1: speaker change points, small batches ----------
splits, ctx_lines = {}, []
for i in range(0, len(segs), BATCH):
    b = segs[i:i + BATCH]
    splits.update(label(b, "\n".join(ctx_lines[-8:])))
    for s in b:
        ctx_lines += runs_text(s, splits.get(s["segment_id"], [{"start_word": 0, "speaker": "UNCLEAR"}]))
    print(f"[*] speaker split: segments {b[0]['segment_id']}-{b[-1]['segment_id']}")

# ---------- Pass 1b: re-split segments that still have a very long single-speaker run ----------
by_seg = {s["segment_id"]: s for s in segs}
def longest_run(s):
    rs = sorted(int(r.get("start_word", 0)) for r in splits.get(s["segment_id"], [])) or [0]
    b = rs + [len(s["words"])]
    return max(b[i + 1] - b[i] for i in range(len(rs)))
redo = [s for s in segs if longest_run(s) > LONG_TURN]
print(f"[*] re-split pass on {len(redo)} segments with runs > {LONG_TURN} words")
for s in redo:
    idx = segs.index(s)
    ctx = []
    for p in segs[max(0, idx - 3):idx]:
        ctx += runs_text(p, splits.get(p["segment_id"], []))
    new = label([s], "\n".join(ctx))
    if new.get(s["segment_id"]):
        splits[s["segment_id"]] = new[s["segment_id"]]

# ---------- Build exact-text pieces ----------
pieces, fixed = [], 0
for s in segs:
    n = len(s["words"])
    runs = splits.get(s["segment_id"]) or [{"start_word": 0, "speaker": "UNCLEAR"}]
    clean = []
    for r in sorted(runs, key=lambda r: int(r.get("start_word", 0))):
        k = int(r.get("start_word", 0))
        if 0 <= k < n and (not clean or k > clean[-1][0]):
            clean.append((k, r.get("speaker", "UNCLEAR")))
    if not clean or clean[0][0] != 0:
        clean.insert(0, (0, clean[0][1] if clean else "UNCLEAR")); fixed += 1
    dur, chars = s["end_sec"] - s["start_sec"], max(1, len(s["text"]))
    for idx, (k, spk) in enumerate(clean):
        end_k = clean[idx + 1][0] if idx + 1 < len(clean) else n
        txt = " ".join(s["words"][k:end_k])
        pre = len(" ".join(s["words"][:k]))
        t0 = s["start_sec"] + dur * pre / chars
        t1 = s["start_sec"] + dur * (pre + len(txt)) / chars
        pieces.append({"speaker": spk, "start_sec": round(t0, 2), "end_sec": round(t1, 2),
                       "text": txt, "segment_id": s["segment_id"], "approx": len(clean) > 1})

# ---------- Merge consecutive same-speaker pieces into turns ----------
turns = []
for p in pieces:
    if turns and turns[-1]["speaker"] == p["speaker"] and p["start_sec"] - turns[-1]["end_sec"] < 2.0:
        t = turns[-1]; t["transcript"] += " " + p["text"]; t["end_sec"] = p["end_sec"]
        t["segment_ids"] = sorted(set(t["segment_ids"] + [p["segment_id"]])); t["timestamp_approx"] |= p["approx"]
    else:
        turns.append({"speaker": p["speaker"], "start_sec": p["start_sec"], "end_sec": p["end_sec"],
                      "transcript": p["text"], "segment_ids": [p["segment_id"]], "timestamp_approx": p["approx"]})
for i, t in enumerate(turns, 1):
    t["turn_id"] = i
    t["timestamp"] = f"{mm(t['start_sec'])}-{mm(t['end_sec'])}"
turns = [{k: t[k] for k in ["turn_id", "speaker", "timestamp", "start_sec", "end_sec", "transcript", "timestamp_approx", "segment_ids"]} for t in turns]
by_turn = {t["turn_id"]: t for t in turns}
print(f"[*] {len(turns)} turns | split-point fixes: {fixed}")

# ---------- Pass 2: findings over the whole conversation ----------
conv = "\n".join(f"T{t['turn_id']} [{t['timestamp']}] {t['speaker']}: {t['transcript']}" for t in turns)
fr = ask("You extract business findings from insurance call transcripts with strict grounding.",
f"""Rules: use only what is in the transcript; never invent names, numbers or companies.
Company as spoken: "बजाज लाइफ इंश्योरेंस" = Bajaj Life Insurance (never "Allianz"). The ASR is noisy; mark garbled evidence as low confidence.
Do not output transcript text; cite turn ids.

Return JSON: {{"summary": "3-4 sentence English summary",
 "findings": [{{"turn_ids": [int], "category": "product|benefit|exclusion|premium_amount|sum_assured|personal_info|health_disclosure|objection|compliance_disclosure|follow_up|pii",
   "speaker": "AGENT|CUSTOMER", "finding": "one grounded English sentence", "confidence": "high|medium|low"}}]}}

Conversation:
{conv}""")

findings = []
for f in fr.get("findings", []):
    ids = sorted(i for i in f.get("turn_ids", []) if i in by_turn)
    if not ids: continue
    findings.append({"timestamp": f"{mm(by_turn[ids[0]]['start_sec'])}-{mm(by_turn[ids[-1]]['end_sec'])}",
                     "start_sec": by_turn[ids[0]]["start_sec"], "category": f.get("category"),
                     "speaker": f.get("speaker"), "finding": f.get("finding"), "confidence": f.get("confidence"),
                     "turn_ids": ids,
                     "exact_transcript": [f"[{by_turn[i]['timestamp']}] {by_turn[i]['speaker']}: {by_turn[i]['transcript']}" for i in ids]})
findings.sort(key=lambda x: x["start_sec"])

def talk(spk):
    ts = [t for t in turns if t["speaker"] == spk]
    return {"turns": len(ts), "talk_time_sec": round(sum(t["end_sec"] - t["start_sec"] for t in ts), 1)}

final = {
    "file_name": data["file_name"], "duration_sec": data["duration_sec"], "duration": mm(data["duration_sec"]),
    "asr_model": data["model"], "asr_decoder": args.decoder, "language": data.get("language", "hi"), "llm_model": MODEL,
    "notes": ["'transcript' fields are exact ASR output, unedited.",
              "Speaker labels are LLM-inferred from content. Where a segment was split between speakers, timestamps are proportional estimates (timestamp_approx=true).",
              "Findings are LLM-generated; check 'exact_transcript' and the audio before acting on numbers."],
    "summary": fr.get("summary"),
    "speakers": {"AGENT": talk("AGENT"), "CUSTOMER": talk("CUSTOMER"), "UNCLEAR": talk("UNCLEAR")},
    "conversation": turns,
    "agent_transcript": [{"turn_id": t["turn_id"], "timestamp": t["timestamp"], "transcript": t["transcript"]} for t in turns if t["speaker"] == "AGENT"],
    "customer_transcript": [{"turn_id": t["turn_id"], "timestamp": t["timestamp"], "transcript": t["transcript"]} for t in turns if t["speaker"] == "CUSTOMER"],
    "findings": findings,
}
json.dump(final, open(OUT_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

md = [f"# {data['file_name']}", "", f"Duration {final['duration']} | ASR `{data['model']}` ({args.decoder}) | LLM `{MODEL}`", "",
      "> Transcript text is exact ASR output. Speakers and findings are LLM-inferred. `~` = estimated timestamp.", "",
      "## Summary", "", str(final["summary"]), "",
      "## Findings", "", "| Time | Category | Speaker | Finding | Conf. |", "|---|---|---|---|---|"]
md += [f"| {f['timestamp']} | {f['category']} | {f['speaker']} | {f['finding']} | {f['confidence']} |" for f in findings]
md += ["", "## Conversation", ""]
md += [f"**[{t['timestamp']}{'~' if t['timestamp_approx'] else ''}] {t['speaker']}:** {t['transcript']}  " for t in turns]
open(OUT_MD, "w", encoding="utf-8").write("\n".join(md))

print(f"[*] Speakers: {dict(Counter(t['speaker'] for t in turns))} | findings: {len(findings)}")
print(f"[*] Saved {OUT_JSON}\n[*] Saved {OUT_MD}")
