from pathlib import Path
from openai import AzureOpenAI

env_path = Path(r"D:\ASR\.env")
vals = {}
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip().strip("\"'")

client = AzureOpenAI(
    azure_endpoint=vals["LLM_BASE_URL"],
    api_key=vals["LLM_API_KEY"],
    api_version=vals.get("LLM_AZURE_API_VERSION", "2024-02-01")
)

try:
    resp = client.chat.completions.create(
        model=vals["LLM_MODEL"],
        messages=[{"role": "user", "content": "Hello! Reply with 'OK - LLM Connected Successfully!'"}],
        max_completion_tokens=50
    )
    print("SUCCESS! Response:", resp.choices[0].message.content)
except Exception as e:
    print("Connection error:", e)
