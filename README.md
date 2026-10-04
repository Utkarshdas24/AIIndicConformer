# IndicConformer-600M ASR & LLM Diarization Pipeline

An enterprise-ready, GPU-accelerated Automatic Speech Recognition (ASR), Speaker Diarization, and Call Intelligence pipeline designed for Indian multilingual call recordings.

Built around **AI4Bharat IndicConformer-600M Multilingual** and **Azure OpenAI (gpt-5.4-nano)**, optimized for high accuracy on low-bitrate (8 kHz) telephony audio.

---

## Key Capabilities

1. **GPU-Accelerated ASR (IndicConformer-600M)**:
   - **RNNT Decoder**: Empirically verified lowest Word Error Rate (**WER 8.73%** in Hindi vs 9.32% CTC).
   - **Adaptive Segmentation & Anti-Drop VAD**: Dynamically caps speech segments to $\le 8.0$ seconds and re-splits any under-transcribed audio to ensure zero dropped text on phone recordings.
   - **3.9× Real-Time Throughput**: Transcribes a full 27-minute call in **7.0 minutes** on a laptop GPU (NVIDIA RTX 3050 6GB).

2. **Grounded LLM Speaker Diarization**:
   - Accurately distinguishes **AGENT** vs **CUSTOMER** without hallucinating or rewriting transcript text.
   - Leverages full-call speaker profiling (gender cues, honorifics, company introduction) before labeling micro-turn switches.

3. **Per-Turn English Summaries & Business Findings**:
   - Every conversational turn contains:
     - Exact, unedited ASR transcript.
     - 1–2 sentence English summary / gist.
     - Structured business findings (categories: `product`, `health_disclosure`, `personal_info`, `compliance_disclosure`, `follow_up`, `objection`, etc.) with confidence scores.

4. **Transparent Cost & Hardware Metrics**:
   - Built-in telemetry tracking prompt tokens, completion tokens, estimated LLM cost in USD/INR (~₹1.60 INR per 27-minute call), peak GPU VRAM, and CPU utilization.

5. **Fully Dockerized**:
   - Multi-stage Docker container with NVIDIA GPU passthrough and volume mounts for zero-setup deployment.

---

## Benchmark Results (Hindi Kathbath Dataset)

| Decoder | Hindi WER | Hindi CER | Speedup Factor | Selected Status |
|---|---|---|---|---|
| **RNNT (ONNX GPU)** | **8.73%** | **3.79%** | **3.90× real-time** | **CHOSEN (Best Accuracy)** |
| **CTC (PyTorch GPU)** | 9.32% | 4.01% | 4.43× real-time | Alternative (Faster, lower accuracy) |

---

## Project Structure

```text
ASR/
├── Dockerfile                   # GPU-accelerated container definition
├── docker-compose.yml           # One-command Docker deployment
├── .dockerignore                # Excludes large binaries & virtual environments
├── .gitignore                   # Excludes models (>2.4 GB), .env secrets, audio & reports
├── .env.example                 # Configuration template for API credentials
├── requirements.txt             # Python dependencies
│
├── run_pipeline.py              # Master CLI entrypoint (Runs Stages 1, 2 & 3)
├── asr_segments.py              # Stage 1: Energy VAD + GPU IndicConformer RNNT decoding
├── final_call_json.py           # Stage 2: LLM Speaker Profiling & Turn Alignment
├── enrich_final_json.py         # Stage 3: Per-turn English summaries, findings & metrics
├── transcribe.py                # IndicConformer model loader & audio preprocessor
├── compare_decoders.py          # Benchmark tool comparing CTC vs RNNT WER/CER
│
├── recordings/                  # Input audio folder (.gitkeep tracked, audio ignored)
├── results/                     # Pipeline output folder (.gitkeep tracked, outputs ignored)
├── archive/                     # Archived benchmark and prototype scripts
└── models/                      # (Optional) Offline model weights directory (ignored by git)
```

---

## Quickstart Guide

### 1. Configure Credentials
Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
LLM_PROVIDER=azure
LLM_BASE_URL=https://<your-resource-name>.openai.azure.com
LLM_MODEL=gpt-5.4-nano
LLM_API_KEY=your_azure_openai_key_here
LLM_AZURE_API_VERSION=2025-04-01-preview
LLM_USE_MAX_COMPLETION_TOKENS=true
LLM_REASONING_EFFORT=medium
```

---

### 2. Run with Docker (Recommended)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. **Build container**:
   ```bash
   docker compose build
   ```

2. **Transcribe an audio call**:
   Place your `.wav` file in `recordings/`, then run:
   ```bash
   docker compose run asr recordings/your_call.wav
   ```

Outputs will be saved in `results/<filename>_final.json` and `results/<filename>_final.md`.

---

### 3. Run Locally (Python 3.10–3.13)

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
   *Note for GPU*: Ensure `onnxruntime-gpu==1.22.0` is installed matching your CUDA version.

3. **Run the master pipeline**:
   ```bash
   python run_pipeline.py recordings/record-1769162120024.wav --lang hi --decoder rnnt
   ```

   **CLI Flags**:
   - `audio`: Path to input audio file.
   - `--lang`: Indic language code (e.g. `hi`, `mr`, `bn`, `gu`, `ta`, `te`). Default: `hi`.
   - `--decoder`: `rnnt` (recommended for accuracy) or `ctc`. Default: `rnnt`.
   - `--out-dir`: Output directory. Default: `results`.
   - `--skip-asr`: Skip ASR if segments JSON already exists (re-runs LLM diarization only).

---

## Output Format Specification

The output `results/<filename>_final.json` provides:

```json
{
  "file_name": "record-1769162120024.wav",
  "duration": "27:07",
  "asr_model": "sunilmahendrakar/indic-conformer-600m-multilingual",
  "asr_decoder": "rnnt",
  "summary": "3-4 sentence comprehensive English summary of the call...",
  "speakers": {
    "AGENT": { "turns": 89, "talk_time_sec": 1091.2 },
    "CUSTOMER": { "turns": 87, "talk_time_sec": 440.9 }
  },
  "conversation": [
    {
      "turn_id": 6,
      "speaker": "AGENT",
      "timestamp": "00:42-01:16",
      "start_sec": 42.83,
      "end_sec": 76.79,
      "transcript": "तो इस बारे में तो पहले तो आप सरल के बारे में बता दीजिए सरल भीमा...",
      "summary": "Agent acknowledges customer's salaried status and confirms availability of both Saral Jeevan Bima and Bajaj eTouch, highlighting a specialized plan for diabetic individuals. She places the call on a 1-minute hold to transfer the customer to a senior specialist.",
      "findings": [
        {
          "category": "product",
          "finding": "Agent indicates two plan options: one specifically for diabetics and another regular option under Bajaj Life Insurance e-Touch.",
          "confidence": "high"
        },
        {
          "category": "compliance_disclosure",
          "finding": "Agent requests the customer to hold while transferring the call to a senior representative.",
          "confidence": "high"
        }
      ],
      "timestamp_approx": false,
      "segment_ids": [10, 11, 12, 13, 14, 15]
    }
  ],
  "agent_transcript": [ ... ],
  "customer_transcript": [ ... ],
  "findings": [ ... ],
  "performance_and_cost_report": {
    "execution_time": {
      "audio_duration_min": 27.13,
      "asr_transcription_time_min": 7.0,
      "llm_enrichment_time_min": 2.46,
      "total_pipeline_time_min": 17.98
    },
    "llm_usage_and_cost": {
      "model": "gpt-5.4-nano",
      "total_tokens": 58533,
      "estimated_total_pipeline_cost_usd": 0.0187,
      "estimated_total_pipeline_cost_inr": 1.63
    },
    "hardware_utilization": {
      "gpu": {
        "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "vram_allocated_peak_mb": 3480
      }
    }
  }
}
```

---

## Git Push Preparation

Before pushing to GitHub:
1. Verify no secrets or model weights are staged:
   ```bash
   git status
   ```
2. Add files and make your initial commit:
   ```bash
   git add .
   git commit -m "feat: complete IndicConformer GPU ASR, LLM Diarization & Docker pipeline"
   ```
3. Push to your remote repository:
   ```bash
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```
