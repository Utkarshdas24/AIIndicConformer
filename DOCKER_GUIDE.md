# Docker Guide: IndicConformer GPU ASR & LLM Diarization

This repository is fully containerized with NVIDIA GPU acceleration (CUDA 12.4 + PyTorch 2.6.0 + ONNX Runtime CUDA Execution Provider).

---

## 1. Prerequisites

1. **Docker Engine / Docker Desktop** with WSL2 backend.
2. **NVIDIA GPU** (RTX 3050, L4, T4, A100, etc.) with driver version >= 525.
3. **NVIDIA Container Toolkit** installed (enables `--gpus all` inside Docker):
   ```bash
   # Verify NVIDIA GPU passthrough in Docker:
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

---

## 2. Directory Structure for Docker Mounts

```
D:\ASR\
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env                       # Azure OpenAI & HF tokens
├── requirements.txt
├── run_pipeline.py            # Master CLI entrypoint
├── recordings/                # Put your input .wav files here
├── results/                   # Final JSON and MD reports are saved here
├── models/                    # (Optional) Offline IndicConformer weights
└── hf_cache/                  # Hugging Face cache (preserved across runs)
```

---

## 3. Quick Start with Docker Compose (Recommended)

### Step A: Build the Container
```bash
docker compose build
```

### Step B: Transcribe an Audio Call
Place your `.wav` file into the `recordings/` folder (e.g., `recordings/call.wav`), then run:

```bash
docker compose run asr recordings/call.wav
```

To run with options (language, decoder):
```bash
docker compose run asr recordings/call.wav --lang hi --decoder rnnt
```

Outputs will be automatically saved to your local `./results/` directory:
- `results/<stem>_final.json` (Structured turns with speaker, timestamp, transcript, English summary & turn findings)
- `results/<stem>_final.md` (Formatted markdown report)

---

## 4. Running with Standard Docker CLI

If you prefer `docker run`:

### Build:
```bash
docker build -t indic-asr-pipeline:latest .
```

### Run:
```bash
docker run --gpus all --ipc=host \
  --env-file .env \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/recordings:/app/recordings \
  -v $(pwd)/hf_cache:/app/hf_cache \
  indic-asr-pipeline:latest recordings/record-1769162120024.wav --lang hi --decoder rnnt
```

*On Windows PowerShell, replace `$(pwd)` with `${PWD}`.*

---

## 5. Environment Variables (`.env`)

The container expects your `.env` file containing:
```ini
LLM_PROVIDER=azure
LLM_BASE_URL=https://<your-azure-openai-resource>.openai.azure.com
LLM_MODEL=gpt-5.4-nano
LLM_API_KEY=<your-api-key>
LLM_AZURE_API_VERSION=2025-04-01-preview
LLM_USE_MAX_COMPLETION_TOKENS=true
LLM_REASONING_EFFORT=medium
HF_TOKEN=<your-hf-token>
```

---

## 6. Offline Model Execution

If you have downloaded the 2.38 GB IndicConformer weights to `models/indic-conformer-600m`:
- The container mounts `./models:/app/models` and points `INDIC_MODEL_DIR=/app/models/indic-conformer-600m`.
- The pipeline will automatically load weights locally without downloading from Hugging Face.
