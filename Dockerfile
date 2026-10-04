# Multi-stage GPU-accelerated Dockerfile for IndicConformer ASR & Call Intelligence
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Configure HuggingFace and PyTorch offline/persistent cache locations
ENV HF_HOME=/app/hf_cache
ENV TORCH_HOME=/app/.cache/torch

# Install system audio libraries, ffmpeg, and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and default configuration
COPY . .

# Create necessary runtime directories
RUN mkdir -p /app/results /app/recordings /app/models /app/hf_cache

# Set default execution command
ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--help"]
