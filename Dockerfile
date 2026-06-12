# Histopathology patch retrieval — GPU-capable image
# Base: CUDA 12.1 runtime. Falls back to CPU if no GPU is present.
# Dataset, model weights, and index are NOT baked in — mount as volumes.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/model_cache \
    TRANSFORMERS_CACHE=/model_cache \
    TORCH_HOME=/model_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libgl1 libglib2.0-0 curl git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code only (no data, no weights)
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml .

# Volume mount points (created so Docker doesn't auto-create them as root-owned)
RUN mkdir -p /data /index_store /model_cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
