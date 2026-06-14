# Histopathology patch retrieval — GPU-capable image
# Base: CUDA 12.9 runtime on Ubuntu 24.04. Falls back to CPU if no GPU is present.
# Dataset, model weights, and index are NOT baked in — mount as volumes.

FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/model_cache \
    TRANSFORMERS_CACHE=/model_cache \
    TORCH_HOME=/model_cache \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_INDEX_STRATEGY=unsafe-best-match

# Copy uv binary from the official image (no pip needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev \
        libgl1 libglib2.0-0 curl git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Copy application code only (no data, no weights)
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml .

# Volume mount points (created so Docker doesn't auto-create them as root-owned)
RUN mkdir -p /data /index_store /model_cache

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
