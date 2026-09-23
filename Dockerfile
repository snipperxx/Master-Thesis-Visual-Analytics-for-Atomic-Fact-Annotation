# Phase-3 — containerized Flask app. Ollama stays NATIVE on the host (GPU),
# reached via host.docker.internal; this image is CPU-only.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/data/hf_cache \
    OLLAMA_URL=http://host.docker.internal:11434 \
    NEO4J_URI=bolt://neo4j:7687 \
    NEO4J_USER=neo4j

# libgomp1: scikit-learn/scipy runtime; build-essential: any sdist-only deps; curl: debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch BEFORE requirements so sentence-transformers won't pull ~2 GB of CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "-m", "ui.app", "--host", "0.0.0.0", "--port", "5000"]
