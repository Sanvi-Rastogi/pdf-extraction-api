FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libmagic1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 torchvision==0.22.1
RUN pip install --no-cache-dir --timeout=300 docling
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

RUN mkdir -p test_pdfs results && chmod -R 777 test_pdfs results


ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface

ENV HF_HUB_OFFLINE=0

# Download embedding model
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer

SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model downloaded")
EOF


# Download Unstructured hi_res model
RUN python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="unstructuredio/yolo_x_layout"
)

print("Unstructured hi_res model downloaded")
EOF


# Download Docling models
RUN python - <<'EOF'
from pathlib import Path
from docling.utils.model_downloader import download_models

download_models(
    output_dir=Path("/root/.cache/docling/models")
)

print("Docling models downloaded")
EOF


ENV DOCLING_ARTIFACTS_PATH=/root/.cache/docling/models

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY . .

CMD ["python", "run_extraction.py"]