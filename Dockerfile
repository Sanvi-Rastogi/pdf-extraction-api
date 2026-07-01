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
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install torch first (large download)
RUN pip install --no-cache-dir --timeout=300 torch

# Install marker (depends on torch)
RUN pip install --no-cache-dir --timeout=300 marker-pdf

# Install docling
RUN pip install --no-cache-dir --timeout=300 docling

# Install everything else
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p test_pdfs results
#Fix permissions — important on Linux
# RUN mkdir -p test_pdfs results && \
#     chmod -R 777 test_pdfs results

CMD ["python", "run_extraction.py"]