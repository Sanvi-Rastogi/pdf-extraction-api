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

RUN pip install --no-cache-dir --timeout=300 torch
RUN pip install --no-cache-dir --timeout=300 marker-pdf
RUN pip install --no-cache-dir --timeout=300 docling
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY . .

RUN mkdir -p test_pdfs results && chmod -R 777 test_pdfs results

# Pre-download Docling models during build
RUN python -c "from docling.document_converter import DocumentConverter, PdfFormatOption; from docling.datamodel.pipeline_options import PdfPipelineOptions; from docling.datamodel.base_models import InputFormat; opts = PdfPipelineOptions(); opts.do_ocr = False; opts.do_table_structure = True; DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}); print('Docling models ready.')"

# Pre-download Unstructured models during build
RUN python -c "from unstructured.partition.pdf import partition_pdf; print('Unstructured models ready.')" || echo "Skipped"

CMD ["python", "run_extraction.py"]