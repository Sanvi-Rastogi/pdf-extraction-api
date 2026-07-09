# PDF Extraction & Semantic Chunking Benchmark Tool

A Dockerized CLI tool that benchmarks 7 different PDF extraction methods
and implements RAGFlow-inspired semantic chunking strategies for academic
document processing pipelines.

Built as part of a research internship project to identify the best PDF
loader and chunking strategy for academic document RAG (Retrieval-Augmented
Generation) pipelines.

---

## Project Structure

```
pdf-extraction-api/
├── run_extraction.py           # Main CLI script
├── extractors/
│   ├── __init__.py
│   ├── pypdf_extractor.py      # PyPDF baseline loader
│   ├── pdfplumber_extractor.py # Table-aware loader
│   ├── pymupdf_extractor.py    # Fast loader with image detection
│   ├── unstructured_extractor.py # Element classification loader
│   ├── docling_extractor.py    # IBM Docling structured loader
│   └── marker_extractor.py     # Datalab Marker loader
├── chunkers/
│   ├── __init__.py
│   ├── naive_chunker.py        # Sentence-boundary aware chunking
│   ├── layout_chunker.py       # Element-type aware chunking
│   ├── table_chunker.py        # Table-preserving chunking
│   └── semantic_chunker.py     # Embedding-based semantic chunking
├── utils/
│   ├── __init__.py
│   └── memory_tracker.py       # RAM and GPU memory tracking
├── Dockerfile                  # Mac build (no Marker pre-download)
├── Dockerfile.linux            # Linux build (all models pre-downloaded)
├── docker-compose.yml          # Linux configuration
├── docker-compose.mac.yml      # Mac configuration
├── requirements.txt
├── test_pdfs/                  # Place your PDFs here
└── results/                    # All output saved here automatically
```

---

## Loaders Compared

| Loader              | Tables | Images | Headers | Equations | Speed          |
| ------------------- | ------ | ------ | ------- | --------- | -------------- |
| PyPDF (Baseline)    | ❌     | ❌     | ❌      | ❌        | Fastest (0.2s) |
| pdfplumber          | ✅     | ❌     | ❌      | ❌        | Fast (0.4s)    |
| PyMuPDF (fitz)      | ❌     | ✅     | ❌      | ❌        | Fastest (0.2s) |
| Unstructured (fast) | ❌     | ❌     | ✅      | ❌        | Medium (5-10s) |
| Unstructured OCR    | ✅     | ✅     | ✅      | ❌        | Slow (~90s)    |
| Docling (IBM)       | ✅     | ❌     | ✅      | ❌        | Slow (~100s)   |
| Marker (Datalab)    | ✅     | ✅     | ✅      | ✅        | Slow (~120s)   |

---

## Chunking Strategies

Inspired by RAGFlow's deepdoc pipeline architecture.

**Naive Chunker** — splits text at sentence boundaries using regex, never cuts mid-sentence. Sentences accumulate until token limit is reached with configurable sentence-level overlap between chunks.

**Layout Chunker** — uses element-type tags from Unstructured loader to apply RAGFlow's core layout rules: Title elements always trigger new chunk boundaries, Table elements are atomic chunks that are never split, Footer and Header elements are discarded as noise.

**Table Chunker** — extracts tables from pdfplumber output as standalone atomic chunks with surrounding paragraph context prepended. Ensures table data is never split across chunk boundaries.

**Semantic Chunker** — encodes all sentences using the all-MiniLM-L6-v2 sentence transformer model and computes cosine similarity between consecutive sentence pairs. Where similarity drops below a configurable threshold, a semantic topic shift is detected and a new chunk boundary is inserted.

---

## Output

After running, the `results/` folder contains:

```
results/
├── yourfile__pypdf.json                         # metrics only
├── yourfile__pypdf.txt                          # full extracted text
├── yourfile__pdfplumber.json
├── yourfile__pdfplumber.txt
├── yourfile__pymupdf.json
├── yourfile__pymupdf.txt
├── yourfile__unstructured_fast.json
├── yourfile__unstructured_fast.txt
├── yourfile__docling.json
├── yourfile__docling.txt
├── yourfile__pypdf__naive_chunks.json           # sentence-aware chunks
├── yourfile__pdfplumber__table_chunks.json      # tables preserved intact
├── yourfile__unstructured_fast__layout_chunks.json  # heading-based chunks
├── yourfile__docling__naive_chunks.json
├── yourfile__docling__semantic_chunks.json      # embedding-based chunks
├── comparison__yourfile.json                    # extraction summary
└── chunk_summary__yourfile.json                 # chunking summary
```

Each `.txt` file contains:

```
=== Loader Name | filename.pdf ===

Status      : success
Time        : 0.44s
Pages       : 11
Chars       : 29711
Word Count  : 4498
Has Tables  : True
Has Headers : False
Images      : 0

=== EXTRACTED CONTENT ===

... full extracted text here ...
```

Each chunk JSON contains:

```json
{
  "file": "filename.pdf",
  "loader": "Docling (IBM)",
  "chunker": "semantic",
  "total_chunks": 28,
  "chunks": [
    {
      "chunk_index": 0,
      "content": "The Transformer model relies entirely on attention...",
      "char_count": 487,
      "token_estimate": 121,
      "sentence_count": 4,
      "chunker": "semantic"
    }
  ]
}
```

---

## Memory Tracking

Every loader tracks:

- Process RAM before and after extraction
- RAM delta — how much memory the loader consumed
- Peak RAM usage during extraction
- GPU availability and GPU RAM used if GPU is present

---

## Usage on Linux

Internet connection required during build only. After build completes the tool runs fully offline.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-extraction-api.git
cd pdf-extraction-api
```

### Step 2: Create required folders

```bash
mkdir -p test_pdfs results
```

### Step 3: Add your PDFs

```bash
cp /path/to/your/file.pdf test_pdfs/
```

### Step 4: Build the Docker image

All ML models (Docling, Unstructured OCR, Marker, sentence-transformers)
are downloaded and baked into the image during build so all future runs
are fully offline. Build time is approximately 30-40 minutes on first run.

```bash
docker-compose build
```

### Step 5: Run extraction

Run fast loaders only (PyPDF, pdfplumber, PyMuPDF, Unstructured fast, Docling):

```bash
docker-compose run pdf-extractor
```

Include Unstructured OCR hi_res (~90s per PDF):

```bash
docker-compose run pdf-extractor python run_extraction.py --ocr
```

Include Marker (~120s per PDF, needs 6GB+ free RAM):

```bash
docker-compose run pdf-extractor python run_extraction.py --marker
```

Run extraction and chunking:

```bash
docker-compose run pdf-extractor python run_extraction.py --chunk
```

Run everything:

```bash
docker-compose run pdf-extractor python run_extraction.py --ocr --chunk
```

Run Marker separately after other loaders (recommended to avoid RAM conflicts):

```bash
# Step 1 — run all other loaders first
docker-compose run pdf-extractor python run_extraction.py --ocr --chunk

# Step 2 — run Marker separately
docker-compose run pdf-extractor python run_extraction.py --marker
```

### Step 6: View results

```bash
# List all output files
ls results/

# View comparison summary
cat results/comparison__yourfile.json

# View chunk summary
cat results/chunk_summary__yourfile.json

# View extracted text from a specific loader
cat results/yourfile__docling.txt

# Check word counts across all loaders
grep "Word Count" results/yourfile__*.txt
```

---

## Usage on Mac

Marker is disabled on Mac due to RAM constraints — requires 6GB+ free RAM.
All other 6 loaders and all 4 chunkers work normally.
Internet connection required during build only.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-extraction-api.git
cd pdf-extraction-api
```

### Step 2: Create required folders

```bash
mkdir -p test_pdfs results
```

### Step 3: Add your PDFs

```bash
cp /path/to/your/file.pdf test_pdfs/
```

### Step 4: Build the Docker image

Uses `Dockerfile` (Mac version) via `docker-compose.mac.yml`.
Marker models are not downloaded during Mac build.
Build time is approximately 15-20 minutes on first run.

```bash
docker-compose -f docker-compose.mac.yml build
```

### Step 5: Run extraction

Run fast loaders only:

```bash
docker-compose -f docker-compose.mac.yml run pdf-extractor
```

Include Unstructured OCR hi_res:

```bash
docker-compose -f docker-compose.mac.yml run pdf-extractor \
  python run_extraction.py --ocr
```

Run extraction and chunking:

```bash
docker-compose -f docker-compose.mac.yml run pdf-extractor \
  python run_extraction.py --chunk
```

Run everything available on Mac:

```bash
docker-compose -f docker-compose.mac.yml run pdf-extractor \
  python run_extraction.py --ocr --chunk
```

### Step 6: View results

```bash
ls results/
cat results/comparison__yourfile.json
cat results/chunk_summary__yourfile.json
grep "Word Count" results/yourfile__*.txt
```

---

## CLI Flags

| Flag       | Description                                                                | Extra Time      |
| ---------- | -------------------------------------------------------------------------- | --------------- |
| (none)     | Run 5 fast loaders: PyPDF, pdfplumber, PyMuPDF, Unstructured fast, Docling | —               |
| `--ocr`    | Also run Unstructured OCR hi_res                                           | ~90s per PDF    |
| `--marker` | Also run Marker (Linux only, needs 6GB+ RAM)                               | ~120s per PDF   |
| `--chunk`  | Run all 4 chunking strategies after extraction                             | ~30-60s per PDF |

---

## Requirements

### Linux

- Docker Engine 20.0+
- Docker Compose 2.0+
- 8GB+ free RAM for Marker
- 30GB+ free disk space for Docker image with all models
- Internet connection during build only

### Mac

- Docker Desktop
- 4GB+ free RAM
- 20GB+ free disk space
- Internet connection during build only
- Apple Silicon (M1/M2/M3) or Intel supported

---

## Why Two Dockerfiles?

The Mac version (`Dockerfile`) skips pre-downloading Marker models
since Mac does not have enough free RAM to run Marker. It sets
`SKIP_MARKER=true` and takes around 15-20 minutes to build.

The Linux version (`Dockerfile.linux`) pre-downloads all models
including Marker (~3GB) during build so the tool runs completely
offline after that. It sets `SKIP_MARKER=false` and takes around
30-40 minutes to build due to the additional model downloads.

Both versions pre-download Docling, Unstructured OCR, and
sentence-transformers models during build so those also work
offline after the first build.

---

## Benchmark Results (Sample — Attention Is All You Need, 11 pages)

| Loader           | Chars      | Tables | Images | Time   | RAM Delta |
| ---------------- | ---------- | ------ | ------ | ------ | --------- |
| PyPDF            | 28,542     | ❌     | ❌     | 0.19s  | 45 MB     |
| pdfplumber       | 28,198     | ✅     | ❌     | 0.44s  | 52 MB     |
| PyMuPDF          | 30,222     | ❌     | ✅ 3   | 0.17s  | 38 MB     |
| Unstructured     | 27,360     | ❌     | ❌     | 5.68s  | 210 MB    |
| Unstructured OCR | 30,701     | ✅     | ✅     | 70.75s | 890 MB    |
| Docling          | 43,329     | ✅     | ❌     | 106s   | 1831 MB   |
| Marker           | Linux only | ✅     | ✅     | ~120s  | ~3000 MB  |

---

## Tech Stack

| Component        | Technology                                                |
| ---------------- | --------------------------------------------------------- |
| Containerization | Docker + Docker Compose                                   |
| PDF Loaders      | PyPDF, pdfplumber, PyMuPDF, Unstructured, Docling, Marker |
| Chunking         | Custom RAGFlow-inspired pipeline                          |
| Embeddings       | sentence-transformers all-MiniLM-L6-v2                    |
| Memory Tracking  | psutil + PyTorch CUDA                                     |
| Output Format    | JSON (metrics + chunks) + TXT (extracted content)         |
| Language         | Python 3.11                                               |
