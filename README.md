# PDF Extraction Benchmark Tool

A Dockerized CLI tool that benchmarks 7 different PDF extraction methods —
comparing text extraction quality, table detection, image awareness,
memory usage, and processing speed across the same document.

Built as part of a research project to identify the best PDF loader
for academic document RAG (Retrieval-Augmented Generation) pipelines.

---

## Project Structure

```
pdf-extraction-api/
├── run_extraction.py           # Main CLI script
├── extractors/
│   ├── __init__.py
│   ├── pypdf_extractor.py      # PyPDF loader
│   ├── pdfplumber_extractor.py # pdfplumber loader
│   ├── pymupdf_extractor.py    # PyMuPDF loader
│   ├── unstructured_extractor.py # Unstructured loader
│   ├── docling_extractor.py    # Docling (IBM) loader
│   └── marker_extractor.py     # Marker (Datalab) loader
├── utils/
│   ├── __init__.py
│   └── memory_tracker.py       # RAM and GPU memory tracking
├── Dockerfile                  # Mac build (no Marker pre-download)
├── Dockerfile.linux            # Linux build (all models pre-downloaded)
├── docker-compose.yml          # Linux configuration
├── docker-compose.mac.yml      # Mac configuration
├── requirements.txt
├── test_pdfs/                  # Put your PDFs here
└── results/                    # Output saved here automatically
```

---

## Memory Tracking

Every loader tracks:

- Process RAM before and after extraction
- RAM delta (how much memory the loader consumed)
- Peak RAM usage
- GPU availability and GPU RAM used (if GPU present)

---

## Usage on Linux

> Requires Docker and Docker Compose installed.
> Internet connection required during build only.
> After build completes — fully offline.

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

Copy your PDF files into the `test_pdfs/` folder:

```bash
cp /path/to/your/file.pdf test_pdfs/
```

### Step 4: Build the Docker image

> This step requires internet. It downloads all ML models
> (Docling, Unstructured OCR, Marker) and bakes them into
> the image so all future runs are fully offline.
> Build time: approximately 30-40 minutes on first run.

```bash
docker-compose build
```

### Step 5: Run the benchmark

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

Run everything (all 7 loaders):

```bash
docker-compose run pdf-extractor python run_extraction.py --ocr --marker
```

Run Marker separately (recommended to avoid RAM conflicts):

```bash
# Step 1 — run all other loaders first
docker-compose run pdf-extractor python run_extraction.py --ocr

# Step 2 — run Marker separately after Step 1 finishes
docker-compose run pdf-extractor python run_extraction.py --marker
```

---

## Usage on Mac

> Marker is disabled on Mac due to RAM constraints (requires 6GB+ free RAM).
> All other 6 loaders work normally.
> Internet connection required during build only.

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

> Uses `Dockerfile` (Mac version) via `docker-compose.mac.yml`.
> Marker models are NOT downloaded during Mac build.
> Build time: approximately 15-20 minutes on first run.

```bash
docker-compose -f docker-compose.mac.yml build
```

### Step 5: Run the benchmark

Run fast loaders only (PyPDF, pdfplumber, PyMuPDF, Unstructured fast, Docling):

```bash
docker-compose -f docker-compose.mac.yml run pdf-extractor
```

Include Unstructured OCR hi_res (~90s per PDF):

```bash
docker-compose -f docker-compose.mac.yml run pdf-extractor \
  python run_extraction.py --ocr
```

> Note: `--marker` flag is ignored on Mac since `SKIP_MARKER=true`
> is set in `docker-compose.mac.yml`. Marker endpoint returns a
> graceful skip message instead of crashing.

---

## Output

After running, the `results/` folder contains:

```
results/
├── yourfile__pypdf.json                  # metrics only
├── yourfile__pypdf.txt                   # full extracted text
├── yourfile__pdfplumber.json
├── yourfile__pdfplumber.txt
├── yourfile__pymupdf.json
├── yourfile__pymupdf.txt
├── yourfile__unstructured_fast.json
├── yourfile__unstructured_fast.txt
├── yourfile__docling.json
├── yourfile__docling.txt                 # markdown formatted
├── yourfile__marker.json
├── yourfile__marker.txt                  # markdown with equations
└── comparison__yourfile.json             # side-by-side summary
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

---

## Requirements

### Linux

- Docker Engine 20.0+
- Docker Compose 2.0+
- 8GB+ free RAM (for Marker)
- 30GB+ free disk space (for Docker image with all models)
- Internet connection (build time only)

### Mac

- Docker Desktop
- 4GB+ free RAM
- 20GB+ free disk space
- Internet connection (build time only)
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

Both versions pre-download Docling and Unstructured OCR models
during build so those also work offline after the first build.

---

## Benchmark Results (Sample — Attention Is All You Need Paper)

Tested on: 11-page research paper with tables and figures

| Loader           | Chars  | Tables | Images | Time   | RAM Delta |
| ---------------- | ------ | ------ | ------ | ------ | --------- |
| PyPDF            | 28,542 | ❌     | ❌     | 0.19s  | 45 MB     |
| pdfplumber       | 28,198 | ✅     | ❌     | 0.44s  | 52 MB     |
| PyMuPDF          | 30,222 | ❌     | ✅     | 0.17s  | 38 MB     |
| Unstructured     | 27,360 | ❌     | ❌     | 5.68s  | 210 MB    |
| Unstructured OCR | 30,701 | ✅     | ✅     | 70.75s | 890 MB    |
| Docling          | 43,329 | ✅     | ❌     | 106s   | 1831 MB   |
| Marker           | TBD\*  | ✅     | ✅     | TBD\*  | TBD\*     |

\*Marker requires Linux with 6GB+ free RAM — not testable on Mac.
