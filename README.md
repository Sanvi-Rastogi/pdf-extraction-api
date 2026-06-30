# PDF Extraction Benchmark Tool

A Dockerized CLI tool that benchmarks 6 different PDF extraction methods
(PyPDF, pdfplumber, PyMuPDF, Unstructured, Docling, Marker) — comparing
text extraction quality, table detection, image awareness, memory usage,
and speed.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/pdf-extraction-api.git
cd pdf-extraction-api
mkdir -p test_pdfs results marker_cache docling_cache
```

Place your PDF files inside `test_pdfs/`

## Run

```bash
# Build the Docker image
docker-compose build

# Run fast loaders (PyPDF, pdfplumber, PyMuPDF, Unstructured, Docling)
docker-compose run pdf-extractor

# Include slow OCR mode
docker-compose run pdf-extractor python run_extraction.py --ocr

# Include Marker (requires 6GB+ RAM, set SKIP_MARKER=false in docker-compose.yml)
docker-compose run pdf-extractor python run_extraction.py --marker

# Run everything
docker-compose run pdf-extractor python run_extraction.py --ocr --marker
```

## Output

Results saved to `results/` folder:

- `*.json` — extraction metrics (chars, tables, images, time, RAM usage)
- `*.txt` — full extracted text content per loader
- `comparison__*.json` — side-by-side summary across all loaders

## Loaders Compared

| Loader           | Tables | Images | Headers | Speed   |
| ---------------- | ------ | ------ | ------- | ------- |
| PyPDF            | ❌     | ❌     | ❌      | Fastest |
| pdfplumber       | ✅     | ❌     | ❌      | Fast    |
| PyMuPDF          | ❌     | ✅     | ❌      | Fastest |
| Unstructured     | ❌     | ❌     | ✅      | Medium  |
| Unstructured OCR | ✅     | ✅     | ✅      | Slow    |
| Docling          | ✅     | ❌     | ✅      | Slow    |
| Marker           | ✅     | ✅     | ✅      | Slow    |

## Memory Tracking

Each extraction tracks process RAM delta and GPU availability,
useful for understanding resource requirements before deploying
to production.
