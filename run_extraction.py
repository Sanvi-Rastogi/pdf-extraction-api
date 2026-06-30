# ─────────────────────────────────────────────────────────────────────────────
# run_extraction.py
# CLI script that runs all PDF extraction loaders on every PDF found
# in the test_pdfs/ folder. No UI, no API — just run from terminal.
#
# Usage:
#   python run_extraction.py                    # run fast loaders only
#   python run_extraction.py --ocr               # also run Unstructured OCR
#   python run_extraction.py --marker             # also run Marker
#   python run_extraction.py --ocr --marker       # run everything
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from extractors import (
    pypdf_extractor,
    pdfplumber_extractor,
    pymupdf_extractor,
    unstructured_extractor,
    docling_extractor,
)

console = Console()

PDF_DIR = Path("test_pdfs")
RESULTS_DIR = Path("results")
PDF_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Check if Marker should be skipped (set in docker-compose.yml)
SKIP_MARKER = os.getenv("SKIP_MARKER", "false").lower() == "true"
if not SKIP_MARKER:
    from extractors import marker_extractor


# ── Save Results ─────────────────────────────────────────────────────────────

def save_result(result: dict, filename: str, loader: str):
    """
    Save extraction result in TWO formats:
    1. JSON file — metrics only (no content), for quick scanning
    2. TXT file — full extracted content, for reading actual extracted text
    """
    safe_loader = loader.replace(" ", "_").replace("/", "_").lower()
    safe_file = Path(filename).stem

    # Save metrics-only JSON
    json_path = RESULTS_DIR / f"{safe_file}__{safe_loader}.json"
    with open(json_path, "w") as f:
        json.dump(
            {k: v for k, v in result.items() if k != "content"},
            f, indent=2
        )

    # Save full extracted content as readable TXT
    if result.get("status") == "success" and "content" in result:
        txt_path = RESULTS_DIR / f"{safe_file}__{safe_loader}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== {result.get('loader', loader)} | {filename} ===\n\n")
            f.write(f"Status      : {result.get('status')}\n")
            f.write(f"Time        : {result.get('time_sec')}s\n")
            f.write(f"Pages       : {result.get('pages')}\n")
            f.write(f"Chars       : {result.get('total_chars')}\n")
            # Word count helps verify nothing was truncated
            word_count = len(result["content"].split())
            f.write(f"Word Count  : {word_count}\n")
            f.write(f"Has Tables  : {result.get('has_table_structure', False)}\n")
            f.write(f"Has Headers : {result.get('has_headers', False)}\n")
            f.write(f"Images      : {result.get('images_detected', 0)}\n\n")
            f.write("=== EXTRACTED CONTENT ===\n\n")
            f.write(result["content"])


# ── Main Runner ───────────────────────────────────────────────────────────────

def run(include_ocr: bool, include_marker: bool):
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        console.print(Panel(
            "[red]No PDF files found in test_pdfs/ folder.\n"
            "Add PDFs there and run again.[/red]",
            title="ERROR"
        ))
        return

    for pdf_path in pdf_files:
        console.print(Panel(
            f"[bold cyan]Testing: {pdf_path.name}[/bold cyan]",
            title="PDF EXTRACTION BENCHMARK"
        ))

        results = {}

        # ── Always run these fast loaders ────────────────────────────────────
        console.print("[yellow]Running PyPDF...[/yellow]")
        results["pypdf"] = pypdf_extractor.extract(str(pdf_path))

        console.print("[yellow]Running pdfplumber...[/yellow]")
        results["pdfplumber"] = pdfplumber_extractor.extract(str(pdf_path))

        console.print("[yellow]Running PyMuPDF...[/yellow]")
        results["pymupdf"] = pymupdf_extractor.extract(str(pdf_path))

        console.print("[yellow]Running Unstructured (fast)...[/yellow]")
        results["unstructured_fast"] = unstructured_extractor.extract(
            str(pdf_path), strategy="fast"
        )

        console.print("[yellow]Running Docling...[/yellow]")
        results["docling"] = docling_extractor.extract(str(pdf_path))

        # ── Optional: Unstructured OCR (slow) ─────────────────────────────────
        if include_ocr:
            console.print("[yellow]Running Unstructured OCR hi_res (slow)...[/yellow]")
            results["unstructured_ocr"] = unstructured_extractor.extract(
                str(pdf_path), strategy="hi_res"
            )

        # ── Optional: Marker (needs lots of RAM) ──────────────────────────────
        if include_marker:
            if SKIP_MARKER:
                console.print(
                    "[red]Marker skipped — SKIP_MARKER=true on this system[/red]"
                )
                results["marker"] = {
                    "loader": "Marker (Datalab)",
                    "status": "skipped",
                    "reason": "SKIP_MARKER enabled on this system"
                }
            else:
                console.print("[yellow]Running Marker (this takes time)...[/yellow]")
                results["marker"] = marker_extractor.extract(str(pdf_path))

        # ── Save every result to results/ folder ──────────────────────────────
        for loader_key, result in results.items():
            save_result(result, pdf_path.name, loader_key)

        # ── Build and print summary table ─────────────────────────────────────
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Loader", width=28)
        table.add_column("Status", width=10)
        table.add_column("Pages", width=7)
        table.add_column("Chars", width=8)
        table.add_column("Words", width=8)
        table.add_column("Tables", width=8)
        table.add_column("Images", width=8)
        table.add_column("Headers", width=8)
        table.add_column("Time(s)", width=8)
        table.add_column("RAM Δ(MB)", width=10)

        summary = []

        for loader_key, result in results.items():
            status = result.get("status", "failed")

            if status == "success":
                mem = result.get("memory", {})
                word_count = len(result.get("content", "").split())

                table.add_row(
                    result.get("loader", loader_key),
                    "[green]✓[/green]",
                    str(result.get("pages", "-")),
                    str(result.get("total_chars", 0)),
                    str(word_count),
                    "[green]Yes[/green]" if result.get("has_table_structure") else "No",
                    str(result.get("images_detected", 0)),
                    "[green]Yes[/green]" if result.get("has_headers") else "No",
                    str(result.get("time_sec", "-")),
                    str(mem.get("ram_delta_mb", "-")),
                )

                summary.append({
                    "loader": result.get("loader", loader_key),
                    "status": "success",
                    "pages": result.get("pages"),
                    "total_chars": result.get("total_chars"),
                    "word_count": word_count,
                    "has_tables": result.get("has_table_structure", False),
                    "images_detected": result.get("images_detected", 0),
                    "has_headers": result.get("has_headers", False),
                    "time_sec": result.get("time_sec"),
                    "ram_delta_mb": mem.get("ram_delta_mb"),
                })
            else:
                table.add_row(
                    result.get("loader", loader_key),
                    "[red]✗ FAIL[/red]",
                    "-", "-", "-", "-", "-", "-", "-", "-"
                )
                summary.append({
                    "loader": result.get("loader", loader_key),
                    "status": status,
                    "error": result.get("error", result.get("reason", "unknown"))
                })

        console.print("\n")
        console.print(table)

        # Save comparison summary
        comp_path = RESULTS_DIR / f"comparison__{pdf_path.stem}.json"
        with open(comp_path, "w") as f:
            json.dump({
                "file": pdf_path.name,
                "summary": summary
            }, f, indent=2)

        console.print(f"\n[bold]Summary saved to {comp_path}[/bold]")
        console.print(f"[bold]Full extracted text saved as .txt files in results/[/bold]\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PDF extraction benchmark across multiple loaders"
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Also run Unstructured OCR hi_res (slow, ~90s per PDF)"
    )
    parser.add_argument(
        "--marker", action="store_true",
        help="Also run Marker (needs 6GB+ RAM)"
    )
    args = parser.parse_args()

    run(include_ocr=args.ocr, include_marker=args.marker)


#Clone the repo
# git clone https://github.com/Sanvi-Rastogi/pdf-extraction-api.git
# cd pdf-extraction-api

#Create folders
# mkdir -p test_pdfs results marker_cache docling_cache

# Copy your PDFs into test_pdfs/

# Update docker-compose.yml — change SKIP_MARKER to false (as sys will have enough memory)

#Build and run:
# docker-compose build
# docker-compose run pdf-extractor python run_extraction.py --ocr --marker