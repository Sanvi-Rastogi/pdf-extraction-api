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
from typing import List
from chunkers import naive_chunker, layout_chunker, semantic_chunker, table_chunker

console = Console()

PDF_DIR = Path("test_pdfs")
RESULTS_DIR = Path("results")
PDF_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Check if Marker should be skipped (set in docker-compose.yml)
SKIP_MARKER = os.getenv("SKIP_MARKER", "false").lower() == "true"
if not SKIP_MARKER:
    from extractors import marker_extractor


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


def save_chunks(chunks: List[dict], filename: str, loader: str, chunker: str):
    """
    Save chunks to results folder as JSON.
    Each chunk has its content, metadata, and token estimate.
    """
    safe_loader = loader.replace(" ", "_").lower()
    safe_chunker = chunker.replace(" ", "_").lower()
    safe_file = Path(filename).stem

    out_path = RESULTS_DIR / \
        f"{safe_file}__{safe_loader}__{safe_chunker}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "file": filename,
            "loader": loader,
            "chunker": chunker,
            "total_chunks": len(chunks),
            "chunks": chunks
        }, f, indent=2)

    console.print(
        f"[green]✓ {len(chunks)} chunks saved to {out_path}[/green]"
    )
    return len(chunks)

def run_chunking(results: dict, pdf_name: str):
    """
    Run chunking on extracted content from each loader.
    Different loaders get different chunking strategies:

    - pypdf/pymupdf    → naive chunker (sentence-aware)
    - unstructured     → layout chunker (element-type aware)
    - pdfplumber       → table chunker (preserves tables)
    - docling          → naive chunker (already markdown-structured)
    - semantic         → semantic chunker (embedding-based) on best result
    """
    console.print(Panel(
        "[bold cyan]Running Semantic Chunking...[/bold cyan]",
        title="CHUNKING"
    ))

    chunk_summary = []

    for loader_key, result in results.items():
        if result.get("status") != "success":
            continue

        content = result.get("content", "")
        if not content:
            continue

        loader_name = result.get("loader", loader_key)
        console.print(f"\n[yellow]Chunking {loader_name}...[/yellow]")

        try:
            # Choose chunking strategy based on loader
            if loader_key == "pdfplumber":
                # pdfplumber has [TABLE N] markers — use table chunker
                chunks = table_chunker.chunk(content)
                chunker_name = "table"

            elif loader_key == "unstructured_fast":
                # Unstructured has [ElementType] markers — use layout chunker
                chunks = layout_chunker.chunk(content)
                chunker_name = "layout"

            elif loader_key == "docling":
                # Docling outputs clean markdown — naive works well
                chunks = naive_chunker.chunk(content, max_tokens=512)
                chunker_name = "naive"

            else:
                # pypdf, pymupdf — plain text, use naive chunker
                chunks = naive_chunker.chunk(content, max_tokens=512)
                chunker_name = "naive"

            # Save chunks
            total = save_chunks(chunks, pdf_name, loader_name, chunker_name)

            chunk_summary.append({
                "loader": loader_name,
                "chunker": chunker_name,
                "total_chunks": total,
                "avg_chunk_chars": round(
                    sum(c["char_count"] for c in chunks) / max(total, 1), 1
                ),
                "avg_tokens": round(
                    sum(c["token_estimate"] for c in chunks) / max(total, 1), 1
                )
            })

        except Exception as e:
            console.print(f"[red]Chunking failed for {loader_name}: {e}[/red]")

    # Run semantic chunker on best result (most chars — usually Docling)
    best_result = max(
        [r for r in results.values() if r.get("status") == "success"],
        key=lambda x: x.get("total_chars", 0),
        default=None
    )

    if best_result:
        console.print(
            f"\n[yellow]Running semantic chunker on best result "
            f"({best_result.get('loader')})...[/yellow]"
        )
        try:
            semantic_chunks = semantic_chunker.chunk(
                best_result.get("content", ""),
                max_tokens=512,
                similarity_threshold=0.3
            )
            total = save_chunks(
                semantic_chunks,
                pdf_name,
                best_result.get("loader"),
                "semantic"
            )
            chunk_summary.append({
                "loader": best_result.get("loader"),
                "chunker": "semantic",
                "total_chunks": total,
                "avg_chunk_chars": round(
                    sum(c["char_count"] for c in semantic_chunks) / max(total, 1), 1
                ),
                "avg_tokens": round(
                    sum(c["token_estimate"] for c in semantic_chunks) / max(total, 1), 1
                )
            })
        except Exception as e:
            console.print(f"[red]Semantic chunking failed: {e}[/red]")

    # Print chunking summary table
    chunk_table = Table(show_header=True, header_style="bold magenta")
    chunk_table.add_column("Loader", width=28)
    chunk_table.add_column("Chunker", width=12)
    chunk_table.add_column("Total Chunks", width=14)
    chunk_table.add_column("Avg Chars/Chunk", width=16)
    chunk_table.add_column("Avg Tokens/Chunk", width=17)

    for entry in chunk_summary:
        chunk_table.add_row(
            entry["loader"],
            entry["chunker"],
            str(entry["total_chunks"]),
            str(entry["avg_chunk_chars"]),
            str(entry["avg_tokens"])
        )

    console.print("\n")
    console.print(chunk_table)

    # Save chunk summary
    summary_path = RESULTS_DIR / f"chunk_summary__{Path(pdf_name).stem}.json"
    with open(summary_path, "w") as f:
        json.dump(chunk_summary, f, indent=2)

    console.print(f"\n[bold]Chunk summary saved to {summary_path}[/bold]")

# main 
def run(include_ocr: bool, include_marker: bool, include_chunk: bool = False):
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

        # Always run fast loaders
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

        # Optional: Unstructured OCR bec slow
        if include_ocr:
            console.print("[yellow]Running Unstructured OCR hi_res (slow)...[/yellow]")
            results["unstructured_ocr"] = unstructured_extractor.extract(
                str(pdf_path), strategy="hi_res"
            )

        # Optional: Marker bec needs lots of RAM
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

        # Save to results folder
        for loader_key, result in results.items():
            save_result(result, pdf_path.name, loader_key)

        # print summary table
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

        if include_chunk:
            run_chunking(results, pdf_path.name)



# entry point
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
    parser.add_argument(
        "--chunk", action="store_true",
        help="Run semantic chunking on extracted content"
    )
    args = parser.parse_args()

    run(include_ocr=args.ocr, include_marker=args.marker, include_chunk=args.chunk)


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