import os
import json
import argparse
from pathlib import Path
from typing import List
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
from chunkers import (
    naive_chunker,
    layout_chunker,
    table_chunker,
    semantic_chunker,
)

console = Console()

PDF_DIR = Path("test_pdfs")
RESULTS_DIR = Path("results")
PDF_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SKIP_MARKER = os.getenv("SKIP_MARKER", "false").lower() == "true"
if not SKIP_MARKER:
    from extractors import marker_extractor


def save_result(result: dict, filename: str, loader: str):
    safe_loader = loader.replace(" ", "_").replace("/", "_").lower()
    safe_file = Path(filename).stem

    json_path = RESULTS_DIR / f"{safe_file}__{safe_loader}.json"
    with open(json_path, "w") as f:
        json.dump(
            {k: v for k, v in result.items() if k != "content"},
            f, indent=2
        )

    if result.get("status") == "success" and "content" in result:
        txt_path = RESULTS_DIR / f"{safe_file}__{safe_loader}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(
                f"=== {result.get('loader', loader)} | {filename} ===\n\n"
            )
            f.write(f"Status      : {result.get('status')}\n")
            f.write(f"Time        : {result.get('time_sec')}s\n")
            f.write(f"Pages       : {result.get('pages')}\n")
            f.write(f"Chars       : {result.get('total_chars')}\n")
            word_count = len(result["content"].split())
            f.write(f"Word Count  : {word_count}\n")
            f.write(
                f"Has Tables  : {result.get('has_table_structure', False)}\n"
            )
            f.write(f"Has Headers : {result.get('has_headers', False)}\n")
            f.write(f"Images      : {result.get('images_detected', 0)}\n\n")
            f.write("=== EXTRACTED CONTENT ===\n\n")
            f.write(result["content"])


def save_chunks(
    chunks: List[dict],
    filename: str,
    loader: str,
    chunker: str
):
    safe_loader = loader.replace(" ", "_").replace("/", "_").lower()
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
        f"[green]✓ {len(chunks)} chunks → {out_path.name}[/green]"
    )
    return len(chunks)


# Chunking
def run_chunking(results: dict, pdf_name: str):
    """Run all 4 chunking strategies on Docling output only."""
    console.print(Panel(
        "[bold cyan]Running Chunking on Docling Output...[/bold cyan]",
        title="CHUNKING"
    ))

    docling_result = results.get("docling", {})

    if docling_result.get("status") != "success":
        console.print(
            "[red]Docling failed — cannot run chunking.[/red]"
        )
        return

    content = docling_result.get("content", "")
    loader_name = docling_result.get("loader", "Docling (IBM)")

    if not content:
        console.print("[red]Docling output empty.[/red]")
        return

    console.print(
        f"[green]Source: Docling — "
        f"{len(content)} chars, "
        f"{len(content.split())} words[/green]\n"
    )

    chunk_summary = []

    # Naive
    console.print("[yellow]Running naive chunker...[/yellow]")
    try:
        chunks = naive_chunker.chunk(content, max_tokens=512)
        total = save_chunks(chunks, pdf_name, loader_name, "naive")
        chunk_summary.append({
            "chunker": "naive",
            "description": "Sentence-boundary aware",
            "total_chunks": total,
            "avg_chunk_chars": round(
                sum(c["char_count"] for c in chunks) / max(total, 1), 1
            ),
            "avg_tokens": round(
                sum(c["token_estimate"] for c in chunks) / max(total, 1), 1
            ),
            "status": "success"
        })
    except Exception as e:
        console.print(f"[red]Naive chunker failed: {e}[/red]")
        chunk_summary.append({"chunker": "naive", "status": f"failed: {e}"})

    # Layout
    console.print("[yellow]Running layout chunker...[/yellow]")
    try:
        chunks = layout_chunker.chunk(content, max_tokens=512)
        total = save_chunks(chunks, pdf_name, loader_name, "layout")
        chunk_summary.append({
            "chunker": "layout",
            "description": "Heading-boundary aware",
            "total_chunks": total,
            "avg_chunk_chars": round(
                sum(c["char_count"] for c in chunks) / max(total, 1), 1
            ),
            "avg_tokens": round(
                sum(c["token_estimate"] for c in chunks) / max(total, 1), 1
            ),
            "status": "success"
        })
    except Exception as e:
        console.print(f"[red]Layout chunker failed: {e}[/red]")
        chunk_summary.append({"chunker": "layout", "status": f"failed: {e}"})

    # Table
    console.print("[yellow]Running table chunker...[/yellow]")
    try:
        chunks = table_chunker.chunk(content, max_tokens=512)
        total = save_chunks(chunks, pdf_name, loader_name, "table")
        chunk_summary.append({
            "chunker": "table",
            "description": "Table-preserving atomic chunks",
            "total_chunks": total,
            "avg_chunk_chars": round(
                sum(c["char_count"] for c in chunks) / max(total, 1), 1
            ),
            "avg_tokens": round(
                sum(c["token_estimate"] for c in chunks) / max(total, 1), 1
            ),
            "status": "success"
        })
    except Exception as e:
        console.print(f"[red]Table chunker failed: {e}[/red]")
        chunk_summary.append({"chunker": "table", "status": f"failed: {e}"})

    # Semantic
    console.print("[yellow]Running semantic chunker...[/yellow]")
    try:
        chunks = semantic_chunker.chunk(
            content,
            max_tokens=512,
            similarity_threshold=0.3
        )
        total = save_chunks(chunks, pdf_name, loader_name, "semantic")
        chunk_summary.append({
            "chunker": "semantic",
            "description": "Embedding-based topic shift detection",
            "total_chunks": total,
            "avg_chunk_chars": round(
                sum(c["char_count"] for c in chunks) / max(total, 1), 1
            ),
            "avg_tokens": round(
                sum(c["token_estimate"] for c in chunks) / max(total, 1), 1
            ),
            "status": "success"
        })
    except Exception as e:
        console.print(f"[red]Semantic chunker failed: {e}[/red]")
        chunk_summary.append(
            {"chunker": "semantic", "status": f"failed: {e}"}
        )

    chunk_table = Table(show_header=True, header_style="bold magenta")
    chunk_table.add_column("Chunker", width=12)
    chunk_table.add_column("Description", width=30)
    chunk_table.add_column("Chunks", width=8)
    chunk_table.add_column("Avg Chars", width=12)
    chunk_table.add_column("Avg Tokens", width=12)
    chunk_table.add_column("Status", width=10)

    for entry in chunk_summary:
        if entry.get("status") == "success":
            chunk_table.add_row(
                entry["chunker"],
                entry.get("description", ""),
                str(entry["total_chunks"]),
                str(entry["avg_chunk_chars"]),
                str(entry["avg_tokens"]),
                "[green]✓[/green]"
            )
        else:
            chunk_table.add_row(
                entry["chunker"], "-", "-", "-", "-",
                "[red]✗ FAIL[/red]"
            )

    console.print("\n")
    console.print(chunk_table)

    summary_path = RESULTS_DIR / f"chunk_summary__{Path(pdf_name).stem}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "file": pdf_name,
            "extraction_source": "Docling (IBM)",
            "total_chars": len(content),
            "word_count": len(content.split()),
            "chunking_strategies": chunk_summary
        }, f, indent=2)

    console.print(f"\n[bold]Chunk summary → {summary_path.name}[/bold]")



def run(
    include_ocr: bool,
    include_marker: bool,
    include_chunk: bool
):
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        console.print(Panel(
            "[red]No PDFs found in test_pdfs/ folder.[/red]",
            title="ERROR"
        ))
        return

    for pdf_path in pdf_files:
        console.print(Panel(
            f"[bold cyan]Testing: {pdf_path.name}[/bold cyan]",
            title="PDF EXTRACTION BENCHMARK"
        ))

        results = {}

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

        if include_ocr:
            console.print(
                "[yellow]Running Unstructured OCR hi_res...[/yellow]"
            )
            results["unstructured_ocr"] = unstructured_extractor.extract(
                str(pdf_path), strategy="hi_res"
            )

        if include_marker:
            if SKIP_MARKER:
                console.print(
                    "[red]Marker skipped — SKIP_MARKER=true[/red]"
                )
                results["marker"] = {
                    "loader": "Marker (Datalab)",
                    "status": "skipped",
                    "reason": "SKIP_MARKER enabled"
                }
            else:
                console.print("[yellow]Running Marker...[/yellow]")
                results["marker"] = marker_extractor.extract(str(pdf_path))

        for loader_key, result in results.items():
            save_result(result, pdf_path.name, loader_key)

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
                    "[green]Yes[/green]" if result.get(
                        "has_table_structure"
                    ) else "No",
                    str(result.get("images_detected", 0)),
                    "[green]Yes[/green]" if result.get(
                        "has_headers"
                    ) else "No",
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
                    "error": result.get(
                        "error", result.get("reason", "unknown")
                    )
                })

        console.print("\n")
        console.print(table)

        comp_path = RESULTS_DIR / f"comparison__{pdf_path.stem}.json"
        with open(comp_path, "w") as f:
            json.dump({"file": pdf_path.name, "summary": summary}, f, indent=2)

        console.print(f"\n[bold]Summary → {comp_path.name}[/bold]")
        console.print(
            f"[bold]Extracted text saved as .txt files in results/[/bold]\n"
        )

        if include_chunk:
            run_chunking(results, pdf_path.name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF extraction benchmark — runs fully offline after build"
    )
    parser.add_argument(
        "--ocr", action="store_true",
        help="Also run Unstructured OCR hi_res (~90s per PDF)"
    )
    parser.add_argument(
        "--marker", action="store_true",
        help="Also run Marker (needs 6GB+ RAM, Linux only)"
    )
    parser.add_argument(
        "--chunk", action="store_true",
        help="Run all 4 chunking strategies on Docling output"
    )
    args = parser.parse_args()

    run(
        include_ocr=args.ocr,
        include_marker=args.marker,
        include_chunk=args.chunk
    )