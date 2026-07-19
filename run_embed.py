# ─────────────────────────────────────────────────────────────────────────────
# run_embed.py
# Reads chunk JSON files from results/ folder, converts chunks to
# embeddings using all-MiniLM-L6-v2, and stores everything in
# ChromaDB — a local offline vector database.
#
# Combines semantic and layout chunks for better retrieval coverage:
# - Semantic chunks: good for concept-based questions
# - Layout chunks: good for section-based questions
#
# Usage:
#   python run_embed.py
#   python run_embed.py --pdf yourfile.pdf
#   python run_embed.py --strategy semantic
#   python run_embed.py --strategy layout
#   python run_embed.py --strategy both (default)
# ─────────────────────────────────────────────────────────────────────────────

import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

console = Console()

RESULTS_DIR = Path("results")
VECTOR_STORE_DIR = Path("vector_store")
VECTOR_STORE_DIR.mkdir(exist_ok=True)


def load_chunks_from_json(json_path: Path) -> list:
    """Load chunks from a JSON file saved by run_extraction.py"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("chunks", [])


def get_chunk_files(pdf_stem: str, strategy: str) -> list:
    """
    Find chunk JSON files for a given PDF and strategy.

    strategy options:
    - semantic: only semantic chunks
    - layout: only layout chunks
    - both: semantic + layout combined (recommended)
    - all: all 4 chunkers
    """
    pattern_map = {
        "semantic": [f"*{pdf_stem}*semantic_chunks.json"],
        "layout": [f"*{pdf_stem}*layout_chunks.json"],
        "both": [
            f"*{pdf_stem}*semantic_chunks.json",
            f"*{pdf_stem}*layout_chunks.json"
        ],
        "all": [f"*{pdf_stem}*_chunks.json"]
    }

    patterns = pattern_map.get(strategy, pattern_map["both"])
    files = []
    for pattern in patterns:
        files.extend(list(RESULTS_DIR.glob(pattern)))

    return list(set(files))  # deduplicate


def store_in_chromadb(
    chunks: list,
    collection_name: str,
    pdf_name: str,
    chunker_name: str
):
    """
    Store chunks and their embeddings in ChromaDB.

    ChromaDB stores together:
    - id: unique identifier for each chunk
    - document: the actual text content of the chunk
    - embedding: 384-dim vector representation
    - metadata: chunk_index, chunker, pdf name, heading etc.

    Args:
        chunks: list of chunk dicts from JSON files
        collection_name: ChromaDB collection to store in
        pdf_name: source PDF filename
        chunker_name: which chunker produced these chunks
    """
    import chromadb
    from utils.embedder import embed_texts

    # Initialize ChromaDB with local persistent storage
    # PersistentClient saves to disk — works offline forever
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    # Get or create collection
    # distance_function: cosine similarity (best for normalized embeddings)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    if not chunks:
        console.print(f"[red]No chunks to store for {chunker_name}[/red]")
        return 0

    # Prepare data for ChromaDB
    texts = [chunk["content"] for chunk in chunks]
    ids = [
        f"{Path(pdf_name).stem}__{chunker_name}__chunk_{chunk['chunk_index']}"
        for chunk in chunks
    ]
    metadatas = [
        {
            "chunk_index": chunk.get("chunk_index", i),
            "chunker": chunk.get("chunker", chunker_name),
            "pdf_name": pdf_name,
            "char_count": chunk.get("char_count", 0),
            "token_estimate": chunk.get("token_estimate", 0),
            "heading": chunk.get("heading", ""),
            "element_type": chunk.get("element_type", "text"),
        }
        for i, chunk in enumerate(chunks)
    ]

    console.print(
        f"[yellow]Embedding {len(texts)} chunks "
        f"from {chunker_name}...[/yellow]"
    )

    # Generate embeddings
    embeddings = embed_texts(texts)

    # Store in ChromaDB — chunks + embeddings + metadata all together
    # ChromaDB handles deduplication by ID
    # If same chunk ID exists — it updates it
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    console.print(
        f"[green]✓ Stored {len(texts)} chunks "
        f"in collection '{collection_name}'[/green]"
    )

    return len(texts)


def run_embed(pdf_filter: str = None, strategy: str = "both"):
    """
    Main function — reads chunk files and stores in ChromaDB.

    Args:
        pdf_filter: optional PDF filename to process only one PDF
        strategy: which chunks to embed (semantic/layout/both/all)
    """
    console.print(Panel(
        f"[bold cyan]Embedding Chunks → ChromaDB[/bold cyan]\n"
        f"Strategy: {strategy}",
        title="EMBEDDING"
    ))

    # Find all chunk summary files to get list of processed PDFs
    summary_files = list(RESULTS_DIR.glob("chunk_summary__*.json"))

    if not summary_files:
        console.print(
            "[red]No chunk summaries found. "
            "Run run_extraction.py --chunk first.[/red]"
        )
        return

    total_stored = 0

    for summary_file in summary_files:
        # Get PDF name from summary file
        pdf_stem = summary_file.stem.replace("chunk_summary__", "")

        # Filter to specific PDF if requested
        if pdf_filter and pdf_filter not in pdf_stem:
            continue

        console.print(f"\n[bold]Processing: {pdf_stem}[/bold]")

        # Find chunk files matching strategy
        chunk_files = get_chunk_files(pdf_stem, strategy)

        if not chunk_files:
            console.print(
                f"[red]No chunk files found for {pdf_stem} "
                f"with strategy '{strategy}'[/red]"
            )
            continue

        # Use PDF stem as collection name
        # One collection per PDF keeps things organized
        collection_name = pdf_stem[:50].replace(
            " ", "_"
        ).replace("-", "_").lower()

        for chunk_file in chunk_files:
            # Determine chunker name from filename
            if "semantic" in chunk_file.name:
                chunker_name = "semantic"
            elif "layout" in chunk_file.name:
                chunker_name = "layout"
            elif "naive" in chunk_file.name:
                chunker_name = "naive"
            elif "table" in chunk_file.name:
                chunker_name = "table"
            else:
                chunker_name = "unknown"

            console.print(
                f"  Loading {chunker_name} chunks from {chunk_file.name}..."
            )
            chunks = load_chunks_from_json(chunk_file)

            stored = store_in_chromadb(
                chunks=chunks,
                collection_name=collection_name,
                pdf_name=pdf_stem,
                chunker_name=chunker_name
            )
            total_stored += stored

    console.print(
        f"\n[bold green]Done! Total chunks stored: {total_stored}[/bold green]"
    )
    console.print(
        f"[bold]Vector store saved to: {VECTOR_STORE_DIR}/[/bold]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed chunks and store in ChromaDB vector database"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Filter to specific PDF filename"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["semantic", "layout", "both", "all"],
        help="Which chunks to embed (default: both)"
    )
    args = parser.parse_args()

    run_embed(
        pdf_filter=args.pdf,
        strategy=args.strategy
    )