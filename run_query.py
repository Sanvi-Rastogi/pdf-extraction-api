import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

VECTOR_STORE_DIR = Path("vector_store")
RESULTS_DIR = Path("results")


def query_chromadb(
    question: str,
    collection_name: str = None,
    top_k: int = 4
) -> list:

    import chromadb
    from utils.embedder import embed_query

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    if collection_name:
        collections = [client.get_collection(collection_name)]
    else:
        collection_names = [c.name for c in client.list_collections()]
        if not collection_names:
            console.print(
                "[red]No collections found. "
                "Run run_embed.py first.[/red]"
            )
            return []
        collections = [
            client.get_collection(name) for name in collection_names
        ]

    console.print(f"[yellow]Embedding question...[/yellow]")
    question_embedding = embed_query(question)

    all_results = []

    for collection in collections:
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]

            similarity = round(1 - (distance / 2), 4)

            all_results.append({
                "rank": i + 1,
                "content": doc,
                "similarity": similarity,
                "chunker": metadata.get("chunker", "unknown"),
                "heading": metadata.get("heading", ""),
                "pdf_name": metadata.get("pdf_name", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "char_count": metadata.get("char_count", 0),
            })

    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    return all_results[:top_k]


def display_results(question: str, results: list):

    console.print(Panel(
        f"[bold cyan]{question}[/bold cyan]",
        title="QUERY"
    ))

    if not results:
        console.print("[red]No results found.[/red]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", width=6)
    table.add_column("Similarity", width=12)
    table.add_column("Chunker", width=10)
    table.add_column("Heading", width=30)
    table.add_column("Chars", width=8)

    for r in results:
        table.add_row(
            str(r["rank"]),
            str(r["similarity"]),
            r["chunker"],
            r["heading"][:28] if r["heading"] else "-",
            str(r["char_count"])
        )

    console.print(table)

    console.print("\n[bold]Retrieved Chunks:[/bold]\n")

    for r in results:
        console.print(
            f"[bold cyan]── Rank {r['rank']} "
            f"(similarity: {r['similarity']}, "
            f"chunker: {r['chunker']})[/bold cyan]"
        )
        if r["heading"]:
            console.print(f"[yellow]Section: {r['heading']}[/yellow]")
        console.print(r["content"])
        console.print()

    output_path = RESULTS_DIR / "last_query_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "question": question,
            "results": results
        }, f, indent=2)

    console.print(
        f"[bold]Results saved to {output_path}[/bold]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query ChromaDB vector store with a question"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Specific collection to search (default: all)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Number of chunks to retrieve (default: 4)"
    )
    args = parser.parse_args()

    results = query_chromadb(
        question=args.question,
        collection_name=args.collection,
        top_k=args.top_k
    )

    display_results(args.question, results)