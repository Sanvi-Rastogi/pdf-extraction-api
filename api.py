import os
import uuid
import shutil
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

from extractors import docling_extractor
from chunkers import (
    naive_chunker,
    layout_chunker,
    table_chunker,
    semantic_chunker,
)

app = FastAPI(title="PDF Extraction API", version="1.0.0")

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
VECTOR_STORE_DIR = Path("vector_store")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)


def save_upload(file: UploadFile) -> str:
    path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return str(path)


def save_result(result: dict, filename: str):
    safe = Path(filename).stem
    with open(RESULTS_DIR / f"{safe}__docling.json", "w") as f:
        json.dump(
            {k: v for k, v in result.items() if k != "content"},
            f, indent=2
        )
    if result.get("status") == "success" and "content" in result:
        with open(
            RESULTS_DIR / f"{safe}__docling.txt", "w", encoding="utf-8"
        ) as f:
            f.write(result["content"])


def save_chunks(chunks: list, filename: str, chunker: str):
    safe = Path(filename).stem
    out = RESULTS_DIR / f"{safe}__docling__{chunker}_chunks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "file": filename,
            "chunker": chunker,
            "total_chunks": len(chunks),
            "chunks": chunks
        }, f, indent=2)


def run_chunker(content: str, strategy: str) -> dict:
    chunkers = {
        "naive": lambda c: naive_chunker.chunk(c, max_tokens=512),
        "layout": lambda c: layout_chunker.chunk(c, max_tokens=512),
        "table": lambda c: table_chunker.chunk(c, max_tokens=512),
        "semantic": lambda c: semantic_chunker.chunk(
            c, max_tokens=512, similarity_threshold=0.3
        ),
    }

    if strategy not in chunkers:
        return {"error": f"Unknown strategy '{strategy}'"}

    try:
        chunks = chunkers[strategy](content)
        return {
            "strategy": strategy,
            "total_chunks": len(chunks),
            "avg_chars": round(
                sum(c["char_count"] for c in chunks) / max(len(chunks), 1),
                1
            ),
            "avg_tokens": round(
                sum(c["token_estimate"] for c in chunks) /
                max(len(chunks), 1),
                1
            ),
            "chunks": chunks
        }
    except Exception as e:
        return {"strategy": strategy, "error": str(e)}


@app.get("/")
def root():
    return {
        "endpoints": {
            "POST /extract": "Extract PDF with Docling",
            "POST /chunk":   "Extract + chunk (choose strategy)",
            "POST /embed":   "Embed chunks into ChromaDB",
            "POST /query":   "Query vector database",
            "GET  /health":  "System status",
        },
        "chunk_strategies": [
            "naive",
            "layout",
            "table",
            "semantic",
            "both (semantic + layout)",
            "all (all 4 strategies)"
        ]
    }


@app.get("/health")
def health():
    import psutil
    vm = psutil.virtual_memory()

    chroma_info = "empty"
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        cols = client.list_collections()
        total = sum(client.get_collection(c.name).count() for c in cols)
        chroma_info = f"{len(cols)} collections, {total} chunks"
    except Exception as e:
        chroma_info = f"error: {e}"

    return {
        "status": "running",
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "chromadb": chroma_info,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files supported"}
        )

    file_path = save_upload(file)
    try:
        result = docling_extractor.extract(file_path)
        save_result(result, file.filename)

        return JSONResponse(content={
            "file": file.filename,
            "status": result.get("status"),
            "total_chars": result.get("total_chars"),
            "word_count": len(result.get("content", "").split()),
            "has_tables": result.get("has_table_structure"),
            "has_headers": result.get("has_headers"),
            "images_detected": result.get("images_detected"),
            "time_sec": result.get("time_sec"),
            "memory": result.get("memory"),
            "saved_to": f"results/{Path(file.filename).stem}__docling.txt"
        })
    finally:
        os.remove(file_path)


@app.post("/chunk")
async def chunk(
    file: UploadFile = File(...),
    strategy: str = Query(
        "both",
        description=(
            "Chunking strategy: "
            "naive | layout | table | semantic | both | all"
        )
    )
):
    
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Only PDF files supported"}
        )

    file_path = save_upload(file)
    try:
        result = docling_extractor.extract(file_path)
        save_result(result, file.filename)

        if result.get("status") != "success":
            return JSONResponse(content={
                "file": file.filename,
                "extraction": "failed",
                "error": result.get("error")
            })

        content = result.get("content", "")

        if strategy == "both":
            strategies = ["semantic", "layout"]
        elif strategy == "all":
            strategies = ["naive", "layout", "table", "semantic"]
        else:
            strategies = [strategy]

        chunk_results = {}
        for s in strategies:
            chunk_data = run_chunker(content, s)
            save_chunks(
                chunk_data.get("chunks", []),
                file.filename,
                s
            )
            chunk_results[s] = {
                k: v for k, v in chunk_data.items()
                if k != "chunks"
            }

        return JSONResponse(content={
            "file": file.filename,
            "extraction": {
                "total_chars": result.get("total_chars"),
                "word_count": len(content.split()),
                "has_tables": result.get("has_table_structure"),
                "has_headers": result.get("has_headers"),
                "time_sec": result.get("time_sec"),
            },
            "chunking": chunk_results,
            "saved_to": "results/ folder"
        })
    finally:
        os.remove(file_path)


@app.post("/embed")
async def embed(
    pdf_stem: str = Query(
        ...,
        description="PDF filename without extension e.g. my_paper"
    ),
    strategy: str = Query(
        "both",
        description="Which chunks to embed: naive|layout|table|semantic|both|all"
    )
):
    import chromadb
    from utils.embedder import embed_texts

    if strategy == "both":
        patterns = [
            f"*{pdf_stem}*semantic_chunks.json",
            f"*{pdf_stem}*layout_chunks.json"
        ]
    elif strategy == "all":
        patterns = [f"*{pdf_stem}*_chunks.json"]
    else:
        patterns = [f"*{pdf_stem}*{strategy}_chunks.json"]

    chunk_files = []
    for p in patterns:
        chunk_files.extend(list(RESULTS_DIR.glob(p)))
    chunk_files = list(set(chunk_files))

    if not chunk_files:
        return JSONResponse(
            status_code=404,
            content={
                "error": (
                    f"No chunk files found for '{pdf_stem}' "
                    f"with strategy '{strategy}'. "
                    f"Run /chunk first."
                )
            }
        )

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection_name = pdf_stem[:50].replace(
        " ", "_"
    ).replace("-", "_").lower()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    stored = []
    for chunk_file in chunk_files:
        chunker_name = "unknown"
        for name in ["semantic", "layout", "naive", "table"]:
            if name in chunk_file.name:
                chunker_name = name
                break

        with open(chunk_file) as f:
            data = json.load(f)
        chunks = data.get("chunks", [])
        if not chunks:
            continue

        texts = [c["content"] for c in chunks]
        ids = [
            f"{pdf_stem}__{chunker_name}__chunk_{c['chunk_index']}"
            for c in chunks
        ]
        metadatas = [{
            "chunk_index": c.get("chunk_index", i),
            "chunker": chunker_name,
            "pdf_name": pdf_stem,
            "char_count": c.get("char_count", 0),
            "heading": str(c.get("heading", "")),
        } for i, c in enumerate(chunks)]

        embeddings = embed_texts(texts)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        stored.append({
            "chunker": chunker_name,
            "chunks_stored": len(texts)
        })

    return JSONResponse(content={
        "collection": collection_name,
        "total_stored": sum(s["chunks_stored"] for s in stored),
        "details": stored
    })


@app.post("/query")
async def query(
    question: str = Query(..., description="Your question"),
    top_k: int = Query(4, description="Number of results"),
    collection: Optional[str] = Query(
        None, description="Collection name (None = search all)"
    )
):
    import chromadb
    from utils.embedder import embed_query

    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    if collection:
        try:
            collections = [client.get_collection(collection)]
        except Exception:
            return JSONResponse(
                status_code=404,
                content={"error": f"Collection '{collection}' not found"}
            )
    else:
        cols = client.list_collections()
        if not cols:
            return JSONResponse(
                status_code=404,
                content={"error": "No collections found. Run /embed first."}
            )
        collections = [client.get_collection(c.name) for c in cols]

    question_embedding = embed_query(question)
    all_results = []

    for col in collections:
        if col.count() == 0:
            continue
        res = col.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"]
        )
        for i, doc in enumerate(res["documents"][0]):
            meta = res["metadatas"][0][i]
            similarity = round(1 - (res["distances"][0][i] / 2), 4)
            all_results.append({
                "content": doc,
                "similarity": similarity,
                "chunker": meta.get("chunker"),
                "heading": meta.get("heading", ""),
                "pdf_name": meta.get("pdf_name"),
            })

    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    return JSONResponse(content={
        "question": question,
        "results": all_results[:top_k]
    })