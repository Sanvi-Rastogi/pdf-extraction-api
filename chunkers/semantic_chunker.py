import re
from typing import List
import numpy as np


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences at punctuation boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def get_embeddings(sentences: List[str]) -> np.ndarray:
    """
    Convert sentences to embedding vectors using HuggingFace model.
    Uses the same model as your Smart Academic Assistant for consistency.
    """
    from sentence_transformers import SentenceTransformer

    # Use lightweight model for speed
    # Same model family as your RAG project
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=False)
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    )


def find_breakpoints(
    embeddings: np.ndarray,
    threshold: float = 0.3
) -> List[int]:
    """
    Find indices where topic shifts occur.

    How it works:
    1. Calculate similarity between each consecutive pair of sentences
    2. Where similarity drops below threshold → topic shift → chunk boundary
    3. Return list of sentence indices where new chunks should start

    Args:
        embeddings: sentence embedding matrix
        threshold: similarity below this = new chunk
                   lower = more chunks, higher = fewer larger chunks
    """
    breakpoints = [0]  # always start with index 0

    for i in range(1, len(embeddings)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])

        # If similarity drops sharply, this is a topic shift
        if similarity < threshold:
            breakpoints.append(i)

    return breakpoints


def chunk(
    text: str,
    max_tokens: int = 512,
    similarity_threshold: float = 0.3
) -> List[dict]:
    """
    Create semantically coherent chunks using sentence embeddings.
    Inspired by RAGFlow's semantic chunking strategy.

    How it works:
    1. Split text into sentences
    2. Embed all sentences using MiniLM model
    3. Find where semantic similarity drops (topic shifts)
    4. Those drop points become chunk boundaries
    5. Merge small adjacent chunks if they're under token limit

    Args:
        text: full extracted text
        max_tokens: max tokens per chunk
        similarity_threshold: cosine similarity below this triggers new chunk

    Returns:
        list of semantically coherent chunk dicts
    """
    max_chars = max_tokens * 4

    sentences = split_into_sentences(text)

    if len(sentences) < 2:
        return [{
            "chunk_index": 0,
            "content": text,
            "char_count": len(text),
            "token_estimate": len(text) // 4,
            "sentence_count": len(sentences),
            "chunker": "semantic",
            "avg_similarity": None,
        }]

    # Get embeddings for all sentences
    embeddings = get_embeddings(sentences)

    # Find where topic shifts happen
    breakpoints = find_breakpoints(embeddings, threshold=similarity_threshold)

    # Build chunks from breakpoints
    raw_chunks = []
    for i, start in enumerate(breakpoints):
        end = breakpoints[i+1] if i+1 < len(breakpoints) else len(sentences)
        chunk_sentences = sentences[start:end]
        raw_chunks.append(chunk_sentences)

    # Merge chunks that are too small, split chunks that are too large
    final_chunks = []
    chunk_index = 0
    current_sentences = []
    current_length = 0

    for raw_chunk in raw_chunks:
        chunk_text = " ".join(raw_chunk)

        # If adding this chunk would exceed limit — save what we have
        if current_length + len(chunk_text) > max_chars and current_sentences:
            joined = " ".join(current_sentences)
            final_chunks.append({
                "chunk_index": chunk_index,
                "content": joined,
                "char_count": len(joined),
                "token_estimate": len(joined) // 4,
                "sentence_count": len(current_sentences),
                "chunker": "semantic",
            })
            chunk_index += 1
            current_sentences = []
            current_length = 0

        current_sentences.extend(raw_chunk)
        current_length += len(chunk_text)

    # Save final chunk
    if current_sentences:
        joined = " ".join(current_sentences)
        final_chunks.append({
            "chunk_index": chunk_index,
            "content": joined,
            "char_count": len(joined),
            "token_estimate": len(joined) // 4,
            "sentence_count": len(current_sentences),
            "chunker": "semantic",
        })

    return final_chunks