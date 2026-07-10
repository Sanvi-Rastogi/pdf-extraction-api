import re
from typing import List
import numpy as np


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def get_embeddings(sentences: List[str]) -> np.ndarray:
    """
    Load sentence transformer from Docker cached models only.
    No internet required.
    """
    from sentence_transformers import SentenceTransformer
    import os

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME",
            "/root/.cache/huggingface"
        ),
        local_files_only=True
    )

    return model.encode(sentences, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    )


def find_breakpoints(
    embeddings: np.ndarray,
    threshold: float = 0.3
) -> List[int]:
    breakpoints = [0]
    for i in range(1, len(embeddings)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < threshold:
            breakpoints.append(i)
    return breakpoints


def chunk(
    text: str,
    max_tokens: int = 512,
    similarity_threshold: float = 0.3
) -> List[dict]:
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
        }]

    embeddings = get_embeddings(sentences)
    breakpoints = find_breakpoints(embeddings, threshold=similarity_threshold)

    raw_chunks = []
    for i, start in enumerate(breakpoints):
        end = breakpoints[i+1] if i+1 < len(breakpoints) else len(sentences)
        raw_chunks.append(sentences[start:end])

    final_chunks = []
    chunk_index = 0
    current_sentences = []
    current_length = 0

    for raw_chunk in raw_chunks:
        chunk_text = " ".join(raw_chunk)
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