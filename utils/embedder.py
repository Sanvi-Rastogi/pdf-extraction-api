# ─────────────────────────────────────────────────────────────────────────────
# embedder.py
# Utility for converting text chunks into embeddings using
# the locally cached all-MiniLM-L6-v2 sentence transformer model.
# Runs completely offline — model loaded from Docker cache.
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
from typing import List


def get_embedding_model():
    """
    Load sentence transformer from local Docker cache.
    local_files_only=True ensures no internet calls.
    Same model used in semantic chunker for consistency.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME",
            "/root/.cache/huggingface"
        ),
        local_files_only=True
    )
    return model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Convert list of text strings to embedding vectors.

    Args:
        texts: list of text strings to embed

    Returns:
        numpy array of shape (len(texts), 384)
        each row is a 384-dimensional embedding vector
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,          # process 32 chunks at a time
        normalize_embeddings=True  # normalize for cosine similarity
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for retrieval.
    Uses same model as chunk embeddings for consistency.
    """
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    return embedding[0]