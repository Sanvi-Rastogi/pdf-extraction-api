import os
import numpy as np
from typing import List


def get_embedding_model():
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
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,          
        normalize_embeddings=True 
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    model = get_embedding_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    return embedding[0]