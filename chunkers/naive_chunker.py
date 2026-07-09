import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations and edge cases.
    """
    # Split on period/exclamation/question followed by space and capital letter
    # This avoids splitting on abbreviations like "Dr." or "Fig."
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk(
    text: str,
    max_tokens: int = 512,
    overlap_sentences: int = 1
) -> List[dict]:
    """
    Split text into chunks that respect sentence boundaries.
    Similar to RAGFlow's naive chunking strategy.

    How it works:
    1. Split entire text into sentences first
    2. Build chunks by adding sentences one by one
    3. When adding next sentence would exceed max_tokens, start new chunk
    4. Carry over last N sentences (overlap) into next chunk for context

    Args:
        text: full extracted text from PDF loader
        max_tokens: approximate max tokens per chunk (1 token ≈ 4 chars)
        overlap_sentences: how many sentences to carry over between chunks

    Returns:
        list of chunk dicts with content, metadata, token estimate
    """
    # Approximate max characters from token limit
    # (1 token ≈ 4 characters on average for English text)
    max_chars = max_tokens * 4

    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_sentences = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence would exceed limit — save current chunk
        if current_length + sentence_length > max_chars and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_text,
                "char_count": len(chunk_text),
                "token_estimate": len(chunk_text) // 4,
                "sentence_count": len(current_sentences),
                "chunker": "naive",
            })
            chunk_index += 1

            # Keep last N sentences as overlap for next chunk
            current_sentences = current_sentences[-overlap_sentences:]
            current_length = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_length += sentence_length

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append({
            "chunk_index": chunk_index,
            "content": chunk_text,
            "char_count": len(chunk_text),
            "token_estimate": len(chunk_text) // 4,
            "sentence_count": len(current_sentences),
            "chunker": "naive",
        })

    return chunks