import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk(
    text: str,
    max_tokens: int = 512,
    overlap_sentences: int = 1
) -> List[dict]:
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
            current_sentences = current_sentences[-overlap_sentences:]
            current_length = sum(len(s) for s in current_sentences)

        current_sentences.append(sentence)
        current_length += sentence_length

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