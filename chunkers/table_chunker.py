import re
from typing import List


def extract_tables_from_pdfplumber(content: str) -> List[dict]:
    """
    Parse pdfplumber output which marks tables as:
    [TABLE N on page M]
    col1 | col2 | col3
    val1 | val2 | val3

    Returns each table as a separate chunk with page context.
    """
    chunks = []
    chunk_index = 0

    # Split content by page
    pages = re.split(r'--- Page \d+ ---', content)

    for page_num, page_content in enumerate(pages):
        if not page_content.strip():
            continue

        # Find all tables on this page
        table_pattern = re.compile(
            r'\[TABLE (\d+) on page (\d+)\](.*?)(?=\[TABLE|\Z)',
            re.DOTALL
        )

        # Get text before first table (potential heading/context)
        first_table_match = re.search(r'\[TABLE', page_content)
        pre_table_text = ""
        if first_table_match:
            pre_table_text = page_content[:first_table_match.start()].strip()

        for match in table_pattern.finditer(page_content):
            table_num = match.group(1)
            page_ref = match.group(2)
            table_content = match.group(3).strip()

            # Build chunk: context + table
            # Including surrounding text helps RAG understand what table is about
            chunk_content = ""
            if pre_table_text:
                chunk_content += f"Context: {pre_table_text[:200]}\n\n"
            chunk_content += f"[TABLE {table_num} - Page {page_ref}]\n"
            chunk_content += table_content

            chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_content,
                "char_count": len(chunk_content),
                "token_estimate": len(chunk_content) // 4,
                "element_type": "Table",
                "page": int(page_ref),
                "table_number": int(table_num),
                "chunker": "table",
            })
            chunk_index += 1

    return chunks


def extract_non_table_text(content: str) -> str:
    """Extract only the non-table text for separate chunking."""
    # Remove table blocks
    cleaned = re.sub(
        r'\[TABLE \d+ on page \d+\].*?(?=\[TABLE|\Z)',
        '',
        content,
        flags=re.DOTALL
    )
    # Remove page separators
    cleaned = re.sub(r'--- Page \d+ ---', '', cleaned)
    return cleaned.strip()


def chunk(content: str, max_tokens: int = 512) -> List[dict]:
    """
    Chunk content with special handling for tables.
    Inspired by RAGFlow's table chunking strategy.

    Strategy:
    1. Extract all tables as atomic chunks with surrounding context
    2. Extract non-table text separately
    3. Return combined list with tables preserved intact

    Args:
        content: extracted text from pdfplumber (has [TABLE N] markers)
        max_tokens: max tokens for non-table chunks

    Returns:
        list of chunks — tables are atomic, text is split by sentences
    """
    from chunkers.naive_chunker import chunk as naive_chunk

    # Get table chunks (atomic — never split)
    table_chunks = extract_tables_from_pdfplumber(content)

    # Get non-table text and chunk it normally
    non_table_text = extract_non_table_text(content)
    text_chunks = naive_chunk(non_table_text, max_tokens=max_tokens)

    # Tag text chunks
    for c in text_chunks:
        c["element_type"] = "text"

    # Merge and re-index
    all_chunks = table_chunks + text_chunks
    for i, c in enumerate(all_chunks):
        c["chunk_index"] = i

    return all_chunks