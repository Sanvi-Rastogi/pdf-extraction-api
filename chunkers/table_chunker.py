from typing import List


def extract_markdown_tables(content: str) -> List[dict]:
    """Extract Markdown tables from Docling output as atomic chunks."""
    chunks = []
    chunk_index = 0
    lines = content.split("\n")
    i = 0
    current_heading = ""

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("#"):
            current_heading = line.lstrip("#").strip()
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            table_content = "\n".join(table_lines)
            chunk_content = ""
            if current_heading:
                chunk_content += f"Section: {current_heading}\n\n"
            chunk_content += table_content

            chunks.append({
                "chunk_index": chunk_index,
                "content": chunk_content,
                "heading": current_heading,
                "char_count": len(chunk_content),
                "token_estimate": len(chunk_content) // 4,
                "element_type": "Table",
                "chunker": "table",
            })
            chunk_index += 1
        else:
            i += 1

    return chunks


def extract_non_table_text(content: str) -> str:
    """Remove all Markdown table blocks from content."""
    lines = content.split("\n")
    non_table_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|"):
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
        else:
            non_table_lines.append(lines[i])
            i += 1

    return "\n".join(non_table_lines).strip()


def chunk(content: str, max_tokens: int = 512) -> List[dict]:
    """
    Chunk Docling markdown with table preservation.
    Tables are atomic chunks. Non-table text uses naive chunker.
    """
    from chunkers.naive_chunker import chunk as naive_chunk

    table_chunks = extract_markdown_tables(content)
    non_table_text = extract_non_table_text(content)
    text_chunks = naive_chunk(non_table_text, max_tokens=max_tokens)

    for c in text_chunks:
        c["element_type"] = "text"

    all_chunks = table_chunks + text_chunks
    for i, c in enumerate(all_chunks):
        c["chunk_index"] = i

    return all_chunks