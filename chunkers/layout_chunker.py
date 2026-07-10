import re
from typing import List


def parse_content(content: str) -> List[dict]:
    """
    Parse Docling Markdown output into elements.
    Handles # ## ### headers and | table rows.
    """
    elements = []

    for line in content.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        if line_stripped.startswith("### "):
            elements.append({
                "type": "Title",
                "content": line_stripped[4:],
                "level": 3
            })
        elif line_stripped.startswith("## "):
            elements.append({
                "type": "Title",
                "content": line_stripped[3:],
                "level": 2
            })
        elif line_stripped.startswith("# "):
            elements.append({
                "type": "Title",
                "content": line_stripped[2:],
                "level": 1
            })
        elif line_stripped.startswith("|"):
            elements.append({
                "type": "Table",
                "content": line_stripped,
                "level": 0
            })
        else:
            elements.append({
                "type": "NarrativeText",
                "content": line_stripped,
                "level": 0
            })

    return elements


def chunk(content: str, max_tokens: int = 512) -> List[dict]:
    """
    Layout-aware chunking using Docling Markdown structure.
    Titles start new chunks. Tables are atomic. Text accumulates.
    """
    max_chars = max_tokens * 4
    elements = parse_content(content)

    if not elements:
        return []

    chunks = []
    chunk_index = 0
    current_content = []
    current_length = 0
    current_heading = ""
    skip_types = {"Footer", "Header"}

    for element in elements:
        etype = element["type"]
        econtent = element["content"]

        if etype in skip_types:
            continue

        # tables: atomic chunks
        if etype == "Table":
            if current_content:
                joined = "\n\n".join(current_content)
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": joined,
                    "heading": current_heading,
                    "char_count": len(joined),
                    "token_estimate": len(joined) // 4,
                    "element_type": "mixed",
                    "chunker": "layout",
                })
                chunk_index += 1
                current_content = []
                current_length = 0

            chunks.append({
                "chunk_index": chunk_index,
                "content": econtent,
                "heading": current_heading,
                "char_count": len(econtent),
                "token_estimate": len(econtent) // 4,
                "element_type": "Table",
                "chunker": "layout",
            })
            chunk_index += 1
            continue

        # titles: start new chunks
        if etype == "Title":
            if current_content:
                joined = "\n\n".join(current_content)
                chunks.append({
                    "chunk_index": chunk_index,
                    "content": joined,
                    "heading": current_heading,
                    "char_count": len(joined),
                    "token_estimate": len(joined) // 4,
                    "element_type": "section",
                    "chunker": "layout",
                })
                chunk_index += 1
                current_content = []
                current_length = 0

            current_heading = econtent
            current_content.append(econtent)
            current_length = len(econtent)
            continue

        # regular text
        if current_length + len(econtent) > max_chars and current_content:
            joined = "\n\n".join(current_content)
            chunks.append({
                "chunk_index": chunk_index,
                "content": joined,
                "heading": current_heading,
                "char_count": len(joined),
                "token_estimate": len(joined) // 4,
                "element_type": "section",
                "chunker": "layout",
            })
            chunk_index += 1
            current_content = [current_heading] if current_heading else []
            current_length = len(current_heading)

        current_content.append(econtent)
        current_length += len(econtent)

    if current_content:
        joined = "\n\n".join(current_content)
        chunks.append({
            "chunk_index": chunk_index,
            "content": joined,
            "heading": current_heading,
            "char_count": len(joined),
            "token_estimate": len(joined) // 4,
            "element_type": "section",
            "chunker": "layout",
        })

    return chunks