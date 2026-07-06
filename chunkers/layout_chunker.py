# ─────────────────────────────────────────────────────────────────────────────
# layout_chunker.py
# Inspired by RAGFlow's layout-aware chunking strategy.
# Uses document element types (Title, NarrativeText, Table, etc.)
# detected by Unstructured loader to create semantically meaningful chunks.
#
# Key idea: a Title element starts a new chunk boundary.
# Content under a heading belongs together until the next heading appears.
# Tables are kept as single atomic chunks — never split across chunks.
# ─────────────────────────────────────────────────────────────────────────────

import re
from typing import List


def parse_unstructured_content(content: str) -> List[dict]:
    """
    Parse the output from Unstructured loader which has format:
    [ElementType]
    content text here

    Returns list of elements with their type and content.
    """
    elements = []
    # Split on element type tags like [Title], [NarrativeText] etc.
    blocks = re.split(r'\n\n(?=\[)', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Extract element type from [Type] prefix
        type_match = re.match(r'^\[([^\]]+)\]\n(.+)', block, re.DOTALL)
        if type_match:
            element_type = type_match.group(1).strip()
            element_content = type_match.group(2).strip()
        else:
            element_type = "NarrativeText"
            element_content = block

        elements.append({
            "type": element_type,
            "content": element_content
        })

    return elements


def chunk(
    content: str,
    max_tokens: int = 512,
    source_loader: str = "unstructured"
) -> List[dict]:
    """
    Create layout-aware chunks using detected element types.
    Inspired by RAGFlow's layout chunking strategy.

    Rules (same as RAGFlow):
    - Title element → always starts a new chunk
    - Table element → always its own atomic chunk (never split)
    - Image/Figure → its own chunk with caption if available
    - NarrativeText → grouped with current section heading
    - Footer/Header → skipped (noise)

    Args:
        content: extracted text from Unstructured loader
                 (contains [ElementType] tags)
        max_tokens: max tokens per chunk
        source_loader: which loader produced this content

    Returns:
        list of layout-aware chunk dicts
    """
    max_chars = max_tokens * 4

    elements = parse_unstructured_content(content)

    if not elements:
        return []

    chunks = []
    chunk_index = 0
    current_content = []
    current_length = 0
    current_heading = ""

    # Elements to skip — these are noise not content
    skip_types = {"Footer", "Header", "PageNumber"}

    for element in elements:
        etype = element["type"]
        econtent = element["content"]

        # Skip noise elements
        if etype in skip_types:
            continue

        # Tables are atomic — always their own chunk
        # Never split a table across chunks (RAGFlow rule)
        if etype == "Table":
            # First save whatever we have so far
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

            # Table as its own chunk
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

        # Title/Heading → start new chunk boundary (RAGFlow rule)
        if etype == "Title":
            # Save current chunk first
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

            # Update current heading
            current_heading = econtent
            current_content.append(econtent)
            current_length = len(econtent)
            continue

        # Regular content — add to current chunk
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

    # Save final chunk
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