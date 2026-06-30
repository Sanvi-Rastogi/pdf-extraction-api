import time
from utils.memory_tracker import get_memory_snapshot, measure_memory_delta


def extract(file_path: str, strategy: str = "fast") -> dict:
    loader_name = f"Unstructured ({strategy})"
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader

        mem_before = get_memory_snapshot()
        start = time.time()

        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy=strategy
        )
        docs = loader.load()
        elapsed = round(time.time() - start, 2)
        mem_after = get_memory_snapshot()

        # Count element types
        element_types = {}
        for doc in docs:
            etype = doc.metadata.get("category", "Unknown")
            element_types[etype] = element_types.get(etype, 0) + 1

        content = "\n\n".join(
            f"[{d.metadata.get('category', 'Text')}]\n{d.page_content}"
            for d in docs
        )

        pages = len(set(
            d.metadata.get("page_number", 0) for d in docs
        ))

        return {
            "loader": loader_name,
            "status": "success",
            "pages": pages,
            "time_sec": elapsed,
            "total_chars": len(content),
            "total_lines": len(content.split("\n")),
            "has_table_structure": "Table" in element_types,
            "images_detected": element_types.get("Image", 0),
            "element_types": element_types,
            "has_numbers": any(c.isdigit() for c in content),
            "content": content,
            "memory": measure_memory_delta(mem_before, mem_after),
        }
    except Exception as e:
        return {
            "loader": loader_name,
            "status": "failed",
            "error": str(e)
        }