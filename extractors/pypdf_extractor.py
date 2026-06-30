import time
from utils.memory_tracker import get_memory_snapshot, measure_memory_delta


def extract(file_path: str) -> dict:
    loader_name = "PyPDFLoader (Baseline)"
    try:
        from langchain_community.document_loaders import PyPDFLoader

        mem_before = get_memory_snapshot()
        start = time.time()

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        elapsed = round(time.time() - start, 2)

        mem_after = get_memory_snapshot()
        content = "\n\n".join(d.page_content for d in docs)

        return {
            "loader": loader_name,
            "status": "success",
            "pages": len(docs),
            "time_sec": elapsed,
            "total_chars": len(content),
            "total_lines": len(content.split("\n")),
            "has_table_structure": "|" in content,
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