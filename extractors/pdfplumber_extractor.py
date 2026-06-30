import time
from utils.memory_tracker import get_memory_snapshot, measure_memory_delta


def extract(file_path: str) -> dict:
    loader_name = "pdfplumber"
    try:
        import pdfplumber

        mem_before = get_memory_snapshot()
        start = time.time()

        all_content = []
        page_count = 0

        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""

                tables = page.extract_tables()
                table_text = ""
                for t_idx, table in enumerate(tables):
                    table_text += f"\n[TABLE {t_idx+1} - Page {page_num+1}]\n"
                    for row in table:
                        cleaned = [
                            str(c).strip() if c else "" for c in row
                        ]
                        table_text += " | ".join(cleaned) + "\n"

                all_content.append(
                    f"--- Page {page_num+1} ---\n{page_text}\n{table_text}"
                )

        elapsed = round(time.time() - start, 2)
        mem_after = get_memory_snapshot()
        content = "\n\n".join(all_content)

        return {
            "loader": loader_name,
            "status": "success",
            "pages": page_count,
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