import time
from utils.memory_tracker import get_memory_snapshot, measure_memory_delta


def extract(file_path: str) -> dict:
    loader_name = "PyMuPDF (fitz)"
    try:
        import fitz

        mem_before = get_memory_snapshot()
        start = time.time()

        doc = fitz.open(file_path)
        all_content = []
        total_images = 0
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text("text")
            images = page.get_images(full=True)
            total_images += len(images)
            image_notes = ""
            if images:
                image_notes = (
                    f"\n[{len(images)} IMAGE(S) DETECTED "
                    f"on page {page_num+1}]\n"
                )
            all_content.append(
                f"--- Page {page_num+1} ---\n{text}{image_notes}"
            )

        doc.close()
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
            "has_table_structure": False,
            "has_numbers": any(c.isdigit() for c in content),
            "images_detected": total_images,
            "has_headers": False,
            "content": content,
            "memory": measure_memory_delta(mem_before, mem_after),
        }
    except Exception as e:
        return {"loader": loader_name, "status": "failed", "error": str(e)}