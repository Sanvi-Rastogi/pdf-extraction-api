import time
from utils.memory_tracker import get_memory_snapshot, measure_memory_delta


def extract(file_path: str) -> dict:
    loader_name = "Docling (IBM)"
    try:
        from docling.document_converter import (
            DocumentConverter, PdfFormatOption
        )
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        mem_before = get_memory_snapshot()
        start = time.time()

        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
            }
        )

        result = converter.convert(file_path)
        content = result.document.export_to_markdown()
        elapsed = round(time.time() - start, 2)
        mem_after = get_memory_snapshot()

        lines = content.strip().split("\n")
        non_empty = [l for l in lines if l.strip()]

        return {
            "loader": loader_name,
            "status": "success",
            "pages": "N/A",
            "time_sec": elapsed,
            "total_chars": len(content),
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty),
            "has_table_structure": "|" in content,
            "has_numbers": any(c.isdigit() for c in content),
            "images_detected": 0,
            "has_headers": any(l.startswith("#") for l in lines),
            "content": content,
            "memory": measure_memory_delta(mem_before, mem_after),
        }
    except Exception as e:
        return {"loader": loader_name, "status": "failed", "error": str(e)}