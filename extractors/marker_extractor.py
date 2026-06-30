import time
from utils.memory_tracker import get_memory_snapshot, measure_memory_delta


def extract(file_path: str) -> dict:
    loader_name = "Marker (Datalab)"
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
        from marker.output import text_from_rendered

        mem_before = get_memory_snapshot()
        start = time.time()

        config = {
            "output_format": "markdown",
            "use_llm": False,
            "force_ocr": False,
            "workers": 1,
            "batch_multiplier": 1,
            "torch_device": "cpu",
        }

        config_parser = ConfigParser(config)
        model_dict = create_model_dict()

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        rendered = converter(file_path)
        content, _, images = text_from_rendered(rendered)
        elapsed = round(time.time() - start, 2)
        mem_after = get_memory_snapshot()

        lines = content.strip().split("\n")
        non_empty = [l for l in lines if l.strip()]

        return {
            "loader": loader_name,
            "status": "success",
            "time_sec": elapsed,
            "total_chars": len(content),
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty),
            "has_table_structure": "|" in content,
            "has_headers": any(l.startswith("#") for l in lines),
            "has_equations": "$$" in content or "\\[" in content,
            "images_extracted": len(images) if images else 0,
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