# ─────────────────────────────────────────────────────────────────────────────
# image_reader.py
# Extracts embedded images from PDFs using PyMuPDF and sends them
# to a locally running LLaVA vision model via Ollama for description.
# Runs completely offline after initial model download.
# ─────────────────────────────────────────────────────────────────────────────

import os
import base64
import requests
from typing import List


# Get Ollama host from environment — defaults to localhost
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def extract_images_from_pdf(file_path: str) -> List[dict]:
    """
    Extract all embedded images from PDF using PyMuPDF.
    Returns list of images with page number and base64 encoded bytes.
    """
    import fitz

    doc = fitz.open(file_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Skip very small images (likely icons or artifacts)
                if len(image_bytes) < 5000:
                    continue

                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                images.append({
                    "page": page_num + 1,
                    "image_index": img_index + 1,
                    "extension": image_ext,
                    "image_b64": image_b64,
                    "size_bytes": len(image_bytes)
                })
            except Exception:
                continue

    doc.close()
    return images


def describe_image(image_b64: str, page: int, index: int) -> str:
    """
    Send image to LLaVA vision model running in Ollama.
    Returns text description of what the image contains.
    Runs completely offline.
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "llava",
                "prompt": (
                    "You are analyzing a figure from an academic paper. "
                    "Describe this image in detail. "
                    "If it contains an architecture diagram, explain the components and connections. "
                    "If it contains a chart or graph, describe the axes, data, and trends. "
                    "If it contains equations or text, transcribe them exactly. "
                    "If it contains a flowchart, explain the flow step by step. "
                    "Be thorough and precise."
                ),
                "images": [image_b64],
                "stream": False
            },
            timeout=120  # vision inference can be slow on CPU
        )

        print(response.status_code)
        print(response.text)

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"[Image description failed: HTTP {response.status_code}]"

    except requests.exceptions.ConnectionError:
        return "[Ollama not running — start with: docker-compose up ollama -d]"
    except Exception as e:
        return f"[Image description error: {e}]"


def extract_and_describe_images(file_path: str) -> str:
    """
    Main function — extracts all images from PDF and
    returns their descriptions as a formatted string
    ready to append to Docling's Markdown output.
    """
    images = extract_images_from_pdf(file_path)
    print("Found images:", len(images))

    if not images:
        return ""

    descriptions = []
    descriptions.append("\n\n## Extracted Image Content\n")

    for img in images:
        print(
            f"  Describing image {img['image_index']} "
            f"on page {img['page']}..."
        )

        description = describe_image(
            img["image_b64"],
            img["page"],
            img["image_index"]
        )

        print("Description:")
        print(description)
        print("----------------")

        descriptions.append(
            f"\n### Figure {img['image_index']} (Page {img['page']})\n\n"
            f"{description}\n"
        )

    print("Descriptions length:", len(descriptions))
    print("Final string length:", len("\n".join(descriptions)))

    return "\n".join(descriptions)