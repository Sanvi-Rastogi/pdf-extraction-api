import os
import base64
import requests
from io import BytesIO

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "moondream")

PROMPT = """
Describe this figure in 2-3 sentences.

Include:
- diagrams and component relationships
- charts and trends
- tables
- equations or readable text
- the main idea

Be concise but complete.
"""

def describe_image(image_b64: str, page: int, index: int) -> str:
    """
    Send image to the vision model running in Ollama.
    Returns a description of the image.
    Runs completely offline.
    """
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Describe this image.",
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_predict": 80
                }
            },
            timeout=300
        )

        if response.status_code != 200:
            print(response.text)

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"[Image description failed: HTTP {response.status_code}]"

    except requests.exceptions.Timeout:
        return "[Image description timed out]"
    except requests.exceptions.ConnectionError:
        return "[Ollama not running — start with: docker-compose up ollama -d]"
    except Exception as e:
        return f"[Image description error: {e}]"


def extract_and_describe_images(document) -> str:
    """
    Read Docling pictures directly and describe them with Ollama.
    """

    pictures = document.pictures
    print("Found images:", len(pictures))

    if not pictures:
        return ""

    descriptions = []
    descriptions.append("\n\n## Extracted Image Content\n")

    total = len(pictures)

    for i, picture in enumerate(pictures, start=1):
        img = picture.get_image(document)

        if img is None:
            continue

        os.makedirs("results/docling_images", exist_ok=True)
        img.save(f"results/docling_images/figure_{i}.png")

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        image_b64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        page = picture.prov[0].page_no if picture.prov else -1

        print(f"  Describing image {i}/{total} (Page {page})...")

        description = describe_image(
            image_b64=image_b64,
            page=page,
            index=i
        )

        descriptions.append(
            f"\n### Figure {i} (Page {page})\n\n"
            f"{description}\n"
        )

    return "\n".join(descriptions)