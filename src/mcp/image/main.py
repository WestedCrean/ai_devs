import base64
import os
import re
from pathlib import Path

from openai import OpenAI
from fastmcp import FastMCP

FILES_DIR = Path(os.environ.get("FILES_DIR", "/data/files"))
FILES_DIR.mkdir(parents=True, exist_ok=True)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GENERATE_MODEL = "openai/gpt-5-image-mini"
EDIT_MODEL = "google/gemini-3.1-flash-image-preview"
VISION_MODEL = "google/gemini-3.1-flash-image-preview"

mcp = FastMCP("Image MCP Server")


def _client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


def _safe_path(filename: str) -> Path:
    if "/" in filename or "\\" in filename:
        raise ValueError(f"Subdirectories are not allowed: {filename!r}")
    path = FILES_DIR / filename
    if path.resolve().parent != FILES_DIR.resolve():
        raise ValueError(f"Path traversal detected: {filename!r}")
    return path


def _extract_b64(data_url: str) -> str:
    m = re.match(r"data:[^;]+;base64,(.+)", data_url, re.DOTALL)
    if not m:
        raise ValueError(f"Not a base64 data URL: {data_url[:60]!r}")
    return m.group(1).replace("\n", "")


def _find_image_in_parts(parts) -> str | None:
    """Search a list of content parts for an image_url data URI."""
    for part in parts:
        if isinstance(part, dict):
            if part.get("type") == "image_url":
                return _extract_b64(part["image_url"]["url"])
            # Plain text with embedded data URI
            text = part.get("text", "")
        else:
            if getattr(part, "type", None) == "image_url":
                url = part.image_url.url if hasattr(part.image_url, "url") else part.image_url["url"]
                return _extract_b64(url)
            text = getattr(part, "text", "")
        m = re.search(r"data:[^;]+;base64,([A-Za-z0-9+/=\n]+)", text)
        if m:
            return m.group(1).replace("\n", "")
    return None


def _chat_generate(model: str, messages: list) -> bytes:
    """Call chat completions, extract the image from the response, return raw bytes."""
    client = _client()
    response = client.chat.completions.create(model=model, messages=messages)
    raw_msg = response.model_dump()["choices"][0]["message"]

    # 1. Non-standard `images` field (gpt-5-image-mini style)
    images = raw_msg.get("images") or []
    b64 = _find_image_in_parts(images)

    # 2. Standard `content` field (Gemini / other models)
    if b64 is None:
        content = raw_msg.get("content") or []
        parts = [content] if isinstance(content, str) else (content if isinstance(content, list) else [])
        b64 = _find_image_in_parts(parts)

    if b64 is None:
        raise ValueError("No image data found in model response")
    return base64.b64decode(b64)


@mcp.tool
def generate_image(prompt: str, filename: str) -> str:
    """Generate an image from a text prompt and save it to the storage folder."""
    path = _safe_path(filename)
    image_data = _chat_generate(
        GENERATE_MODEL,
        [{"role": "user", "content": prompt}],
    )
    path.write_bytes(image_data)
    return f"Generated image saved to {filename!r} ({len(image_data)} bytes)"


@mcp.tool
def edit_image(image_b64: str, prompt: str, filename: str) -> str:
    """Edit an image using Gemini. Provide the source image as a base64 string,
    describe the edit in prompt, and specify the output filename to save into
    the storage folder."""
    path = _safe_path(filename)
    image_data = _chat_generate(
        EDIT_MODEL,
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    path.write_bytes(image_data)
    return f"Edited image saved to {filename!r} ({len(image_data)} bytes)"


@mcp.tool
def vision_describe(filename: str, prompt: str) -> str:
    """Read an image from the storage folder and answer a question about it using a vision model.

    Use this to extract text, data, or descriptions from image files (PNG, JPG, etc.).
    Example: vision_describe('map.png', 'List all route codes shown in this diagram.')
    """
    path = _safe_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filename!r}")
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lstrip(".").lower() or "png"
    mime = f"image/{suffix}"

    client = _client()
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    raw_msg = response.model_dump()["choices"][0]["message"]
    content = raw_msg.get("content") or ""
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
