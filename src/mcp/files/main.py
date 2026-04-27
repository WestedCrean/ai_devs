import base64
import os
from pathlib import Path

from fastmcp import FastMCP
from loguru import logger

FILES_DIR = Path(os.environ.get("FILES_DIR", "/data/files"))
FILES_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("Files MCP Server")


def _safe_path(filename: str) -> Path:
    """Resolve filename inside FILES_DIR, rejecting path traversal or subdirectories."""
    if "/" in filename or "\\" in filename:
        raise ValueError(f"Subdirectories are not allowed: {filename!r}")
    path = FILES_DIR / filename
    if path.resolve().parent != FILES_DIR.resolve():
        raise ValueError(f"Path traversal detected: {filename!r}")
    return path


@mcp.tool
def list_files() -> list[str]:
    """List all files in the storage folder."""
    logger.info("tool=list_files dir={}", FILES_DIR)
    try:
        files = sorted(p.name for p in FILES_DIR.iterdir() if p.is_file())
        logger.info("tool=list_files result_count={}", len(files))
        return files
    except Exception:
        logger.exception("tool=list_files failed")
        raise


@mcp.tool
def read_file(filename: str) -> str:
    """Read the contents of a file from the storage folder."""
    logger.info("tool=read_file filename={}", filename)
    try:
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")
        content = path.read_text(encoding="utf-8")
        logger.info("tool=read_file filename={} chars={}", filename, len(content))
        return content
    except Exception:
        logger.exception("tool=read_file filename={} failed", filename)
        raise


@mcp.tool
def read_file_b64(filename: str) -> str:
    """Read a binary file from the storage folder and return its contents as a base64 string.
    Use this for images and other non-text files."""
    logger.info("tool=read_file_b64 filename={}", filename)
    try:
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        logger.info(
            "tool=read_file_b64 filename={} bytes={} encoded_chars={}",
            filename,
            len(data),
            len(encoded),
        )
        return encoded
    except Exception:
        logger.exception("tool=read_file_b64 filename={} failed", filename)
        raise


@mcp.tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the storage folder, creating it if it does not exist."""
    logger.info("tool=write_file filename={} chars={}", filename, len(content))
    try:
        path = _safe_path(filename)
        path.write_text(content, encoding="utf-8")
        result = f"Written {len(content)} characters to {filename!r}"
        logger.info("tool=write_file filename={} completed", filename)
        return result
    except Exception:
        logger.exception("tool=write_file filename={} failed", filename)
        raise


@mcp.tool
def delete_file(filename: str) -> str:
    """Delete a file from the storage folder."""
    logger.info("tool=delete_file filename={}", filename)
    try:
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")
        path.unlink()
        logger.info("tool=delete_file filename={} completed", filename)
        return f"Deleted {filename!r}"
    except Exception:
        logger.exception("tool=delete_file filename={} failed", filename)
        raise


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
