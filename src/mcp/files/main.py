import os
from pathlib import Path

from fastmcp import FastMCP

FILES_DIR = Path(os.environ.get("FILES_DIR", "/data/files"))
FILES_DIR.mkdir(parents=True, exist_ok=True)

mcp = FastMCP("Files MCP Server")


def _safe_path(filename: str) -> Path:
    """Resolve filename inside FILES_DIR, rejecting any path traversal or subdirectory."""
    # Reject any path separator in the filename
    if "/" in filename or "\\" in filename:
        raise ValueError(f"Subdirectories are not allowed: {filename!r}")
    path = FILES_DIR / filename
    # Extra guard: ensure resolved path is still inside FILES_DIR
    if path.resolve().parent != FILES_DIR.resolve():
        raise ValueError(f"Path traversal detected: {filename!r}")
    return path


@mcp.tool
def list_files() -> list[str]:
    """List all files in the storage folder."""
    return sorted(p.name for p in FILES_DIR.iterdir() if p.is_file())


@mcp.tool
def read_file(filename: str) -> str:
    """Read the contents of a file from the storage folder."""
    path = _safe_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filename!r}")
    return path.read_text(encoding="utf-8")


@mcp.tool
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the storage folder, creating it if it does not exist."""
    path = _safe_path(filename)
    path.write_text(content, encoding="utf-8")
    return f"Written {len(content)} characters to {filename!r}"


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
