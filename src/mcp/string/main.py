import os
import subprocess
from pathlib import Path

from fastmcp import FastMCP

FILES_DIR = Path(os.environ.get("FILES_DIR", "/data/files"))

mcp = FastMCP("String MCP Server")


def _safe_path(filename: str) -> Path:
    """Resolve filename inside FILES_DIR, rejecting path traversal or subdirectories."""
    if "/" in filename or "\\" in filename:
        raise ValueError(f"Subdirectories are not allowed: {filename!r}")
    path = FILES_DIR / filename
    if path.resolve().parent != FILES_DIR.resolve():
        raise ValueError(f"Path traversal detected: {filename!r}")
    return path


@mcp.tool
def head(filename: str, lines: int = 10) -> str:
    """Read the first N lines of a file from the storage folder."""
    path = _safe_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filename!r}")
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[:lines])


@mcp.tool
def tail(filename: str, lines: int = 10) -> str:
    """Read the last N lines of a file from the storage folder."""
    path = _safe_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filename!r}")
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])


@mcp.tool
def ripgrep(pattern: str, filename: str = "") -> str:
    """Search for a regex pattern in files using ripgrep.

    Leave filename empty to search all files in the storage folder.
    Returns matching lines in the format 'filename:line_number:match'.
    """
    target = _safe_path(filename) if filename else FILES_DIR
    result = subprocess.run(
        ["rg", "--no-heading", "--with-filename", "-n", pattern, str(target)],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    if not output:
        return "No matches found."
    # Strip the FILES_DIR prefix so paths are relative (e.g. "hello.txt:1:Hello")
    prefix = str(FILES_DIR) + "/"
    return "\n".join(
        line[len(prefix):] if line.startswith(prefix) else line
        for line in output.splitlines()
    )


@mcp.tool
def replace(filename: str, old: str, new: str) -> str:
    """Replace all occurrences of a literal string in a file with another string.

    Returns the number of substitutions made.
    """
    path = _safe_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {filename!r}")
    content = path.read_text(encoding="utf-8")
    updated = content.replace(old, new)
    count = content.count(old)
    path.write_text(updated, encoding="utf-8")
    return f"Replaced {count} occurrence(s) of {old!r} with {new!r} in {filename!r}"


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
