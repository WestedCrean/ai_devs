import os
import subprocess
from pathlib import Path

from fastmcp import FastMCP
from loguru import logger

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
    logger.info("tool=head filename={} lines={}", filename, lines)
    try:
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")
        output = "\n".join(path.read_text(encoding="utf-8").splitlines()[:lines])
        logger.info("tool=head filename={} result_chars={}", filename, len(output))
        return output
    except Exception:
        logger.exception("tool=head filename={} failed", filename)
        raise


@mcp.tool
def tail(filename: str, lines: int = 10) -> str:
    """Read the last N lines of a file from the storage folder."""
    logger.info("tool=tail filename={} lines={}", filename, lines)
    try:
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")
        output = "\n".join(path.read_text(encoding="utf-8").splitlines()[-lines:])
        logger.info("tool=tail filename={} result_chars={}", filename, len(output))
        return output
    except Exception:
        logger.exception("tool=tail filename={} failed", filename)
        raise


@mcp.tool
def ripgrep(
    pattern: str,
    filename: str = "",
    limit: int = 50,
    offset: int = 0,
    max_chars: int = 12_000,
) -> str:
    """Search for a regex pattern in files using ripgrep.

    Leave filename empty to search all files in the storage folder.
    Results are paginated with offset and limit. Returns matching lines in the
    format 'filename:line_number:match' plus pagination metadata.
    """
    logger.info(
        "tool=ripgrep pattern={!r} filename={} limit={} offset={} max_chars={}",
        pattern,
        filename,
        limit,
        offset,
        max_chars,
    )
    try:
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        max_chars = max(1_000, min(max_chars, 50_000))
        target = _safe_path(filename) if filename else FILES_DIR
        result = subprocess.run(
            ["rg", "--no-heading", "--with-filename", "-n", pattern, str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        if not output:
            logger.info("tool=ripgrep pattern={!r} matches=0", pattern)
            return "No matches found."
        # Strip the FILES_DIR prefix so paths are relative (e.g. "hello.txt:1:Hello")
        prefix = str(FILES_DIR) + "/"
        matches = [
            line[len(prefix) :] if line.startswith(prefix) else line
            for line in output.splitlines()
        ]
        total_matches = len(matches)
        page_matches = matches[offset : offset + limit]
        next_offset = offset + len(page_matches)
        has_more = next_offset < total_matches
        body = "\n".join(page_matches)
        truncated_by_chars = False
        if len(body) > max_chars:
            body = body[:max_chars]
            truncated_by_chars = True

        metadata = [
            f"ripgrep matches: total={total_matches}, offset={offset}, "
            f"returned={len(page_matches)}, limit={limit}",
        ]
        if has_more:
            metadata.append(f"More matches available. Call again with offset={next_offset}.")
        if truncated_by_chars:
            metadata.append(
                f"Page truncated by max_chars={max_chars}. Use a smaller limit or higher offset."
            )

        stripped = "\n".join(
            [*metadata, "", body]
            if body
            else [*metadata, "", "No matches returned for this offset."]
        )
        if len(stripped) > max_chars:
            suffix = f"\n[Tool output truncated by max_chars={max_chars}]"
            keep_chars = max(0, max_chars - len(suffix))
            stripped = stripped[:keep_chars] + suffix
        logger.info(
            "tool=ripgrep pattern={!r} matches={} result_chars={}",
            pattern,
            total_matches,
            len(stripped),
        )
        return stripped
    except Exception:
        logger.exception("tool=ripgrep pattern={!r} filename={} failed", pattern, filename)
        raise


@mcp.tool
def replace(filename: str, old: str, new: str) -> str:
    """Replace all occurrences of a literal string in a file with another string.

    Returns the number of substitutions made.
    """
    logger.info(
        "tool=replace filename={} old_chars={} new_chars={}",
        filename,
        len(old),
        len(new),
    )
    try:
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")
        content = path.read_text(encoding="utf-8")
        updated = content.replace(old, new)
        count = content.count(old)
        path.write_text(updated, encoding="utf-8")
        logger.info("tool=replace filename={} replacements={}", filename, count)
        return f"Replaced {count} occurrence(s) of {old!r} with {new!r} in {filename!r}"
    except Exception:
        logger.exception("tool=replace filename={} failed", filename)
        raise


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
