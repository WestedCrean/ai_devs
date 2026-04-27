import os
import subprocess
from collections import defaultdict
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
def read_line(filename: str, line_number: int) -> str:
    """Read one 1-based line from a file in the storage folder.

    Args:
        filename: File name in the storage folder.
        line_number: 1-based line number to read.
    """
    logger.info("tool=read_line filename={} line_number={}", filename, line_number)
    try:
        if line_number < 1:
            raise ValueError("line_number must be >= 1")
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")

        lines = path.read_text(encoding="utf-8").splitlines()
        line_count = len(lines)
        if line_number > line_count:
            raise IndexError(
                f"line_number {line_number} is outside file range 1..{line_count}"
            )

        line = lines[line_number - 1]
        output = f"{filename}:{line_number}:{line}"
        logger.info("tool=read_line filename={} result_chars={}", filename, len(output))
        return output
    except Exception:
        logger.exception(
            "tool=read_line filename={} line_number={} failed",
            filename,
            line_number,
        )
        raise


@mcp.tool
def replace_line(filename: str, line_number: int, new_line: str) -> str:
    """Replace one 1-based line in a file in the storage folder.

    Args:
        filename: File name in the storage folder.
        line_number: 1-based line number to replace.
        new_line: Replacement line content without a trailing newline.
    """
    logger.info(
        "tool=replace_line filename={} line_number={} new_chars={}",
        filename,
        line_number,
        len(new_line),
    )
    try:
        if line_number < 1:
            raise ValueError("line_number must be >= 1")
        if "\n" in new_line or "\r" in new_line:
            raise ValueError("new_line must contain exactly one line")

        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")

        content = path.read_text(encoding="utf-8")
        had_trailing_newline = content.endswith("\n")
        lines = content.splitlines()
        line_count = len(lines)
        if line_number > line_count:
            raise IndexError(
                f"line_number {line_number} is outside file range 1..{line_count}"
            )

        old_line = lines[line_number - 1]
        lines[line_number - 1] = new_line
        updated = "\n".join(lines)
        if had_trailing_newline:
            updated += "\n"
        path.write_text(updated, encoding="utf-8")

        logger.info("tool=replace_line filename={} completed", filename)
        return (
            f"Replaced line {line_number} in {filename!r}. "
            f"Old: {old_line!r}. New: {new_line!r}."
        )
    except Exception:
        logger.exception(
            "tool=replace_line filename={} line_number={} failed",
            filename,
            line_number,
        )
        raise


@mcp.tool
def ripgrep(
    pattern: str,
    filename: str = "",
    limit: int = 20,
    offset: int = 0,
    max_chars: int = 12_000,
    output_filename: str = "",
) -> str:
    """Search for a regex pattern in files using ripgrep.

    Leave filename empty to search all files in the storage folder.
    Results are paginated with offset and limit. Default limit is 20. Matching
    lines use the format 'filename:line_number:match' plus pagination metadata.

    Prefer setting output_filename for broad searches, high limits, or patterns
    likely to return many matches. This appends the current result page to a file
    in the storage folder and returns only metadata plus a short preview, keeping
    large search results out of the agent context. Use follow-up calls with
    offset and the same output_filename to append later pages to the same file.
    """
    logger.info(
        "tool=ripgrep pattern={!r} filename={} limit={} offset={} max_chars={} output_filename={}",
        pattern,
        filename,
        limit,
        offset,
        max_chars,
        output_filename,
    )
    try:
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        max_chars = max(1_000, min(max_chars, 50_000))
        target = _safe_path(filename) if filename else FILES_DIR
        output_path = _safe_path(output_filename) if output_filename else None
        result = subprocess.run(
            ["rg", "--no-heading", "--with-filename", "-n", pattern, str(target)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
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
        if output_path:
            output_entry = "\n".join(
                [
                    f"# ripgrep pattern={pattern!r} filename={filename!r} "
                    f"offset={offset} limit={limit} returned={len(page_matches)}",
                    body,
                    "",
                ]
            )
            with output_path.open("a", encoding="utf-8") as file:
                file.write(output_entry)
            output_chars = len(body)
        truncated_by_chars = False
        if not output_path and len(body) > max_chars:
            body = body[:max_chars]
            truncated_by_chars = True

        metadata = [
            f"ripgrep matches: total={total_matches}, offset={offset}, "
            f"returned={len(page_matches)}, limit={limit}",
        ]
        if output_path:
            metadata.append(
                f"Appended current page to {output_filename!r} "
                f"({len(page_matches)} lines, {output_chars} chars)."
            )
        if has_more:
            metadata.append(f"More matches available. Call again with offset={next_offset}.")
        if truncated_by_chars:
            metadata.append(
                f"Page truncated by max_chars={max_chars}. Use a smaller limit or higher offset."
            )

        if output_path:
            preview_lines = page_matches[:10]
            preview = "\n".join(preview_lines)
            if len(page_matches) > len(preview_lines):
                preview += (
                    f"\n... {len(page_matches) - len(preview_lines)} "
                    "more lines written to file"
                )
            body = preview

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
def summarize_log_patterns(
    filename: str,
    severities: str = "WARN,ERRO,CRIT",
    components: str = "",
    max_examples_per_group: int = 3,
) -> str:
    """Summarize log lines by severity and component with representative examples.

    Args:
        filename: Log file name from the storage folder.
        severities: Comma-separated severities to include, for example WARN,ERRO,CRIT.
        components: Optional comma-separated component IDs or prefixes to include.
        max_examples_per_group: Maximum representative lines per severity/component group.
    """
    logger.info(
        "tool=summarize_log_patterns filename={} severities={!r} components={!r}",
        filename,
        severities,
        components,
    )
    try:
        path = _safe_path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filename!r}")

        severity_set = {
            item.strip().upper()
            for item in severities.split(",")
            if item.strip()
        }
        component_filters = [
            item.strip().upper()
            for item in components.split(",")
            if item.strip()
        ]
        max_examples_per_group = max(1, min(max_examples_per_group, 10))

        counts: dict[tuple[str, str], int] = defaultdict(int)
        examples: dict[tuple[str, str], list[str]] = defaultdict(list)
        total_scanned = 0
        total_matched = 0

        for line in path.read_text(encoding="utf-8").splitlines():
            total_scanned += 1
            upper_line = line.upper()
            severity = next(
                (item for item in severity_set if f"[{item}]" in upper_line),
                "",
            )
            if not severity:
                continue

            component = _extract_component(line)
            if component_filters and not any(
                component.startswith(item) or item in upper_line
                for item in component_filters
            ):
                continue

            key = (severity, component or "UNKNOWN")
            counts[key] += 1
            total_matched += 1
            if len(examples[key]) < max_examples_per_group:
                examples[key].append(line)

        if not counts:
            return (
                f"No matching log lines found. Scanned lines: {total_scanned}. "
                f"Severities: {', '.join(sorted(severity_set)) or 'none'}."
            )

        lines = [
            f"Scanned lines: {total_scanned}. Matched lines: {total_matched}.",
            "Groups:",
        ]
        for (severity, component), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        ):
            lines.append(f"- {severity} {component}: {count}")
            lines.extend(f"  example: {example}" for example in examples[(severity, component)])

        output = "\n".join(lines)
        logger.info(
            "tool=summarize_log_patterns filename={} groups={} result_chars={}",
            filename,
            len(counts),
            len(output),
        )
        return output
    except Exception:
        logger.exception("tool=summarize_log_patterns filename={} failed", filename)
        raise


def _extract_component(line: str) -> str:
    """Extract a likely component identifier from one log line."""
    for token in line.replace(",", " ").split():
        stripped = token.strip("[]:;()")
        if any(char.isdigit() for char in stripped) and any(
            char.isalpha() for char in stripped
        ):
            return stripped.upper()
    return "UNKNOWN"


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
