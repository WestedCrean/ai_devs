import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console

from src.ai_devs_core import (
    AIDevsClient,
    FAgent,
    complete,
    create_agent,
    discover_mcp_tools,
    get_config,
)
from src.ai_devs_core.memory import ObservedMemory
from src.ai_devs_core.session import RefinedSessionManager
from src.ai_devs_core.utils import count_tokens


env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

DATA_SAVE_PATH = Path("./data")
MCP_FILES_PATH = DATA_SAVE_PATH / "mcp-files"
TASK_NAME = "failure"
SESSION_CONTEXT_TOKEN_LIMIT = 20_000
VERIFY_TOKEN_LIMIT = 1_500

SYSTEM_PROMPT = """
You are solving the failure log task for a power plant. The source log is large, so your
job is to build a compact, evidence-rich payload without loading broad search output into
the chat context.

Final payload requirements:
- Send logs with verify_reflected_failure_logs or verify_failure_logs only.
- The logs field must be a string with one event per line.
- Each event line must preserve YYYY-MM-DD date, HH:MM time, severity, component ID, and
  the operational clue.
- Keep the final payload under 1,500 tokens.
- Include only events useful for failure analysis: power, cooling, water pumps, firmware,
  controllers, safety systems, sensors, thresholds, trips, missing telemetry, and related
  anomalies.

Use this workflow:
1. Download logs with download_server_logs.
2. Collect broad candidates into files with collect_log_candidates. For broad searches,
   always write to output_filename and read back only small pages.
3. Load selected candidate pages into memory with add_failure_observations_from_file.
4. Check progress with get_failure_workflow_status, then compress with reflect_failure_memory.
5. Submit with verify_reflected_failure_logs.
6. If Headquarters says components are missing, call search_missing_component for each
   missing component, load those candidates, reflect again, and verify again.

Hard rules:
- Do not use broad raw ripgrep or read_file on failure.csv in the main chat.
- Do not paste hundreds of log lines into add_failure_observations manually.
- Prefer the task-level tools because they write large results to files and return metadata.
- Treat verification feedback as source-of-truth guidance for the next search.
"""

MCP_DEFINITIONS = {
    "files": "http://localhost:8002/mcp",
    "string": "http://localhost:8003/mcp",
}
ALLOWED_MCP_TOOLS = {"list_files", "head", "tail", "read_line"}

class _LazyAIDevsClient:
    """Defer API client construction until a method is used."""

    def __init__(self) -> None:
        self._client: AIDevsClient | None = None

    def _get_client(self) -> AIDevsClient:
        if self._client is None:
            try:
                config = get_config()
            except ValidationError as exc:  # pragma: no cover - guarded for runtime UX
                raise RuntimeError(
                    "Missing AI Devs environment configuration. Set required env vars "
                    "or provide a .env file before running this lesson."
                ) from exc
            self._client = AIDevsClient(
                api_url=config.AI_DEVS_API_URL,
                api_key=config.AI_DEVS_API_KEY,
            )
        return self._client

    def get_task(self, *args, **kwargs):
        return self._get_client().get_task(*args, **kwargs)

    def verify(self, *args, **kwargs):
        return self._get_client().verify(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._get_client(), name)


ai_devs_core = _LazyAIDevsClient()

FAILURE_MEMORY_TASK = (
    "Store candidate power-plant failure log events and compress them into the "
    "verify payload format: one relevant event per line, under 1500 tokens."
)
failure_memory = ObservedMemory(current_task=FAILURE_MEMORY_TASK)
reflected_failure_logs = ""
latest_verification_feedback = ""
failure_memory_agent: FAgent | None = None

LOG_LINE_RE = re.compile(
    r"^\[?(?P<date>\d{4}-\d{2}-\d{2})[ T]"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\]?\s+"
    r"\[(?P<severity>[A-Z]+)\]\s+"
    r"(?P<component>[A-Za-z][A-Za-z0-9_-]*)\s+"
    r"(?P<message>.*)$"
)


class FailureLogCompression(BaseModel):
    """Compressed failure log payload."""

    logs: str


class FailureCandidate(BaseModel):
    """One candidate failure event with source metadata."""

    timestamp: str = ""
    severity: str = "INFO"
    component: str = "UNKNOWN"
    source_file: str
    source_line: int
    message: str
    reason: str = ""

    model_config = ConfigDict(extra="forbid")

    def as_output_line(self) -> str:
        """Render a stable file-backed candidate line."""
        prefix = f"{self.source_file}:{self.source_line}:"
        event = self.message
        if self.timestamp and self.component != "UNKNOWN":
            event = (
                f"{self.timestamp} [{self.severity}] {self.component} {self.message}"
            )
        return f"{prefix}{event}"

    def as_observation(self) -> str:
        """Render the candidate as a compact memory observation."""
        source = f"source={self.source_file}:{self.source_line}"
        reason = f" reason={self.reason}" if self.reason else ""
        if self.timestamp and self.component != "UNKNOWN":
            return (
                f"{self.timestamp} [{self.severity}] {self.component} {source}"
                f"{reason} {self.message}"
            )
        return f"[{self.severity}] {self.component} {source}{reason} {self.message}"


def _split_csv(value: str | None) -> list[str]:
    """Split comma-separated tool input into non-empty values."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_tool_filename(filename: str) -> str:
    """Return a filename constrained to MCP files storage."""
    name = Path(filename).name
    if not name:
        raise ValueError("filename must not be empty")
    return name


def _mcp_file(filename: str) -> Path:
    """Return the resolved MCP storage path for a tool filename."""
    return MCP_FILES_PATH / _safe_tool_filename(filename)


def _compile_patterns(patterns: str | None) -> list[re.Pattern[str]]:
    """Compile comma-separated regex patterns with case-insensitive matching."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in _split_csv(patterns)]


def _split_log_lines(lines: str) -> list[str]:
    """Return non-empty stripped log lines."""
    return [line.strip() for line in lines.splitlines() if line.strip()]


def _content_chunks_to_text(content: object) -> str:
    """Return user-visible text from provider content chunks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if hasattr(item, "thinking"):
                continue
            part = _content_chunks_to_text(item)
            if part:
                parts.append(part)
        return "\n".join(parts)

    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text

    nested = getattr(content, "content", None)
    if nested is not None:
        return _content_chunks_to_text(nested)

    return str(content)


def _extract_reflected_logs(content: object) -> str:
    """Return reflected log text from plain or accidentally wrapped model output."""
    text = _content_chunks_to_text(content).strip()
    if not text:
        return ""

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(parsed, dict) and isinstance(parsed.get("logs"), str):
        return parsed["logs"].strip()
    return text


def _fit_log_lines_to_token_budget(logs: str, target_tokens: int) -> str:
    """Trim reflected logs by whole lines until they fit the token budget."""
    lines = _split_log_lines(logs)
    kept: list[str] = []
    for line in lines:
        candidate = "\n".join([*kept, line])
        if count_tokens(candidate) > target_tokens:
            continue
        kept.append(line)
    return "\n".join(kept)


def _count_file_lines(path: Path) -> int:
    """Count lines in a text file without returning its contents."""
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def _parse_candidate_line(line: str, fallback_filename: str, index: int) -> FailureCandidate:
    """Parse a stored candidate line into structured metadata."""
    source_file = fallback_filename
    source_line = index + 1
    event_line = line.strip()

    source_match = re.match(r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<event>.*)$", event_line)
    if source_match:
        source_file = Path(source_match.group("file")).name
        source_line = int(source_match.group("line"))
        event_line = source_match.group("event").strip()

    event_match = LOG_LINE_RE.match(event_line)
    if not event_match:
        return FailureCandidate(
            source_file=source_file,
            source_line=source_line,
            message=event_line,
            reason="unparsed candidate line",
        )

    timestamp = f"{event_match.group('date')} {event_match.group('time')[:5]}"
    return FailureCandidate(
        timestamp=timestamp,
        severity=event_match.group("severity"),
        component=event_match.group("component"),
        source_file=source_file,
        source_line=source_line,
        message=event_match.group("message").strip(),
    )


def _candidate_reason(
    candidate: FailureCandidate,
    include_patterns: list[re.Pattern[str]],
    components: set[str],
) -> str:
    """Explain why a candidate was selected."""
    reasons = [candidate.severity]
    component_upper = candidate.component.upper()
    if component_upper in components:
        reasons.append(f"component:{candidate.component}")
    for pattern in include_patterns:
        if pattern.search(candidate.as_observation()):
            reasons.append(f"pattern:{pattern.pattern}")
            break
    return ",".join(reasons)


def _line_matches_filters(
    candidate: FailureCandidate,
    raw_line: str,
    severities: set[str],
    components: set[str],
    include_patterns: list[re.Pattern[str]],
    exclude_patterns: list[re.Pattern[str]],
) -> bool:
    """Return whether a parsed candidate should be written to the output file."""
    searchable = f"{candidate.as_observation()} {raw_line}"
    if exclude_patterns and any(pattern.search(searchable) for pattern in exclude_patterns):
        return False
    if severities and candidate.severity.upper() not in severities:
        return False
    if components:
        component_upper = candidate.component.upper()
        if component_upper not in components and not any(
            component in searchable.upper() for component in components
        ):
            return False
    if include_patterns and not any(pattern.search(searchable) for pattern in include_patterns):
        return False
    return True


def download_server_logs() -> str:
    """Download server logs into MCP files storage and return file metadata."""
    file_path = ai_devs_core.download_dataset_file(
        dataset=TASK_NAME,
        save_path=MCP_FILES_PATH,
        download_always=True,
    )
    stat = file_path.stat()
    line_count = _count_file_lines(file_path)
    return (
        f"Downloaded logs to MCP files storage as {file_path.name!r}. "
        f"Lines: {line_count}. Bytes: {stat.st_size}. "
        "Next use collect_log_candidates with output_filename to collect broad matches "
        "without placing all matches in chat."
    )


def collect_log_candidates(
    filename: str,
    output_filename: str,
    severities: str = "WARN,ERRO,CRIT",
    components: str = "",
    include_patterns: str = "",
    exclude_patterns: str = "",
) -> str:
    """Collect filtered log events into output_filename and return metadata only.

    filename: Source log filename in MCP storage, usually failure.csv.
    output_filename: Destination file in MCP storage. Use this by default for every broad
        search so raw matches stay out of the agent context. Existing files are overwritten.
    severities: Comma-separated severities to include, for example WARN,ERRO,CRIT.
    components: Optional comma-separated component IDs to require, for example PWR01,ECCS8.
    include_patterns: Optional comma-separated regex patterns; at least one must match.
    exclude_patterns: Optional comma-separated regex patterns; matching lines are skipped.
    """
    source_path = _mcp_file(filename)
    output_path = _mcp_file(output_filename)
    severity_set = {item.upper() for item in _split_csv(severities)}
    component_set = {item.upper() for item in _split_csv(components)}
    include_regexes = _compile_patterns(include_patterns)
    exclude_regexes = _compile_patterns(exclude_patterns)

    if not source_path.exists():
        return f"Source file {source_path.name!r} does not exist. Run download_server_logs first."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scanned = 0
    written = 0
    severity_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()

    with source_path.open("r", encoding="utf-8") as source:
        with output_path.open("w", encoding="utf-8") as output:
            for line_number, raw_line in enumerate(source, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                scanned += 1
                candidate = _parse_candidate_line(
                    f"{source_path.name}:{line_number}:{stripped}",
                    source_path.name,
                    line_number - 1,
                )
                if not _line_matches_filters(
                    candidate,
                    stripped,
                    severity_set,
                    component_set,
                    include_regexes,
                    exclude_regexes,
                ):
                    continue
                candidate.reason = _candidate_reason(
                    candidate,
                    include_regexes,
                    component_set,
                )
                output.write(candidate.as_output_line() + "\n")
                written += 1
                severity_counts[candidate.severity] += 1
                component_counts[candidate.component] += 1

    top_components = ", ".join(
        f"{component}:{count}" for component, count in component_counts.most_common(12)
    )
    severity_summary = ", ".join(
        f"{severity}:{count}" for severity, count in severity_counts.most_common()
    )
    return (
        f"Collected candidates into {output_path.name!r}. Scanned lines: {scanned}. "
        f"Written candidates: {written}. Severities: {severity_summary or 'none'}. "
        f"Top components: {top_components or 'none'}. "
        f"Load pages with add_failure_observations_from_file(filename={output_path.name!r}, "
        "offset=0, limit=50)."
    )


def add_failure_observations_from_file(
    filename: str,
    offset: int = 0,
    limit: int = 50,
) -> str:
    """Load a small page of stored candidate lines into failure memory.

    filename: Candidate file in MCP storage created by collect_log_candidates or
        search_missing_component.
    offset: Zero-based line offset to start from.
    limit: Number of candidate lines to load. Keep this small; default is 50.
    """
    path = _mcp_file(filename)
    if not path.exists():
        return f"Candidate file {path.name!r} does not exist."

    bounded_offset = max(0, offset)
    bounded_limit = min(max(1, limit), 200)
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = lines[bounded_offset : bounded_offset + bounded_limit]
    existing = set(failure_memory.observations)
    added = 0
    skipped_duplicates = 0
    components: Counter[str] = Counter()

    for index, line in enumerate(selected, start=bounded_offset):
        candidate = _parse_candidate_line(line, path.name, index)
        observation = candidate.as_observation()
        if observation in existing:
            skipped_duplicates += 1
            continue
        failure_memory.add(observation)
        existing.add(observation)
        added += 1
        components[candidate.component] += 1

    next_offset = bounded_offset + len(selected)
    more = next_offset < len(lines)
    component_summary = ", ".join(
        f"{component}:{count}" for component, count in components.most_common(10)
    )
    return (
        f"Loaded {added} new observations from {path.name!r}; duplicates skipped: "
        f"{skipped_duplicates}. File lines: {len(lines)}. Next offset: "
        f"{next_offset if more else 'done'}. Stored observations: "
        f"{len(failure_memory.observations)}. Approx memory tokens: "
        f"{failure_memory.observation_tokens}. Components loaded: "
        f"{component_summary or 'none'}."
    )


def search_missing_component(
    component: str,
    filename: str = "failure.csv",
    output_filename: str = "",
) -> str:
    """Search one missing component into a file and return metadata only.

    component: Component ID from verification feedback, for example PWR01.
    filename: Source log filename in MCP storage, usually failure.csv.
    output_filename: Destination candidate file. Defaults to missing_<component>.txt.
    """
    component_id = component.strip()
    if not component_id:
        return "component must not be empty"

    safe_component = re.sub(r"[^A-Za-z0-9_-]+", "_", component_id).strip("_")
    destination = output_filename or f"missing_{safe_component}.txt"
    return collect_log_candidates(
        filename=filename,
        output_filename=destination,
        severities="WARN,ERRO,CRIT",
        components=component_id,
        include_patterns=component_id,
        exclude_patterns="",
    )


def get_failure_workflow_status() -> str:
    """Return compact status for files, memory, reflected logs, and verification feedback."""
    files = []
    if MCP_FILES_PATH.exists():
        for path in sorted(MCP_FILES_PATH.iterdir()):
            if path.is_file():
                files.append(f"{path.name}:{path.stat().st_size}B")

    reflected_tokens = count_tokens(reflected_failure_logs) if reflected_failure_logs else 0
    feedback_preview = latest_verification_feedback[:1_000]
    if len(latest_verification_feedback) > len(feedback_preview):
        feedback_preview += "... truncated"

    return (
        f"MCP files: {', '.join(files[:30]) or 'none'}. "
        f"Stored observations: {len(failure_memory.observations)} "
        f"({failure_memory.observation_tokens} tokens). "
        f"Reflected logs: {len(_split_log_lines(reflected_failure_logs))} lines, "
        f"{reflected_tokens} tokens. "
        f"Latest verification feedback: {feedback_preview or 'none'}."
    )


def reflect_failure_memory(target_tokens: int = VERIFY_TOKEN_LIMIT) -> str:
    """Compress stored candidate events into verify-ready failure logs."""
    global reflected_failure_logs

    if not failure_memory.observations:
        return (
            "No candidate observations stored. Use collect_log_candidates and "
            "add_failure_observations_from_file first."
        )

    memory_agent = failure_memory_agent or FAgent(model_id="mistral-large-latest")
    prompt = f"""
Compress these candidate power-plant failure log events into the final verify payload.

Hard requirements:
- Return plain text only. Do not return JSON, Markdown, code fences, commentary, or labels.
- One line equals one event.
- Preserve date in YYYY-MM-DD format, time in HH:MM, severity, and component ID.
- Keep only events relevant to failure analysis: power, cooling, water pumps, software,
  reactor safety systems, sensors, controllers, and directly related components.
- Prefer CRIT, ERRO, WARN, anomalies, trips, thresholds, missing telemetry, and feedback
  from technicians.
- Remove duplicates and routine OK/INFO events unless they explain the failure timeline.
- Paraphrase aggressively, but do not remove causal clues.
- Fit within {target_tokens} tokens.

Candidate observations:
{chr(10).join(failure_memory.observations)}
"""
    response = memory_agent.chat_completion(
        chat_history=[{"role": "user", "content": prompt}],
        max_steps=1,
    )
    reflected_failure_logs = _extract_reflected_logs(response.choices[0].message.content)
    reflected_failure_logs = _fit_log_lines_to_token_budget(
        reflected_failure_logs,
        target_tokens,
    )
    if not reflected_failure_logs:
        return (
            "Reflection returned no visible log text. Retry reflect_failure_memory; "
            "if it repeats, load fewer observations or lower target_tokens."
        )

    token_count = count_tokens(reflected_failure_logs)

    preview = reflected_failure_logs[:1_000]
    if len(reflected_failure_logs) > len(preview):
        preview += "\n... preview truncated; use verify_reflected_failure_logs to submit"

    return (
        f"Reflected logs stored. Tokens: {token_count}/{target_tokens}. "
        f"Lines: {len(_split_log_lines(reflected_failure_logs))}.\n{preview}"
    )


def verify_failure_logs(logs: str) -> str:
    """Submit an explicit compact logs string and store Headquarters feedback."""
    global latest_verification_feedback

    token_count = count_tokens(logs)
    if token_count > VERIFY_TOKEN_LIMIT:
        return (
            f"Logs are {token_count} tokens, above the {VERIFY_TOKEN_LIMIT} token limit. "
            "Compress before verifying."
        )

    result = ai_devs_core.verify(TASK_NAME, {"logs": logs})
    latest_verification_feedback = str(result)
    failure_memory.add(
        f"{datetime.now().date().isoformat()} [HIGH] Verification feedback: "
        f"{latest_verification_feedback}"
    )
    return latest_verification_feedback


def verify_reflected_failure_logs() -> str:
    """Submit the stored reflected failure logs to Headquarters."""
    if not reflected_failure_logs:
        return "No reflected logs stored. Run reflect_failure_memory first."
    return verify_failure_logs(reflected_failure_logs)


def create_native_tools(memory_agent: FAgent | None = None) -> list:
    """Create native tools exposed to the lesson agent."""
    global failure_memory_agent

    failure_memory_agent = memory_agent
    return [
        download_server_logs,
        collect_log_candidates,
        add_failure_observations_from_file,
        search_missing_component,
        get_failure_workflow_status,
        reflect_failure_memory,
        verify_reflected_failure_logs,
        verify_failure_logs,
        count_tokens,
    ]


def _discover_safe_mcp_tools() -> list:
    """Return only narrow MCP file inspection tools for this lesson."""
    tools = discover_mcp_tools(MCP_DEFINITIONS)
    return [tool for tool in tools if tool.__name__ in ALLOWED_MCP_TOOLS]


def _reset_lesson_state() -> None:
    """Clear in-memory lesson state for a new chat attempt."""
    global reflected_failure_logs, latest_verification_feedback, failure_memory_agent

    failure_memory.observations = []
    failure_memory.raw_messages = []
    reflected_failure_logs = ""
    latest_verification_feedback = ""
    failure_memory_agent = None


def main() -> None:
    """Run the interactive failure-log lesson chat."""
    console = Console()
    agent = create_agent("mistral", "mistral-small-latest")
    memory_agent = FAgent(model_id="mistral-large-latest")
    native_tools = create_native_tools(memory_agent)
    mcp_tools = _discover_safe_mcp_tools()
    logger.info("Using {} native tools: {}", len(native_tools), [t.__name__ for t in native_tools])
    logger.info("Using {} MCP tools: {}", len(mcp_tools), [t.__name__ for t in mcp_tools])
    session_manager = RefinedSessionManager(
        agent=agent,
        system_prompt=SYSTEM_PROMPT,
        max_context_tokens=SESSION_CONTEXT_TOKEN_LIMIT,
    )
    prompt_session = PromptSession("> ", multiline=False)

    while True:
        try:
            query = prompt_session.prompt()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "/exit":
            break
        if query == "/clear":
            console.print("Clearing the conversation context")
            session_manager = RefinedSessionManager(
                agent=agent,
                system_prompt=SYSTEM_PROMPT,
                max_context_tokens=SESSION_CONTEXT_TOKEN_LIMIT,
            )
            _reset_lesson_state()
            continue
        try:
            session_manager.add_user_message(query)
            final_response = complete(
                session_manager=session_manager,
                agent=agent,
                tools=mcp_tools + native_tools,
            )
            session_manager.add_agent_message(final_response)
        except Exception as e:
            logger.error("Exception: {}, ", e)
            raise


if __name__ == "__main__":
    main()
