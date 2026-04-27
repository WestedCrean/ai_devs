import os
from pathlib import Path
import pathlib

from dotenv import load_dotenv

from loguru import logger
from pydantic import BaseModel
from enum import StrEnum

from prompt_toolkit import PromptSession
from rich.console import Console
from src.ai_devs_core import (
    AIDevsClient,
    get_config,
    discover_mcp_tools,
    complete,
)
from src.ai_devs_core.memory import ObservedMemory

from src.ai_devs_core.session import (
    RefinedSessionManager,
)
from src.ai_devs_core.agent import (
    BaseAgent,
    FAgent,
    OAgent,
    ORAgent,
   create_agent 
)
from src.ai_devs_core.utils import count_tokens


# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

config = get_config()
DATA_SAVE_PATH = pathlib.Path("./data")
MCP_FILES_PATH = DATA_SAVE_PATH / "mcp-files"
TASK_NAME = "failure"
SESSION_CONTEXT_TOKEN_LIMIT = 20_000

SYSTEM_PROMPT = """
Yesterday, a failure occurred at the power plant. 
You have access to the full system log file from that day—but it's huge. 
Your task is to prepare a condensed version of the logs that:
- contains only events relevant to failure analysis (power, cooling, water pumps, software, and other power plant components),
- fits 1,500 tokens,
- maintains a multi-line format—one event per line.

You send the condensed logs to Headquarters via verify tool. Technicians verify whether they can be used to analyze the cause of the failure. If so, you receive a flag. Task parameter name is: failure

Download the full log file using download_server_logs tool. The file changes at midnight (new timestamps), so download it again if you're working the night shift.

How to submit a response?

With verify tool, example answer:

verify( task="failure", answer = {
"logs": "[2026-02-26 06:04] [CRIT] ECCS8 runaway outlet temp. Protection interlock initiated reactor trip.\n[2026-02-26 06:11] [WARN] PWR01 input ripple crossed warning limits.\n[2026-02-26 10:15] [CRIT] WTANK07 coolant below critical threshold. Hard trip initiated."
})

The logs field is a string - lines separated by \n. Each line is a single event.

Format Requirements
One line = one event - do not combine multiple events on one line.
Date in YYYY-MM-DD format - technicians need to know which day the event occurred.
Time in HH:MM or H:MM format - to place the event in time.
You can abbreviate and paraphrase - it's important to retain the timestamp, severity level, and component ID.
Do not exceed 1,500 tokens - this is the hard limit of the Central Office system. You can check the token count via count_tokens tool
What should be done in the task?
Download the log file - check its size. How many lines does it have? How many tokens does the entire file take up?
Filter out important events - from thousands of entries, select only those related to power plant components and failures. How can you determine which events significantly contributed to the failure? Which are the most important?
Compress to the limit - make sure the resulting file fits within 1,500 tokens. You can shorten event descriptions to retain key information.
Send and read the response - The central office returns detailed feedback from the technicians: what's missing, which components are unclear, or insufficiently described. Use this information to improve the logs.
Correct and resend - iterate based on the feedback until the technicians confirm completeness and you receive the {FLG:...} flag.

Tips
The log file is large - how can you meaningfully search it? What model can help? Expensive models will generate high costs if you repeatedly work with large datasets.
The feedback from the technicians is very precise - The central office provides precise information about which components couldn't be analyzed. This is a valuable clue as to what's missing in the logs - it's worth using it to supplement the resulting file.
Is it worth sending everything important at the beginning? - How many tokens do the WARN/ERRO/CRIT events themselves take up? Will they really fit within the limit without further compression? Or is it better to start with a smaller set and update based on feedback? Consider which approach will yield faster results.
Count tokens before sending – sending logs exceeding the limit will result in rejection. Build token counting into a separate step before verification. Use a conservative conversion rate.
Agent-based approach – this task lends itself well to automation with a Function Calling agent, which can search the file, build the resulting log, count tokens, and iteratively send it for verification based on feedback. It's worth having a tool for searching logs instead of keeping them entirely in the main agent's memory. A subagent can handle the search.

Important implementation detail:
download_server_logs saves the full log file into the shared MCP files storage and returns only metadata.
After downloading, inspect and manipulate failure.csv with MCP tools like list_files, head, tail, ripgrep, read_file, write_file, and replace.

How to use ripgrep:
- Read the metadata at the top of every ripgrep result before acting on the matches.
- If the result says "More matches available. Call again with offset=N", then the current result is only one page. Call ripgrep again with the same pattern and filename, using offset=N, until no "More matches available" line remains or you have enough evidence for the current search.
- If total is much larger than returned, do not assume you have seen every relevant line. Page through more results, narrow the regex, or use output_filename for broad searches.
- For large searches, set output_filename so pages are written to a file, then inspect that file with head, tail, read_file, or more targeted ripgrep calls.
- When collecting candidates for add_failure_observations, prefer complete log lines from all relevant pages, not only the first page of a search.

Recommended tool workflow:
- Use add_failure_observations to store newline-separated candidate log events found with search tools.
- Use get_failure_memory_status to inspect the current candidate set.
- Use reflect_failure_memory to compress stored candidates into a verify-ready logs payload.
- Use verify_reflected_failure_logs to submit the stored reflected payload without copying it through chat.
"""


MCP_DEFINITIONS = {
    "files": "http://localhost:8002/mcp",
    "string": "http://localhost:8003/mcp",
}

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
)

FAILURE_MEMORY_TASK = (
    "Store candidate power-plant failure log events and compress them into the "
    "verify payload format: one relevant event per line, under 1500 tokens."
)
failure_memory = ObservedMemory(current_task=FAILURE_MEMORY_TASK)
reflected_failure_logs = ""


class FailureLogCompression(BaseModel):
    """Compressed failure log payload."""

    logs: str


def _split_log_lines(lines: str) -> list[str]:
    """Return non-empty stripped log lines."""
    return [line.strip() for line in lines.splitlines() if line.strip()]


def _count_file_lines(path: pathlib.Path) -> int:
    """Count lines in a text file without returning its contents."""
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


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
        "Use MCP tools such as head, tail, ripgrep, read_file, write_file, "
        "and replace with filename 'failure.csv'."
    )


def create_native_tools(memory_agent: FAgent) -> list:
    """Create native tools exposed to the lesson agent."""
    def add_failure_observations(lines: str) -> str:
        """Add newline-separated candidate failure log events to task memory."""
        added = 0
        existing = set(failure_memory.observations)
        for line in _split_log_lines(lines):
            if line not in existing:
                failure_memory.add(line)
                existing.add(line)
                added += 1

        return (
            f"Added {added} new candidate lines. "
            f"Stored lines: {len(failure_memory.observations)}. "
            f"Approx tokens: {failure_memory.observation_tokens}."
        )

    def get_failure_memory_status(max_lines: int = 20) -> str:
        """Return a preview and token count for stored candidate failure events."""
        preview = "\n".join(failure_memory.observations[:max_lines])
        if len(failure_memory.observations) > max_lines:
            preview += f"\n... {len(failure_memory.observations) - max_lines} more lines"
        return (
            f"Stored lines: {len(failure_memory.observations)}. "
            f"Approx tokens: {failure_memory.observation_tokens}.\n"
            f"{preview or 'No candidate lines stored.'}"
        )

    def reflect_failure_memory(target_tokens: int = 1_500) -> str:
        """Compress stored candidate events into verify-ready failure logs."""
        global reflected_failure_logs

        if not failure_memory.observations:
            return "No candidate lines stored. Add lines with add_failure_observations first."

        prompt = f"""
Compress these candidate power-plant failure log events into the final verify payload.

Hard requirements:
- Put only the final logs string content in the logs field.
- One line equals one event.
- Preserve date in YYYY-MM-DD format, time in HH:MM, severity, and component ID.
- Keep only events relevant to failure analysis: power, cooling, water pumps, software,
  reactor safety systems, sensors, controllers, and directly related components.
- Prefer CRIT, ERRO, WARN, anomalies, trips, thresholds, missing telemetry, and feedback
  from technicians.
- Remove duplicates and routine OK/INFO events unless they explain the failure timeline.
- Paraphrase aggressively, but do not remove causal clues.
- Fit within {target_tokens} tokens.

Candidate lines:
{chr(10).join(failure_memory.observations)}
"""
        response = memory_agent.chat_completion(
            chat_history=[{"role": "user", "content": prompt}],
            response_schema=FailureLogCompression,
            max_steps=1,
        )
        parsed: FailureLogCompression = response.choices[0].message.parsed
        reflected_failure_logs = parsed.logs.strip()
        token_count = count_tokens(reflected_failure_logs)

        preview = reflected_failure_logs[:1_000]
        if len(reflected_failure_logs) > len(preview):
            preview += "\n... preview truncated; use verify_reflected_failure_logs to submit"

        return (
            f"Reflected logs stored. Tokens: {token_count}/{target_tokens}. "
            f"Lines: {len(_split_log_lines(reflected_failure_logs))}.\n{preview}"
        )

    def verify_reflected_failure_logs() -> str:
        """Submit the stored reflected failure logs to Headquarters."""
        if not reflected_failure_logs:
            return "No reflected logs stored. Run reflect_failure_memory first."

        token_count = count_tokens(reflected_failure_logs)
        if token_count > 1_500:
            return (
                f"Reflected logs are {token_count} tokens, above the 1500 token limit. "
                "Run reflect_failure_memory again with a smaller target."
            )

        return str(ai_devs_core.verify(TASK_NAME, {"logs": reflected_failure_logs}))

    return [
        download_server_logs,
        add_failure_observations,
        get_failure_memory_status,
        reflect_failure_memory,
        verify_reflected_failure_logs,
        count_tokens,
        ai_devs_core.verify,
    ]


def main():
    """Main chat endpoint for operators"""
    global reflected_failure_logs

    console = Console()
    agent = create_agent("openrouter", "google/gemini-3-flash-preview") # create_agent("mistral", "mistral-small-latest") # 
    memory_agent = FAgent(model_id="mistral-small-latest")
    native_tools = create_native_tools(memory_agent)
    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(native_tools)} native tools: {[t.__name__ for t in native_tools]}")
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")  # ty:ignore[unresolved-attribute]
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
        elif query == "/clear":
            console.print("Clearing the conversation context")
            session_manager = RefinedSessionManager(
                agent=agent,
                system_prompt=SYSTEM_PROMPT,
                max_context_tokens=SESSION_CONTEXT_TOKEN_LIMIT,
            )
            failure_memory.observations = []
            failure_memory.raw_messages = []
            reflected_failure_logs = ""
        try:
            session_manager.add_user_message(query)
            final_response = complete(
                session_manager=session_manager,
                agent=agent,
                tools=mcp_tools + native_tools,
            )
            session_manager.add_agent_message(final_response)
        except Exception as e:
            logger.error(f"Exception: {e}, ")
            raise e


if __name__ == "__main__":
    main()
