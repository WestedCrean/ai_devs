from jsonschema.validators import create
from pathlib import Path
import pathlib

from dotenv import load_dotenv

from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console
from src.ai_devs_core import (
    FAgent,
    AIDevsClient,
    get_config,
    discover_mcp_tools,
    complete,
)

from src.ai_devs_core.session import (
    BaseSessionManager
)

from src.ai_devs_core.memory import ObservedMemory

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
"""


MCP_DEFINITIONS = {
    "files": "http://localhost:8002/mcp",
    "string": "http://localhost:8003/mcp",
}

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
)

memory = ObservedMemory()


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


def create_native_tools() -> list:
    """Create native tools exposed to the lesson agent."""
    return [
        download_server_logs,
        count_tokens,
        ai_devs_core.verify
    ]


def main():
    """Main chat endpoint for operators"""
    console = Console()
    agent = FAgent(
        model_id="mistral-large-latest"
    )  # mistral-large-latest labs-leanstral-2603
    native_tools = create_native_tools()
    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(native_tools)} native tools: {[t.__name__ for t in native_tools]}")
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")  # ty:ignore[unresolved-attribute]
    session_manager = BaseSessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
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
            session_manager = BaseSessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
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
